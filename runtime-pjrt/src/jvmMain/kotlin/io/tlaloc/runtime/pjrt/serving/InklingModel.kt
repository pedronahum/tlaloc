package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureMulti
import io.tlaloc.autograd.slice
import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostBytesStorage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.nn.InklingCheckpoint
import io.tlaloc.nn.InklingConfig
import io.tlaloc.nn.InklingLoop
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Path

/**
 * Inkling 2 (`inkling2_loop`, Thinking Machines' Inkling-2-Mini) in the calling JVM: the weights on
 * the device, [InklingLoop.chunk] compiled for chunks of 1, 2, 4, … [maxChunk] tokens (each on first
 * use), and the model's state (K/V caches of [contextLength] rows and the short-convolution
 * histories of every layer application) kept on the device and updated in place.
 *
 * Each chunk attends over a span of the caches: the smallest power of two at least [minSpan] that
 * holds `position + n` (capped at [contextLength]), with one program per (chunk size, span), so a
 * short conversation does not pay for long caches.
 *
 * [append] runs tokens at the current [position], split into power-of-two chunks, so no padding
 * enters the caches or the convolution histories; a later turn of a chat appends to the same
 * state. [generate] samples a continuation. Not thread-safe: one sequence at a time.
 *
 * ```kotlin
 * InklingModel.open(snapshot, contextLength = 8192).use { m ->
 *     val out = m.generate(promptIds, maxTokens = 256, stop = setOf(endId))
 * }
 * ```
 */
class InklingModel private constructor(
    private val session: PjrtSession,
    val config: InklingConfig,
    val contextLength: Int,
    val maxChunk: Int,
    val minSpan: Int,
    private val exactF32Dots: Boolean,
    private val routedExperts: Boolean,
    private val weights: DeviceWeights,
    private val ownsSession: Boolean,
) : AutoCloseable {
    private val dims = InklingLoop.stateDims(config, contextLength)
    private var state: List<PjrtBuffer> = zeroState()
    private val programs = HashMap<Pair<Int, Int>, String>()

    /** The tokens in the state: the next token goes at this position. */
    var position: Int = 0
        private set

    /** Vocabulary size of the logits [append] and [generate] see. */
    val vocabSize: Int get() = config.unpaddedVocabSize

    private fun zeroState(): List<PjrtBuffer> = dims.map { d -> session.bufferFromHostF32(FloatArray(d.fold(1) { a, b -> a * b }), d.toList()) }

    /** Forgets every token: the state is zeros, [position] 0. */
    fun reset() {
        state.forEach { runCatching { it.close() } }
        state = zeroState()
        position = 0
    }

    /** The cache rows a chunk ending at [end] attends: a power of two ≥ [minSpan] and ≥ [end], at most [contextLength]. */
    fun spanFor(end: Int): Int {
        var s = minSpan
        while (s < end) s *= 2
        return minOf(s, contextLength)
    }

    /**
     * The program for chunks of [n] tokens attending [span] cache rows: ids `[n]`, the position, the
     * weights and the state in; the last row's logits `[1, V]` and the new state out (each state input
     * aliased to its output).
     */
    private fun program(n: Int, span: Int): String = programs.getOrPut(n to span) {
        val ins = listOf<DTensor<*, *>>(
            DTensor<Shape, I32>(HostI32Storage(IntArray(n)), intArrayOf(n), I32),
            DTensor<Shape, F32>(HostF32Storage(FloatArray(1)), intArrayOf(1), F32),
        ) + weights.specs + dims.map { DTensor<Shape, F32>(HostF32Storage(FloatArray(0)), it, F32) }
        val nw = weights.keys.size
        val fn = captureMulti(ins, "inkling_chunk_${n}_${span}_$contextLength", symbolic = true) { leaves ->
            val out = InklingLoop.chunk(config, contextLength, leaves[0], leaves[1], leaves.drop(2 + nw), weights.leaves(leaves, 2), routedExperts, span)
            listOf<Tracer<Shape>>(if (n == 1) out[0] else out[0].slice(n - 1, n, 0)) + out.drop(1)
        }
        val aliases = dims.indices.associate { j -> 2 + nw + j to 1 + j }
        fn.toStablehlo(outputAliases = aliases).let { if (exactF32Dots) highestF32Dots(it) else it }
    }

    /** Compiles the chunk programs of [sizes] at the spans of [spans] now rather than on first use. */
    fun warmUp(sizes: List<Int> = chunkSizes(), spans: List<Int> = listOf(spanFor(1))) {
        for (sp in spans) for (n in sizes) if (n <= sp) session.prepareStablehlo(program(n, sp))
    }

    /** [minSpan], twice it, … up to [contextLength]: the spans a conversation can reach. */
    fun spans(): List<Int> = generateSequence(minSpan) { it * 2 }.takeWhile { it < contextLength }.toList() + contextLength

    /** 1, 2, 4, … [maxChunk]. */
    fun chunkSizes(): List<Int> = generateSequence(1) { it * 2 }.takeWhile { it <= maxChunk }.toList()

    private fun run(ids: IntArray): FloatArray {
        val n = ids.size
        val idBuf = session.bufferFromHostI32(ids, listOf(n))
        val posBuf = session.bufferFromHostF32(floatArrayOf(position.toFloat()), listOf(1))
        val out = try {
            session.executeStablehlo(program(n, spanFor(position + n)), listOf(idBuf, posBuf) + weights.buffers + state)
        } finally {
            idBuf.close()
            posBuf.close()
        }
        state.forEach { runCatching { it.close() } }
        state = out.drop(1)
        val logits = out[0].toFloatArray(config.unpaddedVocabSize)
        out[0].close()
        position += n
        return logits
    }

    /**
     * Runs [ids] at [position] (the largest power-of-two chunks up to [maxChunk] first) and returns
     * the logits after the last of them.
     */
    fun append(ids: IntArray): FloatArray {
        require(ids.isNotEmpty()) { "InklingModel.append: no tokens" }
        require(position + ids.size <= contextLength) { "InklingModel: ${position + ids.size} tokens; the caches hold $contextLength" }
        var logits = FloatArray(0)
        var i = 0
        while (i < ids.size) {
            var n = maxChunk
            while (n > ids.size - i) n /= 2
            logits = run(ids.copyOfRange(i, i + n))
            i += n
        }
        return logits
    }

    /**
     * Appends [prompt], then up to [maxTokens] tokens chosen by [sample] from each step's logits
     * (greedy by default), stopping after a token in [stop]. Each chosen token goes to [onToken] as
     * it is chosen. Returns the chosen tokens; the state then holds all but the last of them (the
     * last is fed on the next [append]).
     */
    fun generate(
        prompt: IntArray,
        maxTokens: Int,
        stop: Set<Int> = emptySet(),
        sample: (FloatArray) -> Int = ::argmax,
        onToken: (Int) -> Unit = {},
    ): IntArray {
        var logits = append(prompt)
        val out = ArrayList<Int>()
        while (out.size < maxTokens) {
            val next = sample(logits)
            out += next
            onToken(next)
            if (next in stop || out.size == maxTokens || position >= contextLength) break
            logits = run(intArrayOf(next))
        }
        return out.toIntArray()
    }

    override fun close() {
        state.forEach { runCatching { it.close() } }
        weights.close()
        if (ownsSession) session.close()
    }

    /**
     * The weights on the device, in [InklingCheckpoint.forEachWeight]'s order. Each is uploaded as
     * the reader produces it and only its shape is kept (a symbolic capture reads no values), so
     * the host never holds the model.
     */
    class DeviceWeights(
        val session: PjrtSession,
        ck: InklingCheckpoint,
        val bf16: Boolean,
        log: (String) -> Unit = {},
        quantizedExperts: Boolean = false,
    ) : AutoCloseable {
        val keys = ArrayList<String>()
        val specs = ArrayList<DTensor<*, *>>()
        val buffers = ArrayList<PjrtBuffer>()

        init {
            try {
                ck.forEachWeight(bf16, log, quantizedExperts) { k, t ->
                    keys += k
                    when (val s = t.storage) {
                        is HostBytesStorage -> {
                            buffers += session.bufferFromHostBytes(s.data, t.dims.toList(), t.dtype)
                            specs += DTensor<Shape, io.tlaloc.core.DType>(HostBytesStorage(ByteArray(0)), t.dims, t.dtype)
                        }
                        is HostBf16Storage -> {
                            buffers += session.bufferFromHostBf16(s.data, t.dims.toList())
                            specs += DTensor<Shape, BF16>(HostBf16Storage(ShortArray(0)), t.dims, BF16)
                        }
                        else -> {
                            @Suppress("UNCHECKED_CAST")
                            buffers += session.bufferFromHostF32((t as DTensor<*, F32>).hostF32(), t.dims.toList())
                            specs += DTensor<Shape, F32>(HostF32Storage(FloatArray(0)), t.dims, F32)
                        }
                    }
                }
            } catch (e: Throwable) {
                close()
                throw e
            }
        }

        /** The weight tracers of a capture whose inputs from [offset] on are [specs]. */
        fun leaves(from: List<Tracer<Shape>>, offset: Int): InklingLoop.Weights =
            InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to from[offset + i] }, if (bf16) BF16 else F32)

        override fun close() = buffers.forEach { runCatching { it.close() } }
    }

    companion object {
        /** The index of the largest logit. */
        fun argmax(logits: FloatArray): Int {
            var best = 0
            for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
            return best
        }

        private val F32_DOT = Regex("""(stablehlo\.dot_general [^\n]*?)( : \(tensor<[^>]*xf32>, tensor<[^>]*xf32>\))""")

        /**
         * [mlir] with every f32 `dot_general` at `precision = [HIGHEST, HIGHEST]`. The emitter leaves a
         * MATMUL's precision to XLA, which runs f32 dots as TF32 on recent NVIDIA GPUs; Inkling's
         * routing (top-k of sigmoid scores) can flip on that rounding.
         */
        fun highestF32Dots(mlir: String): String = F32_DOT.replace(mlir) { m ->
            if ("algorithm" in m.groupValues[1] || "precision" in m.groupValues[1]) m.value
            else m.groupValues[1] + ", precision = [HIGHEST, HIGHEST]" + m.groupValues[2]
        }

        /**
         * Reads the checkpoint in [dir] (a snapshot of `thinkingmachines/Inkling-2-Mini`) onto the
         * device: projections in bf16 when [bf16] (as stored), experts dequantized from NVFP4 and
         * MXFP8. [contextLength] sizes the caches (a multiple of nothing in particular; every
         * position below it is usable); [maxChunk] is the largest chunk [append] runs at once (a
         * power of two); [quantizedExperts] keeps the experts the files hold in NVFP4 or MXFP8 as FP8
         * codes with their block scales on the device (about 1.1 bytes a weight instead of bf16's 2) and
         * dequantizes them in each program, to the same values; [minSpan] is the shortest span of the caches a chunk attends (see the class).
         * [exactF32Dots] runs the f32 dots at full precision (see [highestF32Dots]).
         * [routedExperts] runs a chunk's experts as ragged dots on their own tokens rather than every
         * expert on every token; XLA:CPU has no `ragged_dot`, so it is off there by default.
         */
        fun open(
            dir: Path,
            contextLength: Int = 4096,
            maxChunk: Int = 256,
            minSpan: Int = minOf(512, contextLength),
            bf16: Boolean = true,
            quantizedExperts: Boolean = true,
            target: PjrtTarget = PjrtTarget.Cuda,
            exactF32Dots: Boolean = true,
            routedExperts: Boolean = target != PjrtTarget.LlvmCpu,
            log: (String) -> Unit = {},
        ): InklingModel {
            require(maxChunk > 0 && maxChunk and (maxChunk - 1) == 0) { "InklingModel: maxChunk $maxChunk is not a power of two" }
            require(maxChunk <= contextLength) { "InklingModel: maxChunk $maxChunk past the context $contextLength" }
            require(minSpan in 1..contextLength) { "InklingModel: minSpan $minSpan outside 1..$contextLength" }
            val session = when (target) {
                PjrtTarget.LlvmCpu -> PjrtSession(plugin = PjrtBinaries.requireCpuPlugin("InklingModel"), target = target)
                else -> PjrtSession(target = target)
            }
            try {
                InklingCheckpoint(dir).use { ck ->
                    val weights = DeviceWeights(session, ck, bf16, log, quantizedExperts)
                    return InklingModel(session, ck.config, contextLength, maxChunk, minSpan, exactF32Dots, routedExperts, weights, ownsSession = true)
                }
            } catch (e: Throwable) {
                runCatching { session.close() }
                throw e
            }
        }

        /**
         * As [open], on a [session] the caller owns (and closes after this model) and an open
         * checkpoint [ck] (read here; it may be closed afterwards).
         */
        fun open(
            session: PjrtSession,
            ck: InklingCheckpoint,
            contextLength: Int = 4096,
            maxChunk: Int = 256,
            minSpan: Int = minOf(512, contextLength),
            bf16: Boolean = true,
            quantizedExperts: Boolean = true,
            exactF32Dots: Boolean = true,
            routedExperts: Boolean = session.target != PjrtTarget.LlvmCpu,
            log: (String) -> Unit = {},
        ): InklingModel {
            require(maxChunk > 0 && maxChunk and (maxChunk - 1) == 0) { "InklingModel: maxChunk $maxChunk is not a power of two" }
            require(maxChunk <= contextLength) { "InklingModel: maxChunk $maxChunk past the context $contextLength" }
            require(minSpan in 1..contextLength) { "InklingModel: minSpan $minSpan outside 1..$contextLength" }
            return InklingModel(session, ck.config, contextLength, maxChunk, minSpan, exactF32Dots, routedExperts, DeviceWeights(session, ck, bf16, log, quantizedExperts), ownsSession = false)
        }
    }
}
