package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.nio.file.Path

/**
 * A teacher's next-token distribution at every position of a sequence, for distillation: a
 * checkpoint's scoring entry ([DecodeGraphSpec.scoreTopK]) compiled once for one chunk length,
 * its weights staged on the device once (quantized as [HfDecoderConfig] says, as a serving
 * export stages them), its KV pools kept on the device and updated in place chunk after chunk.
 *
 * [score] returns, for each position `i` of `ids`, the [topK] most likely tokens after
 * `ids[0..i]` and their log-probabilities under the full softmax. A sequence runs in chunks of
 * [chunk] tokens; the last one is right-aligned behind padding rows, whose results are dropped.
 */
class TeacherScorer private constructor(
    private val session: PjrtSession,
    val config: HfDecoderConfig,
    val spec: DecodeGraphSpec,
    private val program: String,
    private val weights: List<PjrtBuffer>,
    private var pools: List<PjrtBuffer>,
) : AutoCloseable {
    val topK: Int = spec.scoreTopK!!
    val chunk: Int = spec.tokensPerSeq
    val maxContext: Int = spec.bucket.maxContext
    private val blockSize = spec.model.blockSize
    private val tables = IntArray(spec.maxBlocksPerSeq) { 1 + it }

    /** Top-k ids (`[n * topK]`) and log-probabilities (`[n * topK]`), row `i` after `ids[0..i]`. */
    fun score(ids: IntArray): Pair<IntArray, FloatArray> {
        require(ids.isNotEmpty() && ids.size <= maxContext) { "TeacherScorer: ${ids.size} tokens; the pools hold $maxContext" }
        val outIds = IntArray(ids.size * topK)
        val outLp = FloatArray(ids.size * topK)
        var start = 0
        while (start < ids.size) {
            val real = minOf(chunk, ids.size - start)
            val pad = chunk - real
            val tok = IntArray(chunk) { if (it < pad) 0 else ids[start + it - pad] }
            val pos = IntArray(chunk) { if (it < pad) 0 else start + it - pad }
            val slots = IntArray(chunk) { if (it < pad) -1 else blockSize + start + it - pad }
            val small = listOf(
                session.bufferFromHostI32(tok, listOf(1, chunk)),
                session.bufferFromHostI32(pos, listOf(1, chunk)),
                session.bufferFromHostI32(tables, listOf(1, tables.size)),
                session.bufferFromHostI32(intArrayOf(start + real), listOf(1)),
                session.bufferFromHostI32(slots, listOf(chunk)),
            )
            val out = try {
                session.executeStablehlo(program, small + pools + weights)
            } finally {
                small.forEach { it.close() }
            }
            pools.forEach { runCatching { it.close() } }
            pools = out.drop(2)
            val gotIds = out[0].toIntArray(chunk * topK)
            val gotLp = out[1].toFloatArray(chunk * topK)
            out[0].close()
            out[1].close()
            System.arraycopy(gotIds, pad * topK, outIds, start * topK, real * topK)
            System.arraycopy(gotLp, pad * topK, outLp, start * topK, real * topK)
            start += real
        }
        return outIds to outLp
    }

    override fun close() {
        pools.forEach { runCatching { it.close() } }
        weights.forEach { runCatching { it.close() } }
        session.close()
    }

    companion object {
        /** Little-endian [bytes] of [dtype] onto the device. */
        private fun upload(session: PjrtSession, bytes: ByteArray, dims: List<Int>, dtype: io.tlaloc.core.DType): PjrtBuffer {
            val bb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            return when (dtype.sizeBytes) {
                1 -> session.bufferFromHostBytes(bytes, dims, dtype)
                2 -> ShortArray(bytes.size / 2).also { bb.asShortBuffer().get(it) }.let { require(dtype == io.tlaloc.core.BF16) { "TeacherScorer: a 2-byte $dtype slot" }; session.bufferFromHostBf16(it, dims) }
                4 -> if (dtype == io.tlaloc.core.I32) session.bufferFromHostI32(IntArray(bytes.size / 4).also { bb.asIntBuffer().get(it) }, dims)
                else session.bufferFromHostF32(FloatArray(bytes.size / 4).also { bb.asFloatBuffer().get(it) }, dims)
                else -> throw IllegalArgumentException("TeacherScorer: no upload for $dtype")
            }
        }

        /**
         * Opens the checkpoint in [dir] (no draft), applies [configure] to its config (the weight
         * dtype and quantization a serving export would use), builds the scoring entry for
         * [chunk]-token calls and sequences of up to [maxContext] tokens, and stages the weights.
         * [kernels]: the CUDA kernel library (`libtlaloc_kernels.so`) when the entry uses it.
         */
        fun open(
            dir: Path,
            topK: Int,
            chunk: Int = 512,
            maxContext: Int = 16384,
            blockSize: Int = 16,
            target: PjrtTarget = PjrtTarget.Cuda,
            kernels: Path? = null,
            log: (String) -> Unit = {},
            configure: (HfDecoderConfig) -> HfDecoderConfig = { it },
        ): TeacherScorer {
            require(maxContext % blockSize == 0 && chunk <= maxContext) { "TeacherScorer: maxContext must be a multiple of blockSize and >= chunk" }
            if (kernels != null) {
                require(Files.isRegularFile(kernels)) { "TeacherScorer: no kernel library at $kernels" }
                PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("TeacherScorer"), kernels)
            }
            val session = when (target) {
                PjrtTarget.LlvmCpu -> PjrtSession(plugin = PjrtBinaries.requireCpuPlugin("TeacherScorer"), target = target)
                else -> PjrtSession(target = target)
            }
            try {
                HfCheckpoint.open(dir).use { ckpt ->
                    val config = configure(ckpt.config)
                    require(config.mtpDraftTokens == 0 && config.dflash == null) { "TeacherScorer: a scoring entry has no drafts" }
                    val model = config.toDecodeModelShape(numBlocks = 1 + maxContext / blockSize, blockSize = blockSize, cudaKernels = kernels != null)
                    val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, maxContext), DecodeGraphKind.PREFILL, prefillChunk = chunk, scoreTopK = topK)
                    val fn = HfDecoderGraph.build(spec, config)
                    val poolIn = spec.inputs.withIndex().filter { (_, s) -> s.role == DecodeSlotRole.KV_POOL_IN }
                    val aliases = poolIn.withIndex().associate { (j, iv) -> iv.index to spec.leadingOutputs + j }
                    val program = fn.toStablehlo(outputAliases = aliases)
                    val t0 = System.nanoTime()
                    val slots = HfDecoderGraph.weightSlots(config)
                    val weights = ArrayList<PjrtBuffer>(slots.size)
                    try {
                        for ((i, slot) in slots.withIndex()) {
                            val bytes = ByteArrayOutputStream().also { HfStagedWeights.writeSlot(ckpt, config, i, it) }.toByteArray()
                            weights += upload(session, bytes, slot.type.dims, slot.type.dtype)
                        }
                        log("TeacherScorer: ${slots.size} weight slots staged in ${(System.nanoTime() - t0) / 1_000_000_000} s")
                        val pools = poolIn.map { (_, s) ->
                            val n = s.type.dims.fold(1L) { a, b -> a * b } * s.type.dtype.sizeBytes
                            upload(session, ByteArray(Math.toIntExact(n)), s.type.dims, s.type.dtype)
                        }
                        val t1 = System.nanoTime()
                        session.prepareStablehlo(program)
                        log("TeacherScorer: scoring entry (chunk $chunk, context $maxContext, top $topK) compiled in ${(System.nanoTime() - t1) / 1_000_000_000} s")
                        return TeacherScorer(session, config, spec, program, weights, pools)
                    } catch (e: Throwable) {
                        weights.forEach { runCatching { it.close() } }
                        throw e
                    }
                }
            } catch (e: Throwable) {
                runCatching { session.close() }
                throw e
            }
        }
    }
}
