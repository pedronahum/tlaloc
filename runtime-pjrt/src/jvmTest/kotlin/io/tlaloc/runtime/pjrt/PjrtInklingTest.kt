package io.tlaloc.runtime.pjrt

import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.cast
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.nn.InklingCheckpoint
import io.tlaloc.nn.InklingLoop
import io.tlaloc.runtime.pjrt.serving.InklingModel
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [InklingLoop.forward] compiled for the GPU: the tiny checkpoint against transformers' modeling
 * code, and, with TLALOC_INKLING_DIR (a snapshot of thinkingmachines/Inkling-2-Mini) and
 * TLALOC_INKLING_ORACLE (what `harness/python/inkling_real_oracle.py` wrote), the released model in
 * bf16 against the same code on the same prompt.
 *
 * One program per padded length: the sequence is right-padded, and since every part of the model
 * is causal (attention, the short convolutions, routing per token, the block mixing per position)
 * the logits of the real positions do not depend on the padding.
 */
class PjrtInklingTest {

    /** One compiled forward per padded length over [dev]'s weights. */
    class Runner(val dev: InklingModel.DeviceWeights, val config: io.tlaloc.nn.InklingConfig) {
        private val session = dev.session
        private val programs = HashMap<Int, String>()

        fun program(len: Int): String = programs.getOrPut(len) {
            val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(len)), intArrayOf(len), I32)
            val fn = captureN(listOf<DTensor<*, *>>(ids) + dev.specs, "inkling_$len", symbolic = true) { leaves ->
                InklingLoop.forward(config, leaves[0], dev.leaves(leaves, 1))
            }
            fn.toStablehlo().let { if (tf32) it else highestF32Dots(it) }
        }

        /** Logits `[ids.size, V]` of [ids], computed at length [len] (right-padded with id 0). */
        fun logits(ids: IntArray, len: Int = ids.size): FloatArray {
            require(ids.size <= len)
            val v = config.unpaddedVocabSize
            val padded = IntArray(len).also { ids.copyInto(it) }
            val idBuf = session.bufferFromHostI32(padded, listOf(len))
            val out = session.executeStablehlo(program(len), listOf(idBuf) + dev.buffers)
            idBuf.close()
            val all = out[0].toFloatArray(len * v)
            out.forEach { it.close() }
            return all.copyOf(ids.size * v)
        }

        /** [n] greedy tokens after [prompt], each by a forward over the whole sequence at length [len]. */
        fun greedy(prompt: IntArray, n: Int, len: Int): Pair<IntArray, FloatArray> {
            val v = config.unpaddedVocabSize
            var seq = prompt
            val margins = FloatArray(n)
            repeat(n) { step ->
                val lg = logits(seq, len)
                val row = lg.copyOfRange((seq.size - 1) * v, seq.size * v)
                val best = row.indices.maxBy { row[it] }
                margins[step] = row[best] - row.filterIndexed { i, _ -> i != best }.max()
                seq += best
            }
            return seq.copyOfRange(prompt.size, seq.size) to margins
        }

    }

    companion object {
        /**
         * The f32 dots run at `precision = HIGHEST` (f32 arithmetic) unless TLALOC_INKLING_TF32=1: the
         * emitter leaves a MATMUL's precision to XLA, which runs it as TF32 on the GB10, and PyTorch's
         * matmuls (the oracle's) do not use TF32 by default. On the tiny checkpoint TF32 moved the
         * 19-token prompt's logits by 0.81 (a routing choice flipped); HIGHEST matches within 5e-5.
         */
        val tf32 = System.getenv("TLALOC_INKLING_TF32") == "1"

        fun highestF32Dots(mlir: String): String = InklingModel.highestF32Dots(mlir)
    }

    /**
     * A chunk program's text grows with the model, not with the cache length: every constant the
     * size of a cache or of `[t, ctx]` is one element broadcast. At an 8,192-row context, writing
     * them out element by element made a 16-token chunk of the released model too large to emit.
     */
    @Test
    fun chunkProgramsStaySmallAtLongContexts() {
        val dir = Path.of("../nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            val weights = ck.hostWeights()
            val keys = weights.keys.toList()
            val sizes = listOf(64, 8192).map { ctx ->
                val ins = listOf<DTensor<*, *>>(
                    DTensor<Shape, I32>(HostI32Storage(IntArray(16)), intArrayOf(16), I32),
                    DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(FloatArray(1)), intArrayOf(1), io.tlaloc.core.F32),
                ) + weights.values + InklingLoop.stateDims(c, ctx).map {
                    DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(FloatArray(0)), it, io.tlaloc.core.F32)
                }
                val fn = io.tlaloc.autograd.captureMulti(ins, "chunk_$ctx", symbolic = true) { leaves ->
                    val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[2 + i] })
                    InklingLoop.chunk(c, ctx, leaves[0], leaves[1], leaves.drop(2 + keys.size), w)
                }
                fn.toStablehlo().length
            }
            println("tiny_inkling 16-token chunk program: ${sizes[0]} characters at context 64, ${sizes[1]} at 8192")
            // The iota over the cache rows is the one constant that grows (one number per row).
            assertTrue(sizes[1] < sizes[0] + 8192 * 16, "chunk program text ${sizes[0]} -> ${sizes[1]} characters")
        }
    }

    /**
     * The fixture's NVFP4 and MXFP8 stacks kept quantized on the device ([InklingCheckpoint.quantizedStack])
     * and dequantized in a program ([InklingLoop.dequantize]) equal the host dequantization, in f32
     * and rounded to bf16, element for element.
     */
    private fun quantizedStacksMatch(session: PjrtSession) {
        val dir = Path.of("../nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")
        val meta = parseJson(Files.readString(dir.resolve("quantized_experts.json"))) as JsonObject
        val e = (meta["experts"] as JsonNumber).value.toInt()
        val rows = (meta["rows"] as JsonNumber).value.toInt()
        val cols = (meta["cols"] as JsonNumber).value.toInt()
        io.tlaloc.core.io.SafetensorsFile.open(dir.resolve("quantized_experts.safetensors")).use { f ->
            fun load(n: String) = if (n in f.names) f.load(n) else null
            for (name in listOf("nv.w", "mx.w")) {
                val host = InklingCheckpoint.dequantizeExperts(name, e, rows, cols, ::load) // [E, rows, cols]
                val parts = InklingCheckpoint.quantizedStack(name, e, rows, cols, load = ::load).toList() // [E, cols, rows]
                for (dtype in listOf<io.tlaloc.core.DType>(io.tlaloc.core.F32, io.tlaloc.core.BF16)) {
                    val specs = parts.map { t ->
                        when (val st = t.storage) {
                            is io.tlaloc.core.HostBytesStorage -> DTensor<Shape, io.tlaloc.core.DType>(io.tlaloc.core.HostBytesStorage(ByteArray(0)), t.dims, t.dtype)
                            else -> t
                        }
                    }
                    val fn = io.tlaloc.autograd.captureMulti(specs, "dequant", symbolic = true) { (codes, scales, factor) ->
                        val q = InklingLoop.QuantizedExperts(codes, scales, factor, codes.dims[1] / scales.dims[1], dtype)
                        listOf(InklingLoop.dequantize(q).let { if (dtype == io.tlaloc.core.F32) it else it.cast(io.tlaloc.core.F32) })
                    }
                    val bufs = parts.map { t ->
                        when (val st = t.storage) {
                            is io.tlaloc.core.HostBytesStorage -> session.bufferFromHostBytes(st.data, t.dims.toList(), t.dtype)
                            is io.tlaloc.core.HostBf16Storage -> session.bufferFromHostBf16(st.data, t.dims.toList())
                            else -> session.bufferFromHostF32((st as io.tlaloc.core.HostF32Storage).data, t.dims.toList())
                        }
                    }
                    val out = session.executeStablehlo(fn.toStablehlo(), bufs)
                    val got = out[0].toFloatArray(e * rows * cols)
                    (bufs + out).forEach { it.close() }
                    var bad = 0
                    for (x in 0 until e) for (r in 0 until rows) for (c in 0 until cols) {
                        val h = host[(x * rows + r) * cols + c]
                        val want = if (dtype == io.tlaloc.core.F32) h else io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(h))
                        if (got[(x * cols + c) * rows + r] != want) bad++
                    }
                    println("$name dequantized on ${session.target} to ${dtype.name}: $bad of ${host.size} differ")
                    assertEquals(0, bad, "$name to ${dtype.name} on ${session.target}")
                }
            }
        }
    }

    @Test
    fun quantizedExpertsDequantizeOnTheCpuPluginAsOnTheHost() {
        val plugin = PjrtBinaries.cpuPluginPath
        assumeTrue(plugin != null, "no XLA CPU plugin; scripts/pjrt-cpu-plugin/build.sh")
        PjrtSession(plugin = plugin!!, target = PjrtTarget.LlvmCpu, options = null).use { quantizedStacksMatch(it) }
    }

    @Test
    fun quantizedExpertsDequantizeOnTheGpuAsOnTheHost() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        PjrtSession(target = PjrtTarget.Cuda).use { quantizedStacksMatch(it) }
    }

    /**
     * [InklingLoop.batchChunk] in the Triton backend's convention on the tiny checkpoint: rows
     * right-aligned with padding, padding rows that name a live slot, a slot reused from position 0
     * with a finished sequence's state still in it; every real row's logits against transformers'.
     */
    private fun batchedWorst(session: PjrtSession, routed: Boolean): Double {
        val dir = Path.of("../nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        val prompts = (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }
        val seqs = prompts.map { jsonInts(it["ids"] as JsonArray) }
        val want = prompts.map { p -> (p["logits"] as JsonArray).elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } } }
        val a = seqs.indices.maxBy { seqs[it].size } // the long prompt
        val bq = seqs.indices.minBy { seqs[it].size } // the short one
        val ctx = 24
        val numSlots = 3
        var worst = 0.0
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            val v = c.unpaddedVocabSize
            val weights = ck.hostWeights()
            val keys = weights.keys.toList()
            val wBufs = weights.values.map { session.bufferFromHostF32((it as DTensor<*, io.tlaloc.core.F32>).hostF32(), it.dims.toList()) }
            val pd = InklingLoop.poolDims(c, ctx, numSlots)
            var pools = pd.map { session.bufferFromHostF32(FloatArray(it.fold(1) { x, y -> x * y }), it.toList()) }
            fun i32(n: IntArray, dims: IntArray) = DTensor<Shape, I32>(HostI32Storage(n), dims, I32)
            val programs = HashMap<Pair<Int, Int>, String>()
            fun program(b: Int, t: Int) = programs.getOrPut(b to t) {
                val ins = listOf<DTensor<*, *>>(i32(IntArray(b * t), intArrayOf(b, t)), i32(IntArray(b * t), intArrayOf(b, t)), i32(IntArray(b * t), intArrayOf(b * t)), i32(IntArray(b), intArrayOf(b))) +
                    weights.values + pd.map { DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(FloatArray(0)), it, io.tlaloc.core.F32) }
                val fn = io.tlaloc.autograd.captureMulti(ins, "batch_${b}_$t", symbolic = true) { l ->
                    val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to l[4 + i] }, io.tlaloc.core.F32)
                    InklingLoop.batchChunk(c, ctx, l[0], l[1], l[2], l[3], l.drop(4 + keys.size), w, routed)
                }
                val aliases = pd.indices.associate { j -> 4 + keys.size + j to 1 + j }
                fn.toStablehlo(outputAliases = aliases).let { if (tf32) it else highestF32Dots(it) }
            }
            /** One call; each row: (sequence index or -1, first position, token count, state slot). Checks the real rows. */
            fun call(t: Int, rows: List<IntArray>, label: String) {
                val b = rows.size
                val ids = IntArray(b * t); val pos = IntArray(b * t); val map = IntArray(b * t) { -1 }; val st = IntArray(b)
                for ((r, row) in rows.withIndex()) {
                    val (sq, p0, n, slot) = row.toList()
                    st[r] = if (sq < 0) 0 else slot
                    if (sq < 0) continue
                    for (j in 0 until n) {
                        val i = r * t + (t - n) + j
                        ids[i] = seqs[sq][p0 + j]; pos[i] = p0 + j; map[i] = 1000 + i // any non-negative value
                    }
                }
                val bufs = listOf(session.bufferFromHostI32(ids, listOf(b, t)), session.bufferFromHostI32(pos, listOf(b, t)), session.bufferFromHostI32(map, listOf(b * t)), session.bufferFromHostI32(st, listOf(b)))
                val out = session.executeStablehlo(program(b, t), bufs + wBufs + pools)
                bufs.forEach { it.close() }
                pools.forEach { runCatching { it.close() } }
                pools = out.drop(1)
                val logits = out[0].toFloatArray(b * v)
                out[0].close()
                for ((r, row) in rows.withIndex()) {
                    val (sq, p0, n, _) = row.toList()
                    if (sq < 0) continue
                    var d = 0.0
                    val wantRow = want[sq][p0 + n - 1]
                    for (j in 0 until v) d = maxOf(d, abs(wantRow[j] - logits[r * v + j]))
                    println("tiny_inkling batch ($label, routed $routed) on ${session.target}, row $r: sequence $sq through position ${p0 + n - 1}, worst logit difference $d")
                    worst = maxOf(worst, d)
                }
            }
            // A: the long prompt in slot 2; B: the short one in slot 0, right-aligned behind padding.
            call(8, listOf(intArrayOf(a, 0, 8, 2), intArrayOf(bq, 0, 5, 0)), "prefill")
            // Decode: A, a padding row (it names slot 0, B's), B.
            call(1, listOf(intArrayOf(a, 8, 1, 2), intArrayOf(-1, 0, 0, 0), intArrayOf(bq, 5, 1, 0)), "decode")
            // B is done; C (the long prompt again) takes slot 0 from position 0, beside A's next 4 tokens.
            call(4, listOf(intArrayOf(a, 9, 4, 2), intArrayOf(a, 0, 3, 0)), "slot reuse")
            call(1, listOf(intArrayOf(a, 3, 1, 0), intArrayOf(a, 13, 1, 2)), "decode after reuse")
            (wBufs + pools).forEach { runCatching { it.close() } }
        }
        return worst
    }

    @Test
    fun batchedChunksOnTheCpuPluginMatchTheModelingCode() {
        val plugin = PjrtBinaries.cpuPluginPath
        assumeTrue(plugin != null, "no XLA CPU plugin; scripts/pjrt-cpu-plugin/build.sh")
        val worst = PjrtSession(plugin = plugin!!, target = PjrtTarget.LlvmCpu, options = null).use { batchedWorst(it, routed = false) }
        assertTrue(worst < 5e-4, "worst logit difference $worst")
    }

    @Test
    fun batchedChunksOnTheGpuMatchTheModelingCode() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeTrue(!tf32, "TLALOC_INKLING_TF32=1")
        val worst = PjrtSession(target = PjrtTarget.Cuda).use { batchedWorst(it, routed = true) }
        assertTrue(worst < 5e-4, "worst logit difference $worst")
    }

    private fun jsonInts(a: JsonArray) = a.elements.map { (it as JsonNumber).value.toInt() }.toIntArray()

    /** The tiny checkpoint's logits from [session] against the oracle's, at its own length and padded to 32. */
    private fun tinyWorst(session: PjrtSession): Double {
        val dir = Path.of("../nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        var worstAll = 0.0
        InklingCheckpoint(dir).use { ck ->
            val v = ck.config.unpaddedVocabSize
            InklingModel.DeviceWeights(session, ck, bf16 = false).use { dev ->
                val r = Runner(dev, ck.config)
                for (p in (fixture["prompts"] as JsonArray).elements) {
                    p as JsonObject
                    val ids = jsonInts(p["ids"] as JsonArray)
                    val want = (p["logits"] as JsonArray).elements.flatMap { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
                    for (len in listOf(ids.size, 32)) {
                        val got = r.logits(ids, len)
                        var worst = 0.0
                        for (i in want.indices) worst = maxOf(worst, abs(want[i] - got[i]))
                        println("tiny_inkling on ${session.target}, ${ids.size} tokens at length $len: worst logit difference $worst")
                        worstAll = maxOf(worstAll, worst)
                    }
                }
            }
            for (p in (fixture["prompts"] as JsonArray).elements) {
                p as JsonObject
                val ids = jsonInts(p["ids"] as JsonArray)
                val want = (p["logits"] as JsonArray).elements.flatMap { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
                // minSpan 24 attends every cache row; 4 attends spans of 4, 8, 16 and 24 rows as the prompt grows.
                for (minSpan in listOf(24, 4)) InklingModel.open(session, ck, contextLength = 24, maxChunk = 8, minSpan = minSpan, bf16 = false, exactF32Dots = !tf32).use { m ->
                    // Token by token (chunks of one: the decode step).
                    var worst = 0.0
                    for ((pos, id) in ids.withIndex()) {
                        val got = m.append(intArrayOf(id))
                        for (j in 0 until v) worst = maxOf(worst, abs(want[pos * v + j] - got[j]))
                    }
                    println("tiny_inkling decode steps (minSpan ${m.minSpan}) on ${session.target}, ${ids.size} tokens: worst logit difference $worst")
                    worstAll = maxOf(worstAll, worst)
                    // The first half in power-of-two chunks, the rest token by token from its state.
                    m.reset()
                    val split = ids.size / 2 + 1
                    worst = 0.0
                    val last = m.append(ids.copyOf(split))
                    for (j in 0 until v) worst = maxOf(worst, abs(want[(split - 1) * v + j] - last[j]))
                    for (pos in split until ids.size) {
                        val got = m.append(intArrayOf(ids[pos]))
                        for (j in 0 until v) worst = maxOf(worst, abs(want[pos * v + j] - got[j]))
                    }
                    println("tiny_inkling $split tokens in chunks then decode on ${session.target}, ${ids.size} tokens: worst logit difference $worst")
                    worstAll = maxOf(worstAll, worst)
                    // Two appends after a reset: 5 tokens, then the rest (a later turn).
                    m.reset()
                    m.append(ids.copyOf(5))
                    val end = m.append(ids.copyOfRange(5, ids.size))
                    worst = 0.0
                    for (j in 0 until v) worst = maxOf(worst, abs(want[(ids.size - 1) * v + j] - end[j]))
                    println("tiny_inkling 5 + ${ids.size - 5} tokens appended on ${session.target}: worst logit difference $worst")
                    worstAll = maxOf(worstAll, worst)
                }
            }
        }
        return worstAll
    }

    /** The lowering itself: XLA's CPU backend computes f32 dots in f32, so the program matches the modeling code. */
    @Test
    fun tinyCheckpointOnTheCpuPluginMatchesTheModelingCode() {
        val plugin = PjrtBinaries.cpuPluginPath
        assumeTrue(plugin != null, "no XLA CPU plugin; scripts/pjrt-cpu-plugin/build.sh")
        val worst = PjrtSession(plugin = plugin!!, target = PjrtTarget.LlvmCpu, options = null).use { tinyWorst(it) }
        assertTrue(worst < 5e-4, "worst logit difference $worst")
    }

    /** The same on the GPU, with the f32 dots at HIGHEST (see [tf32]). */
    @Test
    fun tinyCheckpointOnTheGpuMatchesTheModelingCode() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeTrue(!tf32, "TLALOC_INKLING_TF32=1")
        val worst = PjrtSession(target = PjrtTarget.Cuda).use { tinyWorst(it) }
        assertTrue(worst < 5e-4, "worst logit difference $worst")
    }

    @Test
    fun releasedModelMatchesTheModelingCode() {
        val dir = System.getenv("TLALOC_INKLING_DIR")
        val oracle = System.getenv("TLALOC_INKLING_ORACLE")
        assumeTrue(dir != null && oracle != null, "TLALOC_INKLING_DIR and TLALOC_INKLING_ORACLE not set")
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        val o = parseJson(Files.readString(Path.of(oracle))) as JsonObject
        val ids = jsonInts(o["ids"] as JsonArray)
        val wantGreedy = jsonInts(o["greedy"] as JsonArray)
        val wantLast = (o["last_logits"] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }
        val t0 = System.nanoTime()
        // The prompt in power-of-two chunks (16 + 8 + 4 + 1 for 29 tokens), the continuation token by
        // token, then 32 steady-state decode steps.
        val ctx = ((ids.size + wantGreedy.size + 32 + 63) / 64) * 64
        // The experts stay FP8 codes and scales on the device (InklingModel's default);
        // TLALOC_INKLING_QUANTIZED=0 dequantizes them to bf16 on the host instead.
        val quantized = System.getenv("TLALOC_INKLING_QUANTIZED") != "0"
        println("inkling: experts ${if (quantized) "quantized on the device" else "dequantized to bf16 on the host"}")
        InklingModel.open(Path.of(dir), contextLength = ctx, maxChunk = 16, quantizedExperts = quantized, exactF32Dots = !tf32) { println(it) }.use { m ->
            println("inkling: weights read and uploaded in %.0f s".format((System.nanoTime() - t0) / 1e9))
            val wantTop = wantLast.indices.maxBy { wantLast[it] }
            val t3 = System.nanoTime()
            var row = m.append(ids)
            println("inkling: ${ids.size} prompt tokens in %.1f s (compiles included)".format((System.nanoTime() - t3) / 1e9))
            m.reset()
            val t3b = System.nanoTime()
            row = m.append(ids)
            println("inkling: ${ids.size} prompt tokens again in %.2f s".format((System.nanoTime() - t3b) / 1e9))
            var worst = 0f
            for (i in row.indices) worst = maxOf(worst, abs(row[i] - wantLast[i]))
            val top = row.indices.maxBy { row[it] }
            println("inkling: last position worst logit difference $worst (oracle top logits ${wantLast.sortedDescending().take(3)}); argmax $top, oracle $wantTop")
            val got = IntArray(wantGreedy.size)
            val margins = FloatArray(wantGreedy.size)
            val t4 = System.nanoTime()
            for (k in got.indices) {
                val best = InklingModel.argmax(row)
                got[k] = best
                margins[k] = row[best] - row.filterIndexed { i, _ -> i != best }.max()
                if (k + 1 < got.size) row = m.append(intArrayOf(best))
            }
            val genS = (System.nanoTime() - t4) / 1e9
            println("inkling: ${got.size - 1} decode steps in %.2f s, %.1f tokens/s (the first compiles)".format(genS, (got.size - 1) / genS))
            println("inkling: greedy  ${got.toList()}")
            println("inkling: oracle  ${wantGreedy.toList()}")
            println("inkling: margins ${margins.map { "%.2f".format(it) }}")
            val firstDiff = got.indices.firstOrNull { got[it] != wantGreedy[it] }
            // A choice may differ where the top two logits are within bf16 noise; up to there they agree.
            assertTrue(firstDiff == null || margins[firstDiff] < 0.5f, "greedy differs at $firstDiff with margin ${firstDiff?.let { margins[it] }}")
            // Steady-state decode: 32 more steps after the compile.
            val t5 = System.nanoTime()
            var next = got.last()
            repeat(32) { next = InklingModel.argmax(m.append(intArrayOf(next))) }
            println("inkling: 32 decode steps in %.2f s, %.1f tokens/s".format((System.nanoTime() - t5) / 1e9, 32 / ((System.nanoTime() - t5) / 1e9)))
            assertEquals(wantTop, top)
        }
    }
}
