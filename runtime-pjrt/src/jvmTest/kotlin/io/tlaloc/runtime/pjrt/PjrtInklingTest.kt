package io.tlaloc.runtime.pjrt

import io.tlaloc.autograd.captureN
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
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
        InklingModel.open(Path.of(dir), contextLength = ctx, maxChunk = 16, exactF32Dots = !tf32) { println(it) }.use { m ->
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
