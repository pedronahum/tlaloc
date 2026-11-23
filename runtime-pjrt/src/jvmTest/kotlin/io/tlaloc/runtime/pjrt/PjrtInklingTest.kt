package io.tlaloc.runtime.pjrt

import io.tlaloc.autograd.captureN
import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
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
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
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

    /**
     * The weights on the device once, and one compiled forward per padded length. Each weight is
     * uploaded as the checkpoint reader produces it and only its shape is kept (a symbolic capture
     * reads no values), so the host never holds the model.
     */
    class Runner(val session: PjrtSession, val ck: InklingCheckpoint, bf16: Boolean, log: (String) -> Unit = {}) : AutoCloseable {
        private val keys = ArrayList<String>()
        private val specs = ArrayList<DTensor<*, *>>()
        private val buffers = ArrayList<PjrtBuffer>()

        init {
            ck.forEachWeight(bf16, log) { k, t ->
                keys += k
                when (val s = t.storage) {
                    is HostBf16Storage -> {
                        buffers += session.bufferFromHostBf16(s.data, t.dims.toList())
                        specs += DTensor<Shape, BF16>(HostBf16Storage(ShortArray(0)), t.dims, BF16)
                    }
                    else -> {
                        buffers += session.bufferFromHostF32((t as DTensor<*, F32>).hostF32(), t.dims.toList())
                        specs += DTensor<Shape, F32>(io.tlaloc.core.HostF32Storage(FloatArray(0)), t.dims, F32)
                    }
                }
            }
        }
        private val programs = HashMap<Int, String>()

        fun program(len: Int): String = programs.getOrPut(len) {
            val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(len)), intArrayOf(len), I32)
            val fn = captureN(listOf<DTensor<*, *>>(ids) + specs, "inkling_$len", symbolic = true) { leaves ->
                InklingLoop.forward(ck.config, leaves[0], InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[1 + i] }))
            }
            fn.toStablehlo().let { if (tf32) it else highestF32Dots(it) }
        }

        /** Logits `[ids.size, V]` of [ids], computed at length [len] (right-padded with id 0). */
        fun logits(ids: IntArray, len: Int = ids.size): FloatArray {
            require(ids.size <= len)
            val v = ck.config.unpaddedVocabSize
            val padded = IntArray(len).also { ids.copyInto(it) }
            val idBuf = session.bufferFromHostI32(padded, listOf(len))
            val out = session.executeStablehlo(program(len), listOf(idBuf) + buffers)
            idBuf.close()
            val all = out[0].toFloatArray(len * v)
            out.forEach { it.close() }
            return all.copyOf(ids.size * v)
        }

        /** [n] greedy tokens after [prompt], each by a forward over the whole sequence at length [len]. */
        fun greedy(prompt: IntArray, n: Int, len: Int): Pair<IntArray, FloatArray> {
            val v = ck.config.unpaddedVocabSize
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

        override fun close() = buffers.forEach { it.close() }
    }

    companion object {
        /**
         * The f32 dots run at `precision = HIGHEST` (f32 arithmetic) unless TLALOC_INKLING_TF32=1: the
         * emitter leaves a MATMUL's precision to XLA, which runs it as TF32 on the GB10, and PyTorch's
         * matmuls (the oracle's) do not use TF32 by default. On the tiny checkpoint TF32 moved the
         * 19-token prompt's logits by 0.81 (a routing choice flipped); HIGHEST matches within 5e-5.
         */
        val tf32 = System.getenv("TLALOC_INKLING_TF32") == "1"

        private val F32_DOT = Regex("""(stablehlo\.dot_general [^\n]*?)( : \(tensor<[^>]*xf32>, tensor<[^>]*xf32>\))""")

        fun highestF32Dots(mlir: String): String = F32_DOT.replace(mlir) { m ->
            if ("algorithm" in m.groupValues[1] || "precision" in m.groupValues[1]) m.value
            else m.groupValues[1] + ", precision = [HIGHEST, HIGHEST]" + m.groupValues[2]
        }
    }

    private fun jsonInts(a: JsonArray) = a.elements.map { (it as JsonNumber).value.toInt() }.toIntArray()

    /** The tiny checkpoint's logits from [session] against the oracle's, at its own length and padded to 32. */
    private fun tinyWorst(session: PjrtSession): Double {
        val dir = Path.of("../nn/src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        var worstAll = 0.0
        InklingCheckpoint(dir).use { ck ->
            Runner(session, ck, bf16 = false).use { r ->
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
        }
        return worstAll
    }

    /** The lowering itself: XLA's CPU backend computes f32 dots in f32, so the program matches the modeling code. */
    @Test
    fun tinyCheckpointOnTheCpuPluginMatchesTheModelingCode() {
        val plugin = PjrtBinaries.cpuPluginPath
        assumeTrue(plugin != null, "no XLA CPU plugin; scripts/pjrt-cpu-plugin/build.sh")
        val worst = PjrtSession(plugin = plugin!!, target = PjrtTarget.LlvmCpu, options = null).use { tinyWorst(it) }
        assertTrue(worst < 2e-4, "worst logit difference $worst")
    }

    /** The same on the GPU, with the f32 dots at HIGHEST (see [tf32]). */
    @Test
    fun tinyCheckpointOnTheGpuMatchesTheModelingCode() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeTrue(!tf32, "TLALOC_INKLING_TF32=1")
        val worst = PjrtSession(target = PjrtTarget.Cuda).use { tinyWorst(it) }
        assertTrue(worst < 2e-4, "worst logit difference $worst")
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
        PjrtSession(target = PjrtTarget.Cuda).use { s ->
            InklingCheckpoint(Path.of(dir)).use { ck ->
                val t0 = System.nanoTime()
                Runner(s, ck, bf16 = true) { println(it) }.use { r ->
                    println("inkling: weights read and uploaded in %.0f s".format((System.nanoTime() - t0) / 1e9))
                    val len = ((ids.size + wantGreedy.size + 15) / 16) * 16
                    val tc = System.nanoTime()
                    r.program(len)
                    println("inkling: captured and emitted at length $len in %.1f s".format((System.nanoTime() - tc) / 1e9))
                    val t1 = System.nanoTime()
                    val lg = r.logits(ids, len)
                    println("inkling: first forward at length $len in %.1f s (compile included)".format((System.nanoTime() - t1) / 1e9))
                    val v = ck.config.unpaddedVocabSize
                    val last = lg.copyOfRange((ids.size - 1) * v, ids.size * v)
                    var worst = 0f
                    for (i in last.indices) worst = maxOf(worst, abs(last[i] - wantLast[i]))
                    val top = last.indices.maxBy { last[it] }
                    println("inkling: last position worst logit difference $worst; argmax $top, oracle ${wantLast.indices.maxBy { wantLast[it] }}")
                    val t2 = System.nanoTime()
                    val (got, margins) = r.greedy(ids, wantGreedy.size, len)
                    println("inkling: ${wantGreedy.size} greedy tokens in %.1f s".format((System.nanoTime() - t2) / 1e9))
                    println("inkling: greedy  ${got.toList()}")
                    println("inkling: oracle  ${wantGreedy.toList()}")
                    println("inkling: margins ${margins.map { "%.2f".format(it) }}")
                    val firstDiff = got.indices.firstOrNull { got[it] != wantGreedy[it] }
                    // A greedy choice may differ where the top two logits are within bf16 noise; up to there they agree.
                    assertTrue(firstDiff == null || margins[firstDiff] < 0.5f, "greedy differs at $firstDiff with margin ${firstDiff?.let { margins[it] }}")
                    assertEquals(wantLast.indices.maxBy { wantLast[it] }, top)
                }
            }
        }
    }
}
