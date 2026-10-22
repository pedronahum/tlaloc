package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.SampleTokensAttrs
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * SAMPLE_TOKENS without `kernel` (the StableHLO form, what a TPU runs) on XLA:GPU against
 * [SampleTokensAttrs.reference], row for row: greedy rows, top-k, top-p, both, neither; drafts
 * kept, dropped and absent; logits with ties and signed zeros. A vocabulary of 1,000 (every token
 * a candidate) and of 6,000 (4,096 candidates): flat rows that keep every token take the branch
 * over the whole vocabulary, peaked rows do not.
 */
class PjrtSampleTokensXlaTest {

    private fun fn(n: Int, v: Int): DxirFunction = DxirBuilder.function("sample") {
        val ins = listOf(
            param("logits", DxirType(F32, listOf(n, v))), param("draft", DxirType(I32, listOf(n))),
            param("temperature", DxirType(F32, listOf(n))), param("topP", DxirType(F32, listOf(n))),
            param("topK", DxirType(I32, listOf(n))), param("seed", DxirType(I32, listOf(n))),
            param("position", DxirType(I32, listOf(n))),
        )
        listOf(op(OpKind.SAMPLE_TOKENS, ins, DxirType(I32, listOf(n))))
    }

    private fun case(v: Int, flat: Boolean, rounds: Int) {
        val n = 8
        val rnd = Random(v + if (flat) 1 else 0)
        val f = fn(n, v)
        val text = f.toStablehlo()
        assertTrue("tlaloc_sample" !in text && "chlo.top_k" in text, "the StableHLO form, not the kernel")
        TestBackend.session().use { s ->
            var branchRows = 0
            repeat(rounds) { round ->
                val logits = FloatArray(n * v) { i ->
                    val r = i / v
                    when {
                        flat -> (rnd.nextInt(5) - 2) * 0.25f          // few values: ties everywhere
                        r % 3 == 0 -> if (rnd.nextInt(7) == 0) 0f else -0f  // signed zeros
                        else -> (rnd.nextFloat() * 12f - 6f) * (if (i % v < 30) 2f else 1f)
                    }
                }
                val temps = FloatArray(n) { if (it == 0) 0f else listOf(0.3f, 0.8f, 1f, 1.7f)[it % 4] }
                val topP = FloatArray(n) { listOf(1f, 0.9f, 0.5f, 0.05f)[(it + round) % 4] }
                val topK = IntArray(n) { listOf(0, 20, 1, v + 5, 300)[(it + round) % 5] }
                val seed = IntArray(n) { rnd.nextInt(1 shl 24) }
                val pos = IntArray(n) { rnd.nextInt(1 shl 20) }
                // Drafts: the argmax (likely kept), a random id, none.
                val draft = IntArray(n) { r ->
                    when ((r + round) % 3) {
                        0 -> (0 until v).maxByOrNull { logits[r * v + it] }!!
                        1 -> rnd.nextInt(v)
                        else -> -1
                    }
                }
                val p = SampleTokensAttrs.Parsed(n, v)
                val want = SampleTokensAttrs.reference(p, logits, draft, temps, topP, topK, seed, pos)
                val got = s.runOnHost(f, listOf<Any>(logits, draft, temps, topP, topK, seed, pos)).single() as IntArray
                assertContentEquals(want, got, "V=$v flat=$flat round $round")
                if (v > SampleTokensAttrs.XLA_CANDIDATES) {
                    branchRows += (0 until n).count { r -> temps[r] > 0f && (topK[r] <= 0 || topK[r] > SampleTokensAttrs.XLA_CANDIDATES) }
                }
            }
            println("[sample-xla] V=$v flat=$flat: $rounds rounds of $n rows equal the reference ($branchRows rows that may keep more than the candidates)")
        }
    }

    @Test
    fun theStableHloFormDrawsTheReferenceTokens() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        case(1000, flat = false, rounds = 6)
        case(1000, flat = true, rounds = 4)
        case(6000, flat = false, rounds = 6)
        case(6000, flat = true, rounds = 4)
    }

    /** Time per call at Qwen3.6-35B-A3B's vocabulary, 16 rows (4 sequences of a verify step). TLALOC_SAMPLE_BENCH=1. */
    @Test
    fun timeAtTheVocabularyOfQwen36() {
        assumeTrue(System.getenv("TLALOC_SAMPLE_BENCH") == "1", "set TLALOC_SAMPLE_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val n = 16
        val v = 248320
        val rnd = Random(5)
        val f = fn(n, v)
        val mlir = f.toStablehlo()
        TestBackend.session().use { s ->
            // LM-shaped rows: a few tokens well above a broad mass.
            val logits = s.bufferFromHostF32(FloatArray(n * v) { i -> if (i % v < 40) 8f + rnd.nextFloat() * 6f else rnd.nextFloat() * 4f - 2f }, listOf(n, v))
            for ((label, k, pp) in listOf(Triple("top-k 20, top-p 0.9", 20, 0.9f), Triple("top-p 0.9 alone", 0, 0.9f))) {
                val ins = listOf(
                    logits, s.bufferFromHostI32(IntArray(n) { 3 }, listOf(n)), s.bufferFromHostF32(FloatArray(n) { 0.7f }, listOf(n)),
                    s.bufferFromHostF32(FloatArray(n) { pp }, listOf(n)), s.bufferFromHostI32(IntArray(n) { k }, listOf(n)),
                    s.bufferFromHostI32(IntArray(n) { it }, listOf(n)), s.bufferFromHostI32(IntArray(n) { 100 + it }, listOf(n)),
                )
                s.prepareStablehlo(mlir)
                repeat(3) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                val iters = 20
                val t0 = System.nanoTime()
                repeat(iters) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                println("[sample-xla-time] N=$n V=$v $label: %.3f ms per call".format((System.nanoTime() - t0) / 1e6 / iters))
            }
        }
    }
}
