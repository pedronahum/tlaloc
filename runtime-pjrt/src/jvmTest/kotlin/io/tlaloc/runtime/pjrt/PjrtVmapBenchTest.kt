@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.DxirVmapTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Per-example gradients on the GPU: one batched program from `DxirVmapTransform` against
 * a loop that runs the single-example gradient program once per example. A measurement,
 * not a pass/fail claim about speed: it runs only with `TLALOC_VMAP_BENCH=1` and prints the
 * medians. The two must agree (1e-4 of the largest entry, TF32 dots off).
 *
 * Model: `sum(softmax(tanh(x · W1) · W2) ⊙ onehot)`, x [1, 64], W1 [64, 256], W2 [256, 10];
 * the gradient with respect to W1 per example.
 */
class PjrtVmapBenchTest {

    private fun t(vararg d: Int) = DxirType(F32, d.toList())

    private val loss: DxirFunction = DxirBuilder.function("mlpLoss") {
        val w1 = param("w1", t(64, 256))
        val w2 = param("w2", t(256, 10))
        val x = param("x", t(1, 64))
        val y = param("y", t(1, 10))
        val h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w1), t(1, 256))), t(1, 256))
        val p = op(OpKind.SOFTMAX, listOf(op(OpKind.MATMUL, listOf(h, w2), t(1, 10))), t(1, 10))
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(p, y), t(1, 10))), t()))
    }

    private fun data(n: Int, seed: Int) = FloatArray(n) { kotlin.math.sin(0.37 * it + seed).toFloat() * 0.3f }

    @Test
    fun `batched per-example gradients against a loop of single-example programs`() {
        assumeTrue(System.getenv("TLALOC_VMAP_BENCH") == "1", "TLALOC_VMAP_BENCH is not 1 — skipping the measurement")
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // grad with respect to w1; w2, x, y inputs only (trailing).
        val single = DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 3)
        TestBackend.session(portableF32Dots = true).use { session ->
            for (batch in listOf(16, 64, 256)) {
                val batched = DxirVmapTransform.apply(single, listOf(false, false, true, true), batch)
                val w1 = data(64 * 256, 1)
                val w2 = data(256 * 10, 2)
                val xs = data(batch * 64, 3)
                val ys = FloatArray(batch * 10) { if (it % 10 == (it / 10) % 10) 1f else 0f }
                fun runBatched() = session.runOn(batched, listOf(w1, w2, xs, ys))[0]
                fun runLoop(): FloatArray {
                    val out = FloatArray(batch * 64 * 256)
                    for (i in 0 until batch) {
                        val g = session.runOn(
                            single,
                            listOf(w1, w2, xs.copyOfRange(64 * i, 64 * i + 64), ys.copyOfRange(10 * i, 10 * i + 10)),
                        )[0]
                        g.copyInto(out, i * 64 * 256)
                    }
                    return out
                }
                val a = runBatched()
                val b = runLoop()
                val scale = maxOf(1f, b.maxOf { kotlin.math.abs(it) })
                val worst = a.indices.maxOf { kotlin.math.abs(a[it] - b[it]) } / scale
                assertTrue(worst <= 1e-4f, "batched and loop disagree by $worst")
                fun median(f: () -> Unit): Double {
                    val times = (0 until 7).map {
                        val t0 = System.nanoTime(); f(); (System.nanoTime() - t0) / 1e6
                    }.sorted()
                    return times[3]
                }
                val tb = median { runBatched() }
                val tl = median { runLoop() }
                println("vmap bench: batch $batch — one batched program %.2f ms, loop of $batch programs %.2f ms (%.1fx)".format(tb, tl, tl / tb))
            }
        }
    }
}
