package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.Activation
import io.tlaloc.nn.Dense
import io.tlaloc.nn.Precision
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.458 (G1d) — ONE mixed-precision training step on the compiled lane,
 * certified against the host interpreter. LOCAL certification (CUDA plugin
 * on the GB10 Blackwell); NO TPU claim — G2b re-runs this on libtpu.
 *
 * This is the PRODUCT path, not a hand-built twin: the graph under test is
 * `io.tlaloc.nn.capture(model, precision = MIXED_BF16)`'s own gradient
 * function — the injected boundary casts, the bf16 compute region, the f32
 * loss, `DxirReverseTransform`'s reverse — emitted through the :stablehlo
 * pipeline and executed by real XLA via [PjrtSession.runOn] (f32 buffers at
 * the boundary: the params ARE f32 master weights, so the session's f32 lane
 * is the honest transport; the bf16 lives inside the executable).
 *
 * THE DOCUMENTED FLOOR (from the §0.4.457 measurements, derived here rather
 * than guessed): the only certified device-vs-interpreter divergence source
 * for bf16 graphs is an ELIDED NARROWING — XLA's simplifier folds
 * f32→bf16→f32 convert pairs (composed with mul-by-one folding in reverse
 * graphs), so the device may read a RAW value where the interpreter reads
 * its snap; each elision moves a value by at most one bf16 ulp, 2⁻⁸
 * relative. A conservative count of elidable narrowings along any path is
 * the TOTAL number of bf16-typed ops in the gradient function (every bf16
 * op output is one potential snap the device may skip), so the per-tensor
 * tolerance is R·2⁻⁸·max(1, max|interpreter value|) with R counted on the
 * actual graph. Exact lanes (everything bf16-representable) must agree
 * BIT-FOR-BIT — elision of a no-op snap changes nothing, so any exact-lane
 * divergence would be an adjoint bug, not a rounding story.
 */
class PjrtMixedPrecisionSmokeTest {

    private fun tinyModel(
        w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray,
    ): Sequential = Sequential(
        Dense(
            Tensors.f32Matrix<Sym, Sym>(2, 2, w1),
            Tensors.f32Vector<Sym>(b1),
            Activation.Relu,
        ),
        Dense(
            Tensors.f32Matrix<Sym, Sym>(2, 1, w2),
            Tensors.f32Vector<Sym>(b2),
        ),
    )

    private fun mse(target: FloatArray, rows: Int): (io.tlaloc.autograd.Tracer<io.tlaloc.core.Shape>) -> io.tlaloc.autograd.Tracer<*> = { y ->
        val t = y.constant<io.tlaloc.core.Shape>(target, intArrayOf(rows, 1))
        val diff = y - t
        (diff * diff).mean()
    }

    @Test
    fun mixedTrainingStepOnGpuMatchesHostAtDocumentedFloor() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // The snap-lane model from the host suite: non-bf16-representable
        // values everywhere, so narrowings BITE and elision is observable.
        val w1 = floatArrayOf(0.3f, 1.2f, 0.7f, -0.35f)
        val b1 = floatArrayOf(0.1f, -0.2f)
        val w2 = floatArrayOf(1.1f, 0.9f)
        val b2 = floatArrayOf(0.05f)
        val xv = floatArrayOf(1.005f, -0.4f)
        val model = tinyModel(w1, b1, w2, b2)
        val x = Tensors.f32Matrix<Sym, Sym>(1, 2, xv)
        val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mse(floatArrayOf(0.5f), 1))

        // The step's value bindings, exactly as CapturedStep.run builds them:
        // inputs first, then parameters in key order (all f32 master values).
        val values = listOf(xv) + model.parameters.map { it.tensor.hostF32() }
        val want = DxirInterpreter.evalFunction(step.gradient, values)

        // R: the conservative elidable-narrowing count, from the graph itself.
        val r = step.gradient.body.filterIsInstance<DxirOp>().count { it.type.dtype == BF16 }
        assertTrue(r > 0, "the mixed gradient function carries no bf16 ops — the capture flag did nothing")

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(step.gradient, values)
            assertTrue(got.size == want.size, "output arity: GPU ${got.size} vs host ${want.size}")
            var anyNonZero = false
            for (k in want.indices) {
                var maxAbsWant = 0f
                for (v in want[k]) maxAbsWant = maxOf(maxAbsWant, abs(v))
                val tol = r * (1f / 256f) * maxOf(1f, maxAbsWant)
                for (i in want[k].indices) {
                    val diff = abs(got[k][i] - want[k][i])
                    if (got[k][i] != 0f) anyNonZero = true
                    assertTrue(
                        diff <= tol,
                        "output $k lane $i: GPU ${got[k][i]} vs interpreter ${want[k][i]} " +
                            "(|diff| $diff > $tol = R·2⁻⁸·scale, R=$r) — beyond the elided-narrowing floor",
                    )
                }
            }
            assertTrue(anyNonZero, "GPU mixed gradients all-zero — the silent-zero failure mode")

            // The training step itself: fold the GPU gradients into an SGD
            // update and compare against the host-gradient update — the
            // per-weight divergence is the gradient divergence scaled by lr,
            // covered by the same floor.
            val lr = 0.1f
            val keys = step.parameterKeys
            for (j in keys.indices) {
                val p = model.parameters[j]
                val gHost = want[1 + 1 + j] // (loss, dInput, dParams...)
                val gGpu = got[1 + 1 + j]
                val base = p.tensor.hostF32()
                var maxAbsG = 0f
                for (v in gHost) maxAbsG = maxOf(maxAbsG, abs(v))
                // update diff = lr·|Δg| ≤ lr·(gradient floor for this tensor).
                val tol = lr * r * (1f / 256f) * maxOf(1f, maxAbsG)
                for (i in base.indices) {
                    val uHost = base[i] - lr * gHost[i]
                    val uGpu = base[i] - lr * gGpu[i]
                    assertTrue(
                        abs(uHost - uGpu) <= tol,
                        "updated '${keys[j]}'[$i]: host $uHost vs GPU $uGpu beyond the scaled floor $tol",
                    )
                }
            }
            println(
                "[pjrt-mp] mixed-precision training step (nn capture, MIXED_BF16) on GB10: " +
                    "loss GPU=${got[0][0]} host=${want[0][0]}, R=$r bf16 ops, floor R·2⁻⁸·scale held",
            )
        }
    }

    /**
     * The exact lane on the compiled lane: every value bf16-representable,
     * every product/sum exact — the device must agree with the interpreter
     * BIT-FOR-BIT on the loss and every gradient (elidable no-op snaps are
     * invisible; any divergence here is an adjoint bug, not rounding).
     */
    @Test
    fun mixedExactLaneBitMatchesHostOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val model = tinyModel(
            floatArrayOf(0.5f, 2f, 1f, 0.25f), floatArrayOf(0.5f, -1f),
            floatArrayOf(2f, 0.5f), floatArrayOf(0.25f),
        )
        val xv = floatArrayOf(1f, 2f, 0.5f, -4f)
        val x = Tensors.f32Matrix<Sym, Sym>(2, 2, xv)
        val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mse(floatArrayOf(8f, 0f), 2))
        val values = listOf(xv) + model.parameters.map { it.tensor.hostF32() }
        val want = DxirInterpreter.evalFunction(step.gradient, values)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(step.gradient, values)
            for (k in want.indices) {
                for (i in want[k].indices) {
                    assertTrue(
                        got[k][i].toRawBits() == want[k][i].toRawBits(),
                        "exact lane output $k[$i]: GPU ${got[k][i]} != interpreter ${want[k][i]} — " +
                            "bf16-exact mixed steps must be engine-independent",
                    )
                }
            }
            assertTrue(
                want.drop(1).any { arr -> arr.any { it != 0f } },
                "exact lane vacuous — all gradients zero",
            )
            println("[pjrt-mp] exact-lane mixed step BIT-EXACT (loss + all gradients) vs interpreter on GB10")
        }
    }
}
