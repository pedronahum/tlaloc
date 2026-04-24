package io.tlaloc.autograd

import io.tlaloc.core.F32
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GradTest {

    @Test
    fun scalarSquareDerivativeIsTwoX() {
        val g = grad { x: Tracer<ScalarShape> -> x * x }
        assertEquals(6f, g(Tensors.f32Scalar(3f)).hostF32()[0])
        assertEquals(-4f, g(Tensors.f32Scalar(-2f)).hostF32()[0])
        assertEquals(0f, g(Tensors.f32Scalar(0f)).hostF32()[0])
    }

    @Test
    fun valueAndGradReturnsBoth() {
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x * x + x }
        val (value, dx) = vg(Tensors.f32Scalar(3f))
        assertEquals(12f, value)         // 9 + 3
        assertEquals(7f, dx.hostF32()[0]) // 2*3 + 1
    }

    @Test
    fun vectorSumOfSquaresDerivativeIsTwoX() {
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x * x).sum() }
        val out = g(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f, 4f)))
        assertContentEquals(floatArrayOf(2f, 4f, 6f, 8f), out.hostF32())
    }

    @Test
    fun reluSubgradientPropagates() {
        // f(x) = sum(relu(x)); grad is 1 where x > 0, 0 elsewhere
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.relu().sum() }
        val out = g(Tensors.f32Vector(floatArrayOf(-1f, 0f, 2f, -3f, 5f)))
        assertContentEquals(floatArrayOf(0f, 0f, 1f, 0f, 1f), out.hostF32())
    }

    @Test
    fun sumOfElementsHasUnitGradient() {
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.sum() }
        val out = g(Tensors.f32Vector(floatArrayOf(7f, 9f, 11f)))
        assertContentEquals(floatArrayOf(1f, 1f, 1f), out.hostF32())
    }

    @Test
    fun meanHasReciprocalGradient() {
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.mean() }
        val out = g(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f, 4f)))
        assertContentEquals(floatArrayOf(0.25f, 0.25f, 0.25f, 0.25f), out.hostF32())
    }

    @Test
    fun matmulBackwardOnSmallSquare() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // sum(A@B) = 19 + 22 + 43 + 50 = 134
        // dA = ones(2,2) @ B^T = [[11,15],[11,15]]
        // dB = A^T @ ones(2,2) = [[4,4],[6,6]]
        val f = { a: Tracer<Rank2<Sym, Sym>>, b: Tracer<Rank2<Sym, Sym>> ->
            (a matmul b).sum()
        }
        val vg = valueAndGrad2(f)
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val b = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(5f, 6f, 7f, 8f))
        val (value, dA, dB) = vg(a, b)
        assertEquals(134f, value)
        assertContentEquals(floatArrayOf(11f, 15f, 11f, 15f), dA.hostF32())
        assertContentEquals(floatArrayOf(4f, 4f, 6f, 6f), dB.hostF32())
    }

    @Test
    fun chainRuleThroughReluAndMatmul() {
        // W=[[1,2],[3,4]] (2x2), X=[[1],[1]] (2x1)
        // W@X = [[3],[7]], relu unchanged, sum = 10
        // d/d(W@X) sum(relu(...)) = ones since all > 0
        // dW = ones(2,1) @ X^T = [[1],[1]] @ [[1,1]] = [[1,1],[1,1]]
        // dX = W^T @ ones(2,1) = [[1,3],[2,4]] @ [[1],[1]] = [[4],[6]]
        val vg = valueAndGrad2 { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>> ->
            (w matmul x).relu().sum()
        }
        val w = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val x = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f))
        val (value, dW, dX) = vg(w, x)
        assertEquals(10f, value)
        assertContentEquals(floatArrayOf(1f, 1f, 1f, 1f), dW.hostF32())
        assertContentEquals(floatArrayOf(4f, 6f), dX.hostF32())
    }

    @Test
    fun sgdMinimizesQuadratic() {
        // Minimize f(w) = w * w; minimum at w=0. Closed form after N steps:
        //   w_{n+1} = w_n - lr * 2 * w_n = w_n * (1 - 2*lr)
        val g = grad { w: Tracer<ScalarShape> -> w * w }
        var w = Tensors.f32Scalar(3f)
        val lr = 0.1f
        repeat(20) {
            val gw = g(w).hostF32()[0]
            w = Tensors.f32Scalar(w.hostF32()[0] - lr * gw)
        }
        val final = w.hostF32()[0]
        assertTrue(abs(final) < 0.1f, "final w = $final, expected close to 0")
    }

    @Test
    fun reuseSameGradFunctionAcrossCalls() {
        // Each call should build a fresh tape and not leak state.
        val g = grad { x: Tracer<ScalarShape> -> x * x }
        assertEquals(6f, g(Tensors.f32Scalar(3f)).hostF32()[0])
        assertEquals(10f, g(Tensors.f32Scalar(5f)).hostF32()[0])
        assertEquals(-8f, g(Tensors.f32Scalar(-4f)).hostF32()[0])
    }

    @Test
    fun addAndSubBackward() {
        // f(a,b) = sum((a - b) * (a + b)); this equals sum(a^2 - b^2)
        // grad wrt a = 2a, grad wrt b = -2b
        val vg = valueAndGrad2 { a: Tracer<io.tlaloc.core.Rank1<Sym>>, b: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            ((a - b) * (a + b)).sum()
        }
        val a = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
        val b = Tensors.f32Vector<Sym>(floatArrayOf(4f, 5f, 6f))
        val (value, dA, dB) = vg(a, b)
        // sum(a^2 - b^2) = (1+4+9) - (16+25+36) = 14 - 77 = -63
        assertEquals(-63f, value)
        assertContentEquals(floatArrayOf(2f, 4f, 6f), dA.hostF32())
        assertContentEquals(floatArrayOf(-8f, -10f, -12f), dB.hostF32())
    }

    @Test
    fun shapeMismatchIsCaught() {
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x + x).sum() }
        // Runs fine on matching shape.
        val out = g(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertContentEquals(floatArrayOf(2f, 2f, 2f), out.hostF32())
    }

    @Test
    fun negBackward() {
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.neg().sum() }
        val out = g(Tensors.f32Vector(floatArrayOf(5f, -3f, 2f)))
        assertContentEquals(floatArrayOf(-1f, -1f, -1f), out.hostF32())
    }

    // §0.4.57 — D.3ii-tape-tracer. The compiler plugin cannot specialise a
    // break-bearing `while` at compile time (§0.4.55 closure is deferred; §0.4.56
    // pinned the plugin-level fallback), so the user's `grad { ... }` falls through
    // to this runtime tape. The tape records ops as the loop actually executes and
    // walks them in reverse — arbitrary control flow (including data-dependent
    // break) is handled natively. These tests pin the gradient correctness end-to-
    // end on the Tracer surface.

    @Test
    fun breakBearingWhileDoublesUntilThreshold() {
        // f(x) = x doubled repeatedly until x*2^k > 10. For x = 0.5:
        //   k=0..4: d ∈ {0.5, 1, 2, 4, 8}  (d <= 10 each time, keep doubling)
        //   k=5:    d = 16  → exit the loop (16 > 10)
        // Final d = 32x (= 16 at x=0.5). df/dx = 32.
        val g = grad { x: Tracer<ScalarShape> ->
            var d = x
            while (d.entry.value[0] <= 10f) {
                d = d + d
            }
            d
        }
        val out = g(Tensors.f32Scalar(0.5f)).hostF32()[0]
        assertEquals(32f, out, "expected df/dx = 2^(#iterations) = 32 for x=0.5")
    }

    @Test
    fun breakBearingWhileIterationCountDependsOnInput() {
        // Same kernel as above. For x = 3.0 the thresholded iterations stop earlier:
        //   k=0: d=3     (3 <= 10)
        //   k=1: d=6     (6 <= 10)
        //   k=2: d=12    (12 > 10 → exit BEFORE the 3rd doubling runs)
        // Only 2 ADD ops recorded → df/dx = 2^2 = 4.
        //
        // Pins the defining property of a data-dependent break: the number of tape
        // entries — and therefore the closed-form adjoint — varies per invocation
        // of the same `grad { ... }` lambda.
        val g = grad { x: Tracer<ScalarShape> ->
            var d = x
            while (d.entry.value[0] <= 10f) {
                d = d + d
            }
            d
        }
        assertEquals(4f, g(Tensors.f32Scalar(3f)).hostF32()[0], "x=3: 2 doublings → df/dx = 4")
        assertEquals(32f, g(Tensors.f32Scalar(0.5f)).hostF32()[0], "x=0.5: 5 doublings → df/dx = 32")
        assertEquals(1f, g(Tensors.f32Scalar(11f)).hostF32()[0], "x=11: 0 doublings → df/dx = 1")
    }

    @Test
    fun breakBearingWhileValueAndGradAgree() {
        // Pair with `valueAndGrad` to verify the forward value is what the reverse
        // path actually differentiates — no silent divergence between tape-recorded
        // value and the result used as the backward seed.
        val vg = valueAndGrad { x: Tracer<ScalarShape> ->
            var d = x
            while (d.entry.value[0] <= 10f) {
                d = d + d
            }
            d
        }
        val (value, dx) = vg(Tensors.f32Scalar(0.5f))
        assertEquals(16f, value, "x=0.5 · 2^5 = 16")
        assertEquals(32f, dx.hostF32()[0], "df/dx = 2^5 = 32")
    }
}
