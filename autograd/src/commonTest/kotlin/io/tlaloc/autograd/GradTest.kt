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
}
