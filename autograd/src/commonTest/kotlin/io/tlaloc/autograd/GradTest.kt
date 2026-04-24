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

    // §0.4.63 — unary math surface on Tracer. These wrap VjpRegistry rules that
    // have existed since §0.4.22 but had no tape path until now.

    @Test
    fun sqrtBackwardIsHalfOverSqrt() {
        // d/dx sqrt(x) = 1 / (2 sqrt(x)). Forward value and backward grad both pinned.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.sqrt() }
        val (value, dx) = vg(Tensors.f32Scalar(4f))
        assertEquals(2f, value)
        assertEquals(0.25f, dx.hostF32()[0])  // 1 / (2·2) = 0.25
    }

    @Test
    fun expBackwardIsExp() {
        // d/dx exp(x) = exp(x). At x = 0: forward = 1, grad = 1.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.exp() }
        val (v0, d0) = vg(Tensors.f32Scalar(0f))
        assertEquals(1f, v0)
        assertEquals(1f, d0.hostF32()[0])
        // At x = 1: forward = exp(1), grad = exp(1).
        val (v1, d1) = vg(Tensors.f32Scalar(1f))
        val e = kotlin.math.E.toFloat()
        assertTrue(abs(v1 - e) < 1e-5f, "expected exp(1) ≈ e; got $v1")
        assertTrue(abs(d1.hostF32()[0] - e) < 1e-5f, "expected grad(exp) = exp; got ${d1.hostF32()[0]}")
    }

    @Test
    fun logBackwardIsReciprocal() {
        // d/dx log(x) = 1/x. At x = 4: forward = ln(4) ≈ 1.386, grad = 0.25.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.log() }
        val (v, dx) = vg(Tensors.f32Scalar(4f))
        assertTrue(abs(v - kotlin.math.ln(4f)) < 1e-5f, "forward = ln(4); got $v")
        assertEquals(0.25f, dx.hostF32()[0])
    }

    @Test
    fun tanhBackwardIsOneMinusTanhSquared() {
        // d/dx tanh(x) = 1 - tanh(x)^2. At x = 0: forward = 0, grad = 1.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.tanh() }
        val (v0, d0) = vg(Tensors.f32Scalar(0f))
        assertEquals(0f, v0)
        assertEquals(1f, d0.hostF32()[0])
        // At x = 1: forward = tanh(1), grad = 1 - tanh(1)^2.
        val (v1, d1) = vg(Tensors.f32Scalar(1f))
        val t1 = kotlin.math.tanh(1f)
        assertTrue(abs(v1 - t1) < 1e-5f, "forward = tanh(1); got $v1")
        assertTrue(abs(d1.hostF32()[0] - (1f - t1 * t1)) < 1e-5f, "grad = 1 - tanh^2; got ${d1.hostF32()[0]}")
    }

    @Test
    fun sigmoidBackwardIsSigmoidTimesOneMinusSigmoid() {
        // d/dx σ(x) = σ(x)(1 - σ(x)). At x = 0: forward = 0.5, grad = 0.25.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.sigmoid() }
        val (v0, d0) = vg(Tensors.f32Scalar(0f))
        assertEquals(0.5f, v0)
        assertEquals(0.25f, d0.hostF32()[0])
        // Large positive: σ ≈ 1, grad ≈ 0.
        val (_, dLarge) = vg(Tensors.f32Scalar(10f))
        assertTrue(dLarge.hostF32()[0] < 1e-3f, "grad should be ~0 at x=10; got ${dLarge.hostF32()[0]}")
    }

    // §0.4.64 — `Tracer.pow(other)` closes the last VjpRegistry rule without a
    // Tracer surface wrapper. Both operands are differentiable, so valueAndGrad2
    // seeds both sides of the reverse walk.

    // §0.4.65 — `Tracer.constant(f)` wraps a scalar leaf flagged opaque so
    // Backward skips materialising any grad contribution targeting it. Confirms
    // both ergonomics (shorter call site) and correctness (the grad wrt x still
    // agrees with the two-leaf variant from §0.4.64).

    @Test
    fun constantTracerEnablesSquaringWithoutExpLeaf() {
        // f(x) = x^2 using `x.constant(2f)`. Expected grad_x = 2x. At x = 3:
        //   value = 9, grad = 6. Identical to §0.4.64's two-leaf variant but
        //   the constant doesn't surface a second gradient output to the user.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.pow(x.constant(2f)) }
        val (value, dx) = vg(Tensors.f32Scalar(3f))
        assertEquals(9f, value)
        assertEquals(6f, dx.hostF32()[0])
    }

    @Test
    fun constantTracerMatchesTwoLeafVariant() {
        // Cross-check: x.pow(x.constant(2f)) and valueAndGrad2 { x, e -> x.pow(e) }
        // at e=2 give the same grad_x. Pins the constant-skip optimisation as
        // a no-op on user-visible output.
        val withConstant = valueAndGrad { x: Tracer<ScalarShape> -> x.pow(x.constant(2f)) }
        val twoLeaves = valueAndGrad2 { x: Tracer<ScalarShape>, e: Tracer<ScalarShape> -> x.pow(e) }
        for (xv in listOf(1f, 2f, 4f, 0.5f)) {
            val (_, dxCon) = withConstant(Tensors.f32Scalar(xv))
            val (_, dxTwo, _) = twoLeaves(Tensors.f32Scalar(xv), Tensors.f32Scalar(2f))
            assertEquals(dxTwo.hostF32()[0], dxCon.hostF32()[0], "grad_x mismatch at x=$xv")
        }
    }

    // §0.4.70 — rank-N `constant(FloatArray, IntArray)` generalises §0.4.67 and
    // §0.4.65. Caller supplies flat values + dims; the phantom Shape type is
    // picked via type inference at the call site.

    @Test
    fun constantRank2MatmulProducesExpectedGradient() {
        // x: rank-1 [3], m: rank-2 [3, 1] constant. Forward: (matrix · vector-as-matrix).sum().
        // Simpler: wire x through two rank-1 constants as a dot-product-style reduction.
        //
        // For this test, exercise constant(FloatArray, IntArray) at rank-1 via the
        // generic overload — picks the same Tracer<Rank1<Sym>> shape §0.4.67's
        // dedicated overload produces, but through the type-inference path.
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            val c: Tracer<io.tlaloc.core.Rank1<Sym>> =
                x.constant(floatArrayOf(2f, 3f, 4f), intArrayOf(3))
            (x * c).sum()
        }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(10f, 20f, 30f)))
        // sum(10·2 + 20·3 + 30·4) = 20 + 60 + 120 = 200
        assertEquals(200f, value)
        val g = dx.hostF32()
        // d/dx_i sum(x_i · c_i) = c_i → [2, 3, 4]
        assertEquals(2f, g[0])
        assertEquals(3f, g[1])
        assertEquals(4f, g[2])
    }

    @Test
    fun constantRankNValidatesDimsProduct() {
        // Size mismatch: 3 values against a 2x2 dims claim should fail loudly.
        var caught: IllegalArgumentException? = null
        grad { x: Tracer<ScalarShape> ->
            try {
                val _m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> =
                    x.constant(floatArrayOf(1f, 2f, 3f), intArrayOf(2, 2))
            } catch (e: IllegalArgumentException) {
                caught = e
            }
            x
        }(Tensors.f32Scalar(1f))
        assertTrue(caught != null, "expected IllegalArgumentException for size mismatch")
        assertTrue(
            caught!!.message!!.contains("doesn't match"),
            "message should name the mismatch; got: ${caught!!.message}",
        )
    }

    @Test
    fun constantRankNRejectsEmptyDims() {
        // Empty dims → would silently become ScalarShape; force the user to use
        // the dedicated `constant(Float)` instead.
        var caught: IllegalArgumentException? = null
        grad { x: Tracer<ScalarShape> ->
            try {
                val _c: Tracer<io.tlaloc.core.Rank1<Sym>> =
                    x.constant(floatArrayOf(1f), IntArray(0))
            } catch (e: IllegalArgumentException) {
                caught = e
            }
            x
        }(Tensors.f32Scalar(1f))
        assertTrue(caught != null, "expected IllegalArgumentException for empty dims")
        assertTrue(
            caught!!.message!!.contains("empty dims"),
            "message should name the failure mode; got: ${caught!!.message}",
        )
    }

    @Test
    fun constantLikeEnablesRank1Offset() {
        // §0.4.69 — `x.constantLike(5f)` creates a rank-1 constant with the same
        // shape as `x`. Lets `x + x.constantLike(5f)` work without broadcasting:
        // both operands are same-shape, so the existing `plus` applies directly.
        // For x = [1, 2, 3]: f(x) = sum(x + 5) = 1+2+3 + 15 = 21, grad_x = [1,1,1].
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + x.constantLike(5f)).sum()
        }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(21f, value)
        val g = dx.hostF32()
        assertEquals(1f, g[0])
        assertEquals(1f, g[1])
        assertEquals(1f, g[2])
    }

    @Test
    fun constantLikePreservesShape() {
        // Matches receiver shape across different ranks: scalar and rank-1.
        val vgScalar = valueAndGrad { x: Tracer<ScalarShape> -> x * x.constantLike(3f) }
        val (v, dx) = vgScalar(Tensors.f32Scalar(2f))
        assertEquals(6f, v)
        assertEquals(3f, dx.hostF32()[0], "grad through (x * 3) = 3")

        val vgRank1 = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x * x.constantLike(0.5f)).sum()
        }
        val (v1, d1) = vgRank1(Tensors.f32Vector(floatArrayOf(4f, 8f, 12f)))
        assertEquals(12f, v1)  // (4 + 8 + 12) * 0.5 = 12
        val g1 = d1.hostF32()
        assertEquals(0.5f, g1[0])
        assertEquals(0.5f, g1[1])
        assertEquals(0.5f, g1[2])
    }

    @Test
    fun constantLikeIsNonDifferentiable() {
        // The constantLike leaf is flagged `isConstant = true`, so even if the
        // user deliberately referenced the returned Tracer elsewhere (say in a
        // product with itself), its contribution to backward doesn't show up
        // on any Tracer the user can retrieve via `grad` / `valueAndGrad`.
        // This test pins: grad_x through `x + c + c` where `c = x.constantLike(5f)`
        // is still 1 per element, not 1 + <something involving c's nonexistent grad>.
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            val c = x.constantLike(5f)
            (x + c + c).sum()  // = sum(x) + 2 · sum(c) — grad_x is still [1, 1, 1]
        }
        val (_, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 1f)))
        val g = dx.hostF32()
        assertEquals(1f, g[0])
        assertEquals(1f, g[1])
    }

    @Test
    fun constantRank1PerElementExponentSchedule() {
        // §0.4.67 — rank-1 `x.constant(FloatArray)` lets each element of a
        // rank-1 tracer get its own exponent. For x = [2, 3, 4] with
        // exponents [3, 2, 1], `sum(x^e)` = 8 + 9 + 4 = 21.
        // grad_x[i] = e[i] · x[i]^(e[i] - 1) = [3·4, 2·3, 1·1] = [12, 6, 1].
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            x.pow(x.constant(floatArrayOf(3f, 2f, 1f))).sum()
        }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(2f, 3f, 4f)))
        assertEquals(21f, value)
        val g = dx.hostF32()
        assertTrue(abs(g[0] - 12f) < 1e-5f, "grad_x[0] = 12; got ${g[0]}")
        assertTrue(abs(g[1] - 6f) < 1e-5f, "grad_x[1] = 6; got ${g[1]}")
        assertTrue(abs(g[2] - 1f) < 1e-5f, "grad_x[2] = 1; got ${g[2]}")
    }

    @Test
    fun constantRank1DefensivelyCopiesInput() {
        // Pin that a caller mutation to the passed-in array doesn't desync
        // the tape's cached values. Subtle gotcha if `constant(FloatArray)`
        // stored the reference — a grad computed after the mutation would
        // use stale values.
        val g = grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            val src = floatArrayOf(1f, 1f)
            val c = x.constant(src)
            src[0] = 999f  // mutate AFTER handing off to constant()
            src[1] = 999f
            (x + c).sum()  // grad_x should still be [1, 1] — c's values
            // frozen at constant() time. `c` is non-differentiable so the
            // mutation doesn't even appear in the value computation.
        }
        val out = g(Tensors.f32Vector(floatArrayOf(5f, 10f)))
        val arr = out.hostF32()
        assertEquals(1f, arr[0])
        assertEquals(1f, arr[1])
    }

    @Test
    fun constantRank1RejectsEmptyArray() {
        // Empty arrays can't form a rank-1 tensor — pin the require() fires
        // loudly instead of producing a zero-dim runtime surprise.
        var caught: IllegalArgumentException? = null
        grad { x: Tracer<ScalarShape> ->
            try {
                x.constant(FloatArray(0))
            } catch (e: IllegalArgumentException) {
                caught = e
            }
            x
        }(Tensors.f32Scalar(1f))
        assertTrue(caught != null, "expected IllegalArgumentException for empty FloatArray")
        assertTrue(caught!!.message!!.contains("empty"), "message should name the failure; got: ${caught!!.message}")
    }

    @Test
    fun constantTracerAsAdditiveOffsetLeavesGradUnchanged() {
        // Adding a constant offset shifts forward but leaves grad_x at 1.
        // Pins that `.constant()` composes with arithmetic operators, not only
        // with pow.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x + x.constant(5f) }
        val (v, dx) = vg(Tensors.f32Scalar(3f))
        assertEquals(8f, v)
        assertEquals(1f, dx.hostF32()[0], "grad_x through (x + const) should be 1")
    }

    @Test
    fun powBackwardAtIntegerExponent() {
        // f(x, e) = x^e. At x = 3, e = 2:
        //   value = 9
        //   grad_x = e · x^(e-1) = 2 · 3^1 = 6
        //   grad_e = x^e · ln(x) = 9 · ln 3
        val vg = valueAndGrad2 { x: Tracer<ScalarShape>, e: Tracer<ScalarShape> -> x.pow(e) }
        val (value, dx, de) = vg(Tensors.f32Scalar(3f), Tensors.f32Scalar(2f))
        assertEquals(9f, value)
        assertEquals(6f, dx.hostF32()[0])
        val expected = 9f * kotlin.math.ln(3f)
        assertTrue(
            abs(de.hostF32()[0] - expected) < 1e-4f,
            "grad_e = x^e · ln(x) ≈ $expected; got ${de.hostF32()[0]}",
        )
    }

    @Test
    fun powBackwardAtFractionalExponent() {
        // f(x, e) = x^e. At x = 4, e = 0.5 (square root):
        //   value = 2
        //   grad_x = 0.5 · 4^(-0.5) = 0.25
        //   grad_e = 2 · ln(4)
        val vg = valueAndGrad2 { x: Tracer<ScalarShape>, e: Tracer<ScalarShape> -> x.pow(e) }
        val (value, dx, de) = vg(Tensors.f32Scalar(4f), Tensors.f32Scalar(0.5f))
        assertEquals(2f, value)
        assertTrue(abs(dx.hostF32()[0] - 0.25f) < 1e-6f, "grad_x ≈ 0.25; got ${dx.hostF32()[0]}")
        val expected = 2f * kotlin.math.ln(4f)
        assertTrue(
            abs(de.hostF32()[0] - expected) < 1e-4f,
            "grad_e = 2·ln(4) ≈ $expected; got ${de.hostF32()[0]}",
        )
    }

    @Test
    fun powRank1BroadcastsElementwise() {
        // f(x, e) = sum(x^e) with same-shape rank-1 operands.
        // At x = [2, 3], e = [3, 2]:
        //   value = 8 + 9 = 17
        //   grad_x_i = e_i · x_i^(e_i - 1) = [3·4, 2·3] = [12, 6]
        //   grad_e_i = x_i^{e_i} · ln(x_i) = [8·ln2, 9·ln3]
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, e: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            x.pow(e).sum()
        }
        val (value, dx, de) = vg(
            Tensors.f32Vector(floatArrayOf(2f, 3f)),
            Tensors.f32Vector(floatArrayOf(3f, 2f)),
        )
        assertEquals(17f, value)
        val gx = dx.hostF32()
        assertTrue(abs(gx[0] - 12f) < 1e-5f, "grad_x[0] = 12; got ${gx[0]}")
        assertTrue(abs(gx[1] - 6f) < 1e-5f, "grad_x[1] = 6; got ${gx[1]}")
        val ge = de.hostF32()
        val expected0 = 8f * kotlin.math.ln(2f)
        val expected1 = 9f * kotlin.math.ln(3f)
        assertTrue(abs(ge[0] - expected0) < 1e-4f, "grad_e[0] ≈ $expected0; got ${ge[0]}")
        assertTrue(abs(ge[1] - expected1) < 1e-4f, "grad_e[1] ≈ $expected1; got ${ge[1]}")
    }

    @Test
    fun chainedUnaryMathRoundTrips() {
        // f(x) = sqrt(exp(x)); f'(x) = sqrt(exp(x)) / 2 = f(x) / 2. At x = 0:
        // f = 1, f' = 0.5.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.exp().sqrt() }
        val (v, dx) = vg(Tensors.f32Scalar(0f))
        assertEquals(1f, v)
        assertEquals(0.5f, dx.hostF32()[0])
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
        //
        // §0.4.59 — uses `d.peek()` instead of `d.entry.value[0]`; public API
        // replaces the internal backing-array reach-through.
        val g = grad { x: Tracer<ScalarShape> ->
            var d = x
            while (d.scalar <= 10f) {
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
            while (d.scalar <= 10f) {
                d = d + d
            }
            d
        }
        assertEquals(4f, g(Tensors.f32Scalar(3f)).hostF32()[0], "x=3: 2 doublings → df/dx = 4")
        assertEquals(32f, g(Tensors.f32Scalar(0.5f)).hostF32()[0], "x=0.5: 5 doublings → df/dx = 32")
        assertEquals(1f, g(Tensors.f32Scalar(11f)).hostF32()[0], "x=11: 0 doublings → df/dx = 1")
    }

    @Test
    fun peekReturnsCurrentForwardValueForScalar() {
        // §0.4.59 — `peek()` reads the tape-recorded forward value without
        // allocating. Exercised inside a traced function so `d.peek()` returns
        // the rolling value of `d = d + d` as the tape accrues ADDs.
        val recorded = mutableListOf<Float>()
        val g = grad { x: Tracer<ScalarShape> ->
            var d = x
            recorded.add(d.peek())
            d = d + d
            recorded.add(d.peek())
            d = d + d
            recorded.add(d.peek())
            d
        }
        g(Tensors.f32Scalar(1.5f))
        assertEquals(listOf(1.5f, 3f, 6f), recorded)
    }

    @Test
    fun scalarPropertyDelegatesToPeek() {
        // §0.4.62 — `Tracer<ScalarShape>.scalar` is a type-safe extension that
        // compiles only on rank-0 tracers. Behaviour equals `peek()`; pin it
        // so the shortcut doesn't silently diverge from the underlying read.
        val observed = mutableListOf<Float>()
        val g = grad { x: Tracer<ScalarShape> ->
            var d = x
            observed.add(d.scalar)
            d = d + d
            observed.add(d.scalar)
            d
        }
        g(Tensors.f32Scalar(2.5f))
        assertEquals(listOf(2.5f, 5f), observed)
        // `scalar` must agree with `peek()` at every step.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x + x + x }
        val (value, _) = vg(Tensors.f32Scalar(4f))
        assertEquals(12f, value)
    }

    @Test
    fun peekRejectsOutOfBoundsIndex() {
        // §0.4.59 — `peek(index)` validates bounds. Pins the contract so future
        // refactors can't silently return stale/garbage floats for bad indices.
        var caught: IllegalArgumentException? = null
        grad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            try {
                x.peek(99)
            } catch (e: IllegalArgumentException) {
                caught = e
            }
            x.sum()
        }(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertTrue(caught != null, "expected IllegalArgumentException for peek(99) on size-3 tracer")
        assertTrue(caught!!.message!!.contains("out of bounds"),
            "message should name the failure mode; got: ${caught!!.message}")
    }

    @Test
    fun breakBearingWhileValueAndGradAgree() {
        // Pair with `valueAndGrad` to verify the forward value is what the reverse
        // path actually differentiates — no silent divergence between tape-recorded
        // value and the result used as the backward seed.
        val vg = valueAndGrad { x: Tracer<ScalarShape> ->
            var d = x
            while (d.scalar <= 10f) {
                d = d + d
            }
            d
        }
        val (value, dx) = vg(Tensors.f32Scalar(0.5f))
        assertEquals(16f, value, "x=0.5 · 2^5 = 16")
        assertEquals(32f, dx.hostF32()[0], "df/dx = 2^5 = 32")
    }
}
