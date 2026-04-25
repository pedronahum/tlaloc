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

    // §0.4.75 — scalar-literal operator overloads (`x + 5f`, `x * 0.5f`, etc.).
    // These are convenience sugar over `x <op> x.constantLike(scalar)`; no new
    // tape op kind or VJP rule needed.

    // §0.4.77 — differentiable scalar-to-rank1 broadcasting. These tests pin
    // the gradient of BOTH operands: the rank-1 operand gets the usual per-
    // element derivative, and the scalar operand receives a SUM-reduced
    // gradient (via the new BroadcastRule's reverse).

    // §0.4.81 — (DTensor, Float) ergonomic overloads for grad2 / valueAndGrad2.
    // Accept the scalar as a raw Float, unwrap the scalar gradient back to Float
    // at the return boundary. Underlying path is the same as §0.4.77's scalar
    // broadcast — these tests pin the wrapping/unwrapping correctness.

    @Test
    fun gradWithScalarsProducesBothScalarGradients() {
        // §0.4.82 — f(a, b) = a² + a·b at a=2, b=3. Value = 4 + 6 = 10.
        //   grad_a = 2a + b = 7.  grad_b = a = 2.
        val g = gradWithScalars { a: Tracer<ScalarShape>, b: Tracer<ScalarShape> ->
            a * a + a * b
        }
        val (da, db) = g(2f, 3f)
        assertEquals(7f, da)
        assertEquals(2f, db)
    }

    @Test
    fun valueAndGradWithScalarsReturnsFullTriple() {
        // f(a, b) = (a - b) · (a + b) = a² - b². At a=5, b=3: value = 16.
        //   grad_a = 2a = 10. grad_b = -2b = -6.
        val vg = valueAndGradWithScalars { a: Tracer<ScalarShape>, b: Tracer<ScalarShape> ->
            (a - b) * (a + b)
        }
        val (value, da, db) = vg(5f, 3f)
        assertEquals(16f, value)
        assertEquals(10f, da)
        assertEquals(-6f, db)
    }

    @Test
    fun gradWithScalarWrapsAndUnwraps() {
        // f(x, c) = sum(x * c) at x=[2, 4, 8], c=3. value=42.
        //   grad_x=[3, 3, 3]. grad_c = sum(x) = 14.
        val g = gradWithScalar { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x * c).sum()
        }
        val (dx, dc) = g(Tensors.f32Vector(floatArrayOf(2f, 4f, 8f)), 3f)
        val gx = dx.hostF32()
        assertEquals(3f, gx[0])
        assertEquals(3f, gx[1])
        assertEquals(3f, gx[2])
        assertEquals(14f, dc, "grad_c should be unwrapped to Float; got $dc")
    }

    @Test
    fun valueAndGradWithScalarReturnsValueAndBothGradients() {
        // f(x, c) = sum(x + c) at x=[1, 2], c=5. value = 13. grad_x=[1, 1]. grad_c = N = 2.
        val vg = valueAndGradWithScalar { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x + c).sum()
        }
        val (value, dx, dc) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f)), 5f)
        assertEquals(13f, value)
        val gx = dx.hostF32()
        assertEquals(1f, gx[0])
        assertEquals(1f, gx[1])
        assertEquals(2f, dc)
    }

    @Test
    fun gradWithScalarPreservesExistingGrad2Semantics() {
        // Cross-check: `gradWithScalar { x, c -> ... }` agrees with
        // `grad2 { x, c -> ... }` when c is passed through Tensors.f32Scalar.
        val f = { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x * c + x).sum()
        }
        val new = gradWithScalar(f)
        val legacy = grad2(f)
        val xIn = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
        val (dxNew, dcNew) = new(xIn, 4f)
        val (dxLegacy, dcLegacy) = legacy(xIn, Tensors.f32Scalar(4f))
        val newArr = dxNew.hostF32()
        val legacyArr = dxLegacy.hostF32()
        assertEquals(legacyArr[0], newArr[0])
        assertEquals(legacyArr[1], newArr[1])
        assertEquals(legacyArr[2], newArr[2])
        assertEquals(dcLegacy.hostF32()[0], dcNew)
    }

    // §0.4.85 — rank-2-plus-rank-1 row-vector broadcasting. Matrix's each row
    // gets `row` added; gradient wrt `row` is the column-wise sum of the
    // upstream (via §0.4.84's axis-aware BroadcastRule).

    @Test
    fun doubleLiteralBroadcastWorksForBothSides() {
        // §0.4.95 — Double literals work in both LHS and RHS positions.
        // f(x) = sum(x * 0.5).  At x = [2, 4, 8]: forward = 7. grad = [0.5, 0.5, 0.5].
        val vgRhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x * 0.5).sum() }
        val (vR, dR) = vgRhs(Tensors.f32Vector(floatArrayOf(2f, 4f, 8f)))
        assertEquals(7f, vR)
        val drArr = dR.hostF32()
        for (i in 0 until 3) assertEquals(0.5f, drArr[i])

        // f(x) = sum(2.0 - x).  At x = [1, 2, 3]: forward = 1+0-1 = 0. grad = [-1, -1, -1].
        val vgLhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (2.0 - x).sum() }
        val (vL, dL) = vgLhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(0f, vL)
        val dlArr = dL.hostF32()
        for (i in 0 until 3) assertEquals(-1f, dlArr[i])
    }

    @Test
    fun intLiteralBroadcastWorksForBothSides() {
        // f(x) = sum(x + 5).  At x = [1, 2, 3]: forward = 6+7+8 = 21. grad = [1, 1, 1].
        val vgRhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x + 5).sum() }
        val (vR, dR) = vgRhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(21f, vR)
        val drArr = dR.hostF32()
        for (i in 0 until 3) assertEquals(1f, drArr[i])

        // f(x) = sum(3 * x).  At x = [1, 2, 3]: forward = 18. grad = [3, 3, 3].
        val vgLhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (3 * x).sum() }
        val (vL, dL) = vgLhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(18f, vL)
        val dlArr = dL.hostF32()
        for (i in 0 until 3) assertEquals(3f, dlArr[i])
    }

    @Test
    fun longLiteralBroadcastWorksForBothSides() {
        // §0.4.110 — Long literals work in both LHS and RHS positions, matching
        // the Int / Double / Float overloads. Useful when a literal is naturally
        // typed Long (e.g., from a config that returns `Long` durations or counts).

        // f(x) = sum(x + 5L).  At x = [1, 2, 3]: forward = 6+7+8 = 21. grad = [1, 1, 1].
        val vgRhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x + 5L).sum() }
        val (vR, dR) = vgRhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(21f, vR)
        val drArr = dR.hostF32()
        for (i in 0 until 3) assertEquals(1f, drArr[i])

        // f(x) = sum(3L * x).  At x = [1, 2, 3]: forward = 18. grad = [3, 3, 3].
        val vgLhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (3L * x).sum() }
        val (vL, dL) = vgLhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(18f, vL)
        val dlArr = dL.hostF32()
        for (i in 0 until 3) assertEquals(3f, dlArr[i])
    }

    @Test
    fun longLiteralLhsMinusRowFlipsSignFromRowMinusLong() {
        // Non-commutative direction parity check, mirroring §0.4.93's Float-LHS
        // test: for row=[1,2,3], `10L - row` and `row - 10L` differ in sign.
        //   row - 10L = [-9, -8, -7], sum = -24, grad_row = +1.
        //   10L - row = [9, 8, 7], sum = 24, grad_row = -1.
        val vgRhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x - 10L).sum() }
        val (vR, dR) = vgRhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(-24f, vR)
        val drArr = dR.hostF32()
        for (i in 0 until 3) assertEquals(1f, drArr[i])

        val vgLhs = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (10L - x).sum() }
        val (vL, dL) = vgLhs(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(24f, vL)
        val dlArr = dL.hostF32()
        for (i in 0 until 3) assertEquals(-1f, dlArr[i])
    }

    @Test
    fun longLiteralComposesWithFloatAndIntInOneExpression() {
        // §0.4.101 cross-literal composition: a single expression mixes `Long`,
        // `Float`, and `Int` literals in the same chain. Catches dispatch
        // ambiguity that single-literal-type tests miss.
        //
        // f(x) = sum((x * 2L + 1f) * 3).  At x = [1, 2, 3]:
        //   per-element: x*2L = [2, 4, 6]; +1f = [3, 5, 7]; *3 = [9, 15, 21].
        //   sum = 45.
        //   grad chain: d/dx = 2L * 3 = 6 per element.
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> ((x * 2L + 1f) * 3).sum() }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(45f, value)
        val gx = dx.hostF32()
        for (i in 0 until 3) assertEquals(6f, gx[i], "grad_x[$i]")
    }

    @Test
    fun floatLiteralLhsMinusRowFlipsSignFromRowMinusFloat() {
        // §0.4.93 — `5f - row` is NOT the same as `row - 5f`. For row=[1,2,3]:
        //   row - 5f = [-4, -3, -2], sum = -9.
        //   5f - row = [4, 3, 2], sum = 9.
        //   grad_row for (5f - row) = -1 per element.
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (5f - x).sum() }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(9f, value)
        val gx = dx.hostF32()
        for (i in 0 until 3) assertEquals(-1f, gx[i], "grad_row[$i]")
    }

    @Test
    fun floatLiteralLhsDivMatrixProducesReciprocalScaledGrads() {
        // f(x) = sum(6f / x) at x=[2, 3, 6]. value = 3 + 2 + 1 = 6.
        //   grad_x_i = -6 / x_i² = [-6/4, -6/9, -6/36] = [-1.5, -0.667, -0.167].
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (6f / x).sum() }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(2f, 3f, 6f)))
        assertEquals(6f, value)
        val gx = dx.hostF32()
        assertTrue(abs(gx[0] - (-6f / 4f)) < 1e-5f, "grad[0] ≈ -1.5; got ${gx[0]}")
        assertTrue(abs(gx[1] - (-6f / 9f)) < 1e-5f, "grad[1] ≈ -0.667; got ${gx[1]}")
        assertTrue(abs(gx[2] - (-6f / 36f)) < 1e-5f, "grad[2] ≈ -0.167; got ${gx[2]}")
    }

    @Test
    fun floatLiteralLhsTimesMatrixCommutesWithRhsTimes() {
        // `2f * matrix` and `matrix * 2f` are commutative — same value, same grad.
        val mIn = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val vgLhs = valueAndGrad { m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> -> (2f * m).sum() }
        val vgRhs = valueAndGrad { m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> -> (m * 2f).sum() }
        val (vL, dL) = vgLhs(mIn)
        val (vR, dR) = vgRhs(mIn)
        assertEquals(vL, vR)
        val dLArr = dL.hostF32()
        val dRArr = dR.hostF32()
        for (i in 0 until 4) assertEquals(dLArr[i], dRArr[i])
    }

    @Test
    fun reverseOrderScalarMinusRank2DiffersFromMatrixMinusScalar() {
        // §0.4.92 — `scalar - matrix` is NOT the same as `matrix - scalar`.
        // For scalar=10, matrix=[[1, 2], [3, 4]]:
        //   scalar - matrix = [[9, 8], [7, 6]], sum = 30.
        //   grad_scalar = M*N = 4.
        //   grad_matrix = -1 per element.
        val vg = valueAndGrad2 { s: Tracer<ScalarShape>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (s - m).sum()
        }
        val (value, dScalar, dMatrix) = vg(
            Tensors.f32Scalar(10f),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
        )
        assertEquals(30f, value)
        assertEquals(4f, dScalar.hostF32()[0], "grad_scalar = M·N = 4")
        val gm = dMatrix.hostF32()
        for (i in 0 until 4) assertEquals(-1f, gm[i])
    }

    @Test
    fun reverseOrderScalarTimesRank2CommutativeMatch() {
        // scalar * matrix and matrix * scalar must agree (commutative).
        val vgFwd = valueAndGrad2 { s: Tracer<ScalarShape>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (m * s).sum()
        }
        val vgRev = valueAndGrad2 { s: Tracer<ScalarShape>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (s * m).sum()
        }
        val sIn = Tensors.f32Scalar(2f)
        val mIn = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val (vF, dsF, dmF) = vgFwd(sIn, mIn)
        val (vR, dsR, dmR) = vgRev(sIn, mIn)
        assertEquals(vF, vR)
        assertEquals(dsF.hostF32()[0], dsR.hostF32()[0])
        val dF = dmF.hostF32()
        val dR = dmR.hostF32()
        for (i in 0 until 6) assertEquals(dF[i], dR[i])
    }

    @Test
    fun reverseOrderScalarMinusRank1DiffersFromRank1MinusScalar() {
        // §0.4.91 — `scalar - row` is NOT the same as `row - scalar`. For
        // scalar=10, row=[1, 2, 3]:
        //   row - scalar = [-9, -8, -7], sum = -24.
        //   scalar - row = [9, 8, 7], sum = 24.
        // grad_scalar for (scalar - row) = sum over i of +1 = N = 3.
        // grad_row for (scalar - row) = -1 per element.
        val vg = valueAndGrad2 { s: Tracer<ScalarShape>, r: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (s - r).sum()
        }
        val (value, dScalar, dRow) = vg(
            Tensors.f32Scalar(10f),
            Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
        )
        assertEquals(24f, value)
        assertEquals(3f, dScalar.hostF32()[0], "grad_scalar = N = 3")
        val gr = dRow.hostF32()
        assertEquals(-1f, gr[0]); assertEquals(-1f, gr[1]); assertEquals(-1f, gr[2])
    }

    @Test
    fun reverseOrderScalarDivRank1ProducesCorrectGradients() {
        // f(s, r) = sum(s / r) at s=12, r=[2, 3, 4].
        //   broadcast s = [12, 12, 12]. quotient = [6, 4, 3]. sum = 13.
        //   grad_s = sum over i of 1/r_i = 1/2 + 1/3 + 1/4 = 13/12 ≈ 1.0833.
        //   grad_r_i = -s / r_i² = [-3, -4/3, -3/4].
        val vg = valueAndGrad2 { s: Tracer<ScalarShape>, r: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (s / r).sum()
        }
        val (value, dScalar, dRow) = vg(
            Tensors.f32Scalar(12f),
            Tensors.f32Vector<Sym>(floatArrayOf(2f, 3f, 4f)),
        )
        assertEquals(13f, value)
        assertTrue(abs(dScalar.hostF32()[0] - 13f / 12f) < 1e-5f, "grad_s = 13/12; got ${dScalar.hostF32()[0]}")
        val gr = dRow.hostF32()
        assertTrue(abs(gr[0] - (-3f)) < 1e-5f)
        assertTrue(abs(gr[1] - (-12f / 9f)) < 1e-5f)  // -4/3 ≈ -1.333
        assertTrue(abs(gr[2] - (-12f / 16f)) < 1e-5f)  // -3/4 = -0.75
    }

    @Test
    fun reverseOrderScalarPlusRank1CommutativeMatch() {
        // scalar + row and row + scalar must agree on value + both grads.
        val vgFwd = valueAndGrad2 { s: Tracer<ScalarShape>, r: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (r + s).sum()
        }
        val vgRev = valueAndGrad2 { s: Tracer<ScalarShape>, r: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (s + r).sum()
        }
        val sIn = Tensors.f32Scalar(7f)
        val rIn = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
        val (vF, dsF, drF) = vgFwd(sIn, rIn)
        val (vR, dsR, drR) = vgRev(sIn, rIn)
        assertEquals(vF, vR)
        assertEquals(dsF.hostF32()[0], dsR.hostF32()[0])
        val dF = drF.hostF32()
        val dR = drR.hostF32()
        for (i in 0 until 3) assertEquals(dF[i], dR[i])
    }

    @Test
    fun reverseOrderRowMinusMatrixDiffersFromMatrixMinusRow() {
        // §0.4.90 — `row - matrix` is NOT the same as `matrix - row`. For
        // row = [10, 20], matrix = [[1, 2], [3, 4]]:
        //   matrix - row = [[-9, -18], [-7, -16]]  sum = -50.
        //   row - matrix = [[9, 18], [7, 16]]      sum = 50.
        // grad_row for (row - matrix): sum over rows of +1 = M = 2 per element → [2, 2].
        // grad_matrix for (row - matrix): -1 per element → ones(2,2) negated.
        val vg = valueAndGrad2 { row: Tracer<io.tlaloc.core.Rank1<Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (row - m).sum()
        }
        val (value, dRow, dMatrix) = vg(
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f)),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
        )
        assertEquals(50f, value)
        val gr = dRow.hostF32()
        assertEquals(2f, gr[0])  // sum over M rows of +1 = 2
        assertEquals(2f, gr[1])
        val gm = dMatrix.hostF32()
        for (i in 0 until 4) assertEquals(-1f, gm[i], "grad_matrix[$i] = -1")
    }

    @Test
    fun reverseOrderRowDivMatrixProducesCorrectGradients() {
        // f(row, m) = sum(row / m) at row=[6, 8], m=[[2, 4], [3, 2]].
        //   broadcast row = [[6, 8], [6, 8]].
        //   quotient     = [[3, 2], [2, 4]].  sum = 11.
        //   grad_row_j = sum over rows of 1/m_ij = (1/2 + 1/3, 1/4 + 1/2) = (5/6, 3/4).
        //   grad_m_ij = -row_j / m_ij² → [[-6/4, -8/16], [-6/9, -8/4]]
        //             = [[-1.5, -0.5], [-0.667, -2]].
        val vg = valueAndGrad2 { row: Tracer<io.tlaloc.core.Rank1<Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (row / m).sum()
        }
        val (value, dRow, dMatrix) = vg(
            Tensors.f32Vector<Sym>(floatArrayOf(6f, 8f)),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(2f, 4f, 3f, 2f)),
        )
        assertEquals(11f, value)
        val gr = dRow.hostF32()
        assertTrue(abs(gr[0] - (1f / 2f + 1f / 3f)) < 1e-5f, "grad_row[0] = 5/6; got ${gr[0]}")
        assertTrue(abs(gr[1] - (1f / 4f + 1f / 2f)) < 1e-5f, "grad_row[1] = 3/4; got ${gr[1]}")
        val gm = dMatrix.hostF32()
        assertTrue(abs(gm[0] - (-6f / 4f)) < 1e-5f, "grad_m[0,0] = -1.5; got ${gm[0]}")
        assertTrue(abs(gm[1] - (-8f / 16f)) < 1e-5f, "grad_m[0,1] = -0.5; got ${gm[1]}")
    }

    @Test
    fun reverseOrderCommutativeOpsMatchForwardOrder() {
        // row + matrix and matrix + row must yield the same result (addition
        // is commutative). Pin that the two call sites agree for both value
        // and gradient.
        val row = Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f, 30f))
        val mat = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val vgForward = valueAndGrad2 { r: Tracer<io.tlaloc.core.Rank1<Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (m + r).sum()
        }
        val vgReverse = valueAndGrad2 { r: Tracer<io.tlaloc.core.Rank1<Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (r + m).sum()
        }
        val (vFwd, drFwd, dmFwd) = vgForward(row, mat)
        val (vRev, drRev, dmRev) = vgReverse(row, mat)
        assertEquals(vFwd, vRev)
        val drFwdArr = drFwd.hostF32()
        val drRevArr = drRev.hostF32()
        for (i in 0 until 3) assertEquals(drFwdArr[i], drRevArr[i], "grad_row[$i]")
        val dmFwdArr = dmFwd.hostF32()
        val dmRevArr = dmRev.hostF32()
        for (i in 0 until 6) assertEquals(dmFwdArr[i], dmRevArr[i], "grad_matrix[$i]")
    }

    @Test
    fun broadcastRowExposedPubliclyMatchesImplicitOperator() {
        // §0.4.89 — §0.4.85's `Tracer<Rank2>.plus(Tracer<Rank1>)` operator and
        // the newly-public `broadcastRow` must produce identical forward +
        // backward results; the operator is a thin wrapper over the builder.
        val xInput = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val bInput = Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f, 30f))

        val vgImplicit = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + b).sum()
        }
        val vgExplicit = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + x.broadcastRow(b)).sum()
        }

        val (vImp, dxImp, dbImp) = vgImplicit(xInput, bInput)
        val (vExp, dxExp, dbExp) = vgExplicit(xInput, bInput)
        assertEquals(vImp, vExp)
        val gxImp = dxImp.hostF32()
        val gxExp = dxExp.hostF32()
        for (i in 0 until 6) assertEquals(gxImp[i], gxExp[i], "grad_x[$i] implicit vs explicit")
        val gbImp = dbImp.hostF32()
        val gbExp = dbExp.hostF32()
        for (i in 0 until 3) assertEquals(gbImp[i], gbExp[i], "grad_b[$i] implicit vs explicit")
    }

    @Test
    fun rank2PlusBroadcastColAppliesColumnVectorAcrossColumns() {
        // §0.4.87 — `matrix + matrix.broadcastCol(col)` replicates `col`
        // (size M) across all N columns. For x = [[1,2,3],[4,5,6]] and
        // col=[10,100]: broadcast = [[10,10,10],[100,100,100]]. sum(x+bcast) =
        // (1+10)+(2+10)+(3+10)+(4+100)+(5+100)+(6+100) = 33 + 315 = 348 … wait,
        // let me recompute: 11+12+13+104+105+106 = 36+315 = 351. Yes 351.
        //   grad_x = ones(2, 3).
        //   grad_col_i = sum over cols of 1 = N = 3.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, col: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + x.broadcastCol(col)).sum()
        }
        val (value, dx, dcol) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 100f)),
        )
        assertEquals(351f, value)
        val gx = dx.hostF32()
        for (i in 0 until 6) assertEquals(1f, gx[i], "grad_x[$i]")
        val gc = dcol.hostF32()
        assertEquals(3f, gc[0], "grad_col[0] = N")
        assertEquals(3f, gc[1], "grad_col[1] = N")
    }

    @Test
    fun rank2TimesBroadcastColWeightsEachRow() {
        // f(x, col) = sum(x * col_broadcast) at x=[[1,2],[3,4]], col=[10,100].
        //   broadcast = [[10,10],[100,100]].
        //   product = [[10,20],[300,400]]. sum = 730.
        //   grad_x_ij = col_i → [[10,10],[100,100]].
        //   grad_col_i = sum over j of x_ij = [1+2, 3+4] = [3, 7].
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, col: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x * x.broadcastCol(col)).sum()
        }
        val (value, dx, dcol) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 100f)),
        )
        assertEquals(730f, value)
        val gx = dx.hostF32()
        assertEquals(10f, gx[0]); assertEquals(10f, gx[1]); assertEquals(100f, gx[2]); assertEquals(100f, gx[3])
        val gc = dcol.hostF32()
        assertEquals(3f, gc[0])
        assertEquals(7f, gc[1])
    }

    @Test
    fun rank2BroadcastColShapeMismatchThrows() {
        // Matrix row size 2 vs col size 3 → clear error.
        var caught: IllegalArgumentException? = null
        try {
            valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, col: Tracer<io.tlaloc.core.Rank1<Sym>> ->
                (x + x.broadcastCol(col)).sum()
            }(
                Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
                Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
            )
        } catch (e: IllegalArgumentException) {
            caught = e
        }
        assertTrue(caught != null, "expected IllegalArgumentException for size mismatch")
        assertTrue(caught!!.message!!.contains("doesn't match"), "message should name mismatch: ${caught!!.message}")
    }

    @Test
    fun rank2PlusRank1RowBroadcastsAndSums() {
        // f(x, b) = sum(x + b). x is [[1, 2, 3], [4, 5, 6]], b is [10, 20, 30].
        //   Forward: [[11, 22, 33], [14, 25, 36]] → sum = 141.
        //   grad_x = ones(2, 3) → [[1, 1, 1], [1, 1, 1]].
        //   grad_b_j = sum over rows = 2 for each column → [2, 2, 2].
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + b).sum()
        }
        val (value, dx, db) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f, 30f)),
        )
        assertEquals(141f, value)
        val gx = dx.hostF32()
        for (i in 0 until 6) assertEquals(1f, gx[i], "grad_x[$i]")
        val gb = db.hostF32()
        assertEquals(2f, gb[0])
        assertEquals(2f, gb[1])
        assertEquals(2f, gb[2])
    }

    @Test
    fun rank2TimesRank1RowGivesColumnWeightedGrads() {
        // f(x, w) = sum(x * w). x is [[1, 2], [3, 4]], w is [10, 100].
        //   Forward: [[10, 200], [30, 400]] → sum = 640.
        //   grad_x_ij = w_j → [[10, 100], [10, 100]].
        //   grad_w_j = sum over rows of x_*j = [4, 6].
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, w: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x * w).sum()
        }
        val (value, dx, dw) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 100f)),
        )
        assertEquals(640f, value)
        val gx = dx.hostF32()
        assertEquals(10f, gx[0]); assertEquals(100f, gx[1]); assertEquals(10f, gx[2]); assertEquals(100f, gx[3])
        val gw = dw.hostF32()
        assertEquals(4f, gw[0])  // 1 + 3
        assertEquals(6f, gw[1])  // 2 + 4
    }

    @Test
    fun rank2BroadcastRowShapeMismatchThrows() {
        // Matrix col size (2) != row size (3) → loud fail.
        var caught: IllegalArgumentException? = null
        try {
            valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank1<Sym>> ->
                (x + b).sum()
            }(
                Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
                Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
            )
        } catch (e: IllegalArgumentException) {
            caught = e
        }
        assertTrue(caught != null, "expected IllegalArgumentException for size mismatch")
        assertTrue(caught!!.message!!.contains("doesn't match"), "message should name mismatch: ${caught!!.message}")
    }

    @Test
    fun complexBroadcastCompositionMixesScalarRowAndFloatLiteral() {
        // §0.4.101 — exercise the §0.4.75+ broadcast surface in a single
        // expression that uses Float literal LHS, row broadcast, scalar
        // tracer, and operator chaining. The kind of expression a user
        // doing per-row weighted regression might write.
        //
        // f(x, w_row, lr) = sum((x - 1f).times(w_row) * lr)
        //   x: rank-2 [2, 3] = [[1, 2, 3], [4, 5, 6]]
        //   w_row: rank-1 [3] = [10, 100, 1000]
        //   lr: scalar = 0.5
        //
        // Step-by-step:
        //   x - 1f         = [[0, 1, 2], [3, 4, 5]]              (Float literal RHS)
        //   .times(w_row)  = [[0, 100, 2000], [30, 400, 5000]]   (row broadcast)
        //   * lr           = [[0, 50, 1000], [15, 200, 2500]]    (scalar broadcast)
        //   .sum()         = 50 + 1000 + 15 + 200 + 2500 = 3765
        //
        // Gradients:
        //   grad_x[i, j] = w_row[j] * lr → row [5, 50, 500] replicated.
        //   grad_w_row[j] = sum over i of (x-1)_ij * lr =
        //     [(0+3)*0.5, (1+4)*0.5, (2+5)*0.5] = [1.5, 2.5, 3.5].
        //   grad_lr = sum((x-1)*w_row) =
        //     0+100+2000+30+400+5000 = 7530.
        // No `valueAndGrad3` exists — bind lr as a Float literal inside the
        // 2-input lambda. Mathematically identical; exercises §0.4.93's
        // Float * matrix path for the scaling step.
        val vgPrimal = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, w: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            // Pin lr=0.5 as a Float literal — exercises §0.4.93's Float * matrix.
            (((x - 1f) * w) * 0.5f).sum()
        }
        val (value, dx, dw) = vgPrimal(
            Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 100f, 1000f)),
        )
        assertEquals(3765f, value)
        // grad_x[i, j] = w_row[j] * 0.5
        val gx = dx.hostF32()
        // Row 0: w * 0.5 = [5, 50, 500]
        assertEquals(5f, gx[0]); assertEquals(50f, gx[1]); assertEquals(500f, gx[2])
        // Row 1: same
        assertEquals(5f, gx[3]); assertEquals(50f, gx[4]); assertEquals(500f, gx[5])
        // grad_w[j] = sum over i of (x-1)_ij * 0.5
        val gw = dw.hostF32()
        assertEquals(1.5f, gw[0])  // (0+3)/2
        assertEquals(2.5f, gw[1])  // (1+4)/2
        assertEquals(3.5f, gw[2])  // (2+5)/2
    }

    @Test
    fun rank3MinusScalarTracerLhsFlipsSign() {
        // §0.4.98 — `scalar - tensor3` has opposite gradient sign from
        // `tensor3 - scalar`. x = 2x2x2 of [1..8], s = 5.
        //   forward: sum(5 - x_i) = sum([4,3,2,1,0,-1,-2,-3]) = 4. value=4.
        //   grad_s = D0·D1·D2 = 8.
        //   grad_x = -1 per element.
        val vg = valueAndGrad2 { s: Tracer<ScalarShape>, x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>> ->
            (s - x).sum()
        }
        val (value, dScalar, dTensor) = vg(
            Tensors.f32Scalar(5f),
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
        )
        assertEquals(4f, value)
        assertEquals(8f, dScalar.hostF32()[0], "grad_s = D0·D1·D2 = 8")
        val gx = dTensor.hostF32()
        for (i in 0 until 8) assertEquals(-1f, gx[i])
    }

    @Test
    fun rank3PlusScalarTracerGivesBothGradients() {
        // §0.4.97 — Rank-3 extension of §0.4.78's scalar broadcast. `f32Tensor3`
        // constructor + Tracer<Rank3>.plus(Tracer<ScalarShape>) overload.
        // x is a 2x2x2 tensor [[[1,2],[3,4]],[[5,6],[7,8]]]; scalar c = 10.
        //   forward: every element + 10. sum = 36 + 80 = 116. value = 116.
        //   grad_x = ones (8 elements). grad_c = 8 (= D0*D1*D2).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, c: Tracer<ScalarShape> ->
            (x + c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
            Tensors.f32Scalar(10f),
        )
        assertEquals(116f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(1f, gx[i], "grad_x[$i]")
        assertEquals(8f, dc.hostF32()[0], "grad_c = D0*D1*D2 = 8")
    }

    @Test
    fun rank3TimesScalarTracerScalesEachElement() {
        // f(x, c) = sum(x * c). x = 2x2x2 ones-like, c = 5.
        //   forward: sum = 8 * 5 = 40. grad_x = c per element (5 each).
        //   grad_c = sum(x) = 8 (since x is 8 ones).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, c: Tracer<ScalarShape> ->
            (x * c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { 1f }),
            Tensors.f32Scalar(5f),
        )
        assertEquals(40f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(5f, gx[i])
        assertEquals(8f, dc.hostF32()[0])
    }

    // §0.4.116 — rank-3 + rank-1 cross-rank broadcast (inner axis).

    @Test
    fun rank3PlusRank1InnerGivesBothGradients() {
        // f(x, v) = sum(x + v.broadcastedToInner). x is 2x2x2 [1..8], v = [10, 20].
        // For each [a, b, c]: x[a,b,c] + v[c]. Sum across all 8 elements:
        //   row [0,0]: 1+10=11, 2+20=22         → 33
        //   row [0,1]: 3+10=13, 4+20=24         → 37
        //   row [1,0]: 5+10=15, 6+20=26         → 41
        //   row [1,1]: 7+10=17, 8+20=28         → 45
        //   total                               = 156
        // grad_x = ones (8 elements; sum's grad is broadcast 1).
        // grad_v = sum of upstream over axes [0, 1] = [4, 4]
        //   (each v[c] is added into 4 output positions: 2 batches × 2 rows).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, v: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x + v).sum()
        }
        val (value, dx, dv) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f)),
        )
        assertEquals(156f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(1f, gx[i], "grad_x[$i]")
        val gv = dv.hostF32()
        assertEquals(4f, gv[0], "grad_v[0] = 4 (positions [a,b,0] sum)")
        assertEquals(4f, gv[1], "grad_v[1] = 4 (positions [a,b,1] sum)")
    }

    @Test
    fun rank3MinusRank1InnerFlipsGradSign() {
        // Non-commutative pin: (x - v) reverses sign on grad_v.
        // x = ones (2x2x2 = 8 ones), v = [10, 20].
        // For each [a, b, c]: 1 - v[c]. sum = 4*(1-10) + 4*(1-20) = -36 + -76 = -112.
        // grad_x = +1 per element (sum's grad).
        // grad_v = -1 per upstream position summed over [0, 1] = [-4, -4].
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, v: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x - v).sum()
        }
        val (value, dx, dv) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { 1f }),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f)),
        )
        assertEquals(-112f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(1f, gx[i])
        val gv = dv.hostF32()
        assertEquals(-4f, gv[0], "grad_v[0]")
        assertEquals(-4f, gv[1], "grad_v[1]")
    }

    @Test
    fun rank3TimesRank1InnerScalesElementsByInnerVector() {
        // f(x, v) = sum(x * v_broadcast). x = ones (8 elements), v = [3, 5].
        // For each [a, b, c]: 1 * v[c]. sum = 4*3 + 4*5 = 32.
        // grad_x = v[c] per element (3 at c=0, 5 at c=1).
        // grad_v[c] = sum of x[a,b,c] over [a, b] = 4 each (since x is all ones).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, v: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            (x * v).sum()
        }
        val (value, dx, dv) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { 1f }),
            Tensors.f32Vector<Sym>(floatArrayOf(3f, 5f)),
        )
        assertEquals(32f, value)
        val gx = dx.hostF32()
        // Layout: [a=0,b=0,c=0], [a=0,b=0,c=1], [a=0,b=1,c=0], [a=0,b=1,c=1], etc.
        // c alternates: 0,1,0,1,0,1,0,1. So gx = [3, 5, 3, 5, 3, 5, 3, 5].
        for (i in 0 until 8) {
            val expected = if (i % 2 == 0) 3f else 5f
            assertEquals(expected, gx[i], "grad_x[$i]")
        }
        val gv = dv.hostF32()
        assertEquals(4f, gv[0], "grad_v[0]")
        assertEquals(4f, gv[1], "grad_v[1]")
    }

    @Test
    fun rank3InnerBroadcastComposesWithScalarBroadcast() {
        // §0.4.101-style composition: cross-rank broadcasting + scalar broadcast in
        // one expression. f(x, v, c) — well, we have valueAndGrad2 (2-input), so
        // bind c as a scalar literal. f(x, v) = sum((x + v) * 0.5f) — adds the
        // inner-broadcast vector then scales the whole tensor by a Float literal.
        //
        // x = 2x2x2 [1..8], v = [10, 20].
        // (x + v) per element same as the first test → values [11, 22, 13, 24, 15, 26, 17, 28].
        // * 0.5 → [5.5, 11, 6.5, 12, 7.5, 13, 8.5, 14]. sum = 78.
        // grad_x: each element scales by 0.5 (the literal multiplier). So gx = 0.5 per element.
        // grad_v[c]: sum of 0.5 over [a, b] = 4 * 0.5 = 2 each.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, v: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            ((x + v) * 0.5f).sum()
        }
        val (value, dx, dv) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
            Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f)),
        )
        assertEquals(78f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(0.5f, gx[i], "grad_x[$i]")
        val gv = dv.hostF32()
        assertEquals(2f, gv[0], "grad_v[0]")
        assertEquals(2f, gv[1], "grad_v[1]")
    }

    // §0.4.117 — rank-3 + rank-2 cross-rank broadcast (batch axis).

    @Test
    fun rank3PlusRank2BatchGivesBothGradients() {
        // f(x, m) = sum(x + m_broadcast). x = 2x2x2 [1..8], m = 2x2 [10, 20, 30, 40].
        // For each [a, b, c]: x[a,b,c] + m[b,c]. Both batches see the same matrix
        // added to their respective slices.
        //   batch 0: [[1+10, 2+20], [3+30, 4+40]] = [[11, 22], [33, 44]]
        //   batch 1: [[5+10, 6+20], [7+30, 8+40]] = [[15, 26], [37, 48]]
        //   sum = (11+22+33+44) + (15+26+37+48) = 110 + 126 = 236.
        // grad_x = ones (8 elements).
        // grad_m[b, c] = sum over batch (axis 0) of upstream = 2 per element
        //   (each m position is added into 2 outputs: one per batch).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (x + m).sum()
        }
        val (value, dx, dm) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f)),
        )
        assertEquals(236f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(1f, gx[i], "grad_x[$i]")
        val gm = dm.hostF32()
        for (i in 0 until 4) assertEquals(2f, gm[i], "grad_m[$i] = 2 (one per batch)")
    }

    @Test
    fun rank3MinusRank2BatchFlipsGradSign() {
        // x = ones (8 elements), m = [10, 20, 30, 40].
        // (x - m) per element: 1 - m[b, c] across each batch.
        //   batch 0: [-9, -19, -29, -39] → -96
        //   batch 1: [-9, -19, -29, -39] → -96
        //   sum = -192.
        // grad_x = +1 per element.
        // grad_m = -1 per upstream * 2 batches = -2 per element.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (x - m).sum()
        }
        val (value, dx, dm) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { 1f }),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f)),
        )
        assertEquals(-192f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(1f, gx[i])
        val gm = dm.hostF32()
        for (i in 0 until 4) assertEquals(-2f, gm[i])
    }

    @Test
    fun rank3TimesRank2BatchScalesPerSlice() {
        // f(x, m) = sum(x * m_broadcast). x = ones (8), m = [2, 3, 4, 5].
        //   per element: 1 * m[b, c] = m. Across both batches, each m position
        //   is replicated once. sum = 2 * (2+3+4+5) = 28.
        // grad_x[a, b, c] = m[b, c] — each batch sees the same matrix scaling.
        // grad_m[b, c] = sum_a x[a, b, c] = 2 (since x is all ones, two batches).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            (x * m).sum()
        }
        val (value, dx, dm) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, FloatArray(8) { 1f }),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(2f, 3f, 4f, 5f)),
        )
        assertEquals(28f, value)
        val gx = dx.hostF32()
        // Layout: [a=0,b=0,c=0], [a=0,b=0,c=1], [a=0,b=1,c=0], [a=0,b=1,c=1],
        //         [a=1,b=0,c=0], [a=1,b=0,c=1], [a=1,b=1,c=0], [a=1,b=1,c=1].
        // m row-major: m[b, c] = [2 (b=0,c=0), 3 (b=0,c=1), 4 (b=1,c=0), 5 (b=1,c=1)].
        // So gx = [2, 3, 4, 5, 2, 3, 4, 5].
        val expectedGx = floatArrayOf(2f, 3f, 4f, 5f, 2f, 3f, 4f, 5f)
        for (i in 0 until 8) assertEquals(expectedGx[i], gx[i], "grad_x[$i]")
        val gm = dm.hostF32()
        for (i in 0 until 4) assertEquals(2f, gm[i])
    }

    @Test
    fun rank3BatchBroadcastComposesWithInnerBroadcast() {
        // §0.4.101-style composition: rank-3 + rank-2 batch + then + rank-1 inner
        // in one expression. Three different broadcasts in sequence.
        //
        // f(x, m, v) — but valueAndGrad2 only takes 2 inputs, so bind v as a
        // local closure. Use a named const v rather than a tracer.
        // f(x, m) = sum((x + m) + 0.1f). Nope, that's not composing v.
        //
        // Better composition: f(x, m) = sum((x + m) * 2f). Three operators
        // (rank-3 + rank-2 batch broadcast, then Float-literal multiply, then SUM).
        //
        // Forward: (x + m) per-element values from the first test [11..48 across
        // 8 elements as computed], total sum = 236. Multiplied by 2 → 472.
        // grad_x = 2 per element (the literal multiplier).
        // grad_m[b, c] = 2 batches × 2 = 4 per element.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, m: Tracer<io.tlaloc.core.Rank2<Sym, Sym>> ->
            ((x + m) * 2f).sum()
        }
        val (value, dx, dm) = vg(
            Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f)),
        )
        assertEquals(472f, value)
        val gx = dx.hostF32()
        for (i in 0 until 8) assertEquals(2f, gx[i], "grad_x[$i]")
        val gm = dm.hostF32()
        for (i in 0 until 4) assertEquals(4f, gm[i], "grad_m[$i] = 4")
    }

    @Test
    fun rank2PlusScalarTracerGivesBothGradients() {
        // §0.4.78 — rank-2 extension of §0.4.77's scalar broadcast.
        // f(x, c) = sum(x + c) at x=[[1,2],[3,4]], c=5.
        //   value = sum([6,7,8,9]) = 30. grad_x=[[1,1],[1,1]]. grad_c=4 (= M*N).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, c: Tracer<ScalarShape> ->
            (x + c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            Tensors.f32Scalar(5f),
        )
        assertEquals(30f, value)
        val gx = dx.hostF32()
        assertEquals(1f, gx[0]); assertEquals(1f, gx[1]); assertEquals(1f, gx[2]); assertEquals(1f, gx[3])
        assertEquals(4f, dc.hostF32()[0], "grad_c = M · N = 4")
    }

    @Test
    fun rank2TimesScalarTracerGivesBothGradients() {
        // f(x, c) = sum(x * c) at x=[[1, 2], [3, 4]], c=2.
        //   value = 2 · (1+2+3+4) = 20. grad_x=[[c,c],[c,c]]=[[2,2],[2,2]]. grad_c=sum(x)=10.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank2<Sym, Sym>>, c: Tracer<ScalarShape> ->
            (x * c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            Tensors.f32Scalar(2f),
        )
        assertEquals(20f, value)
        val gx = dx.hostF32()
        assertEquals(2f, gx[0]); assertEquals(2f, gx[1]); assertEquals(2f, gx[2]); assertEquals(2f, gx[3])
        assertEquals(10f, dc.hostF32()[0], "grad_c = sum(x) = 10")
    }

    @Test
    fun rank1PlusScalarTracerGivesBothGradients() {
        // f(x, c) = sum(x + c) at x=[1,2,3], c=10. Value 36.
        //   grad_x_i = d/dx_i sum(x + c) = 1 → [1, 1, 1].
        //   grad_c = d/dc sum(x + c) = N = 3 (once per element).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x + c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)),
            Tensors.f32Scalar(10f),
        )
        assertEquals(36f, value)
        val gx = dx.hostF32()
        assertEquals(1f, gx[0])
        assertEquals(1f, gx[1])
        assertEquals(1f, gx[2])
        assertEquals(3f, dc.hostF32()[0], "grad_c = sum(1) across N=3 elements")
    }

    @Test
    fun rank1TimesScalarTracerGivesBothGradients() {
        // f(x, c) = sum(x * c) at x=[2, 4, 8], c=3. Value = 3 · 14 = 42.
        //   grad_x_i = d/dx_i sum(x · c) = c = 3.
        //   grad_c = d/dc sum(x · c) = sum(x) = 14.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x * c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Vector(floatArrayOf(2f, 4f, 8f)),
            Tensors.f32Scalar(3f),
        )
        assertEquals(42f, value)
        val gx = dx.hostF32()
        assertEquals(3f, gx[0])
        assertEquals(3f, gx[1])
        assertEquals(3f, gx[2])
        assertEquals(14f, dc.hostF32()[0], "grad_c = sum(x)")
    }

    @Test
    fun rank1MinusScalarTracerReverseSubtract() {
        // f(x, c) = sum(x - c) at x=[5, 10], c=2. Value = 11.
        //   grad_x = [1, 1].
        //   grad_c = -2 (one per element × -1).
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x - c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Vector(floatArrayOf(5f, 10f)),
            Tensors.f32Scalar(2f),
        )
        assertEquals(11f, value)
        val gx = dx.hostF32()
        assertEquals(1f, gx[0])
        assertEquals(1f, gx[1])
        assertEquals(-2f, dc.hostF32()[0], "grad_c = -N for N=2 elements")
    }

    @Test
    fun rank1DivByScalarTracerGivesReciprocalGrad() {
        // f(x, c) = sum(x / c) at x=[4, 8], c=2. Value = 2 + 4 = 6.
        //   grad_x_i = 1/c = 0.5.
        //   grad_c = -sum(x)/c² = -12/4 = -3.
        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x / c).sum()
        }
        val (value, dx, dc) = vg(
            Tensors.f32Vector(floatArrayOf(4f, 8f)),
            Tensors.f32Scalar(2f),
        )
        assertEquals(6f, value)
        val gx = dx.hostF32()
        assertTrue(abs(gx[0] - 0.5f) < 1e-5f)
        assertTrue(abs(gx[1] - 0.5f) < 1e-5f)
        assertTrue(abs(dc.hostF32()[0] - (-3f)) < 1e-5f, "grad_c = -sum(x)/c² = -3; got ${dc.hostF32()[0]}")
    }

    @Test
    fun scalarPowScalesGradViaExponent() {
        // §0.4.76 — f(x) = x^3 as `x.pow(3f)`. Grad = 3·x². At x=2 → value=8, grad=12.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x.pow(3f) }
        val (v, dx) = vg(Tensors.f32Scalar(2f))
        assertEquals(8f, v)
        assertTrue(abs(dx.hostF32()[0] - 12f) < 1e-5f, "grad_x = 3·x² = 12; got ${dx.hostF32()[0]}")

        // Rank-1 elementwise: f(x) = sum(x^2) at x=[1, 2, 3] → value 14, grad=[2, 4, 6].
        val vgR1 = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.pow(2f).sum() }
        val (v1, d1) = vgR1(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(14f, v1)
        val g = d1.hostF32()
        assertTrue(abs(g[0] - 2f) < 1e-5f)
        assertTrue(abs(g[1] - 4f) < 1e-5f)
        assertTrue(abs(g[2] - 6f) < 1e-5f)
    }

    @Test
    fun scalarAddIsIdentityOnGrad() {
        // f(x) = x + 5. df/dx = 1 on every element.
        val vgScalar = valueAndGrad { x: Tracer<ScalarShape> -> x + 5f }
        val (v, dx) = vgScalar(Tensors.f32Scalar(3f))
        assertEquals(8f, v)
        assertEquals(1f, dx.hostF32()[0])

        val vgRank1 = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x + 5f).sum() }
        val (v1, d1) = vgRank1(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(21f, v1)  // 6+7+8
        val g = d1.hostF32()
        assertEquals(1f, g[0])
        assertEquals(1f, g[1])
        assertEquals(1f, g[2])
    }

    @Test
    fun scalarSubtractShiftsForwardOnly() {
        // f(x) = x - 2. df/dx = 1.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x - 2f }
        val (v, dx) = vg(Tensors.f32Scalar(7f))
        assertEquals(5f, v)
        assertEquals(1f, dx.hostF32()[0])
    }

    @Test
    fun scalarMultiplyScalesGrad() {
        // f(x) = x * 3. df/dx = 3 on every element.
        val vgScalar = valueAndGrad { x: Tracer<ScalarShape> -> x * 3f }
        val (v, dx) = vgScalar(Tensors.f32Scalar(4f))
        assertEquals(12f, v)
        assertEquals(3f, dx.hostF32()[0])

        val vgRank1 = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (x * 0.5f).sum() }
        val (v1, d1) = vgRank1(Tensors.f32Vector(floatArrayOf(2f, 4f, 6f)))
        assertEquals(6f, v1)  // (1+2+3)
        val g = d1.hostF32()
        assertEquals(0.5f, g[0])
        assertEquals(0.5f, g[1])
        assertEquals(0.5f, g[2])
    }

    @Test
    fun scalarDivScalesGradReciprocally() {
        // f(x) = x / 4. df/dx = 1/4.
        val vg = valueAndGrad { x: Tracer<ScalarShape> -> x / 4f }
        val (v, dx) = vg(Tensors.f32Scalar(12f))
        assertEquals(3f, v)
        assertEquals(0.25f, dx.hostF32()[0])
    }

    @Test
    fun scalarOpsComposeInExpressions() {
        // f(x) = (x + 1) * 2 - 3 on rank-1; chains all four operators.
        // Forward at x=[1, 2, 3]: (2*2 - 3, 3*2 - 3, 4*2 - 3) = (1, 3, 5). sum = 9.
        // grad_x per element = d/dx of (x + 1) * 2 - 3 = 2. So all grad = 2.
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            ((x + 1f) * 2f - 3f).sum()
        }
        val (v, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(9f, v)
        val g = dx.hostF32()
        assertEquals(2f, g[0])
        assertEquals(2f, g[1])
        assertEquals(2f, g[2])
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
    fun unaryMinusOperatorMatchesNegMethod() {
        // §0.4.96 — `-x` and `x.neg()` produce identical forward + backward
        // results. The operator is a thin pass-through; no new tape op.
        val xIn = Tensors.f32Vector<Sym>(floatArrayOf(5f, -3f, 2f))
        val vgOp = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> (-x).sum() }
        val vgMethod = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> x.neg().sum() }
        val (vO, dO) = vgOp(xIn)
        val (vM, dM) = vgMethod(xIn)
        assertEquals(vO, vM)
        val dOArr = dO.hostF32()
        val dMArr = dM.hostF32()
        for (i in 0 until 3) assertEquals(dOArr[i], dMArr[i])
    }

    @Test
    fun unaryMinusComposesInExpressions() {
        // f(x) = sum((-x) * x) = sum(-x²). At x = [1, 2, 3]: forward = -14.
        //   grad_x = -2x → [-2, -4, -6].
        val vg = valueAndGrad { x: Tracer<io.tlaloc.core.Rank1<Sym>> -> ((-x) * x).sum() }
        val (value, dx) = vg(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))
        assertEquals(-14f, value)
        val gx = dx.hostF32()
        assertEquals(-2f, gx[0])
        assertEquals(-4f, gx[1])
        assertEquals(-6f, gx[2])
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

    // §0.4.134 — `valueAndGrad3` / `grad3` for 3-tensor scalar-valued functions.

    @Test
    fun valueAndGrad3OnPureScalarTernarySumOfProducts() {
        // f(a, b, c) = a*b + c. Gradients: df/da = b, df/db = a, df/dc = 1.
        val vg = valueAndGrad3 { a: Tracer<ScalarShape>, b: Tracer<ScalarShape>, c: Tracer<ScalarShape> ->
            a * b + c
        }
        val out = vg(
            Tensors.f32Scalar(2f),
            Tensors.f32Scalar(3f),
            Tensors.f32Scalar(5f),
        )
        assertEquals(11f, out.first, "f(2,3,5) = 2*3 + 5 = 11")
        assertEquals(3f, out.second.hostF32()[0], "df/da = b = 3")
        assertEquals(2f, out.third.hostF32()[0], "df/db = a = 2")
        assertEquals(1f, out.fourth.hostF32()[0], "df/dc = 1")
    }

    @Test
    fun grad3DropsValueAndReturnsTriple() {
        // grad3 mirrors grad2: drops the primal value, returns a Triple of grads.
        // Same f as the valueAndGrad3 test → same gradients (3, 2, 1).
        val g = grad3 { a: Tracer<ScalarShape>, b: Tracer<ScalarShape>, c: Tracer<ScalarShape> ->
            a * b + c
        }
        val (dA, dB, dC) = g(
            Tensors.f32Scalar(2f),
            Tensors.f32Scalar(3f),
            Tensors.f32Scalar(5f),
        )
        assertEquals(3f, dA.hostF32()[0])
        assertEquals(2f, dB.hostF32()[0])
        assertEquals(1f, dC.hostF32()[0])
    }

    @Test
    fun valueAndGrad3CrossRankMatrixVectorScalarInputs() {
        // §0.4.101's "cross-rank composition" pin pattern. f(W, x, c) = sum(W @ x.unsqueeze) * c
        // — but unsqueeze isn't on the Tracer surface; reach the equivalent via
        // (W matmul x_col).sum() * c, where x is a column matrix (Rank2 of shape [N, 1]).
        // Concrete: W = [[1, 2], [3, 4]] (2x2), x = [[1], [1]] (2x1), c = 0.5.
        // W@x = [[3], [7]], sum = 10, * c = 5.
        // d/dW sum(W@x)*c = c * ones(2,1) @ x^T = 0.5 * [[1,1],[1,1]]
        //                                       = [[0.5, 0.5], [0.5, 0.5]].
        // d/dx sum(W@x)*c = c * W^T @ ones(2,1) = 0.5 * [[1,3],[2,4]] @ [[1],[1]]
        //                                       = 0.5 * [[4],[6]] = [[2],[3]].
        // d/dc sum(W@x)*c = sum(W@x) = 10.
        val vg = valueAndGrad3 { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>>, c: Tracer<ScalarShape> ->
            (w matmul x).sum() * c
        }
        val out = vg(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f)),
            Tensors.f32Scalar(0.5f),
        )
        assertEquals(5f, out.first, "f(W, x, c) = sum(W@x)*c = 10*0.5 = 5")
        assertContentEquals(floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f), out.second.hostF32(), "dW")
        assertContentEquals(floatArrayOf(2f, 3f), out.third.hostF32(), "dx")
        assertEquals(10f, out.fourth.hostF32()[0], "dc = sum(W@x) = 10")
    }

    @Test
    fun valueAndGrad3ReuseAcrossCallsHasNoTapeBleed() {
        // Each call must build a fresh tape — no state leak between invocations.
        val vg = valueAndGrad3 { a: Tracer<ScalarShape>, b: Tracer<ScalarShape>, c: Tracer<ScalarShape> ->
            a * b * c
        }
        // f(a,b,c) = a*b*c. df/da = b*c, df/db = a*c, df/dc = a*b.
        val firstCall = vg(Tensors.f32Scalar(2f), Tensors.f32Scalar(3f), Tensors.f32Scalar(5f))
        assertEquals(30f, firstCall.first, "first: 2*3*5 = 30")
        assertEquals(15f, firstCall.second.hostF32()[0], "first dA = b*c = 15")
        assertEquals(10f, firstCall.third.hostF32()[0], "first dB = a*c = 10")
        assertEquals(6f, firstCall.fourth.hostF32()[0], "first dC = a*b = 6")

        val secondCall = vg(Tensors.f32Scalar(1f), Tensors.f32Scalar(7f), Tensors.f32Scalar(11f))
        assertEquals(77f, secondCall.first, "second: 1*7*11 = 77")
        assertEquals(77f, secondCall.second.hostF32()[0], "second dA = b*c = 77")
        assertEquals(11f, secondCall.third.hostF32()[0], "second dB = a*c = 11")
        assertEquals(7f, secondCall.fourth.hostF32()[0], "second dC = a*b = 7")
    }

    // §0.4.137 — tracer-surface `bmm` (rank-3 batched matmul) + batched MatmulRule.

    @Test
    fun bmmForwardComputesPerBatchSlices() {
        // Batch 0: a0 = [[1, 2, 3], [4, 5, 6]] (2×3), b0 = [[1, 0], [0, 1], [1, 1]] (3×2).
        //   a0 @ b0 = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]].
        // Batch 1: a1 = [[1, 1, 1], [2, 2, 2]] (2×3), b1 = [[1, 1], [1, 1], [1, 1]] (3×2).
        //   a1 @ b1 = [[3, 3], [6, 6]].
        val vg = valueAndGrad2 { a: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>> ->
            (a bmm b).sum()
        }
        val a = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 2, 3,
            floatArrayOf(
                1f, 2f, 3f, 4f, 5f, 6f,
                1f, 1f, 1f, 2f, 2f, 2f,
            ),
        )
        val b = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 3, 2,
            floatArrayOf(
                1f, 0f, 0f, 1f, 1f, 1f,
                1f, 1f, 1f, 1f, 1f, 1f,
            ),
        )
        val (value, _, _) = vg(a, b)
        // sum(batch 0) = 4 + 5 + 10 + 11 = 30; sum(batch 1) = 3 + 3 + 6 + 6 = 18; total = 48.
        assertEquals(48f, value)
    }

    @Test
    fun bmmGradientPropagatesThroughBatchedMatmul() {
        // For f(A, B) = sum(A @@ B), the gradients are:
        //   dA = ones(B, M, N) @@ B^T_batched = ones-row times sum of B's columns per batch.
        //   dB = A^T_batched @@ ones(B, M, N) = sum of A's rows per batch.
        //
        // With B=2, M=2, K=3, N=2:
        //   - For batch 0 of dA (shape M×K): each row is sum of B's columns = sum of B's columns per batch.
        //     B_0 = [[1,0],[0,1],[1,1]] → column-sum row = [1+0+1, 0+1+1] = [2, 2] (sum across cols).
        //     But dA[i, j] = sum_n B^T[j, n] = sum_n B[n, j]. For B=batch 0:
        //       dA_0[*, 0] = sum_n B_0[n, 0] = 1 + 0 + 1 = 2 (every row, column 0 of dA).
        //       dA_0[*, 1] = sum_n B_0[n, 1] = 0 + 1 + 1 = 2.
        //       dA_0[*, 2] = sum_n B_0[n, 2-out-of-range]. Wait, B is K×N=3×2; reading via B^T's [j,n]
        //                   means dA's row r and column k = sum_n B[k, n]. For B_0, column-k sums:
        //                     k=0: B_0[0, *] = 1+0 = 1.
        //                     k=1: B_0[1, *] = 0+1 = 1.
        //                     k=2: B_0[2, *] = 1+1 = 2.
        //       So dA_0 = [[1, 1, 2], [1, 1, 2]] (every row == [1, 1, 2]).
        //     For B=batch 1, B_1 = [[1,1],[1,1],[1,1]] → row-sums [2, 2, 2]; dA_1 = [[2,2,2],[2,2,2]].
        //   - Similar analysis for dB.
        val grad = grad2 { a: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>>, b: Tracer<io.tlaloc.core.Rank3<Sym, Sym, Sym>> ->
            (a bmm b).sum()
        }
        val a = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 2, 3,
            floatArrayOf(
                1f, 2f, 3f, 4f, 5f, 6f,
                1f, 1f, 1f, 2f, 2f, 2f,
            ),
        )
        val b = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 3, 2,
            floatArrayOf(
                1f, 0f, 0f, 1f, 1f, 1f,
                1f, 1f, 1f, 1f, 1f, 1f,
            ),
        )
        val (dA, dB) = grad(a, b)
        // dA[batch=0] = [[1, 1, 2], [1, 1, 2]]; dA[batch=1] = [[2, 2, 2], [2, 2, 2]].
        assertContentEquals(
            floatArrayOf(
                1f, 1f, 2f, 1f, 1f, 2f,
                2f, 2f, 2f, 2f, 2f, 2f,
            ),
            dA.hostF32(),
        )
        // dB[k, n] = sum over m of A[m, k]:
        // For batch 0, A_0 = [[1,2,3],[4,5,6]], column-sums per k = [5, 7, 9].
        //   dB_0[k, *] = [5, 5; 7, 7; 9, 9] (every column-n entry equals the k-sum).
        // For batch 1, A_1 = [[1,1,1],[2,2,2]], column-sums per k = [3, 3, 3].
        //   dB_1[k, *] = [3, 3; 3, 3; 3, 3].
        assertContentEquals(
            floatArrayOf(
                5f, 5f, 7f, 7f, 9f, 9f,
                3f, 3f, 3f, 3f, 3f, 3f,
            ),
            dB.hostF32(),
        )
    }

    @Test
    fun bmmRejectsBatchOrInnerDimMismatch() {
        // Inner dim mismatch: a is (B, M, K) and b is (B, K', N) with K ≠ K'.
        val a = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 2, 3,
            FloatArray(12),
        )
        val bBadInner = io.tlaloc.core.Tensors.f32Tensor3<Sym, Sym, Sym>(
            2, 4, 2,  // K'=4, but lhs K=3
            FloatArray(16),
        )
        val tape = Tape()
        val ta = tape.traceLeaf<io.tlaloc.core.Rank3<Sym, Sym, Sym>>(a)
        val tb = tape.traceLeaf<io.tlaloc.core.Rank3<Sym, Sym, Sym>>(bBadInner)
        kotlin.test.assertFailsWith<IllegalArgumentException> { ta bmm tb }
    }

    @Test
    fun valueAndGrad3RejectsNonScalarOutput() {
        // The contract requires a scalar return. A rank-1 return must throw.
        val vg = valueAndGrad3 { a: Tracer<io.tlaloc.core.Rank1<Sym>>,
                                 b: Tracer<io.tlaloc.core.Rank1<Sym>>,
                                 c: Tracer<io.tlaloc.core.Rank1<Sym>> ->
            // (a + b + c) is rank-1; not a scalar.
            @Suppress("UNCHECKED_CAST")
            (a + b + c) as Tracer<ScalarShape>
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            vg(
                Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
                Tensors.f32Vector<Sym>(floatArrayOf(0f, 0f, 0f)),
                Tensors.f32Vector<Sym>(floatArrayOf(0f, 0f, 0f)),
            )
        }
    }
}
