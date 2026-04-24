package io.tlaloc.autograd

import io.tlaloc.core.F32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Equivalence tests for the §0.4.6 dxir-eval bridge: for each primal, builds the same
 * expression as a [DxirFunction] and (where the tape surface supports it) as a
 * [Tracer]-based lambda, runs both paths, and asserts the gradient values match.
 *
 * The runtime-tape path is `valueAndGrad` → `backward` (which now routes elementwise
 * arms through `VjpRegistry` + `DxirInterpreter`). The SCT path is
 * `DxirReverseTransform.apply` → `DxirInterpreter.evalFunction`. Matching numerical
 * output confirms the two paths share one math source-of-truth for every op in the
 * registry's current scope (ADD / SUB / MUL / DIV / NEG).
 *
 * As of §0.4.7 the registry also covers `RELU` (via `ReluRule = upstream * STEP(x)`),
 * and `DIV` has a tape-side implementation in `TracedOps.kt`, so the reciprocal case
 * now pins tape against SCT directly in addition to the closed-form derivative.
 */
class DxirBridgeEquivalenceTest {

    private val f32 = DxirType(F32, emptyList())
    private val tol = 1e-5f

    private fun tapeGrad(f: (Tracer<ScalarShape>) -> Tracer<ScalarShape>, input: Float): Float =
        valueAndGrad(f)(Tensors.f32Scalar(input)).second.hostF32()[0]

    private fun sctGrad(primal: DxirFunction, input: Float): Float {
        val grad = DxirReverseTransform.apply(primal)
        return DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(input))).single()[0]
    }

    private fun assertClose(expected: Float, actual: Float, msg: String = "") {
        assertTrue(
            abs(expected - actual) < tol,
            "$msg expected=$expected actual=$actual |Δ|=${abs(expected - actual)}",
        )
    }

    @Test
    fun identity() {
        // f(x) = x,  f'(x) = 1.
        val primal = DxirBuilder.function("id") {
            val x = param("x", f32)
            listOf(x)
        }
        for (v in listOf(-3.0f, 0.0f, 2.5f, 7.0f)) {
            val tape = tapeGrad({ x -> x }, v)
            val sct = sctGrad(primal, v)
            assertEquals(1.0f, sct)
            assertClose(tape, sct, "identity at v=$v")
        }
    }

    @Test
    fun unaryNeg() {
        // f(x) = -x,  f'(x) = -1.
        val primal = DxirBuilder.function("neg") {
            val x = param("x", f32)
            val y = op(OpKind.NEG, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(-3.0f, 0.0f, 2.5f, 7.0f)) {
            val tape = tapeGrad({ x -> x.neg() }, v)
            val sct = sctGrad(primal, v)
            assertEquals(-1.0f, sct)
            assertClose(tape, sct, "neg at v=$v")
        }
    }

    @Test
    fun subtractionAliased() {
        // f(x) = x - x  (degenerate but exercises SUB with aliased operands),  f'(x) = 0.
        val primal = DxirBuilder.function("xsubx") {
            val x = param("x", f32)
            val y = op(OpKind.SUB, listOf(x, x), f32)
            listOf(y)
        }
        for (v in listOf(-3.0f, 0.0f, 2.5f, 7.0f)) {
            val tape = tapeGrad({ x -> x - x }, v)
            val sct = sctGrad(primal, v)
            assertEquals(0.0f, sct)
            assertClose(tape, sct, "x-x at v=$v")
        }
    }

    @Test
    fun mulSquaredAliased() {
        // f(x) = x * x,  f'(x) = 2x.  The aliased-operand gotcha: MulRule returns two
        // contributions both keyed on the same DxirParam; position-based mapping in
        // `applyRegistryRule` correctly routes both to the same tape id and accumulates.
        val primal = DxirBuilder.function("sq") {
            val x = param("x", f32)
            val y = op(OpKind.MUL, listOf(x, x), f32)
            listOf(y)
        }
        for (v in listOf(-3.0f, 0.0f, 2.5f, 7.0f)) {
            val tape = tapeGrad({ x -> x * x }, v)
            val sct = sctGrad(primal, v)
            assertEquals(2.0f * v, sct)
            assertClose(tape, sct, "x*x at v=$v")
        }
    }

    @Test
    fun cubic() {
        // f(x) = x * x * x,  f'(x) = 3x².
        val primal = DxirBuilder.function("cube") {
            val x = param("x", f32)
            val xx = op(OpKind.MUL, listOf(x, x), f32)
            val xxx = op(OpKind.MUL, listOf(xx, x), f32)
            listOf(xxx)
        }
        for (v in listOf(-2.0f, 0.5f, 3.0f)) {
            val tape = tapeGrad({ x -> x * x * x }, v)
            val sct = sctGrad(primal, v)
            assertClose(3.0f * v * v, sct, "cubic expected at v=$v")
            assertClose(tape, sct, "cubic tape-vs-sct at v=$v")
        }
    }

    @Test
    fun polynomialWithValBindings() {
        // f(x) = x² * x² + x  ≡  x⁴ + x,  f'(x) = 4x³ + 1.
        val primal = DxirBuilder.function("poly") {
            val x = param("x", f32)
            val xx = op(OpKind.MUL, listOf(x, x), f32)
            val xxxx = op(OpKind.MUL, listOf(xx, xx), f32)
            val out = op(OpKind.ADD, listOf(xxxx, x), f32)
            listOf(out)
        }
        for (v in listOf(-2.0f, 0.5f, 3.0f)) {
            val tape = tapeGrad({ x -> val y = x * x; y * y + x }, v)
            val sct = sctGrad(primal, v)
            assertClose(4.0f * v * v * v + 1.0f, sct, "poly expected at v=$v")
            assertClose(tape, sct, "poly tape-vs-sct at v=$v")
        }
    }

    @Test
    fun reciprocal() {
        // f(x) = 1 / x,  f'(x) = -1/x².  Exercises the full bridge (registry lookup,
        // DivRule emission, interpreter evaluation) against both the closed-form
        // derivative and the tape path.
        //
        // The Tracer API has no constant-literal surface, so the tape-side formulation
        // uses `valueAndGrad2 { one, x -> one / x }` — feeding `1` as the first argument
        // and extracting only the second gradient (∂x) for the comparison. The first
        // gradient (∂one = 1/x) is discarded.
        val primal = DxirBuilder.function("rec") {
            val x = param("x", f32)
            val one = const(1.0f, f32)
            val y = op(OpKind.DIV, listOf(one, x), f32)
            listOf(y)
        }
        for (v in listOf(-3.0f, 0.5f, 2.0f, 4.0f)) {
            val sct = sctGrad(primal, v)
            val expected = -1.0f / (v * v)
            assertClose(expected, sct, "1/x closed-form at v=$v")
            val (_, _, gradX) = valueAndGrad2<ScalarShape, ScalarShape> { oneT, x -> oneT / x }(
                Tensors.f32Scalar(1.0f),
                Tensors.f32Scalar(v),
            )
            assertClose(gradX.hostF32()[0], sct, "1/x tape-vs-sct at v=$v")
        }
    }

    @Test
    fun reluAtPositive() {
        // f(x) = relu(x) for x > 0 is identity, so f'(x) = 1.
        val primal = DxirBuilder.function("relu_pos") {
            val x = param("x", f32)
            val y = op(OpKind.RELU, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(0.25f, 1.0f, 3.0f, 7.5f)) {
            val tape = tapeGrad({ x -> x.relu() }, v)
            val sct = sctGrad(primal, v)
            assertClose(1.0f, sct, "ReluRule grad at v=$v")
            assertClose(tape, sct, "relu tape-vs-sct at v=$v")
        }
    }

    @Test
    fun reluAtNonPositive() {
        // f(x) = relu(x) for x <= 0 is constantly 0, so f'(x) = 0 (including at x = 0,
        // matching XLA's `compare GT 0` semantics in STEP).
        val primal = DxirBuilder.function("relu_nonpos") {
            val x = param("x", f32)
            val y = op(OpKind.RELU, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(-3.0f, -0.5f, 0.0f)) {
            val tape = tapeGrad({ x -> x.relu() }, v)
            val sct = sctGrad(primal, v)
            assertClose(0.0f, sct, "ReluRule grad at v=$v (should be 0 for x <= 0)")
            assertClose(tape, sct, "relu tape-vs-sct at v=$v")
        }
    }

    // §0.4.66 — tape-vs-SCT parity for the §0.4.63 unary math ops (SQRT/EXP/LOG/TANH/
    // SIGMOID) and §0.4.64's POW. Both paths route through VjpRegistry's rules, so
    // numerical drift between them would mean one of the two dispatch sites is
    // mis-evaluating.

    @Test
    fun scalarBroadcastMatchesTapeAndSctPaths() {
        // §0.4.79 — BROADCAST-based scalar broadcasting (§0.4.77). Constructs the
        // SAME primal two ways:
        //   * Tape:  valueAndGrad2 { x, c -> (x * c).sum() }, which the
        //            Tracer.times(scalar) overload records as
        //            BROADCAST(c) → bcast, MUL(x, bcast), SUM(prod).
        //   * SCT:   DxirFunction with the exact same ops in the same order,
        //            then DxirReverseTransform.apply + evalFunction.
        // Cross-check both grad_x and grad_c on a representative input.
        val f32vec3 = DxirType(F32, listOf(3))
        val primal = io.tlaloc.ir.DxirBuilder.function("x_times_c_sum") {
            val x = param("x", f32vec3)
            val c = param("c", f32)
            val bcastC = op(
                OpKind.BROADCAST,
                listOf(c),
                f32vec3,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val prod = op(OpKind.MUL, listOf(x, bcastC), f32vec3)
            val s = op(OpKind.SUM, listOf(prod), f32)
            listOf(s)
        }
        val gradFn = DxirReverseTransform.apply(primal)
        val xInput = floatArrayOf(2f, 4f, 8f)
        val cInput = floatArrayOf(3f)
        val sctOut = DxirInterpreter.evalFunction(gradFn, listOf(xInput, cInput))
        val sctDx = sctOut[0]
        val sctDc = sctOut[1][0]

        val vg = valueAndGrad2 { x: Tracer<io.tlaloc.core.Rank1<Sym>>, c: Tracer<ScalarShape> ->
            (x * c).sum()
        }
        val (_, tapeDx, tapeDc) = vg(Tensors.f32Vector(xInput), Tensors.f32Scalar(cInput[0]))
        val tapeDxArr = tapeDx.hostF32()

        assertClose(tapeDxArr[0], sctDx[0], "broadcast dx[0]")
        assertClose(tapeDxArr[1], sctDx[1], "broadcast dx[1]")
        assertClose(tapeDxArr[2], sctDx[2], "broadcast dx[2]")
        assertClose(tapeDc.hostF32()[0], sctDc, "broadcast dc")
    }

    @Test
    fun sqrtGradMatchesClosedFormAndTape() {
        // d/dx sqrt(x) = 1 / (2 sqrt(x)).
        val primal = DxirBuilder.function("sqrt") {
            val x = param("x", f32)
            val y = op(OpKind.SQRT, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(0.25f, 1.0f, 4.0f, 9.0f)) {
            val tape = tapeGrad({ x -> x.sqrt() }, v)
            val sct = sctGrad(primal, v)
            val expected = 0.5f / kotlin.math.sqrt(v)
            assertClose(expected, sct, "sqrt closed-form at v=$v")
            assertClose(tape, sct, "sqrt tape-vs-sct at v=$v")
        }
    }

    @Test
    fun expGradEqualsForwardExp() {
        // d/dx exp(x) = exp(x). Both forward and backward paths rely on the same
        // `kotlin.math.exp` under the hood, so any drift would be in dispatch glue.
        val primal = DxirBuilder.function("exp") {
            val x = param("x", f32)
            val y = op(OpKind.EXP, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(-1f, 0f, 0.5f, 2f)) {
            val tape = tapeGrad({ x -> x.exp() }, v)
            val sct = sctGrad(primal, v)
            val expected = kotlin.math.exp(v)
            assertClose(expected, sct, "exp closed-form at v=$v")
            assertClose(tape, sct, "exp tape-vs-sct at v=$v")
        }
    }

    @Test
    fun logGradIsReciprocal() {
        // d/dx log(x) = 1/x. Defined only for x > 0.
        val primal = DxirBuilder.function("log") {
            val x = param("x", f32)
            val y = op(OpKind.LOG, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(0.5f, 1f, 2f, 10f)) {
            val tape = tapeGrad({ x -> x.log() }, v)
            val sct = sctGrad(primal, v)
            assertClose(1f / v, sct, "log closed-form at v=$v")
            assertClose(tape, sct, "log tape-vs-sct at v=$v")
        }
    }

    @Test
    fun tanhGradIsOneMinusTanhSquared() {
        // d/dx tanh(x) = 1 - tanh(x)^2.
        val primal = DxirBuilder.function("tanh") {
            val x = param("x", f32)
            val y = op(OpKind.TANH, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(-2f, 0f, 0.5f, 3f)) {
            val tape = tapeGrad({ x -> x.tanh() }, v)
            val sct = sctGrad(primal, v)
            val t = kotlin.math.tanh(v)
            assertClose(1f - t * t, sct, "tanh closed-form at v=$v")
            assertClose(tape, sct, "tanh tape-vs-sct at v=$v")
        }
    }

    @Test
    fun sigmoidGradIsSigmoidTimesOneMinusSigmoid() {
        // d/dx σ(x) = σ(x)(1 - σ(x)).
        val primal = DxirBuilder.function("sig") {
            val x = param("x", f32)
            val y = op(OpKind.SIGMOID, listOf(x), f32)
            listOf(y)
        }
        for (v in listOf(-3f, 0f, 1f, 4f)) {
            val tape = tapeGrad({ x -> x.sigmoid() }, v)
            val sct = sctGrad(primal, v)
            val s = 1f / (1f + kotlin.math.exp(-v))
            assertClose(s * (1f - s), sct, "sigmoid closed-form at v=$v")
            assertClose(tape, sct, "sigmoid tape-vs-sct at v=$v")
        }
    }

    @Test
    fun powGradBothParamsMatchTape() {
        // f(x, e) = x^e. Reproduces the §0.4.64 GradTest via the bridge: build
        // a two-param primal with POW, take SCT grads through
        // DxirReverseTransform, and cross-check against the tape's valueAndGrad2.
        val primal = DxirBuilder.function("pow") {
            val x = param("x", f32)
            val e = param("e", f32)
            val y = op(OpKind.POW, listOf(x, e), f32)
            listOf(y)
        }
        val gradFn = DxirReverseTransform.apply(primal)
        val vg = valueAndGrad2 { x: Tracer<ScalarShape>, e: Tracer<ScalarShape> -> x.pow(e) }
        for ((xv, ev) in listOf(3f to 2f, 4f to 0.5f, 2f to 3f)) {
            val sctOut = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(xv), floatArrayOf(ev)))
            val sctDx = sctOut[0][0]
            val sctDe = sctOut[1][0]
            val (_, tapeDx, tapeDe) = vg(Tensors.f32Scalar(xv), Tensors.f32Scalar(ev))
            assertClose(tapeDx.hostF32()[0], sctDx, "pow grad_x tape-vs-sct at (x=$xv, e=$ev)")
            assertClose(tapeDe.hostF32()[0], sctDe, "pow grad_e tape-vs-sct at (x=$xv, e=$ev)")
        }
    }

    @Test
    fun sumGrad() {
        // f(x) = sum(x) over x: f32[4],  ∂f/∂xᵢ = 1 for all i.  Exercises SumRule's
        // BROADCAST(upstream, target=x.dims) through both paths.  The scratch-op
        // widening in Backward.applyRegistryRule lets the tape-side scratch BROADCAST
        // produce a size-4 FloatArray via the shape-aware DxirInterpreter.
        val primal = DxirBuilder.function("sum_vec") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.SUM, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        val gradFn = DxirReverseTransform.apply(primal)
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val sct = DxirInterpreter.evalFunction(gradFn, listOf(input)).single()
        val expected = floatArrayOf(1f, 1f, 1f, 1f)
        for (i in expected.indices) assertClose(expected[i], sct[i], "sum SCT[$i]")

        val tapeGrad = grad { x: Tracer<Rank1<Sym>> -> x.sum() }(Tensors.f32Vector(input))
        val tapeArr = tapeGrad.hostF32()
        for (i in expected.indices) assertClose(tapeArr[i], sct[i], "sum tape-vs-sct[$i]")
    }

    @Test
    fun meanGrad() {
        // f(x) = mean(x) over x: f32[4],  ∂f/∂xᵢ = 1/4.  Exercises MeanRule's
        // BROADCAST(MUL(upstream, 1/N), target=x.dims).
        val primal = DxirBuilder.function("mean_vec") {
            val x = param("x", DxirType(F32, listOf(4)))
            val y = op(OpKind.MEAN, listOf(x), DxirType(F32, emptyList()))
            listOf(y)
        }
        val gradFn = DxirReverseTransform.apply(primal)
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val sct = DxirInterpreter.evalFunction(gradFn, listOf(input)).single()
        val expected = floatArrayOf(0.25f, 0.25f, 0.25f, 0.25f)
        for (i in expected.indices) assertClose(expected[i], sct[i], "mean SCT[$i]")

        val tapeGrad = grad { x: Tracer<Rank1<Sym>> -> x.mean() }(Tensors.f32Vector(input))
        val tapeArr = tapeGrad.hostF32()
        for (i in expected.indices) assertClose(tapeArr[i], sct[i], "mean tape-vs-sct[$i]")
    }

    @Test
    fun matmulGrad() {
        // f(A, B) = sum(A @ B) for A, B ∈ f32[2, 2].  ∂f/∂A = ones(2,2) @ Bᵀ,
        // ∂f/∂B = Aᵀ @ ones(2,2).  Exercises the full bridge for MatmulRule:
        // two TRANSPOSEs + two MATMULs in the scratch body, evaluated via
        // DxirInterpreter's rank-2 arms.  The outer SUM keeps the scalar-return
        // hard gate happy.  Concrete numbers match GradTest.matmulBackwardOnSmallSquare.
        val mat = DxirType(F32, listOf(2, 2))
        val primal = DxirBuilder.function("sum_matmul") {
            val a = param("a", mat)
            val b = param("b", mat)
            val ab = op(OpKind.MATMUL, listOf(a, b), mat)
            val s = op(OpKind.SUM, listOf(ab), DxirType(F32, emptyList()))
            listOf(s)
        }
        val aData = floatArrayOf(1f, 2f, 3f, 4f)
        val bData = floatArrayOf(5f, 6f, 7f, 8f)
        val gradFn = DxirReverseTransform.apply(primal)
        val sct = DxirInterpreter.evalFunction(gradFn, listOf(aData, bData))
        val expectedDA = floatArrayOf(11f, 15f, 11f, 15f)
        val expectedDB = floatArrayOf(4f, 4f, 6f, 6f)
        for (i in expectedDA.indices) assertClose(expectedDA[i], sct[0][i], "matmul SCT dA[$i]")
        for (i in expectedDB.indices) assertClose(expectedDB[i], sct[1][i], "matmul SCT dB[$i]")

        val (_, tapeDA, tapeDB) = valueAndGrad2<Rank2<Sym, Sym>, Rank2<Sym, Sym>> { a, b ->
            (a matmul b).sum()
        }(
            Tensors.f32Matrix<Sym, Sym>(2, 2, aData),
            Tensors.f32Matrix<Sym, Sym>(2, 2, bData),
        )
        val tapeAArr = tapeDA.hostF32()
        val tapeBArr = tapeDB.hostF32()
        for (i in expectedDA.indices) assertClose(tapeAArr[i], sct[0][i], "matmul tape-vs-sct dA[$i]")
        for (i in expectedDB.indices) assertClose(tapeBArr[i], sct[1][i], "matmul tape-vs-sct dB[$i]")
    }

    @Test
    fun twoParamMulPlusA() {
        // f(a, b) = a * b + a,  ∂f/∂a = b + 1,  ∂f/∂b = a.  Multi-param SCT + tape grad2.
        val primal = DxirBuilder.function("ab_plus_a") {
            val a = param("a", f32)
            val b = param("b", f32)
            val ab = op(OpKind.MUL, listOf(a, b), f32)
            val out = op(OpKind.ADD, listOf(ab, a), f32)
            listOf(out)
        }
        val grad = DxirReverseTransform.apply(primal)
        for ((av, bv) in listOf(1.0f to 2.0f, -3.0f to 0.5f, 4.0f to 4.0f)) {
            val sct = DxirInterpreter.evalFunction(
                grad,
                listOf(floatArrayOf(av), floatArrayOf(bv)),
            )
            assertClose(bv + 1.0f, sct[0][0], "∂a at ($av, $bv)")
            assertClose(av, sct[1][0], "∂b at ($av, $bv)")

            val (_, gradA, gradB) = valueAndGrad2<ScalarShape, ScalarShape> { a, b -> a * b + a }(
                Tensors.f32Scalar(av),
                Tensors.f32Scalar(bv),
            )
            assertClose(gradA.hostF32()[0], sct[0][0], "tape-∂a at ($av, $bv)")
            assertClose(gradB.hostF32()[0], sct[1][0], "tape-∂b at ($av, $bv)")
        }
    }
}
