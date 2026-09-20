package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.403 — Phase B3: the COARSENED forward arm in [DxirForwardTransform].
 * The tangent of a coarsened op is the forward transform of its stored
 * `primal_body`, spliced inline — the mirror image of
 * [DxirReverseTransform]'s `handleCoarsenedAdjoint` consuming
 * `gradient_body`.
 *
 * Oracles, per the house discipline:
 *  - **decomposeCoarsened**: `tangent(coarsened f)` must numerically equal
 *    `tangent(decomposeCoarsened(f))` — the decomposed body takes the
 *    ordinary per-op tangent path, so agreement pins the splice against an
 *    independent code path for free.
 *  - **Analytic pins** for every surface (2·a·da for the square, etc.).
 *  - **JVP⇄VJP cross-identity**: through a [PhiCalculus.coarsenFunction]
 *    product, the forward tangent (via `primal_body`) must equal
 *    `grad · da` where grad flows through `handleCoarsenedAdjoint` (via
 *    `gradient_body`) — two entirely different attrs, two transforms.
 */
class DxirForwardCoarsenedTest {

    private val f32s = DxirType(F32, emptyList())
    private val f32v3 = DxirType(F32, listOf(3))

    /** primal: x → x·x. */
    private fun squarePrimal(): DxirFunction = DxirBuilder.function("square_primal") {
        val x = param("x", f32s)
        listOf(op(OpKind.MUL, listOf(x, x), f32s))
    }

    /** gradient: (upstream, x) → 2·upstream·x. */
    private fun squareGradient(): DxirFunction = DxirBuilder.function("square_gradient") {
        val up = param("upstream", f32s)
        val x = param("x", f32s)
        val two = const(2f, f32s)
        listOf(op(OpKind.MUL, listOf(up, op(OpKind.MUL, listOf(two, x), f32s)), f32s))
    }

    @Test
    fun handBuiltCoarsenedTangentMatchesDecomposedOracleAndAnalytic() {
        // g(a) = COARSENED(a; primal a·a) + a  →  dg = 2·a·da + da.
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = squarePrimal(),
                gradientBody = squareGradient(),
                readsPrimalIndices = setOf(0),
            )
            listOf(op(OpKind.ADD, listOf(c, a), f32s))
        }
        val jvpCoarse = DxirForwardTransform.apply(outer)
        val jvpDecomp = DxirForwardTransform.apply(decomposeCoarsened(outer))
        for (a in listOf(-2f, 0f, 0.5f, 3f)) {
            for (da in listOf(1f, -0.25f, 2f)) {
                val inputs = listOf(floatArrayOf(a), floatArrayOf(da))
                val outC = DxirInterpreter.evalFunction(jvpCoarse, inputs)
                val outD = DxirInterpreter.evalFunction(jvpDecomp, inputs)
                // Return layout: (y, dy).
                assertEquals(a * a + a, outC[0][0], "primal value drift at a=$a")
                val want = 2f * a * da + da
                assertTrue(
                    abs(outC[1][0] - want) < 1e-5f,
                    "analytic tangent at a=$a da=$da: got ${outC[1][0]}, want $want",
                )
                assertEquals(
                    outD[1][0], outC[1][0],
                    "coarsened vs decomposed tangent drift at a=$a da=$da",
                )
            }
        }
    }

    @Test
    fun coarsenFunctionProductSatisfiesJvpVjpCrossIdentity() {
        // f(a) = a³ + 2a. tangent = (3a² + 2)·da; grad = 3a² + 2.
        // The forward arm consumes primal_body; DxirReverseTransform's
        // handleCoarsenedAdjoint consumes gradient_body — independent paths.
        val fn = DxirBuilder.function("cubic") {
            val a = param("a", f32s)
            val two = const(2f, f32s)
            val a2 = op(OpKind.MUL, listOf(a, a), f32s)
            val a3 = op(OpKind.MUL, listOf(a2, a), f32s)
            listOf(op(OpKind.ADD, listOf(a3, op(OpKind.MUL, listOf(two, a), f32s)), f32s))
        }
        val coarsened = PhiCalculus.coarsenFunction(fn, engine = null)
        // Preflight: the product really is COARSENED-bearing, or this test
        // silently degrades into a plain straight-line re-run.
        assertTrue(
            coarsened.body.any { it is DxirOp && it.op == OpKind.COARSENED },
            "coarsenFunction produced no COARSENED op — test precondition broken",
        )
        val jvp = DxirForwardTransform.apply(coarsened)
        val grad = DxirReverseTransform.apply(coarsened)
        for (a in listOf(-1.5f, 0f, 1f, 2.5f)) {
            for (da in listOf(1f, 0.5f, -2f)) {
                val dy = DxirInterpreter.evalFunction(
                    jvp, listOf(floatArrayOf(a), floatArrayOf(da)),
                )[1][0]
                val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(a)))[0][0]
                val want = (3f * a * a + 2f) * da
                assertTrue(
                    abs(dy - want) < 1e-4f,
                    "analytic tangent at a=$a da=$da: got $dy, want $want",
                )
                assertTrue(
                    abs(dy - g * da) < 1e-4f,
                    "JVP⇄VJP cross-identity broken at a=$a da=$da: dy=$dy, grad·da=${g * da}",
                )
            }
        }
    }

    @Test
    fun tensorTwoOperandCoarsenedThreadsProductRuleTangents() {
        // g(x, w) = COARSENED(x, w; primal x⊙w)  →  dy = dx⊙w + x⊙dw.
        // Pins multi-operand param seeding (values THEN tangents) and
        // non-scalar types through the splice.
        val primal = DxirBuilder.function("mul_primal") {
            val x = param("x", f32v3)
            val w = param("w", f32v3)
            listOf(op(OpKind.MUL, listOf(x, w), f32v3))
        }
        val gradient = DxirBuilder.function("mul_gradient") {
            val up = param("upstream", f32v3)
            val x = param("x", f32v3)
            val w = param("w", f32v3)
            listOf(op(OpKind.MUL, listOf(up, w), f32v3), op(OpKind.MUL, listOf(up, x), f32v3))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32v3)
            val w = param("w", f32v3)
            listOf(
                coarsened(
                    operands = listOf(x, w),
                    primalBody = primal,
                    gradientBody = gradient,
                    readsPrimalIndices = setOf(0, 1),
                ),
            )
        }
        val jvpCoarse = DxirForwardTransform.apply(outer)
        val jvpDecomp = DxirForwardTransform.apply(decomposeCoarsened(outer))
        val x = floatArrayOf(1f, -2f, 0.5f)
        val w = floatArrayOf(3f, 0f, -1.5f)
        val dx = floatArrayOf(0.1f, 0.2f, -0.3f)
        val dw = floatArrayOf(-1f, 0.5f, 2f)
        val inputs = listOf(x, w, dx, dw)
        val outC = DxirInterpreter.evalFunction(jvpCoarse, inputs)
        val outD = DxirInterpreter.evalFunction(jvpDecomp, inputs)
        for (i in 0 until 3) {
            val want = dx[i] * w[i] + x[i] * dw[i]
            assertTrue(
                abs(outC[1][i] - want) < 1e-6f,
                "product-rule tangent [$i]: got ${outC[1][i]}, want $want",
            )
            assertEquals(outD[1][i], outC[1][i], "coarsened vs decomposed drift at [$i]")
        }
    }

    @Test
    fun nestedCoarsenedRecursesThroughPrimalBodies() {
        // mid(a) = COARSENED(a; primal a·a) + a  (contains a COARSENED);
        // outer(a) = COARSENED(a; primal_body = mid)  →  tangent (2a+1)·da,
        // exercising apply's recursion into a primal_body that itself
        // carries a COARSENED op.
        val mid = DxirBuilder.function("mid") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = squarePrimal(),
                gradientBody = squareGradient(),
                readsPrimalIndices = setOf(0),
            )
            listOf(op(OpKind.ADD, listOf(c, a), f32s))
        }
        // Hand gradient for mid: (up, a) → up·(2a + 1).
        val midGradient = DxirBuilder.function("mid_gradient") {
            val up = param("upstream", f32s)
            val a = param("a", f32s)
            val two = const(2f, f32s)
            val one = const(1f, f32s)
            val slope = op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(two, a), f32s), one), f32s)
            listOf(op(OpKind.MUL, listOf(up, slope), f32s))
        }
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            listOf(
                coarsened(
                    operands = listOf(a),
                    primalBody = mid,
                    gradientBody = midGradient,
                    readsPrimalIndices = setOf(0),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        for (a in listOf(-1f, 0f, 2f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(a), floatArrayOf(1f)))
            assertEquals(a * a + a, out[0][0], "nested primal value drift at a=$a")
            val want = 2f * a + 1f
            assertTrue(
                abs(out[1][0] - want) < 1e-5f,
                "nested tangent at a=$a: got ${out[1][0]}, want $want",
            )
        }
    }

    @Test
    fun multiResultCoarsenedRefusedLoudlyByName() {
        // primal (a, b) → (a+b, a−b): a 2-result COARSENED must refuse with
        // the named message, not an incidental invariant failure downstream.
        val primal = DxirBuilder.function("addsub") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            listOf(op(OpKind.ADD, listOf(a, b), f32s), op(OpKind.SUB, listOf(a, b), f32s))
        }
        val gradient = DxirBuilder.function("addsub_gradient") {
            val up0 = param("up0", f32s)
            val up1 = param("up1", f32s)
            param("a", f32s)
            param("b", f32s)
            listOf(
                op(OpKind.ADD, listOf(up0, up1), f32s),
                op(OpKind.SUB, listOf(up0, up1), f32s),
            )
        }
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = gradient,
                readsPrimalIndices = emptySet(),
            )
            listOf(op(OpKind.ADD, listOf(c.result(0), c.result(1)), f32s))
        }
        try {
            DxirForwardTransform.apply(outer)
            fail("expected multi-result COARSENED refusal")
        } catch (e: IllegalStateException) {
            assertTrue(
                "multi-result COARSENED" in (e.message ?: ""),
                "refusal must name multi-result COARSENED; got: ${e.message}",
            )
        }
    }

    // NOTE: a COARSENED with a missing `primal_body` attr is unconstructible —
    // DxirFunction.init's validateCoarsenedShape refuses it at build time
    // (pinned in CoarsenedOpTest), so the transform's own missing-attr error
    // is defensive dead code, mirroring decomposeCoarsened's.
}
