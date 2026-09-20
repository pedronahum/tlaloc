package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
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
 *
 * §0.4.430 closes the two §0.4.403/407 recorded tails: MULTI-result
 * COARSENED tangents (per-result tracking through one splice) and IF
 * inside a `primal_body` (the widened splice clones the recursion's
 * value-IF/tangent-IF pairs yield-only). Production coarseners emit
 * single-result, region-free bodies today, so those pins are synthetic
 * by design — the transform contract is the deliverable.
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

    // ── §0.4.430 — multi-result COARSENED tangents (the §0.4.403 recorded
    // tail). Production coarseners emit single-result only today
    // (PhiCalculus.coarsenFunction bails on multi-return primals), so these
    // pins are SYNTHETIC by design — the transform CONTRACT is the
    // deliverable, per-result tangent tracking through one splice.

    /** primal (a, b) → (a + b·b, a·b) — nonlinear per result. */
    private fun mrPrimal(): DxirFunction = DxirBuilder.function("mr_primal") {
        val a = param("a", f32s)
        val b = param("b", f32s)
        val bb = op(OpKind.MUL, listOf(b, b), f32s)
        listOf(op(OpKind.ADD, listOf(a, bb), f32s), op(OpKind.MUL, listOf(a, b), f32s))
    }

    /** gradient (up0, up1, a, b) → (up0 + up1·b, up0·2b + up1·a) — the K+N
     * upstream convention handleCoarsenedAdjoint reads (§0.4.179). */
    private fun mrGradient(): DxirFunction = DxirBuilder.function("mr_gradient") {
        val up0 = param("up0", f32s)
        val up1 = param("up1", f32s)
        val a = param("a", f32s)
        val b = param("b", f32s)
        val two = const(2f, f32s)
        val da = op(OpKind.ADD, listOf(up0, op(OpKind.MUL, listOf(up1, b), f32s)), f32s)
        val db = op(
            OpKind.ADD,
            listOf(
                op(OpKind.MUL, listOf(up0, op(OpKind.MUL, listOf(two, b), f32s)), f32s),
                op(OpKind.MUL, listOf(up1, a), f32s),
            ),
            f32s,
        )
        listOf(da, db)
    }

    @Test
    fun multiResultCoarsenedTangentsMatchDecomposedEquivalentAndAnalytic() {
        // outer(a, b) = (y0, y0·y1) where (y0, y1) = COARSENED(a, b) =
        // (a + b², a·b) — a DIRECT result return AND a consuming op, so both
        // per-index tangent resolution paths (returns map + operand refs) pin.
        // dy0 = da + 2b·db;  dy1 = b·da + a·db;  dz = dy0·y1 + y0·dy1.
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = mrPrimal(),
                gradientBody = mrGradient(),
                readsPrimalIndices = setOf(0, 1),
            )
            listOf(c.result(0), op(OpKind.MUL, listOf(c.result(0), c.result(1)), f32s))
        }
        // Decomposed equivalent: the same math with plain ops — the ordinary
        // per-op tangent path, never reading primal_body as an attr
        // (decomposeCoarsened itself is single-result v1, so the equivalent
        // is hand-built).
        val decomposed = DxirBuilder.function("outer_decomposed") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val bb = op(OpKind.MUL, listOf(b, b), f32s)
            val y0 = op(OpKind.ADD, listOf(a, bb), f32s)
            val y1 = op(OpKind.MUL, listOf(a, b), f32s)
            listOf(y0, op(OpKind.MUL, listOf(y0, y1), f32s))
        }
        val jvpC = DxirForwardTransform.apply(outer)
        val jvpD = DxirForwardTransform.apply(decomposed)
        // Quarter-integer grid — exact in f32.
        for ((a, b) in listOf(1.5f to 0.5f, -0.75f to -1.25f, 2f to 0.25f)) {
            for ((da, db) in listOf(1f to 0.25f, -0.5f to 2f)) {
                val inputs = listOf(
                    floatArrayOf(a), floatArrayOf(b), floatArrayOf(da), floatArrayOf(db),
                )
                val outC = DxirInterpreter.evalFunction(jvpC, inputs)
                val outD = DxirInterpreter.evalFunction(jvpD, inputs)
                // Return layout: (y0, z, dy0, dz).
                val y0 = a + b * b
                val y1 = a * b
                val dy0 = da + 2f * b * db
                val dy1 = b * da + a * db
                val dz = dy0 * y1 + y0 * dy1
                assertEquals(y0, outC[0][0], "y0 drift at a=$a b=$b")
                assertEquals(y0 * y1, outC[1][0], "z drift at a=$a b=$b")
                assertEquals(dy0, outC[2][0], "analytic dy0 at a=$a b=$b da=$da db=$db")
                assertEquals(dz, outC[3][0], "analytic dz at a=$a b=$b da=$da db=$db")
                for (i in 0 until 4) {
                    assertEquals(
                        outD[i][0], outC[i][0],
                        "coarsened vs decomposed drift at output $i, a=$a b=$b da=$da db=$db",
                    )
                }
            }
        }
    }

    @Test
    fun multiResultCoarsenedSatisfiesJvpVjpCrossIdentity() {
        // z = c0·c1 over the SAME multi-result node: the forward arm consumes
        // primal_body (one splice, per-result tangents) while
        // handleCoarsenedAdjoint consumes gradient_body (K=2 upstreams,
        // §0.4.179) — two attrs, two transforms, one number:
        // dz == ∇z·(da, db).
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = mrPrimal(),
                gradientBody = mrGradient(),
                readsPrimalIndices = setOf(0, 1),
            )
            listOf(op(OpKind.MUL, listOf(c.result(0), c.result(1)), f32s))
        }
        val jvp = DxirForwardTransform.apply(outer)
        val grad = DxirReverseTransform.apply(outer)
        for ((a, b) in listOf(1.5f to 0.5f, -0.75f to 2f, 0.25f to -1.5f)) {
            val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(a), floatArrayOf(b)))
            for ((da, db) in listOf(1f to 0f, 0f to 1f, -0.5f to 0.25f)) {
                val dz = DxirInterpreter.evalFunction(
                    jvp,
                    listOf(floatArrayOf(a), floatArrayOf(b), floatArrayOf(da), floatArrayOf(db)),
                )[1][0]
                val want = g[0][0] * da + g[1][0] * db
                assertTrue(
                    abs(dz - want) <= 1e-5f * maxOf(1f, abs(want)),
                    "JVP⇄VJP cross-identity broken at a=$a b=$b da=$da db=$db: " +
                        "dz=$dz vs ⟨∇z,(da,db)⟩=$want",
                )
            }
        }
    }

    // ── §0.4.430 — IF inside a COARSENED primal_body (the §0.4.407 recorded
    // tail): apply's recursion transforms the IF into a value-IF/tangent-IF
    // pair, and the widened splice clones them into the outer stream
    // yield-only — the emitter-compatible shape.

    /** primal x → if (x > 0) x·x else −x, real ops in the branches. */
    private fun ifPrimalBody(): DxirFunction = DxirBuilder.function("if_primal") {
        val x = param("x", f32s)
        val pred = op(OpKind.STEP, listOf(x), DxirType(Bool, emptyList()))
        val ifResult = ifOp(
            cond = pred,
            types = listOf(f32s),
            thenRegion = region {
                val sq = op(OpKind.MUL, listOf(x, x), f32s)
                yields(sq)
            },
            elseRegion = region {
                val neg = op(OpKind.NEG, listOf(x), f32s)
                yields(neg)
            },
        )
        listOf(ifResult)
    }

    /** gradient (up, x) → up·(s·2x − (1−s)), s = STEP(x) as f32 — region-free
     * mask math computing the same piecewise slope (2x above, −1 below). */
    private fun ifGradientBody(): DxirFunction = DxirBuilder.function("if_gradient") {
        val up = param("upstream", f32s)
        val x = param("x", f32s)
        val one = const(1f, f32s)
        val two = const(2f, f32s)
        val s = op(OpKind.STEP, listOf(x), f32s)
        val twoX = op(OpKind.MUL, listOf(two, x), f32s)
        val term1 = op(OpKind.MUL, listOf(s, twoX), f32s)
        val oneMinusS = op(OpKind.SUB, listOf(one, s), f32s)
        val slope = op(OpKind.SUB, listOf(term1, oneMinusS), f32s)
        listOf(op(OpKind.MUL, listOf(up, slope), f32s))
    }

    @Test
    fun ifInsidePrimalBodySplicesOnBothSidesOfTheBranch() {
        // g(x) = COARSENED(x; primal if (x>0) x² else −x) → dg = 2x·dx above,
        // −dx below. Oracles: analytic both branches; the JVP⇄VJP
        // cross-identity (reverse consumes the region-FREE gradient_body's
        // mask math — an entirely independent encoding); the §0.4.407
        // top-level IF arm over the same body (a different code path: walk
        // arm vs splice clone); and the structural yield-only pin.
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = ifPrimalBody(),
                    gradientBody = ifGradientBody(),
                    readsPrimalIndices = setOf(0),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        val jvpTop = DxirForwardTransform.apply(ifPrimalBody())
        val grad = DxirReverseTransform.apply(outer)

        // The spliced IFs arrive yield-only — exactly two (value + tangent),
        // the only IF shape DxirToIrSynthesis.irIfOp and the emitter accept.
        val ifOps = jvp.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.IF }
        assertEquals(2, ifOps.size, "value IF + tangent IF through the splice")
        for (node in ifOps) {
            assertTrue(
                node.regions.all { r -> r.blocks.single().body.isEmpty() },
                "spliced IFs must be yield-only (empty-body regions)",
            )
        }

        for (x in listOf(2f, 0.5f, -0.75f, -3f)) {
            for (dx in listOf(1f, -0.5f)) {
                val out = DxirInterpreter.evalFunction(
                    jvp, listOf(floatArrayOf(x), floatArrayOf(dx)),
                )
                val wantY = if (x > 0f) x * x else -x
                val wantD = if (x > 0f) 2f * x * dx else -dx
                assertEquals(wantY, out[0][0], "primal value drift at x=$x")
                assertTrue(
                    abs(out[1][0] - wantD) < 1e-5f,
                    "analytic tangent at x=$x dx=$dx: got ${out[1][0]}, want $wantD",
                )
                val outTop = DxirInterpreter.evalFunction(
                    jvpTop, listOf(floatArrayOf(x), floatArrayOf(dx)),
                )
                assertEquals(
                    outTop[1][0], out[1][0],
                    "splice vs top-level IF arm drift at x=$x dx=$dx",
                )
                val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))[0][0]
                assertTrue(
                    abs(out[1][0] - g * dx) < 1e-5f,
                    "JVP⇄VJP cross-identity broken at x=$x dx=$dx: " +
                        "dy=${out[1][0]} vs grad·dx=${g * dx}",
                )
            }
        }
    }

    @Test
    fun multiResultIfInsidePrimalBodySplices() {
        // primal x → r0 + r1 where (r0, r1) = if (x>0) (−x, x²) else (x, x²)
        // — the §0.4.155 multi-live shape INSIDE a primal_body: the splice
        // must clone a multi-result IF and re-wrap DxirOpResult references
        // through the clone. x>0: dy = (2x−1)·dx;  x≤0: dy = (2x+1)·dx.
        val primal = DxirBuilder.function("mr_if_primal") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), DxirType(Bool, emptyList()))
            val negX = op(OpKind.NEG, listOf(x), f32s)
            val xSq = op(OpKind.MUL, listOf(x, x), f32s)
            val mrIf = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32s, f32s),
                regions = listOf(
                    region { yields(negX, xSq) },
                    region { yields(x, xSq) },
                ),
            )
            listOf(op(OpKind.ADD, listOf(mrIf.result(0), mrIf.result(1)), f32s))
        }
        // gradient (up, x) → up·(s·(2x−1) + (1−s)·(2x+1)), s = STEP(x) as f32.
        val gradient = DxirBuilder.function("mr_if_gradient") {
            val up = param("upstream", f32s)
            val x = param("x", f32s)
            val one = const(1f, f32s)
            val two = const(2f, f32s)
            val s = op(OpKind.STEP, listOf(x), f32s)
            val twoX = op(OpKind.MUL, listOf(two, x), f32s)
            val above = op(OpKind.SUB, listOf(twoX, one), f32s)
            val below = op(OpKind.ADD, listOf(twoX, one), f32s)
            val oneMinusS = op(OpKind.SUB, listOf(one, s), f32s)
            val slope = op(
                OpKind.ADD,
                listOf(
                    op(OpKind.MUL, listOf(s, above), f32s),
                    op(OpKind.MUL, listOf(oneMinusS, below), f32s),
                ),
                f32s,
            )
            listOf(op(OpKind.MUL, listOf(up, slope), f32s))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = primal,
                    gradientBody = gradient,
                    readsPrimalIndices = setOf(0),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        val jvpTop = DxirForwardTransform.apply(primal)
        val grad = DxirReverseTransform.apply(outer)
        for (x in listOf(2f, 0.5f, -0.75f, -3f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            val wantY = if (x > 0f) -x + x * x else x + x * x
            val wantD = if (x > 0f) 2f * x - 1f else 2f * x + 1f
            assertEquals(wantY, out[0][0], "primal value drift at x=$x")
            assertTrue(
                abs(out[1][0] - wantD) < 1e-5f,
                "analytic tangent at x=$x: got ${out[1][0]}, want $wantD",
            )
            val top = DxirInterpreter.evalFunction(jvpTop, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertEquals(top[1][0], out[1][0], "splice vs top-level MR-IF arm drift at x=$x")
            val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))[0][0]
            assertTrue(
                abs(out[1][0] - g) < 1e-5f,
                "JVP⇄VJP cross-identity broken at x=$x: dy=${out[1][0]} vs grad=$g",
            )
        }
    }

    @Test
    fun nestedMultiResultCoarsenedInsidePrimalBodyClonesThroughTheSplice() {
        // inner (x) → (x², 2x) as a 2-result COARSENED; mid(x) = Σ inner =
        // x² + 2x; outer = COARSENED(x; primal_body = mid). apply's recursion
        // hits the MR arm inside mid's transform, and the OUTER splice must
        // then clone the resulting multi-result COARSENED value node (the
        // opMulti splice path) with per-index result re-wrapping.
        // dy = (2x + 2)·dx.
        val innerPrimal = DxirBuilder.function("inner_primal") {
            val x = param("x", f32s)
            listOf(op(OpKind.MUL, listOf(x, x), f32s), op(OpKind.ADD, listOf(x, x), f32s))
        }
        val innerGradient = DxirBuilder.function("inner_gradient") {
            val up0 = param("up0", f32s)
            val up1 = param("up1", f32s)
            val x = param("x", f32s)
            val two = const(2f, f32s)
            listOf(
                op(
                    OpKind.ADD,
                    listOf(
                        op(OpKind.MUL, listOf(up0, op(OpKind.MUL, listOf(two, x), f32s)), f32s),
                        op(OpKind.MUL, listOf(up1, two), f32s),
                    ),
                    f32s,
                ),
            )
        }
        val mid = DxirBuilder.function("mid") {
            val x = param("x", f32s)
            val c = coarsened(
                operands = listOf(x),
                primalBody = innerPrimal,
                gradientBody = innerGradient,
                readsPrimalIndices = setOf(0),
            )
            listOf(op(OpKind.ADD, listOf(c.result(0), c.result(1)), f32s))
        }
        val midGradient = DxirBuilder.function("mid_gradient") {
            val up = param("upstream", f32s)
            val x = param("x", f32s)
            val two = const(2f, f32s)
            val slope = op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(two, x), f32s), two), f32s)
            listOf(op(OpKind.MUL, listOf(up, slope), f32s))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = mid,
                    gradientBody = midGradient,
                    readsPrimalIndices = setOf(0),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        for (x in listOf(-1.5f, 0.25f, 2f)) {
            for (dx in listOf(1f, -0.5f)) {
                val out = DxirInterpreter.evalFunction(
                    jvp, listOf(floatArrayOf(x), floatArrayOf(dx)),
                )
                assertEquals(x * x + 2f * x, out[0][0], "nested MR primal value drift at x=$x")
                val want = (2f * x + 2f) * dx
                assertTrue(
                    abs(out[1][0] - want) < 1e-5f,
                    "nested MR tangent at x=$x dx=$dx: got ${out[1][0]}, want $want",
                )
            }
        }
    }

    // NOTE: a COARSENED with a missing `primal_body` attr is unconstructible —
    // DxirFunction.init's validateCoarsenedShape refuses it at build time
    // (pinned in CoarsenedOpTest), so the transform's own missing-attr error
    // is defensive dead code, mirroring decomposeCoarsened's. A multi-result
    // COARSENED carrying a USER tangent_body is likewise unconstructible
    // (validateCoarsenedShape's single-result tangent contract), so the
    // transform's MR-user check is defensive too.
}
