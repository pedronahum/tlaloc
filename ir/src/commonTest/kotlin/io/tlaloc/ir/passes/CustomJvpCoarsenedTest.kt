package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.416 — Phase B5, the forward side, at the IR level: a COARSENED node
 * carrying a USER `tangent_body` (`user_gradient = true` — the customJvp /
 * customVjpJvp call-forms' lowering product) must (1) splice the USER's
 * tangent verbatim in forward mode — proven by a deliberately
 * NON-mathematical jvpFn whose answer differs from auto-differentiating the
 * primal, so no fallback can pass; (2) REFUSE reverse mode loudly by name
 * when NO `gradient_body` is present (customJvp-only — the exact mirror of
 * the §0.4.415 forward refusal); (3) with BOTH bodies (customVjpJvp), run
 * each mode's OWN body — pinned separately with deliberately INCONSISTENT
 * bodies (the design doc's semantic-fork principle made explicit) and pinned
 * jointly with CONSISTENT bodies through the JVP⇄VJP cross-identity;
 * (4) compose under hessian (forward-over-reverse): the tangent_body splice
 * fires inside a reverse-produced body when the COARSENED survives the
 * reverse clone; and (5) honour the runtime shape contract — a
 * sentinel-typed user tangent wraps in CHECK_SHAPE_LIKE against the node's
 * own value, while statically concrete ones stay unwrapped (the construction
 * type check already pinned them).
 */
class CustomJvpCoarsenedTest {

    private val f32s = DxirType(F32, emptyList())

    /** primal: x → x·x. */
    private fun squarePrimal(): DxirFunction = DxirBuilder.function("customJvp_primal") {
        val x = param("x", f32s)
        listOf(op(OpKind.MUL, listOf(x, x), f32s))
    }

    /** NON-mathematical user tangent: (x, dx) → 3·dx — auto-differentiating
     * the primal would give 2x·dx, so only the user body yields 3·dx. */
    private fun threeTimesTangentJvp(): DxirFunction = DxirBuilder.function("customJvp_jvp") {
        param("x", f32s)
        val dx = param("dx", f32s)
        val three = const(3f, f32s)
        listOf(op(OpKind.MUL, listOf(dx, three), f32s))
    }

    /** The TRUE tangent of the square: (x, dx) → 2x·dx. */
    private fun trueSquareJvp(): DxirFunction = DxirBuilder.function("customJvp_jvp") {
        val x = param("x", f32s)
        val dx = param("dx", f32s)
        val two = const(2f, f32s)
        listOf(op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(x, dx), f32s), two), f32s))
    }

    /** The TRUE adjoint of the square: (upstream, x) → 2x·upstream. */
    private fun trueSquareVjp(): DxirFunction = DxirBuilder.function("customVjp_vjp") {
        val up = param("upstream", f32s)
        val x = param("x", f32s)
        val two = const(2f, f32s)
        listOf(op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(up, x), f32s), two), f32s))
    }

    @Test
    fun userTangentBodyIsWhatRunsInForwardMode() {
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = null,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                    tangentBody = threeTimesTangentJvp(),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        for (x in listOf(-2f, 0.5f, 5f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertEquals(x * x, out[0][0], "the value stream still runs f at x=$x")
            // The USER's 3·dx, never the auto-tangent's 2x·dx.
            assertEquals(3f, out[1][0], "user tangent must run verbatim at x=$x")
        }
    }

    @Test
    fun reverseModeRefusesCustomJvpOnlyLoudlyByName() {
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = null,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                    tangentBody = threeTimesTangentJvp(),
                ),
            )
        }
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(outer) }
        assertTrue(
            "customJvp" in ex.message.orEmpty(),
            "the reverse refusal must name customJvp; got: ${ex.message}",
        )
    }

    @Test
    fun customVjpJvpRunsEachModesOwnBodySeparately() {
        // DELIBERATELY inconsistent bodies: vjpFn = 5·upstream, jvpFn = 3·dx.
        // Each mode must run ITS body — reverse yields 5, forward yields 3,
        // and neither may fall back to the other or to the primal's 2x.
        val vjp = DxirBuilder.function("customVjpJvp_vjp") {
            val up = param("upstream", f32s)
            param("x", f32s)
            val five = const(5f, f32s)
            listOf(op(OpKind.MUL, listOf(up, five), f32s))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = vjp,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                    tangentBody = threeTimesTangentJvp(),
                ),
            )
        }
        val grad = DxirReverseTransform.apply(outer)
        assertEquals(
            5f,
            DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))[0][0],
            "reverse mode runs the user vjpFn (5), not the jvpFn (3) or the math (2x)",
        )
        val jvp = DxirForwardTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(4f), floatArrayOf(1f)))
        assertEquals(16f, out[0][0], "value stream")
        assertEquals(
            3f,
            out[1][0],
            "forward mode runs the user jvpFn (3), not the vjpFn (5) or the math (2x)",
        )
    }

    @Test
    fun crossIdentityHoldsWithConsistentBodies() {
        // CONSISTENT bodies (both the true derivative of x²): the JVP⇄VJP
        // cross-identity ⟨∇f(x), v⟩ = jvp_f(x, v) must hold across a sweep —
        // two entirely different user encodings of the same derivative
        // producing one number.
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = trueSquareVjp(),
                    readsPrimalIndices = setOf(0),
                    userGradient = true,
                    tangentBody = trueSquareJvp(),
                ),
            )
        }
        val grad = DxirReverseTransform.apply(outer)
        val jvp = DxirForwardTransform.apply(outer)
        for (x in listOf(-1.5f, 0.25f, 2f, 7f)) {
            for (v in listOf(1f, -0.5f, 2.5f)) {
                val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))[0][0]
                val dy = DxirInterpreter.evalFunction(
                    jvp,
                    listOf(floatArrayOf(x), floatArrayOf(v)),
                )[1][0]
                assertTrue(
                    abs(g * v - dy) < 1e-4f,
                    "cross-identity at x=$x, v=$v: ⟨grad, v⟩=${g * v} vs jvp=$dy",
                )
            }
        }
    }

    @Test
    fun hessianComposesThroughTheSplicedTangentBody() {
        // forward-over-reverse where the COARSENED SURVIVES into the
        // reverse-produced body: a downstream MUL's adjoint reads its
        // operands, so the reverse transform clones the coarsened node into
        // the gradient function for value recomputation — and the outer
        // forward pass then meets a user_gradient COARSENED mid-body,
        // exercising the tangent_body splice INSIDE a reverse-produced body.
        // With consistent bodies (c(x) = x², true vjp and jvp), f(x) =
        // Σ c(x)·c(x) = Σ x⁴ gives ∇f = 4x³ and H·v = 12x²·v.
        val v3 = DxirType(F32, listOf(3))
        val primal = DxirBuilder.function("p") {
            val x = param("x", v3)
            listOf(op(OpKind.MUL, listOf(x, x), v3))
        }
        val vjp = DxirBuilder.function("g") {
            val up = param("upstream", v3)
            val x = param("x", v3)
            val two = const(2f, f32s)
            listOf(op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(up, x), v3), two), v3))
        }
        val jvpBody = DxirBuilder.function("t") {
            val x = param("x", v3)
            val dx = param("dx", v3)
            val two = const(2f, f32s)
            listOf(op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(x, dx), v3), two), v3))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", v3)
            val c = coarsened(
                operands = listOf(x),
                primalBody = primal,
                gradientBody = vjp,
                readsPrimalIndices = setOf(0),
                userGradient = true,
                tangentBody = jvpBody,
            )
            val sq = op(OpKind.MUL, listOf(c, c), v3)
            listOf(op(OpKind.SUM, listOf(sq), f32s))
        }
        val grad = DxirReverseTransform.apply(outer)
        assertTrue(
            grad.body.any { it is DxirOp && it.op == OpKind.COARSENED },
            "precondition: the COARSENED must survive into the gradient body " +
                "(MUL's adjoint reads the coarsened result) for this test to exercise the splice",
        )
        val hvp = DxirForwardTransform.apply(grad)
        val x = floatArrayOf(1f, -2f, 3f)
        val v = floatArrayOf(0.5f, 1f, -1f)
        val out = DxirInterpreter.evalFunction(hvp, listOf(x, v))
        // Value half: ∇f(x) = 4x³; tangent half: H·v = 12x²·v.
        for (i in 0 until 3) {
            val g = 4f * x[i] * x[i] * x[i]
            val hv = 12f * x[i] * x[i] * v[i]
            assertTrue(
                abs(out[0][i] - g) < 1e-4f,
                "gradient value stream [$i]: got ${out[0][i]}, want $g",
            )
            assertTrue(
                abs(out[1][i] - hv) < 1e-3f,
                "H·v through the spliced tangent [$i]: got ${out[1][i]}, want $hv",
            )
        }
    }

    @Test
    fun sentinelUserTangentWrapsInCheckShapeLike() {
        // A sentinel-typed result makes the tangent's runtime extent
        // statically undecidable, so the splice must wrap the user tangent in
        // the runtime assert against the node's own value clone; a fully
        // concrete node stays unwrapped (the construction-time type check
        // already pinned its shape).
        val sv = DxirType(F32, listOf(-1))
        val primal = DxirBuilder.function("p") {
            val x = param("x", sv)
            listOf(op(OpKind.MUL, listOf(x, x), sv))
        }
        val jvpBody = DxirBuilder.function("t") {
            param("x", sv)
            val dx = param("dx", sv)
            val three = const(3f, f32s)
            listOf(op(OpKind.MUL, listOf(dx, three), sv))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", sv)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = primal,
                    gradientBody = null,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                    tangentBody = jvpBody,
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        assertTrue(
            jvp.body.any { it is DxirOp && it.op == OpKind.CHECK_SHAPE_LIKE },
            "sentinel user tangent must wrap in CHECK_SHAPE_LIKE:\n" +
                jvp.body.filterIsInstance<DxirOp>().map { it.op }.toString(),
        )

        val concreteOuter = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = null,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                    tangentBody = threeTimesTangentJvp(),
                ),
            )
        }
        val concreteJvp = DxirForwardTransform.apply(concreteOuter)
        assertTrue(
            concreteJvp.body.none { it is DxirOp && it.op == OpKind.CHECK_SHAPE_LIKE },
            "statically concrete user tangents must not pay the runtime assert",
        )
    }

    @Test
    fun tangentBodyContractIsValidatedAtConstruction() {
        // (1) A user node with NEITHER gradient_body NOR tangent_body is
        // unconstructible — user_gradient alone doesn't lift the §0.4.31
        // gradient_body requirement.
        val exNoBodies = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("outer") {
                val x = param("x", f32s)
                listOf(
                    coarsened(
                        operands = listOf(x),
                        primalBody = squarePrimal(),
                        gradientBody = null,
                        readsPrimalIndices = emptySet(),
                        userGradient = true,
                    ),
                )
            }
        }
        assertTrue(
            "gradient_body" in exNoBodies.message.orEmpty(),
            "expected the gradient_body requirement; got: ${exNoBodies.message}",
        )
        // (2) tangent_body param count must be 2·N (primals…, tangents…).
        val badTangent = DxirBuilder.function("t") {
            val dx = param("dx", f32s)
            listOf(op(OpKind.MUL, listOf(dx, const(3f, f32s)), f32s))
        }
        val exBadParams = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("outer") {
                val x = param("x", f32s)
                listOf(
                    coarsened(
                        operands = listOf(x),
                        primalBody = squarePrimal(),
                        gradientBody = null,
                        readsPrimalIndices = emptySet(),
                        userGradient = true,
                        tangentBody = badTangent,
                    ),
                )
            }
        }
        assertTrue(
            "tangent_body.params count" in exBadParams.message.orEmpty(),
            "expected the 2·N param-count check; got: ${exBadParams.message}",
        )
        // (3) tangent_body without user_gradient is refused — machine
        // coarsening derives tangents from primal_body, never stores them.
        val exNoUser = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("outer") {
                val x = param("x", f32s)
                listOf(
                    coarsened(
                        operands = listOf(x),
                        primalBody = squarePrimal(),
                        gradientBody = trueSquareVjp(),
                        readsPrimalIndices = emptySet(),
                        userGradient = false,
                        tangentBody = threeTimesTangentJvp(),
                    ),
                )
            }
        }
        assertTrue(
            "without `user_gradient`" in exNoUser.message.orEmpty(),
            "expected the user_gradient pairing check; got: ${exNoUser.message}",
        )
    }
}
