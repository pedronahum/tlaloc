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
 * §0.4.415 — Phase B5 (customVjp) at the IR level: a COARSENED node whose
 * `gradient_body` is USER-supplied (`user_gradient = true`, the customVjp
 * call-form's lowering product) must (1) splice the USER's adjoint verbatim in
 * reverse mode — proven by a deliberately NON-mathematical vjpFn whose answer
 * differs from what composing the primal would give, so no fallback can pass;
 * (2) differentiate AGAIN through the spliced body (rev∘custom nesting);
 * (3) REFUSE forward mode loudly by name (the ratified refuse-unless-jvpFn
 * policy — auto-differentiating `primal_body` would silently disagree with a
 * deliberately divergent user adjoint); and (4) enforce the VJP shape
 * contract: concrete mismatches fail AT TRANSFORM TIME, sentinel-bearing
 * contributions wrap in the CHECK_SHAPE_LIKE runtime assert whose interpreter
 * arm fails loudly.
 */
class CustomVjpCoarsenedTest {

    private val f32s = DxirType(F32, emptyList())

    /** primal: x → x·x. */
    private fun squarePrimal(): DxirFunction = DxirBuilder.function("customVjp_primal") {
        val x = param("x", f32s)
        listOf(op(OpKind.MUL, listOf(x, x), f32s))
    }

    /** NON-mathematical user adjoint: (upstream, x) → 3·upstream — composition
     * of the primal would give 2x·upstream, so only the user body yields 3. */
    private fun threeTimesUpstreamVjp(): DxirFunction = DxirBuilder.function("customVjp_vjp") {
        val up = param("upstream", f32s)
        param("x", f32s)
        val three = const(3f, f32s)
        listOf(op(OpKind.MUL, listOf(up, three), f32s))
    }

    @Test
    fun userGradientBodyIsWhatRunsInReverseMode() {
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = threeTimesUpstreamVjp(),
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                ),
            )
        }
        val grad = DxirReverseTransform.apply(outer)
        for (x in listOf(-2f, 0.5f, 5f)) {
            val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))
            // The USER's 3, never the composition's 2x.
            assertEquals(3f, out[0][0], "user adjoint must run verbatim at x=$x")
        }
    }

    @Test
    fun reverseOverCustomDifferentiatesTheSplicedUserBody() {
        // vjpFn = upstream·x·3 → g(x) = 3x (seed 1); the SECOND reverse pass
        // differentiates the spliced user ops themselves → g'(x) = 3.
        val vjp = DxirBuilder.function("customVjp_vjp") {
            val up = param("upstream", f32s)
            val x = param("x", f32s)
            val three = const(3f, f32s)
            listOf(op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(up, x), f32s), three), f32s))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = vjp,
                    readsPrimalIndices = setOf(0),
                    userGradient = true,
                ),
            )
        }
        val g = DxirReverseTransform.apply(outer)
        assertEquals(15f, DxirInterpreter.evalFunction(g, listOf(floatArrayOf(5f)))[0][0], "g(5) = 3·5")
        val gg = DxirReverseTransform.apply(g)
        assertEquals(3f, DxirInterpreter.evalFunction(gg, listOf(floatArrayOf(5f)))[0][0], "g'(x) = 3")
    }

    @Test
    fun forwardModeRefusesUserGradientLoudlyByName() {
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = threeTimesUpstreamVjp(),
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                ),
            )
        }
        val ex = assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(outer) }
        assertTrue(
            "user_gradient" in ex.message.orEmpty(),
            "the forward refusal must name the user_gradient attr; got: ${ex.message}",
        )
    }

    @Test
    fun machineCoarsenedForwardStillWorksUnchanged() {
        // The refusal keys on the ATTR, not on COARSENED itself: a
        // machine-derived node (no user_gradient) keeps its §0.4.403 tangent.
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32s)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = squarePrimal(),
                    gradientBody = threeTimesUpstreamVjp(),
                    readsPrimalIndices = emptySet(),
                ),
            )
        }
        val jvp = DxirForwardTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(3f), floatArrayOf(1f)))
        assertEquals(9f, out[0][0], "primal value")
        assertTrue(abs(out[1][0] - 6f) < 1e-5f, "auto-tangent of the primal: 2·3·1")
    }

    @Test
    fun concreteShapeContractViolationFailsAtTransformTime() {
        val v3 = DxirType(F32, listOf(3))
        val primal = DxirBuilder.function("customVjp_primal") {
            val x = param("x", v3)
            listOf(op(OpKind.SUM, listOf(x), f32s))
        }
        // d_x deliberately shaped [2] for an operand of shape [3].
        val badVjp = DxirBuilder.function("customVjp_vjp") {
            param("upstream", f32s)
            param("x", v3)
            listOf(const(floatArrayOf(1f, 2f), DxirType(F32, listOf(2))))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", v3)
            listOf(
                coarsened(
                    operands = listOf(x),
                    primalBody = primal,
                    gradientBody = badVjp,
                    readsPrimalIndices = emptySet(),
                    userGradient = true,
                ),
            )
        }
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(outer) }
        assertTrue(
            "violates the VJP shape contract" in ex.message.orEmpty(),
            "expected the loud shape-contract failure; got: ${ex.message}",
        )
    }

    @Test
    fun sentinelContributionsWrapInCheckShapeLike() {
        // Under `grad {}`-style -1 sentinel dims the contract is undecidable
        // at transform time, so the splice must emit the runtime assert.
        val sv = DxirType(F32, listOf(-1))
        val primal = DxirBuilder.function("customVjp_primal") {
            val x = param("x", sv)
            listOf(op(OpKind.MUL, listOf(x, x), sv))
        }
        val vjp = DxirBuilder.function("customVjp_vjp") {
            val up = param("upstream", sv)
            val x = param("x", sv)
            listOf(op(OpKind.MUL, listOf(up, x), sv))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", sv)
            val c = coarsened(
                operands = listOf(x),
                primalBody = primal,
                gradientBody = vjp,
                readsPrimalIndices = setOf(0),
                userGradient = true,
            )
            listOf(op(OpKind.SUM, listOf(c), f32s))
        }
        val grad = DxirReverseTransform.apply(outer)
        assertTrue(
            grad.body.any { it is DxirOp && it.op == OpKind.CHECK_SHAPE_LIKE },
            "sentinel-dims user contribution must be wrapped in CHECK_SHAPE_LIKE:\n" +
                grad.body.filterIsInstance<DxirOp>().map { it.op }.toString(),
        )
    }

    @Test
    fun machineCoarsenedSpliceStaysUnwrapped() {
        // No user_gradient attr → the machine contract holds by construction
        // and the splice must stay byte-identical: no CHECK_SHAPE_LIKE.
        val sv = DxirType(F32, listOf(-1))
        val primal = DxirBuilder.function("p") {
            val x = param("x", sv)
            listOf(op(OpKind.MUL, listOf(x, x), sv))
        }
        val vjp = DxirBuilder.function("g") {
            val up = param("upstream", sv)
            val x = param("x", sv)
            listOf(op(OpKind.MUL, listOf(up, x), sv))
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", sv)
            val c = coarsened(listOf(x), primal, vjp, setOf(0))
            listOf(op(OpKind.SUM, listOf(c), f32s))
        }
        val grad = DxirReverseTransform.apply(outer)
        assertTrue(
            grad.body.none { it is DxirOp && it.op == OpKind.CHECK_SHAPE_LIKE },
            "machine gradient bodies must not pay the runtime assert",
        )
    }

    @Test
    fun checkShapeLikeInterpreterAssertsAndPassesThrough() {
        val pass = DxirBuilder.function("pass") {
            val v = param("v", DxirType(F32, listOf(3)))
            val t = param("t", DxirType(F32, listOf(3)))
            listOf(op(OpKind.CHECK_SHAPE_LIKE, listOf(v, t), DxirType(F32, listOf(3))))
        }
        val out = DxirInterpreter.evalFunction(
            pass,
            listOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(9f, 9f, 9f)),
        )
        assertEquals(listOf(1f, 2f, 3f), out[0].toList(), "matched shapes pass the value through")

        val fail = DxirBuilder.function("fail") {
            val v = param("v", DxirType(F32, listOf(2)))
            val t = param("t", DxirType(F32, listOf(3)))
            listOf(op(OpKind.CHECK_SHAPE_LIKE, listOf(v, t), DxirType(F32, listOf(3))))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fail, listOf(floatArrayOf(1f, 2f), floatArrayOf(9f, 9f, 9f)))
        }
        assertTrue(
            "CHECK_SHAPE_LIKE failed" in ex.message.orEmpty() &&
                "violates the VJP shape contract" in ex.message.orEmpty(),
            "expected the loud runtime shape assert; got: ${ex.message}",
        )
    }

    @Test
    fun checkShapeLikeComposesUnderBothTransforms() {
        // The assert is a value-identity: reverse passes the upstream through
        // (CheckShapeLikeRule) and forward checks the tangent against the same
        // template (the hessian = forward-over-reverse composition path).
        val v3 = DxirType(F32, listOf(3))
        val fn = DxirBuilder.function("f") {
            val v = param("v", v3)
            val t = param("t", v3)
            val checked = op(OpKind.CHECK_SHAPE_LIKE, listOf(v, t), v3)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(checked, checked), v3)), f32s))
        }
        // reverse: d/dv Σ c² = 2v (the check is identity), d/dt = 0.
        val grad = DxirReverseTransform.apply(fn)
        val g = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1f, -2f, 3f), floatArrayOf(0f, 0f, 0f)),
        )
        assertEquals(listOf(2f, -4f, 6f), g[0].toList(), "dv through the identity check")
        // forward: dy = Σ 2·v·dv.
        val jvp = DxirForwardTransform.apply(fn)
        val out = DxirInterpreter.evalFunction(
            jvp,
            listOf(
                floatArrayOf(1f, -2f, 3f), floatArrayOf(0f, 0f, 0f),
                floatArrayOf(1f, 1f, 1f), floatArrayOf(0f, 0f, 0f),
            ),
        )
        val want = 2f * (1f - 2f + 3f)
        assertTrue(abs(out[1][0] - want) < 1e-5f, "tangent through the check: got ${out[1][0]}, want $want")
    }
}
