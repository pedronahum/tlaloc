package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * §0.4.419 — Phase E1c-pre: the PARAM-ADDRESSED structural zero
 * (`OpKind.ZEROS_LIKE`). §0.4.400's anonymous integer-zero const could not say
 * which param it zeroed under -1 sentinel dims, so `grad {}` admitted exactly
 * one integer param per lambda — and a CSR sparse operand carries two (colIdx,
 * rowPtr). The reverse transform now emits the zero as ZEROS_LIKE on the
 * cloned param ITSELF, so any number of integer params is unambiguous by
 * construction. This suite pins the addressing at IR level:
 *
 * 1. **The headline pin** — a body with TWO integer params of DIFFERENT sizes:
 *    each ZEROS_LIKE return references its own cloned param (node identity),
 *    and the interpreter materialises each zero at ITS param's extent — the
 *    disambiguation an anonymous const structurally cannot express.
 * 2. **Forward-over-reverse composes through it** — the tangent arm (d/dx 0 =
 *    0, another ZEROS_LIKE on the same template) lets jvp-of-grad transform
 *    and run: the hessian path through gradient bodies containing the op.
 * 3. **The VjpRule is closed under itself** — the template's contribution is
 *    ZEROS_LIKE(template), never an anonymous const, so reverse composes
 *    through gradient bodies to any order without reintroducing the ambiguity
 *    the op exists to remove.
 */
class DxirZerosLikeTest {

    private val scalar = DxirType(F32, emptyList())

    /** loss = Σ embedding(table, idxA) + Σ embedding(table, idxB), N=4 ≠ M=3. */
    private fun twoIndexParamFn() = DxirBuilder.function("two_idx_loss") {
        val table = param("table", DxirType(F32, listOf(3, 2)))
        val idxA = param("idxA", DxirType(I32, listOf(4)))
        val idxB = param("idxB", DxirType(I32, listOf(3)))
        val embA = op(OpKind.EMBEDDING, listOf(table, idxA), DxirType(F32, listOf(4, 2)))
        val embB = op(OpKind.EMBEDDING, listOf(table, idxB), DxirType(F32, listOf(3, 2)))
        val sA = op(OpKind.SUM, listOf(embA), scalar)
        val sB = op(OpKind.SUM, listOf(embB), scalar)
        listOf(op(OpKind.ADD, listOf(sA, sB), scalar))
    }

    @Test
    fun eachIntegerParamGetsItsOwnParamAddressedZero() {
        val grad = DxirReverseTransform.apply(twoIndexParamFn())

        // Structure: returns = (d_table, d_idxA, d_idxB); each integer return
        // is a ZEROS_LIKE whose single operand IS that param's clone — the
        // addressing an anonymous const cannot carry.
        assertEquals(3, grad.returns.size, "grad must return one gradient per param")
        val dIdxA = grad.returns[1]
        val dIdxB = grad.returns[2]
        assertTrue(dIdxA is DxirOp && dIdxA.op == OpKind.ZEROS_LIKE, "d_idxA must be ZEROS_LIKE, got $dIdxA")
        assertTrue(dIdxB is DxirOp && dIdxB.op == OpKind.ZEROS_LIKE, "d_idxB must be ZEROS_LIKE, got $dIdxB")
        assertSame(grad.params[1], (dIdxA as DxirOp).operands[0], "d_idxA must address the idxA param")
        assertSame(grad.params[2], (dIdxB as DxirOp).operands[0], "d_idxB must address the idxB param")

        // Interpretation: idx = [0, 2, 0, 1] and [1, 1, 2] over vocab 3 →
        // dTable counts = A(2,1,1) + B(0,2,1) = [2, 3, 2] per row, and the two
        // integer zeros land at THEIR OWN extents (4 vs 3 — the sizes that
        // would collapse into ambiguity without addressing).
        val table = FloatArray(6) { it.toFloat() }
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(table, floatArrayOf(0f, 2f, 0f, 1f), floatArrayOf(1f, 1f, 2f)),
        )
        val wantCounts = floatArrayOf(2f, 3f, 2f)
        for (v in 0 until 3) {
            for (d in 0 until 2) {
                assertTrue(
                    abs(out[0][v * 2 + d] - wantCounts[v]) < 1e-6f,
                    "dTable[$v,$d] = ${out[0][v * 2 + d]}, want ${wantCounts[v]}",
                )
            }
        }
        assertEquals(4, out[1].size, "d_idxA must have idxA's extent")
        assertEquals(3, out[2].size, "d_idxB must have idxB's extent")
        assertTrue(out[1].all { it == 0f } && out[2].all { it == 0f }, "structural zeros must be exact")
    }

    @Test
    fun forwardOverReverseComposesThroughZerosLike() {
        // jvp(grad): the fwd-over-rev hessian path THROUGH a gradient body
        // containing ZEROS_LIKE ops — the transform must not reject the op
        // (the tangent arm exists) and the structural zeros' tangents must be
        // zeros at their params' own extents. The loss here is x²-shaped in
        // the FLOAT param with the integer params riding as pass-throughs, so
        // the pin isolates ZEROS_LIKE itself (embedding-containing hessians
        // are §0.4.423's DxirFusedAdjointTangentTest; the ZEROS_LIKE returns
        // appear for the integer params regardless, which is what this pins).
        val fn = DxirBuilder.function("quad_loss") {
            val x = param("x", DxirType(F32, listOf(3)))
            val idxA = param("idxA", DxirType(I32, listOf(4)))
            val idxB = param("idxB", DxirType(I32, listOf(3)))
            // Keep the integer params ALIVE without differentiable use — the
            // param-list contract (one gradient per param) is what forces the
            // ZEROS_LIKE returns; idxA/idxB need no body consumer for that.
            require(idxA.type.dtype == I32 && idxB.type.dtype == I32)
            val sq = op(OpKind.MUL, listOf(x, x), DxirType(F32, listOf(3)))
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        assertTrue(
            grad.returns.drop(1).all { it is DxirOp && (it as DxirOp).op == OpKind.ZEROS_LIKE },
            "both integer params must return ZEROS_LIKE",
        )
        val jvpOfGrad = DxirForwardTransform.apply(grad)
        val x = floatArrayOf(1f, 2f, 3f)
        val dx = floatArrayOf(1f, 1f, 1f)
        val idxA = floatArrayOf(0f, 2f, 0f, 1f)
        val idxB = floatArrayOf(1f, 1f, 2f)
        val out = DxirInterpreter.evalFunction(
            jvpOfGrad,
            listOf(x, idxA, idxB, dx, FloatArray(4), FloatArray(3)),
        )
        // Layout: primals then tangents. d_x = 2x; its tangent (the HVP
        // against dx=1) is 2·dx = [2,2,2]; the integer zeros' tangents are
        // zeros at their OWN extents (4 vs 3).
        assertEquals(6, out.size, "jvp(grad) returns 3 primals + 3 tangents")
        for (i in 0 until 3) {
            assertTrue(abs(out[0][i] - 2f * x[i]) < 1e-6f, "d_x[$i] = ${out[0][i]}, want ${2f * x[i]}")
            assertTrue(abs(out[3][i] - 2f) < 1e-6f, "HVP[$i] = ${out[3][i]}, want 2.0")
        }
        assertEquals(4, out[4].size, "tangent of d_idxA keeps idxA's extent")
        assertEquals(3, out[5].size, "tangent of d_idxB keeps idxB's extent")
        assertTrue(out[4].all { it == 0f } && out[5].all { it == 0f }, "tangents of structural zeros are zero")
    }

    @Test
    fun zerosLikeRuleContributesZerosLikeToItsOwnTemplate() {
        // The rule's contribution is ZEROS_LIKE(template) — param-addressed
        // like the op itself, never an anonymous const — so reverse mode
        // composes through gradient bodies containing it to any order.
        val fn = DxirBuilder.function("zl") {
            val idx = param("idx", DxirType(I32, listOf(5)))
            val zl = op(OpKind.ZEROS_LIKE, listOf(idx), DxirType(I32, listOf(5)))
            listOf(zl)
        }
        val zlOp = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.ZEROS_LIKE }
        val rule = VjpRegistry[OpKind.ZEROS_LIKE]
            ?: error("ZEROS_LIKE must be registered in VjpRegistry")
        val contributions = DxirBuilder.function("probe") {
            val idx = param("u", DxirType(I32, listOf(5)))
            val built = rule.apply(zlOp, idx, this)
            assertEquals(1, built.size, "one contribution, to the template")
            val (target, contribution) = built.single()
            assertSame(zlOp.operands[0], target, "the contribution targets the template")
            assertTrue(
                contribution is DxirOp && contribution.op == OpKind.ZEROS_LIKE,
                "the contribution is itself param-addressed, got $contribution",
            )
            assertSame(
                zlOp.operands[0],
                (contribution as DxirOp).operands[0],
                "the contribution addresses the SAME template",
            )
            listOf(idx)
        }
        assertTrue(contributions.params.isNotEmpty()) // silence unused-result; assertions above are the test
    }

    @Test
    fun interpreterZerosLikeNeverReadsTheTemplatesValues() {
        // Template contributes SHAPE ONLY: zeros come out whatever the
        // template holds, dims read off the operand's type.
        val fn = DxirBuilder.function("zl_eval") {
            val idx = param("idx", DxirType(I32, listOf(2, 3)))
            listOf(op(OpKind.ZEROS_LIKE, listOf(idx), DxirType(I32, listOf(2, 3))))
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(7f, -1f, 3f, 9f, 0f, 5f))).single()
        assertEquals(6, out.size, "rank-2 template's element count")
        assertTrue(out.all { it == 0f }, "all zeros regardless of template values, got ${out.toList()}")
    }
}
