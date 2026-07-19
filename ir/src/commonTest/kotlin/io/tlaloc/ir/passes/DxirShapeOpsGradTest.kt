package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.367 — Phase A2a (DiffKT parity): the RESHAPE family + permutation
 * TRANSPOSE differentiate through both transforms at IR level. TransposeRule
 * (inverse-perm) and ReshapeRule predate this §; what's new here is pinning
 * the general rank-3 permutation round-trip and the flatten (rank-collapsing
 * reshape) gradient — the shapes the §0.4.367 user surface emits.
 */
class DxirShapeOpsGradTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun transposePermRank3GradientInvertsPermutation() {
        // loss = Σ transpose(x, [2,0,1]) ⊙ w  ⇒  dx = transpose(w, inv([2,0,1])).
        // x: [2,3,4] → xT: [4,2,3]; w: [4,2,3]. dx[i,j,k] = w[k,i,j].
        val fn = DxirBuilder.function("perm3_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3, 4)))
            val w = param("w", DxirType(F32, listOf(4, 2, 3)))
            val xT = op(
                OpKind.TRANSPOSE, listOf(x), DxirType(F32, listOf(4, 2, 3)),
                attrs = mapOf("permutation" to listOf(2, 0, 1)),
            )
            val p = op(OpKind.MUL, listOf(xT, w), DxirType(F32, listOf(4, 2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = FloatArray(24) { it.toFloat() }
        val w = FloatArray(24) { (it * 0.5f) - 3f }
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        // dx[i,j,k] = w[k,i,j]  (w strides: [6, 3, 1] on shape [4,2,3]).
        for (i in 0 until 2) for (j in 0 until 3) for (k in 0 until 4) {
            val got = out[0][i * 12 + j * 4 + k]
            val want = w[k * 6 + i * 3 + j]
            assertTrue(abs(got - want) < 1e-6f, "dx[$i,$j,$k] = $got, want $want")
        }
        // dw = xT: dw[k,i,j] = x[i,j,k].
        for (k in 0 until 4) for (i in 0 until 2) for (j in 0 until 3) {
            val got = out[1][k * 6 + i * 3 + j]
            val want = x[i * 12 + j * 4 + k]
            assertTrue(abs(got - want) < 1e-6f, "dw[$k,$i,$j] = $got, want $want")
        }
    }

    @Test
    fun flattenGradientReshapesBack() {
        // loss = Σ flatten(x) ⊙ w  ⇒  dx = reshape(w, x.shape) — the
        // rank-collapsing RESHAPE adjoint (identity Jacobian, row-major).
        val fn = DxirBuilder.function("flatten_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(6)))
            val flat = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(6)))
            val p = op(OpKind.MUL, listOf(flat, w), DxirType(F32, listOf(6)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val w = floatArrayOf(0.5f, -1f, 2f, 0.25f, -3f, 1.5f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        assertEquals(w.toList(), out[0].toList(), "dx = w relaid out at x's shape")
        assertEquals(x.toList(), out[1].toList(), "dw = flatten(x)")
    }

    @Test
    fun shapeChainJvpVjpCrossIdentity() {
        // f(x, w) = Σ transpose(reshape(x, [3,2]), [1,0]) ⊙ w — a reshape →
        // transpose chain; ⟨∇f, v⟩ must equal the forward tangent.
        val fn = DxirBuilder.function("shape_chain") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(2, 3)))
            val r = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(3, 2)))
            val t = op(
                OpKind.TRANSPOSE, listOf(r), DxirType(F32, listOf(2, 3)),
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            val p = op(OpKind.MUL, listOf(t, w), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vw = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.19f, -0.07f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 6) dot += grads[0][i].toDouble() * vx[i] + grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through reshape∘transpose: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
