package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * §0.4.396 — REVERSE (DiffKT `flip`, Phase C3) at IR level. Interpreter pins
 * cover a single trailing axis, both axes, and a LEADING axis on a
 * non-square shape (where the copy is not one contiguous run — the stride
 * walk must invert per-axis, not merely reverse the flat array). Gradient
 * oracle: `∇_a Σ flip(a) ⊙ b = flip(b)` — REVERSE is self-adjoint, and the
 * non-uniform upstream `b` makes a dropped or mis-indexed adjoint fail
 * loudly. The JVP⇄VJP cross-identity runs through `Σ tanh(flip(a) ⊙ b)` so
 * the forward tangent's linear REVERSE arm is exercised inside a nonlinear
 * chain rather than cancelling out.
 */
class DxirFlipGradTest {

    private val scalar = DxirType(F32, emptyList())
    private val r23 = DxirType(F32, listOf(2, 3))

    private fun flipFn(axes: List<Int>, type: DxirType) = DxirBuilder.function("flip") {
        val x = param("x", type)
        listOf(op(OpKind.REVERSE, listOf(x), type, attrs = mapOf("dimensions" to axes)))
    }

    @Test
    fun interpreterFlipsTrailingAxis() {
        // [[1,2,3],[4,5,6]] flipped along axis 1 → [[3,2,1],[6,5,4]].
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val out = DxirInterpreter.evalFunction(flipFn(listOf(1), r23), listOf(x)).single()
        assertContentEquals(floatArrayOf(3f, 2f, 1f, 6f, 5f, 4f), out)
    }

    @Test
    fun interpreterFlipsLeadingAxisNonContiguously() {
        // Axis 0 on [2,3]: whole ROWS swap — [[4,5,6],[1,2,3]] — which a flat
        // element reversal would get wrong.
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val out = DxirInterpreter.evalFunction(flipFn(listOf(0), r23), listOf(x)).single()
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 1f, 2f, 3f), out)
    }

    @Test
    fun interpreterFlipsBothAxes() {
        // Both axes = full element reversal on rank-2.
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val out = DxirInterpreter.evalFunction(flipFn(listOf(0, 1), r23), listOf(x)).single()
        assertContentEquals(floatArrayOf(6f, 5f, 4f, 3f, 2f, 1f), out)
        // Involution: flipping twice is the identity.
        val fn2 = DxirBuilder.function("flip2") {
            val p = param("x", r23)
            val once = op(OpKind.REVERSE, listOf(p), r23, attrs = mapOf("dimensions" to listOf(0, 1)))
            listOf(op(OpKind.REVERSE, listOf(once), r23, attrs = mapOf("dimensions" to listOf(0, 1))))
        }
        assertContentEquals(x, DxirInterpreter.evalFunction(fn2, listOf(x)).single())
    }

    @Test
    fun flipGradientIsTheFlippedUpstream() {
        // loss = Σ flip(a, [0]) ⊙ b  ⇒  dA = flip(b, [0]), dB = flip(a, [0]).
        val fn = DxirBuilder.function("flip_loss") {
            val a = param("a", r23)
            val b = param("b", r23)
            val f = op(OpKind.REVERSE, listOf(a), r23, attrs = mapOf("dimensions" to listOf(0)))
            val p = op(OpKind.MUL, listOf(f, b), r23)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val b = floatArrayOf(0.5f, -1f, 2f, 0.25f, 3f, -0.75f)
        val out = DxirInterpreter.evalFunction(grad, listOf(a, b))
        // flip(b, [0]) = rows of b swapped.
        assertContentEquals(floatArrayOf(0.25f, 3f, -0.75f, 0.5f, -1f, 2f), out[0])
        // flip(a, [0]) = rows of a swapped.
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 1f, 2f, 3f), out[1])
    }

    @Test
    fun flipJvpVjpCrossIdentity() {
        // f(a, b) = Σ tanh(flip(a, [0, 1]) ⊙ b): ⟨∇f, v⟩ == forward tangent.
        val fn = DxirBuilder.function("flip_chain") {
            val a = param("a", r23)
            val b = param("b", r23)
            val f = op(OpKind.REVERSE, listOf(a), r23, attrs = mapOf("dimensions" to listOf(0, 1)))
            val p = op(OpKind.MUL, listOf(f, b), r23)
            val t = op(OpKind.TANH, listOf(p), r23)
            listOf(op(OpKind.SUM, listOf(t), scalar))
        }
        val a = floatArrayOf(0.3f, -1.2f, 0.9f, -0.4f, 0.7f, -0.2f)
        val b = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 0.4f, 1.1f)
        val va = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.19f, -0.07f)
        val vb = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.23f, 0.05f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, b))
        var dot = 0.0
        for (i in 0 until 6) dot += grads[0][i].toDouble() * va[i] + grads[1][i].toDouble() * vb[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(a, b, va, vb),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through flip: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
