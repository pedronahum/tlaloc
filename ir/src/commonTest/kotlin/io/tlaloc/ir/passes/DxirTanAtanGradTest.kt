package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.tan
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.395 — Phase C2 trig tails (DiffKT parity): TAN and ATAN differentiate
 * through both transforms at IR level. Oracles: hand-computed analytic pins
 * with a NON-uniform upstream (`Σ f(x) ⊙ w`, so a dropped upstream factor
 * cannot pass) + the JVP⇄VJP dot-product cross-identity through a body that
 * chains both ops. Probe points stay well inside (−π/2, π/2) — tan's poles
 * are IEEE-finite-but-huge in Double, and a pin next to one would be a test
 * of rounding, not of the adjoint.
 */
class DxirTanAtanGradTest {

    private val scalar = DxirType(F32, emptyList())
    private val r1 = DxirType(F32, listOf(4))

    @Test
    fun tanGradientMatchesAnalytic() {
        // loss = Σ tan(x) ⊙ w  ⇒  dx = w ⊙ (1 + tan²(x)), dw = tan(x).
        val fn = DxirBuilder.function("tan_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            val t = op(OpKind.TAN, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(t, w), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.3f, -0.7f, 1.1f, 0.0f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        for (i in 0 until 4) {
            val t = tan(x[i].toDouble())
            val wantDx = (w[i] * (1.0 + t * t)).toFloat()
            assertTrue(abs(out[0][i] - wantDx) < 1e-4f, "dx[$i] = ${out[0][i]}, want $wantDx")
            assertTrue(abs(out[1][i] - t.toFloat()) < 1e-5f, "dw[$i] = ${out[1][i]}, want ${t.toFloat()}")
        }
    }

    @Test
    fun atanGradientMatchesAnalytic() {
        // loss = Σ atan(x) ⊙ w  ⇒  dx = w / (1 + x²), dw = atan(x). Probes
        // include |x| > π/2 — atan is total, unlike tan.
        val fn = DxirBuilder.function("atan_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            val a = op(OpKind.ATAN, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(a, w), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.5f, -3.0f, 10.0f, 0.0f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        for (i in 0 until 4) {
            val wantDx = (w[i] / (1.0 + x[i].toDouble() * x[i])).toFloat()
            val wantDw = atan(x[i].toDouble()).toFloat()
            assertTrue(abs(out[0][i] - wantDx) < 1e-4f, "dx[$i] = ${out[0][i]}, want $wantDx")
            assertTrue(abs(out[1][i] - wantDw) < 1e-5f, "dw[$i] = ${out[1][i]}, want $wantDw")
        }
    }

    @Test
    fun tanAtanJvpVjpCrossIdentity() {
        // f(x, w) = Σ atan(tan(x) ⊙ w) — both new rules chained in one body;
        // ⟨∇f, v⟩ must equal the forward tangent.
        val fn = DxirBuilder.function("tan_atan_chain") {
            val x = param("x", r1)
            val w = param("w", r1)
            val t = op(OpKind.TAN, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(t, w), r1)
            val a = op(OpKind.ATAN, listOf(p), r1)
            listOf(op(OpKind.SUM, listOf(a), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 0.9f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val vw = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 4) dot += grads[0][i].toDouble() * vx[i] + grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-4,
            "JVP⇄VJP cross-identity broken through atan∘tan: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
