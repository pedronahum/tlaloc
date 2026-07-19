package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.363 — MAXPOOL2D / AVGPOOL2D differentiability (DiffKT-gap item
 * 4, pooling half). Forward semantics pinned by hand (incl. padding and
 * the count_include_pad averaging convention); gradients pinned by an
 * analytic maxpool routing pin, central differences, and the
 * forward/reverse cross-identity — the JVP maxpool arm and the VJP
 * maxpool rule share the tie convention, so the identity is exact.
 */
class DxirPoolingTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun maxAndAvgPoolForwardHandComputed() {
        // x (1,1,4,4) = 1..16, window 2, stride 2:
        // max → [[6,8],[14,16]]; avg → [[3.5,5.5],[11.5,13.5]].
        val xT = DxirType(F32, listOf(1, 1, 4, 4))
        val yT = DxirType(F32, listOf(1, 1, 2, 2))
        val attrs = mapOf("window" to listOf(2, 2))
        val fn = DxirBuilder.function("pools") {
            val x = param("x", xT)
            listOf(
                op(OpKind.MAXPOOL2D, listOf(x), yT, attrs = attrs),
                op(OpKind.AVGPOOL2D, listOf(x), yT, attrs = attrs),
            )
        }
        val x = FloatArray(16) { (it + 1).toFloat() }
        val out = DxirInterpreter.evalFunction(fn, listOf(x))
        assertEquals(listOf(6f, 8f, 14f, 16f), out[0].toList())
        assertEquals(listOf(3.5f, 5.5f, 11.5f, 13.5f), out[1].toList())
    }

    @Test
    fun avgPoolPaddingUsesFullWindowCount() {
        // x (1,1,2,2) = [[1,2],[3,4]], window 2, stride 2, pad 1 all sides →
        // 2×2 output of corner windows; count_include_pad divides by 4:
        // [[1/4, 2/4], [3/4, 4/4]].
        val fn = DxirBuilder.function("avg_pad") {
            val x = param("x", DxirType(F32, listOf(1, 1, 2, 2)))
            listOf(
                op(
                    OpKind.AVGPOOL2D, listOf(x), DxirType(F32, listOf(1, 1, 2, 2)),
                    attrs = mapOf(
                        "window" to listOf(2, 2),
                        "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                    ),
                ),
            )
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f))).single()
        assertEquals(listOf(0.25f, 0.5f, 0.75f, 1f), out.toList())
    }

    @Test
    fun maxPoolGradientRoutesToArgmax() {
        // loss = sum(maxpool(x)): dx = 1 exactly at each window's max.
        val fn = DxirBuilder.function("maxpool_loss") {
            val x = param("x", DxirType(F32, listOf(1, 1, 4, 4)))
            val y = op(
                OpKind.MAXPOOL2D, listOf(x), DxirType(F32, listOf(1, 1, 2, 2)),
                attrs = mapOf("window" to listOf(2, 2)),
            )
            listOf(op(OpKind.SUM, listOf(y), scalar))
        }
        val x = floatArrayOf(
            1f, 6f, 2f, 3f,
            5f, 4f, 8f, 7f,
            9f, 10f, 11f, 12f,
            14f, 13f, 15f, 16f,
        )
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x)).single()
        val want = listOf(
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 0f,
            1f, 0f, 0f, 1f,
        )
        assertEquals(want, out.toList())
    }

    /** Overlapping avgpool (window 3, stride 2, pad 1) over (2,3,6,6); loss = Σ y². */
    private fun avgLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 3, 6, 6))
        val yT = DxirType(F32, listOf(2, 3, 3, 3))
        return DxirBuilder.function("avg_loss") {
            val x = param("x", xT)
            val y = op(
                OpKind.AVGPOOL2D, listOf(x), yT,
                attrs = mapOf(
                    "window" to listOf(3, 3),
                    "window_strides" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 0), listOf(1, 0)),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    /** Non-overlapping maxpool (window 2 == stride) over (2,3,6,6); loss = Σ y². */
    private fun maxLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 3, 6, 6))
        val yT = DxirType(F32, listOf(2, 3, 3, 3))
        return DxirBuilder.function("max_loss") {
            val x = param("x", xT)
            val y = op(OpKind.MAXPOOL2D, listOf(x), yT, attrs = mapOf("window" to listOf(2, 2)))
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    private fun checkOracles(fn: DxirFunction, seed: Int) {
        val rng = kotlin.random.Random(seed)
        // Well-separated distinct values (a shuffled 0.07-grid): maxpool is
        // piecewise-linear, and central differences straddles a kink whenever
        // a window's top-two entries lie within ε of each other — separation
        // ≥ 0.07 » ε·max|v| keeps the FD oracle on one linear piece.
        val n = 2 * 3 * 6 * 6
        val perm = (0 until n).shuffled(rng)
        val x = FloatArray(n) { perm[it] * 0.07f - 0.035f * n }
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x))
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]

        // Oracle 1: forward mode (analytic, tight).
        val tangent = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, vx),
        )[1].single()
        assertTrue(
            abs(dot - tangent) <= 1e-4 * maxOf(1.0, abs(dot)),
            "${fn.name}: reverse and forward modes disagree: ⟨∇f,v⟩=$dot vs jvp=$tangent",
        )

        // Oracle 2: central differences on the primal (no AD involved).
        val eps = 1e-2f
        fun shifted(s: Float): Float {
            val xS = FloatArray(x.size) { x[it] + s * vx[it] }
            return DxirInterpreter.evalFunction(fn, listOf(xS)).single().single()
        }
        val fd = (shifted(eps) - shifted(-eps)).toDouble() / (2 * eps)
        assertTrue(
            abs(dot - fd) <= 1e-2 * maxOf(1.0, abs(fd)),
            "${fn.name}: VJP disagrees with central differences: ⟨∇f,v⟩=$dot vs fd=$fd",
        )
    }

    @Test
    fun avgPoolGradientMatchesBothOracles() = checkOracles(avgLossFn(), seed = 42)

    @Test
    fun maxPoolGradientMatchesBothOracles() = checkOracles(maxLossFn(), seed = 43)
}
