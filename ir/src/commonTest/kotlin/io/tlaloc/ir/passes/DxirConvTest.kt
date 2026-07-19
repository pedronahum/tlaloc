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
 * §0.4.362 — CONV2D differentiability (DiffKT-gap item 4, conv half).
 * Forward semantics pinned by hand-computed examples (incl. the
 * `window_reversal` tap flip the adjoint relies on); gradients pinned
 * by two independent oracles: central differences on the primal, and
 * the forward transform via `⟨∇f, v⟩ == jvp.tangent` — [Conv2dRule]'s
 * CONV_TRANSPOSE2D/lhs_dilation machinery never appears in either
 * oracle's own computation.
 */
class DxirConvTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun convForwardHandComputed() {
        // x = [[1..9]] (1,1,3,3), w = [[1,0],[0,2]] (1,1,2,2), valid conv:
        // out[y][x] = x[y][x] + 2·x[y+1][x+1] → [[11,14],[20,23]].
        val fn = DxirBuilder.function("conv") {
            val x = param("x", DxirType(F32, listOf(1, 1, 3, 3)))
            val w = param("w", DxirType(F32, listOf(1, 1, 2, 2)))
            listOf(op(OpKind.CONV2D, listOf(x, w), DxirType(F32, listOf(1, 1, 2, 2))))
        }
        val x = FloatArray(9) { (it + 1).toFloat() }
        val w = floatArrayOf(1f, 0f, 0f, 2f)
        val out = DxirInterpreter.evalFunction(fn, listOf(x, w)).single()
        assertEquals(listOf(11f, 14f, 20f, 23f), out.toList())
    }

    @Test
    fun convForwardWindowReversalFlipsTaps() {
        // Same tensors with window_reversal: taps flip spatially, so
        // out[y][x] = 2·x[y][x] + x[y+1][x+1] → [[7,10],[16,19]].
        val fn = DxirBuilder.function("conv_rev") {
            val x = param("x", DxirType(F32, listOf(1, 1, 3, 3)))
            val w = param("w", DxirType(F32, listOf(1, 1, 2, 2)))
            listOf(
                op(
                    OpKind.CONV2D, listOf(x, w), DxirType(F32, listOf(1, 1, 2, 2)),
                    attrs = mapOf("window_reversal" to listOf(true, true)),
                ),
            )
        }
        val x = FloatArray(9) { (it + 1).toFloat() }
        val w = floatArrayOf(1f, 0f, 0f, 2f)
        val out = DxirInterpreter.evalFunction(fn, listOf(x, w)).single()
        assertEquals(listOf(7f, 10f, 16f, 19f), out.toList())
    }

    /**
     * General-attr conv loss: N=2, Ci=2, Co=3, 5×4 input, 3×2 kernel,
     * strides [2,1], asymmetric padding [[1,0],[1,1]], rhs_dilation [1,2].
     * loss = Σ y² so the upstream is non-uniform (2y).
     */
    private fun convLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 2, 5, 4))
        val wT = DxirType(F32, listOf(3, 2, 3, 2))
        // hOut = (5+1+0−3)/2+1 = 2; kEffW = (2−1)·2+1 = 3; wOut = (4+1+1−3)/1+1 = 4.
        val yT = DxirType(F32, listOf(2, 3, 2, 4))
        return DxirBuilder.function("conv_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(
                OpKind.CONV2D, listOf(x, w), yT,
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    private fun randomInputs(): Pair<FloatArray, FloatArray> {
        val rng = kotlin.random.Random(42)
        val x = FloatArray(2 * 2 * 5 * 4) { rng.nextFloat() - 0.5f }
        val w = FloatArray(3 * 2 * 3 * 2) { rng.nextFloat() - 0.5f }
        return x to w
    }

    @Test
    fun convVjpMatchesCentralDifferences() {
        val fn = convLossFn()
        val (x, w) = randomInputs()
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))

        // Directional central differences along a fixed direction v:
        // ⟨∇f, v⟩ vs (f(x+εv) − f(x−εv)) / 2ε. Aggregating into one
        // scalar keeps the f32 oracle well-conditioned.
        val rng = kotlin.random.Random(7)
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
        val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

        val eps = 1e-2f
        fun shifted(s: Float): Float {
            val xS = FloatArray(x.size) { x[it] + s * vx[it] }
            val wS = FloatArray(w.size) { w[it] + s * vw[it] }
            return DxirInterpreter.evalFunction(fn, listOf(xS, wS)).single().single()
        }
        val fd = (shifted(eps) - shifted(-eps)).toDouble() / (2 * eps)
        assertTrue(
            abs(dot - fd) <= 1e-2 * maxOf(1.0, abs(fd)),
            "conv VJP disagrees with central differences: ⟨∇f,v⟩=$dot vs fd=$fd",
        )
    }

    @Test
    fun convVjpAgreesWithJvp() {
        // ⟨∇f(x,w), (vx,vw)⟩ == jvp_f((x,w), (vx,vw)).tangent — both sides
        // analytic; the JVP side uses only the bilinear product rule (the
        // primal op twice), never Conv2dRule's transposed machinery.
        val fn = convLossFn()
        val (x, w) = randomInputs()
        val rng = kotlin.random.Random(11)
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
        val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

        val tangent = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )[1].single()

        assertTrue(
            abs(dot - tangent) <= 1e-4 * maxOf(1.0, abs(dot)),
            "conv reverse and forward modes disagree: ⟨∇f,v⟩=$dot vs jvp=$tangent",
        )
    }

    @Test
    fun convGradientShapesMatchOperands() {
        val fn = convLossFn()
        val grad = DxirReverseTransform.apply(fn)
        assertEquals(listOf(2, 2, 5, 4), grad.returns[0].type.dims, "dX carries x's shape")
        assertEquals(listOf(3, 2, 3, 2), grad.returns[1].type.dims, "dW carries w's shape")
    }
}
