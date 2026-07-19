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
 * §0.4.366 — Phase A1 (DiffKT parity): axis-wise reductions with SQUEEZED
 * result shapes (DiffKT's `keepDims = false` default) differentiate through
 * both transforms, and MEAN gains its interpreter arm (it had none — an
 * audit-flagged orphan). The §0.4.359 rules covered full-reduce and
 * keepdims shapes only; the squeezed arm inserts a RESHAPE to the keepdims
 * spelling before the stretch BROADCAST (see `VjpRegistry.reshapeToKeepdims`).
 *
 * Oracles: hand-computed pins + the JVP⇄VJP dot-product cross-identity
 * (`⟨∇f(x), v⟩ == jvp_f(x, v).tangent` — each transform certifies the other).
 */
class DxirAxisReductionTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun meanInterpreterFullAxisAndKeepdims() {
        // Full reduce → scalar mean; axis 1 squeezed → row means; axis 0
        // keepdims → column means at [1, 3]. Divisor = reduced extents only.
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val full = DxirBuilder.function("mean_full") {
            val p = param("x", DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.MEAN, listOf(p), scalar))
        }
        assertEquals(3.5f, DxirInterpreter.evalFunction(full, listOf(x)).single().single())

        val rows = DxirBuilder.function("mean_rows") {
            val p = param("x", DxirType(F32, listOf(2, 3)))
            listOf(
                op(
                    OpKind.MEAN, listOf(p), DxirType(F32, listOf(2)),
                    attrs = mapOf("reduction_dims" to listOf(1)),
                ),
            )
        }
        assertEquals(listOf(2f, 5f), DxirInterpreter.evalFunction(rows, listOf(x)).single().toList())

        val cols = DxirBuilder.function("mean_cols_keep") {
            val p = param("x", DxirType(F32, listOf(2, 3)))
            listOf(
                op(
                    OpKind.MEAN, listOf(p), DxirType(F32, listOf(1, 3)),
                    attrs = mapOf("reduction_dims" to listOf(0)),
                ),
            )
        }
        assertEquals(
            listOf(2.5f, 3.5f, 4.5f),
            DxirInterpreter.evalFunction(cols, listOf(x)).single().toList(),
        )
    }

    @Test
    fun sumAxisSqueezedGradientIsWeightMatrix() {
        // f(x, w) = Σ_j (Σ_i x[i,j]) · w[j] with the column sum SQUEEZED to
        // rank-1: dx[i,j] = w[j] (the un-reduce must stretch the rank-1
        // upstream back over axis 0), dw[j] = Σ_i x[i,j].
        val fn = DxirBuilder.function("sum_axis_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(3)))
            val s = op(
                OpKind.SUM, listOf(x), DxirType(F32, listOf(3)),
                attrs = mapOf("reduction_dims" to listOf(0)),
            )
            val p = op(OpKind.MUL, listOf(s, w), DxirType(F32, listOf(3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val w = floatArrayOf(0.5f, -1f, 2f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        assertEquals(
            listOf(0.5f, -1f, 2f, 0.5f, -1f, 2f),
            out[0].toList(),
            "dx[i,j] = w[j] stretched over the reduced axis",
        )
        assertEquals(listOf(5f, 7f, 9f), out[1].toList(), "dw = column sums of x")
    }

    @Test
    fun meanAxisGradientSqueezedAndKeepdims() {
        // f(x) = Σ mean(x, axis=1): dx = 1/3 everywhere — identical whether
        // the primal squeezed the axis or kept it at size 1.
        for (keep in listOf(false, true)) {
            val outType = if (keep) DxirType(F32, listOf(2, 1)) else DxirType(F32, listOf(2))
            val fn = DxirBuilder.function("mean_axis_loss") {
                val x = param("x", DxirType(F32, listOf(2, 3)))
                val m = op(
                    OpKind.MEAN, listOf(x), outType,
                    attrs = mapOf("reduction_dims" to listOf(1)),
                )
                listOf(op(OpKind.SUM, listOf(m), scalar))
            }
            val grad = DxirReverseTransform.apply(fn)
            val out = DxirInterpreter.evalFunction(
                grad, listOf(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
            )
            for (v in out.single()) {
                assertTrue(abs(v - 1f / 3f) < 1e-6f, "d mean/dx = 1/3 (keep=$keep), got $v")
            }
        }
    }

    @Test
    fun maxAxisSqueezedGradientRoutesToMaxima() {
        // The §0.4.359 twin pinned the KEEPDIMS shape; this pins the squeezed
        // one (the new RESHAPE-before-stretch arm). Same data, same routing:
        // row 1 ties at 5 twice — full upstream to every tie.
        val fn = DxirBuilder.function("max_axis_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val m = op(
                OpKind.MAX, listOf(x), DxirType(F32, listOf(2)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val out = DxirInterpreter.evalFunction(
            grad, listOf(floatArrayOf(1f, 3f, 2f, 5f, 4f, 5f)),
        )
        assertEquals(listOf(0f, 1f, 0f, 1f, 0f, 1f), out.single().toList())
    }

    @Test
    fun minAxisSqueezedForwardRoutesTangent() {
        // Forward-mode twin at the squeezed shape: the tangent of each row's
        // min is the tangent at its argmin (the transform now reshapes the
        // primal output to keepdims before the extremum-mask broadcast).
        val fn = DxirBuilder.function("min_axis_fwd") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            listOf(
                op(
                    OpKind.MIN, listOf(x), DxirType(F32, listOf(2)),
                    attrs = mapOf("reduction_dims" to listOf(1)),
                ),
            )
        }
        val x = floatArrayOf(3f, 1f, 2f, 5f, 6f, 4f)
        val dx = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        val out = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, dx))
        assertEquals(listOf(1f, 4f), out[0].toList(), "primal row minima")
        assertEquals(listOf(0.2f, 0.6f), out[1].toList(), "tangent routed to each row's argmin")
    }

    @Test
    fun axisReductionJvpVjpCrossIdentity() {
        // f(x, w) = Σ_i mean_j(x[i,·]) · w[i] (squeezed axis mean feeding a
        // rank-1 product). ⟨∇f, (vx, vw)⟩ must equal the forward transform's
        // tangent — the two AD modes certify each other with no shared code
        // path through the axis plumbing.
        val fn = DxirBuilder.function("mean_axis_cross") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(2)))
            val m = op(
                OpKind.MEAN, listOf(x), DxirType(F32, listOf(2)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            val p = op(OpKind.MUL, listOf(m, w), DxirType(F32, listOf(2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vw = floatArrayOf(-0.29f, 0.31f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in x.indices) dot += grads[0][i].toDouble() * vx[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken: ⟨grad, v⟩=$dot vs tangent=$tangent",
        )
    }
}
