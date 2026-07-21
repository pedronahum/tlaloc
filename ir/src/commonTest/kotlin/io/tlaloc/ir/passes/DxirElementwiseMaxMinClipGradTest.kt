package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.369 — Phase A4 (DiffKT parity): elementwise `maximum(a, b)` /
 * `minimum(a, b)` / `clip(x, lo, hi)` / `outerProduct(a, b)` differentiate at
 * IR level. All four are SUGAR — no new VjpRule — so this pins that the exact
 * op compositions the K2 plugin emits produce the analytic gradients through
 * the existing WhereRule / CompareRule / MatmulRule / ReshapeRule chains.
 */
class DxirElementwiseMaxMinClipGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** maximum(a, b) = WHERE(COMPARE(a, b, GE), a, b): full upstream to the
     *  larger operand, ties (a == b) to `a` via the `>=` mask. */
    @Test
    fun maximumRoutesUpstreamToLargerOperand() {
        val t = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("maximum") {
            val a = param("a", t)
            val b = param("b", t)
            val cmp = op(OpKind.COMPARE, listOf(a, b), DxirType(Bool, t.dims), attrs = mapOf("direction" to "GE"))
            val mx = op(OpKind.WHERE, listOf(cmp, a, b), t)
            val w = param("w", t)
            val weighted = op(OpKind.MUL, listOf(w, mx), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        //                a>=b?  T     F     T(tie) F
        val a = floatArrayOf(3f, -1f,  2f,   0.5f)
        val b = floatArrayOf(1f,  4f,  2f,   2.0f)
        val w = floatArrayOf(0.5f, 1f, -2f,  3f)
        val out = DxirInterpreter.evalFunction(grad, listOf(a, b, w))
        // da = w where a>=b else 0; db = w where a<b else 0.
        val da = floatArrayOf(0.5f, 0f, -2f, 0f)
        val db = floatArrayOf(0f, 1f, 0f, 3f)
        assertClose(da, out[0], "maximum da")
        assertClose(db, out[1], "maximum db")
    }

    /** minimum(a, b) = WHERE(COMPARE(a, b, LE), a, b): upstream to the smaller
     *  operand, ties to `a`. */
    @Test
    fun minimumRoutesUpstreamToSmallerOperand() {
        val t = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("minimum") {
            val a = param("a", t)
            val b = param("b", t)
            val cmp = op(OpKind.COMPARE, listOf(a, b), DxirType(Bool, t.dims), attrs = mapOf("direction" to "LE"))
            val mn = op(OpKind.WHERE, listOf(cmp, a, b), t)
            val w = param("w", t)
            val weighted = op(OpKind.MUL, listOf(w, mn), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        //                a<=b?  F     T     T(tie) T
        val a = floatArrayOf(3f, -1f,  2f,   0.5f)
        val b = floatArrayOf(1f,  4f,  2f,   2.0f)
        val w = floatArrayOf(0.5f, 1f, -2f,  3f)
        val out = DxirInterpreter.evalFunction(grad, listOf(a, b, w))
        val da = floatArrayOf(0f, 1f, -2f, 3f)
        val db = floatArrayOf(0.5f, 0f, 0f, 0f)
        assertClose(da, out[0], "minimum da")
        assertClose(db, out[1], "minimum db")
    }

    /** clip(x, lo, hi) = minimum(maximum(x, lo), hi): gradient 1 inside
     *  [lo, hi], 0 outside — the classic hard-clip straight-through region. */
    @Test
    fun clipGradientIsOneInsideBoundsZeroOutside() {
        val lo = -1.0f
        val hi = 2.0f
        val t = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("clip") {
            val x = param("x", t)
            val loC = const(lo, t)
            val hiC = const(hi, t)
            val ge = op(OpKind.COMPARE, listOf(x, loC), DxirType(Bool, t.dims), attrs = mapOf("direction" to "GE"))
            val maxed = op(OpKind.WHERE, listOf(ge, x, loC), t)
            val le = op(OpKind.COMPARE, listOf(maxed, hiC), DxirType(Bool, t.dims), attrs = mapOf("direction" to "LE"))
            val clipped = op(OpKind.WHERE, listOf(le, maxed, hiC), t)
            val w = param("w", t)
            val weighted = op(OpKind.MUL, listOf(w, clipped), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        //                 below  lo(tie) inside hi(tie) above
        val x = floatArrayOf(-3f,  -1f,    0.5f,  2f,    5f)
        val w = floatArrayOf(1f,    2f,    3f,    4f,    5f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        // In-bounds means lo <= x <= hi. At x == lo the GE mask keeps x (route
        // through); at x == hi the LE mask keeps x. So both boundaries pass.
        val dx = floatArrayOf(0f, 2f, 3f, 4f, 0f)
        assertClose(dx, out[0], "clip dx")
    }

    /** outerProduct(a, b) = MATMUL(reshape(a, [n,1]), reshape(b, [1,m])):
     *  da_i = Σ_j upstream[i,j]·b[j]; db_j = Σ_i upstream[i,j]·a[i]. */
    @Test
    fun outerProductGradientContractsUpstream() {
        val n = 3
        val m = 2
        val av = DxirType(F32, listOf(n))
        val bv = DxirType(F32, listOf(m))
        val fn = DxirBuilder.function("outer") {
            val a = param("a", av)
            val b = param("b", bv)
            val aU = op(OpKind.RESHAPE, listOf(a), DxirType(F32, listOf(n, 1)))
            val bU = op(OpKind.RESHAPE, listOf(b), DxirType(F32, listOf(1, m)))
            val outer = op(OpKind.MATMUL, listOf(aU, bU), DxirType(F32, listOf(n, m)))
            // Weight by W then sum so the upstream into the outer product is W.
            val w = param("w", DxirType(F32, listOf(n, m)))
            val weighted = op(OpKind.MUL, listOf(w, outer), DxirType(F32, listOf(n, m)))
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(0.5f, -1f)
        val w = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f) // [n,m] row-major
        val out = DxirInterpreter.evalFunction(grad, listOf(a, b, w))
        // upstream into outer product = w. da_i = Σ_j w[i,j]·b[j];
        // db_j = Σ_i w[i,j]·a[i].
        val da = FloatArray(n)
        val db = FloatArray(m)
        for (i in 0 until n) for (j in 0 until m) {
            da[i] += w[i * m + j] * b[j]
            db[j] += w[i * m + j] * a[i]
        }
        assertClose(da, out[0], "outer da")
        assertClose(db, out[1], "outer db")
    }

    private fun assertClose(want: FloatArray, got: FloatArray, label: String) {
        assertTrue(want.size == got.size, "$label size: want ${want.size}, got ${got.size}")
        var maxAbs = 0f
        for (i in want.indices) maxAbs = maxOf(maxAbs, abs(want[i] - got[i]))
        assertTrue(
            maxAbs <= 1e-5f,
            "$label diverges: max|diff|=$maxAbs\n  want=${want.toList()}\n  got =${got.toList()}",
        )
    }
}
