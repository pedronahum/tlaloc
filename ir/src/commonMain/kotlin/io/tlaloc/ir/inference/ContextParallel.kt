package io.tlaloc.ir.inference

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirEmitter
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * Attention over a context held in slices (decode context parallelism: each device holds some
 * of a sequence's pages), merged from each slice's PAGED_ATTENTION with
 * [io.tlaloc.ir.PagedAttentionAttrs.RETURN_LSE]:
 * ```
 *   m   = max_i lse_i                 (0 where every slice is -inf)
 *   w_i = exp(lse_i - m)
 *   lse = m + log(sum_i w_i)
 *   out = sum_i (w_i / sum_i w_i) out_i   (0 for a row no slice has a position of)
 * ```
 * Across devices the slices' lse are all-gathered and the weighted outputs reduce-scattered
 * (or all-reduced); on one device, or in a test, the slices are merged here.
 */
object ContextParallel {

    /** Merges [outs] (`[S, H, D]`) and their log-sum-exps [lses] (`[S, H]` f32): the merged output and lse. */
    fun merge(b: DxirEmitter, outs: List<DxirNode>, lses: List<DxirNode>): Pair<DxirNode, DxirNode> = with(b) {
        require(outs.isNotEmpty() && outs.size == lses.size) { "ContextParallel.merge: one lse per output" }
        val lseT = lses[0].type
        val outT = outs[0].type
        require(lseT.dtype == F32 && lseT.rank == 2 && outT.rank == 3 && outT.dims.take(2) == lseT.dims) {
            "ContextParallel.merge: outputs [S, H, D] and f32 lse [S, H], got $outT and $lseT"
        }
        val pT = DxirType(Bool, lseT.dims)
        fun cmp(dir: String, x: DxirNode, y: DxirNode) = op(OpKind.COMPARE, listOf(x, y), pT, attrs = mapOf("direction" to dir))
        fun where(c: DxirNode, x: DxirNode, y: DxirNode) = op(OpKind.WHERE, listOf(c, x, y), x.type)
        var m: DxirNode = lses[0]
        for (l in lses.drop(1)) m = where(cmp("GT", l, m), l, m)
        val ninf = const(Float.NEGATIVE_INFINITY, lseT)
        val zero = const(0f, lseT)
        val mS = where(cmp("EQ", m, ninf), zero, m)
        val w = lses.map { op(OpKind.EXP, listOf(op(OpKind.SUB, listOf(it, mS), lseT)), lseT) }
        var sum: DxirNode = w[0]
        for (x in w.drop(1)) sum = op(OpKind.ADD, listOf(sum, x), lseT)
        val lse = op(OpKind.ADD, listOf(mS, op(OpKind.LOG, listOf(sum), lseT)), lseT)
        val den = where(cmp("GT", sum, zero), sum, const(1f, lseT))
        var out: DxirNode? = null
        for (i in outs.indices) {
            val wi = op(OpKind.DIV, listOf(w[i], den), lseT)
            val wb = op(OpKind.BROADCAST, listOf(wi), DxirType(F32, outT.dims), attrs = mapOf("broadcast_dimensions" to listOf(0, 1)))
            val o32 = if (outT.dtype == F32) outs[i] else op(OpKind.CAST, listOf(outs[i]), DxirType(F32, outT.dims))
            val term = op(OpKind.MUL, listOf(wb, o32), DxirType(F32, outT.dims))
            out = out?.let { op(OpKind.ADD, listOf(it, term), DxirType(F32, outT.dims)) } ?: term
        }
        out!! to lse
    }
}
