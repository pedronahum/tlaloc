package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * §0.4.353 — conservative static shape validation over a [DxirFunction]'s
 * top-level body. Reports only **certain** errors (a mismatch that cannot
 * execute), never style opinions; symbolic/unknown dims (any dim ≤ 0)
 * are skipped, so running this on plugin-lowered bodies whose dims
 * resolve at call time is silent-by-design until literal dims appear.
 *
 * Checked in v1:
 * - `MATMUL`: lhs last dim vs rhs first dim (the Tlaloc contraction
 *   convention, `S = A · B` contracting last(A) × first(B)).
 * - Elementwise binaries (`ADD`/`SUB`/`MUL`/`DIV`/`POW`): a right-aligned
 *   axis position where both sides are concrete, differ, and neither is 1 —
 *   the one case NumPy broadcasting cannot save either, so the interpreter,
 *   the emitter's `broadcast_in_dim` injection and the host broadcasting ops
 *   would all reject it at runtime. Phase A5c made these ops implicitly
 *   broadcasting, which is what lets rank-differing operands be checked here
 *   too (v1 skipped them and left them to an emitter that refused them).
 *
 * Consumed by the K2 checker (compile-time red squiggles,
 * `TENSOR_SHAPE_MISMATCH`) and usable as a library pass on
 * directly-built DXIR.
 */
fun validateDxirShapes(fn: DxirFunction): List<String> = buildList {
    val elementwise = setOf(OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW)
    for (node in fn.body) {
        if (node !is DxirOp) continue
        when {
            node.op == OpKind.MATMUL && node.operands.size == 2 -> {
                val l = node.operands[0].type.dims
                val r = node.operands[1].type.dims
                if (l.isNotEmpty() && r.isNotEmpty()) {
                    val k1 = l.last()
                    val k2 = r.first()
                    if (k1 > 0 && k2 > 0 && k1 != k2) {
                        add(
                            "MATMUL contract-dim mismatch: lhs ${l} contracts its last dim ($k1) " +
                                "against rhs ${r} first dim ($k2)",
                        )
                    }
                }
            }
            node.op in elementwise && node.operands.size == 2 -> {
                val l = node.operands[0].type.dims
                val r = node.operands[1].type.dims
                // Phase A5c — the elementwise binaries broadcast implicitly, so the
                // check is NumPy's, right-aligned: a rank-deficient operand is
                // treated as having size-1 leading axes. (v1 only compared
                // equal-rank pairs and left rank-differing ones to the emitter,
                // which then refused them outright.)
                val rank = maxOf(l.size, r.size)
                for (k in 0 until rank) {
                    val a = alignedDim(l, k, rank)
                    val b = alignedDim(r, k, rank)
                    if (a > 0 && b > 0 && a != b && a != 1 && b != 1) {
                        add(
                            "${node.op} operand shapes are incompatible at dim $k: " +
                                "$l vs $r ($a ≠ $b, neither broadcastable)",
                        )
                        break
                    }
                }
            }
        }
    }
}

/** Right-aligned axis [k] of [dims] within a rank-[rank] result: 1 for axes the operand lacks. */
private fun alignedDim(dims: List<Int>, k: Int, rank: Int): Int {
    val i = k - (rank - dims.size)
    return if (i < 0) 1 else dims[i]
}
