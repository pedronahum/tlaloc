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
 * - Elementwise binaries (`ADD`/`SUB`/`MUL`/`DIV`/`POW`) at equal rank:
 *   a dim position where both sides are concrete, differ, and neither is
 *   1 (the emitter's broadcast rules cannot save it). Rank-differing
 *   operands are left to the emitter's broadcast machinery.
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
                if (l.size == r.size) {
                    for (i in l.indices) {
                        val a = l[i]
                        val b = r[i]
                        if (a > 0 && b > 0 && a != b && a != 1 && b != 1) {
                            add(
                                "${node.op} operand shapes are incompatible at dim $i: " +
                                    "$l vs $r ($a ≠ $b, neither broadcastable)",
                            )
                            break
                        }
                    }
                }
            }
        }
    }
}
