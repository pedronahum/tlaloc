package io.tlaloc.plugin

import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * A two-operand `BROADCAST` carries a shape template (operand 1). Synthesis lowers it to
 * `broadcastLike` or `stretchLike`, which take the template's run-time shape, and types
 * the result with the template's IrType. That is sound only when the op's own result
 * shape is the template's: same rank, and equal extents wherever both are known.
 * Synthesis refuses a templated `BROADCAST` that fails this check rather than give it
 * the template's type.
 */
internal object BroadcastTemplate {
    fun agrees(op: DxirOp): Boolean {
        if (op.op != OpKind.BROADCAST || op.operands.size != 2) return false
        val result = op.type.dims
        val template = op.operands[1].type.dims
        if (result.size != template.size) return false
        return result.indices.all { i -> result[i] < 0 || template[i] < 0 || result[i] == template[i] }
    }
}
