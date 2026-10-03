package io.tlaloc.ir

import io.tlaloc.core.BF16
import io.tlaloc.core.F32

/**
 * [OpKind.MOE_EXPERTS]'s operands and attribute, one parser for every layer
 * that touches the kind.
 * ```
 *   0 x            [R, H]       the routed rows, in the weights' dtype
 *   1 routerLogits [R, E]       f32
 *   2 gateUp       [E, 2I, H]   each expert's gate and up projections, [gate | up] along 2I
 *   3 down         [E, H, I]    each expert's down projection
 *   -> y           [R, H]       f32
 * attrs: top_k: Int (1..E)
 * ```
 * The expert weights keep a checkpoint's `[out, in]` layout.
 */
object MoeExpertsAttrs {

    data class Parsed(val rows: Int, val hidden: Int, val experts: Int, val intermediate: Int, val topK: Int)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.MOE_EXPERTS) { "$layer: MoeExpertsAttrs.parse called on ${op.op}" }
        require(op.operands.size == 4) {
            "$layer: MOE_EXPERTS takes 4 operands (x, routerLogits, gateUp, down), got ${op.operands.size}"
        }
        val (x, logits, gu, down) = op.operands.map { it.type }
        require(x.rank == 2) { "$layer: MOE_EXPERTS x must be [R, H], got ${x.dims}" }
        val (r, h) = x.dims
        require(logits.rank == 2 && logits.dims[0] == r && logits.dtype == F32) {
            "$layer: MOE_EXPERTS routerLogits must be f32 [$r, E], got $logits"
        }
        val e = logits.dims[1]
        require(gu.rank == 3 && gu.dims[0] == e && gu.dims[2] == h && gu.dims[1] % 2 == 0) {
            "$layer: MOE_EXPERTS gateUp must be [$e, 2I, $h], got ${gu.dims}"
        }
        val i = gu.dims[1] / 2
        require(down.dims == listOf(e, h, i)) { "$layer: MOE_EXPERTS down must be [$e, $h, $i], got ${down.dims}" }
        require(gu.dtype == x.dtype && down.dtype == x.dtype && (x.dtype == F32 || x.dtype == BF16)) {
            "$layer: MOE_EXPERTS x and the expert weights share one dtype, f32 or bf16; got " +
                "${x.dtype}, ${gu.dtype}, ${down.dtype}"
        }
        val k = (op.attrs["top_k"] as? Number)?.toInt()
            ?: throw IllegalArgumentException("$layer: MOE_EXPERTS needs an integer top_k attribute")
        require(k in 1..e) { "$layer: MOE_EXPERTS top_k $k must be in 1..$e" }
        require(op.attrs.keys == setOf("top_k")) { "$layer: MOE_EXPERTS takes only top_k, got ${op.attrs.keys}" }
        require(op.types.size == 1 && op.type == io.tlaloc.ir.DxirType(F32, listOf(r, h))) {
            "$layer: MOE_EXPERTS returns f32 [$r, $h], got ${op.types}"
        }
        return Parsed(r, h, e, i, k)
    }
}
