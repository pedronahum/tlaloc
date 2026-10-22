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
 * `route_kernel = true` routes with the CUDA kernel `tlaloc_moe_route` (one warp per row)
 * instead of a sort over every expert; the routing is the same.
 * The expert weights keep a checkpoint's `[out, in]` layout. Quantized, they
 * are int8 or e4m3fn codes and two more operands give an f32 scale per expert
 * and output channel, `gateUpScale [E, 2I]` after `gateUp` and `downScale
 * [E, H]` after `down`: the weight is `code * scale`, and x is in the compute
 * dtype (f32 or bf16).
 *
 * NVFP4 (8 operands): x f32, routerLogits, then per expert the packed NVFP4
 * of gateUp `[2I, H]` and of down `[H, I]` ([Nvfp4MatmulAttrs]), each as
 * codes `[E, T, K / 64, 512]` u8, group scales `[E, T, K / 64, 64]` u8 and
 * row scales `[E, N]` f32. The rows are rounded to bf16 for the products,
 * as is silu(g) * u before the down projection; the products sum in f32.
 * With `act_quant = "nvfp4"` ([ACT_QUANT]) the rows are quantized to NVFP4 instead
 * ([Nvfp4MatmulAttrs.quantizeActivations]: an f32 scale per row, an e4m3 scale per 16
 * values), and so is silu(g) * u, kept in f32 until then, per (row, expert) pair: W4A4, the
 * products on the block-scaled FP4 tensor-core instruction. The kernel adds a row's experts in
 * the order they finish, so its last bits can differ between runs.
 */
object MoeExpertsAttrs {

    data class Parsed(
        val rows: Int,
        val hidden: Int,
        val experts: Int,
        val intermediate: Int,
        val topK: Int,
        /** Whether the expert weights are codes with per-channel scales. */
        val quantized: Boolean = false,
        /** Whether the expert weights are packed NVFP4. */
        val nvfp4: Boolean = false,
        /**
         * Sigmoid routing with a selection bias (the last operand, `[E]` f32):
         * the top k of `sigmoid(logits) + bias` are chosen, their sigmoids
         * normalized and multiplied by [routedScale]. Softmax routing otherwise.
         */
        val sigmoidBias: Boolean = false,
        val routedScale: Double = 1.0,
        /**
         * The experts as two `chlo.ragged_dot`s over the pairs sorted by expert (attribute
         * `lowering = "ragged_dot"`), the form XLA:TPU runs as its grouped matmul. Unquantized
         * weights only.
         */
        val raggedDot: Boolean = false,
        /** Whether the routing's top-k runs as the CUDA kernel `tlaloc_moe_route` ([ROUTE_KERNEL]). */
        val routeKernel: Boolean = false,
        /** Whether the activations are quantized to NVFP4 (W4A4, [ACT_QUANT]); NVFP4 experts only. */
        val actFp4: Boolean = false,
    )

    /** `act_quant = "nvfp4"`: NVFP4 experts with the activations quantized to NVFP4 as well. */
    const val ACT_QUANT = "act_quant"
    const val NVFP4 = "nvfp4"

    /** `route_kernel = true`: route with `tlaloc_moe_route` (triton/kernels/moe_route.cu). */
    const val ROUTE_KERNEL = "route_kernel"

    /** Attributes that choose a lowering and leave the arithmetic as it is. */
    private val LOWERING_ATTRS = setOf(LOWERING, ROUTE_KERNEL, ACT_QUANT)

    /** The `lowering` attribute and its one value. */
    const val LOWERING = "lowering"
    const val RAGGED_DOT = "ragged_dot"

    /** XLA:TPU falls back to an expander that TorchTPU records as wrong unless the ragged
     * dimension is a multiple of this (its `grouped_mm_aten_kernels.cc`). */
    const val RAGGED_ROW_MULTIPLE = 256

    /** The `routing` attribute's value for sigmoid routing with a selection bias. */
    const val SIGMOID_BIAS = "sigmoid_bias"

    /** The operand index of gateUp, down and their scales (-1 unquantized). */
    fun gateUpScaleIndex(p: Parsed) = if (p.quantized) 3 else -1
    fun downIndex(p: Parsed) = if (p.quantized) 4 else 3
    fun downScaleIndex(p: Parsed) = if (p.quantized) 5 else -1

    fun parse(op: DxirOp, layer: String): Parsed {
        val routeKernel = op.attrs[ROUTE_KERNEL]
        require(routeKernel == null || routeKernel is Boolean) { "$layer: MOE_EXPERTS $ROUTE_KERNEL is a Boolean, got $routeKernel" }
        val act = op.attrs[ACT_QUANT]
        val p = parseShapes(op, layer).copy(routeKernel = routeKernel == true, actFp4 = act != null)
        if (act != null) {
            require(act == NVFP4) { "$layer: MOE_EXPERTS $ACT_QUANT is '$NVFP4' or absent, got '$act'" }
            require(p.nvfp4) { "$layer: MOE_EXPERTS $ACT_QUANT '$NVFP4' takes NVFP4 experts" }
            require(p.topK >= 2 && p.hidden % 256 == 0 && p.intermediate % 256 == 0 && p.hidden <= 4096 && p.intermediate <= 4096) {
                "$layer: MOE_EXPERTS $ACT_QUANT '$NVFP4' needs top_k >= 2 and H and I in multiples of 256, at most 4096 " +
                    "(got top_k ${p.topK}, ${p.hidden}, ${p.intermediate})"
            }
        }
        val lowering = op.attrs[LOWERING] as? String ?: return p
        require(lowering == RAGGED_DOT) { "$layer: MOE_EXPERTS lowering is '$RAGGED_DOT' or absent, got '$lowering'" }
        require(!p.quantized && !p.nvfp4) {
            "$layer: MOE_EXPERTS lowering '$RAGGED_DOT' takes unquantized expert weights"
        }
        return p.copy(raggedDot = true)
    }

    private fun parseShapes(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.MOE_EXPERTS) { "$layer: MoeExpertsAttrs.parse called on ${op.op}" }
        val routing = op.attrs["routing"] as? String
        if (routing != null) {
            require(routing == SIGMOID_BIAS) { "$layer: MOE_EXPERTS routing is '$SIGMOID_BIAS' or absent (softmax), got '$routing'" }
            val bias = op.operands.last().type
            val withoutBias = DxirOp(op.id, op.op, op.operands.dropLast(1), op.attrs - "routing" - "routed_scale", op.types, op.sharding, op.regions)
            val p = parseShapes(withoutBias, layer)
            require(bias == io.tlaloc.ir.DxirType(F32, listOf(p.experts))) { "$layer: MOE_EXPERTS router bias must be f32 [${p.experts}], got $bias" }
            val scale = (op.attrs["routed_scale"] as? Number)?.toDouble() ?: 1.0
            return p.copy(sigmoidBias = true, routedScale = scale)
        }
        if (op.operands.size == 8) return parseNvfp4(op, layer)
        require(op.operands.size == 4 || op.operands.size == 6) {
            "$layer: MOE_EXPERTS takes 4 operands (x, routerLogits, gateUp, down) or, quantized, 6 " +
                "(x, routerLogits, gateUp, gateUpScale, down, downScale), got ${op.operands.size}"
        }
        val quantized = op.operands.size == 6
        val ts = op.operands.map { it.type }
        val x = ts[0]
        val logits = ts[1]
        val gu = ts[2]
        val down = ts[if (quantized) 4 else 3]
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
        require(x.dtype == F32 || x.dtype == BF16) { "$layer: MOE_EXPERTS x is f32 or bf16, got ${x.dtype}" }
        if (quantized) {
            require(gu.dtype == down.dtype && (gu.dtype == io.tlaloc.core.I8 || gu.dtype == io.tlaloc.core.F8E4M3FN)) {
                "$layer: quantized MOE_EXPERTS weights are int8 or e4m3fn codes, got ${gu.dtype}, ${down.dtype}"
            }
            require(ts[3] == io.tlaloc.ir.DxirType(F32, listOf(e, 2 * i)) && ts[5] == io.tlaloc.ir.DxirType(F32, listOf(e, h))) {
                "$layer: MOE_EXPERTS scales must be f32 [$e, ${2 * i}] and [$e, $h], got ${ts[3]} and ${ts[5]}"
            }
        } else {
            require(gu.dtype == x.dtype && down.dtype == x.dtype) {
                "$layer: MOE_EXPERTS x and the expert weights share one dtype; got ${x.dtype}, ${gu.dtype}, ${down.dtype}"
            }
        }
        val k = (op.attrs["top_k"] as? Number)?.toInt()
            ?: throw IllegalArgumentException("$layer: MOE_EXPERTS needs an integer top_k attribute")
        require(k in 1..e) { "$layer: MOE_EXPERTS top_k $k must be in 1..$e" }
        require(op.attrs.keys - LOWERING_ATTRS == setOf("top_k")) { "$layer: MOE_EXPERTS takes only top_k, got ${op.attrs.keys}" }
        require(op.types.size == 1 && op.type == io.tlaloc.ir.DxirType(F32, listOf(r, h))) {
            "$layer: MOE_EXPERTS returns f32 [$r, $h], got ${op.types}"
        }
        return Parsed(r, h, e, i, k, quantized)
    }

    private fun parseNvfp4(op: DxirOp, layer: String): Parsed {
        val ts = op.operands.map { it.type }
        val x = ts[0]
        require(x.rank == 2 && x.dtype == F32) { "$layer: NVFP4 MOE_EXPERTS x must be f32 [R, H], got $x" }
        val (r, h) = x.dims
        val logits = ts[1]
        require(logits.rank == 2 && logits.dims[0] == r && logits.dtype == F32) {
            "$layer: MOE_EXPERTS routerLogits must be f32 [$r, E], got $logits"
        }
        val e = logits.dims[1]
        val guS2 = ts[4]
        require(guS2.dtype == F32 && guS2.rank == 2 && guS2.dims[0] == e && guS2.dims[1] % 2 == 0) {
            "$layer: NVFP4 MOE_EXPERTS gate/up row scales must be f32 [$e, 2I], got $guS2"
        }
        val i = guS2.dims[1] / 2
        require(h % 64 == 0 && i % 64 == 0) { "$layer: NVFP4 MOE_EXPERTS needs H and I in groups of 64, got $h, $i" }
        fun packed(at: Int, n: Int, k: Int, what: String) {
            val t = (n + 15) / 16
            require(ts[at] == io.tlaloc.ir.DxirType(io.tlaloc.core.U8, listOf(e, t, k / 64, 512))) {
                "$layer: NVFP4 MOE_EXPERTS $what codes must be u8 [$e, $t, ${k / 64}, 512], got ${ts[at]}"
            }
            require(ts[at + 1] == io.tlaloc.ir.DxirType(io.tlaloc.core.U8, listOf(e, t, k / 64, 64))) {
                "$layer: NVFP4 MOE_EXPERTS $what scales must be u8 [$e, $t, ${k / 64}, 64], got ${ts[at + 1]}"
            }
            require(ts[at + 2] == io.tlaloc.ir.DxirType(F32, listOf(e, n))) {
                "$layer: NVFP4 MOE_EXPERTS $what row scales must be f32 [$e, $n], got ${ts[at + 2]}"
            }
        }
        packed(2, 2 * i, h, "gate/up")
        packed(5, h, i, "down")
        val k = (op.attrs["top_k"] as? Number)?.toInt()
            ?: throw IllegalArgumentException("$layer: MOE_EXPERTS needs an integer top_k attribute")
        require(k in 1..e) { "$layer: MOE_EXPERTS top_k $k must be in 1..$e" }
        require(op.attrs.keys - LOWERING_ATTRS == setOf("top_k")) { "$layer: MOE_EXPERTS takes only top_k, got ${op.attrs.keys}" }
        require(op.types.size == 1 && op.type == io.tlaloc.ir.DxirType(F32, listOf(r, h))) {
            "$layer: MOE_EXPERTS returns f32 [$r, $h], got ${op.types}"
        }
        return Parsed(r, h, e, i, k, quantized = false, nvfp4 = true)
    }

    /** The scratch, in bytes, of tlaloc_moe_fp4 (moe::ScratchBytes in triton/kernels/moe_fp4.cu). */
    fun nvfp4ScratchBytes(p: Parsed): Long {
        val pairs = p.rows.toLong() * p.topK
        return 4 * (p.experts + 1 + pairs) + 4 * pairs * 2 * p.intermediate + 2 * pairs * p.intermediate + 4 * pairs * p.hidden + 64
    }
}

/**
 * [fn] with every unquantized MOE_EXPERTS op lowered as two `chlo.ragged_dot`s
 * ([MoeExpertsAttrs.RAGGED_DOT]); quantized and NVFP4 experts keep their forms. A TPU session
 * applies it, since XLA:TPU runs a ragged dot as its grouped matmul.
 */
fun raggedDotMoe(fn: DxirFunction): DxirFunction {
    var changed = false
    val body = fn.body.map { n ->
        if (n is DxirOp && n.op == OpKind.MOE_EXPERTS && MoeExpertsAttrs.LOWERING !in n.attrs) {
            val p = MoeExpertsAttrs.parse(n, "raggedDotMoe")
            if (p.quantized || p.nvfp4) n
            else {
                changed = true
                DxirOp(n.id, n.op, n.operands, n.attrs + (MoeExpertsAttrs.LOWERING to MoeExpertsAttrs.RAGGED_DOT), n.types, n.sharding, n.regions)
            }
        } else n
    }
    if (!changed) return fn
    // Returns and later operands may hold the old op objects; nodes are matched by id downstream.
    val byId = body.associateBy { it.id }
    return DxirFunction(fn.name, fn.params, body, fn.returns.map { byId[it.id] ?: it }, fn.meshes)
}
