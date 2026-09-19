package io.tlaloc.ir.recognizer.cost

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor

/**
 * Layer 3 §0.4.255+ — per-op + per-function cost estimator.
 *
 * Roofline-style: each op contributes a `flops` term (FLOPs needed to
 * compute it) and a `bytesMoved` term (bytes read + written if executed
 * unfused). The function aggregator sums per-op costs, with the
 * exception that `OpKind.COARSENED` ops are charged either:
 *
 * - as the sum of their `primal_body`'s op costs (if no `kernel_descriptor`
 *   attribute is present — i.e., the L3.3 lowering chose decompose), or
 * - as the same FLOP count but with bytes-moved limited to operand
 *   inputs + final output (if a `kernel_descriptor` is present — i.e.,
 *   a vendor-fused kernel will be emitted by StableHLO emit). This
 *   captures the headline benefit of fused attention kernels:
 *   intermediate `S` and `P` tensors never round-trip to HBM.
 *
 * # Scope (v1)
 *
 * - Single-result ops only.
 * - No region-bearing ops (IF / WHILE / MANUAL_COMPUTATION). Estimating
 *   through control flow needs a frequency model — out of v1.
 * - No collective costs (ALL_REDUCE / ALL_GATHER) — placeholder zeros.
 *   Layer 4's sharding-aware cost model adds the network terms.
 *
 * # Per-op formulas
 *
 * Documented inline at each branch of [estimateOp]. The dominant terms
 * are textbook (matmul = 2·m·k·n FLOPs; elementwise = N FLOPs; reduce =
 * N − 1 FLOPs ≈ N). Approximations preferred to vendor-specific
 * micro-models — the cost model is a *relative* estimator, not a
 * benchmark.
 */
fun estimateCost(fn: DxirFunction): CostEstimate {
    var total = CostEstimate.ZERO
    for (node in fn.body) {
        if (node !is DxirOp) continue
        total += estimateOp(node)
    }
    return total
}

/**
 * Estimate the cost of a single [DxirOp]. For [OpKind.COARSENED], the
 * estimate respects whether a [KernelDescriptor] is annotated (fused
 * memory cost) or absent (decompose-equivalent cost).
 */
fun estimateOp(op: DxirOp): CostEstimate {
    if (op.op == OpKind.COARSENED) return estimateCoarsened(op)

    val flops = computeFlops(op)
    val bytes = computeBytesMoved(op)
    return CostEstimate(flops, bytes)
}

/**
 * COARSENED cost dispatches on whether a `kernel_descriptor` is set.
 *
 * - **No descriptor** → decompose path: we charge the sum of the
 *   `primal_body`'s op costs (intermediate tensors round-trip to HBM
 *   between ops, so the bytes are conservative).
 * - **Descriptor set** → fused-kernel path: same FLOP count, but bytes
 *   capped to operand inputs + final outputs. Models the typical
 *   vendor-fused-kernel benefit.
 */
private fun estimateCoarsened(op: DxirOp): CostEstimate {
    val primal = op.attrs["primal_body"] as? DxirFunction
        ?: return CostEstimate.ZERO
    val decomposed = estimateCost(primal)

    val hasKernel = op.attrs[KernelDescriptor.ATTR_KEY] != null
    if (!hasKernel) return decomposed

    // Fused path: keep FLOPs, lower bytes to (operands + outputs) only.
    val operandBytes = op.operands.sumOf { typeBytes(it.type).toDouble() }
    val outputBytes = op.types.sumOf { typeBytes(it).toDouble() }
    return CostEstimate(decomposed.flops, operandBytes + outputBytes)
}

/**
 * FLOPs for a single op, by op kind. See per-branch comments for the
 * derivation.
 */
private fun computeFlops(op: DxirOp): Double = when (op.op) {
    // Linear algebra.
    //   matmul [m, k] · [k, n] → [m, n]: 2·m·k·n FLOPs (one MAC per output).
    //   Batched matmul scales by batch product.
    OpKind.MATMUL -> matmulFlops(op)

    // Convolution: stub — convolutional cost in v1 isn't first-class
    // (Tlaloc's wedge audiences don't drive conv heavy work). We ship
    // a placeholder that scales with output volume × in-channels × kernel
    // volume; real IREE/StableHLO cost models can override later.
    // §0.4.385 — the fused conv adjoints cost the same order as the conv they
    // adjoint (the kernel one adds two transposes, which are pure data movement
    // and already accounted under bytes).
    OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D,
    OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT,
    -> {
        // Conservative: out_elements × in_channels × 9 (3×3 kernel).
        op.type.elementCount.toDouble() * 9.0 * 1.0
    }

    // §0.4.363 — window pooling: one compare/add per window tap per output.
    // §0.4.386 — AVGPOOL2D_GRAD is the same order of work spread the other way
    // (each output upstream lands on `window` input elements).
    OpKind.MAXPOOL2D, OpKind.AVGPOOL2D, OpKind.AVGPOOL2D_GRAD ->
        op.type.elementCount.toDouble() * 9.0

    OpKind.DOT -> {
        // Vector dot product [N] · [N] → scalar: 2N − 1 ≈ 2N.
        2.0 * op.operands[0].type.elementCount.toDouble()
    }

    // Pure data movement — zero compute, captured under bytes.
    OpKind.TRANSPOSE, OpKind.BROADCAST, OpKind.RESHAPE,
    OpKind.SLICE, OpKind.GATHER, OpKind.SCATTER, OpKind.SCATTER_ADD,
    // §0.4.360 — PAD is data movement; WHERE/COMPARE are 1 op/element
    // (folded into the elementwise bucket below by their users; kept at
    // movement-cost here since they never dominate a kernel decision).
    OpKind.CONCAT, OpKind.SPLIT, OpKind.CAST, OpKind.PAD, OpKind.WHERE, OpKind.COMPARE -> 0.0

    // Elementwise binary (one op per output element).
    OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW,
    OpKind.LAND, OpKind.NOT -> op.type.elementCount.toDouble()

    // Elementwise unary.
    OpKind.NEG, OpKind.ABS, OpKind.SIGN, OpKind.STEP -> op.type.elementCount.toDouble()
    OpKind.EXP, OpKind.LOG, OpKind.SQRT, OpKind.RSQRT -> 4.0 * op.type.elementCount.toDouble()
    OpKind.TANH, OpKind.SIGMOID -> 6.0 * op.type.elementCount.toDouble()
    OpKind.RELU, OpKind.GELU, OpKind.SILU -> 5.0 * op.type.elementCount.toDouble()
    OpKind.SIN, OpKind.COS -> 8.0 * op.type.elementCount.toDouble()

    // Reductions over an axis (or all): N − 1 ≈ N adds per reduced
    // element.
    OpKind.SUM, OpKind.MEAN, OpKind.MAX, OpKind.MIN, OpKind.ARGMAX -> {
        val inputElems = op.operands[0].type.elementCount.toDouble()
        inputElems
    }

    // §0.4.373 — SUM_TO (numpy unbroadcast): one add per value element into
    // the (smaller) template-shaped accumulator, like a reduction.
    OpKind.SUM_TO -> op.operands[0].type.elementCount.toDouble()

    // §0.4.374 — PAD_TO (zero-pad to template): one write per value element
    // into the (larger) template-shaped output, like a copy.
    OpKind.PAD_TO -> op.operands[0].type.elementCount.toDouble()

    // Phase A2b — SLICE_LIKE (runtime-extent window slice, CONCAT's adjoint):
    // one write per output element, like a copy.
    OpKind.SLICE_LIKE -> op.type.elementCount.toDouble()

    // Softmax = max + sub + exp + sum + div per element of the reduced
    // axis. ~5 FLOPs per input element, plus the reduction.
    OpKind.SOFTMAX, OpKind.LOGSUMEXP -> 5.0 * op.operands[0].type.elementCount.toDouble()

    // Normalization compounds — model as weighted sums.
    OpKind.RMSNORM -> 5.0 * op.type.elementCount.toDouble()
    OpKind.LAYERNORM -> 8.0 * op.type.elementCount.toDouble()
    OpKind.BATCHNORM -> 8.0 * op.type.elementCount.toDouble()

    OpKind.SCALED_DOT_PRODUCT_ATTENTION -> {
        // Pre-fused attention: roughly 4·b·h·m·n·d FLOPs for forward
        // (two matmuls + softmax) where dims are inferred from the
        // operand shapes. v1 approximation: 4 × output volume × middle
        // dim if available.
        val outputType = op.type
        val r = outputType.rank
        val mid = if (r >= 2) outputType.dims[r - 1] else 64
        4.0 * outputType.elementCount.toDouble() * mid
    }

    OpKind.EMBEDDING -> op.type.elementCount.toDouble()
    // §0.4.370 — embedding adjoint: one scatter-add per upstream element.
    OpKind.EMBEDDING_GRAD -> op.operands[1].type.elementCount.toDouble()
    OpKind.CROSS_ENTROPY -> 5.0 * op.operands[0].type.elementCount.toDouble()

    // Control flow: structural, no per-op compute.
    OpKind.IF, OpKind.WHILE -> 0.0

    // Sharding markers — purely metadata.
    OpKind.SHARD_CONSTRAINT -> 0.0

    // Manual computation: nested region; v1 doesn't recurse.
    OpKind.MANUAL_COMPUTATION -> 0.0

    // Collectives: placeholder. L4 adds network-cost terms.
    OpKind.ALL_REDUCE, OpKind.ALL_GATHER, OpKind.REDUCE_SCATTER -> 0.0

    // COARSENED handled separately by [estimateCoarsened].
    OpKind.COARSENED -> error("estimateCoarsened should have been called")
}

/**
 * MATMUL FLOPs: for `[B0..Bk-1, M, K] · [B0..Bk-1, K, N] → [B0..Bk-1, M, N]`,
 * the count is `2·M·K·N · prod(B0..Bk-1)`. Falls back to a conservative
 * `2 * out_elements * inner_dim` if the operand shapes don't follow the
 * canonical convention.
 */
private fun matmulFlops(op: DxirOp): Double {
    val a = op.operands.getOrNull(0)?.type ?: return 0.0
    val b = op.operands.getOrNull(1)?.type ?: return 0.0
    val ar = a.rank
    val br = b.rank
    if (ar < 2 || br < 2) return 2.0 * op.type.elementCount.toDouble()
    val m = a.dims[ar - 2]
    val k = a.dims[ar - 1]
    val n = b.dims[br - 1]
    val batchProd = (0 until ar - 2).fold(1L) { acc, i -> acc * a.dims[i] }
    return 2.0 * batchProd * m * k * n
}

/**
 * Bytes-moved for a single op: read all inputs once, write all outputs
 * once. This is the unfused estimate the function aggregator uses by
 * default. Fused-kernel paths override this via [estimateCoarsened].
 */
private fun computeBytesMoved(op: DxirOp): Double {
    val inputBytes = op.operands.sumOf { typeBytes(it.type).toDouble() }
    val outputBytes = op.types.sumOf { typeBytes(it).toDouble() }
    return inputBytes + outputBytes
}

/** Total bytes for a tensor of [type], rounded to the nearest byte. */
internal fun typeBytes(type: DxirType): Long =
    type.elementCount * type.dtype.sizeBytes

/**
 * Pick the appropriate device peak for the dtype family of [op]. F32 →
 * F32 peak, BF16/FP16 (collapsed) → BF16 peak, FP8 if available else
 * BF16 fallback. Other dtypes (Int, Bool) fall back to F32 peak as a
 * rough estimate.
 */
internal fun peakForDtype(op: DxirOp, device: DeviceDescriptor): Double {
    val dtype = op.type.dtype
    return when (dtype) {
        F32 -> device.peakFlopsF32
        F64 -> device.peakFlopsF32 / 2.0  // F64 typically ½ F32 on tensor cores
        else -> device.peakFlopsBf16  // BF16 / FP16 / lower-precision fallback
    }
}

/**
 * Pick a peak appropriate for the *function's* dominant dtype. Picks
 * the mode dtype across body ops; ties broken by F32 first, then BF16.
 * Used by [estimateRooflineSeconds] when caller doesn't know the
 * function's primary dtype upfront.
 */
internal fun dominantDeviceFlops(fn: DxirFunction, device: DeviceDescriptor): Double {
    val counts = HashMap<io.tlaloc.core.DType, Int>()
    for (n in fn.body) if (n is DxirOp) counts.merge(n.type.dtype, 1) { a, b -> a + b }
    val mode = counts.maxByOrNull { it.value }?.key ?: F32
    return when (mode) {
        F32 -> device.peakFlopsF32
        F64 -> device.peakFlopsF32 / 2.0
        else -> device.peakFlopsBf16
    }
}

/**
 * Convenience: roofline-style time estimate for [fn] on [device], in
 * microseconds. Picks the dominant-dtype peak automatically.
 */
fun estimateRooflineMicros(fn: DxirFunction, device: DeviceDescriptor): Double {
    val cost = estimateCost(fn)
    val peak = dominantDeviceFlops(fn, device)
    return cost.roofineSeconds(device, peak) * 1e6
}
