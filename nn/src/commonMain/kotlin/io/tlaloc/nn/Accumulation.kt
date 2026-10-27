package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInline

/**
 * This step over [microbatches] microbatches in one program: the loss is the mean of the
 * microbatch losses and the parameter gradients are the mean of theirs, so a batch that does not
 * fit the device as one runs as [microbatches] smaller ones within one step.
 *
 * The step takes `inputCount × microbatches` inputs, microbatch by microbatch (each microbatch's
 * inputs and targets in the original order), then the parameters. It returns the mean loss, the
 * gradient of the mean loss for each input, and the parameter gradients; [withDeviceAdamW],
 * [withDeviceSGD] and [withDeviceMuon] apply to it as to any step.
 *
 * With [sequential] each microbatch's inputs pass through an [OpKind.OPTIMIZATION_BARRIER]
 * anchored on the previous microbatches' accumulated loss and gradients, so the compiler runs
 * the microbatches one after the other and holds one microbatch's activations at a time. Apply
 * [CapturedStep.withRemat] before this, not after. On the GB10 a 4-layer CausalLM (d 512) over
 * 8 × 256 tokens peaks at 508 MiB as one batch and 256 MiB as 4 microbatches in sequence.
 *
 * Without [sequential] the compiler may interleave the microbatches (467 MiB in the same case).
 * XLA:GPU then merges the matrix products of different microbatches that share a weight; at the
 * vendored XLA commit its v2 GEMM fusions of the merged products compute wrong parameter
 * gradients, and a CUDA `PjrtSession` compiles with them off.
 */
fun CapturedStep.withGradientAccumulation(microbatches: Int, sequential: Boolean = true): CapturedStep {
    require(microbatches >= 1) { "withGradientAccumulation: microbatches must be >= 1, got $microbatches" }
    require(this.microbatches == 1) { "withGradientAccumulation: this step already accumulates ${this.microbatches} microbatches" }
    if (microbatches == 1) return this
    val nIn = inputCount
    val nP = parameterKeys.size
    var next = maxOf(DxirInline.maxId(primal), DxirInline.maxId(gradient)) + 1
    val id = { next++ }

    // Microbatch inputs, shared by both functions; the parameters keep their nodes.
    val inputs = (0 until microbatches).map { k ->
        (0 until nIn).map { j -> gradient.params[j].let { p -> DxirParam(id(), "${p.name}_mb$k", p.type, p.sharding) } }
    }

    fun MutableList<DxirNode>.add(a: DxirNode, b: DxirNode): DxirNode =
        DxirOp(id(), OpKind.ADD, listOf(a, b), type = a.type).also { this += it }

    /** [x] / microbatches; integer values (an input's zero gradient) are left as they are. */
    fun MutableList<DxirNode>.mean(x: DxirNode): DxirNode {
        if (x.type.dtype != F32) return x
        val n = DxirConst(id(), microbatches.toFloat(), DxirType(F32, emptyList())).also { this += it }
        val nx = if (x.type.dims.isEmpty()) n
        else DxirOp(id(), OpKind.BROADCAST, listOf(n), mapOf("broadcast_dimensions" to emptyList<Int>()), x.type).also { this += it }
        return DxirOp(id(), OpKind.DIV, listOf(x, nx), type = x.type).also { this += it }
    }

    // The primal: the mean of the microbatch losses.
    val primalBody = ArrayList<DxirNode>()
    val losses = inputs.map { DxirInline.inline(primal, it + primal.params.drop(nIn), id, primalBody)[0] }
    val primalLoss = primalBody.mean(losses.drop(1).fold(losses[0]) { s, x -> primalBody.add(s, x) })
    val newPrimal = DxirFunction(
        "${primal.name}_x$microbatches", inputs.flatten() + primal.params.drop(nIn), primalBody, listOf(primalLoss), primal.meshes,
    )

    // The gradient: each microbatch's gradient, added to the running sums as it is produced.
    val body = ArrayList<DxirNode>()
    var lossSum: DxirNode? = null
    var gradSums: List<DxirNode> = emptyList()
    val inputGrads = ArrayList<DxirNode>()
    for ((k, xs) in inputs.withIndex()) {
        val anchors = listOfNotNull(lossSum) + gradSums
        val args = if (!sequential || k == 0) xs else xs.map { x ->
            DxirOp(id(), OpKind.OPTIMIZATION_BARRIER, listOf(x) + anchors, types = listOf(x.type)).also { body += it }
        }
        val outs = DxirInline.inline(gradient, args + gradient.params.drop(nIn), id, body)
        lossSum = lossSum?.let { body.add(it, outs[0]) } ?: outs[0]
        val g = outs.subList(1 + nIn, 1 + nIn + nP)
        gradSums = if (k == 0) g else gradSums.zip(g) { s, x -> body.add(s, x) }
        inputGrads += outs.subList(1, 1 + nIn)
    }
    val returns = listOf(body.mean(lossSum!!)) + inputGrads.map { body.mean(it) } + gradSums.map { body.mean(it) }
    val newGradient = DxirFunction(
        "${gradient.name}_x$microbatches", inputs.flatten() + gradient.params.drop(nIn), body, returns, gradient.meshes,
    )
    return CapturedStep(newPrimal, newGradient, parameterKeys, nIn * microbatches, frozenKeys, microbatches)
}
