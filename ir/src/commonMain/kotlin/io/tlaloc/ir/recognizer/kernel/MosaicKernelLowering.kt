package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.MosaicKernelAttrs
import io.tlaloc.ir.OpKind

/**
 * The claiming template for [OpKind.MOSAIC_KERNEL]: on a Google TPU target
 * the op becomes `stablehlo.custom_call @tpu_custom_call` carrying its
 * [MosaicKernel]; on any other target it declines.
 */
@ExperimentalTlalocApi
val MosaicTpuKernel: KernelTemplate = KernelTemplate { node, target ->
    if (node.op != OpKind.MOSAIC_KERNEL || target.vendor != "google") return@KernelTemplate null
    val kernel = MosaicKernelAttrs.parse(node, "MosaicTpuKernel").kernel
    KernelDescriptor(
        kernelName = MosaicKernel.CALL_TARGET,
        vendor = "google",
        targetArch = target.arch ?: "tpu",
        outputOperandAliases = kernel.outputOperandAliases,
        mosaic = kernel,
    )
}

/**
 * Resolve every [OpKind.MOSAIC_KERNEL] in [fn] for [target]:
 *
 * - [template] claims it (by default: the target is a Google TPU) → the op
 *   stays, annotated with a [KernelDescriptor] under
 *   [KernelDescriptor.ATTR_KEY], and emits `stablehlo.custom_call`;
 * - otherwise, if the op declares `reference_fallback` → the op is replaced
 *   by its `reference` body, inlined;
 * - otherwise → refused by name. A TPU-only kernel never runs as something
 *   else without saying so.
 *
 * Returns [fn] unchanged when it has no MOSAIC_KERNEL op. Ops with nested
 * regions are outside this pass's scope and refused when a MOSAIC_KERNEL is
 * present.
 */
@ExperimentalTlalocApi
fun lowerMosaicKernels(
    fn: DxirFunction,
    target: KernelTarget,
    template: KernelTemplate = MosaicTpuKernel,
): DxirFunction {
    if (fn.body.none { it is DxirOp && it.op == OpKind.MOSAIC_KERNEL }) return fn
    return DxirBuilder.function(fn.name) {
        for (m in fn.meshes) declareMesh(m)
        val values = HashMap<Int, List<DxirNode>>()
        for (p in fn.params) values[p.id] = listOf(param(p.name, p.type, p.sharding))
        cloneResolving(fn.body, values, this, target, template)
        fn.returns.map { resolve(it, values) }
    }
}

private fun resolve(node: DxirNode, values: Map<Int, List<DxirNode>>): DxirNode {
    val mapped = values[node.id]
        ?: error("lowerMosaicKernels: value id=${node.id} used before it is defined")
    return if (node is DxirOpResult) mapped[node.index] else mapped[0]
}

@ExperimentalTlalocApi
private fun cloneResolving(
    body: List<DxirNode>,
    values: HashMap<Int, List<DxirNode>>,
    b: DxirBuilder,
    target: KernelTarget,
    template: KernelTemplate,
) {
    for (node in body) {
        when (node) {
            is DxirConst -> values[node.id] = listOf(b.const(node.value, node.type, node.sharding))
            is DxirOp -> {
                require(node.regions.isEmpty()) {
                    "lowerMosaicKernels: op id=${node.id} (${node.op}) has nested regions, which this " +
                        "pass does not rewrite; resolve MOSAIC_KERNEL ops before control flow is formed"
                }
                val operands = node.operands.map { resolve(it, values) }
                if (node.op == OpKind.MOSAIC_KERNEL) {
                    val parsed = MosaicKernelAttrs.parse(node, "lowerMosaicKernels")
                    val descriptor = template.pickFor(node, target)
                    when {
                        descriptor != null -> {
                            val claimed = b.opMulti(
                                node.op, operands, node.types,
                                node.attrs + (KernelDescriptor.ATTR_KEY to descriptor),
                                node.sharding,
                            )
                            values[node.id] = List(node.numResults) { claimed.result(it) }
                        }
                        parsed.referenceFallback -> {
                            val inner = HashMap<Int, List<DxirNode>>()
                            for ((i, p) in parsed.reference.params.withIndex()) inner[p.id] = listOf(operands[i])
                            cloneResolving(parsed.reference.body, inner, b, target, template)
                            values[node.id] = parsed.reference.returns.map { resolve(it, inner) }
                        }
                        else -> error(
                            "lowerMosaicKernels: MOSAIC_KERNEL '${parsed.kernel.kernelName}' is a TPU " +
                                "kernel and target ${target.vendor}/${target.arch} is not a TPU; the op " +
                                "declares no reference fallback, so it is refused rather than silently " +
                                "replaced. Build it with referenceFallback = true to run its reference " +
                                "decomposition on this target.",
                        )
                    }
                } else {
                    val clone = b.opMulti(node.op, operands, node.types, node.attrs, node.sharding)
                    values[node.id] = List(node.numResults) { clone.result(it) }
                }
            }
            is DxirParam -> error("lowerMosaicKernels: unexpected param id=${node.id} in a function body")
            else -> error("lowerMosaicKernels: unsupported body node $node (id=${node.id})")
        }
    }
}
