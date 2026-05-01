package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.OpKind

/**
 * Layer 3 §0.4.253+ — kernel template registry + decompose-fallback
 * driver.
 *
 * # Pipeline shape
 *
 * ```
 * fn (with COARSENED ops post-L3.2)
 *   → lowerKernelChoice(fn, target)
 *   → fn' (each COARSENED either *annotated* with a KernelDescriptor
 *           for downstream custom-call emit, or *decomposed* by inlining
 *           its primal_body back into the outer body)
 * ```
 *
 * # Two paths per COARSENED
 *
 * 1. **Kernel selected** — the per-pattern [KernelTemplate] returned a
 *    [KernelDescriptor] for [target]. The COARSENED op is rewritten in
 *    place with the descriptor stashed under
 *    [KernelDescriptor.ATTR_KEY]. StableHLO emit (post-Layer-3 work)
 *    consumes that attr to emit `stablehlo.custom_call` with the
 *    kernel name.
 *
 * 2. **No kernel — decompose** — the template returned `null`. The
 *    COARSENED op is replaced by the inlined contents of its
 *    `primal_body`, with primal-body params resolved to the COARSENED's
 *    operands. The function shape after decomposition matches what the
 *    user originally wrote (recognizer would no-op on the result, by
 *    design — decomposed code is no longer a "FlashAttention" pattern).
 *
 * # Why annotate instead of wrapping in MANUAL_COMPUTATION?
 *
 * The L3 plan's intent ("custom-call envelopes via OpKind.MANUAL_COMPUTATION
 * reuse") was to give downstream lowering a uniform op shape to dispatch
 * on. In Tlaloc today, `MANUAL_COMPUTATION` is reserved for Shardy's
 * `sdy.manual_computation` (in_shardings/out_shardings/manual_axes
 * required by the existing emitter at stablehlo/Emitter.kt:1131-1135).
 * Reusing the op for kernel custom-calls would require extending the
 * SDY emitter to dispatch on attrs — out of L3.3 scope.
 *
 * Annotating the existing COARSENED op with a [KernelDescriptor] attr
 * achieves the same separation (pattern-recognized + kernel-tagged
 * sub-graphs are distinguishable at lowering time) without an emitter
 * change. Tracked as audit OQ-Layer3-1: "evaluate adding a dedicated
 * `KERNEL_CALL` opkind once a second backend (IREE, Triton) needs the
 * dispatch shape".
 *
 * # Default registry
 *
 * [defaultKernelTemplates] ships one entry — `"FlashAttention"` →
 * [FlashAttentionKernel]. Add patterns by adding a new file +
 * registry entry, mirroring the L3.2 coarsener registry pattern.
 */
fun lowerKernelChoice(
    fn: DxirFunction,
    target: KernelTarget,
    registry: Map<String, KernelTemplate> = defaultKernelTemplates,
): DxirFunction {
    val coarsenedOps = fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
    if (coarsenedOps.isEmpty()) return fn

    // Per-coarsened decision: kernel descriptor or null (decompose).
    // Keyed by COARSENED op id.
    val perCoarsened = HashMap<Int, KernelDescriptor?>()
    for (co in coarsenedOps) {
        val template = lookupTemplate(co, registry) ?: continue
        perCoarsened[co.id] = template.pickFor(co, target)
    }
    if (perCoarsened.isEmpty()) return fn

    return DxirBuilder.function(fn.name) {
        val nodeMap = HashMap<Int, DxirNode>()
        for (p in fn.params) {
            nodeMap[p.id] = param(p.name, p.type, p.sharding)
        }
        for (node in fn.body) {
            when {
                node is DxirOp && node.op == OpKind.COARSENED && node.id in perCoarsened -> {
                    val descriptor = perCoarsened[node.id]
                    val outerOperands = node.operands.map {
                        nodeMap[it.id] ?: error("KernelLowering: operand id=${it.id} not in nodeMap")
                    }
                    if (descriptor != null) {
                        // Annotate path — emit a fresh COARSENED with
                        // the descriptor stashed in attrs alongside the
                        // existing primal_body / gradient_body /
                        // reads_primal_indices triple.
                        val annotatedAttrs = node.attrs + mapOf(KernelDescriptor.ATTR_KEY to descriptor)
                        // Multi-result single-typed COARSENED: there's
                        // exactly one result. Use the generic op() (not
                        // the convenience coarsened() builder, which
                        // hard-codes the 3-attr map).
                        val annotated = op(
                            kind = OpKind.COARSENED,
                            operands = outerOperands,
                            type = node.type,
                            attrs = annotatedAttrs,
                            sharding = node.sharding,
                        )
                        nodeMap[node.id] = annotated
                    } else {
                        // Decompose path — inline primal_body's body in
                        // place of the COARSENED.
                        inlineCoarsenedPrimal(node, outerOperands, nodeMap, this)
                    }
                }

                node is DxirConst -> {
                    nodeMap[node.id] = const(node.value, node.type, node.sharding)
                }

                node is DxirOp -> {
                    require(node.regions.isEmpty()) {
                        "KernelLowering: cloning ops with nested regions (id=${node.id}, op=${node.op}) is out of v1 scope"
                    }
                    require(!node.isMultiResult) {
                        "KernelLowering: cloning multi-result ops (id=${node.id}, op=${node.op}) is out of v1 scope"
                    }
                    val operands = node.operands.map {
                        require(it !is DxirOpResult) {
                            "KernelLowering: DxirOpResult operand out of v1 scope"
                        }
                        nodeMap[it.id]
                            ?: error("KernelLowering: operand id=${it.id} not in nodeMap")
                    }
                    val newOp = op(node.op, operands, node.type, node.attrs, node.sharding)
                    nodeMap[node.id] = newOp
                }

                else -> error("KernelLowering: unsupported body node $node (id=${node.id})")
            }
        }
        fn.returns.map {
            nodeMap[it.id] ?: error("KernelLowering: return ref id=${it.id} not in nodeMap")
        }
    }
}

/**
 * Look up a [KernelTemplate] for [coarsened] in [registry].
 *
 * Bridges the L3.2 coarsener's `primal_body.name = "<pattern>_primal"`
 * convention (snake_case) to the registry's CamelCase keys
 * (`"FlashAttention"`). The mapping is case-insensitive, hyphen-/
 * underscore-stripped.
 */
private fun lookupTemplate(
    coarsened: DxirOp,
    registry: Map<String, KernelTemplate>,
): KernelTemplate? {
    val primalName = (coarsened.attrs["primal_body"] as? DxirFunction)?.name ?: return null
    val patternKey = primalName.removeSuffix("_primal").replace("_", "")
    return registry.entries.firstOrNull { it.key.equals(patternKey, ignoreCase = true) }?.value
}

/**
 * Inline a `COARSENED` op's primal_body into the outer builder, mapping
 * primal_params positionally to [outerOperands] and primal_body ops to
 * fresh outer-scope clones. The COARSENED op's id is mapped to the
 * primal_body's last return (which becomes the value any consumer of
 * the original COARSENED references).
 *
 * v1 limitation: assumes single-result COARSENED, no regions inside the
 * primal_body, no multi-result ops in the primal_body. This matches
 * what L3.2's [coarsenRecognizedPatterns] produces today.
 */
private fun inlineCoarsenedPrimal(
    coarsenedOp: DxirOp,
    outerOperands: List<DxirNode>,
    outerNodeMap: HashMap<Int, DxirNode>,
    builder: DxirBuilder,
) {
    val primal = coarsenedOp.attrs["primal_body"] as DxirFunction
    require(primal.params.size == outerOperands.size) {
        "KernelLowering: primal_body.params count ${primal.params.size} ≠ outer operand count ${outerOperands.size}"
    }
    require(primal.returns.size == 1) {
        "KernelLowering: v1 inline supports single-result primal only; got ${primal.returns.size}"
    }

    // Map primal-scope ids → outer-scope nodes.
    val inlineMap = HashMap<Int, DxirNode>()
    for ((i, p) in primal.params.withIndex()) {
        inlineMap[p.id] = outerOperands[i]
    }
    for (n in primal.body) {
        when (n) {
            is DxirParam -> error("KernelLowering: unexpected param in primal_body.body")
            is DxirConst -> {
                inlineMap[n.id] = builder.const(n.value, n.type, n.sharding)
            }
            is DxirOp -> {
                require(n.regions.isEmpty()) {
                    "KernelLowering: primal_body op id=${n.id} has nested regions; out of v1 scope"
                }
                require(!n.isMultiResult) {
                    "KernelLowering: primal_body op id=${n.id} is multi-result; out of v1 scope"
                }
                val operands = n.operands.map {
                    require(it !is DxirOpResult) {
                        "KernelLowering: DxirOpResult in primal_body operands; out of v1 scope"
                    }
                    inlineMap[it.id]
                        ?: error("KernelLowering: primal-scope id=${it.id} not in inlineMap")
                }
                val newOp = builder.op(n.op, operands, n.type, n.attrs, n.sharding)
                inlineMap[n.id] = newOp
            }
            else -> error("KernelLowering: unsupported primal_body node $n")
        }
    }
    // The COARSENED op's value is the primal_body's single return —
    // map the COARSENED id to that return's outer-scope node.
    val ret = primal.returns.single()
    val outerReturn = inlineMap[ret.id]
        ?: error("KernelLowering: primal return id=${ret.id} not in inlineMap")
    outerNodeMap[coarsenedOp.id] = outerReturn
}

/**
 * v1 default registry — mirrors L3.2's `defaultCoarseners`. One entry
 * per recognized pattern. Adding a pattern = new file + one entry.
 */
val defaultKernelTemplates: Map<String, KernelTemplate> = mapOf(
    "FlashAttention" to FlashAttentionKernel,
)
