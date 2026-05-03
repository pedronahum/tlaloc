package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor

/**
 * Layer 4 §0.4.275 — Decompose every COARSENED op without an attached
 * `kernel_descriptor` by inlining its `primal_body` back into the parent
 * function. Returns the rewritten function (or [fn] unchanged if there's
 * nothing to decompose).
 *
 * # Why this exists
 *
 * The Phase-1 coarsening pipeline produces COARSENED ops carrying a
 * `primal_body` (the analytical forward) and a `gradient_body` (the
 * analytical VJP). Layer 3.3's [io.tlaloc.ir.recognizer.kernel.lowerKernelChoice]
 * then attaches a `kernel_descriptor` for targets that have a fused
 * vendor kernel for that pattern (e.g., `flash_attn_v3` on H100). The
 * StableHLO emitter consumes these descriptors as `stablehlo.custom_call`
 * dispatches.
 *
 * For targets *without* fused-kernel coverage — the CPU baseline being
 * the obvious case, but also any partial kernel registry — `lowerKernelChoice`
 * leaves the COARSENED un-annotated. The :stablehlo emitter then errors
 * with "COARSENED requires kernel_descriptor or decomposition". This pass
 * provides the decomposition: each un-annotated COARSENED is rewritten
 * inline as its decomposed primal_body, so the function lowers cleanly
 * to standard StableHLO ops.
 *
 * # CPU baseline strategy
 *
 * Per the dual-track Llama-decoder benchmark plan
 * (`memory/llama_benchmark_dual_track.md`, Phase 3a — CPU first), this
 * is the rewrite that makes "CPU = decomposed" work as the reference
 * baseline. GPU/TPU runs pick fused custom_calls via `lowerKernelChoice`;
 * CPU runs decompose to primitives via this pass; the eventual
 * benchmark headline ("X.Y× speedup vs CPU baseline") then measures the
 * actual coarsening win.
 *
 * # Scope (v1)
 *
 * - **Single-result COARSENED only.** Matches [CoarsenedBundle]'s v1
 *   scope (single-result patterns). Multi-result decomposition would
 *   need [DxirOpResult]-aware operand remapping in callers; out of scope.
 * - **Region-bearing ops in primal_body declined.** No COARSENED's
 *   primal_body produces nested regions in v1; if one ever does, this
 *   pass needs the same region-clone path that `VjpCoarsener` skips.
 * - **`kernel_descriptor` attr is the discriminator.** Presence ⇒ keep
 *   COARSENED for downstream custom_call emit; absence ⇒ decompose.
 *   No partial-decomposition mode (e.g. "decompose RmsNorm but keep
 *   FlashAttention coarsened"); the kernel_descriptor pass already
 *   decided per-op.
 */
fun decomposeCoarsened(fn: DxirFunction): DxirFunction {
    val toDecompose = fn.body.filterIsInstance<DxirOp>()
        .filter { it.op == OpKind.COARSENED && it.attrs[KernelDescriptor.ATTR_KEY] == null }
        .map { it.id }
        .toSet()

    if (toDecompose.isEmpty()) return fn

    return DxirBuilder.function(fn.name) {
        val nodeMap = HashMap<Int, DxirNode>()
        for (p in fn.params) {
            nodeMap[p.id] = param(p.name, p.type, p.sharding)
        }
        for (node in fn.body) {
            when {
                node is DxirOp && node.id in toDecompose -> {
                    val primalBody = node.attrs["primal_body"] as? DxirFunction
                        ?: error(
                            "decomposeCoarsened: COARSENED op id=${node.id} missing 'primal_body' " +
                                "attr — invariant violated by upstream coarsener",
                        )
                    val outerOperands = node.operands.map { operand ->
                        nodeMap[operand.id]
                            ?: error("decomposeCoarsened: operand id=${operand.id} not in nodeMap (came before anchor?)")
                    }
                    val results = inlinePrimalBody(this, primalBody, outerOperands)
                    require(results.size == 1) {
                        "decomposeCoarsened: only single-result COARSENED supported in v1 " +
                            "(op id=${node.id} primal_body returned ${results.size} values)"
                    }
                    nodeMap[node.id] = results[0]
                }

                node is DxirConst -> {
                    val newConst = const(node.value, node.type, node.sharding)
                    nodeMap[node.id] = newConst
                }

                node is DxirOp -> {
                    require(node.regions.isEmpty()) {
                        "decomposeCoarsened: cloning ops with nested regions (id=${node.id}, op=${node.op}) " +
                            "is out of v1 scope"
                    }
                    require(!node.isMultiResult) {
                        "decomposeCoarsened: cloning multi-result ops (id=${node.id}, op=${node.op}) " +
                            "is out of v1 scope"
                    }
                    val remappedOperands = node.operands.map { operand ->
                        require(operand !is DxirOpResult) {
                            "decomposeCoarsened: operand is DxirOpResult; multi-result chains out of v1 scope"
                        }
                        nodeMap[operand.id]
                            ?: error("decomposeCoarsened: operand id=${operand.id} not in nodeMap")
                    }
                    val newOp = op(
                        kind = node.op,
                        operands = remappedOperands,
                        type = node.type,
                        attrs = node.attrs,
                        sharding = node.sharding,
                    )
                    nodeMap[node.id] = newOp
                }

                else -> error("decomposeCoarsened: unsupported body node $node (id=${node.id})")
            }
        }

        fn.returns.map {
            nodeMap[it.id] ?: error("decomposeCoarsened: return ref id=${it.id} not in nodeMap")
        }
    }
}

/**
 * Splice the body of [primalBody] into the current builder, with
 * [primalBody]'s params bound positionally to [outerOperands]. Returns
 * the outer-function nodes that correspond to [primalBody].returns.
 *
 * Helper for [decomposeCoarsened]; could lift to a public utility once a
 * second caller surfaces.
 */
private fun inlinePrimalBody(
    builder: DxirBuilder,
    primalBody: DxirFunction,
    outerOperands: List<DxirNode>,
): List<DxirNode> {
    require(primalBody.params.size == outerOperands.size) {
        "inlinePrimalBody: primal_body has ${primalBody.params.size} params but COARSENED has " +
            "${outerOperands.size} operands"
    }

    val innerToOuter = HashMap<Int, DxirNode>()
    for ((p, outer) in primalBody.params.zip(outerOperands)) {
        innerToOuter[p.id] = outer
    }

    for (node in primalBody.body) {
        when (node) {
            is DxirConst -> {
                innerToOuter[node.id] = builder.const(node.value, node.type, node.sharding)
            }

            is DxirOp -> {
                require(node.regions.isEmpty()) {
                    "inlinePrimalBody: ops with nested regions not supported (id=${node.id}, op=${node.op})"
                }
                require(!node.isMultiResult) {
                    "inlinePrimalBody: multi-result ops not supported (id=${node.id}, op=${node.op})"
                }
                val remappedOperands = node.operands.map { operand ->
                    require(operand !is DxirOpResult) {
                        "inlinePrimalBody: operand is DxirOpResult; multi-result chains not supported"
                    }
                    innerToOuter[operand.id]
                        ?: error("inlinePrimalBody: operand id=${operand.id} not in inner-to-outer map")
                }
                val newOp = builder.op(
                    kind = node.op,
                    operands = remappedOperands,
                    type = node.type,
                    attrs = node.attrs,
                    sharding = node.sharding,
                )
                innerToOuter[node.id] = newOp
            }

            else -> error("inlinePrimalBody: unsupported body node $node (id=${node.id})")
        }
    }

    return primalBody.returns.map {
        innerToOuter[it.id]
            ?: error("inlinePrimalBody: return id=${it.id} not in inner-to-outer map")
    }
}
