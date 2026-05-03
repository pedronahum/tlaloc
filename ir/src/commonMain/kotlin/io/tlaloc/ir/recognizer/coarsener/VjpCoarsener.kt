package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * Layer 3 §0.4.252+ — VJP coarsener registry + driver.
 *
 * Consumes the output of `recognizeAll(fn)` and rewrites [fn] so each
 * recognized sub-graph is replaced by a single `OpKind.COARSENED` op
 * carrying the analytical primal + gradient bodies for that pattern.
 *
 * # Pipeline shape
 *
 * ```
 * fn (raw user code)
 *   → recognizer.recognizeAll(fn)         [L3.0/L3.1]
 *   → coarsenRecognizedPatterns(fn, ms)   [this file, L3.2]
 *   → fn' (with COARSENED envelopes)
 *   → DxirReverseTransform                [pre-existing]
 *   → grad fn (analytical VJPs spliced in)
 * ```
 *
 * # Registry shape
 *
 * Coarseners are registered by `RecognitionMatch.patternName`. v1
 * supplies a single entry — `FlashAttention`. RmsNorm / RoPE /
 * CrossEntropy coarseners can land in subsequent §0.4.x phases by
 * adding a per-pattern file + one [defaultCoarseners] entry. No
 * build-system changes required.
 *
 * # Scope (first cut)
 *
 * - Single top-level matches (no nested matches inside region-bearing
 *   ops). `recognizeAll`'s `resolveLargestMatch` already strips
 *   overlapping matches, so the input list is non-overlapping.
 * - Single-result COARSENED only — every pattern v1 emits one tensor.
 * - Skip patterns whose matched sub-graph has *external* consumers of
 *   non-anchor ops. Those would need the absorbed ops to remain in the
 *   outer function as well — out of v1's scope.
 * - No multi-result ops or `DxirOpResult` references in the body. Will
 *   panic loudly if one is encountered; that path needs the
 *   result-index-aware nodeMap discussed in `PhiCalculus.cloneGradNode`.
 */
fun coarsenRecognizedPatterns(
    fn: DxirFunction,
    matches: List<RecognitionMatch>,
    coarseners: Map<String, PatternCoarsener> = defaultCoarseners,
): DxirFunction {
    if (matches.isEmpty()) return fn

    // 1. Build (anchorOpId → bundle) and the union of absorbed op ids.
    //    Skip matches with no registered coarsener or that the coarsener
    //    declines (returns null).
    val anchorToBundle = HashMap<Int, CoarsenedBundle>()
    val absorbedOpIds = HashSet<Int>()

    for (match in matches) {
        val coarsener = coarseners[match.patternName] ?: continue
        val bundle = coarsener.coarsen(match) ?: continue

        // Defensive: if any absorbed id overlaps an already-claimed one,
        // skip — resolveLargestMatch should have prevented this, but
        // belt-and-braces matters here because an overlap would corrupt
        // the rewrite.
        if (bundle.absorbedOpIds.any { it in absorbedOpIds }) continue

        // External-consumer check: every absorbed op other than the
        // anchor must have all its consumers also absorbed. If a
        // non-anchor absorbed op has an external consumer, we'd lose its
        // value when we drop it from the outer body. Skip the match.
        if (!hasNoExternalConsumers(fn, bundle)) continue

        anchorToBundle[bundle.anchorOpId] = bundle
        absorbedOpIds += bundle.absorbedOpIds
    }

    if (anchorToBundle.isEmpty()) return fn

    // 2. Rebuild the function. Walk fn.body in order; clone non-absorbed
    //    nodes; emit a COARSENED at each anchor; skip non-anchor
    //    absorbed nodes.
    return DxirBuilder.function(fn.name) {
        val nodeMap = HashMap<Int, DxirNode>()

        for (p in fn.params) {
            val newP = param(p.name, p.type, p.sharding)
            nodeMap[p.id] = newP
        }

        for (node in fn.body) {
            when {
                node.id in anchorToBundle -> {
                    val bundle = anchorToBundle[node.id]!!
                    val remappedOperands = bundle.outerOperands.map {
                        nodeMap[it.id]
                            ?: error("VjpCoarsener: outer operand id=${it.id} not in nodeMap (came before anchor?)")
                    }
                    val coarsenedOp = coarsened(
                        operands = remappedOperands,
                        primalBody = bundle.primalBody,
                        gradientBody = bundle.gradientBody,
                        readsPrimalIndices = bundle.readsPrimalIndices,
                    )
                    nodeMap[node.id] = coarsenedOp
                }

                node.id in absorbedOpIds -> {
                    // Non-anchor absorbed op: skip. Its computation
                    // lives inside the COARSENED's primal_body now.
                }

                node is DxirConst -> {
                    val newConst = const(node.value, node.type, node.sharding)
                    nodeMap[node.id] = newConst
                }

                node is DxirOp -> {
                    require(node.regions.isEmpty()) {
                        "VjpCoarsener: cloning ops with nested regions (id=${node.id}, op=${node.op}) " +
                            "is out of v1 scope; recognizers shouldn't produce matches inside region-bearing ops"
                    }
                    require(!node.isMultiResult) {
                        "VjpCoarsener: cloning multi-result ops (id=${node.id}, op=${node.op}) " +
                            "is out of v1 scope; would need DxirOpResult-aware operand remap"
                    }
                    val remappedOperands = node.operands.map {
                        require(it !is DxirOpResult) {
                            "VjpCoarsener: operand is DxirOpResult; multi-result chains out of v1 scope"
                        }
                        nodeMap[it.id]
                            ?: error("VjpCoarsener: operand id=${it.id} not in nodeMap (out of order?)")
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

                else -> error("VjpCoarsener: unsupported body node $node (id=${node.id})")
            }
        }

        fn.returns.map {
            nodeMap[it.id]
                ?: error("VjpCoarsener: return ref id=${it.id} not in nodeMap")
        }
    }
}

/**
 * Verify that every absorbed op other than the anchor has its consumers
 * fully contained inside the absorbed set. Returns `false` if any
 * non-anchor absorbed op is consumed by an op outside the match.
 */
private fun hasNoExternalConsumers(fn: DxirFunction, bundle: CoarsenedBundle): Boolean {
    for (node in fn.body) {
        if (node !is DxirOp) continue
        if (node.id in bundle.absorbedOpIds) continue
        for (operand in node.operands) {
            if (operand.id == bundle.anchorOpId) continue  // anchor is exposed
            if (operand.id in bundle.absorbedOpIds) return false
        }
    }
    // Also check fn.returns — a non-anchor absorbed op feeding a return
    // would also be lost.
    for (ret in fn.returns) {
        if (ret.id == bundle.anchorOpId) continue
        if (ret.id in bundle.absorbedOpIds) return false
    }
    return true
}

/**
 * Compute the set of [primalBody] param indices that the [gradientBody]
 * actually dereferences. Mirrors `PhiCalculus.computeGradientReads`'s
 * shape — duplicated here to keep the L3.2 coarsener self-contained
 * (the φ-calculus helper is `private`).
 *
 * For multi-result patterns (K > 1), the first K params of [gradientBody]
 * are upstream gradients; primal-operand params start at index K.
 */
internal fun computeGradientReads(gradientBody: DxirFunction, numUpstreamParams: Int): Set<Int> {
    val paramIds = gradientBody.params.map { it.id }
    val reads = LinkedHashSet<Int>()

    fun scanOp(op: DxirOp) {
        for (operand in op.operands) {
            val paramIdx = paramIds.indexOf(operand.id)
            if (paramIdx >= numUpstreamParams) reads += paramIdx - numUpstreamParams
        }
        for (region in op.regions) for (block in region.blocks) {
            for (bodyNode in block.body) if (bodyNode is DxirOp) scanOp(bodyNode)
        }
    }

    for (n in gradientBody.body) if (n is DxirOp) scanOp(n)
    for (r in gradientBody.returns) {
        val paramIdx = paramIds.indexOf(r.id)
        if (paramIdx >= numUpstreamParams) reads += paramIdx - numUpstreamParams
    }
    return reads
}

/**
 * v1 default registry: one coarsener per pattern. Adding a recognizer +
 * its coarsener = new file + one entry here.
 */
val defaultCoarseners: Map<String, PatternCoarsener> = mapOf(
    "FlashAttention" to PatternCoarsener { m -> coarsenFlashAttention(m as RecognitionMatch.FlashAttention) },
    "RmsNorm" to PatternCoarsener { m -> coarsenRmsNorm(m as RecognitionMatch.RmsNorm) },
)
