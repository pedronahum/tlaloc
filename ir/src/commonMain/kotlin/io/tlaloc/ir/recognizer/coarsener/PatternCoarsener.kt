package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * Layer 3 §0.4.252+ — interface for per-pattern VJP coarseners.
 *
 * A `PatternCoarsener` consumes a [RecognitionMatch] and produces a
 * [CoarsenedBundle] — everything the generic [coarsenRecognizedPatterns]
 * driver needs to splice an `OpKind.COARSENED` envelope into the outer
 * function in place of the matched sub-graph.
 *
 * Coarseners are registered by `RecognitionMatch.patternName`. Adding a
 * new pattern = new file with a `coarsenXxx(match)` function + one entry
 * in [defaultCoarseners].
 */
fun interface PatternCoarsener {
    /**
     * Build the coarsened envelope for [match], or return `null` to
     * decline the rewrite (e.g. the matched sub-graph has external
     * consumers the coarsener can't safely subsume).
     */
    fun coarsen(match: RecognitionMatch): CoarsenedBundle?
}

/**
 * Everything the generic coarsening driver needs to splice a
 * single-result `OpKind.COARSENED` op into the outer function.
 *
 * @property absorbedOpIds SSA ids of every op in the outer function this
 *   match subsumes. The driver skips these when rebuilding the function;
 *   they live only inside [primalBody] going forward.
 * @property anchorOpId The id of the op whose result the new COARSENED
 *   *replaces* — typically the last op of the matched sub-graph.
 *   Consumers that previously referenced this op now reference the
 *   COARSENED's first result.
 * @property outerOperands Outer-function nodes that become the COARSENED
 *   op's operands. Must align positionally with [primalBody].params.
 * @property primalBody The simplified primal — a free-standing
 *   [DxirFunction] whose params correspond to [outerOperands] and whose
 *   returns correspond to the COARSENED op's results.
 * @property gradientBody The analytical VJP. Signature:
 *   `(upstream_0, …, upstream_K-1, *primal_operands) → (d_operand_0, …)`.
 *   For single-result patterns (K=1) this is `(upstream, *primals) → grads`.
 * @property readsPrimalIndices Indices into [primalBody].params that the
 *   gradient body actually dereferences. Consumed by
 *   `DxirReverseTransform.computeUsedByAdjoint` to know which operand
 *   subgraphs must be cloned into the gradient-function scope.
 *
 * v1 limitation: only single-result COARSENED is supported. Multi-result
 * patterns (a recognized region producing multiple distinct outputs)
 * would need [anchorOpId] to be a list and consumers remapped per-result.
 */
data class CoarsenedBundle(
    val absorbedOpIds: Set<Int>,
    val anchorOpId: Int,
    val outerOperands: List<DxirNode>,
    val primalBody: DxirFunction,
    val gradientBody: DxirFunction,
    val readsPrimalIndices: Set<Int>,
)
