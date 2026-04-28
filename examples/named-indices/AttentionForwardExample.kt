@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.named_indices

/**
 * Layer 1 §0.4.241+ — attention-block forward pass with named indices.
 *
 * Demonstrates how named axes anchor a multi-step typed computation: each
 * tensor carries the semantic name of every axis (`Batch`, `SeqLen`,
 * `Heads`, `Dim`), so an experienced reader can tell at a glance what each
 * matmul contracts over without consulting the implementation.
 *
 * v1 simplification: this example expresses just the QK^T attention-score
 * matrix (the first contraction in a transformer attention block), where
 * the named structure is most informative. The full softmax + V matmul
 * follow the same pattern but require op support that lives outside Layer
 * 1's scope (e.g. `softmax` over a named axis is a v2 enhancement).
 *
 * Copy into a project that depends on `io.tlaloc:core` to run.
 */

import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.Dim
import io.tlaloc.core.F32
import io.tlaloc.core.Heads
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.contract

/**
 * Compute query × key^T for one head of one batch element.
 *
 * Real transformer code carries the `Batch` and `Heads` axes through every
 * tensor and uses batched matmul to contract. Layer 1's v1 supports rank-2
 * × rank-2; rank-3 batched named contraction is a v1 follow-up
 * (tracked in the audit's open-issues section).
 *
 * Returns the rank-2 attention-score matrix typed
 * `DTensor<Rank2<Named<SeqLen, Sym>, Named<SeqLen, Sym>>, F32>`. Both axes
 * carry the *same* name `SeqLen` (queries' positions and keys' positions);
 * v1 named-contract over a self-named axis isn't supported — for the full
 * pipeline you'd promote one axis to a distinct name like `KeySeqLen` to
 * disambiguate.
 */
fun attentionScores(
    queries: DTensor<Rank2<Named<SeqLen, Sym>, Named<Dim, Sym>>, F32>,
    keysTransposed: DTensor<Rank2<Named<Dim, Sym>, Named<SeqLen, Sym>>, F32>,
): DTensor<Rank2<Named<SeqLen, Sym>, Named<SeqLen, Sym>>, F32> {
    // Contract over the shared `Dim` axis: queries.axis-1 × keysTransposed.axis-0.
    return queries contract keysTransposed
}

fun main() {
    // Single batch element, single head: 4 query positions × 8 features.
    val queries: DTensor<Rank2<Named<SeqLen, Sym>, Named<Dim, Sym>>, F32> =
        Tensors.f32Matrix<Named<SeqLen, Sym>, Named<Dim, Sym>>(
            d0 = 4, d1 = 8,
            data = FloatArray(4 * 8) { it.toFloat() / 32f },
        )

    // 8 features × 4 key positions (already transposed for the contraction).
    val keysT: DTensor<Rank2<Named<Dim, Sym>, Named<SeqLen, Sym>>, F32> =
        Tensors.f32Matrix<Named<Dim, Sym>, Named<SeqLen, Sym>>(
            d0 = 8, d1 = 4,
            data = FloatArray(8 * 4) { (it + 1).toFloat() / 32f },
        )

    val scores = attentionScores(queries, keysT)
    println("Attention scores dims: ${scores.dims.toList()}  (expected [4, 4])")
    val flat = scores.hostF32()
    for (q in 0 until scores.dims[0]) {
        val row = (0 until scores.dims[1])
            .joinToString("  ") { k -> "%.4f".format(flat[q * scores.dims[1] + k]) }
        println("  q=$q  $row")
    }

    // Type-system note: the Batch and Heads axes don't appear in this rank-2
    // example, but a real implementation would carry them as outer axes:
    //
    //   val q: DTensor<Rank4<Named<Batch, ..>, Named<Heads, ..>,
    //                        Named<SeqLen, ..>, Named<Dim, ..>>, F32>
    //
    // A v2 enhancement adds rank-3 / rank-4 batched named contraction. The
    // audit's open-issues section in docs/audits/named_indices_audit.md
    // tracks this.
    @Suppress("UNUSED_VARIABLE")
    val _typeSystemDocumentation = listOf(Batch.name, Heads.name)
}
