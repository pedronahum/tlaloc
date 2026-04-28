@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.named_indices

/**
 * Layer 1 §0.4.241+ + Layer 1.5 §0.4.242+ — attention-block forward pass
 * with named indices.
 *
 * Demonstrates how named axes anchor a multi-step typed computation. As of
 * Layer 1.5 (§0.4.242), Tlaloc's `contract` operator supports the full
 * rank-4 attention shape with two batching axes (`Batch`, `Heads`) and a
 * contracting axis (`Dim`) — the typed surface matches a transformer's
 * attention layer exactly, no rank-2 simplification needed.
 *
 * The full softmax + V matmul follow the same pattern but require op
 * support that lives outside Layer 1's scope (e.g. `softmax` over a named
 * axis is a v2 enhancement).
 *
 * Copy into a project that depends on `io.tlaloc:core` to run.
 */

import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.Dim
import io.tlaloc.core.F32
import io.tlaloc.core.Heads
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank4
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.contract

/**
 * QK^T attention-score axis: an alias for the *key* sequence dimension to
 * disambiguate from queries' [SeqLen] axis. (A self-name conflict —
 * `(Batch, Heads, SeqLen, Dim) × (Batch, Heads, Dim, SeqLen)` — would not
 * type-check because the contracting axis would also match positionally
 * with the surviving query-side axis. Distinct names are the natural fix.)
 */
object KeySeqLen : IndexName {
    override val name = "key_seq"
}

/**
 * Attention-score forward pass: `softmax(QK^T / sqrt(d))` reduced to just
 * the QK^T core (the other pieces are ops outside Layer 1's scope).
 *
 * Type-level structure:
 *
 *     Q : (Batch, Heads, SeqLen, Dim)
 *     K^T: (Batch, Heads, Dim,    KeySeqLen)
 *     →    (Batch, Heads, SeqLen, KeySeqLen)
 *
 * Shared axes: Batch (positions 0,0 → batching), Heads (positions 1,1 →
 * batching), Dim (positions 3,2 → contracting). The K2 plugin's lowering
 * emits `stablehlo.dot_general` with `lhs_batching_dims = [0, 1]` and
 * `lhs_contracting_dims = [3]` automatically.
 */
fun attentionScores(
    queries: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<Dim, Sym>>, F32>,
    keysTransposed: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<Dim, Sym>, Named<KeySeqLen, Sym>>, F32>,
): DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<KeySeqLen, Sym>>, F32> {
    return queries contract keysTransposed
}

fun main() {
    val nb = 1
    val nh = 2
    val tq = 4
    val tk = 4
    val d = 8

    val queries: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<Dim, Sym>>, F32> =
        DTensor(
            io.tlaloc.core.HostF32Storage(FloatArray(nb * nh * tq * d) { it.toFloat() / 64f }),
            intArrayOf(nb, nh, tq, d),
            F32,
        )
    val keysT: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<Dim, Sym>, Named<KeySeqLen, Sym>>, F32> =
        DTensor(
            io.tlaloc.core.HostF32Storage(FloatArray(nb * nh * d * tk) { (it + 1).toFloat() / 64f }),
            intArrayOf(nb, nh, d, tk),
            F32,
        )

    val scores = attentionScores(queries, keysT)
    println("Attention scores dims: ${scores.dims.toList()}  (expected [$nb, $nh, $tq, $tk])")
    val flat = scores.hostF32()
    for (head in 0 until nh) {
        println("--- head $head ---")
        for (q in 0 until tq) {
            val row = (0 until tk)
                .joinToString("  ") { k ->
                    val idx = head * tq * tk + q * tk + k
                    "%.4f".format(flat[idx])
                }
            println("  q=$q  $row")
        }
    }
}
