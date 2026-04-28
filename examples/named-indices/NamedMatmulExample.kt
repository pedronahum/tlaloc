@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.named_indices

/**
 * Layer 1 §0.4.241+ — basic named-index contraction example.
 *
 * Demonstrates the simplest non-trivial use of named indices: a rank-2 matrix
 * multiply where the contracted axis is named (`SeqLen`) and the surviving
 * axes carry distinct names (`Batch` × `Hidden`).
 *
 * The K2 plugin is **not** required to make this code type-check — Kotlin's
 * native type-inference machinery enforces the contraction structure (Refined
 * Option A; see `docs/audits/named_indices_audit.md`). The plugin's job is to
 * lower `contract` calls inside `grad { }` lambdas to DXIR; outside lambdas
 * (as here), the runtime body in `core/ops/NamedOps.kt` executes.
 *
 * Copy into a project that depends on `io.tlaloc:core` to run.
 */

import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.contract

fun main() {
    // Activations: 2 (batch) × 3 (sequence positions). Named axes anchor on
    // the per-axis semantics — the type system records that axis 0 is the
    // batch dimension and axis 1 is the sequence dimension.
    val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> =
        Tensors.f32Matrix<Named<Batch, Sym>, Named<SeqLen, Sym>>(
            d0 = 2, d1 = 3,
            data = floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f,
            ),
        )

    // Weights: 3 (sequence positions, must match activations' axis 1) ×
    // 4 (hidden). Named axis 0 is `SeqLen` — same name as activations' axis 1
    // — that's the contraction signal Kotlin picks up.
    val weights: DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> =
        Tensors.f32Matrix<Named<SeqLen, Sym>, Named<Hidden, Sym>>(
            d0 = 3, d1 = 4,
            data = floatArrayOf(
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f,
            ),
        )

    // Contract over the shared `SeqLen` axis. Result type is inferred by
    // Kotlin as `DTensor<Rank2<Named<Batch, Sym>, Named<Hidden, Sym>>, F32>`
    // — the surviving axes are batch (from activations) and hidden (from
    // weights).
    val output: DTensor<Rank2<Named<Batch, Sym>, Named<Hidden, Sym>>, F32> =
        activations contract weights

    println("Output dims: ${output.dims.toList()}  (expected [2, 4])")
    println("Output values:")
    val flat = output.hostF32()
    for (b in 0 until output.dims[0]) {
        val row = (0 until output.dims[1])
            .joinToString("  ") { h -> "%.3f".format(flat[b * output.dims[1] + h]) }
        println("  [$b] $row")
    }
}
