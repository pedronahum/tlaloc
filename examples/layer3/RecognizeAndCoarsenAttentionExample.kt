@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.layer3

/**
 * Layer 3 §0.4.250+ — recognize + coarsen the canonical FlashAttention shape.
 *
 * Walks the smallest end-to-end loop of the L3 pipeline:
 *
 * 1. Build a hand-crafted DXIR function carrying the user-written
 *    `MATMUL → SOFTMAX → MATMUL` shape (the typical unfused attention
 *    forward).
 * 2. Run `recognizeAll` — finds one `RecognitionMatch.FlashAttention`.
 * 3. Run `coarsenRecognizedPatterns` — rewrites the function so the three
 *    matched ops are absorbed into a single `OpKind.COARSENED` op
 *    carrying the analytical primal_body + gradient_body.
 *
 * The output is a function with one COARSENED op, ready for L3.3 kernel
 * lowering. Copy-paste runnable; depends on the `:ir` module.
 */

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeAll

fun main() {
    // Rank-2 attention. Q [M, K] · K [K, N] → S [M, N];
    // SOFTMAX(S) [M, N]; · V [N, D_v] → O [M, D_v].
    val qType = DxirType(F32, listOf(8, 4))   // [M=8, K=4]
    val kType = DxirType(F32, listOf(4, 8))   // [K=4, N=8]
    val vType = DxirType(F32, listOf(8, 4))   // [N=8, D_v=4]
    val sType = DxirType(F32, listOf(8, 8))   // [M, N]
    val oType = DxirType(F32, listOf(8, 4))   // [M, D_v]

    // 1. The user's hand-written attention forward.
    val user = DxirBuilder.function("attention") {
        val q = param("Q", qType)
        val k = param("K", kType)
        val v = param("V", vType)
        val qk = op(OpKind.MATMUL, listOf(q, k), sType)
        val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
        val out = op(OpKind.MATMUL, listOf(sm, v), oType)
        listOf(out)
    }
    println("Before:  ${user.body.filterIsInstance<DxirOp>().map { it.op }}")
    // → Before:  [MATMUL, SOFTMAX, MATMUL]

    // 2. Recognize compound forms.
    val matches = recognizeAll(user)
    println("Matches: ${matches.map { it.patternName }}")
    // → Matches: [FlashAttention]

    // 3. Coarsen — three ops become one COARSENED.
    val coarsened = coarsenRecognizedPatterns(user, matches)
    val ops = coarsened.body.filterIsInstance<DxirOp>()
    println("After:   ${ops.map { it.op }}")
    // → After:   [COARSENED]

    val co = ops.single()
    val primal = co.attrs["primal_body"] as io.tlaloc.ir.DxirFunction
    val gradient = co.attrs["gradient_body"] as io.tlaloc.ir.DxirFunction
    @Suppress("UNCHECKED_CAST")
    val reads = co.attrs["reads_primal_indices"] as Set<Int>

    println("\nCOARSENED envelope:")
    println("  operands:           ${co.operands.size} (Q, K, V)")
    println("  result type:        ${co.type}")
    println("  primal_body name:   ${primal.name}")
    println("  primal_body ops:    ${primal.body.filterIsInstance<DxirOp>().map { it.op }}")
    println("  gradient_body name: ${gradient.name}")
    println("  gradient signature: ${gradient.params.size} params (1 dO + 3 primal) → ${gradient.returns.size} grads")
    println("  reads_primal:       $reads (Q=0, K=1, V=2 — all referenced)")
}
