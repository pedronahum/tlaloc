@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.layer3

/**
 * Layer 3 §0.4.253+ — kernel-lowering across the per-target matrix.
 *
 * Walks the seven canonical device targets and shows what
 * `lowerKernelChoice` decides for each. Three outcomes:
 *
 * - **Fused kernel** — the COARSENED op gets a `kernel_descriptor` attr.
 *   StableHLO emit will materialize a `stablehlo.custom_call` to the
 *   vendor's fused implementation.
 * - **Decompose** — no kernel for this target (e.g. CPU). The COARSENED
 *   op is replaced by the inlined primal_body's three primitive ops.
 * - **Pass-through** — registry doesn't know about this pattern.
 *   Function unchanged. (Not exercised here; ships when registry is
 *   empty.)
 */

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeAll

fun main() {
    val sType = DxirType(F32, listOf(8, 8))
    val oType = DxirType(F32, listOf(8, 4))

    val attn = DxirBuilder.function("attn") {
        val q = param("Q", DxirType(F32, listOf(8, 4)))
        val k = param("K", DxirType(F32, listOf(4, 8)))
        val v = param("V", DxirType(F32, listOf(8, 4)))
        val qk = op(OpKind.MATMUL, listOf(q, k), sType)
        val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
        val out = op(OpKind.MATMUL, listOf(sm, v), oType)
        listOf(out)
    }
    val coarsened = coarsenRecognizedPatterns(attn, recognizeAll(attn))

    val targets = listOf(
        KernelTarget.NVIDIA_H100,
        KernelTarget.NVIDIA_A100,
        KernelTarget.AMD_MI300X,
        KernelTarget.GOOGLE_TPU_V4,
        KernelTarget.GOOGLE_TPU_V5E,
        KernelTarget.GOOGLE_TPU_V6E,
        KernelTarget.AWS_TRAINIUM2,
        KernelTarget.CPU_GENERIC,
    )

    println("Per-target kernel lowering decisions:")
    println("%-20s  %-32s  %s".format("target", "decision", "kernel"))
    println("-".repeat(70))
    for (t in targets) {
        val lowered = lowerKernelChoice(coarsened, t)
        val ops = lowered.body.filterIsInstance<DxirOp>()
        val coarsenedOp = ops.firstOrNull { it.op == OpKind.COARSENED }
        val tag = "${t.vendor}/${t.arch}"
        if (coarsenedOp != null) {
            val descriptor = coarsenedOp.attrs[KernelDescriptor.ATTR_KEY] as KernelDescriptor
            println("%-20s  %-32s  %s".format(tag, "fused (kernel custom_call)", descriptor.kernelName))
        } else {
            // No COARSENED → decomposed back to MATMUL → SOFTMAX → MATMUL.
            val opNames = ops.joinToString(",") { it.op.name }
            println("%-20s  %-32s  %s".format(tag, "decomposed", opNames))
        }
    }
}
