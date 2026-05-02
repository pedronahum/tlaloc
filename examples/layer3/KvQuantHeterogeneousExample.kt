@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.layer3

/**
 * Layer 3 §0.4.257+ — best-effort KV-quant across heterogeneous targets.
 *
 * The user requests FP8 KV cache. Across H100 / A100 / Trainium2 / TPU
 * v5e, only some targets can honour the request:
 *
 * - **H100, Trainium2** — native FP8 support → annotation applied.
 * - **A100, TPU v5e** — no FP8 path → silently declined with a
 *   structured diagnostic explaining why. The COARSENED op stays
 *   un-annotated; downstream codegen falls back to the kernel's default
 *   precision.
 *
 * The user's compile call doesn't fail — heterogeneous targets are a
 * first-class deployment shape. The diagnostic stream is what surfaces
 * the per-target accept/decline decision.
 */

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.applyKvQuantWithDiagnostics
import io.tlaloc.ir.recognizer.recognizeAll

fun main() {
    val attn = DxirBuilder.function("attn") {
        val q = param("Q", DxirType(F32, listOf(8, 4)))
        val k = param("K", DxirType(F32, listOf(4, 8)))
        val v = param("V", DxirType(F32, listOf(8, 4)))
        val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(8, 8)))
        val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(8, 8)))
        val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(8, 4)))
        listOf(out)
    }
    val coarsened = coarsenRecognizedPatterns(attn, recognizeAll(attn))
    val request = KvQuantConfig.FP8_PER_HEAD

    val targets = listOf(
        KernelTarget.NVIDIA_H100,
        KernelTarget.NVIDIA_A100,
        KernelTarget.AWS_TRAINIUM2,
        KernelTarget.GOOGLE_TPU_V5E,
    )

    println("Requesting ${request.dtype.nameTag} (${request.scaleStrategy}) KV cache:")
    println()
    for (t in targets) {
        val lowered = lowerKernelChoice(coarsened, t)
        val (afterQuant, diagnostics) = applyKvQuantWithDiagnostics(lowered, request)
        val co = afterQuant.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val attached = co.attrs[KvQuantConfig.ATTR_KEY] != null
        val tag = "${t.vendor}/${t.arch}"
        if (attached) {
            println("  %-22s  ✓ accepted (will quantize KV cache to ${request.dtype.nameTag})".format(tag))
        } else {
            val why = diagnostics.singleOrNull()?.reason ?: "no diagnostic"
            println("  %-22s  ✗ declined — %s".format(tag, why))
        }
    }
}
