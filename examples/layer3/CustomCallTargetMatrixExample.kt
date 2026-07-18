@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.layer3

/**
 * Layer 3 §0.4.325 — **the artifact-level evidence that Tlaloc decides
 * per-device kernel choice upstream of the runtime, in a way JAX/XLA
 * does not.**
 *
 * Same Kotlin source — a recognizable `MATMUL → SOFTMAX → MATMUL`
 * attention block — compiled through the kernel-lowering pipeline
 * (`recognize → coarsen → lowerKernelChoice(target) → decomposeCoarsened
 * → toStablehlo`) for four device targets. Each target produces a
 * **structurally different MLIR artifact**, naming the per-vendor
 * fused-attention kernel:
 *
 * | Target            | Custom call emitted                   |
 * |-------------------|---------------------------------------|
 * | NVIDIA GB10       | `stablehlo.custom_call @flash_attn_v3` |
 * | GOOGLE TPU v6e    | `stablehlo.custom_call @tpu_pallas_flash_attention` |
 * | AWS Trainium2     | `stablehlo.custom_call @nki_flash_attention` |
 * | CPU_GENERIC       | (none — fully decomposed)             |
 *
 * Run via: `./gradlew :examples:run --args='CustomCallTargetMatrixExample'`
 * (or copy + paste the relevant `main()` body into a fresh project).
 *
 * # Why this is the differentiator
 *
 * JAX/XLA emits the same StableHLO regardless of target; XLA's
 * auto-fusion picks one codegen path at runtime. **Tlaloc's artifact
 * carries the decision upstream of the runtime** — the bytes shipped to
 * GB10 are different from the bytes shipped to TPU. JAX literally
 * cannot consume Tlaloc's GB10 artifact, because `@flash_attn_v3` isn't
 * a custom-call symbol JAX/XLA ships (see
 * `JaxRejectsTlalocCustomCallMlirTest` for the negative pin).
 *
 * The `CPU_GENERIC` row demonstrates the always-available fallback —
 * when no per-target kernel exists, the COARSENED's analytical
 * `primal_body` is inlined, producing portable StableHLO.
 *
 * # What's NOT here
 *
 * - **No runtime dispatch.** Phase 2 (PJRT custom-call symbol
 *   registration via FFM) closes the loop; until then, the artifact
 *   carries the decision but the runtime executes the decomposed form.
 * - **Only FlashAttention has a kernel template in v1.** The other
 *   recognized patterns (RmsNorm / RoPE / TransformerMLP / CrossEntropy
 *   / GroupedQueryAttention) coarsen but decompose at lowering time.
 *   Per-pattern kernel template files land alongside future vendor
 *   integrations.
 */

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.stablehlo.toStablehlo

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

    val targets = listOf(
        KernelTarget.NVIDIA_GB10,
        KernelTarget.NVIDIA_H100,
        KernelTarget.GOOGLE_TPU_V6E,
        KernelTarget.AWS_TRAINIUM2,
        KernelTarget.CPU_GENERIC,
    )

    println("=".repeat(78))
    println("Per-target artifact divergence — same Kotlin source, four MLIR shapes")
    println("=".repeat(78))
    println()

    val artifacts = mutableMapOf<KernelTarget, String>()
    for (t in targets) {
        val coarsened = coarsenRecognizedPatterns(attn, recognizeAll(attn))
        val lowered = lowerKernelChoice(coarsened, t)
        val decomposed = decomposeCoarsened(lowered)
        val mlir = decomposed.toStablehlo("")
        artifacts[t] = mlir

        val customCallLines = mlir.lineSequence()
            .filter { "stablehlo.custom_call" in it }
            .map { it.trim() }
            .toList()
        val coarsenedOpsRemaining = decomposed.body.filterIsInstance<DxirOp>()
            .count { it.op == OpKind.COARSENED }

        val tag = "${t.vendor}/${t.arch}"
        println("--- target: $tag")
        if (customCallLines.isEmpty()) {
            println("    (no custom_call — fully decomposed to StableHLO primitives)")
        } else {
            for (line in customCallLines) {
                println("    $line")
            }
        }
        val totalOps = decomposed.body.filterIsInstance<DxirOp>().size
        println("    ops in body: $totalOps (kernel-annotated COARSENED: $coarsenedOpsRemaining)")
        println()
    }

    // Pairwise byte-diff summary — proves the artifacts truly differ.
    println("=".repeat(78))
    println("Pairwise byte-diff (same Kotlin source, four artifacts)")
    println("=".repeat(78))
    val keys = targets.toList()
    for (i in keys.indices) {
        for (j in (i + 1) until keys.size) {
            val a = keys[i]; val b = keys[j]
            val same = artifacts[a] == artifacts[b]
            val tagA = "${a.vendor}/${a.arch}".padEnd(20)
            val tagB = "${b.vendor}/${b.arch}".padEnd(20)
            println("  $tagA vs $tagB → ${if (same) "IDENTICAL" else "DIFFERENT"}")
        }
    }
    println()
    println("See also: harness/python/run_jax_compile_check.py +")
    println("          benchmarks .../JaxRejectsTlalocCustomCallMlirTest.kt")
    println("for the matching negative pin (JAX cannot compile the GB10 artifact).")
}
