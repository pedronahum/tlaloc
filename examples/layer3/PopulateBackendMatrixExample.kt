@file:Suppress("MatchingDeclarationName", "unused")

package io.tlaloc.examples.layer3

/**
 * Layer 3 §0.4.258+ — populate the manifest's `backendMatrix` across all
 * seven device targets and dump it as JSON.
 *
 * This is the form the runtime side ([com.netflix.maestro.engine.tlaloc.TlalocPodSpecBuilder])
 * consumes — given the JSON below + the cluster's (vendor, arch), it
 * picks the best-matching row and translates it into K8s pod-spec
 * fields (nodeSelector + accelerators + gpu).
 *
 * The output also illustrates the cost-ordered relative ranking the
 * scheduler can use when choosing among targets at deployment time.
 */

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.maestro.populateBackendMatrix

fun main() {
    val attn = DxirBuilder.function("attn") {
        val q = param("Q", DxirType(F32, listOf(64, 64)))
        val k = param("K", DxirType(F32, listOf(64, 64)))
        val v = param("V", DxirType(F32, listOf(64, 64)))
        val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(64, 64)))
        val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(64, 64)))
        val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(64, 64)))
        listOf(out)
    }

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

    val matrix = populateBackendMatrix(attn, targets, kvQuant = KvQuantConfig.FP8_PER_HEAD)

    println("Backend matrix rows (sorted by cost):")
    println("%-22s  %-32s  %-12s  %s".format("target", "kernel", "kv_quant", "cost (us)"))
    println("-".repeat(80))
    val sorted = matrix.sortedBy { it.costMicroseconds ?: Double.POSITIVE_INFINITY }
    for (row in sorted) {
        val tag = "${row.vendor}/${row.arch}"
        val kernel = row.kernelName ?: "(decompose)"
        val kvq = row.kvQuantDtype ?: "—"
        val cost = row.costMicroseconds?.let { "%.2f".format(it) } ?: "—"
        println("%-22s  %-32s  %-12s  %s".format(tag, kernel, kvq, cost))
    }

    // The matrix serialises to JSON for transport to the runtime side.
    println("\nSerialized manifest fragment (truncated):")
    val firstThree = matrix.take(3).joinToString(",", "[", ",...]") { it.toJson() }
    println(firstThree)

    println("\nThis JSON populates `manifest.backendMatrix`. The runtime's")
    println("TlalocPodSpecBuilder reads it + the cluster's (vendor, arch),")
    println("then translates the picked row into K8s nodeSelector +")
    println("accelerator labels + gpu count at job-launch time.")
}
