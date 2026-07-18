package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.RmsNormKernel
import io.tlaloc.ir.recognizer.kernel.RopeKernel
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v1.8 (§0.4.336) — **the recognizer-driven loop closes**: the
 * LlamaDecoder-medium forward pipeline
 * `recognize → coarsen → lowerKernelChoice(GB10, {RmsNorm→[RmsNormKernel]})
 * → decomposeCoarsened → toStablehlo` emits two typed-FFI
 * `stablehlo.custom_call @kptx_rms_norm` ops (pre-attn + pre-MLP norms),
 * and that emitter-produced program **executes on the GB10 with the
 * hand-written PTX rms_norm computing inside it**, agreeing with the
 * full-decompose program on the same PJRT-CUDA client.
 *
 * Unlike §0.4.331/§0.4.335 (hand-written MLIR isolating kernel
 * correctness) and §0.4.332 (probe handler isolating the attr pipeline),
 * nothing here is hand-written except the PTX itself: the MLIR comes
 * from the recognizer pipeline, the kernel launch from
 * [KptxKernelRegistry], the numbers from a real Llama-shaped forward.
 *
 * The kernel is the **eps-operand variant** of §0.4.331's rms_norm
 * (the coarsened RmsNorm is unweighted — Llama's weight-multiply lives
 * outside the recognized pattern — and eps arrives as the coarsener's
 * `[rows, 1]` operand, loaded per CTA): one CTA per token
 * row, 256 threads, strided fma sum-of-squares, shared-memory tree
 * reduction, `sqrt.rn`+`rcp.rn` normalizer, `out = x · r`.
 */
class LlamaDecoderKptxRmsNormE2ETest {

    private val kptxRegistry: Map<String, KernelTemplate> = mapOf("RmsNorm" to RmsNormKernel)


    /** GPU-less pin: the recognizer pipeline emits exactly two typed-FFI
     * `@kptx_rms_norm` custom_calls on GB10 and zero on CPU_GENERIC. */
    @Test
    fun emitsTwoKptxRmsNormCustomCallsOnGb10() {
        val gb10 = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        ).toStablehlo()
        assertEquals(
            2,
            Regex("stablehlo\\.custom_call @kptx_rms_norm\\(").findAll(gb10).count(),
            "expected the pre-attn and pre-MLP RmsNorms to lower to @kptx_rms_norm",
        )
        assertTrue(
            gb10.contains("api_version = 4 : i32"),
            "KPTX custom_calls must use the typed-FFI convention",
        )

        val cpu = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        ).toStablehlo()
        assertEquals(0, Regex("custom_call").findAll(cpu).count(), "CPU_GENERIC must fully decompose")
    }

    /** GPU-gated: the emitter-produced GB10 program runs with the PTX
     * kernel inside and agrees with the full-decompose program. */
    @Test
    fun kptxLoweredForwardAgreesWithDecomposePath() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxTestKernels.ensureRmsNormRegistered(pluginPath)

        val kptxFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        )
        val decomposeFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        )
        val inputs = llamaSynthesizeInputs(seed = 42L, fn = kptxFn)

        val kptxOut: List<FloatArray>
        val decomposeOut: List<FloatArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            kptxOut = session.runOn(kptxFn, inputs)
            decomposeOut = session.runOn(decomposeFn, inputs)
        }

        assertEquals(decomposeOut.size, kptxOut.size)
        var maxRel = 0f
        for (o in kptxOut.indices) {
            for (i in kptxOut[o].indices) {
                val d = abs(kptxOut[o][i] - decomposeOut[o][i])
                maxRel = max(maxRel, d / max(abs(decomposeOut[o][i]), 1e-6f))
            }
        }
        println(
            "[kptx-llama-e2e] LlamaDecoder-medium forward, 2×@kptx_rms_norm inside: " +
                "loss kptx=${kptxOut[0][0]}, decompose=${decomposeOut[0][0]}, max rel=$maxRel",
        )
        assertTrue(maxRel <= 1e-4f, "kptx-lowered forward diverges from decompose path: max rel $maxRel")
    }

    /** §0.4.349 — two kernel families claimed at once: RmsNorm ×2 + RoPE ×1.
     * Tolerance is wider than the rms-only test: the RoPE kernel's
     * `sin.approx`/`cos.approx` (PTX's only sin/cos) differ from XLA's
     * precise routines at ~1e-6 absolute, which the downstream
     * softmax/CE amplifies into the loss. */
    @Test
    fun kptxRmsNormPlusRopeForwardAgreesWithDecomposePath() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxTestKernels.ensureRmsNormRegistered(pluginPath)
        KptxTestKernels.ensureRopeRegistered(pluginPath)

        val registry: Map<String, KernelTemplate> =
            mapOf("RmsNorm" to RmsNormKernel, "Rope" to RopeKernel)
        val kptxFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, registry,
        )
        val mlir = kptxFn.toStablehlo()
        assertEquals(2, Regex("custom_call @kptx_rms_norm\\(").findAll(mlir).count())
        assertEquals(1, Regex("custom_call @kptx_rope\\(").findAll(mlir).count())

        val decomposeFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, registry,
        )
        val inputs = llamaSynthesizeInputs(seed = 42L, fn = kptxFn)

        val kptxOut: List<FloatArray>
        val decomposeOut: List<FloatArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            kptxOut = session.runOn(kptxFn, inputs)
            decomposeOut = session.runOn(decomposeFn, inputs)
        }

        var maxRel = 0f
        for (o in kptxOut.indices) {
            for (i in kptxOut[o].indices) {
                val d = abs(kptxOut[o][i] - decomposeOut[o][i])
                maxRel = max(maxRel, d / max(abs(decomposeOut[o][i]), 1e-6f))
            }
        }
        println(
            "[kptx-llama-e2e] LlamaDecoder-medium forward, 2×rms_norm + 1×rope inside: " +
                "loss kptx=${kptxOut[0][0]}, decompose=${decomposeOut[0][0]}, max rel=$maxRel",
        )
        assertTrue(maxRel <= 1e-3f, "kptx rms+rope forward diverges from decompose path: max rel $maxRel")
    }
}
