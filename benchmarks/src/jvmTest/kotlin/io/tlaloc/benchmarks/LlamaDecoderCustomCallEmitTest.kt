package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.stablehlo.toStablehlo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.325 — **The "Tlaloc does something JAX doesn't" pin.**
 *
 * Same `LlamaDecoderPrimal` Kotlin source compiled for four
 * [KernelTarget]s through the kernel-lowering pipeline
 * (`recognize → coarsen → lowerKernelChoice(target) → decomposeCoarsened
 * → toStablehlo`) produces **four different MLIR artifacts**, each
 * carrying the per-device kernel decision baked into a
 * `stablehlo.custom_call @<kernel_name>` op:
 *
 * | Target            | Custom-call kernel               | Count |
 * |-------------------|----------------------------------|-------|
 * | NVIDIA GB10       | `flash_attn_v3`                  | 1     |
 * | GOOGLE TPU v6e    | `tpu_pallas_flash_attention`     | 1     |
 * | AWS Trainium2     | `nki_flash_attention`            | 1     |
 * | CPU_GENERIC       | (none — full decompose)          | 0     |
 *
 * The other recognized patterns (RmsNorm × 2, RoPE, TransformerMLP,
 * CrossEntropy) **do not** have entries in `defaultKernelTemplates`
 * today, so their COARSENED ops decompose to primitives even on GPU
 * targets. That's the "1 custom_call" count above — only FlashAttention
 * has a kernel template plumbed in v1.
 *
 * # Why this matters
 *
 * JAX/XLA does not pick across kernel families per-device. JAX emits the
 * same StableHLO regardless of target; XLA's auto-fusion later picks
 * one codegen path. **Tlaloc's artifact carries the per-device decision
 * upstream of the runtime** — the MLIR shipped to GB10 is structurally
 * different from the MLIR shipped to TPU. That's a property of the
 * artifact, not of the runtime.
 *
 * §0.4.310's [LlamaDecoderCoarseningEmitDiagnosticTest] still pins the
 * **legacy** decompose-only pipeline (0 custom_calls everywhere); that
 * test stays green because the kernel-lowering pipeline lives in a
 * separate helper and is opt-in. The benchmark inhabitants
 * (LlamaDecoderPjrtFfmBenchTest, LlamaDecoderIreeBenchmarkTest) keep
 * using the legacy path until Phase 2 wires runtime kernel registration.
 */
class LlamaDecoderCustomCallEmitTest {

    @Test
    fun gb10TargetEmitsFlashAttnV3CustomCall() {
        val mlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10,
        ).toStablehlo("")

        val flashAttnV3 = mlir.lineSequence().count { "stablehlo.custom_call @flash_attn_v3" in it }
        val totalCustomCalls = mlir.lineSequence().count { "stablehlo.custom_call" in it }

        assertEquals(
            1, flashAttnV3,
            "GB10 target should emit exactly one `stablehlo.custom_call @flash_attn_v3` op " +
                "(the FlashAttention COARSENED's kernel descriptor on Grace-Blackwell)",
        )
        assertEquals(
            1, totalCustomCalls,
            "v1 has only FlashAttention in defaultKernelTemplates — other patterns decompose",
        )
    }

    @Test
    fun tpuV6eTargetEmitsPallasFlashAttentionCustomCall() {
        val mlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.GOOGLE_TPU_V6E,
        ).toStablehlo("")

        assertEquals(
            1, mlir.lineSequence().count { "stablehlo.custom_call @tpu_pallas_flash_attention" in it },
            "TPU v6e target should emit `stablehlo.custom_call @tpu_pallas_flash_attention`",
        )
        assertEquals(
            0, mlir.lineSequence().count { "stablehlo.custom_call @flash_attn" in it },
            "TPU target should not emit NVIDIA's flash_attn_* kernel name",
        )
    }

    @Test
    fun trainium2TargetEmitsNkiFlashAttentionCustomCall() {
        val mlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.AWS_TRAINIUM2,
        ).toStablehlo("")

        assertEquals(
            1, mlir.lineSequence().count { "stablehlo.custom_call @nki_flash_attention" in it },
            "Trainium2 target should emit `stablehlo.custom_call @nki_flash_attention` (NKI)",
        )
    }

    @Test
    fun cpuGenericTargetEmitsZeroCustomCalls() {
        val mlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC,
        ).toStablehlo("")

        assertEquals(
            0, mlir.lineSequence().count { "stablehlo.custom_call" in it },
            "CPU_GENERIC should fully decompose — no fused kernel anywhere",
        )
    }

    /**
     * The headline assertion: same Kotlin source compiles to **four
     * different artifacts**, each naming a different per-device kernel.
     * Pairwise inequality on the emitted MLIR is the cleanest way to
     * pin the artifact-level divergence the README claims.
     */
    @Test
    fun sameSourceProducesDifferentArtifactsAcrossTargets() {
        val gb10 = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10,
        ).toStablehlo("")
        val tpu = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.GOOGLE_TPU_V6E,
        ).toStablehlo("")
        val trainium = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.AWS_TRAINIUM2,
        ).toStablehlo("")
        val cpu = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC,
        ).toStablehlo("")

        val artifacts = mapOf(
            "GB10" to gb10, "TPU_V6E" to tpu, "TRAINIUM2" to trainium, "CPU" to cpu,
        )
        val keys = artifacts.keys.toList()
        for (i in keys.indices) {
            for (j in (i + 1) until keys.size) {
                val a = keys[i]; val b = keys[j]
                assertTrue(
                    artifacts[a] != artifacts[b],
                    "$a and $b artifacts must differ — same source, different per-device decisions",
                )
            }
        }

        // Spot-check: each artifact contains its target's kernel name (or
        // is the no-custom-call CPU artifact).
        assertTrue("@flash_attn_v3" in gb10, "GB10 artifact must name flash_attn_v3")
        assertTrue("@tpu_pallas_flash_attention" in tpu, "TPU artifact must name tpu_pallas_flash_attention")
        assertTrue("@nki_flash_attention" in trainium, "Trainium artifact must name nki_flash_attention")
        assertTrue("stablehlo.custom_call" !in cpu, "CPU artifact must contain no custom_call")
    }

    /**
     * Pin the IR-level shape post-pipeline: kernel-annotated COARSENED
     * survives `decomposeCoarsened`, un-annotated COARSENED gets inlined.
     * GB10 should leave exactly one COARSENED in the body (the
     * FlashAttention one carrying the kernel_descriptor); the other 5
     * recognized patterns decompose.
     */
    @Test
    fun gb10IrHasOneKernelAnnotatedCoarsened() {
        val fn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10,
        )
        val ops = fn.body.filterIsInstance<DxirOp>()
        val coarsened = ops.filter { it.op == OpKind.COARSENED }
        assertEquals(
            1, coarsened.size,
            "GB10 pipeline should leave exactly 1 COARSENED (FlashAttention with kernel_descriptor)",
        )
        val descriptor = coarsened.single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertTrue(descriptor != null, "the surviving COARSENED must carry a KernelDescriptor")
        assertEquals("flash_attn_v3", descriptor.kernelName)
        assertEquals("nvidia", descriptor.vendor)
        assertEquals("gb10", descriptor.targetArch)
    }
}
