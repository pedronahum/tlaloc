package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Layer 4.1 §0.4.261 — pin the StableHLO `custom_call` shape emitted for
 * a COARSENED op carrying a [KernelDescriptor]. Round-trip through
 * `stablehlo-translate` is deferred to L4 follow-up (see
 * `setup-dgx-spark-userspace.sh`); these tests pin emitted text only.
 */
class CoarsenedCustomCallTest {

    private val mDim = 8
    private val kDim = 4
    private val nDim = 8
    private val dvDim = 4

    private val qType = DxirType(F32, listOf(mDim, kDim))
    private val kType = DxirType(F32, listOf(kDim, nDim))
    private val vType = DxirType(F32, listOf(nDim, dvDim))
    private val sType = DxirType(F32, listOf(mDim, nDim))
    private val oType = DxirType(F32, listOf(mDim, dvDim))

    private fun buildCoarsenedFn(): DxirFunction {
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val out = op(OpKind.MATMUL, listOf(sm, v), oType)
            listOf(out)
        }
        return coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))
    }

    private fun emitFor(target: KernelTarget): String =
        lowerKernelChoice(buildCoarsenedFn(), target).toStablehlo()

    @Test
    fun h100EmitsFlashAttnV3CustomCall() {
        val mlir = emitFor(KernelTarget.NVIDIA_H100)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v3"), mlir)
        assertTrue(
            mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, fp8_e5m2, int8]"),
            mlir,
        )
        assertTrue(mlir.contains("has_side_effect = false"), mlir)
    }

    @Test
    fun a100EmitsFlashAttnV2CustomCall() {
        val mlir = emitFor(KernelTarget.NVIDIA_A100)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v2"), mlir)
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, int8]"), mlir)
    }

    @Test
    fun mi300xEmitsFlashAttnAmdCustomCall() {
        val mlir = emitFor(KernelTarget.AMD_MI300X)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_amd"), mlir)
    }

    @Test
    fun tpuV5eEmitsTpuPallasFlashAttention() {
        val mlir = emitFor(KernelTarget.GOOGLE_TPU_V5E)
        assertTrue(mlir.contains("stablehlo.custom_call @tpu_pallas_flash_attention"), mlir)
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, int8]"), mlir)
    }

    @Test
    fun tpuV6eIncludesFp8InSupportedKvDtypes() {
        val mlir = emitFor(KernelTarget.GOOGLE_TPU_V6E)
        assertTrue(mlir.contains("stablehlo.custom_call @tpu_pallas_flash_attention"), mlir)
        // v6e gets the FP8 kernel variant; older v4/v5 stay BF16/INT8.
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, int8]"), mlir)
    }

    @Test
    fun trainium2EmitsNkiFlashAttention() {
        val mlir = emitFor(KernelTarget.AWS_TRAINIUM2)
        assertTrue(mlir.contains("stablehlo.custom_call @nki_flash_attention"), mlir)
    }

    @Test
    fun customCallSyntaxIncludesOperandsAndTypeSignature() {
        val mlir = emitFor(KernelTarget.NVIDIA_H100)
        // Three operands (Q, K, V), single tensor result, type signature
        // with parens around the input types and bare result type.
        assertTrue(
            mlir.contains("(%0, %1, %2)"),
            "expected (Q, K, V) operand list: $mlir",
        )
        assertTrue(
            mlir.contains(
                "(tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>",
            ),
            "expected (qType, kType, vType) -> oType signature: $mlir",
        )
    }

    @Test
    fun decomposeFallbackProducesPlainPrimitivesNoCustomCall() {
        // CPU_GENERIC forces decompose: the COARSENED is replaced by its
        // primal_body (MATMUL → SOFTMAX → MATMUL), so the emitter sees
        // no COARSENED at all and produces no custom_call.
        val mlir = emitFor(KernelTarget.CPU_GENERIC)
        assertFalse(mlir.contains("custom_call"), "decompose path emits no custom_call: $mlir")
        assertTrue(mlir.contains("stablehlo.dot_general"), mlir)
    }

    @Test
    fun coarsenedWithoutDescriptorRaisesEmitterError() {
        // The COARSENED produced by coarsenRecognizedPatterns has the
        // canonical primal_body / gradient_body / reads_primal_indices
        // attrs but no kernel_descriptor — that's the "compiler bug"
        // shape: the lowering chain neither annotated nor decomposed it.
        val fn = buildCoarsenedFn()
        val coarsened = fn.body.filterIsInstance<DxirOp>().single()
        // Sanity: confirm the test's premise — no descriptor on the bare
        // post-coarsen op.
        check(coarsened.attrs[KernelDescriptor.ATTR_KEY] == null) {
            "test invariant: bare coarsened op should not yet carry a descriptor"
        }
        assertFails { fn.toStablehlo() }
    }

    @Test
    fun emptyCustomCallAttrsProducesEmptyBackendConfig() {
        // Synthetic descriptor with no attrs: backend_config = "".
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor("synth_kernel", "tlaloc", "test", customCallAttrs = emptyMap())
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        assertTrue(mlir.contains("stablehlo.custom_call @synth_kernel"), mlir)
        assertTrue(mlir.contains("backend_config = \"\""), mlir)
    }

    @Test
    fun backendConfigSerializesAttrsInAlphabeticKeyOrder() {
        // Synthetic descriptor with multiple attr types in deliberately
        // non-alphabetic insertion order. The encoder must emit them
        // sorted alphabetically so the on-the-wire shape is deterministic.
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "synth_kernel", "tlaloc", "test",
                    customCallAttrs = linkedMapOf(
                        "softmax_scale" to 0.125f,
                        "is_causal" to true,
                        "head_dim" to 64,
                    ),
                )
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        // Alphabetical: head_dim, is_causal, softmax_scale.
        assertTrue(
            mlir.contains(
                "backend_config = \"{head_dim = 64, is_causal = true, softmax_scale = 0.125}\"",
            ),
            mlir,
        )
    }

    @Test
    fun backendConfigEncodesScalarListAndBoolValues() {
        // Each supported value type (Number, Boolean, String, List)
        // round-trips into the expected literal form.
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "synth_kernel", "tlaloc", "test",
                    customCallAttrs = mapOf(
                        "dtype_tag" to "bf16",
                        "is_causal" to false,
                        "head_dim" to 64,
                        "supported_kv_dtypes" to listOf("f32", "bf16", "int8"),
                    ),
                )
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        assertTrue(
            mlir.contains(
                "backend_config = \"{dtype_tag = bf16, head_dim = 64, is_causal = false, " +
                    "supported_kv_dtypes = [f32, bf16, int8]}\"",
            ),
            mlir,
        )
    }
}
