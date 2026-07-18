package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * KPTX v1.6 (§0.4.332) — pin the **typed-FFI** custom_call emit shape:
 * a [KernelDescriptor] with `typedFfi = true` produces
 * `api_version = 4 : i32` plus attrs as a `backend_config` *dictionary*
 * (typed MLIR literals — StableHLO's canonical typed-FFI form), replacing
 * the untyped string. The default (`typedFfi = false`) emit is pinned
 * unchanged by [CoarsenedCustomCallTest] — this convention is strictly
 * opt-in per descriptor.
 */
class TypedFfiCustomCallTest {

    private val qType = DxirType(F32, listOf(8, 4))
    private val kType = DxirType(F32, listOf(4, 8))
    private val vType = DxirType(F32, listOf(8, 4))
    private val sType = DxirType(F32, listOf(8, 8))
    private val oType = DxirType(F32, listOf(8, 4))

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

    private fun emitWithDescriptor(descriptor: KernelDescriptor): String {
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ -> descriptor },
        )
        return lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry).toStablehlo()
    }

    @Test
    fun typedFfiEmitsApiVersion4AndDictBackendConfig() {
        val mlir = emitWithDescriptor(
            KernelDescriptor(
                "kptx_rms_norm", "tlaloc", "gb10", typedFfi = true,
                customCallAttrs = linkedMapOf(
                    "softmax_scale" to 0.125f,
                    "is_causal" to true,
                    "head_dim" to 64,
                    "dtype_tag" to "f32",
                ),
            ),
        )
        assertTrue(mlir.contains("stablehlo.custom_call @kptx_rms_norm"), mlir)
        assertTrue(mlir.contains("api_version = 4 : i32"), mlir)
        // Dictionary form, alphabetic keys, typed literals; strings quoted.
        assertTrue(
            mlir.contains(
                "backend_config = {dtype_tag = \"f32\", head_dim = 64 : i64, " +
                    "is_causal = true, softmax_scale = 0.125 : f32}",
            ),
            mlir,
        )
        assertTrue(mlir.contains("has_side_effect = false"), mlir)
        // The untyped string form must be absent.
        assertFalse(mlir.contains("backend_config = \""), mlir)
    }

    @Test
    fun typedFfiWithEmptyAttrsOmitsBackendConfigEntirely() {
        val mlir = emitWithDescriptor(
            KernelDescriptor("kptx_probe", "tlaloc", "gb10", typedFfi = true),
        )
        assertTrue(mlir.contains("stablehlo.custom_call @kptx_probe"), mlir)
        assertTrue(mlir.contains("api_version = 4 : i32"), mlir)
        assertFalse(mlir.contains("backend_config"), mlir)
    }

    @Test
    fun typedFfiFloatExponentsAreLowercaseMlirLiterals() {
        // Kotlin renders 1e-5f as "1.0E-5"; the MLIR literal must be
        // lowercase-e or the dict fails to parse.
        val mlir = emitWithDescriptor(
            KernelDescriptor(
                "kptx_rms_norm", "tlaloc", "gb10", typedFfi = true,
                customCallAttrs = mapOf("eps" to 1e-5f),
            ),
        )
        assertTrue(mlir.contains("backend_config = {eps = 1.0e-5 : f32}"), mlir)
    }

    @Test
    fun defaultDescriptorStillEmitsUntypedStringForm() {
        // Regression guard: same attrs, typedFfi unset → the §0.4.261
        // string convention, byte-identical to before this commit.
        val mlir = emitWithDescriptor(
            KernelDescriptor(
                "synth_kernel", "tlaloc", "test",
                customCallAttrs = mapOf("head_dim" to 64),
            ),
        )
        assertTrue(mlir.contains("backend_config = \"{head_dim = 64}\""), mlir)
        assertFalse(mlir.contains("api_version"), mlir)
        assertFalse(mlir.contains("mhlo.backend_config"), mlir)
    }
}
