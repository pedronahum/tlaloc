package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.mosaicKernel
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.MosaicKernel
import io.tlaloc.ir.recognizer.kernel.OutputOperandAlias
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import io.tlaloc.stablehlo.mosaic.MosaicRmsNorm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The `tpu_custom_call` emit: JSON backend_config as an escaped MLIR string,
 * several results without a tuple, and `output_operand_aliases`.
 */
class MosaicCustomCallEmitTest {

    private val t = DxirType(F32, listOf(8, 128))
    private val w = DxirType(F32, listOf(1, 128))
    private val kernel = MosaicKernel(kernelName = "tlaloc_probe", bodyBase64 = "TUzvUgE=")

    private fun rmsRef(): DxirFunction = DxirBuilder.function("ref") {
        val x = param("x", t)
        val g = param("g", w)
        val gb = op(OpKind.BROADCAST, listOf(g), t, attrs = mapOf("broadcast_dimensions" to listOf(0, 1)))
        listOf(op(OpKind.MUL, listOf(x, gb), t))
    }

    private fun twoResultRef(): DxirFunction = DxirBuilder.function("ref2") {
        val x = param("x", t)
        val g = param("g", w)
        val gb = op(OpKind.BROADCAST, listOf(g), t, attrs = mapOf("broadcast_dimensions" to listOf(0, 1)))
        listOf(op(OpKind.MUL, listOf(x, gb), t), op(OpKind.ADD, listOf(x, gb), t))
    }

    private fun emitOnTpu(k: MosaicKernel, ref: DxirFunction, returnAll: Boolean = true): String {
        val fn = DxirBuilder.function("main") {
            val x = param("x", t)
            val g = param("g", w)
            val call = mosaicKernel(listOf(x, g), k, ref)
            if (returnAll) List(call.numResults) { call.result(it) } else listOf(call.result(0))
        }
        return lowerMosaicKernels(fn, KernelTarget.GOOGLE_TPU_V6E).toStablehlo()
    }

    @Test
    fun singleResultCallMatchesJaxSpelling() {
        val mlir = emitOnTpu(kernel, rmsRef())
        val expected =
            "stablehlo.custom_call @tpu_custom_call(%0, %1) {backend_config = " +
                "\"{\\22custom_call_config\\22: {\\22body\\22: \\22TUzvUgE=\\22, " +
                "\\22serialization_format\\22: 1, \\22needs_layout_passes\\22: true}}\", " +
                "kernel_name = \"tlaloc_probe\", mhlo.frontend_attributes = {kernel_metadata = \"{}\"}, " +
                "operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], " +
                "result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : " +
                "(tensor<8x128xf32>, tensor<1x128xf32>) -> tensor<8x128xf32>"
        assertTrue(mlir.contains(expected), mlir)
        assertFalse(mlir.contains("has_side_effect"), mlir)
        assertFalse(mlir.contains("api_version"), mlir)
    }

    @Test
    fun severalResultsAreTupleFree() {
        val mlir = emitOnTpu(kernel, twoResultRef())
        assertTrue(Regex("%(\\w+):2 = stablehlo\\.custom_call @tpu_custom_call").containsMatchIn(mlir), mlir)
        assertTrue(mlir.contains("-> (tensor<8x128xf32>, tensor<8x128xf32>)"), mlir)
        assertTrue(Regex("return %(\\w+)#0, %\\1#1 :").containsMatchIn(mlir), mlir)
        assertTrue(
            mlir.contains("result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]"),
            mlir,
        )
        assertFalse(mlir.contains("tuple"), mlir)
    }

    @Test
    fun aliasOnASingleResultHasEmptyOutputTupleIndices() {
        val mlir = emitOnTpu(kernel.copy(inputOutputAliases = mapOf(0 to 0)), rmsRef())
        assertTrue(
            mlir.contains(
                "output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], " +
                    "operand_index = 0, operand_tuple_indices = []>], result_layouts",
            ),
            mlir,
        )
    }

    @Test
    fun aliasOnASecondResultIndexesIt() {
        val mlir = emitOnTpu(kernel.copy(inputOutputAliases = mapOf(0 to 1)), twoResultRef())
        assertTrue(
            mlir.contains(
                "output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [1], " +
                    "operand_index = 0, operand_tuple_indices = []>]",
            ),
            mlir,
        )
    }

    @Test
    fun sideEffectingKernelSaysSo() {
        val mlir = emitOnTpu(kernel.copy(hasSideEffect = true), rmsRef())
        assertTrue(mlir.contains("has_side_effect = true, kernel_name"), mlir)
    }

    @Test
    fun configStringsAreEscapedTwice() {
        // JSON escapes the quote and backslash; MLIR then escapes JSON's
        // quote and backslash characters in turn.
        val k = kernel.copy(customCallConfig = linkedMapOf("note" to "a\"b\\c"))
        val mlir = emitOnTpu(k, rmsRef())
        assertTrue(mlir.contains("\\22note\\22: \\22a\\\\\\22b\\\\\\\\c\\22}}\""), mlir)
    }

    @Test
    fun unresolvedOpIsRefusedByName() {
        val fn = DxirBuilder.function("main") {
            val x = param("x", t)
            val g = param("g", w)
            listOf(mosaicKernel(listOf(x, g), kernel, rmsRef(), referenceFallback = true))
        }
        val e = assertFailsWith<IllegalStateException> { fn.toStablehlo() }
        assertTrue("'tlaloc_probe'" in e.message!! && "lowerMosaicKernels" in e.message!!, e.message)
    }

    @Test
    fun fallbackOnCpuEmitsTheReferenceAndNoCustomCall() {
        val fn = DxirBuilder.function("main") {
            val x = param("x", t)
            val g = param("g", w)
            listOf(mosaicKernel(listOf(x, g), kernel, rmsRef(), referenceFallback = true))
        }
        val mlir = lowerMosaicKernels(fn, KernelTarget.CPU_GENERIC).toStablehlo()
        assertFalse(mlir.contains("custom_call"), mlir)
        assertTrue(mlir.contains("stablehlo.multiply"), mlir)
    }

    @Test
    fun plainDescriptorsCarryAliasesToo() {
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", DxirType(F32, listOf(8, 4)))
            val k = param("K", DxirType(F32, listOf(4, 8)))
            val v = param("V", DxirType(F32, listOf(8, 4)))
            val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(8, 8)))
            val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(8, 8)))
            listOf(op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(8, 4))))
        }
        val coarsened = coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))
        val descriptor = KernelDescriptor(
            "kptx_inplace", "tlaloc", "gb10", typedFfi = true,
            outputOperandAliases = listOf(OutputOperandAlias(outputIndex = 0, operandIndex = 2)),
        )
        val registry = mapOf<String, KernelTemplate>("FlashAttention" to KernelTemplate { _, _ -> descriptor })
        val mlir = lowerKernelChoice(coarsened, KernelTarget.CPU_GENERIC, registry).toStablehlo()
        assertTrue(
            mlir.contains(
                "has_side_effect = false, output_operand_aliases = [#stablehlo.output_operand_alias<" +
                    "output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]",
            ),
            mlir,
        )
    }

    @Test
    fun mlirStringLiteralEscapesLikeMlirPrints() {
        assertEquals("\"plain\"", mlirStringLiteral("plain"))
        assertEquals("\"\\22q\\22 \\\\ \\0A \\C3\\A9\"", mlirStringLiteral("\"q\" \\ \n é"))
    }

    @Test
    fun mosaicRmsNormF32Text() {
        val text = MosaicRmsNorm.emit(rows = 16, hidden = 256)
        assertTrue(text.startsWith("module {\n  func.func @tlaloc_rmsnorm(%x_ref: memref<16x256xf32, #tpu.memory_space<vmem>>"), text)
        assertTrue(text.contains("%sum = vector.multi_reduction <add>, %sq, %zero [1] : vector<16x256xf32> to vector<16xf32>"), text)
        assertTrue(text.contains("%n = arith.constant 0x43800000 : f32"), text)
        assertTrue(text.contains("%eps = arith.constant 0x358637BD : f32"), text)
        assertTrue(text.contains("tpu.vector_store %o_ref[%c0, %c0], %y {strides = array<i32>}"), text)
        assertFalse(text.contains("arith.extf"), text)
    }

    @Test
    fun mosaicRmsNormBf16WidensAndNarrows() {
        val text = MosaicRmsNorm.emit(rows = 16, hidden = 256, dtype = MosaicRmsNorm.Dtype.BF16)
        assertTrue(text.contains("%x = arith.extf %x_in : vector<16x256xbf16> to vector<16x256xf32>"), text)
        assertTrue(text.contains("%y_out = arith.truncf %y : vector<16x256xf32> to vector<16x256xbf16>"), text)
        assertFailsWith<IllegalArgumentException> { MosaicRmsNorm.emit(rows = 12, hidden = 256) }
        assertFailsWith<IllegalArgumentException> { MosaicRmsNorm.emit(rows = 8, hidden = 100) }
    }
}
