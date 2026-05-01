package io.tlaloc.ir.recognizer.quant

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.257+ — KV-quant application tests.
 *
 * Pin the (target, dtype) → annotation matrix. H100/Trainium2/v6e
 * accept FP8; A100/v5e accept INT8 but not FP8; CPU has no kernel and
 * therefore declines silently with a diagnostic.
 */
class KvQuantApplyTest {

    private val qType = DxirType(F32, listOf(8, 4))
    private val kType = DxirType(F32, listOf(4, 8))
    private val vType = DxirType(F32, listOf(8, 4))
    private val sType = DxirType(F32, listOf(8, 8))
    private val oType = DxirType(F32, listOf(8, 4))

    private fun coarsenedAndLowered(target: KernelTarget): io.tlaloc.ir.DxirFunction {
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val out = op(OpKind.MATMUL, listOf(sm, v), oType)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))
        return lowerKernelChoice(coarsened, target)
    }

    @Test
    fun h100AcceptsFp8KvQuant() {
        val fn = coarsenedAndLowered(KernelTarget.NVIDIA_H100)
        val applied = applyKvQuant(fn, KvQuantConfig.FP8_PER_HEAD)
        val co = applied.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val cfg = co.attrs[KvQuantConfig.ATTR_KEY] as? KvQuantConfig
        assertNotNull(cfg, "H100 should accept FP8 KV-quant")
        assertEquals(KvQuantDtype.FP8_E4M3, cfg.dtype)
        assertEquals(KvScaleStrategy.PER_HEAD, cfg.scaleStrategy)
    }

    @Test
    fun a100DeclinesFp8AndProducesDiagnostic() {
        val fn = coarsenedAndLowered(KernelTarget.NVIDIA_A100)
        val (applied, diagnostics) = applyKvQuantWithDiagnostics(fn, KvQuantConfig.FP8_PER_HEAD)
        val co = applied.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        assertNull(co.attrs[KvQuantConfig.ATTR_KEY], "A100 has no native FP8 — kernel should decline")
        assertEquals(1, diagnostics.size)
        assertTrue("flash_attn_v2" in diagnostics.single().reason)
        assertTrue("fp8_e4m3" in diagnostics.single().reason)
    }

    @Test
    fun a100AcceptsInt8() {
        val fn = coarsenedAndLowered(KernelTarget.NVIDIA_A100)
        val (applied, diagnostics) = applyKvQuantWithDiagnostics(fn, KvQuantConfig.INT8_PER_TENSOR)
        val co = applied.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val cfg = co.attrs[KvQuantConfig.ATTR_KEY] as? KvQuantConfig
        assertNotNull(cfg, "A100 supports int8 KV cache")
        assertEquals(KvQuantDtype.INT8, cfg.dtype)
        assertEquals(KvScaleStrategy.PER_TENSOR, cfg.scaleStrategy)
        assertEquals(0, diagnostics.size)
    }

    @Test
    fun trainium2AcceptsFp8() {
        val fn = coarsenedAndLowered(KernelTarget.AWS_TRAINIUM2)
        val applied = applyKvQuant(fn, KvQuantConfig.FP8_PER_HEAD)
        val co = applied.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        assertNotNull(co.attrs[KvQuantConfig.ATTR_KEY], "Trainium2 supports FP8 KV cache")
    }

    @Test
    fun tpuV5eDeclinesFp8ButAcceptsInt8() {
        val fn = coarsenedAndLowered(KernelTarget.GOOGLE_TPU_V5E)
        val (afterFp8, fp8Diag) = applyKvQuantWithDiagnostics(fn, KvQuantConfig.FP8_PER_HEAD)
        val coAfterFp8 = afterFp8.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        assertNull(coAfterFp8.attrs[KvQuantConfig.ATTR_KEY])
        assertEquals(1, fp8Diag.size)

        val (afterInt8, int8Diag) = applyKvQuantWithDiagnostics(fn, KvQuantConfig.INT8_PER_TENSOR)
        val coAfterInt8 = afterInt8.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        assertNotNull(coAfterInt8.attrs[KvQuantConfig.ATTR_KEY])
        assertEquals(0, int8Diag.size)
    }

    @Test
    fun tpuV6eAddsFp8OnTopOfV5eMatrix() {
        val fn = coarsenedAndLowered(KernelTarget.GOOGLE_TPU_V6E)
        val applied = applyKvQuant(fn, KvQuantConfig.FP8_PER_HEAD)
        val co = applied.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        assertNotNull(co.attrs[KvQuantConfig.ATTR_KEY], "TPU v6e (Trillium) supports FP8 — added in §0.4.257")
    }

    @Test
    fun cpuGenericDeclinesEverythingWithDiagnostic() {
        // CPU_GENERIC forces decompose in L3.3, so there's no COARSENED
        // left for KV-quant to annotate. The pass is a no-op + reports
        // no diagnostics (because the no-COARSENED-found short-circuit
        // returns early before we walk descriptor compatibility).
        val fn = coarsenedAndLowered(KernelTarget.CPU_GENERIC)
        val (applied, diag) = applyKvQuantWithDiagnostics(fn, KvQuantConfig.FP8_PER_HEAD)
        val coarsenedOps = applied.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(0, coarsenedOps.size, "CPU forced decompose; no COARSENED to quant")
        assertEquals(0, diag.size, "no COARSENED ops → no diagnostics")
    }

    @Test
    fun coarsenedWithoutKernelDescriptorIsSkipped() {
        // Build a coarsened function but skip L3.3 lowering — there's a
        // COARSENED with no kernel_descriptor attached.
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val out = op(OpKind.MATMUL, listOf(sm, v), oType)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))

        val (applied, diagnostics) = applyKvQuantWithDiagnostics(coarsened, KvQuantConfig.FP8_PER_HEAD)
        val co = applied.body.filterIsInstance<DxirOp>().single()
        assertNull(co.attrs[KvQuantConfig.ATTR_KEY])
        assertEquals(1, diagnostics.size)
        assertTrue("no kernel descriptor" in diagnostics.single().reason)
    }

    @Test
    fun functionWithoutCoarsenedOpsIsNoOp() {
        val fn = DxirBuilder.function("plain") {
            val x = param("x", qType)
            val r = op(OpKind.RELU, listOf(x), qType)
            listOf(r)
        }
        val applied = applyKvQuant(fn, KvQuantConfig.FP8_PER_HEAD)
        // Same shape, same ops — the pass short-circuits on no COARSENED.
        val ops = applied.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.RELU, ops.single().op)
    }

    @Test
    fun configsAreReachableViaCompanionDefaults() {
        val fp8 = KvQuantConfig.FP8_PER_HEAD
        assertEquals(KvQuantDtype.FP8_E4M3, fp8.dtype)
        assertEquals(KvScaleStrategy.PER_HEAD, fp8.scaleStrategy)
        val int8 = KvQuantConfig.INT8_PER_TENSOR
        assertEquals(KvQuantDtype.INT8, int8.dtype)
        assertEquals(KvScaleStrategy.PER_TENSOR, int8.scaleStrategy)
    }

    @Test
    fun bitsPerElementMatchesDtype() {
        // Sanity on the size table — the cost model needs these values
        // to estimate KV-cache memory savings later.
        assertEquals(32, KvQuantDtype.F32.bitsPerElement)
        assertEquals(16, KvQuantDtype.BF16.bitsPerElement)
        assertEquals(8, KvQuantDtype.FP8_E4M3.bitsPerElement)
        assertEquals(8, KvQuantDtype.FP8_E5M2.bitsPerElement)
        assertEquals(8, KvQuantDtype.INT8.bitsPerElement)
        assertEquals(4, KvQuantDtype.INT4.bitsPerElement)
    }
}
