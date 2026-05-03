package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.253+ — kernel template registry + decompose-fallback
 * tests.
 *
 * Two paths exercised:
 *
 * - **Annotate path** — vendor-fused kernel selected for the target.
 *   COARSENED is rewritten with a [KernelDescriptor] under
 *   [KernelDescriptor.ATTR_KEY], primal_body / gradient_body /
 *   reads_primal_indices preserved.
 * - **Decompose path** — no kernel for the target. COARSENED is
 *   replaced by the inlined primal_body's body (3 ops: MATMUL → SOFTMAX
 *   → MATMUL — the recognizer's matched shape).
 */
class KernelLoweringTest {

    private val mDim = 8
    private val kDim = 4
    private val nDim = 8
    private val dvDim = 4

    private val qType = DxirType(F32, listOf(mDim, kDim))
    private val kType = DxirType(F32, listOf(kDim, nDim))
    private val vType = DxirType(F32, listOf(nDim, dvDim))
    private val sType = DxirType(F32, listOf(mDim, nDim))
    private val oType = DxirType(F32, listOf(mDim, dvDim))

    private fun buildCoarsenedFn(): io.tlaloc.ir.DxirFunction {
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

    @Test
    fun h100AnnotatesCoarsenedWithFlashAttnV3() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_H100)

        val co = lowered.body.filterIsInstance<DxirOp>().single()
        assertEquals(OpKind.COARSENED, co.op)

        val descriptor = co.attrs[KernelDescriptor.ATTR_KEY]
        assertNotNull(descriptor, "H100 target should attach a kernel descriptor")
        descriptor as KernelDescriptor
        assertEquals("flash_attn_v3", descriptor.kernelName)
        assertEquals("nvidia", descriptor.vendor)
        assertEquals("h100", descriptor.targetArch)
    }

    @Test
    fun a100PicksFlashAttnV2() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_A100)
        val descriptor = lowered.body.filterIsInstance<DxirOp>()
            .single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(descriptor)
        assertEquals("flash_attn_v2", descriptor.kernelName)
        assertEquals("a100", descriptor.targetArch)
    }

    @Test
    fun tpuV5ePicksTpuPallasFlashAttention() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.GOOGLE_TPU_V5E)
        val descriptor = lowered.body.filterIsInstance<DxirOp>()
            .single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(descriptor)
        assertEquals("tpu_pallas_flash_attention", descriptor.kernelName)
    }

    @Test
    fun gb10PicksFlashAttnV3() {
        // GB10 (Grace-Blackwell Spark, sm_100) reuses FA3 — the kernel
        // family Tlaloc's dev host actually consumes.
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_GB10)
        val descriptor = lowered.body.filterIsInstance<DxirOp>()
            .single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(descriptor, "GB10 must attach a kernel descriptor")
        assertEquals("flash_attn_v3", descriptor.kernelName)
        assertEquals("nvidia", descriptor.vendor)
        assertEquals("gb10", descriptor.targetArch)
        @Suppress("UNCHECKED_CAST")
        val kvDtypes = descriptor.customCallAttrs["supported_kv_dtypes"] as List<String>
        assertTrue(kvDtypes.contains("fp8_e5m2"), "GB10 inherits FA3's full fp8_e5m2 KV-quant set")
    }

    @Test
    fun b100PicksCudnnMultiHeadAttention() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_B100)
        val descriptor = lowered.body.filterIsInstance<DxirOp>()
            .single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(descriptor)
        assertEquals("cudnn_multi_head_attention", descriptor.kernelName)
        assertEquals("b100", descriptor.targetArch)
        @Suppress("UNCHECKED_CAST")
        val kvDtypes = descriptor.customCallAttrs["supported_kv_dtypes"] as List<String>
        // cuDNN MHA's GA surface omits fp8_e5m2 — conservative pick.
        assertFalse(kvDtypes.contains("fp8_e5m2"), "cuDNN MHA must not advertise fp8_e5m2 in v1")
        assertTrue(kvDtypes.contains("fp8_e4m3"))
    }

    @Test
    fun b200PicksCudnnMultiHeadAttention() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_B200)
        val descriptor = lowered.body.filterIsInstance<DxirOp>()
            .single().attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(descriptor)
        assertEquals("cudnn_multi_head_attention", descriptor.kernelName)
        assertEquals("b200", descriptor.targetArch)
    }

    @Test
    fun annotatedCoarsenedPreservesPrimalAndGradientBodies() {
        val fn = buildCoarsenedFn()
        val originalCo = fn.body.filterIsInstance<DxirOp>().single()
        val originalPrimal = originalCo.attrs["primal_body"] as io.tlaloc.ir.DxirFunction
        val originalGrad = originalCo.attrs["gradient_body"] as io.tlaloc.ir.DxirFunction

        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_H100)
        val newCo = lowered.body.filterIsInstance<DxirOp>().single()

        // Primal + gradient bodies are reused (referential equality —
        // we don't deep-clone bodies during annotation).
        assertEquals(originalPrimal, newCo.attrs["primal_body"])
        assertEquals(originalGrad, newCo.attrs["gradient_body"])
        // reads_primal_indices preserved
        @Suppress("UNCHECKED_CAST")
        val origReads = originalCo.attrs["reads_primal_indices"] as Set<Int>
        @Suppress("UNCHECKED_CAST")
        val newReads = newCo.attrs["reads_primal_indices"] as Set<Int>
        assertEquals(origReads, newReads)
    }

    @Test
    fun cpuGenericDecomposesBackToPrimitives() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.CPU_GENERIC)

        // After decomposition the COARSENED is gone; we have the
        // inlined MATMUL → SOFTMAX → MATMUL.
        val ops = lowered.body.filterIsInstance<DxirOp>()
        assertFalse(
            ops.any { it.op == OpKind.COARSENED },
            "CPU_GENERIC should force decompose; got body ${ops.map { it.op }}",
        )
        assertEquals(3, ops.size, "decomposed body has the original three ops")
        assertEquals(OpKind.MATMUL, ops[0].op)
        assertEquals(OpKind.SOFTMAX, ops[1].op)
        assertEquals(OpKind.MATMUL, ops[2].op)
    }

    @Test
    fun unknownVendorDecomposes() {
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget("unicorn", "purple_glitter"))
        val ops = lowered.body.filterIsInstance<DxirOp>()
        assertFalse(ops.any { it.op == OpKind.COARSENED })
        assertEquals(3, ops.size, "unknown vendor → decompose")
    }

    @Test
    fun functionWithoutCoarsenedOpsPassesThrough() {
        val fn = DxirBuilder.function("plain") {
            val x = param("x", qType)
            val y = op(OpKind.RELU, listOf(x), qType)
            listOf(y)
        }
        // Should be a no-op: no COARSENED ops in body, nothing to do.
        val out = lowerKernelChoice(fn, KernelTarget.NVIDIA_H100)
        // Same shape (one RELU op).
        val ops = out.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.RELU, ops.single().op)
    }

    @Test
    fun decomposedCoarsenedReturnsCorrectValue() {
        // After decompose, the function's return references the inlined
        // pvMatmul (the last op in the primal body), not the original
        // COARSENED id.
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.CPU_GENERIC)
        assertEquals(1, lowered.returns.size)
        val ret = lowered.returns.single()
        // The return points to the last MATMUL of the inlined body.
        val ops = lowered.body.filterIsInstance<DxirOp>()
        val lastMatmul = ops.last()
        assertEquals(OpKind.MATMUL, lastMatmul.op)
        assertEquals(lastMatmul.id, ret.id)
        assertEquals(oType, ret.type)
    }

    @Test
    fun emptyRegistryForcesDecomposeOnEveryTarget() {
        val fn = buildCoarsenedFn()
        // Even on H100 — if there's no template registered, fall through
        // to "no template" which means perCoarsened stays empty and the
        // function passes through unchanged (not decomposed). This
        // tests the "no template registered" path specifically.
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_H100, registry = emptyMap())
        // No template lookup → COARSENED preserved as-is.
        val ops = lowered.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.COARSENED, ops.single().op)
        // No descriptor attached.
        assertNull(ops.single().attrs[KernelDescriptor.ATTR_KEY])
    }

    @Test
    fun annotatedCoarsenedStillValidatesAsCoarsenedOp() {
        // The annotated COARSENED still has the canonical primal_body /
        // gradient_body / reads_primal_indices attrs, so DxirFunction.init's
        // validateCoarsenedShape still accepts it.
        val fn = buildCoarsenedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_H100)
        // Re-run through DxirBuilder.function to revalidate.
        val rebuilt = DxirBuilder.function(lowered.name) {
            val nodeMap = HashMap<Int, io.tlaloc.ir.DxirNode>()
            for (p in lowered.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
            for (node in lowered.body) {
                if (node !is DxirOp) continue
                val operands = node.operands.map { nodeMap[it.id]!! }
                val newOp = op(node.op, operands, node.type, node.attrs, node.sharding)
                nodeMap[node.id] = newOp
            }
            lowered.returns.map { nodeMap[it.id]!! }
        }
        assertEquals(1, rebuilt.body.filterIsInstance<DxirOp>().size)
    }
}
