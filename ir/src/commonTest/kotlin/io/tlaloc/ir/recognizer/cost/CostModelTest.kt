package io.tlaloc.ir.recognizer.cost

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.255+ — cost model tests.
 *
 * Pin canonical FLOPs/bytes formulas (matmul = 2·m·k·n; elementwise = N;
 * softmax ≈ 5·N) and verify the COARSENED dispatch — annotated with a
 * `kernel_descriptor` lowers `bytesMoved` to operands+output (the headline
 * fused-attention benefit), without a descriptor it stays at the
 * decompose-equivalent cost.
 */
class CostModelTest {

    @Test
    fun matmulFlopsIsTwoTimesMKN() {
        // [4, 3] · [3, 5] → 2 · 4 · 3 · 5 = 120 FLOPs
        val fn = DxirBuilder.function("mm") {
            val a = param("A", DxirType(F32, listOf(4, 3)))
            val b = param("B", DxirType(F32, listOf(3, 5)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(4, 5)))
            listOf(c)
        }
        val cost = estimateCost(fn)
        assertEquals(120.0, cost.flops, "2·m·k·n = 2·4·3·5 = 120")
        // Bytes: read A (12 elements × 4B = 48), B (15 × 4 = 60), write C (20 × 4 = 80) → 188
        assertEquals(188.0, cost.bytesMoved)
    }

    @Test
    fun batchedMatmulScalesByBatch() {
        // [2, 8, 4, 3] · [2, 8, 3, 5] → 2 · 16 · 4 · 3 · 5 = 1920 FLOPs
        val fn = DxirBuilder.function("bmm") {
            val a = param("A", DxirType(F32, listOf(2, 8, 4, 3)))
            val b = param("B", DxirType(F32, listOf(2, 8, 3, 5)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(2, 8, 4, 5)))
            listOf(c)
        }
        val cost = estimateCost(fn)
        assertEquals(1920.0, cost.flops)
    }

    @Test
    fun elementwiseAddIsOneFlopPerElement() {
        val fn = DxirBuilder.function("add") {
            val x = param("x", DxirType(F32, listOf(100)))
            val y = param("y", DxirType(F32, listOf(100)))
            val z = op(OpKind.ADD, listOf(x, y), DxirType(F32, listOf(100)))
            listOf(z)
        }
        assertEquals(100.0, estimateCost(fn).flops)
    }

    @Test
    fun reductionFlopsScaleWithInput() {
        // SUM over [256] → scalar: ~256 FLOPs.
        val fn = DxirBuilder.function("sum") {
            val x = param("x", DxirType(F32, listOf(256)))
            val s = op(OpKind.SUM, listOf(x), DxirType(F32, listOf()))
            listOf(s)
        }
        assertEquals(256.0, estimateCost(fn).flops)
    }

    @Test
    fun softmaxIsFiveFlopsPerInputElement() {
        // SOFTMAX over [128, 64] over last axis: ~5·N = 5·8192 = 40960 FLOPs.
        val fn = DxirBuilder.function("sm") {
            val x = param("x", DxirType(F32, listOf(128, 64)))
            val s = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(128, 64)))
            listOf(s)
        }
        assertEquals(40960.0, estimateCost(fn).flops)
    }

    @Test
    fun transposeAndBroadcastAreZeroFlops() {
        val fn = DxirBuilder.function("shape") {
            val x = param("x", DxirType(F32, listOf(128, 64)))
            val xt = op(
                OpKind.TRANSPOSE, listOf(x),
                DxirType(F32, listOf(64, 128)),
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            listOf(xt)
        }
        assertEquals(0.0, estimateCost(fn).flops, "TRANSPOSE = data movement, no FLOPs")
        // But bytes moved is non-zero (read in, write out).
        assertTrue(estimateCost(fn).bytesMoved > 0.0)
    }

    @Test
    fun aggregatesAcrossMultipleOps() {
        val fn = DxirBuilder.function("chain") {
            val x = param("x", DxirType(F32, listOf(100)))
            val y = param("y", DxirType(F32, listOf(100)))
            val a = op(OpKind.ADD, listOf(x, y), DxirType(F32, listOf(100)))  // 100 FLOPs
            val b = op(OpKind.MUL, listOf(a, x), DxirType(F32, listOf(100)))  // 100 FLOPs
            listOf(b)
        }
        assertEquals(200.0, estimateCost(fn).flops)
    }

    @Test
    fun coarsenedWithoutKernelEqualsDecomposeCost() {
        // Build canonical attention, coarsen — but no kernel lowering yet,
        // so no kernel_descriptor attr → decompose-equivalent cost.
        val attn = DxirBuilder.function("attn") {
            val q = param("Q", DxirType(F32, listOf(8, 4)))
            val k = param("K", DxirType(F32, listOf(4, 8)))
            val v = param("V", DxirType(F32, listOf(8, 4)))
            val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(8, 8)))
            val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(8, 8)))
            val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(8, 4)))
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(attn, recognizeFlashAttention(attn))
        val coarsenedCost = estimateCost(coarsened)
        val unfusedCost = estimateCost(attn)
        // Decompose path: COARSENED's primal_body has same body shape as
        // the original attn function (3 ops). FLOPs match exactly.
        assertEquals(unfusedCost.flops, coarsenedCost.flops, "decompose FLOPs == unfused")
    }

    @Test
    fun coarsenedWithKernelDescriptorLowersBytesMoved() {
        // Same attention, but lowered with NVIDIA H100 → kernel_descriptor
        // is set, so the cost model elides intermediate bytes.
        val attn = DxirBuilder.function("attn") {
            val q = param("Q", DxirType(F32, listOf(8, 4)))
            val k = param("K", DxirType(F32, listOf(4, 8)))
            val v = param("V", DxirType(F32, listOf(8, 4)))
            val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(8, 8)))
            val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(8, 8)))
            val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(8, 4)))
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(attn, recognizeFlashAttention(attn))
        val withKernel = lowerKernelChoice(coarsened, KernelTarget.NVIDIA_H100)

        val unfusedCost = estimateCost(attn)
        val fusedCost = estimateCost(withKernel)

        // FLOPs unchanged — same matmuls + softmax.
        assertEquals(unfusedCost.flops, fusedCost.flops, "kernel-call FLOPs == unfused")
        // But fused bytes-moved < unfused bytes-moved: intermediate S, P
        // tensors don't round-trip to HBM.
        assertTrue(
            fusedCost.bytesMoved < unfusedCost.bytesMoved,
            "fused bytes (${fusedCost.bytesMoved}) < unfused bytes (${unfusedCost.bytesMoved})",
        )
    }

    @Test
    fun arithmeticIntensityRisesWithFusion() {
        // Same attention shapes, decompose vs fused. Fusion increases
        // FLOPs/byte (the headline reason fused kernels exist).
        val attn = DxirBuilder.function("attn") {
            val q = param("Q", DxirType(F32, listOf(64, 64)))
            val k = param("K", DxirType(F32, listOf(64, 64)))
            val v = param("V", DxirType(F32, listOf(64, 64)))
            val qk = op(OpKind.MATMUL, listOf(q, k), DxirType(F32, listOf(64, 64)))
            val sm = op(OpKind.SOFTMAX, listOf(qk), DxirType(F32, listOf(64, 64)))
            val out = op(OpKind.MATMUL, listOf(sm, v), DxirType(F32, listOf(64, 64)))
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(attn, recognizeFlashAttention(attn))
        val fused = lowerKernelChoice(coarsened, KernelTarget.NVIDIA_H100)

        val unfusedAi = estimateCost(attn).arithmeticIntensity
        val fusedAi = estimateCost(fused).arithmeticIntensity
        assertTrue(fusedAi > unfusedAi, "fused AI ($fusedAi) > unfused AI ($unfusedAi)")
    }

    @Test
    fun rooflineMicrosScalesWithDevice() {
        // Same workload, different devices. H100 should be (much)
        // faster than A100 should be (much) faster than CPU.
        val fn = DxirBuilder.function("mm") {
            val a = param("A", DxirType(F32, listOf(1024, 1024)))
            val b = param("B", DxirType(F32, listOf(1024, 1024)))
            val c = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(1024, 1024)))
            listOf(c)
        }
        val tH100 = estimateRooflineMicros(fn, DeviceDescriptors.H100)
        val tA100 = estimateRooflineMicros(fn, DeviceDescriptors.A100)
        val tCpu = estimateRooflineMicros(fn, DeviceDescriptors.CPU_GENERIC)
        assertTrue(tH100 < tA100, "H100 ($tH100 us) faster than A100 ($tA100 us)")
        assertTrue(tA100 < tCpu, "A100 ($tA100 us) faster than CPU ($tCpu us)")
    }

    @Test
    fun emptyFunctionHasZeroCost() {
        val fn = DxirBuilder.function("empty") {
            val x = param("x", DxirType(F32, listOf(10)))
            listOf(x)
        }
        assertEquals(CostEstimate.ZERO, estimateCost(fn))
    }
}
