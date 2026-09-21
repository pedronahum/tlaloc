package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.kptxInferenceKernelTemplates
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.471 — Phase H4: the OFFLINE half of paged-attention *claiming*.
 *
 * The load-bearing oracle is `KptxPagedAttentionKernelTest` (the kernel vs
 * the gather-composed emission under real XLA on the GB10), but that
 * self-skips without CUDA. What always runs is this: the claimed graph's
 * emission is ONE typed-FFI custom_call carrying all five operands and two
 * results, and — the property that matters — **none of the reference
 * lowering survives**. A claim that left a stray gather behind would still
 * produce the right numbers and quietly pay for the work twice.
 */
class PagedAttentionCustomCallEmitTest {

    private val numSeqs = 2
    private val numHeads = 4
    private val numKvHeads = 2
    private val headDim = 3
    private val blockSize = 2
    private val numBlocks = 6
    private val maxBlocksPerSeq = 3

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tableType = DxirType(I32, listOf(numSeqs, maxBlocksPerSeq))
    private val lensType = DxirType(I32, listOf(numSeqs))

    private fun pagedFn(): DxirFunction = DxirBuilder.function("paged") {
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        val t = param("t", tableType)
        val l = param("l", lensType)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to 0.5)))
    }

    private fun claimedMlir(): String =
        lowerKernelChoice(
            pagedFn(),
            KernelTarget.NVIDIA_GB10,
            inferenceRegistry = kptxInferenceKernelTemplates,
        ).toStablehlo()

    @Test
    fun theClaimedGraphIsOneTypedFfiCustomCall() {
        val text = claimedMlir()
        val calls = text.lines().filter { "stablehlo.custom_call" in it }
        assertEquals(1, calls.size, "one call, not one per stage — the chain is the handler's:\n$text")
        val call = calls.single()
        assertTrue("@kptx_paged_attention" in call, call)
        assertTrue("api_version = 4 : i32" in call, "typed-FFI convention (§0.4.332): $call")
        // All five operands ride the call: the block table and seqLens are
        // device buffers the kernel dereferences, NOT compile-time attrs.
        assertEquals(
            5,
            Regex("""@kptx_paged_attention\(([^)]*)\)""").find(call)!!
                .groupValues[1].split(",").size,
            call,
        )
        assertTrue(
            "(tensor<${numSeqs}x${numHeads}x${headDim}xf32>, " +
                "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>, " +
                "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>, " +
                "tensor<${numSeqs}x${maxBlocksPerSeq}xi32>, tensor<${numSeqs}xi32>)" in call,
            "the integer block table and seqLens ride as device buffers: $call",
        )
        // Result #0 is the op's own value; result #1 is the XLA-owned score
        // matrix the three stages hand between them (§0.4.350/351).
        assertTrue(
            "(tensor<${numSeqs}x${numHeads}x${headDim}xf32>, " +
                "tensor<${numSeqs * numHeads}x${maxBlocksPerSeq * blockSize}xf32>)" in call,
            "out + the [rows, ctx] score scratch: $call",
        )
        assertTrue("scale = 5.0e-01 : f64" in call || "scale = 0.5 : f64" in call, call)
    }

    @Test
    fun claimingReplacesTheReferenceLoweringWholesale() {
        val text = claimedMlir()
        for (leftover in listOf("stablehlo.gather", "stablehlo.dot_general", "stablehlo.reduce", "stablehlo.select")) {
            assertTrue(
                text.lines().none { leftover in it },
                "the fused kernel does this work; '$leftover' surviving means it is done twice:\n$text",
            )
        }
    }

    @Test
    fun theUnclaimedGraphStillEmitsTheReferenceForm() {
        // The safety property of claiming a first-class op kind: declining is
        // not a hole. Same function, no registry ⇒ §0.4.465's lowering, byte
        // for byte.
        val plain = pagedFn().toStablehlo()
        val unclaimed = lowerKernelChoice(pagedFn(), KernelTarget.CPU_GENERIC).toStablehlo()
        assertEquals(plain, unclaimed)
        assertTrue(plain.lines().any { "stablehlo.gather" in it })
        assertTrue(plain.lines().none { "custom_call" in it })
    }
}
