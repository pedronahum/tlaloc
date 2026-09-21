package io.tlaloc.stablehlo

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DequantizeKvAttrs
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.472 — Phase H5: DEQUANTIZE_KV's emission. Three ops and no cleverness,
 * so these are STRUCTURE pins on the three:
 *
 * - the `convert` that widens the integer codes (the place a narrow DType will
 *   one day change and nothing else will notice),
 * - the `broadcast_in_dim ... dims = [2]` that IS the per-head convention —
 *   axis 2 of `[numBlocks, blockSize, numKvHeads, headDim]`,
 * - the `multiply`.
 *
 * And one pin on what must NOT be there: no constant folded in place of the
 * scales operand, because the scales are runtime data a loader computes from
 * the checkpoint it just read.
 */
class DequantizeKvEmitTest {

    private val numBlocks = 4
    private val blockSize = 2
    private val numKvHeads = 2
    private val headDim = 3

    private val codesType = DxirType(I32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val poolType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val bf16PoolType = DxirType(BF16, listOf(numBlocks, blockSize, numKvHeads, headDim))

    private fun fn(
        scalesType: DxirType = DxirType(F32, listOf(numKvHeads)),
        outType: DxirType = poolType,
        tag: String = "int8",
    ): DxirFunction = DxirBuilder.function("dequant") {
        val codes = param("codes", codesType)
        val scales = param("scales", scalesType)
        listOf(
            op(
                OpKind.DEQUANTIZE_KV, listOf(codes, scales), outType,
                mapOf(DequantizeKvAttrs.DTYPE_ATTR to tag),
            ),
        )
    }

    @Test
    fun emitsConvertBroadcastMultiply() {
        val text = fn().toStablehlo()
        assertTrue(
            "(tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xi32>) -> " +
                "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>" in text &&
                text.lines().any { "stablehlo.convert" in it },
            "codes must widen from their integer tensor:\n$text",
        )
        assertTrue(
            text.lines().any { "stablehlo.broadcast_in_dim" in it && "dims = [2]" in it },
            "the scale vector must broadcast along the KV-HEAD axis (2):\n$text",
        )
        assertEquals(
            1, text.lines().count { "stablehlo.multiply" in it },
            "exactly one multiply — dequantization is `code * scale`:\n$text",
        )
    }

    @Test
    fun perTensorScalesRideTheSameBroadcast() {
        val text = fn(scalesType = DxirType(F32, listOf(1))).toStablehlo()
        assertTrue(
            text.lines().any { "stablehlo.broadcast_in_dim" in it && "dims = [2]" in it },
            "a [1] scale needs no arm of its own — size-1 expands:\n$text",
        )
        assertTrue("(tensor<1xf32>)" in text, "the per-tensor scale stays a rank-1 operand:\n$text")
    }

    @Test
    fun scalesAreNeverFoldedIntoAConstant() {
        val text = fn().toStablehlo()
        assertTrue(
            "stablehlo.constant" !in text,
            "the scales are RUNTIME data (a loader computes them from a checkpoint); " +
                "folding one in would pin a checkpoint's numbers into the executable:\n$text",
        )
    }

    @Test
    fun aNarrowerPoolWidensTheScalesFirst() {
        val text = fn(outType = bf16PoolType).toStablehlo()
        assertTrue(
            "(tensor<${numKvHeads}xf32>) -> tensor<${numKvHeads}xbf16>" in text,
            "an f32 scale vector against a bf16 pool must be converted before the multiply:\n$text",
        )
        assertEquals(
            2, text.lines().count { "stablehlo.convert" in it },
            "one convert for the codes, one for the scales:\n$text",
        )
    }

    @Test
    fun fp8IsRefusedAtEmissionByName() {
        val ex = assertFailsWith<IllegalArgumentException> { fn(tag = "fp8_e5m2").toStablehlo() }
        assertTrue("fp8_e5m2" in (ex.message ?: ""), "must name the refused format; got: ${ex.message}")
    }
}
