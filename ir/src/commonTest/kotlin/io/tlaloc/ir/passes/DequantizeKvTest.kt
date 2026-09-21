package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DequantizeKvAttrs
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.inference.KvQuantPool
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.KvQuantDtype
import io.tlaloc.ir.recognizer.quant.KvScaleStrategy
import io.tlaloc.ir.render.KotlinRenderRefusal
import io.tlaloc.ir.render.toKotlinSource
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.472 — Phase H5: DEQUANTIZE_KV in the graph, and the capability it
 * unlocks — PAGED ATTENTION OVER AN INT8 POOL.
 *
 * The oracle story, strongest first:
 *
 * 1. **Paged attention over a quantized pool vs the f32 pool, at a STATED
 *    accuracy floor.** The same decode step is run twice: once against the
 *    original f32 K/V pools, once against pools that were quantized to int8
 *    and dequantized back IN THE GRAPH. The two answers are required to agree
 *    to a floor DERIVED from the quantization bound (see the test), not to a
 *    tolerance someone tuned. This is the number the whole slice exists to
 *    produce: what int8 KV costs in answer quality, measured rather than
 *    asserted.
 * 2. **The op and the host codec are one formula.** The in-graph
 *    DEQUANTIZE_KV and [KvQuantPool.dequantize] are pinned elementwise —
 *    a loader quantizes with the codec and the graph reads with the op, so a
 *    disagreement would read a pool through a different scale than it was
 *    written with and nothing downstream would look wrong.
 * 3. **The named refusals**: inference-only by design, so both AD transforms
 *    and the Kotlin renderer refuse BY NAME; fp8 and a mis-shaped scale vector
 *    are refused by the one parser.
 */
class DequantizeKvTest {

    private val numBlocks = 4
    private val blockSize = 2
    private val numKvHeads = 2
    private val headDim = 3
    private val poolElems = numBlocks * blockSize * numKvHeads * headDim

    private val codesType = DxirType(I32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val poolType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val perHeadScalesType = DxirType(F32, listOf(numKvHeads))
    private val perTensorScalesType = DxirType(F32, listOf(1))

    private val int8PerHead = KvQuantConfig(KvQuantDtype.INT8, KvScaleStrategy.PER_HEAD)

    private fun dequantFn(
        scalesType: DxirType = perHeadScalesType,
        tag: String = "int8",
    ): DxirFunction = DxirBuilder.function("dequant") {
        val codes = param("codes", codesType)
        val scales = param("scales", scalesType)
        listOf(
            op(
                OpKind.DEQUANTIZE_KV, listOf(codes, scales), poolType,
                mapOf(DequantizeKvAttrs.DTYPE_ATTR to tag),
            ),
        )
    }

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 4f
        }
    }

    // --- Oracle 2: one formula, two callers. ---------------------------

    @Test
    fun theGraphOpAndTheHostCodecAgreeElementwise() {
        val pool = pseudo(poolElems, 31)
        val q = KvQuantPool.quantize(pool, numKvHeads, headDim, int8PerHead)
        val fromGraph = DxirInterpreter.evalFunction(
            dequantFn(),
            listOf(FloatArray(q.codes.size) { q.codes[it].toFloat() }, q.scales),
        )[0]
        val fromCodec = q.dequantize(numKvHeads, headDim)
        for (i in fromCodec.indices) {
            assertEquals(fromCodec[i], fromGraph[i], 0f, "element $i: op and codec must be the SAME formula")
        }
    }

    @Test
    fun perTensorScalingIsDerivedFromTheScaleExtent() {
        val pool = pseudo(poolElems, 77)
        val q = KvQuantPool.quantize(pool, numKvHeads, headDim, KvQuantConfig.INT8_PER_TENSOR)
        assertEquals(1, q.scales.size)
        val fromGraph = DxirInterpreter.evalFunction(
            dequantFn(scalesType = perTensorScalesType),
            listOf(FloatArray(q.codes.size) { q.codes[it].toFloat() }, q.scales),
        )[0]
        val fromCodec = q.dequantize(numKvHeads, headDim)
        for (i in fromCodec.indices) assertEquals(fromCodec[i], fromGraph[i], 0f, "element $i")
    }

    // --- Oracle 1: paged attention over an int8 pool. -------------------

    private val numSeqs = 2
    private val numHeads = 4
    private val maxBlocksPerSeq = 2
    private val scale = 0.5

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val tableType = DxirType(I32, listOf(numSeqs, maxBlocksPerSeq))
    private val lensType = DxirType(I32, listOf(numSeqs))

    /** Paged attention reading FLOAT pools — the reference. */
    private fun pagedF32Fn(): DxirFunction = DxirBuilder.function("pagedF32") {
        val q = param("q", qType)
        val k = param("k", poolType)
        val v = param("v", poolType)
        val t = param("t", tableType)
        val l = param("l", lensType)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scale)))
    }

    /**
     * Paged attention reading QUANTIZED pools: the decode step as a serving
     * deployment with an int8 KV cache actually runs it — codes and scales
     * cross the boundary, DEQUANTIZE_KV is the first thing the graph does, and
     * the attention downstream is byte-for-byte the same op.
     */
    private fun pagedInt8Fn(): DxirFunction = DxirBuilder.function("pagedInt8") {
        val q = param("q", qType)
        val kCodes = param("kCodes", codesType)
        val kScales = param("kScales", perHeadScalesType)
        val vCodes = param("vCodes", codesType)
        val vScales = param("vScales", perHeadScalesType)
        val t = param("t", tableType)
        val l = param("l", lensType)
        val attrs = mapOf(DequantizeKvAttrs.DTYPE_ATTR to "int8")
        val k = op(OpKind.DEQUANTIZE_KV, listOf(kCodes, kScales), poolType, attrs)
        val v = op(OpKind.DEQUANTIZE_KV, listOf(vCodes, vScales), poolType, attrs)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scale)))
    }

    @Test
    fun pagedAttentionOverAnInt8PoolMeetsTheDerivedAccuracyFloor() {
        val query = pseudo(numSeqs * numHeads * headDim, 5)
        val k = pseudo(poolElems, 101)
        val v = pseudo(poolElems, 103)
        val table = floatArrayOf(0f, 1f, 2f, 3f)
        val lens = floatArrayOf(4f, 3f) // one full pair of pages, one ragged

        val kq = KvQuantPool.quantize(k, numKvHeads, headDim, int8PerHead)
        val vq = KvQuantPool.quantize(v, numKvHeads, headDim, int8PerHead)

        val reference = DxirInterpreter.evalFunction(pagedF32Fn(), listOf(query, k, v, table, lens))[0]
        val quantized = DxirInterpreter.evalFunction(
            pagedInt8Fn(),
            listOf(
                query,
                FloatArray(kq.codes.size) { kq.codes[it].toFloat() }, kq.scales,
                FloatArray(vq.codes.size) { vq.codes[it].toFloat() }, vq.scales,
                table, lens,
            ),
        )[0]

        // THE FLOOR, DERIVED. The output is a convex combination of V rows
        // (softmax weights sum to 1), so the V error contributes at most
        // max|ΔV| = scaleV/2 directly. The K error moves the WEIGHTS, and a
        // softmax's output moves by at most (max score perturbation) in the
        // worst case of a two-point distribution; the score perturbation is
        // bounded by scale · headDim · max|q| · (scaleK/2). Both terms are
        // computed from the quantization bound and the actual inputs — no
        // tuned tolerance.
        val vTerm = KvQuantPool.maxRoundTripError(vq.scales)
        val maxAbsQ = query.maxOf { abs(it) }
        val scoreWobble = scale.toFloat() * headDim * maxAbsQ * KvQuantPool.maxRoundTripError(kq.scales)
        val vSpan = v.max() - v.min()
        val floor = vTerm + scoreWobble * vSpan

        var worst = 0f
        for (i in reference.indices) worst = maxOf(worst, abs(reference[i] - quantized[i]))
        assertTrue(
            worst <= floor,
            "int8 KV cost $worst per element, above the derived floor $floor " +
                "(vTerm=$vTerm, scoreWobble=$scoreWobble, vSpan=$vSpan)",
        )
        // And it must actually be a QUANTIZED run — if the codes round-tripped
        // exactly the test would pass while proving nothing.
        assertTrue(worst > 0f, "the quantized run must differ from the f32 one; it was identical")
    }

    // --- Oracle 3: the named refusals. ---------------------------------

    private fun dequantLossFn(): DxirFunction = DxirBuilder.function("dequantLoss") {
        val codes = param("codes", codesType)
        val scales = param("scales", perHeadScalesType)
        val pool = op(
            OpKind.DEQUANTIZE_KV, listOf(codes, scales), poolType,
            mapOf(DequantizeKvAttrs.DTYPE_ATTR to "int8"),
        )
        listOf(op(OpKind.SUM, listOf(pool), DxirType(F32, emptyList())))
    }

    private fun assertInferenceOnlyRefusal(ex: Throwable) {
        val msg = ex.message ?: ""
        assertTrue("DEQUANTIZE_KV" in msg, "refusal must name the kind; got: $msg")
        assertTrue("INFERENCE-ONLY" in msg, "refusal must state the rationale; got: $msg")
        assertTrue("STRAIGHT-THROUGH" in msg, "refusal must name the training-time fiction it declines; got: $msg")
        assertTrue("MUL" in msg, "refusal must name the differentiable alternative; got: $msg")
    }

    @Test
    fun reverseTransformRefusesDequantizeKvByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(dequantLossFn()) })
    }

    @Test
    fun forwardTransformRefusesDequantizeKvByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(dequantLossFn()) })
    }

    @Test
    fun kotlinRendererRefusesDequantizeKvAndPointsAtTheCodec() {
        val ex = assertFailsWith<KotlinRenderRefusal> { toKotlinSource(dequantFn()) }
        val msg = ex.message ?: ""
        assertTrue("DEQUANTIZE_KV" in msg, "render refusal must name the kind; got: $msg")
        assertTrue("KvQuantPool.dequantize" in msg, "render refusal must point at the host spelling; got: $msg")
    }

    @Test
    fun fp8IsRefusedByTheOpParserByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                dequantFn(tag = "fp8_e4m3"),
                listOf(FloatArray(poolElems), floatArrayOf(1f, 1f)),
            )
        }
        val msg = ex.message ?: ""
        assertTrue("fp8_e4m3" in msg, "must name the refused format; got: $msg")
        assertTrue("refused BY NAME" in msg, "must be a refusal, not a gap; got: $msg")
    }

    @Test
    fun aMissingDtypeAttrIsRefusedWithTheReasonItCannotBeDerived() {
        val fn = DxirBuilder.function("noTag") {
            val codes = param("codes", codesType)
            val scales = param("scales", perHeadScalesType)
            listOf(op(OpKind.DEQUANTIZE_KV, listOf(codes, scales), poolType, emptyMap()))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, listOf(FloatArray(poolElems), floatArrayOf(1f, 1f)))
        }
        assertTrue(
            "int4" in (ex.message ?: "") || "cannot tell" in (ex.message ?: ""),
            "must say why the shapes cannot supply it; got: ${ex.message}",
        )
    }

    @Test
    fun aScaleVectorOfTheWrongExtentIsRefusedAsAStrategyThatDoesNotExist() {
        val fn = DxirBuilder.function("badScales") {
            val codes = param("codes", codesType)
            val scales = param("scales", DxirType(F32, listOf(numKvHeads + 1)))
            listOf(
                op(
                    OpKind.DEQUANTIZE_KV, listOf(codes, scales), poolType,
                    mapOf(DequantizeKvAttrs.DTYPE_ATTR to "int8"),
                ),
            )
        }
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(fn, listOf(FloatArray(poolElems), FloatArray(numKvHeads + 1) { 1f }))
        }
        assertTrue("DERIVED" in (ex.message ?: ""), "must cite the derived-strategy rule; got: ${ex.message}")
    }

    @Test
    fun floatCodesAreRefusedBecauseCodesAreCodes() {
        val fn = DxirBuilder.function("floatCodes") {
            val codes = param("codes", poolType)
            val scales = param("scales", perHeadScalesType)
            listOf(
                op(
                    OpKind.DEQUANTIZE_KV, listOf(codes, scales), poolType,
                    mapOf(DequantizeKvAttrs.DTYPE_ATTR to "int8"),
                ),
            )
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, listOf(FloatArray(poolElems), floatArrayOf(1f, 1f)))
        }
        assertTrue("integer tensor" in (ex.message ?: ""), "must say codes are integers; got: ${ex.message}")
    }
}
