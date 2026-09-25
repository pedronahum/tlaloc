package io.tlaloc.ir.inference

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 * The prefill graph against the decode loop, in the reference interpreter.
 *
 * A prefill chunk runs a prompt's tokens as rows of one call; the decode loop
 * runs them one step at a time. Every op in the graph is row-independent
 * except PAGED_ATTENTION, and there each prefill row reads the same pool
 * positions with the same causal length the matching decode step read. So the
 * last row's logits and the pool the chunk leaves behind are expected to be
 * BIT-IDENTICAL to the decode loop's, not merely close, and they are pinned
 * with `==`.
 *
 * The model is a small GQA Llama with random weights (no checkpoint), so this
 * runs everywhere. `HfLlamaRealPrefillParityTest` runs the same comparison on
 * two real TinyLlama layers when the checkpoint is present.
 */
class HfLlamaPrefillTest {

    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM",
        modelType = "llama",
        hiddenSize = 16,
        intermediateSize = 40,
        numLayers = 2,
        numHeads = 4,
        numKvHeads = 2,
        headDim = 4,
        vocabSize = 23,
        rmsNormEps = 1e-5,
        ropeTheta = 10000.0,
        maxPositionEmbeddings = 64,
        tieWordEmbeddings = false,
        attentionBias = false,
        torchDtype = "float32",
        ropeScalingType = null,
    )
    private val blockSize = 4
    private val numBlocks = 12
    private val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)

    /** Deterministic random weights, scaled so the logits are not flat. */
    private val weights: List<FloatArray> = run {
        val rng = Random(20260925)
        HfDecoderGraph.weightSlots(config).map { slot ->
            val n = slot.type.dims.fold(1) { a, b -> a * b }
            if (slot.type.dims.size == 1) {
                FloatArray(n) { 1f + 0.1f * (rng.nextFloat() - 0.5f) } // norm gains
            } else {
                FloatArray(n) { 0.6f * (rng.nextFloat() - 0.5f) }
            }
        }
    }

    private val poolSize = numBlocks * blockSize * model.numKvHeads * model.headDim

    private fun emptyPools() = List(2 * config.numLayers) { FloatArray(poolSize) }

    private fun graph(kind: DecodeGraphKind, batch: Int, context: Int): DxirFunction =
        HfDecoderGraph.build(
            HfDecoderGraph.spec(config, model, DecodeBucket(batch, context), kind),
            config,
        )

    private val context = 16
    private val maxBlocks = context / blockSize
    private val decode = graph(DecodeGraphKind.DECODE, 1, context)
    private val prefill16 = graph(DecodeGraphKind.PREFILL, 1, context)

    private fun slotOf(table: IntArray, pos: Int) = table[pos / blockSize] * blockSize + pos % blockSize

    private fun f(vararg v: Int) = FloatArray(v.size) { v[it].toFloat() }

    private class Step(val logits: FloatArray, val pools: List<FloatArray>)

    /** One decode step of one sequence. */
    private fun decodeStep(token: Int, pos: Int, table: IntArray, pools: List<FloatArray>): Step {
        val out = DxirInterpreter.evalFunction(
            decode,
            buildList {
                add(f(token))
                add(f(pos))
                add(FloatArray(maxBlocks) { table[it].toFloat() })
                add(f(pos + 1))
                add(f(slotOf(table, pos)))
                addAll(pools)
                addAll(weights)
            },
        )
        return Step(out[0], out.drop(1))
    }

    private fun decodeLoop(tokens: IntArray, table: IntArray, start: Int = 0, pools0: List<FloatArray> = emptyPools()): Step {
        var pools = pools0
        var logits = FloatArray(0)
        for (i in tokens.indices) {
            val s = decodeStep(tokens[i], start + i, table, pools)
            logits = s.logits
            pools = s.pools
        }
        return Step(logits, pools)
    }

    /**
     * One prefill chunk per sequence, right-aligned: [tokens] of sequence `s`
     * start at position `starts[s]`; padding rows come first.
     */
    private fun prefillCall(
        fn: DxirFunction,
        t: Int,
        seqs: List<Pair<IntArray, IntArray>>, // (tokens, block table)
        starts: IntArray,
        pools: List<FloatArray>,
        rightAligned: Boolean = true,
    ): Step {
        val b = seqs.size
        val tokenIds = FloatArray(b * t)
        val positions = FloatArray(b * t)
        val slots = FloatArray(b * t) { -1f }
        val tables = FloatArray(b * maxBlocks)
        val seqLens = FloatArray(b)
        for ((s, pair) in seqs.withIndex()) {
            val (tokens, table) = pair
            val pad = t - tokens.size
            for (i in tokens.indices) {
                val row = s * t + if (rightAligned) pad + i else i
                tokenIds[row] = tokens[i].toFloat()
                positions[row] = (starts[s] + i).toFloat()
                slots[row] = slotOf(table, starts[s] + i).toFloat()
            }
            for (j in 0 until maxBlocks) tables[s * maxBlocks + j] = table[j].toFloat()
            seqLens[s] = (starts[s] + tokens.size).toFloat()
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            buildList {
                add(tokenIds); add(positions); add(tables); add(seqLens); add(slots)
                addAll(pools)
                addAll(weights)
            },
        )
        return Step(out[0], out.drop(1))
    }

    private val prompt = intArrayOf(1, 7, 3, 19, 11, 5, 2)
    private val table = intArrayOf(3, 7, 5, 9) // deliberately not 0..3 and not ascending

    @Test
    fun thePrefillSignatureIsTheDecodeSignatureWithALongerTokenAxis() {
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL)
        assertEquals(listOf(1, context), spec.tokenIdsType.dims)
        assertEquals(listOf(context), spec.slotMappingType.dims)
        assertEquals(listOf(1, 1, config.vocabSize), spec.logitsType.dims, "last position only")
        assertEquals(decode.params.map { it.name }, prefill16.params.map { it.name })
        assertEquals(decode.returns.size, prefill16.returns.size)
    }

    @Test
    fun aPrefillChunkGivesTheDecodeLoopsLogitsAndPoolsBitForBit() {
        val loop = decodeLoop(prompt, table)
        val pre = prefillCall(prefill16, context, listOf(prompt to table), intArrayOf(0), emptyPools())
        assertEquals(config.vocabSize, pre.logits.size)
        assertContentEquals(loop.logits, pre.logits, "last-position logits")
        // The pools match too. With two layers, a layer-1 K/V at position i
        // depends on layer 0's attention at row i, so a row that saw a later
        // token would leave a different pool: this is the causal mask's check.
        for (p in loop.pools.indices) {
            assertContentEquals(loop.pools[p], pre.pools[p], "pool $p")
        }
        assertEquals(argmax(loop.logits), argmax(pre.logits))
    }

    @Test
    fun aPromptPrefilledInTwoChunksGivesTheSameAnswer() {
        // Chunk 1: positions 0..3 in a 16-token chunk; chunk 2: positions 4..6.
        val first = prompt.copyOfRange(0, 4)
        val rest = prompt.copyOfRange(4, prompt.size)
        val a = prefillCall(prefill16, context, listOf(first to table), intArrayOf(0), emptyPools())
        val b = prefillCall(prefill16, context, listOf(rest to table), intArrayOf(4), a.pools)
        val loop = decodeLoop(prompt, table)
        assertContentEquals(loop.logits, b.logits)
        for (p in loop.pools.indices) assertContentEquals(loop.pools[p], b.pools[p], "pool $p")

        // And decoding on from a prefilled pool continues the decode loop.
        val nextToken = argmax(b.logits)
        val afterPrefill = decodeStep(nextToken, prompt.size, table, b.pools)
        val afterLoop = decodeStep(nextToken, prompt.size, table, loop.pools)
        assertContentEquals(afterLoop.logits, afterPrefill.logits)
    }

    @Test
    fun twoSequencesInOnePrefillCallAreEachTheirOwnDecodeLoop() {
        val fn = graph(DecodeGraphKind.PREFILL, 2, context)
        val other = intArrayOf(4, 4, 17)
        val otherTable = intArrayOf(1, 2, 0, 0)
        val both = prefillCall(fn, context, listOf(prompt to table, other to otherTable), intArrayOf(0, 0), emptyPools())
        val v = config.vocabSize
        assertContentEquals(decodeLoop(prompt, table).logits, both.logits.copyOfRange(0, v))
        assertContentEquals(decodeLoop(other, otherTable).logits, both.logits.copyOfRange(v, 2 * v))
    }

    /** Negative control: the same tokens left-aligned put padding in the last
     *  row, and the graph must then NOT reproduce the decode loop. */
    @Test
    fun aLeftAlignedChunkDoesNotGiveTheDecodeLoopsAnswer() {
        val loop = decodeLoop(prompt, table)
        val left = prefillCall(
            prefill16, context, listOf(prompt to table), intArrayOf(0), emptyPools(), rightAligned = false,
        )
        assertFalse(loop.logits.contentEquals(left.logits), "left-aligned chunk matched the decode loop")
        // The pools still match: the writes do not depend on row order.
        for (p in loop.pools.indices) assertContentEquals(loop.pools[p], left.pools[p])
    }

    private fun argmax(v: FloatArray): Int {
        var best = 0
        for (i in v.indices) if (v[i] > v[best]) best = i
        return best
    }
}
