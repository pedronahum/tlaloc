package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Muse Glimmer family (text only).
 *
 * The real config is meta-models/Muse-Glimmer-30B's own `config.json`,
 * verbatim (`muse_glimmer_30b_config.json`). The arithmetic is certified
 * against transformers' own modeling code on a five-layer model
 * (`muse_glimmer_tiny.json`, written by `harness/python/muse_glimmer_fixture.py tiny`):
 * sliding windows of 3 over an 11-token prompt, two layers without RoPE,
 * gainless q/k norms, the q scale, the attention gate, the four (1 + w)
 * norms, the embedding norm and the soft-capped logits. Its weights are a
 * closed-form function of the tensor name and element index, so this test
 * rebuilds them bit for bit and no weight file is committed. The oracle runs
 * the arithmetic of the bf16-weight graph: bf16 weights, f32 activations,
 * each projection's input rounded to bf16.
 *
 * The fixture holds two sets of logits. `logitsF64Rope` is transformers with
 * its rotary tables computed in float64 and narrowed, which is how
 * [HfDecoderGraph.ropeTables] computes them; the graph must match it to f32
 * accuracy. `logits` is transformers unmodified, whose tables are computed in
 * float32; they differ from the float64 ones by an ulp here and there, and a
 * projection input one ulp from a bf16 rounding boundary then rounds the other
 * way. The two oracles differ by up to 8e-3 of the largest logit on this
 * 32-wide model, and the graph must be no further from `logits` than that
 * plus f32 noise, and pick the same token at every position.
 *
 * The real-weights claim is `triton/verify.sh`'s Muse Glimmer section.
 */
class HfMuseGlimmerTest {

    private fun resource(name: String): String =
        javaClass.getResourceAsStream(name)!!.use { it.readBytes().decodeToString() }

    private val realConfig: HfDecoderConfig by lazy { HfDecoderConfig.parse(resource("muse_glimmer_30b_config.json")) }

    // ------------------------------------------------------------- config

    @Test
    fun theRealConfigParsesAsTheMuseGlimmerFamily() {
        val c = realConfig
        assertEquals(HfModelFamily.MuseGlimmer, c.family)
        assertEquals(6656, c.hiddenSize)
        assertEquals(19968, c.intermediateSize)
        assertEquals(52, c.numLayers)
        assertEquals(32, c.numHeads)
        assertEquals(2, c.numKvHeads)
        assertEquals(128, c.headDim)
        assertEquals(202048, c.vocabSize)
        assertEquals(500000.0, c.ropeTheta)
        assertEquals(1e-5, c.rmsNormEps)
        assertEquals(1e-8, c.outputNormEps)
        assertEquals(20.0, c.finalLogitSoftcap)
        assertEquals(0.19611613513818404, c.logitMultiplier)
        assertEquals(3.87, c.queryScale)
        assertTrue(c.embeddingNorm && c.layerNormGainPlusOne)
        assertTrue(!c.tieWordEmbeddings)
        assertEquals(BF16, c.weightDType)
        assertEquals("silu", c.hiddenAct)
        assertEquals(mapOf(200092 to "image_token_id", 200091 to "video_token_id"), c.refusedTokenIds)
        assertTrue(c.unsupportedFeatures().isEmpty(), "${c.unsupportedFeatures()}")
        c.toDecodeModelShape(numBlocks = 4, blockSize = 16)
    }

    @Test
    fun threeSlidingLayersWithRopeThenOneFullLayerWithout() {
        val c = realConfig
        for (l in 0 until 52) {
            val s = c.layer(l)
            if (l % 4 == 3) {
                assertEquals(AttentionKind.FULL, s.attention, "layer $l")
                assertTrue(!s.rope, "layer $l has no RoPE")
            } else {
                assertEquals(AttentionKind.SLIDING, s.attention, "layer $l")
                assertEquals(2048, s.slidingWindow, "layer $l")
                assertTrue(s.rope, "layer $l has RoPE")
            }
            assertTrue(s.qkNorm && !s.qkNormGain && s.attentionOutputGate, "layer $l")
            assertTrue(s.postAttentionOutputNorm && s.postFeedforwardNorm, "layer $l")
        }
    }

    @Test
    fun twelveTensorsPerLayerAndTheVisionEncoderIsNotRead() {
        val c = realConfig
        val roles = HfDecoderNames.roles(c)
        // The file holds 1436 tensors: these 627 and 809 of the vision encoder.
        assertEquals(1 + 52 * 12 + 2, roles.size)
        val f = c.family
        fun name(l: Int, p: DecoderLayerPart) = HfDecoderNames.hfName(DecoderWeightRole.Layer(l, p), f)
        assertEquals("model.language_model.embed_tokens.weight", HfDecoderNames.hfName(DecoderWeightRole.EmbedTokens, f))
        assertEquals("model.language_model.norm.weight", HfDecoderNames.hfName(DecoderWeightRole.FinalNorm, f))
        assertEquals("lm_head.weight", HfDecoderNames.hfName(DecoderWeightRole.LmHead, f))
        assertEquals("model.language_model.layers.7.pre_feedforward_layernorm.weight", name(7, DecoderLayerPart.POST_ATTENTION_LAYERNORM))
        assertEquals("model.language_model.layers.7.post_attention_layernorm.weight", name(7, DecoderLayerPart.ATTENTION_OUTPUT_NORM))
        assertEquals("model.language_model.layers.7.post_feedforward_layernorm.weight", name(7, DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM))
        assertEquals("model.language_model.layers.7.self_attn.gate_proj.weight", name(7, DecoderLayerPart.ATTN_GATE_PROJ))
        for (r in roles) assertEquals(r, HfDecoderNames.role(HfDecoderNames.hfName(r, f), f))
        assertNull(HfDecoderNames.role("model.vision_tower.layers.0.attn.q_proj.weight", f))
        assertNull(HfDecoderNames.role("model.layers.0.self_attn.q_proj.weight", f))
        assertEquals(
            listOf(4096, 6656),
            HfDecoderNames.expectedDims(DecoderWeightRole.Layer(0, DecoderLayerPart.ATTN_GATE_PROJ), c).toList(),
        )
    }

    @Test
    fun unknownKeysAreRefusedByNameInsideAndOutsideTextConfig() {
        val text = resource("muse_glimmer_30b_config.json")
        val inner = text.replace("\"qk_scale_factor\": 3.87,", "\"qk_scale_factor\": 3.87, \"attn_logit_softcapping\": 50.0,")
        val e1 = assertFailsWith<JsonException> { HfDecoderConfig.parse(inner) }
        assertTrue("'attn_logit_softcapping'" in e1.message!!, e1.message!!)
        val outer = text.replace("\"image_token_id\": 200092,", "\"image_token_id\": 200092, \"audio_config\": {},")
        val e2 = assertFailsWith<JsonException> { HfDecoderConfig.parse(outer) }
        assertTrue("'audio_config'" in e2.message!!, e2.message!!)
    }

    @Test
    fun aLayerRopeThetaOtherThanZeroOrTheGlobalOneIsRefused() {
        val text = resource("muse_glimmer_30b_config.json")
        val bad = text.replaceFirst("\"layer_rope_theta\": [\n      500000.0,", "\"layer_rope_theta\": [\n      10000.0,")
        val e = assertFailsWith<JsonException> { HfDecoderConfig.parse(bad) }
        assertTrue("layer_rope_theta[0] = 10000.0" in e.message!!, e.message!!)
    }

    @Test
    fun imageAndVideoPlaceholdersAreRefusedByName() {
        val c = realConfig
        c.checkTextOnlyTokens(intArrayOf(200000, 791, 6593))
        val e = assertFailsWith<JsonException> { c.checkTextOnlyTokens(intArrayOf(200000, 200092, 3)) }
        assertTrue("image_token_id" in e.message!!, e.message!!)
        val v = assertFailsWith<JsonException> { c.checkTextOnlyTokens(intArrayOf(200091)) }
        assertTrue("video_token_id" in v.message!!, v.message!!)
    }

    // --------------------------------------------- the transformers oracle

    private val fixture: JsonObject by lazy { parseJson(resource("muse_glimmer_tiny.json")) as JsonObject }

    private val tiny: HfDecoderConfig by lazy {
        HfDecoderConfig.parse((fixture["configJson"] as JsonString).value)
    }

    private val prompt: IntArray by lazy {
        (fixture["prompt"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
    }

    private fun logits(key: String): List<DoubleArray> =
        (fixture[key] as JsonArray).elements.map { row ->
            (row as JsonArray).elements.map { (it as JsonNumber).value }.toDoubleArray()
        }

    /** transformers with float64 rotary tables: the graph's own arithmetic. */
    private val expected: List<DoubleArray> by lazy { logits("logitsF64Rope") }

    /** transformers unmodified. */
    private val transformers: List<DoubleArray> by lazy { logits("logits") }

    /** The fixture's weight formula, in the file's own layout: bf16 values as floats. */
    private fun closedForm(name: String, k: Int, dims: IntArray): FloatArray {
        val n = dims.fold(1) { a, b -> a * b }
        val (scale, offset) = when {
            name.endsWith("embed_tokens.weight") -> 1.0 to 0.0
            name.endsWith("language_model.norm.weight") -> 0.2 to 1.0
            dims.size == 1 -> 0.2 to 0.0
            else -> 0.35 to 0.0
        }
        return FloatArray(n) { i ->
            val m = (i.toLong() * 7919 + (k + 1).toLong() * 104729) % 2003
            val u = (m - 1001).toDouble() / 1001.0
            bf16BitsToFloat(floatToBf16Bits((offset + scale * u).toFloat()))
        }
    }

    /** Every text tensor of the tiny model, by HF name, in the file's layout. */
    private val fileTensors: Map<String, FloatArray> by lazy {
        val names = HfDecoderNames.roles(tiny).map { HfDecoderNames.hfName(it, tiny.family) }.sorted()
        names.withIndex().associate { (k, name) ->
            val role = HfDecoderNames.role(name, tiny.family)!!
            name to closedForm(name, k, HfDecoderNames.expectedDims(role, tiny))
        }
    }

    /** The staged weights of [config] (same roles as [tiny]), math layout. */
    private fun staged(config: HfDecoderConfig): List<FloatArray> =
        HfDecoderGraph.weightRoles(config).map { role ->
            val file = fileTensors.getValue(HfDecoderNames.hfName(role, config.family))
            if (HfDecoderNames.isTransposedLinear(role)) {
                val d = HfDecoderNames.expectedDims(role, config)
                HfStagedWeights.transpose(file, d[0], d[1])
            } else {
                file
            }
        }

    private val blockSize = 4
    private val numBlocks = 4
    private val context = numBlocks * blockSize
    private val table = intArrayOf(2, 0, 3, 1)
    private fun slotOf(pos: Int) = table[pos / blockSize] * blockSize + pos % blockSize
    private fun f(vararg v: Int) = FloatArray(v.size) { v[it].toFloat() }

    private fun graph(config: HfDecoderConfig, kind: DecodeGraphKind): DxirFunction {
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
        return HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
    }

    private fun emptyPools(config: HfDecoderConfig) =
        List(2 * config.numLayers) { FloatArray(numBlocks * blockSize * config.numKvHeads * config.headDim) }

    /**
     * Logits at every position of [prompt]: the first [prefillLen] tokens in
     * one prefill call (right-aligned in a chunk of [context] rows), then one
     * decode step per token. With [prefillLen] 0 every token is a decode step.
     */
    private fun run(
        config: HfDecoderConfig,
        prefillLen: Int = 0,
        weights: List<FloatArray> = staged(config),
    ): Map<Int, FloatArray> {
        var pools = emptyPools(config)
        val out = LinkedHashMap<Int, FloatArray>()
        if (prefillLen > 0) {
            val tokenIds = FloatArray(context)
            val positions = FloatArray(context)
            val slots = FloatArray(context) { -1f }
            val pad = context - prefillLen
            for (i in 0 until prefillLen) {
                tokenIds[pad + i] = prompt[i].toFloat()
                positions[pad + i] = i.toFloat()
                slots[pad + i] = slotOf(i).toFloat()
            }
            val res = DxirInterpreter.evalFunction(
                graph(config, DecodeGraphKind.PREFILL),
                buildList {
                    add(tokenIds); add(positions); add(f(*table)); add(f(prefillLen)); add(slots)
                    addAll(pools); addAll(weights)
                },
            )
            pools = res.drop(1)
            out[prefillLen - 1] = res[0]
        }
        val decode = graph(config, DecodeGraphKind.DECODE)
        for (pos in prefillLen until prompt.size) {
            val res = DxirInterpreter.evalFunction(
                decode,
                buildList {
                    add(f(prompt[pos])); add(f(pos)); add(f(*table)); add(f(pos + 1)); add(f(slotOf(pos)))
                    addAll(pools); addAll(weights)
                },
            )
            pools = res.drop(1)
            out[pos] = res[0]
        }
        return out
    }

    /** The worst difference from [oracle] over [got]'s positions, relative to max(1, max |logit|). */
    private fun worst(got: Map<Int, FloatArray>, oracle: List<DoubleArray> = expected): Double {
        var w = 0.0
        for ((pos, row) in got) {
            val want = oracle[pos]
            val denom = max(1.0, want.maxOf { abs(it) })
            for (v in want.indices) w = max(w, abs(row[v] - want[v]) / denom)
        }
        return w
    }

    /**
     * Tolerance against [expected]: the same arithmetic in a different order,
     * f32 throughout; an f32 difference that flipped a bf16 rounding would
     * show as 1e-4 or more on this model, and none does.
     */
    private val tol = 1e-5

    /** What each negative control must move the logits by, at least. */
    private val controlMoves = 1e-2

    @Test
    fun theTransformersStateDictNamesAreTheRolesNames() {
        val want = (fixture["textTensors"] as JsonArray).elements.map { (it as JsonString).value }
        assertEquals(want, fileTensors.keys.toList())
    }

    @Test
    fun theDecodeLoopMatchesTransformersAtEveryPosition() {
        val got = run(tiny)
        val w = worst(got)
        println("[muse-glimmer tiny] decode loop vs transformers (f64 RoPE tables): worst relative difference $w")
        assertTrue(w <= tol, "decode loop vs transformers: worst relative difference $w (tolerance $tol)")

        // Against unmodified transformers: no further than its own f32-vs-f64
        // table difference, and the same token at every position.
        val tables = (0 until prompt.size).associateWith { p -> FloatArray(tiny.vocabSize) { expected[p][it].toFloat() } }
        val spread = worst(tables, transformers)
        val wHf = worst(got, transformers)
        println("[muse-glimmer tiny] vs unmodified transformers: $wHf (its f32 vs f64 RoPE tables differ by $spread)")
        assertTrue(spread in 1e-4..1e-2, "the two oracles differ by $spread")
        assertTrue(wHf <= spread + tol, "graph vs unmodified transformers: $wHf, more than the tables' $spread")
        for ((pos, row) in got) {
            val argmax = row.indices.maxBy { row[it] }
            val want = transformers[pos].indices.maxBy { transformers[pos][it] }
            assertEquals(want, argmax, "argmax at position $pos")
        }
    }

    @Test
    fun prefillThenDecodeMatchesTransformers() {
        val got = run(tiny, prefillLen = 6)
        val w = worst(got)
        println("[muse-glimmer tiny] prefill of 6 then decode vs transformers: worst relative difference $w")
        assertTrue(w <= tol, "prefill + decode vs transformers: worst relative difference $w (tolerance $tol)")
    }

    private fun assertMoves(what: String, config: HfDecoderConfig) {
        val w = worst(run(config))
        assertTrue(w > controlMoves, "$what: the oracle should see it, but the logits moved by only $w")
    }

    /** Negative controls: each feature, removed, is visible to the oracle. */
    @Test
    fun withoutTheSlidingWindowTheLogitsMove() =
        assertMoves("full attention everywhere", tiny.copy(layers = tiny.layers!!.map { it.copy(attention = AttentionKind.FULL, slidingWindow = null) }))

    @Test
    fun withRopeOnEveryLayerTheLogitsMove() =
        assertMoves("RoPE on every layer", tiny.copy(layers = tiny.layers!!.map { it.copy(rope = true) }))

    @Test
    fun withoutSoftCappingTheLogitsMove() = assertMoves("no soft-capping", tiny.copy(finalLogitSoftcap = null))

    @Test
    fun withPlainGainsTheLogitsMove() = assertMoves("w instead of 1 + w", tiny.copy(layerNormGainPlusOne = false))

    @Test
    fun withoutTheQueryScaleTheLogitsMove() = assertMoves("no qk_scale_factor", tiny.copy(queryScale = 1.0))

    // -------------------------------------------------- staging the bytes

    /**
     * The streaming bf16 writer against the in-memory staging: a tiny
     * checkpoint written as bf16 safetensors, each slot written by
     * [HfStagedWeights.writeSlot] with a 64-byte block (so every Linear is
     * transposed in several bands, each read in several chunks), must be the
     * staged floats' bf16 bits.
     */
    /** [body] with the tiny model written as a bf16 safetensors checkpoint. */
    private fun <T> withCheckpoint(body: (HfCheckpoint) -> T): T {
        val dir = Files.createTempDirectory("muse-tiny")
        try {
            Files.writeString(dir.resolve("config.json"), (fixture["configJson"] as JsonString).value)
            val tensors = fileTensors.map { (name, data) ->
                val role = HfDecoderNames.role(name, tiny.family)!!
                SafetensorsTensor(
                    name, BF16, HfDecoderNames.expectedDims(role, tiny),
                    HostBf16Storage(ShortArray(data.size) { floatToBf16Bits(data[it]) }),
                )
            }
            SafetensorsFileWriter.write(dir.resolve("model.safetensors"), tensors)
            return HfCheckpoint.open(dir).use(body)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun theStreamingBf16WriterTransposesInBandsExactly() {
        withCheckpoint { ckpt ->
            assertTrue(ckpt.verifyInventory().isEmpty())
            val want = staged(tiny)
            for (i in want.indices) {
                val bytes = ByteArrayOutputStream()
                val n = HfStagedWeights.writeSlot(ckpt, tiny, i, bytes, blockBytes = 64)
                val b = bytes.toByteArray()
                assertEquals(2L * want[i].size, n)
                val got = FloatArray(want[i].size) {
                    val bits = (b[2 * it].toInt() and 0xFF) or ((b[2 * it + 1].toInt() and 0xFF) shl 8)
                    bf16BitsToFloat(bits.toShort())
                }
                assertContentEquals(want[i], got, "slot $i (${HfDecoderGraph.weightSlots(tiny)[i].name})")
            }
        }
    }

    // ------------------------------------------------------ int8 weights

    private val int8: HfDecoderConfig by lazy { tiny.copy(weightQuant = WeightQuant.INT8) }

    private val linearParts = setOf(
        DecoderLayerPart.Q_PROJ, DecoderLayerPart.K_PROJ, DecoderLayerPart.V_PROJ, DecoderLayerPart.O_PROJ,
        DecoderLayerPart.GATE_PROJ, DecoderLayerPart.UP_PROJ, DecoderLayerPart.DOWN_PROJ,
        DecoderLayerPart.ATTN_GATE_PROJ,
    )

    @Test
    fun int8QuantizesTheLayerLinearsOnlyAndAddsTheirScales() {
        assertEquals(HfDecoderGraph.weightSlots(tiny), HfDecoderGraph.weightSlots(tiny.copy(weightQuant = WeightQuant.NONE)))
        val roles = HfDecoderGraph.weightRoles(int8)
        val sources = HfDecoderGraph.weightSlotSources(int8)
        val slots = HfDecoderGraph.weightSlots(int8)
        val linears = roles.count { it is DecoderWeightRole.Layer && it.part in linearParts }
        assertTrue(linears >= 7 * int8.numLayers, "at least seven Linears per layer, got $linears")
        assertEquals(roles.size + linears, slots.size)
        assertEquals(roles, sources.filter { !it.scale }.map { it.role })
        for ((i, src) in sources.withIndex()) {
            val slot = slots[i]
            val isLinear = src.role is DecoderWeightRole.Layer && (src.role as DecoderWeightRole.Layer).part in linearParts
            when {
                src.scale -> {
                    assertEquals(HfDecoderGraph.slotName(src.role) + "Scale", slot.name)
                    assertEquals(io.tlaloc.core.F32, slot.type.dtype)
                    assertEquals(listOf(slots[i - 1].type.dims.last()), slot.type.dims, slot.name)
                    assertEquals(src.role, sources[i - 1].role)
                }
                isLinear -> assertEquals(io.tlaloc.core.I8, slot.type.dtype, slot.name)
                else -> assertEquals(tiny.weightDType, slot.type.dtype, slot.name)
            }
        }
    }

    /**
     * The quantizer against the formula written out here: per output channel
     * (a row of the file's `[out, in]`), `scale = max|w| / 127` and
     * `code = rint(w / scale)`, staged transposed. The streaming writer, with
     * a 64-byte block, writes the same codes and the scales' f32 bits.
     */
    @Test
    fun theInt8CodesAndScalesFollowTheFormulaAndStreamExactly() {
        withCheckpoint { ckpt ->
            val sources = HfDecoderGraph.weightSlotSources(int8)
            val slots = HfDecoderGraph.weightSlots(int8)
            val staged = HfStagedWeights.stage(ckpt, int8)
            var checked = 0
            for ((i, src) in sources.withIndex()) {
                if (!HfDecoderNames.isQuantized(src.role, int8)) continue
                val file = fileTensors.getValue(HfDecoderNames.hfName(src.role, int8.family))
                val (rows, cols) = HfDecoderNames.expectedDims(src.role, int8).let { it[0] to it[1] }
                val scales = FloatArray(rows) { r ->
                    val m = (0 until cols).maxOf { abs(file[r * cols + it]) }
                    if (m == 0f) 1f else m / 127f
                }
                val bytes = ByteArrayOutputStream()
                val n = HfStagedWeights.writeSlot(ckpt, int8, i, bytes, blockBytes = 64)
                val b = bytes.toByteArray()
                if (src.scale) {
                    assertEquals(4L * rows, n)
                    val buf = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    for (r in 0 until rows) {
                        assertEquals(scales[r].toRawBits(), staged[i][r].toRawBits(), "${slots[i].name}[$r]")
                        assertEquals(scales[r].toRawBits(), buf.getFloat(4 * r).toRawBits(), "${slots[i].name}[$r] written")
                    }
                } else {
                    assertEquals(rows.toLong() * cols, n)
                    for (r in 0 until rows) {
                        var maxCode = 0
                        for (c in 0 until cols) {
                            val code = Math.rint((file[r * cols + c] / scales[r]).toDouble()).toInt()
                            assertEquals(code.toFloat(), staged[i][c * rows + r], "${slots[i].name}[$c, $r]")
                            assertEquals(code.toByte(), b[c * rows + r], "${slots[i].name}[$c, $r] written")
                            maxCode = maxOf(maxCode, abs(code))
                        }
                        assertEquals(127, maxCode, "${slots[i].name}: the largest weight of row $r is code 127")
                    }
                    checked++
                }
            }
            assertTrue(checked >= 7 * int8.numLayers, "checked $checked quantized Linears")
        }
    }

    /**
     * The int8 graph computes `(x @ codes) * scales`: with f32 weights it
     * gives the logits of the unquantized graph fed `codes * scales`, to
     * f32 rounding. Against transformers (the unquantized model) it differs
     * by the quantization error, which is reported; a wrong scale moves it.
     */
    @Test
    fun theInt8GraphIsTheGraphOfTheDequantizedWeights() {
        withCheckpoint { ckpt ->
            val f32 = tiny.copy(weightDType = io.tlaloc.core.F32)
            val q32 = f32.copy(weightQuant = WeightQuant.INT8)
            val staged = HfStagedWeights.stage(ckpt, q32)
            val sources = HfDecoderGraph.weightSlotSources(q32)
            val slots = HfDecoderGraph.weightSlots(q32)
            val dequantized = sources.indices.filter { !sources[it].scale }.map { i ->
                if (!HfDecoderNames.isQuantized(sources[i].role, q32)) {
                    staged[i]
                } else {
                    val out = slots[i].type.dims.last()
                    val scales = staged[i + 1]
                    FloatArray(staged[i].size) { staged[i][it] * scales[it % out] }
                }
            }
            val quantized = run(q32, weights = staged)
            val reference = run(f32, weights = dequantized)
            var w = 0.0
            for ((pos, row) in quantized) {
                val ref = reference.getValue(pos)
                val denom = max(1.0, ref.maxOf { abs(it).toDouble() })
                for (v in row.indices) w = max(w, abs(row[v] - ref[v]) / denom)
            }
            assertTrue(w <= tol, "int8 graph vs dequantized weights: worst relative difference $w")
            val cost = worst(quantized)
            println("[muse-glimmer tiny] int8 weights: $w from the dequantized graph, $cost from transformers")
            // The tiny model's weights are spread evenly over their range, the
            // hardest case for 127 levels per row: a few percent of the largest logit.
            assertTrue(cost > 0.0 && cost < 0.1, "int8 vs transformers: $cost")
            // bf16 activations with int8 weights: the default path of a quantized Muse Glimmer.
            val bf16 = run(int8, weights = HfStagedWeights.stage(ckpt, int8))
            println("[muse-glimmer tiny] int8 weights, bf16 activations: ${worst(bf16)} from transformers")
            assertTrue(worst(bf16) < 0.15, "int8 with bf16 activations vs transformers: ${worst(bf16)}")
            // Control: the scales of the first quantized Linear reversed.
            val first = sources.indexOfFirst { it.scale }
            val wrong = staged.toMutableList().also { it[first] = staged[first].reversedArray() }
            val moved = worst(run(q32, weights = wrong), oracle = quantized.let { m ->
                List(prompt.size) { p -> m.getValue(p).map { it.toDouble() }.toDoubleArray() }
            })
            assertTrue(moved > controlMoves, "reversed scales moved the logits by only $moved")
        }
    }
}
