package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.AttentionKind
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.DecoderLayerSpec
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.maestro.serving.TritonModelRepository.KvMode.CLIENT
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * An artifact whose sliding-window layers keep their KV in a windowed pool:
 * it is `tlaloc-serving-v3`, states the pool class and the layers in it, gives
 * those layers' pools their own roles and dims, adds the window block tables
 * and slot mapping to every entry, aliases every pool to its output, and
 * becomes a Triton model that states the ring's geometry. Without the pool
 * the same model is a v2 artifact, unchanged.
 */
class WindowedArtifactExportTest {

    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM", modelType = "llama",
        hiddenSize = 16, intermediateSize = 24, numLayers = 3,
        numHeads = 4, numKvHeads = 2, headDim = 4, vocabSize = 23,
        rmsNormEps = 1e-5, ropeTheta = 10000.0, maxPositionEmbeddings = 64,
        tieWordEmbeddings = false, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null,
        layers = listOf(
            DecoderLayerSpec(attention = AttentionKind.FULL),
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 8),
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 5),
        ),
    )

    private fun export(dir: Path, windowed: Boolean): ServingManifest {
        val policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 32, blockSize = 4, minContext = 16)
        val window = if (windowed) config.windowedKvPool(4, 32, 17) else null
        val model = config.toDecodeModelShape(numBlocks = 17, blockSize = 4, windowedKv = window)
        val specs = policy.allBuckets.map { HfDecoderGraph.spec(config, model, it) } +
            policy.contextLadder.map { HfDecoderGraph.spec(config, model, DecodeBucket(1, it), DecodeGraphKind.PREFILL) }
        val rng = Random(7)
        return ServingArtifactWriter.export(
            dir = dir, modelName = "tiny-window", modelHash = "tiny-window-hash", model = model,
            ladder = ServingArtifactWriter.ladderOf(policy), specs = specs,
            stageWeight = { slot -> FloatArray(slot.type.dims.fold(1) { a, b -> a * b }) { rng.nextFloat() - 0.5f } },
            build = { HfDecoderGraph.build(it, config) },
        )
    }

    private fun <T> withArtifact(windowed: Boolean, body: (Path, ServingManifest) -> T): T {
        val dir = Files.createTempDirectory("tlaloc-window-artifact")
        try {
            return body(dir, export(dir, windowed))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun theWindowedArtifactIsVersionThreeAndStatesItsPoolClass() = withArtifact(true) { dir, m ->
        assertEquals(ServingManifest.SCHEMA_VERSION_3, m.schemaVersion)
        val w = m.model.windowedKv!!
        assertEquals(ServingWindowedKv(window = 8, layers = listOf(1, 2), numBlocks = 1 + 2 * 3, ringPages = 3), w)
        val text = Files.readString(dir.resolve(ServingManifest.FILE_NAME))
        assertTrue("\"windowedKv\":{\"window\":8,\"layers\":[1,2],\"numBlocks\":7,\"ringPages\":3,\"kvPoolDims\":[7,4,2,4]}" in text, text)
        assertEquals(m, ServingManifest.fromJson(text))
        for (e in m.entries) {
            assertEquals(
                listOf(
                    DecodeSlotRole.TOKEN_IDS, DecodeSlotRole.POSITIONS, DecodeSlotRole.BLOCK_TABLES,
                    DecodeSlotRole.SEQ_LENS, DecodeSlotRole.SLOT_MAPPING,
                    DecodeSlotRole.WINDOW_BLOCK_TABLES, DecodeSlotRole.WINDOW_SLOT_MAPPING,
                ) + List(2) { DecodeSlotRole.KV_POOL_IN } + List(4) { DecodeSlotRole.WINDOW_KV_POOL_IN },
                e.inputs.take(13).map { it.role },
                e.entryId,
            )
            assertEquals(listOf(17, 4, 2, 4), e.inputs[7].type.dims)
            assertEquals(listOf(7, 4, 2, 4), e.inputs[9].type.dims)
            assertEquals((0 until 6).map { listOf(7 + it, 1 + it) }, e.donationPairs)
            val body = Files.readString(dir.resolve(e.bodyPath))
            // Every pool, windowed or not, is written over its own input.
            for (i in 0 until 6) assertTrue("tf.aliasing_output = ${1 + i} : i32" in body, "${e.entryId}: pool $i")
        }
    }

    @Test
    fun withoutTheWindowedPoolTheSameModelIsAnUnchangedVersionTwoArtifact() = withArtifact(false) { dir, m ->
        assertEquals(ServingManifest.SCHEMA_VERSION, m.schemaVersion)
        assertNull(m.model.windowedKv)
        val text = Files.readString(dir.resolve(ServingManifest.FILE_NAME))
        assertFalse("windowedKv" in text || "WINDOW_" in text)
        assertEquals((0 until 6).map { listOf(5 + it, 1 + it) }, m.entries.first().donationPairs)
    }

    @Test
    fun aVersionThatDoesNotMatchThePoolClassesIsRefusedByName() {
        withArtifact(true) { _, m ->
            val e = assertFailsWith<IllegalArgumentException> { m.copy(schemaVersion = ServingManifest.SCHEMA_VERSION) }
            assertTrue("windowed pools are defined in tlaloc-serving-v3" in e.message!!, e.message!!)
            val e2 = assertFailsWith<IllegalArgumentException> { m.copy(model = m.model.copy(windowedKv = null)) }
            assertTrue("with a windowed KV pool" in e2.message!! || "without" in e2.message!!, e2.message!!)
        }
        withArtifact(false) { _, m ->
            val e = assertFailsWith<IllegalArgumentException> { m.copy(schemaVersion = ServingManifest.SCHEMA_VERSION_3) }
            assertTrue("without a windowed KV pool" in e.message!!, e.message!!)
        }
    }

    @Test
    fun theTritonModelStatesTheRingAndBindsTheWindowSlots() = withArtifact(true) { _, m ->
        val sequence = TritonModelRepository.config(m, "tiny_window")
        for (line in listOf(
            "{ name: \"KV_PAGES\" data_type: TYPE_INT32 dims: [ 2 ] }",
            "max_candidate_sequences: 6",
            "parameters: { key: \"kv_window\" value: { string_value: \"8\" } }",
            "parameters: { key: \"kv_window_layers\" value: { string_value: \"1,2\" } }",
            "parameters: { key: \"kv_window_num_blocks\" value: { string_value: \"7\" } }",
            "parameters: { key: \"kv_window_ring_pages\" value: { string_value: \"3\" } }",
        )) assertTrue(line in sequence, "$line\n$sequence")
        val client = TritonModelRepository.config(m, "tiny_window", CLIENT)
        assertTrue(
            "input:slotMapping, input:windowBlockTables, input:windowSlotMapping, state:keyCache0, " +
                "state:valueCache0, state:keyCache1" in client,
            client,
        )
        assertTrue("output:logits, state:keyCache0, state:valueCache0, state:keyCache1, state:valueCache1, state:keyCache2" in client, client)
        assertTrue("{ name: \"windowBlockTables\" data_type: TYPE_INT32 dims: [ -1, -1 ] }" in client, client)
    }
}
