package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.TypeDescriptor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.469 — Phase H3a: the serving manifest is a WIRE FORMAT read by a
 * process that cannot ask a question. These are the pins that say it
 * survives the crossing and that it refuses a malformed artifact by name
 * rather than half-reading it.
 */
class ServingManifestTest {

    private fun slot(name: String, role: DecodeSlotRole, dtype: String, dims: List<Int>) =
        ServingSlot(name, role, TypeDescriptor(dtype, dims))

    private fun entry(batch: Int = 4, context: Int = 4) = ServingEntry(
        kind = DecodeGraphKind.DECODE,
        batch = batch,
        context = context,
        tokensPerSeq = 1,
        maxBlocksPerSeq = context / 2,
        cacheKey = "tlaloc-decode-v1/h/decode/b$batch/c$context/t1/dtF32/kvF32",
        entryPoint = "main",
        bodyPath = "bodies/abc.mlir",
        bodyHash = "abc",
        programPath = "programs/decode_b${batch}_c$context.json",
        inputs = listOf(
            slot("tokenIds", DecodeSlotRole.TOKEN_IDS, "i32", listOf(batch, 1)),
            slot("keyCache0", DecodeSlotRole.KV_POOL_IN, "f32", listOf(6, 2, 2, 2)),
        ),
        outputs = listOf(
            slot("logits", DecodeSlotRole.LOGITS, "f32", listOf(batch, 1, 11)),
            slot("keyCache0Out", DecodeSlotRole.KV_POOL_OUT, "f32", listOf(6, 2, 2, 2)),
        ),
        donationPairs = listOf(listOf(1, 1)),
    )

    private fun manifest(entries: List<ServingEntry> = listOf(entry())) = ServingManifest(
        modelName = "toy",
        modelHash = "h",
        model = ServingModelShape(
            vocabSize = 11, hiddenSize = 8, numHeads = 4, numKvHeads = 2, headDim = 2,
            numLayers = 1, numBlocks = 6, blockSize = 2, dtype = "f32", kvDtype = "f32",
        ),
        bucketLadder = ServingBucketLadder(blockSize = 2, batch = listOf(1, 2, 4), context = listOf(2, 4)),
        weights = ServingWeightsPointer.embedded(),
        entries = entries,
    )

    @Test
    fun aManifestRoundTripsThroughItsOwnJson() {
        val m = manifest(listOf(entry(4, 4), entry(2, 2)))
        val back = ServingManifest.fromJson(m.toJson())
        assertEquals(m, back, "the serving manifest must survive its own serialization intact")
        // And the second crossing is byte-stable: a content-addressed artifact
        // whose manifest re-serialized differently would be a different artifact.
        assertEquals(m.toJson(), back.toJson())
    }

    @Test
    fun slotsKeepTheirRolesAndTypesAcrossTheWire() {
        val back = ServingManifest.fromJson(manifest().toJson())
        val e = back.entryFor(DecodeGraphKind.DECODE, 4, 4)
        assertEquals(DecodeSlotRole.TOKEN_IDS, e.inputs[0].role)
        assertEquals(listOf(4, 1), e.inputs[0].type.dims)
        assertEquals("i32", e.inputs[0].type.dtype)
        assertEquals(DecodeSlotRole.KV_POOL_OUT, e.outputs[1].role)
    }

    @Test
    fun theKvPoolLayoutIsStatedAndNotLeftToBeInferred() {
        val m = manifest()
        assertEquals(listOf("numBlocks", "blockSize", "numKvHeads", "headDim"), m.model.kvPoolAxisOrder)
        assertEquals(listOf(6, 2, 2, 2), m.model.kvPoolDims)
        assertTrue(m.toJson().contains("\"kvPoolAxisOrder\""))
    }

    /**
     * §0.4.472 — Phase H5: the long-reserved `kvQuant` slot, finally carrying
     * a value, and the two facts it keeps apart — what the codes MEAN
     * (`dtype = "int8"`) and what they RIDE (`codeDtype = "i32"`, the v1
     * deferral said out loud in the artifact rather than only in a doc).
     */
    @Test
    fun aQuantizedKvPoolRoundTripsIncludingTheCodeDtypeDeferral() {
        val m = manifest().copy(
            model = ServingModelShape(
                vocabSize = 11, hiddenSize = 8, numHeads = 4, numKvHeads = 2, headDim = 2,
                numLayers = 1, numBlocks = 6, blockSize = 2, dtype = "f32", kvDtype = "i32",
                kvQuant = ServingKvQuant(
                    dtype = "int8", scaleStrategy = ServingKvQuant.PER_HEAD,
                    codeMax = 127, codeDtype = "i32",
                ),
            ),
        )
        val back = ServingManifest.fromJson(m.toJson())
        assertEquals(m, back, "a quantized artifact must survive its own serialization")
        val q = back.model.kvQuant!!
        assertEquals("int8", q.dtype, "what the codes MEAN")
        assertEquals("i32", q.codeDtype, "what the codes RIDE — the v1 deferral, in the artifact")
        assertEquals(127, q.codeMax)
        assertEquals(2, q.scaleCount(numKvHeads = 2), "per-head scaling needs one scale per kv head")
        assertEquals(1, q.copy(scaleStrategy = ServingKvQuant.PER_TENSOR).scaleCount(2))
    }

    /** An artifact written before H5 has no `kvQuant` field, and still loads. */
    @Test
    fun anArtifactWithNoKvQuantFieldStillReadsAsUnquantized() {
        val json = manifest().toJson().replace("\"kvQuant\":null,", "")
        assertEquals(null, ServingManifest.fromJson(json).model.kvQuant)
    }

    @Test
    fun entryForNamesTheCompiledPointsWhenAskedForOneThatIsNotThere() {
        val e = assertFailsWith<IllegalArgumentException> {
            manifest().entryFor(DecodeGraphKind.DECODE, 8, 4)
        }
        assertTrue(e.message!!.contains("(4,4)"), "the refusal must list what IS compiled: ${e.message}")
    }

    @Test
    fun anEntryOffTheDeclaredLadderIsRefused() {
        val e = assertFailsWith<IllegalArgumentException> { manifest(listOf(entry(batch = 3))) }
        assertTrue(e.message!!.contains("not a point of the declared ladder"), e.message!!)
    }

    @Test
    fun duplicateLadderPointsAreRefused() {
        val e = assertFailsWith<IllegalArgumentException> { manifest(listOf(entry(), entry())) }
        assertTrue(e.message!!.contains("duplicate ladder points"), e.message!!)
    }

    @Test
    fun aDonationPairThatAliasesMismatchedTypesIsRefused() {
        val e = assertFailsWith<IllegalArgumentException> {
            entry().copy(donationPairs = listOf(listOf(0, 1)))
        }
        assertTrue(e.message!!.contains("written IN PLACE"), e.message!!)
    }

    @Test
    fun anUnalignedContextLadderIsRefused() {
        val e = assertFailsWith<IllegalArgumentException> {
            ServingBucketLadder(blockSize = 2, batch = listOf(1), context = listOf(3))
        }
        assertTrue(e.message!!.contains("whole number of pages"), e.message!!)
    }

    @Test
    fun aWeightsPointerThatClaimsBothEmbeddedAndAPathIsRefused() {
        val e = assertFailsWith<IllegalArgumentException> {
            ServingWeightsPointer("safetensors", "weights/model.safetensors", embedded = true)
        }
        assertTrue(e.message!!.contains("has not decided"), e.message!!)
        // The staged form is legal and round-trips.
        val staged = ServingWeightsPointer("safetensors", "weights/model.safetensors", embedded = false)
        assertEquals(staged, ServingWeightsPointer.fromJson(io.tlaloc.core.io.parseJson(staged.toJson()) as io.tlaloc.core.io.JsonObject))
    }

    @Test
    fun anArtifactOfAnUnknownSchemaVersionIsRefusedRatherThanPartlyRead() {
        val bad = manifest().toJson().replace(ServingManifest.SCHEMA_VERSION, "tlaloc-serving-v99")
        val e = assertFailsWith<IllegalArgumentException> { ServingManifest.fromJson(bad) }
        assertTrue(e.message!!.contains("tlaloc-serving-v99"), e.message!!)
    }

    @Test
    fun aVersionOneArtifactIsStillReadAndTheWriterWritesVersionTwo() {
        assertEquals("tlaloc-serving-v2", manifest().schemaVersion)
        val v1 = manifest().toJson().replace(ServingManifest.SCHEMA_VERSION, ServingManifest.SCHEMA_VERSION_1)
        assertEquals(ServingManifest.SCHEMA_VERSION_1, ServingManifest.fromJson(v1).schemaVersion)
    }

    @Test
    fun prefillEntriesRoundTripAndAreRefusedInAVersionOneArtifact() {
        val prefill = entry(1, 4).copy(
            kind = DecodeGraphKind.PREFILL,
            tokensPerSeq = 4,
            programPath = "programs/prefill_b1_c4.json",
        )
        val m = manifest(listOf(entry(1, 4), prefill))
        val back = ServingManifest.fromJson(m.toJson())
        assertEquals(DecodeGraphKind.PREFILL, back.entryFor(DecodeGraphKind.PREFILL, 1, 4).kind)
        val v1 = m.toJson().replace(ServingManifest.SCHEMA_VERSION, ServingManifest.SCHEMA_VERSION_1)
        val e = assertFailsWith<IllegalArgumentException> { ServingManifest.fromJson(v1) }
        assertTrue(e.message!!.contains("prefill entries"), e.message!!)
    }

    @Test
    fun aMissingFieldIsNamedRatherThanDefaulted() {
        val bad = manifest().toJson().replace("\"modelHash\":", "\"modelHashX\":")
        val e = assertFailsWith<io.tlaloc.core.io.JsonException> { ServingManifest.fromJson(bad) }
        assertTrue(e.message!!.contains("modelHash"), e.message!!)
    }

    @Test
    fun anUnknownSlotRoleIsRefusedByName() {
        val bad = manifest().toJson().replace("\"TOKEN_IDS\"", "\"TOKEN_IDZ\"")
        val e = assertFailsWith<io.tlaloc.core.io.JsonException> { ServingManifest.fromJson(bad) }
        assertTrue(e.message!!.contains("TOKEN_IDZ"), e.message!!)
    }
}
