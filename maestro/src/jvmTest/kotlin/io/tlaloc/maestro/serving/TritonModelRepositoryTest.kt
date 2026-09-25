package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.maestro.TypeDescriptor
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The Triton model a serving artifact becomes: its `config.pbtxt`, checked as
 * golden text and slot by slot against the manifest, and the files written
 * into the model version directory.
 */
class TritonModelRepositoryTest {

    private val tritonType = mapOf("i32" to "TYPE_INT32", "f32" to "TYPE_FP32")

    @Test
    fun referenceDecodeConfigIsTheGoldenText() {
        withReferenceArtifact { _, manifest ->
            val bodies = manifest.entries.map { it.bodyPath }.distinct().joinToString(", ")
            val expected = """
                # Written by Tlaloc from the serving artifact 'tlaloc-reference-decode'
                # (reference-decode-lcg-v1), 6 entries: decode_b1_c2, decode_b1_c4, decode_b2_c2, decode_b2_c4, decode_b4_c2, decode_b4_c4.
                # The KV pools are state of the model instance: zero at start, updated by every
                # request. The client allocates pages through blockTables and slotMapping.
                name: "reference_decode"
                backend: "tlaloc"
                max_batch_size: 0
                input [
                  { name: "tokenIds" data_type: TYPE_INT32 dims: [ -1, 1 ] },
                  { name: "positions" data_type: TYPE_INT32 dims: [ -1, 1 ] },
                  { name: "blockTables" data_type: TYPE_INT32 dims: [ -1, -1 ] },
                  { name: "seqLens" data_type: TYPE_INT32 dims: [ -1 ] },
                  { name: "slotMapping" data_type: TYPE_INT32 dims: [ -1 ] }
                ]
                output [
                  { name: "logits" data_type: TYPE_FP32 dims: [ -1, 1, 11 ] }
                ]
                instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]
                parameters: { key: "artifact" value: { string_value: "$bodies" } }
                parameters: { key: "entry" value: { string_value: "main" } }
                parameters: { key: "arguments" value: { string_value: "input:tokenIds, input:positions, input:blockTables, input:seqLens, input:slotMapping, state:keyCache0, state:valueCache0" } }
                parameters: { key: "results" value: { string_value: "output:logits, state:keyCache0, state:valueCache0" } }
                parameters: { key: "kv_block_size" value: { string_value: "2" } }
                parameters: { key: "kv_num_blocks" value: { string_value: "6" } }

            """.trimIndent()
            assertEquals(expected, TritonModelRepository.config(manifest, "reference_decode"))
        }
    }

    @Test
    fun everyManifestSlotAppearsWithItsDtypeAndDims() {
        withReferenceArtifact { _, manifest ->
            val config = TritonModelRepository.config(manifest, "reference_decode")
            val first = manifest.entries.first()
            for ((index, slot) in first.inputs.withIndex()) {
                when (slot.role) {
                    DecodeSlotRole.KV_POOL_IN -> assertTrue("state:${slot.name}" in config)
                    else -> {
                        val dims = mergedDims(manifest.entries.map { it.inputs[index].type.dims })
                        val line = "{ name: \"${slot.name}\" data_type: ${tritonType.getValue(slot.type.dtype)} " +
                            "dims: [ ${dims.joinToString(", ")} ] }"
                        assertTrue(line in config, "missing input line: $line")
                        assertTrue("input:${slot.name}" in config)
                    }
                }
            }
            for ((index, slot) in first.outputs.withIndex()) {
                when (slot.role) {
                    DecodeSlotRole.LOGITS -> {
                        val dims = mergedDims(manifest.entries.map { it.outputs[index].type.dims })
                        val line = "{ name: \"${slot.name}\" data_type: ${tritonType.getValue(slot.type.dtype)} " +
                            "dims: [ ${dims.joinToString(", ")} ] }"
                        assertTrue(line in config, "missing output line: $line")
                        assertTrue("output:${slot.name}" in config)
                    }
                    else -> {
                        // A KV_POOL_OUT writes back the KV_POOL_IN its donation pair names.
                        val input = first.inputs[first.donationPairs.single { it[1] == index }[0]]
                        assertTrue(slot.name == input.name + "Out")
                    }
                }
            }
            // Positional order: the arguments and results lists follow the
            // entry signature one for one.
            val arguments = parameter(config, "arguments").split(", ")
            assertEquals(first.inputs.map { it.name }, arguments.map { it.substringAfter(':') })
            val results = parameter(config, "results").split(", ")
            assertEquals(first.outputs.size, results.size)
            // Every body is served.
            for (e in manifest.entries) assertTrue(e.bodyPath in parameter(config, "artifact"))
        }
    }

    @Test
    fun stagedWeightsBecomeWeightArgumentsByTheirFiles() {
        val weight = ServingWeightFile(
            name = "embedTokens", path = "weights/0000_embedTokens.bin", dtype = "f32",
            dims = listOf(11, 8), byteLength = 11L * 8 * 4, sha256 = "0".repeat(64),
        )
        val manifest = syntheticManifest(
            extraInputs = listOf(ServingSlot("embedTokens", DecodeSlotRole.WEIGHT, TypeDescriptor("f32", listOf(11, 8)))),
            weights = ServingWeightsPointer(ServingWeightsPointer.STAGED_FORMAT, "weights", false, listOf(weight)),
        )
        val config = TritonModelRepository.config(manifest, "tiny")
        assertEquals(
            "input:tokenIds, state:keyCache0, weight:weights/0000_embedTokens.bin",
            parameter(config, "arguments"),
        )
        assertTrue("# 1 staged weights are loaded onto the device when the model loads." in config)
        assertTrue("embedTokens\" data_type" !in config, "a weight must not be a request tensor")
    }

    @Test
    fun refusesWhatTheBackendCouldNotBind() {
        withReferenceArtifact { _, manifest ->
            assertFailsWith<IllegalArgumentException> {
                TritonModelRepository.config(manifest, "../escape")
            }
        }
        val unnamedWeight = syntheticManifest(
            extraInputs = listOf(ServingSlot("w", DecodeSlotRole.WEIGHT, TypeDescriptor("f32", listOf(2)))),
        )
        val e = assertFailsWith<IllegalArgumentException> {
            TritonModelRepository.config(unnamedWeight, "tiny")
        }
        assertTrue("WEIGHT slot 'w' is not in the manifest's weight table" in e.message!!)
        val unpaired = syntheticManifest(donationPairs = emptyList())
        val u = assertFailsWith<IllegalArgumentException> { TritonModelRepository.config(unpaired, "tiny") }
        assertTrue("has no donation pair" in u.message!!)
    }

    @Test
    fun writeLaysOutTheModelDirectory() {
        withReferenceArtifact { artifact, manifest ->
            val repo = Files.createTempDirectory("tlaloc-triton-repo")
            try {
                val model = TritonModelRepository.write(artifact, repo, "reference_decode")
                assertEquals(repo.resolve("reference_decode"), model)
                val config = Files.readString(model.resolve("config.pbtxt"))
                assertEquals(TritonModelRepository.config(manifest, "reference_decode"), config)
                val version = model.resolve("1")
                assertTrue(Files.isRegularFile(version.resolve(ServingManifest.FILE_NAME)))
                for (e in manifest.entries) {
                    assertEquals(
                        Files.readString(artifact.resolve(e.bodyPath)),
                        Files.readString(version.resolve(e.bodyPath)),
                    )
                    assertTrue(Files.isRegularFile(version.resolve(e.programPath)))
                }
                // Writing again replaces the files in place.
                TritonModelRepository.write(artifact, repo, "reference_decode")
                assertEquals(config, Files.readString(model.resolve("config.pbtxt")))
            } finally {
                repo.toFile().deleteRecursively()
            }
        }
    }

    // ---------------------------------------------------------------------

    private fun withReferenceArtifact(body: (Path, ServingManifest) -> Unit) {
        val dir = Files.createTempDirectory("tlaloc-reference-artifact")
        try {
            body(dir, ReferenceDecodeGraph.exportTo(dir))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    private fun mergedDims(all: List<List<Int>>): List<Int> =
        all.first().indices.map { d -> all.map { it[d] }.toSet().singleOrNull() ?: -1 }

    private fun parameter(config: String, key: String): String =
        Regex("key: \"$key\" value: \\{ string_value: \"([^\"]*)\" \\}").find(config)!!.groupValues[1]

    private fun syntheticManifest(
        extraInputs: List<ServingSlot> = emptyList(),
        weights: ServingWeightsPointer = ServingWeightsPointer.embedded(),
        donationPairs: List<List<Int>> = listOf(listOf(1, 1)),
    ): ServingManifest {
        val pool = TypeDescriptor("f32", listOf(6, 2, 2, 2))
        val entry = ServingEntry(
            kind = DecodeGraphKind.DECODE, batch = 1, context = 2, tokensPerSeq = 1,
            maxBlocksPerSeq = 1, cacheKey = "k", entryPoint = "main",
            bodyPath = "bodies/a.mlir", bodyHash = "a", programPath = "programs/decode_b1_c2.json",
            inputs = listOf(
                ServingSlot("tokenIds", DecodeSlotRole.TOKEN_IDS, TypeDescriptor("i32", listOf(1, 1))),
                ServingSlot("keyCache0", DecodeSlotRole.KV_POOL_IN, pool),
            ) + extraInputs,
            outputs = listOf(
                ServingSlot("logits", DecodeSlotRole.LOGITS, TypeDescriptor("f32", listOf(1, 1, 11))),
                ServingSlot("keyCache0Out", DecodeSlotRole.KV_POOL_OUT, pool),
            ),
            donationPairs = donationPairs,
        )
        val reference = ReferenceDecodeGraph.MODEL
        return ServingManifest(
            modelName = "tiny", modelHash = "tiny-hash",
            model = ServingModelShape(
                vocabSize = 11, hiddenSize = 8, numHeads = 4, numKvHeads = 2, headDim = 2,
                numLayers = 1, numBlocks = reference.numBlocks, blockSize = reference.blockSize,
                dtype = "f32", kvDtype = "f32",
            ),
            bucketLadder = ServingBucketLadder(blockSize = 2, batch = listOf(1), context = listOf(2)),
            weights = weights,
            entries = listOf(entry),
        )
    }
}
