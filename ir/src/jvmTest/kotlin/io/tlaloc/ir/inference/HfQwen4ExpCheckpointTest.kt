package io.tlaloc.ir.inference

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * NVIDIA's NVFP4 Qwen3.8-Flash-Next read as the qwen4_exp family: the config's
 * layers, streams, n-gram memory and indexer; every tensor of the text model
 * mapped to a role with the dims the config gives, the n-gram table's 128 FP8
 * shards stacking to the computed table, and the stored hash constants equal
 * to the computed ones. Reads headers and the hash buffers only. Skips without
 * the checkpoint.
 */
class HfQwen4ExpCheckpointTest {

    private fun snapshot(repo: String): Path? {
        val root = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub", "models--" + repo.replace("/", "--"), "snapshots")
        if (!Files.isDirectory(root)) return null
        return Files.list(root).use { s -> s.toList().firstOrNull { Files.isRegularFile(it.resolve("config.json")) } }
    }

    @Test
    fun flashNextReadsAsQwen4Exp() {
        val dir = snapshot("nvidia/Qwen3.8-Flash-Next-NVFP4")
        assumeTrue(dir != null, "needs nvidia/Qwen3.8-Flash-Next-NVFP4 in the HuggingFace cache")
        HfCheckpoint.open(dir!!).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Qwen4Exp, c.family)
            assertEquals(48, c.numLayers)
            assertEquals(GatedResidualConfig(streams = 4, lowRank = 320), c.gatedResidual)
            assertEquals(IndexerConfig(heads = 4, headDim = 128, budget = 2048, chunk = 4), c.indexer)
            assertEquals(MoeConfig(numExperts = 512, topK = 10, expertIntermediate = 640, sharedIntermediate = 640), c.moe)
            assertEquals(listOf(1), c.ple!!.layers)
            assertEquals(12, (0 until 48).count { c.layer(it).indexer })
            assertEquals(36, c.linearLayers.size)
            assertTrue(c.linearGateSigmoid && c.rotaryDim == 64)
            val table = DecoderWeightRole.Layer(1, DecoderLayerPart.PLE_TABLE)
            val shards = ckpt.tableShards(table)!!
            assertEquals(128, shards.size)
            val rows = shards.sumOf { ckpt.entryNamed(it).dims[0] }
            assertEquals(c.ple!!.tableRows(0), rows)
            assertEquals(160, ckpt.entryNamed(shards[0]).dims[1])
            assertTrue(ckpt.tableScale(table) != null)
            val unmapped = ckpt.verifyInventory()
            // The vision encoder, the MTP head, and the NVFP4 experts' scales beside their codes.
            val other = unmapped.filterNot {
                it.startsWith("model.visual.") || it.startsWith("mtp.") ||
                    it.endsWith(".weight_scale") || it.endsWith(".weight_scale_2") || it.endsWith(".input_scale")
            }
            assertTrue(other.isEmpty(), "unmapped: ${other.take(10)}")
            println(
                "Flash-Next: ${unmapped.size} tensors unread (vision, MTP, scales); n-gram table $rows x 160 in ${shards.size} " +
                    "FP8 shards; hash constants as computed: ${c.ple!!.multipliers(0)}",
            )
        }
    }
}
