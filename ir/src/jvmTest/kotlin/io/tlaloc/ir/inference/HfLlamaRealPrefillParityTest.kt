package io.tlaloc.ir.inference

import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The prefill graph against the decode loop on the first two layers of the
 * real TinyLlama-1.1B checkpoint, in the reference interpreter.
 *
 * `HfLlamaRealDecodeParityTest` certifies the decode loop against
 * HuggingFace transformers on the same reduced slice. This test certifies the
 * prefill chunk against that decode loop: the last position's logits and
 * every KV pool, bit for bit, for the prompt "The capital of France is".
 *
 * Skips by name when the checkpoint is absent
 * (`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`, or
 * `TLALOC_HF_LLAMA_CHECKPOINT`).
 */
class HfLlamaRealPrefillParityTest {

    /** BOS + "The capital of France is", TinyLlama's tokenizer. */
    private val prompt = intArrayOf(1, 450, 7483, 310, 3444, 338)

    private val layers = 2
    private val blockSize = 8
    private val numBlocks = 4
    private val context = 16

    private fun checkpointDir(): Path? {
        val env = System.getenv("TLALOC_HF_LLAMA_CHECKPOINT")
        val p = if (env != null) Path.of(env) else Path.of(
            System.getProperty("user.home"),
            ".cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0",
        )
        return if (Files.isDirectory(p)) p else null
    }

    @Test
    fun aRealPrefillChunkGivesTheDecodeLoopsLogitsAndPools() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no TinyLlama checkpoint — see HfLlamaCheckpointTest for the fetch command")

        HfCheckpoint.open(dir!!).use { ckpt ->
            val config = ckpt.config.copy(numLayers = layers)
            val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
            val bucket = DecodeBucket(batch = 1, maxContext = context)
            val decode = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket), config)
            val prefill = HfDecoderGraph.build(
                HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL), config,
            )
            val weights = HfStagedWeights.stage(ckpt, config)
            val poolSize = numBlocks * blockSize * model.numKvHeads * model.headDim
            // Pages 2 and 3: the sequence does not start at page 0.
            val table = floatArrayOf(2f, 3f)
            fun slot(pos: Int) = (table[pos / blockSize].toInt() * blockSize + pos % blockSize).toFloat()

            var pools = List(2 * layers) { FloatArray(poolSize) }
            var loopLogits = FloatArray(0)
            for (i in prompt.indices) {
                val out = DxirInterpreter.evalFunction(
                    decode,
                    buildList {
                        add(floatArrayOf(prompt[i].toFloat()))
                        add(floatArrayOf(i.toFloat()))
                        add(table)
                        add(floatArrayOf((i + 1).toFloat()))
                        add(floatArrayOf(slot(i)))
                        addAll(pools)
                        addAll(weights)
                    },
                )
                loopLogits = out[0]
                pools = out.drop(1)
            }

            // Right-aligned chunk: 10 padding rows, then the 6 prompt tokens.
            val pad = context - prompt.size
            val tokenIds = FloatArray(context)
            val positions = FloatArray(context)
            val slots = FloatArray(context) { -1f }
            for (i in prompt.indices) {
                tokenIds[pad + i] = prompt[i].toFloat()
                positions[pad + i] = i.toFloat()
                slots[pad + i] = slot(i)
            }
            val out = DxirInterpreter.evalFunction(
                prefill,
                buildList {
                    add(tokenIds); add(positions); add(table)
                    add(floatArrayOf(prompt.size.toFloat())); add(slots)
                    addAll(List(2 * layers) { FloatArray(poolSize) })
                    addAll(weights)
                },
            )
            assertEquals(config.vocabSize, out[0].size)
            assertEquals(argmax(loopLogits), argmax(out[0]), "argmax of the last position")
            assertContentEquals(loopLogits, out[0], "last-position logits")
            for (p in pools.indices) assertContentEquals(pools[p], out[1 + p], "pool $p")
            println(
                "real prefill parity: ${prompt.size} tokens x $layers layers, one prefill call == " +
                    "${prompt.size} decode steps bit for bit, argmax ${argmax(out[0])}",
            )
        }
    }

    private fun argmax(v: FloatArray): Int {
        var best = 0
        for (i in v.indices) if (v[i] > v[best]) best = i
        return best
    }
}
