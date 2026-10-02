package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.maestro.serving.HfServingExport
import io.tlaloc.maestro.serving.ServingManifest
import io.tlaloc.maestro.serving.ServingRefusedToken
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.Params
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.TestBackend
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [ServingModel] on an artifact `HfServingExport` wrote from a tiny random
 * Qwen3 checkpoint (written by `HfCausalLm.save`, no download): greedy ids
 * equal to the Python serving path's (`run_llama_generate.py` on the same
 * artifact), the prefill and decode-only walks agree, a batch equals its
 * rows run alone, and the next-token logits agree with the interpreter's
 * forward of the same model.
 */
class ServingModelTest {

    private val tmp = Files.createTempDirectory("tlaloc-serving-model-")

    @AfterTest
    fun cleanup() {
        tmp.toFile().deleteRecursively()
    }

    private val json = """{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 32,
        "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "num_hidden_layers": 2, "vocab_size": 50, "rms_norm_eps": 1e-06, "rope_theta": 10000,
        "max_position_embeddings": 64, "tie_word_embeddings": true, "hidden_act": "silu",
        "attention_bias": false, "torch_dtype": "float32"}"""

    private val model = CausalLM.llama(
        CausalLmConfig(50, 32, 2, 4, 48, numKvHeads = 2, headDim = 8, normEps = 1e-6f, tiedEmbeddings = true,
            qkNorm = true, initStd = 0.5f),
        RandomKey.fromSeed(2026),
    )

    /** The checkpoint, exported; [prefill] adds prefill entries. */
    private fun artifact(name: String, prefill: Boolean): Path {
        val src = Files.createDirectories(tmp.resolve("src"))
        Files.writeString(src.resolve("config.json"), json)
        val ckpt = tmp.resolve("ckpt")
        if (!Files.isDirectory(ckpt)) HfCausalLm(model, HfDecoderConfig.parse(json)).save(src, ckpt, dtype = F32)
        val dir = tmp.resolve(name)
        HfCheckpoint.open(ckpt).use { c ->
            HfServingExport.export(
                ckpt = c, dir = dir,
                policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 32, blockSize = 4, minContext = 16),
                numBlocks = 24, modelName = "tiny-qwen3", prefill = prefill,
            )
        }
        return dir
    }

    private fun assumeGpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(PjrtBinaries.cudaAvailable, TestBackend.noDevice)
    }

    /** The interpreter's logits for [ids] at every position. */
    private fun interpreterLogits(ids: IntArray): FloatArray {
        val input = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(1, ids.size), I32)
        val params = model.parameters
        var out: Tracer<Shape>? = null
        captureN(listOf(input) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            model.forward(leaves[0], Params { byKey.getValue(it) }).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    private val prompt = intArrayOf(3, 17, 42, 8, 25, 11)

    @Test
    fun greedyIdsEqualThePythonServingPaths() {
        assumeGpu()
        val dir = artifact("with-prefill", prefill = true)
        val jvm = ServingModel.load(dir).use { m ->
            assertEquals("tiny-qwen3", m.modelName)
            assertEquals(32, m.maxContext)
            m.generate(prompt, 10)
        }
        // run_llama_generate.py walks the smallest context bucket (16 tokens here).
        val python = runPython(dir, prompt, 10)
        assumeTrue(python != null, "python3 cannot run harness/python/run_llama_generate.py here")
        println("[serving-model] JVM ${jvm.toList()}, run_llama_generate.py ${python!!.toList()}")
        assertContentEquals(python, jvm)
    }

    @Test
    fun prefillAndDecodeOnlyWalksAgreeAndLogitsMatchTheInterpreter() {
        assumeGpu()
        val withPrefill = ServingModel.load(artifact("with-prefill", prefill = true))
        val decodeOnly = ServingModel.load(artifact("decode-only", prefill = false))
        try {
            val a = withPrefill.generate(prompt, 10)
            val b = decodeOnly.generate(prompt, 10)
            assertContentEquals(a, b, "prefill and decode-only walks")
            // Twice on one model: the pools carry the previous request's KV,
            // which the next request never reads.
            assertContentEquals(a, withPrefill.generate(prompt, 10), "a second request on the same pools")

            val gpu = decodeOnly.nextTokenLogits(prompt)
            val host = interpreterLogits(prompt).let { it.copyOfRange(it.size - 50, it.size) }
            val gap = gpu.indices.maxOf { abs(gpu[it] - host[it]) }
            val scale = host.maxOf { abs(it) }
            println("[serving-model] next-token logits vs the interpreter: max |diff| $gap (largest |logit| $scale)")
            assertTrue(gap < 2e-2f * maxOf(1f, scale), "GPU and interpreter logits differ by $gap")

            // Greedy from the interpreter, token by token, where the top two logits are not a near tie.
            val ids = prompt.toMutableList()
            for (step in 0 until 10) {
                val row = interpreterLogits(ids.toIntArray()).let { it.copyOfRange(it.size - 50, it.size) }
                val sorted = row.sortedDescending()
                val best = row.indices.maxBy { row[it] }
                if (sorted[0] - sorted[1] < 0.05f) break
                assertEquals(best, a[step], "token $step")
                ids += best
            }
        } finally {
            withPrefill.close()
            decodeOnly.close()
        }
    }

    @Test
    fun aBatchEqualsItsRowsRunAlone() {
        assumeGpu()
        ServingModel.load(artifact("with-prefill", prefill = true)).use { m ->
            val prompts = arrayOf(prompt, intArrayOf(9, 1, 30), intArrayOf(44, 2, 7, 7, 19))
            val batch = m.generateBatch(prompts, 8)
            for ((i, p) in prompts.withIndex()) assertContentEquals(m.generate(p, 8), batch[i], "row $i")
            // A stop token ends a row and is kept.
            val stop = batch[0][2]
            val stopped = m.generate(prompt, 8, intArrayOf(stop))
            assertEquals(stop, stopped.last())
            assertTrue(stopped.size <= 3)
        }
    }

    @Test
    fun whatTheRunnerCannotServeIsRefusedByName() {
        val e = assertFailsWith<IllegalArgumentException> { ServingModel.load(tmp.resolve("nothing")) }
        assertTrue("not a Tlaloc serving artifact" in e.message!!, e.message)
        assumeGpu()
        ServingModel.load(artifact("with-prefill", prefill = true)).use { m ->
            assertFailsWith<IllegalArgumentException> { m.generate(IntArray(30) { 1 }, 8) }  // 38 > 32
            assertFailsWith<IllegalArgumentException> { m.generate(intArrayOf(1, 50), 2) }   // outside the vocabulary
            assertFailsWith<IllegalArgumentException> { m.generate(IntArray(0), 2) }
        }
    }

    @Test
    fun aGeneratedRefusedTokenIsRefusedWhenItIsFedBack() {
        assumeGpu()
        val dir = artifact("refused", prefill = true)
        val free = ServingModel.load(dir).use { it.generate(prompt, 3) }
        val t = free[0]
        assumeTrue(t !in prompt, "the first generated token $t is also in the prompt")
        val file = dir.resolve(ServingManifest.FILE_NAME)
        val m = ServingManifest.fromJson(Files.readString(file))
        Files.writeString(file, m.copy(model = m.model.copy(refusedTokens = listOf(ServingRefusedToken(t, "image_token_id")))).toJson())
        ServingModel.load(dir).use { model ->
            val e = assertFailsWith<IllegalArgumentException> { model.generate(prompt, 3) }
            assertTrue("$t, the model's image_token_id placeholder" in e.message!!, e.message)
            // The last generated token is not fed back, so it is returned.
            assertContentEquals(free.copyOf(1), model.generate(prompt, 1))
            assertFailsWith<IllegalArgumentException> { model.generate(intArrayOf(5, t), 2) }
            // A refusal happens before any device call: the model still serves.
            assertContentEquals(model.generate(intArrayOf(9, 1, 30), 4), ServingModel.load(dir).use { it.generate(intArrayOf(9, 1, 30), 4) })
        }
    }

    @Test
    fun aRequestAfterNonFiniteLogitsEqualsOneOnAFreshModel() {
        assumeGpu()
        // Untied, so an infinite embedding row reaches only the requests that use its token.
        val untied = json.replace("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false")
        val untiedModel = CausalLM.llama(
            CausalLmConfig(50, 32, 2, 4, 48, numKvHeads = 2, headDim = 8, normEps = 1e-6f, tiedEmbeddings = false,
                qkNorm = true, initStd = 0.5f),
            RandomKey.fromSeed(2026),
        )
        val src = Files.createDirectories(tmp.resolve("untied-src"))
        Files.writeString(src.resolve("config.json"), untied)
        val ckpt = tmp.resolve("untied-ckpt")
        HfCausalLm(untiedModel, HfDecoderConfig.parse(untied)).save(src, ckpt, dtype = F32)
        val dir = tmp.resolve("poisoned")
        HfCheckpoint.open(ckpt).use { c ->
            HfServingExport.export(
                ckpt = c, dir = dir,
                policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 32, blockSize = 4, minContext = 16),
                numBlocks = 24, modelName = "tiny-qwen3-untied", prefill = true,
            )
        }
        // Token 49's embedding row, 32 f32 values, set to +Inf.
        val embed = ServingManifest.fromJson(Files.readString(dir.resolve(ServingManifest.FILE_NAME)))
            .weights.table.first { it.name == "embedTokens" }
        java.nio.channels.FileChannel.open(dir.resolve(embed.path), java.nio.file.StandardOpenOption.WRITE).use { ch ->
            val row = java.nio.ByteBuffer.allocate(32 * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            repeat(32) { row.putFloat(Float.POSITIVE_INFINITY) }
            row.flip()
            ch.write(row, 49L * 32 * 4)
        }
        val fresh = ServingModel.load(dir).use { it.generate(prompt, 10) }
        ServingModel.load(dir).use { m ->
            // Longer than the next request, so its KV slots include some the next one never writes.
            val poisoned = intArrayOf(3, 49, 17, 42, 8, 25, 11, 5, 9, 12, 30, 2)
            assertTrue(m.nextTokenLogits(poisoned).any { !it.isFinite() }, "token 49 gives non-finite logits")
            assertContentEquals(fresh, m.generate(prompt, 10), "the next request, as on a fresh model")
            assertTrue(m.nextTokenLogits(prompt).all { it.isFinite() })
        }
    }

    @Test
    fun servingExportWritesAnArtifactServingModelLoads() {
        artifact("with-prefill", prefill = true) // writes the checkpoint
        val dir = ServingExport.export(tmp.resolve("ckpt"), tmp.resolve("exported"), 2, 40)
        val manifest = io.tlaloc.maestro.serving.ServingManifest.fromJson(
            Files.readString(dir.resolve(io.tlaloc.maestro.serving.ServingManifest.FILE_NAME)))
        assertEquals(16, manifest.bucketLadder.blockSize)
        assertEquals(listOf(1, 2), manifest.bucketLadder.batch)
        assertEquals(48, manifest.bucketLadder.maxContext) // 40 rounded up to whole pages
        assertEquals(1 + 2 * 3, manifest.model.numBlocks)
        assertEquals(null, manifest.model.windowedKv)
        assertTrue(manifest.entries.any { it.kind == io.tlaloc.ir.inference.DecodeGraphKind.PREFILL })
        assumeGpu()
        ServingModel.load(dir).use { m ->
            assertEquals(48, m.maxContext)
            ServingModel.load(artifact("with-prefill", prefill = true)).use { other ->
                assertContentEquals(other.generate(prompt, 8), m.generate(prompt, 8))
            }
        }
    }

    /** `run_llama_generate.py` (ctypes engine, stock python3) on [dir], or null when it cannot run here. */
    private fun runPython(dir: Path, prompt: IntArray, maxNew: Int): IntArray? {
        val harness = generateSequence(Path.of(System.getProperty("user.dir"))) { it.parent }
            .map { it.resolve("harness").resolve("python") }.first { Files.isDirectory(it) }
        val req = tmp.resolve("request.json")
        Files.writeString(req, """{"promptTokens": ${prompt.joinToString(",", "[", "]")}, "maxNewTokens": $maxNew}""")
        val out = tmp.resolve("python-out.json")
        val pb = ProcessBuilder("python3", harness.resolve("run_llama_generate.py").toString(),
            "--artifact", dir.toString(), "--request", req.toString(), "--output", out.toString(), "--platform", "cuda")
        pb.environment()["TLALOC_PJRT_PLUGIN_PATH"] = PjrtBinaries.pluginPath.toString()
        pb.environment()["PYTHONPATH"] = harness.toString()
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val log = proc.inputStream.bufferedReader().readText()
        assertTrue(proc.waitFor(600, TimeUnit.SECONDS), "run_llama_generate.py timed out")
        if (proc.exitValue() == 2) {
            println("[skip] run_llama_generate.py: $log")
            return null
        }
        assertEquals(0, proc.exitValue(), "run_llama_generate.py failed:\n$log")
        val o = parseJson(Files.readString(out)) as JsonObject
        return (o["generatedTokens"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
    }
}
