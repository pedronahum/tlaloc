package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.parseJson
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * §0.4.478 (H3c-1) — HF Llama ingestion against files.
 *
 * Two lanes, deliberately separate:
 *
 * **A. HERMETIC.** Tiny safetensors checkpoints the test writes itself, byte
 * by byte, to certify the cases a single downloaded model cannot show: TIED
 * embeddings (TinyLlama is untied), the config/file disagreement about tying
 * in BOTH directions, and a transposed projection caught by name. These run
 * everywhere and are the ones that gate `./gradlew test` on a fresh machine.
 *
 * **B. THE REAL CHECKPOINT.** TinyLlama-1.1B-Chat-v1.0, 2.2 GB of bf16, at
 * `~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`
 * (override with `TLALOC_HF_LLAMA_CHECKPOINT`). SELF-SKIPS when absent — the
 * checkpoint is not in the repo and never will be. Fetch it with:
 *
 * ```
 * ~/.local/venvs/vllm/bin/python -c "from huggingface_hub import snapshot_download; \
 *   snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
 *     local_dir='$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0', \
 *     allow_patterns=['*.json','*.safetensors','tokenizer*'])"
 * ```
 *
 * Lane B carries the claim the whole slice exists for: **HF stores Linear
 * weights TRANSPOSED as `[out_features, in_features]`**, checked against a
 * model whose GQA projections (`[256, 2048]`) and MLP (`[5632, 2048]`,
 * `[2048, 5632]`) are RECTANGULAR — a square model cannot distinguish the two
 * conventions, and asserting the convention against one would be asserting
 * nothing. The VALUES are spot-checked against torch reading the same file
 * (`harness/python/read_hf_llama_probes.py` in `~/.local/venvs/vllm`), at
 * **exact raw bit patterns, no tolerance**: reading bytes is not arithmetic.
 */
class HfLlamaCheckpointTest {

    // ---------------------------------------------------------------- lane A

    @Test
    fun tiedEmbeddingsResolveTheHeadToTheEmbeddingTable() {
        val dir = writeCheckpoint(tied = true, includeLmHead = false)
        HfLlamaCheckpoint.open(dir).use { c ->
            assertTrue(c.config.tieWordEmbeddings)
            assertTrue(c.tiedEmbeddings)
            assertFalse(c.lmHeadPresent)
            assertEquals(
                HfLlamaNames.EMBED_TOKENS,
                c.resolveName(LlamaWeightRole.LmHead),
                "a tied head must read the embedding table's bytes",
            )
            // Every other role still resolves to its own name.
            assertEquals(
                "model.layers.1.mlp.up_proj.weight",
                c.resolveName(LlamaWeightRole.Layer(1, LlamaLayerPart.UP_PROJ)),
            )
            assertTrue(c.verifyInventory().isEmpty())
            val head = c.load(LlamaWeightRole.LmHead)
            val embed = c.load(LlamaWeightRole.EmbedTokens)
            assertEquals(HfLlamaNames.EMBED_TOKENS, head.name)
            assertEquals(listOf(TINY_VOCAB, TINY_HIDDEN), head.dims.toList())
            assertContentEquals(embed.toF32Array(), head.toF32Array())
        }
    }

    @Test
    fun anUntiedCheckpointKeepsTheHeadSeparate() {
        val dir = writeCheckpoint(tied = false, includeLmHead = true)
        HfLlamaCheckpoint.open(dir).use { c ->
            assertFalse(c.tiedEmbeddings)
            assertEquals(HfLlamaNames.LM_HEAD, c.resolveName(LlamaWeightRole.LmHead))
            assertTrue(c.verifyInventory().isEmpty())
            // Different bytes, so the two roles are genuinely distinct.
            val head = c.load(LlamaWeightRole.LmHead).toF32Array()
            val embed = c.load(LlamaWeightRole.EmbedTokens).toF32Array()
            assertTrue(head.indices.any { head[it] != embed[it] })
        }
    }

    @Test
    fun aConfigThatDisagreesWithItsOwnFileAboutTyingIsRefusedBothWays() {
        val saysTiedButHasHead = writeCheckpoint(tied = true, includeLmHead = true)
        val e1 = assertFailsWith<JsonException> { HfLlamaCheckpoint.open(saysTiedButHasHead) }
        assertTrue("tie_word_embeddings=true" in e1.message!!, e1.message!!)
        assertTrue("PRESENT" in e1.message!!, e1.message!!)

        val saysUntiedButHasNoHead = writeCheckpoint(tied = false, includeLmHead = false)
        val e2 = assertFailsWith<JsonException> { HfLlamaCheckpoint.open(saysUntiedButHasNoHead) }
        assertTrue("tie_word_embeddings=false" in e2.message!!, e2.message!!)
        assertTrue("ABSENT" in e2.message!!, e2.message!!)
    }

    @Test
    fun aTransposedProjectionIsCaughtAtLoadWithTheConventionNamed() {
        // k_proj written [hidden, kvProjOut] instead of [kvProjOut, hidden] —
        // exactly what a loader that assumed the OTHER convention would accept.
        val dir = writeCheckpoint(
            tied = false,
            includeLmHead = true,
            overrideDims = mapOf(
                "model.layers.0.self_attn.k_proj.weight" to intArrayOf(TINY_HIDDEN, TINY_KV_OUT),
            ),
        )
        HfLlamaCheckpoint.open(dir).use { c ->
            val e = assertFailsWith<JsonException> {
                c.load(LlamaWeightRole.Layer(0, LlamaLayerPart.K_PROJ))
            }
            assertTrue("k_proj" in e.message!!, e.message!!)
            assertTrue("[$TINY_HIDDEN, $TINY_KV_OUT]" in e.message!!, e.message!!)
            assertTrue("[$TINY_KV_OUT, $TINY_HIDDEN]" in e.message!!, e.message!!)
            assertTrue("TRANSPOSED" in e.message!!, e.message!!)
        }
    }

    @Test
    fun aMissingTensorIsNamedByVerifyInventory() {
        val dir = writeCheckpoint(tied = false, includeLmHead = true, drop = setOf("model.norm.weight"))
        HfLlamaCheckpoint.open(dir).use { c ->
            val e = assertFailsWith<JsonException> { c.verifyInventory() }
            assertTrue("model.norm.weight" in e.message!!, e.message!!)
        }
    }

    @Test
    fun anUnmappedTensorIsReportedAsAFactNotAnError() {
        val dir = writeCheckpoint(
            tied = false,
            includeLmHead = true,
            extra = mapOf("model.layers.0.self_attn.rotary_emb.inv_freq" to intArrayOf(4)),
        )
        HfLlamaCheckpoint.open(dir).use { c ->
            assertEquals(
                listOf("model.layers.0.self_attn.rotary_emb.inv_freq"),
                c.verifyInventory(),
            )
        }
    }

    @Test
    fun aDirectoryWithNoConfigIsRefusedByName() {
        val dir = createTempDirectory("hf-llama-noconfig").toFile().also { it.deleteOnExit() }.toPath()
        val e = assertFailsWith<JsonException> { HfLlamaCheckpoint.open(dir) }
        assertTrue("config.json" in e.message!!, e.message!!)
    }

    // ---------------------------------------------------------------- lane B

    @Test
    fun theRealTinyLlamaCheckpointMatchesItsConfigTensorForTensor() {
        val dir = realCheckpoint() ?: return
        HfLlamaCheckpoint.open(dir).use { c ->
            val cfg = c.config
            assertEquals("LlamaForCausalLM", cfg.architecture)
            assertEquals(2048, cfg.hiddenSize)
            assertEquals(5632, cfg.intermediateSize)
            assertEquals(22, cfg.numLayers)
            assertEquals(32, cfg.numHeads)
            assertEquals(4, cfg.numKvHeads)
            assertEquals(64, cfg.headDim)
            assertEquals(32000, cfg.vocabSize)
            assertEquals(1e-5, cfg.rmsNormEps)
            assertEquals(10000.0, cfg.ropeTheta)
            assertEquals(BF16, cfg.storageDType)
            assertFalse(cfg.tieWordEmbeddings)
            assertFalse(c.tiedEmbeddings)

            // Every role present, nothing in the file left unmapped: 201
            // tensors and 201 roles, a total bijection.
            assertEquals(201, HfLlamaNames.roles(cfg).size)
            assertEquals(emptyList(), c.verifyInventory())
            assertEquals(201, c.names.size)

            // THE LAYOUT CLAIM, against the header of the real file, for all
            // 201 tensors at once: expectedDims — which encodes [out, in] —
            // equals what the producer wrote, everywhere.
            SafetensorsFile.open(dir.resolve("model.safetensors")).use { f ->
                for (role in HfLlamaNames.roles(cfg)) {
                    val e = f.header.entry(c.resolveName(role))
                    assertEquals(
                        HfLlamaNames.expectedDims(role, cfg).toList(),
                        e.dims,
                        "layout disagreement at $role (${e.name})",
                    )
                    assertEquals("BF16", e.wireDType, "dtype at ${e.name}")
                }
                // And the four RECTANGULAR witnesses, spelled out: under the
                // other convention each of these would be its own reverse.
                assertEquals(listOf(256, 2048), f.header.entry(K0).dims)
                assertEquals(listOf(256, 2048), f.header.entry(V0).dims)
                assertEquals(listOf(5632, 2048), f.header.entry(GATE0).dims)
                assertEquals(listOf(2048, 5632), f.header.entry(DOWN0).dims)
            }

            val shape = cfg.toDecodeModelShape(numBlocks = 128, blockSize = 16)
            assertEquals(22, shape.numLayers)
            assertEquals(4, shape.numKvHeads)
            assertEquals(8, shape.numHeads / shape.numKvHeads)
            assertEquals(F32, shape.dtype)
        }
    }

    @Test
    fun theRealCheckpointsBytesAreTheOnesTorchSees() {
        val dir = realCheckpoint() ?: return
        val python = vllmPython() ?: run {
            println("[skip] no ~/.local/venvs/vllm python for the torch weight-value oracle")
            return
        }

        // Probes chosen to exercise the offset arithmetic, not just the first
        // tensor: a rank-1 norm, the first element of a GQA projection, a deep
        // layer's interior, the LAST element of the largest tensor, and the
        // final element of the (untied) head.
        val probes = listOf(
            NORM to 2047,
            K0 to 0,
            "model.layers.11.self_attn.v_proj.weight" to 123_456,
            DOWN0 to 5631,
            "model.embed_tokens.weight" to 32_000 * 2048 - 1,
            "lm_head.weight" to 32_000 * 2048 - 1,
        )

        val oracle = runOracle(python, dir, probes) ?: run {
            println("[skip] the torch weight-value oracle did not run")
            return
        }
        assertEquals(201, (oracle["tensor_count"] as JsonNumber).asInt("tensor_count"))

        HfLlamaCheckpoint.open(dir).use { c ->
            val tensors = (oracle["tensors"] as JsonObject)
            for ((name, index) in probes) {
                val role = HfLlamaNames.role(name)!!
                val t = c.load(role)
                assertEquals(BF16, t.dtype, name)
                val o = tensors.obj(name)
                assertEquals(
                    (o.arr("shape")).asIntList("shape"),
                    t.dims.toList(),
                    "shape disagreement at $name",
                )
                assertEquals(o.reqInt("numel"), t.size, "numel at $name")
                val want = ((o.obj("probes")[index.toString()]) as JsonNumber)
                    .asInt("probes[$index]")
                val got = (t.storage as HostBf16Storage).data[index].toInt() and 0xFFFF
                assertEquals(
                    want,
                    got,
                    "raw bf16 bits at $name[$index]: torch says 0x${want.toString(16)}, " +
                        "Tlaloc's reader says 0x${got.toString(16)} — EXACT is the floor here",
                )
            }
            // The whole-file dtype/shape tables agree too, which is the header
            // read cross-checked against a second implementation.
            val shapes = oracle.obj("shapes")
            val dtypes = oracle.obj("dtypes")
            for (role in HfLlamaNames.roles(c.config)) {
                val n = c.resolveName(role)
                assertEquals(
                    HfLlamaNames.expectedDims(role, c.config).toList(),
                    shapes.arr(n).asIntList(n),
                    "torch's shape for $n",
                )
                assertEquals("BF16", (dtypes[n] as JsonString).value)
            }
        }
    }

    // ------------------------------------------------------------- machinery

    private fun realCheckpoint(): Path? {
        val env = System.getenv("TLALOC_HF_LLAMA_CHECKPOINT")
        val dir = if (env != null) Path.of(env) else Path.of(
            System.getProperty("user.home") ?: return null,
            ".cache", "tlaloc-checkpoints", "TinyLlama__TinyLlama-1.1B-Chat-v1.0",
        )
        if (!Files.isRegularFile(dir.resolve("config.json"))) {
            println("[skip] no HF Llama checkpoint at $dir — see this test's KDoc for the fetch command")
            return null
        }
        return dir
    }

    private fun vllmPython(): Path? {
        val p = Path.of(
            System.getProperty("user.home") ?: return null,
            ".local", "venvs", "vllm", "bin", "python",
        )
        return if (Files.isExecutable(p)) p else null
    }

    private fun runOracle(python: Path, dir: Path, probes: List<Pair<String, Int>>): JsonObject? {
        val script = Path.of("..", "harness", "python", "read_hf_llama_probes.py")
            .toAbsolutePath().normalize()
        if (!Files.isRegularFile(script)) return null
        val cmd = buildList {
            add(python.toString()); add(script.toString()); add(dir.toString())
            for ((n, i) in probes) add("$n@$i")
        }
        val out = Files.createTempFile("hf-probes", ".json")
        val err = Files.createTempFile("hf-probes", ".err")
        try {
            val p = ProcessBuilder(cmd)
                .redirectOutput(out.toFile())
                .redirectError(err.toFile())
                .start()
            if (!p.waitFor(300, TimeUnit.SECONDS)) {
                p.destroyForcibly()
                println("[skip] the torch weight-value oracle timed out")
                return null
            }
            if (p.exitValue() != 0) {
                println("[skip] oracle exit ${p.exitValue()}: ${Files.readString(err).take(800)}")
                return null
            }
            return parseJson(Files.readString(out)) as JsonObject
        } finally {
            Files.deleteIfExists(out)
            Files.deleteIfExists(err)
        }
    }

    /**
     * Write a minimal but REAL safetensors checkpoint directory: an f32 file
     * whose header this repo's own reader parses, plus a config.json. The
     * values are a deterministic ramp per tensor so two roles can be told
     * apart.
     *
     * REJECTED: mocking [io.tlaloc.core.io.WeightSource]. The claims here are
     * about how a DIRECTORY resolves — which file is opened, which name is
     * looked up, what the header says — and a mock replaces exactly the thing
     * under test.
     */
    private fun writeCheckpoint(
        tied: Boolean,
        includeLmHead: Boolean,
        overrideDims: Map<String, IntArray> = emptyMap(),
        drop: Set<String> = emptySet(),
        extra: Map<String, IntArray> = emptyMap(),
    ): Path {
        val cfg = HfLlamaConfig(
            architecture = "LlamaForCausalLM", modelType = "llama",
            hiddenSize = TINY_HIDDEN, intermediateSize = TINY_INTER, numLayers = TINY_LAYERS,
            numHeads = TINY_HEADS, numKvHeads = TINY_KV_HEADS, headDim = TINY_HEAD_DIM,
            vocabSize = TINY_VOCAB, rmsNormEps = 1e-5, ropeTheta = 10000.0,
            maxPositionEmbeddings = 32, tieWordEmbeddings = tied, attentionBias = false,
            torchDtype = "float32", ropeScalingType = null,
        )
        val plan = LinkedHashMap<String, IntArray>()
        for (role in HfLlamaNames.roles(cfg)) {
            if (role == LlamaWeightRole.LmHead && !includeLmHead) continue
            val n = HfLlamaNames.hfName(role)
            if (n in drop) continue
            plan[n] = overrideDims[n] ?: HfLlamaNames.expectedDims(role, cfg)
        }
        plan.putAll(extra)

        val dir = createTempDirectory("hf-llama-$tied-$includeLmHead")
        dir.toFile().deleteOnExit()
        Files.write(
            dir.resolve("config.json"),
            configJson(cfg).toByteArray(StandardCharsets.UTF_8),
        )

        val header = StringBuilder("{")
        val body = java.io.ByteArrayOutputStream()
        var off = 0L
        var seed = 1
        for ((i, e) in plan.entries.withIndex()) {
            val (name, dims) = e
            val count = dims.fold(1) { a, d -> a * d }
            val begin = off
            for (k in 0 until count) {
                val bits = java.lang.Float.floatToRawIntBits((seed * 0.125f) + k * 0.001953125f)
                for (b in 0 until 4) body.write((bits ushr (8 * b)) and 0xFF)
            }
            off += count * 4L
            if (i > 0) header.append(',')
            header.append("\"").append(name).append("\":{\"dtype\":\"F32\",\"shape\":[")
                .append(dims.joinToString(","))
                .append("],\"data_offsets\":[").append(begin).append(',').append(off).append("]}")
            seed++
        }
        header.append("}")
        val headerBytes = header.toString().toByteArray(StandardCharsets.UTF_8)
        val file = java.io.ByteArrayOutputStream()
        val n = headerBytes.size.toLong()
        for (b in 0 until 8) file.write(((n ushr (8 * b)) and 0xFF).toInt())
        file.write(headerBytes)
        file.write(body.toByteArray())
        Files.write(dir.resolve("model.safetensors"), file.toByteArray())
        return dir
    }

    private fun configJson(c: HfLlamaConfig): String = buildString {
        append("{\"architectures\":[\"").append(c.architecture).append("\"],")
        append("\"model_type\":\"").append(c.modelType).append("\",")
        append("\"hidden_size\":").append(c.hiddenSize).append(',')
        append("\"intermediate_size\":").append(c.intermediateSize).append(',')
        append("\"num_hidden_layers\":").append(c.numLayers).append(',')
        append("\"num_attention_heads\":").append(c.numHeads).append(',')
        append("\"num_key_value_heads\":").append(c.numKvHeads).append(',')
        append("\"head_dim\":").append(c.headDim).append(',')
        append("\"vocab_size\":").append(c.vocabSize).append(',')
        append("\"rms_norm_eps\":1e-05,\"rope_theta\":10000.0,")
        append("\"max_position_embeddings\":").append(c.maxPositionEmbeddings).append(',')
        append("\"tie_word_embeddings\":").append(c.tieWordEmbeddings).append(',')
        append("\"attention_bias\":false,\"torch_dtype\":\"float32\",\"rope_scaling\":null}")
    }

    private fun assertContentEquals(a: FloatArray, b: FloatArray) {
        assertEquals(a.size, b.size)
        for (i in a.indices) assertEquals(a[i], b[i], "at $i")
    }

    private fun JsonObject.reqInt(key: String): Int =
        (this[key] as JsonNumber).asInt(key)

    private companion object {
        const val TINY_HIDDEN = 8
        const val TINY_INTER = 12
        const val TINY_LAYERS = 2
        const val TINY_HEADS = 4
        const val TINY_KV_HEADS = 2
        const val TINY_HEAD_DIM = 2
        const val TINY_VOCAB = 16
        const val TINY_KV_OUT = TINY_KV_HEADS * TINY_HEAD_DIM

        const val NORM = "model.norm.weight"
        const val K0 = "model.layers.0.self_attn.k_proj.weight"
        const val V0 = "model.layers.0.self_attn.v_proj.weight"
        const val GATE0 = "model.layers.0.mlp.gate_proj.weight"
        const val DOWN0 = "model.layers.0.mlp.down_proj.weight"
    }
}
