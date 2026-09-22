package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfLlamaCheckpoint
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.480 — Phase H3c-3: **a real Llama serves, and says what HuggingFace says.**
 *
 * §0.4.479 built the decode graph from a real TinyLlama-1.1B checkpoint and
 * certified its LOGITS against transformers. It could not certify the device
 * lane, for one reason it named precisely: the ARTIFACT had nowhere to put a
 * weight table, so the only way to run that graph was the JVM interpreter.
 * This slice gives the manifest a weight table, the writer a staged-weight
 * format, and `tlaloc_serve.py` a way to bind it — and then runs the whole
 * thing on PJRT-CUDA with no framework in the process.
 *
 * ## The claim, and why the floor is not a tolerance
 *
 * The lane greedy-decodes a prompt two ways:
 *
 * 1. **Tlaloc** — `run_llama_generate.py` over an exported artifact, ctypes
 *    engine, PJRT-CUDA, stdlib + `tlaloc_pjrt` only.
 * 2. **HuggingFace** — `hf_llama_greedy_oracle.py`, `AutoModelForCausalLM`
 *    fp32 on CPU, in the vLLM venv (torch 2.13.0+cu130, transformers 5.17.0).
 *
 * and asserts the GENERATED TOKEN IDS ARE EQUAL. Token ids, not logits, and
 * that is the deliberate choice: XLA-GPU's default f32 `dot_general` policy
 * is TF32 (§0.4.477's serving floor is 1e-3 for exactly this reason) while
 * the oracle is fp32 on CPU, so a logit tolerance here would be a number
 * chosen to pass. An argmax is not. Greedy decoding agrees exactly until the
 * two arithmetics disagree about a top-1, and the honest form of the claim is
 * **the length of the prefix that agrees** — so the test asserts the FULL
 * requested budget agrees and reports the first divergence if one appears,
 * rather than shrinking the budget until it is comfortable.
 *
 * The prompt is never written down as ids. `hf_llama_greedy_oracle.py`
 * tokenizes it with the checkpoint's own tokenizer and the ids it returns are
 * what Tlaloc is asked to continue — §0.4.477 refused to fabricate a
 * `config.json`, and a hand-typed token id is the same refusal, smaller.
 *
 * ## Lane split
 *
 * The real lane self-skips without the checkpoint, the vLLM venv or a plugin
 * `.so`. The hermetic lane below it is the exporter's REFUSALS, which need
 * none of those and gate a fresh machine.
 */
class HfLlamaServingArtifactTest {

    /** Long enough to be a sentence, short enough that a step is ~1.4 s. */
    private val maxNew = 6
    private val prompt = "The capital of France is"

    @Test
    fun aRealLlamaServesFromAnArtifactAndAgreesWithHuggingFaceTokenForToken() {
        val ckptDir = resolveCheckpoint() ?: run {
            println(
                "[skip] no TinyLlama checkpoint; fetch it with\n" +
                    "  ~/.local/venvs/vllm/bin/python -c \"from huggingface_hub import " +
                    "snapshot_download; snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', " +
                    "local_dir='$DEFAULT_CKPT')\"",
            )
            return
        }
        val vllmPython = resolveVllmPython() ?: run {
            println("[skip] no ~/.local/venvs/vllm python for the transformers oracle"); return
        }
        val plugin = resolvePluginSo() ?: run {
            println("[skip] no PJRT plugin .so"); return
        }

        val dir = Files.createTempDirectory("tlaloc-llama-artifact")
        try {
            // --- the oracle FIRST: it owns the tokenizer, so it owns the ids.
            val oracleOut = dir.resolve("oracle.json")
            val oracle = runPython(
                vllmPython, "hf_llama_greedy_oracle.py", listOf(
                    "--checkpoint", ckptDir.toString(), "--prompt", prompt,
                    "--max-new", maxNew.toString(), "--output", oracleOut.toString(),
                ), plugin, oracleOut,
            ) ?: return
            val promptIds = oracle.ints("promptTokens")
            val oracleIds = oracle.ints("generatedTokens")
            assertEquals(maxNew, oracleIds.size, "the oracle generated a different budget")

            // --- export the artifact the deployment would ship -----------
            val manifest = HfLlamaCheckpoint.open(ckptDir).use { ckpt ->
                HfLlamaServingExport.export(
                    ckpt = ckpt, dir = dir.resolve("artifact"),
                    policy = DecodeBucketPolicy(
                        maxBatch = 1, maxContext = CONTEXT,
                        blockSize = BLOCK_SIZE, minContext = CONTEXT,
                    ),
                    modelName = ckptDir.fileName.toString(),
                )
            }
            assertEquals(
                oracle.int("numLayers"), manifest.model.numLayers,
                "the two sides must be the same model, not two reductions of one",
            )
            // 201 tensors, 201 staged operands — §0.4.478's inventory, now
            // bound one-to-one to files on disk.
            assertEquals(
                manifest.weights.table.size,
                manifest.entries.first().inputs.count { it.role.name == "WEIGHT" },
                "every WEIGHT operand of the compiled entry must have a file behind it",
            )
            assertTrue(!manifest.weights.embedded)
            assertEquals(ServingWeightsPointer.STAGED_FORMAT, manifest.weights.format)
            for (w in manifest.weights.table) {
                val f = dir.resolve("artifact").resolve(w.path)
                assertTrue(Files.isRegularFile(f), "staged weight ${w.path} is missing")
                assertEquals(
                    w.byteLength, Files.size(f),
                    "staged weight ${w.name} is a different length on disk than the manifest says",
                )
                // The length is the dims, not a number the writer remembered:
                // f32 is 4 bytes and the artifact says so in two places.
                assertEquals(w.count * 4, w.byteLength, "weight ${w.name} byteLength vs dims")
            }
            // The manifest survives the wire it was designed for.
            assertEquals(manifest, ServingManifest.fromJson(manifest.toJson()))

            // --- run it ---------------------------------------------------
            val reqFile = dir.resolve("request.json")
            Files.writeString(
                reqFile,
                "{\"promptTokens\":${promptIds.joinToString(",", "[", "]")}," +
                    "\"maxNewTokens\":$maxNew}",
            )
            val runOut = dir.resolve("generated.json")
            val run = runPython(
                "python3", "run_llama_generate.py", listOf(
                    "--artifact", dir.resolve("artifact").toString(),
                    "--request", reqFile.toString(), "--output", runOut.toString(),
                    "--platform", "cuda", "--verify-weights",
                ), plugin, runOut,
            ) ?: return

            assertEquals("CtypesEngine", run.str("engine"), "the deployment engine, not the oracle")
            assertEquals(manifest.model.numLayers, run.int("numLayers"))
            assertEquals(manifest.weights.table.size, run.int("weightSlots"))
            assertEquals(promptIds, run.ints("promptTokens"))

            val got = run.ints("generatedTokens")
            val agree = got.zip(oracleIds).takeWhile { (a, b) -> a == b }.size
            assertEquals(
                oracleIds, got,
                "a real TinyLlama-1.1B greedy-decoding '$prompt' through the Tlaloc serving " +
                    "artifact on PJRT-CUDA agreed with HuggingFace transformers (fp32/CPU) for " +
                    "$agree of $maxNew tokens and then diverged. Tlaloc $got vs HF $oracleIds. " +
                    "Greedy decoding agrees EXACTLY until TF32 and fp32 disagree about a top-1, " +
                    "so a divergence here is a number to record, not a tolerance to widen",
            )
            // --- §0.4.492 (H3c-4b): the SAME artifact, driven by vLLM ----
            // Two callers, one artifact. `run_llama_generate.py` walks the
            // decode ladder itself; vLLM brings its scheduler, its block
            // manager, its tokenizer and its `LLM` entry point and reaches
            // the same programs through `TlalocWorker`. Any difference is
            // the ADAPTER, which is why the floor is `==` and not a
            // tolerance — both lanes run the same PJRT plugin on the same
            // device against the same 4.2 GiB of staged weights.
            val vllmOut = dir.resolve("vllm-generate.json")
            val vllmRun = runPython(
                vllmPython, "run_vllm_generate_check.py", listOf(
                    "--artifact", dir.resolve("artifact").toString(),
                    "--checkpoint", ckptDir.toString(),
                    "--prompt", prompt, "--max-new", maxNew.toString(),
                    "--max-context", CONTEXT.toString(),
                    "--block-size", BLOCK_SIZE.toString(),
                    "--output", vllmOut.toString(),
                ), plugin, vllmOut,
            )
            if (vllmRun == null) {
                println("[skip] the vLLM generate lane reported an unrunnable environment")
            } else {
                // vLLM tokenized the prompt with the checkpoint's own
                // tokenizer, independently of the oracle. That the two agree
                // is the statement that both lanes were asked the same
                // question; nothing here types a token id.
                assertEquals(
                    promptIds, vllmRun.ints("promptTokens"),
                    "vLLM and the transformers oracle tokenized '$prompt' differently, so " +
                        "the two lanes were never asked the same question",
                )
                val viaVllm = vllmRun.ints("generatedTokens")
                assertEquals(
                    got, viaVllm,
                    "the same artifact, called directly and called through vLLM 0.29.0's " +
                        "LLM.generate(), produced different tokens: direct $got vs vLLM " +
                        "$viaVllm. Both ran the same compiled programs on the same device, " +
                        "so the difference is in the plugin's adaptation and nowhere else",
                )
                assertEquals(
                    oracleIds, viaVllm,
                    "vLLM's generation disagrees with HuggingFace transformers",
                )
                assertEquals(maxNew, viaVllm.size)
                assertEquals("length", vllmRun.str("finishReason"))
                println(
                    "[H3c-4b] vLLM ${vllmRun.str("vllmVersion")} LLM.generate() on the " +
                        "artifact: ${vllmRun.str("text").replace("\n", "\\n")} — " +
                        "$viaVllm, equal to the direct lane and to HF; " +
                        "construct ${vllmRun.num("constructSeconds")}s, " +
                        "generate ${vllmRun.num("generateSeconds")}s",
                )
            }

            println(
                "[H3c-3] ${manifest.modelName}: ${manifest.model.numLayers} layers, " +
                    "${manifest.weights.table.size} staged weights " +
                    "(${manifest.weights.table.sumOf { it.byteLength } / (1024 * 1024)} MiB), " +
                    "$maxNew/$maxNew tokens agree with HF; first step " +
                    "${run.num("firstStepSeconds")}s, median step ${run.num("medianStepMs")}ms",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // --- hermetic: the exporter refuses a half-artifact --------------------

    @Test
    fun theExporterRefusesWeightSlotsWithNoBytesAndBytesWithNoSlots() {
        val dir = Files.createTempDirectory("tlaloc-staged-refusal")
        try {
            // The reference model declares no weight slots (its weights are
            // body constants), so supplying a stager is writing bytes nothing
            // will bind.
            val e = runCatching {
                ServingArtifactWriter.export(
                    dir = dir,
                    modelName = "x", modelHash = "x",
                    model = ReferenceDecodeGraph.MODEL,
                    ladder = ServingArtifactWriter.ladderOf(ReferenceDecodeGraph.POLICY),
                    specs = ServingArtifactWriter.decodeSpecs(
                        ReferenceDecodeGraph.MODEL, ReferenceDecodeGraph.POLICY,
                    ),
                    stageWeight = { FloatArray(1) },
                    build = ReferenceDecodeGraph::build,
                )
            }.exceptionOrNull() ?: fail("a stager for a model with no weight slots was accepted")
            assertTrue(
                e.message!!.contains("bound by nothing"),
                "the refusal must say what is wrong: ${e.message}",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun anEmbeddedPointerCannotAlsoCarryAStagingTable() {
        val e = runCatching {
            ServingWeightsPointer(
                format = "embedded", path = null, embedded = true,
                table = listOf(
                    ServingWeightFile("w", "weights/0000_w.bin", "f32", listOf(2), 8L, "ab"),
                ),
            )
        }.exceptionOrNull() ?: fail("embedded weights with a staging table were accepted")
        assertTrue(e.message!!.contains("exactly one of those two descriptions is wrong"))
    }

    // --- process plumbing --------------------------------------------------

    private fun runPython(
        python: String, script: String, args: List<String>, plugin: String, out: Path,
    ): JsonObject? {
        val pb = ProcessBuilder(
            listOf(python, harnessPython().resolve(script).toString()) + args,
        )
        pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        pb.environment()["TLALOC_PJRT_PLUGIN_PATH"] = plugin
        pb.environment()["PYTHONPATH"] = harnessPython().toString()
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val stdout = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(1800, TimeUnit.SECONDS)) {
            proc.destroyForcibly(); fail("$script timed out")
        }
        val exit = proc.exitValue()
        if (exit == 2) {
            println("[skip] $script reports an unrunnable environment:\n$stdout"); return null
        }
        if (exit != 0) fail("$script failed (exit $exit)\n$stdout")
        return parseJson(Files.readString(out)) as JsonObject
    }

    private fun harnessPython(): Path =
        Path.of(System.getProperty("user.dir")).let { cwd ->
            generateSequence(cwd) { it.parent }
                .map { it.resolve("harness").resolve("python") }
                .first { Files.isDirectory(it) }
        }

    private fun resolveCheckpoint(): Path? {
        System.getenv("TLALOC_HF_LLAMA_CHECKPOINT")?.let {
            val p = Path.of(it); if (Files.isDirectory(p)) return p
        }
        val p = Path.of(System.getProperty("user.home") ?: return null)
            .resolve(DEFAULT_CKPT.removePrefix("~/"))
        return if (Files.isDirectory(p)) p else null
    }

    private fun resolveVllmPython(): String? {
        System.getenv("TLALOC_VLLM_PYTHON")?.let {
            if (Files.isExecutable(Path.of(it))) return it
        }
        val p = Path.of(System.getProperty("user.home") ?: return null,
            ".local", "venvs", "vllm", "bin", "python")
        return if (Files.isExecutable(p)) p.toString() else null
    }

    private fun resolvePluginSo(): String? {
        System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.let {
            if (Files.isReadable(Path.of(it))) return it
        }
        val home = System.getProperty("user.home") ?: return null
        val plugins = Path.of(home, ".local", "venvs", "iree", "lib", "python3.12",
            "site-packages", "jax_plugins")
        if (!Files.isDirectory(plugins)) return null
        Files.walk(plugins, 2).use { s ->
            return s.filter { it.fileName.toString() == "xla_cuda_plugin.so" }
                .findFirst().map { it.toString() }.orElse(null)
        }
    }

    private companion object {
        const val DEFAULT_CKPT: String =
            "~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0"

        /**
         * A DEMO ladder: one batch size, four pages of 16. Every extra ladder
         * point is another full XLA compile of a 22-layer model, and the claim
         * this lane makes is that it serves — H1c already certified that
         * bucketing does not change the answer.
         */
        const val BLOCK_SIZE: Int = 16
        const val CONTEXT: Int = 64
    }
}

private fun JsonObject.num(key: String): Double =
    (fields[key] as? JsonNumber)?.value
        ?: throw IllegalArgumentException("'$key' is not a number")

private fun JsonObject.int(key: String): Int =
    (fields[key] as? JsonNumber)?.asInt(key)
        ?: throw IllegalArgumentException("'$key' is not a number")

private fun JsonObject.ints(key: String): List<Int> =
    (fields[key] as? JsonArray)?.elements?.map {
        (it as? JsonNumber)?.asInt(key) ?: throw IllegalArgumentException("'$key' holds a non-number")
    } ?: throw IllegalArgumentException("'$key' is not an array")
