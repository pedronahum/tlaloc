package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodePadding
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.477 — Phase H7: the LIVE vLLM lane, with vLLM actually installed.
 *
 * This test settles `docs/INFERENCE_SERVING_AUDIT.md` §5's
 * WRITTEN-BUT-UNCERTIFIED item 1. Since §0.4.470 the repo has had exactly two
 * files that import vLLM — `vllm_tlaloc/platform.py` and
 * `vllm_tlaloc/worker.py` — and neither had ever run, because installing
 * vLLM into `~/.local/venvs/iree` would replace the torch every
 * cross-language oracle here is measured against. H7 installs it somewhere
 * ELSE: `~/.local/venvs/vllm`, a venv of its own, holding vLLM 0.29.0 and
 * torch 2.13.0+cu130 and NO jax at all.
 *
 * ## Why the second venv is possible at all
 *
 * Because of H6. The serving path reaches PJRT through `tlaloc_pjrt`'s
 * ctypes binding, so the vLLM venv needs no framework of ours — only a
 * plugin `.so`, named by `$TLALOC_PJRT_PLUGIN_PATH`, and that `.so` is
 * allowed to be the one sitting in the ORACLE venv's site-packages, because
 * `dlopen` of a file in another directory is not a dependency on that
 * directory's Python. Before H6 this test could not have been written: the
 * loader wanted jaxlib, and `pip install jax[cuda12]` beside vLLM's
 * CUDA-13 wheels is the collision the whole rail exists to prevent.
 *
 * ## The two lanes, and the claim that needs both
 *
 * 1. **The live lane** — `run_vllm_live_check.py` in the vLLM venv: platform
 *    discovery, `check_and_update_config` against REAL `vllm.config`
 *    objects, and `TlalocWorker` driven through the v1 worker API returning
 *    a real `vllm.v1.outputs.ModelRunnerOutput`.
 * 2. **The runner lane** — H3b's `run_vllm_tlaloc_check.py` in the ORACLE
 *    venv, the already-certified path that drives `TlalocModelRunner`
 *    directly.
 *
 * Both are given the SAME exported artifact directory and the SAME request.
 * The claim is that **the same artifact, called two ways, in two venvs with
 * different torches and only one of them holding jax, produces the same
 * numbers** — and the floor is not a tolerance: it is `==`. Both lanes run
 * the ctypes engine against the same plugin `.so` on the same device, so a
 * difference of any size would be a difference in the ADAPTER, which is the
 * only thing that differs. A tolerance here would hide exactly the defect
 * the lane exists to find.
 *
 * ## What it found
 *
 * On its first execution, step 0, `KeyError`. `TlalocWorker.execute_model`
 * built the batch before admitting the new sequences, so
 * `decode_requests_from_scheduler_output` asked the runner for the history
 * of a request that had not been added yet — the first step of every server
 * that would ever have started. The vLLM-free unit lane passed it because
 * its `last_token_of` stand-in was a dict literal with an answer for the new
 * id, and a dict is more forgiving than a runner. The fix and its regression
 * test are in `batching.py` and `vllm_tlaloc_test.py` (§0.4.477); the
 * finding is the argument for this test existing.
 *
 * ## What it does NOT cover, stated rather than implied
 *
 * No `LLM.generate()` and no `vllm serve`. Both need a HuggingFace config
 * and tokenizer for a real model, and the artifact this repo can export
 * today is `ReferenceDecodeGraph` — a tiny LCG model with no tokenizer. That
 * is H3c (weight name-mapping + a real Llama), already the audit's largest
 * open item, and it is a MODEL-COVERAGE gap rather than a plugin gap.
 * Wrapping the reference model in a fabricated `config.json` would have
 * certified the fabrication.
 */
class VllmLivePluginTest {

    private val m = ReferenceDecodeGraph.MODEL

    @Test
    fun theLivePluginUnderRealVllmMatchesTheRunnerLaneBitForBit() {
        val vllmPython = resolveVllmPython() ?: run {
            println("[skip] no ~/.local/venvs/vllm python for the live vLLM lane"); return
        }
        val oraclePython = resolveOraclePython() ?: run {
            println("[skip] no oracle venv python for the comparison lane"); return
        }
        val plugin = resolvePluginSo() ?: run {
            println("[skip] no PJRT plugin .so to point the vLLM venv at"); return
        }

        val dir = Files.createTempDirectory("tlaloc-vllm-live")
        try {
            ReferenceDecodeGraph.exportTo(dir)
            val reqFile = dir.resolve("request.json")
            Files.writeString(reqFile, buildRequest())

            val liveOut = dir.resolve("live.json")
            val live = runCheck(
                vllmPython, "run_vllm_live_check.py", dir, reqFile, liveOut, plugin,
            ) ?: return

            // --- 1. discovery: vLLM's own platform resolution ------------
            assertEquals(
                "vllm_tlaloc.platform.TlalocPlatform", live.str("platformClass"),
                "vLLM's out-of-tree platform discovery must land on ours",
            )
            assertEquals("tlaloc", live.str("deviceName"))
            assertEquals("tlaloc-pjrt-cuda:0", live.str("getDeviceName"))
            assertTrue(live.bool("supportsV1"))
            assertTrue(
                !live.bool("isAsyncOutputSupported"),
                "this runner samples on the host inside the step; async output staging " +
                    "assumes it can be ahead of the sampler",
            )
            println("[live] vLLM ${live.str("vllmVersion")} activated the tlaloc platform")

            // --- 2. the config hook, against REAL vllm.config objects ----
            val hook = live.obj("configHook")
            assertEquals("vllm_tlaloc.worker.TlalocWorker", hook.str("workerCls"))
            assertEquals(m.blockSize, hook.int("blockSize"))
            assertEquals(m.numBlocks, hook.int("numGpuBlocksOverride"))
            assertTrue(
                hook.str("cacheConfigClass").startsWith("vllm.config"),
                "the hook must have been validated against vLLM's own CacheConfig, not a " +
                    "stand-in: a refusal that has never met a pydantic validator is not a " +
                    "refusal (got ${hook.str("cacheConfigClass")})",
            )

            // Every refusal fires, BY NAME, on a config vLLM itself built.
            // An empty string is what the script records when nothing was
            // raised, which is why each of these is an exact content check
            // and not a null check.
            val refusals = live.obj("refusals")
            refusals.mustContain("blockSize", "disagrees with the serving artifact's compiled")
            refusals.mustContain("maxModelLen", "exceeds the artifact's top context bucket")
            refusals.mustContain("maxNumSeqs", "exceeds the artifact's top batch bucket")
            refusals.mustContain("worldSize", "multi-device serving is a named deferral")
            live.mustContain("kvConfigRefusal", "cannot be resized by a config")
            live.mustContain("chunkedPrefillRefusal", "named deferral")

            // --- 2b. §0.4.491 (H3c-4a): the attention backend CLASS -------
            // Until this slice `get_attn_backend_cls` RAISED, and vLLM's v1
            // engine core calls it unconditionally — which is where §0.4.480
            // died with a real 22-layer artifact in hand. It now answers with
            // a dotted path, and everything below is that answer examined
            // against vLLM's own code rather than against a docstring.
            assertEquals(
                "vllm_tlaloc.attention.TlalocAttentionBackend", live.str("attnBackendClass"),
                "the platform must hand vLLM the dotted path of the backend class",
            )
            assertEquals(
                live.str("attnBackendClass"), live.str("attnBackendResolves"),
                "vLLM resolves that string with resolve_obj_by_qualname and nothing " +
                    "checks it first; a typo surfaces as an import error inside engine startup",
            )
            assertTrue(
                live.bool("attnBackendIsAttentionBackend"),
                "it must be a real vllm.v1.attention.backend.AttentionBackend subclass",
            )
            // An EXPLICIT --attention-backend is still refused by name: a user
            // who asked for FlashAttention deserves an answer, not a silent
            // substitution.
            live.mustContain("attnBackendRefusal", "OpKind.PAGED_ATTENTION")
            live.mustContain("attnHeadSizeRefusal", "disagrees with the serving artifact")
            live.mustContain("attnMlaRefusal", "named deferral")
            assertEquals(
                "vllm_tlaloc.attention.TlalocAttentionBackend",
                live.str("attnAgreeingConfigPath"),
                "a selector config that AGREES with the artifact must be answered, not refused",
            )

            // The shape agreement, BOTH WAYS, against a real manifest. The
            // plugin derives its page from kvPoolAxisOrder/kvPoolDims; vLLM
            // derives its own from an AttentionSpec through
            // compute_layer_kv_cache_shape_bytes. Two independent
            // derivations of one pool, and they must land on one tuple.
            val attn = live.obj("attnBackend")
            assertEquals(
                listOf(2, m.numBlocks, m.blockSize, m.numKvHeads, m.headDim),
                attn.ints("kvCacheShape"),
                "the backend's KV cache shape is the manifest's kvPoolDims with K and V " +
                    "in front — separate tensors, which is what a Tlaloc pool literally is",
            )
            assertEquals(
                listOf("kv", "numBlocks", "blockSize", "numKvHeads", "headDim"),
                attn.strings("kvCacheAxisNames"),
            )
            assertEquals(
                live.ints("vllmPageShapeBytes"), attn.ints("vllmLogicalPageShapeBytes"),
                "vLLM's own compute_layer_kv_cache_shape_bytes and the plugin's reading " +
                    "of the same manifest must produce the same [B, H, N, C] page",
            )
            assertEquals(
                live.int("vllmPageSizeBytes"), attn.int("pageSizeBytes"),
                "KV_PAGE_BYTES_AGREE: vLLM interleaves K and V into the content axis and " +
                    "Tlaloc stores them as two tensors — the axis orders differ and the " +
                    "BYTES PER PAGE do not, and the byte count is what blocks are handed " +
                    "out against",
            )
            assertEquals(
                m.numBlocks * m.blockSize * m.numKvHeads * m.headDim * 4 * 2 * m.numLayers,
                attn.int("poolBytes"),
            )
            assertEquals("LBNHC", attn.str("kvCacheLayout"))
            assertEquals(listOf("LBNHC"), live.strings("attnKvCacheLayouts"))
            assertEquals(listOf(m.headDim), attn.ints("supportedHeadSizes"))
            assertEquals(listOf(m.blockSize), attn.ints("supportedKernelBlockSizes"))

            // The capability predicates, asked the way vLLM asks them. The
            // interesting one is the NEGATIVE: vLLM's inherited
            // supports_block_size accepts any MULTIPLE of a supported size,
            // because 0.29.0 can subdivide a manager block into kernel
            // blocks. A compiled pool cannot, and this lane is what found it.
            assertTrue(live.bool("attnSupportsCompiledBlockSize"))
            assertTrue(
                !live.bool("attnSupportsDoubleBlockSize"),
                "blockSize is baked into H1a's gather and H1b's scatter; inheriting " +
                    "vLLM's divisibility rule would let the scheduler page memory the " +
                    "compiled program has no slots for",
            )
            assertTrue(live.bool("attnSupportsCompiledHeadSize"))
            assertTrue(!live.bool("attnSupportsOtherHeadSize"))

            // The forward that is never reached, and the two constructors
            // above it. Each says WHY arriving there means vLLM took a path
            // this plugin does not implement.
            live.mustContain("attnImplRefusal", "TlalocAttentionNotReached")
            live.mustContain("attnImplRefusal", "there is no torch attention layer to build")
            live.mustContain("attnBuilderRefusal", "TlalocAttentionNotReached")
            live.mustContain("attnForwardRefusal", "Attention lives inside the compiled artifact")

            // --- 3. what the worker reports about the pool ---------------
            // The artifact's pool, exactly — not a memory profile. A profile
            // more generous than the truth is a silent out-of-bounds write.
            val elems = m.numBlocks * m.blockSize * m.numKvHeads * m.headDim
            assertEquals(
                elems.toLong() * 4L * 2L * m.numLayers.toLong(),
                live.long("availableMemoryBytes"),
                "determine_available_memory must report the compiled pool",
            )
            val layer0 = live.obj("kvCacheSpecLayer0")
            assertEquals(m.blockSize, layer0.int("block_size"))
            assertEquals(m.numKvHeads, layer0.int("num_kv_heads"))
            assertEquals(m.headDim, layer0.int("head_size"))
            assertEquals(m.numLayers, live.arr("kvCacheSpecLayers").size)

            // --- 4. the numbers, against the certified runner lane -------
            val runnerOut = dir.resolve("runner.json")
            val runner = runCheck(
                oraclePython, "run_vllm_tlaloc_check.py", dir, reqFile, runnerOut, null,
            ) ?: return

            val liveSteps = live.arr("steps")
            val runnerSteps = runner.arr("steps")
            assertEquals(STEPS, liveSteps.size)
            assertTrue(runnerSteps.size >= STEPS)

            for (i in 0 until STEPS) {
                val a = liveSteps.obj(i)
                val b = runnerSteps.obj(i)
                assertEquals(
                    "vllm.v1.outputs.ModelRunnerOutput", a.str("outputClass"),
                    "the worker must return vLLM's real output type",
                )
                assertEquals(b.ints("bucket"), a.ints("bucket"), "step $i chose a different bucket")
                assertEquals(b.ints("sampled"), a.ints("sampled"), "step $i sampled differently")
                val x = a.floats("logits")
                val y = b.floats("logits")
                assertEquals(y.size, x.size, "step $i logits width")
                for (k in x.indices) {
                    // `==`, deliberately. Same artifact, same plugin .so, same
                    // device, same ctypes engine — only the CALLER differs, so
                    // any difference at all is an adapter difference.
                    assertEquals(
                        y[k], x[k],
                        "step $i logit $k: the vLLM worker lane and the runner lane must " +
                            "agree bit-for-bit (vLLM venv vs oracle venv)",
                    )
                }
            }

            // One bucket across both steps ⇒ one compile. The executable
            // cache is keyed by the manifest's cacheKey; a second compile
            // here would mean the key moved between callers.
            assertEquals(1, live.int("compileCount"))

            // The request ids are vLLM's STRINGS, not the runner lane's
            // integers — the same sequences under a different naming, which
            // is precisely what the adapter is for.
            val ids = liveSteps.obj(0).strings("reqIds")
            assertEquals(listOf("0", "1", "2"), ids)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // --- plumbing -----------------------------------------------------------

    private val prompts = listOf(3, 5, 7)

    private fun buildRequest(): String = """
        {"prompts":${prompts.joinToString(",", "[", "]")},
         "steps":$STEPS,
         "padding":{"PADDING_TOKEN_ID":${DecodePadding.PADDING_TOKEN_ID},
                    "PADDING_POSITION":${DecodePadding.PADDING_POSITION},
                    "PADDING_BLOCK":${DecodePadding.PADDING_BLOCK},
                    "PADDING_SEQ_LEN":${DecodePadding.PADDING_SEQ_LEN},
                    "PADDING_SLOT":${DecodePadding.PADDING_SLOT}}}
    """.trimIndent()

    /** Runs one harness script; null means the lane self-skipped (exit 2). */
    private fun runCheck(
        python: String, script: String, artifact: Path, request: Path, out: Path,
        plugin: String?,
    ): JsonObject? {
        val pb = ProcessBuilder(
            python, harnessPython().resolve(script).toString(),
            "--artifact", artifact.toString(),
            "--request", request.toString(),
            "--output", out.toString(),
            "--platform", "cuda",
        )
        pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        // The vLLM venv has no jax, so the loader finds its plugin the way a
        // DEPLOYMENT does: by file path, from the environment.
        plugin?.let { pb.environment()["TLALOC_PJRT_PLUGIN_PATH"] = it }
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val stdout = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(1800, TimeUnit.SECONDS)) {
            proc.destroyForcibly(); fail("$script timed out")
        }
        val exit = proc.exitValue()
        val text = if (Files.exists(out)) Files.readString(out) else "{}"
        if (exit == 2) {
            println("[skip] $script reports an unrunnable environment: $text"); return null
        }
        if (exit != 0) fail("$script failed (exit $exit)\n$text\n$stdout")
        val o = parseJson(text) as JsonObject
        assertTrue(o.bool("ok"), "$script reported failure: $text")
        return o
    }

    private fun harnessPython(): Path =
        Path.of("..", "harness", "python").toAbsolutePath().normalize()

    private fun resolveVllmPython(): String? = executableOr(
        System.getenv("TLALOC_VLLM_PYTHON"),
        Path.of(System.getProperty("user.home") ?: return null, ".local", "venvs", "vllm", "bin", "python"),
    )

    private fun resolveOraclePython(): String? = executableOr(
        System.getenv("TLALOC_TORCH_PYTHON"),
        Path.of(System.getProperty("user.home") ?: return null, ".local", "venvs", "iree", "bin", "python"),
    )

    private fun executableOr(env: String?, fallback: Path): String? {
        env?.let { if (Files.isExecutable(Path.of(it))) return it }
        return if (Files.isExecutable(fallback)) fallback.toString() else null
    }

    /**
     * The plugin `.so` the vLLM venv is pointed at. Deliberately the same
     * three-place FILE lookup `tlaloc_serve` documents, honoured here in the
     * one order a JVM test can: an explicit env var, then the oracle venv's
     * jax plugin as "a place a file sits".
     */
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
        const val STEPS = 2
    }
}

// --- small readers over the result JSON, on :core's strict parser -----------
//
// `:core`'s parser is the one H2 wrote for UNTRUSTED checkpoint headers, and
// it is reused here rather than the hand-rolled scanner in
// VllmPluginContractTest: this result is a much wider object (nested refusal
// maps, per-step arrays of objects) and a scanner that is "tiny because the
// file is small" stops being an argument at that width.

private fun JsonObject.bool(key: String): Boolean =
    (fields[key] as? JsonBool)?.value
        ?: throw IllegalArgumentException("'$key' is not a boolean")

private fun JsonObject.num(key: String): Double =
    (fields[key] as? JsonNumber)?.value
        ?: throw IllegalArgumentException("'$key' is not a number")

private fun JsonObject.int(key: String): Int = num(key).toInt()
private fun JsonObject.long(key: String): Long = num(key).toLong()

private fun JsonObject.ints(key: String): List<Int> =
    arr(key).elements.map { (it as JsonNumber).value.toInt() }

private fun JsonObject.floats(key: String): List<Float> =
    arr(key).elements.map { (it as JsonNumber).raw.toFloat() }

private fun JsonObject.strings(key: String): List<String> =
    arr(key).elements.map { (it as JsonString).value }

private fun JsonArray.obj(i: Int): JsonObject = elements[i] as JsonObject

/**
 * Asserts a recorded refusal message both EXISTS and says the right thing.
 * `run_vllm_live_check.py` records the empty string when a call that was
 * supposed to raise did not, so "contains" over an empty string is the
 * failure — which is why this is one helper and not two assertions.
 */
private fun JsonObject.mustContain(key: String, fragment: String) {
    val got = str(key)
    if (!got.contains(fragment)) {
        throw AssertionError(
            if (got.isEmpty()) "'$key': nothing was raised — the refusal did not fire"
            else "'$key' should mention \"$fragment\" but said: $got",
        )
    }
}
