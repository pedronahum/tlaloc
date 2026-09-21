package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodePadding
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.470 — Phase H3b: the `vllm-tlaloc` plugin, certified as far as it can
 * honestly be certified on a machine with no vLLM.
 *
 * **vLLM is not installed and this slice did not install it.** Its dependency
 * closure resolves on this box (vllm 0.29.0, aarch64, py3.12) and comes to 186
 * packages including torch 2.13.0 and 33 CUDA-13 wheels, into the one venv
 * that holds jax-cuda12 0.10.0, torch 2.11.0+cpu and every certification
 * oracle in this repo. The audit records that refusal and the exact command
 * that would run the live-serving lane. What that leaves is a clean line:
 *
 * - `vllm_tlaloc/platform.py` and `worker.py` import vLLM and are UNCERTIFIED;
 * - everything they delegate to — the page pool, the slot arithmetic, the
 *   block tables, bucket selection, the artifact binding, the pool swap
 *   across steps, host-side sampling — is certified HERE, end to end, by
 *   driving `TlalocModelRunner` (the class the worker delegates to) over a
 *   real exported artifact on real PJRT.
 *
 * Three lanes:
 *
 * 1. **The registration contract**, which needs nothing installed at all and
 *    so always runs: the entry point named in `pyproject.toml`, the callable
 *    it names, and the platform class path all agree.
 * 2. **The vLLM-free unit lane** — `python -m unittest vllm_tlaloc_test`,
 *    29 stdlib-only cases over the marshalling arithmetic. Driven from here
 *    so the coverage is part of `./gradlew test` rather than a command
 *    somebody remembers to run.
 * 3. **The plugin-driven decode lane** — export the reference artifact, run
 *    TWO decode steps through the plugin's own bookkeeping, and check both
 *    the NUMBERS (against the host interpreter) and the DECISIONS (block
 *    tables, slots, buckets, page reuse) against an expectation this file
 *    computes independently. Two steps and not one: a single step cannot
 *    tell a KV cache that is threaded across steps from one that is
 *    recomputed, and the pool swap is the runner's most easily-wrong line.
 *
 * Tolerance, per H3a's measurement: CPU `1e-5` (the semantics claim; XLA-CPU
 * computes an f32 dot in f32) and CUDA `1e-3` (the deployment claim; XLA-GPU
 * lowers a default-precision f32 dot through TF32). The decisions half is
 * pinned EXACTLY on both lanes — a page id is not a float.
 */
class VllmPluginContractTest {

    private val m = ReferenceDecodeGraph.MODEL
    private val prompts = intArrayOf(3, 1, 4)
    private val steps = 2

    /** Two decode steps of three sequences at blockSize 2 never leave the
     *  context-2 bucket, which is what makes the executable cache's claim
     *  (one compile for two steps) checkable. */
    private val bucket = DecodeBucket(batch = 4, maxContext = 2)
    private val spec = DecodeGraphSpec(m, bucket, DecodeGraphKind.DECODE)
    private val poolLen = m.numBlocks * m.blockSize * m.numKvHeads * m.headDim

    /**
     * The plugin's allocator reserves page 0 as the padding scratch page and
     * hands out the lowest free page first, so three sequences admitted in
     * order own pages 1, 2, 3. Computed here from the RULE, not copied from
     * a run — that is what makes it an independent expectation.
     */
    private val expectedPages = intArrayOf(1, 2, 3)

    // --- lane 1: registration, which needs nothing installed ------------

    @Test
    fun theEntryPointAndThePlatformClassPathAgree() {
        val dir = harnessPython()
        val pyproject = Files.readString(dir.resolve("pyproject.toml"))
        assertTrue(
            pyproject.contains("[project.entry-points.\"vllm.platform_plugins\"]"),
            "the plugin registers through vLLM's documented entry-point group",
        )
        assertTrue(
            pyproject.contains("tlaloc = \"vllm_tlaloc:register\""),
            "the entry point must name vllm_tlaloc.register",
        )
        val init = Files.readString(dir.resolve("vllm_tlaloc").resolve("__init__.py"))
        val path = Regex("PLATFORM_CLASS_PATH = \"([^\"]+)\"").find(init)?.groupValues?.get(1)
            ?: fail("vllm_tlaloc/__init__.py declares no PLATFORM_CLASS_PATH")
        assertEquals("vllm_tlaloc.platform.TlalocPlatform", path)
        val cls = path.substringAfterLast('.')
        val module = path.substringBeforeLast('.').substringAfterLast('.')
        val src = Files.readString(dir.resolve("vllm_tlaloc").resolve("$module.py"))
        assertTrue(
            src.contains("class $cls("),
            "the entry point names $path and $module.py defines no class $cls; vLLM " +
                "reports that as 'no platform found', which says nothing",
        )
        // The Kotlin side of the seam must be the thing the plugin looks for.
        assertTrue(
            Files.readString(dir.resolve("vllm_tlaloc").resolve("runner.py"))
                .contains(ServingManifest.FILE_NAME.removeSuffix(".json")) ||
                Files.readString(dir.resolve("tlaloc_serve.py")).contains(ServingManifest.FILE_NAME),
            "the plugin's loader must look for ${ServingManifest.FILE_NAME}",
        )
    }

    // --- lane 2: the vLLM-free unit lane --------------------------------

    @Test
    fun thePluginsMarshallingArithmeticPassesItsOwnUnitLane() {
        val python = resolvePython() ?: run {
            println("[skip] no venv python for the vllm_tlaloc unit lane"); return
        }
        val pb = ProcessBuilder(python, "-m", "unittest", "vllm_tlaloc_test", "-v")
        pb.directory(harnessPython().toFile())
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val out = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(300, TimeUnit.SECONDS)) {
            proc.destroyForcibly(); fail("vllm_tlaloc_test timed out")
        }
        if (proc.exitValue() != 0) fail("vllm_tlaloc_test failed:\n$out")
        val ran = Regex("Ran (\\d+) tests").find(out)?.groupValues?.get(1)?.toInt() ?: 0
        assertTrue(ran >= 25, "the unit lane ran only $ran cases; it should be ~29\n$out")
        assertTrue(out.contains("OK"), "unittest did not report OK:\n$out")
    }

    // --- lane 3: the plugin drives a real artifact on real PJRT ---------

    @Test
    fun thePluginDecodesTwoStepsAgainstTheHostInterpreterOnCpu() =
        runPluginLane(platform = "cpu", tolerance = 1e-5f)

    @Test
    fun thePluginDecodesTwoStepsAgainstTheHostInterpreterOnCuda() =
        runPluginLane(platform = "cuda", tolerance = 1e-3f)

    private fun runPluginLane(platform: String, tolerance: Float) {
        val python = resolvePython() ?: run {
            println("[skip] no venv python for the $platform plugin lane"); return
        }
        if (!pythonHasJax(python)) {
            println("[skip] venv python has no importable jax"); return
        }
        val dir = Files.createTempDirectory("tlaloc-vllm-plugin")
        try {
            ReferenceDecodeGraph.exportTo(dir)

            val reqFile = dir.resolve("request.json")
            Files.writeString(reqFile, buildRequest())
            val outFile = dir.resolve("result.json")

            val pb = ProcessBuilder(
                python, scriptPath().toString(),
                "--artifact", dir.toString(),
                "--request", reqFile.toString(),
                "--output", outFile.toString(),
                "--platform", platform,
            )
            pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            pb.redirectErrorStream(true)
            val proc = pb.start()
            val stdout = proc.inputStream.bufferedReader().readText()
            if (!proc.waitFor(900, TimeUnit.SECONDS)) {
                proc.destroyForcibly(); fail("run_vllm_tlaloc_check[$platform] timed out")
            }
            val exit = proc.exitValue()
            val result = if (Files.exists(outFile)) Files.readString(outFile) else "{}"
            if (exit == 2) {
                println("[skip] plugin check reports an unrunnable $platform env: $result"); return
            }
            if (exit != 0) {
                fail("run_vllm_tlaloc_check[$platform] failed (exit $exit)\n$result\n$stdout")
            }

            val got = Json(result)
            assertTrue(got.bool("ok"), "the check reported failure: $result")

            // What the plugin tells vLLM about the deployment, answered from
            // the artifact rather than from a memory profile.
            assertEquals(m.blockSize, got.int("blockSize"))
            assertEquals(m.numBlocks, got.int("numGpuBlocks"))
            assertEquals(ReferenceDecodeGraph.POLICY.maxBatch, got.int("maxBatch"))
            assertEquals(ReferenceDecodeGraph.POLICY.maxContext, got.int("maxContext"))
            assertEquals(
                m.numBlocks - 1, got.int("poolCapacity"),
                "page ${DecodePadding.PADDING_BLOCK} is the padding scratch page and is " +
                    "never allocated to a live sequence",
            )

            // --- the decisions, pinned exactly on both lanes --------------
            val mbs = spec.maxBlocksPerSeq
            var keyPool = FloatArray(poolLen)
            var valuePool = FloatArray(poolLen)
            var feed = prompts.copyOf()

            for (step in 0 until steps) {
                val s = got.obj("steps", step)
                assertContentEquals(
                    listOf(bucket.batch, bucket.maxContext), s.ints("bucket"),
                    "step $step chose a bucket the expectation did not",
                )
                assertContentEquals(
                    List(3) { step }, s.ints("positions"),
                    "step $step: the position written is the one the token occupies, " +
                        "and it advances by exactly one per step",
                )
                assertContentEquals(List(3) { step + 1 }, s.ints("seqLens"))
                assertContentEquals(
                    expectedPages.map { it * m.blockSize + step }, s.ints("slotMapping"),
                    "step $step: slot = page * blockSize + (position % blockSize)",
                )
                assertEquals(
                    expectedPages.joinToString(",") { p ->
                        (listOf(p) + List(mbs - 1) { DecodePadding.PADDING_BLOCK })
                            .joinToString(",")
                    },
                    s.ints("blockTables").joinToString(","),
                    "step $step: each sequence names its own page, padded with the scratch page",
                )
                assertContentEquals(listOf(3, 1, m.vocabSize), s.ints("logitsShape"))

                // --- the numbers, against the host interpreter -------------
                val ref = interpreterStep(feed, step, keyPool, valuePool)
                val v = m.vocabSize
                val gotLogits = s.floats("logits")
                assertEquals(3 * v, gotLogits.size)
                for (row in 0 until 3) {
                    for (j in 0 until v) {
                        assertClose(
                            ref.logits[row * v + j], gotLogits[row * v + j], tolerance,
                            "[$platform] step $step logit[$row][$j]",
                        )
                    }
                }
                assertTrue(
                    gotLogits.distinct().size > 3,
                    "step $step produced a degenerate logit vector; the oracle would " +
                        "pass on anything",
                )

                // Host-side greedy sampling must pick what the interpreter's
                // own argmax picks — the tokens fed back in next step depend
                // on it, so a disagreement here diverges the whole lane.
                val expectSampled = (0 until 3).map { row ->
                    var best = 0
                    for (j in 1 until v) {
                        if (ref.logits[row * v + j] > ref.logits[row * v + best]) best = j
                    }
                    best
                }
                assertContentEquals(
                    expectSampled, s.ints("sampled"),
                    "[$platform] step $step: host-side greedy sampling disagrees with " +
                        "the interpreter's argmax",
                )

                keyPool = ref.keyPool
                valuePool = ref.valuePool
                feed = IntArray(3) { expectSampled[it] }
            }

            // Two steps at one bucket compile ONE executable: the manifest's
            // cacheKey is what the loader keys on, and a second compile here
            // would mean the plugin re-selects a bucket it already has.
            assertEquals(
                1, got.int("compileCountAfterSteps"),
                "two steps in the (${bucket.batch},${bucket.maxContext}) bucket compiled " +
                    "more than one executable",
            )
            // ...and the batch-of-one lifecycle step below IS a different
            // bucket, so it compiles a second program. Both halves matter:
            // the first says the cache works, the second says bucket
            // selection is really selecting and not returning one answer.
            assertEquals(
                2, got.int("compileCount"),
                "a batch of one is the (1,2) bucket and must compile its own entry",
            )

            // The cache is really threaded: step 1 and step 2 see different
            // KV, so identical logits would mean the pool swap is a no-op.
            assertTrue(
                got.obj("steps", 0).floats("logits") != got.obj("steps", 1).floats("logits"),
                "[$platform] the two steps produced identical logits; the KV pool the " +
                    "graph returned was not fed back in",
            )

            // --- the page lifecycle ---------------------------------------
            assertEquals(
                m.numBlocks - 1 - 3, got.int("freePagesBeforeRelease"),
                "three sequences hold one page each",
            )
            assertEquals(
                m.numBlocks - 1 - 2, got.int("freePagesAfterRelease"),
                "freeing a sequence returns its pages; a serving loop that leaks one " +
                    "page per request dies of it",
            )
            assertContentEquals(
                listOf(expectedPages.last()), got.ints("reusedBlocks"),
                "the newcomer must get the freed page back — the free list is kept " +
                    "sorted so that is reproducible and not incidental",
            )
            assertContentEquals(
                listOf(prompts[0]) + (0 until steps).map { got.obj("steps", it).ints("sampled")[0] },
                got.ints("historyOfSeq0"),
                "the runner is the thing that remembers what it decoded; vLLM's " +
                    "scheduler output does not carry it",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // --- helpers ---------------------------------------------------------

    private class Ref(val logits: FloatArray, val keyPool: FloatArray, val valuePool: FloatArray)

    /**
     * The same step in the host interpreter: the padded operands the plugin
     * SHOULD have built, computed here from the rule rather than read back
     * from the plugin's answer.
     */
    private fun interpreterStep(
        tokens: IntArray,
        step: Int,
        keyPool: FloatArray,
        valuePool: FloatArray,
    ): Ref {
        val mbs = spec.maxBlocksPerSeq
        val tables = IntArray(bucket.batch * mbs) { DecodePadding.PADDING_BLOCK }
        for (s in expectedPages.indices) tables[s * mbs] = expectedPages[s]
        val slots = IntArray(3) { expectedPages[it] * m.blockSize + step }
        fun f(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val out = DxirInterpreter.evalFunction(
            ReferenceDecodeGraph.build(spec),
            listOf(
                f(DecodePadding.padTo(tokens, bucket.batch, DecodePadding.PADDING_TOKEN_ID, "tokenIds")),
                f(DecodePadding.padTo(IntArray(3) { step }, bucket.batch, DecodePadding.PADDING_POSITION, "positions")),
                f(tables),
                f(DecodePadding.padTo(IntArray(3) { step + 1 }, bucket.batch, DecodePadding.PADDING_SEQ_LEN, "seqLens")),
                f(DecodePadding.padTo(slots, bucket.batch, DecodePadding.PADDING_SLOT, "slotMapping")),
                keyPool.copyOf(), valuePool.copyOf(),
            ),
        )
        return Ref(out[0], out[1], out[2])
    }

    private fun buildRequest(): String = """
        {"prompts":${prompts.joinToString(",", "[", "]")},
         "steps":$steps,
         "padding":{"PADDING_TOKEN_ID":${DecodePadding.PADDING_TOKEN_ID},
                    "PADDING_POSITION":${DecodePadding.PADDING_POSITION},
                    "PADDING_BLOCK":${DecodePadding.PADDING_BLOCK},
                    "PADDING_SEQ_LEN":${DecodePadding.PADDING_SEQ_LEN},
                    "PADDING_SLOT":${DecodePadding.PADDING_SLOT}}}
    """.trimIndent()

    private fun assertClose(expected: Float, actual: Float, rel: Float, what: String) {
        val tol = rel * max(1f, abs(expected))
        assertTrue(
            abs(expected - actual) <= tol,
            "$what: interpreter $expected vs plugin $actual (tolerance $tol)",
        )
    }

    private fun harnessPython(): Path =
        Path.of("..", "harness", "python").toAbsolutePath().normalize()

    private fun scriptPath(): Path = harnessPython().resolve("run_vllm_tlaloc_check.py")

    private fun resolvePython(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { if (Files.isExecutable(Path.of(it))) return it }
        val home = System.getProperty("user.home") ?: return null
        val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(venv)) venv.toString() else null
    }

    private fun pythonHasJax(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import jax")
        pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(120, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    /**
     * A small reader over the result JSON, extended from H3a's with an array-
     * of-objects accessor (this slice's result carries one object per step).
     * Deliberately tiny: `:core`'s strict parser reads ARTIFACTS; this reads a
     * file a test wrote.
     */
    private class Json(private val text: String) {
        private fun valueAt(from: Int): Pair<String, Int> {
            var j = from
            while (j < text.length && text[j].isWhitespace()) j++
            var depth = 0
            val start = j
            while (j < text.length) {
                val ch = text[j]
                if (ch == '[' || ch == '{') depth++
                if (ch == ']' || ch == '}') {
                    if (depth == 0) break
                    depth--
                }
                if (ch == ',' && depth == 0) break
                j++
            }
            return text.substring(start, j).trim() to j
        }

        private fun raw(key: String): String {
            val i = text.indexOf("\"$key\"")
            require(i >= 0) { "result JSON has no '$key': ${text.take(400)}" }
            return valueAt(text.indexOf(':', i) + 1).first
        }

        /** The n-th object of the array under [key], as its own reader. */
        fun obj(key: String, index: Int): Json {
            val arr = raw(key)
            var depth = 0
            var start = -1
            var seen = 0
            for (j in arr.indices) {
                when (arr[j]) {
                    '{' -> { if (depth == 0) start = j; depth++ }
                    '}' -> {
                        depth--
                        if (depth == 0) {
                            if (seen == index) return Json(arr.substring(start, j + 1))
                            seen++
                        }
                    }
                }
            }
            throw IllegalArgumentException("'$key' has no element $index")
        }

        fun bool(key: String): Boolean = raw(key) == "true"
        fun int(key: String): Int = raw(key).toInt()
        fun ints(key: String): List<Int> =
            raw(key).replace("[", "").replace("]", "").split(',')
                .filter { it.isNotBlank() }.map { it.trim().toInt() }

        fun floats(key: String): List<Float> =
            raw(key).replace("[", "").replace("]", "").split(',')
                .filter { it.isNotBlank() }.map { it.trim().toFloat() }
    }
}
