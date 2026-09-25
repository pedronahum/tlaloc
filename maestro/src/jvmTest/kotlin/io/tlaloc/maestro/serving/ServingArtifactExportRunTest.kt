package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.DecodePadding
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.maestro.ProgramManifest
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.469 — Phase H3a: **the export-then-run certification**, and the
 * evidence for `docs/INFERENCE_SERVING_AUDIT.md`'s central claim.
 *
 * Kotlin exports an artifact directory. A Python process — `python
 * harness/python/run_tlaloc_serve_check.py --artifact <dir>` — loads it,
 * compiles the StableHLO through jaxlib/PJRT, runs a decode step, and
 * writes its outputs. Nothing Tlaloc-shaped is on that process's import
 * path; there is no JVM in it and no gradle in the loop past the point
 * where the directory exists. Then the two sets of numbers are compared.
 *
 * ## Two lanes, two floors, and why that is not hedging
 *
 * The same artifact runs on the **PJRT CPU client** and on the **PJRT CUDA
 * client**, against the reference interpreter on the host.
 *
 * - CPU, `1e-5` relative — the SEMANTICS claim. XLA-CPU computes an f32
 *   dot in f32, so nothing stands between the two sides and a
 *   disagreement is Tlaloc's.
 * - CUDA, `1e-3` relative — the DEPLOYMENT claim. XLA-GPU lowers a
 *   default-precision f32 `dot_general` through TF32, which measurably
 *   puts this step ~5e-4 relative from the host (the CPU lane sits at
 *   ~1e-7 on the identical bytes, which is how we know it is the
 *   backend's precision policy and not an emission bug).
 *
 * Running only the GPU lane at 1e-3 would have been a semantics test with
 * a thousandfold hole in it; running only the CPU lane would not have
 * certified the thing a deployment runs. This is also H1c's deferred
 * "device-side measurement", answered: the honest number is per-backend.
 * It is a DIFFERENT claim from H1c's bit-identical padding invariant —
 * that one is interpreter-vs-interpreter and can afford `==`.
 *
 * What the tolerance cannot hide: the KV pools are **poisoned** at
 * 1000 + index, so a write that never landed, or landed in the wrong slot,
 * is three orders of magnitude out and not epsilon. And the padded row —
 * the fourth row of a bucket-4 graph running three sequences — must leave
 * its scratch page byte-untouched, which is a claim about WHERE the writes
 * went, not how big they were.
 *
 * ## Self-skip
 *
 * The artifact-shape half of this file runs everywhere. The run half
 * self-skips (no venv python, no jax, no CUDA device) the way every
 * subprocess certification in this repo does, and a script exit of 1 —
 * "the run failed" — is a FAILURE, while 2 — "this machine cannot run it"
 * — is a skip. The distinction is in the script, so a broken artifact
 * cannot present itself as an absent GPU.
 */
class ServingArtifactExportRunTest {

    // --- the scenario (H1c's, so the two slices talk about one thing) ---

    private val realTokenIds = intArrayOf(3, 1, 4)
    private val realPositions = intArrayOf(0, 0, 0)
    /** Sequence s owns page pages[s]; page 0 is deliberately NOT used — it
     *  is the scratch page a padded row's block table names. */
    private val pages = intArrayOf(2, 5, 3)
    private val realSeqLens = intArrayOf(1, 1, 1)
    private val m = ReferenceDecodeGraph.MODEL
    private val realSlots = IntArray(3) { pages[it] * m.blockSize }
    private val poolLen = m.numBlocks * m.blockSize * m.numKvHeads * m.headDim

    private fun poisonedPool(seed: Int) = FloatArray(poolLen) { 1000f + seed * 10_000f + it }

    /** The bucket a batch of three at context three lands in. */
    private val bucket = DecodeBucket(batch = 4, maxContext = 4)
    private val spec = DecodeGraphSpec(m, bucket, DecodeGraphKind.DECODE)

    private fun paddedBlockTables(): IntArray {
        val mbs = spec.maxBlocksPerSeq
        val t = IntArray(bucket.batch * mbs) { DecodePadding.PADDING_BLOCK }
        for (s in pages.indices) t[s * mbs] = pages[s]
        return t
    }

    // --- Oracle 1: the artifact describes itself, and truthfully. -------

    @Test
    fun theExportedDirectoryIsSelfDescribingAndContentAddressed() {
        val dir = Files.createTempDirectory("tlaloc-serving-shape")
        try {
            val manifest = ReferenceDecodeGraph.exportTo(dir)

            // Re-read it the way the loader will: off disk, through the parser.
            val onDisk = ServingManifest.fromJson(
                Files.readString(dir.resolve(ServingManifest.FILE_NAME)),
            )
            assertEquals(manifest, onDisk, "the written manifest must parse back to what was written")
            assertEquals(
                ReferenceDecodeGraph.POLICY.allBuckets.size, manifest.entries.size,
                "one entry per decode ladder point",
            )

            for (e in manifest.entries) {
                val body = dir.resolve(e.bodyPath)
                assertTrue(Files.exists(body), "missing body for ${e.entryId}: ${e.bodyPath}")
                val hash = MessageDigest.getInstance("SHA-256")
                    .digest(Files.readAllBytes(body))
                    .joinToString("") { "%02x".format(it) }
                assertEquals(
                    e.bodyHash, hash,
                    "${e.entryId}: the body must hash to the name it is filed under — content " +
                        "addressing is what lets a loader verify an artifact it was handed",
                )
                val text = Files.readString(body)
                assertTrue(
                    text.contains("func.func @main"),
                    "${e.entryId}: the entry point must be @main, as the manifest states — a " +
                        "loader that has to rename a function is editing a program it was asked to run",
                )
                // The op kinds this slice exists to ship actually survived to MLIR.
                assertTrue(text.contains("dot_general"), "${e.entryId}: no matmul in the body?")

                // The per-entry ProgramManifest is present, parses with its OWN
                // parser, and agrees about the content address.
                val pm = ProgramManifest.fromJson(Files.readString(dir.resolve(e.programPath)))
                assertEquals(e.bodyHash, pm.bodyHash, "${e.entryId}: manifests disagree about the body")
                assertEquals(e.inputs.map { it.type }, pm.inputs, "${e.entryId}: input types disagree")
                assertEquals(e.outputs.map { it.type }, pm.outputs, "${e.entryId}: output types disagree")

                // Donation pairs are H1c's, restated: every pool aliases itself.
                assertEquals(
                    2 * m.numLayers, e.donationPairs.size,
                    "${e.entryId}: every KV pool is a donation candidate",
                )
                // And the body tells XLA so: exactly the paired parameters carry
                // tf.aliasing_output, naming the result they pair with.
                val signature = text.substringAfter("func.func @main(").substringBefore(") -> (")
                val params = signature.split(Regex(", (?=%)"))
                assertEquals(e.inputs.size, params.size, "${e.entryId}: parameter count")
                val pairs = e.donationPairs.associate { it[0] to it[1] }
                val aliasAttr = Regex("""\{tf\.aliasing_output = (\d+) : i32\}""")
                for ((i, param) in params.withIndex()) {
                    val alias = aliasAttr.find(param)?.groupValues?.get(1)?.toInt()
                    assertEquals(
                        pairs[i], alias,
                        "${e.entryId}: parameter $i (${e.inputs[i].name}) must alias " +
                            "${pairs[i] ?: "no result"}: $param",
                    )
                }
                assertEquals(
                    spec.executableCacheKey(ReferenceDecodeGraph.MODEL_HASH).substringBefore("/b"),
                    e.cacheKey.substringBefore("/b"),
                    "${e.entryId}: cache keys must share the model component",
                )
            }

            // Bodies are de-duplicated by content: distinct files <= entries.
            val bodyFiles = Files.list(dir.resolve(ServingArtifactWriter.BODIES_DIR)).use { it.count() }
            assertEquals(
                manifest.entries.map { it.bodyHash }.distinct().size.toLong(), bodyFiles,
                "one file per distinct program, not one per ladder point",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun exportingIsDeterministicSoTheBodyHashIsAnAddressAndNotATimestamp() {
        val a = Files.createTempDirectory("tlaloc-serving-a")
        val b = Files.createTempDirectory("tlaloc-serving-b")
        try {
            assertEquals(
                ReferenceDecodeGraph.exportTo(a).toJson(),
                ReferenceDecodeGraph.exportTo(b).toJson(),
                "two exports of one model must be the same artifact, byte for byte",
            )
        } finally {
            a.toFile().deleteRecursively(); b.toFile().deleteRecursively()
        }
    }

    // --- Oracle 2: Python loads it and the numbers agree. ---------------

    /**
     * The CPU lane: the SEMANTICS claim. XLA-CPU computes an f32 dot in
     * f32, so a disagreement here is Tlaloc's — an emission that means
     * something other than what the interpreter does. Pinned at 1e-5
     * relative, a floor the house rule allows precisely because no device
     * precision policy sits between the two sides.
     *
     * §0.4.476 (H6b): this is the one lane that still rides `engine="jax"`,
     * and the reason is a fact about jaxlib rather than a preference —
     * **jaxlib ships no CPU PJRT plugin `.so`**. Its CPU client is a C++
     * class inside the jaxlib extension, not a loadable plugin, so the
     * ctypes binding (which dlopens a plugin and calls `GetPjrtApi`) has
     * nothing to open. Keeping this lane on jax keeps the tight semantics
     * floor; moving it would have meant deleting a certified row to make a
     * sentence tidier. The day a CPU plugin `.so` exists here — or the TPU
     * VM's `libtpu.so` does (G2b) — this lane switches engine and nothing
     * else changes.
     */
    @Test
    fun theExportedArtifactRunsOnThePjrtCpuClientAndAgreesWithTheInterpreter() {
        runLane(platform = "cpu", tolerance = 1e-5f, engine = "jax")
    }

    /**
     * The CUDA lane on the GB10: the DEPLOYMENT claim — the same directory,
     * unmodified, compiled and run by the backend a deployment actually
     * serves on.
     *
     * Its floor is **1e-3 relative**, and the looseness is measured rather
     * than guessed. XLA-GPU lowers a default-precision f32 `dot_general`
     * through TF32 (a 10-bit mantissa), which puts this step's logits
     * ~5e-4 relative from the host while the CPU lane sits at ~1e-7. Two
     * lanes is how that is stated honestly: the tight one says the program
     * MEANS the right thing, the loose one says the artifact RUNS on the
     * real backend, and neither pretends to be the other. A single
     * GPU-only oracle at 1e-3 would have been a semantics test with a
     * thousandfold hole in it.
     *
     * Named deferral: emitting `precision_config = HIGHEST` on dots that
     * want it (an emitter-wide question, and a serving deployment usually
     * wants TF32's speed), and measuring the top-1 consequence on a real
     * Llama's vocabulary, which needs H3b's un-embedded weights.
     *
     * §0.4.476 (H6b): this lane now runs `engine="ctypes"` — the artifact is
     * compiled and executed through `tlaloc_pjrt`, the ctypes binding of the
     * PJRT C API, under an import guard that raises on jax, jaxlib, torch and
     * numpy. The numbers did not move; what moved is what had to be installed
     * to get them. See [theFullServingPathRunsWithNoFrameworkImported].
     */
    @Test
    fun theSameArtifactRunsOnTheCudaClientOnTheGb10() {
        runLane(platform = "cuda", tolerance = 1e-3f, engine = "ctypes")
    }

    /**
     * §0.4.476 (H6b) — **the slice's headline claim, as an assertion.**
     *
     * The CUDA lane above already runs under the guard; this test is about
     * the guard itself, because a negative claim ("this process imported no
     * framework") is worth exactly as much as the thing that would have
     * caught the positive. Three separate facts, and all three are needed:
     *
     * 1. `sys.modules` carries no `jax`, `jaxlib`, `torch` or `numpy` root
     *    after the whole serving path — manifest parse, body verification,
     *    bucket selection, compile, staging, execute, readback — has run.
     * 2. Nothing was even ATTEMPTED: `blocked_attempts` is empty, so this is
     *    not a path that tried and fell back.
     * 3. The guard FIRED when deliberately provoked, in the interpreter where
     *    jax is genuinely installed. Without this, (1) and (2) are equally
     *    consistent with a guard that was never installed (§0.4.474's
     *    lesson: the canary has to be shown to sing).
     *
     * And the run this is asserted on is the one that produced numbers, not a
     * separate hello-world: the same subprocess reports both.
     */
    @Test
    fun theFullServingPathRunsWithNoFrameworkImported() {
        runLane(platform = "cuda", tolerance = 1e-3f, engine = "ctypes", guardOnly = true)
    }

    private fun runLane(
        platform: String,
        tolerance: Float,
        engine: String,
        guardOnly: Boolean = false,
    ) {
        val python = resolvePython() ?: run {
            println("[skip] no venv python at ~/.local/venvs/iree/bin/python"); return
        }
        if (engine == "jax" && !pythonHasJax(python)) {
            println("[skip] venv python has no jax"); return
        }
        if (engine == "ctypes" && resolvePlugin() == null) {
            println("[skip] no PJRT plugin .so resolved for the ctypes engine"); return
        }

        val dir = Files.createTempDirectory("tlaloc-serving-$platform-$engine")
        try {
            ReferenceDecodeGraph.exportTo(dir)

            val keyPool = poisonedPool(0)
            val valuePool = poisonedPool(1)
            val reqFile = dir.resolve("request.json")
            Files.writeString(reqFile, buildRequest(keyPool, valuePool))
            val outFile = dir.resolve("result.json")

            val cmd = mutableListOf(
                python, scriptPath().toString(),
                "--artifact", dir.toString(),
                "--request", reqFile.toString(),
                "--output", outFile.toString(),
                "--platform", platform,
                "--engine", engine,
            )
            resolvePlugin()?.let { if (engine == "ctypes") { cmd += "--plugin"; cmd += it } }
            val pb = ProcessBuilder(cmd)
            // GB10 is unified-memory: never let JAX preallocate (§0.4.333).
            // The ctypes engine reaches the same setting through
            // PjrtClientOptions instead, which is why it does not need a
            // framework to be talked out of eating the machine.
            pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            pb.redirectErrorStream(true)
            val proc = pb.start()
            val stdout = proc.inputStream.bufferedReader().readText()
            assertTrue(
                proc.waitFor(300, TimeUnit.SECONDS),
                "run_tlaloc_serve_check[$platform] timed out\n$stdout",
            )
            val exit = proc.exitValue()
            val result = if (Files.exists(outFile)) Files.readString(outFile) else "{}"
            if (exit == 2) {
                println("[skip] serving check reports an unrunnable $platform environment: $result"); return
            }
            if (exit != 0) {
                fail("run_tlaloc_serve_check[$platform] failed (exit $exit)\nresult: $result\nstdout:\n$stdout")
            }

            val got = SimpleJson(result)
            assertTrue(got.bool("ok"), "script reported failure: $result")

            if (guardOnly) {
                // §0.4.476 (H6b): the three facts, in the order that makes
                // them mean something. See the test's doc comment.
                assertEquals(
                    "[]", got.raw("loaded_forbidden"),
                    "the serving path loaded a framework it claims not to need: $result",
                )
                assertEquals(
                    "[]", got.raw("blocked_attempts"),
                    "the serving path TRIED to import a framework and was stopped; the claim " +
                        "is that it never reaches for one, not that it is prevented: $result",
                )
                assertTrue(
                    got.bool("guard_self_test_fired"),
                    "the import guard did not fire when provoked — so 'nothing was imported' " +
                        "is equally consistent with a guard that was never installed (§0.4.474): $result",
                )
                assertEquals("\"ctypes\"", got.raw("engine"), "the guarded lane must be the ctypes one")
                return
            }
            assertEquals(
                listOf(4, 4), got.ints("bucket"),
                "a batch of 3 at context 3 belongs in bucket (4,4) — bucket selection is the " +
                    "Python side's, read off the manifest's ladder",
            )
            assertEquals(
                listOf(3, 1, m.vocabSize), got.ints("logitsShape"),
                "logits come back SLICED to the real rows: the padding row never leaves the loader",
            )
            assertEquals(ReferenceDecodeGraph.POLICY.allBuckets.size, got.int("entryCount"))
            // The executable cache is keyed by the manifest's cacheKey: a second
            // step at the same bucket must not compile a second executable.
            assertEquals(
                got.int("compileCountAfterFirst"), got.int("compileCountAfterSecond"),
                "a repeated bucket re-compiled; the manifest's cacheKey is not doing its job",
            )
            assertEquals(1, got.int("compileCountAfterFirst"), "one bucket, one executable")
            assertTrue(got.bool("repeatLogitsMatch"), "the same step ran twice gave different logits")

            // The reference: the same padded step in the host interpreter.
            val ref = interpreterStep(keyPool, valuePool)
            val v = m.vocabSize
            val gotLogits = got.floats("logits")
            assertEquals(3 * v, gotLogits.size)
            for (row in 0 until 3) {
                for (j in 0 until v) {
                    assertClose(ref.logits[row * v + j], gotLogits[row * v + j], tolerance, "[$platform] logit[$row][$j]")
                }
            }
            assertTrue(
                gotLogits.distinct().size > 3,
                "the exported model produced a degenerate logit vector; the oracle would pass on anything",
            )

            val gotKey = got.floats("keyPool")
            val gotValue = got.floats("valuePool")
            assertEquals(poolLen, gotKey.size)
            for (i in 0 until poolLen) {
                assertClose(ref.keyPool[i], gotKey[i], tolerance, "[$platform] keyPool[$i]")
                assertClose(ref.valuePool[i], gotValue[i], tolerance, "[$platform] valuePool[$i]")
            }

            // The padded row is INERT, and this half is pinned EXACTLY on both
            // lanes: scratch page 0 — which the padded row's block table names
            // and which it therefore READ — is copied, not computed, so no
            // device precision policy can excuse a difference.
            val perPage = m.blockSize * m.numKvHeads * m.headDim
            for (i in 0 until perPage) {
                assertTrue(
                    keyPool[i] == gotKey[i],
                    "[$platform] scratch page 0 slot $i changed from ${keyPool[i]} to ${gotKey[i]}; " +
                        "the padded row's slotMapping is ${DecodePadding.PADDING_SLOT} and that " +
                        "write must be dropped",
                )
                assertTrue(
                    valuePool[i] == gotValue[i],
                    "[$platform] scratch page 0 (value) slot $i changed",
                )
            }
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // --- helpers --------------------------------------------------------

    /** Relative-with-absolute-floor comparison; see each lane for its number. */
    private fun assertClose(expected: Float, actual: Float, rel: Float, what: String) {
        val tol = rel * max(1f, abs(expected))
        assertTrue(
            abs(expected - actual) <= tol,
            "$what: interpreter $expected vs PJRT $actual (tolerance $tol)",
        )
    }

    private class Ref(val logits: FloatArray, val keyPool: FloatArray, val valuePool: FloatArray)

    private fun interpreterStep(keyPool: FloatArray, valuePool: FloatArray): Ref {
        fun f(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val out = DxirInterpreter.evalFunction(
            ReferenceDecodeGraph.build(spec),
            listOf(
                f(DecodePadding.padTo(realTokenIds, 4, DecodePadding.PADDING_TOKEN_ID, "tokenIds")),
                f(DecodePadding.padTo(realPositions, 4, DecodePadding.PADDING_POSITION, "positions")),
                f(paddedBlockTables()),
                f(DecodePadding.padTo(realSeqLens, 4, DecodePadding.PADDING_SEQ_LEN, "seqLens")),
                f(DecodePadding.padTo(realSlots, 4, DecodePadding.PADDING_SLOT, "slotMapping")),
                keyPool.copyOf(), valuePool.copyOf(),
            ),
        )
        return Ref(out[0], out[1], out[2])
    }

    /**
     * The request the Python side gets. Note the `padding` block: the
     * loader's own copy of the wire convention is checked against
     * [DecodePadding] rather than trusted, so the two processes cannot pad
     * a batch differently and agree anyway.
     */
    private fun buildRequest(keyPool: FloatArray, valuePool: FloatArray): String {
        fun arr(a: IntArray) = a.joinToString(",", "[", "]")
        fun farr(a: FloatArray) = a.joinToString(",", "[", "]")
        val tables = pages.joinToString(",", "[", "]") { "[$it]" }
        return """
            {"tokenIds":${arr(realTokenIds)},
             "positions":${arr(realPositions)},
             "blockTables":$tables,
             "seqLens":${arr(realSeqLens)},
             "slotMapping":${arr(realSlots)},
             "context":3,
             "padding":{"PADDING_TOKEN_ID":${DecodePadding.PADDING_TOKEN_ID},
                        "PADDING_POSITION":${DecodePadding.PADDING_POSITION},
                        "PADDING_BLOCK":${DecodePadding.PADDING_BLOCK},
                        "PADDING_SEQ_LEN":${DecodePadding.PADDING_SEQ_LEN},
                        "PADDING_SLOT":${DecodePadding.PADDING_SLOT}},
             "keyPool":${farr(keyPool)},
             "valuePool":${farr(valuePool)}}
        """.trimIndent()
    }

    private fun resolvePython(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { if (Files.isExecutable(Path.of(it))) return it }
        val home = System.getProperty("user.home") ?: return null
        val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(venv)) venv.toString() else null
    }

    /**
     * §0.4.476 (H6b) — the PJRT plugin `.so` the ctypes engine dlopens.
     *
     * `:maestro` does not depend on `:runtime-pjrt`, so this mirrors
     * `PjrtBinaries`'s resolution order rather than calling it: the env var
     * first (what a deployment sets), then the plugin file that happens to
     * live inside the venv's jax install. Note what that second source is —
     * a FILE PATH, not an import. The whole point of the slice is that the
     * plugin does not know a Python package delivered it.
     */
    private fun resolvePlugin(): String? {
        System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.let { if (Files.exists(Path.of(it))) return it }
        val home = System.getProperty("user.home") ?: return null
        val venv = Path.of(home, ".local", "venvs", "iree", "lib")
        if (!Files.isDirectory(venv)) return null
        Files.list(venv).use { pys ->
            for (py in pys) {
                val p = py.resolve("site-packages")
                    .resolve("jax_plugins").resolve("xla_cuda12").resolve("xla_cuda_plugin.so")
                if (Files.exists(p)) return p.toString()
            }
        }
        return null
    }

    private fun pythonHasJax(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import jax")
        pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(60, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun scriptPath(): Path =
        Path.of("..", "harness", "python", "run_tlaloc_serve_check.py").toAbsolutePath().normalize()

    /**
     * A three-field reader over the result JSON. Deliberately tiny: `:core`'s
     * strict parser is the reader for artifacts, and this is a test reading a
     * file a test wrote.
     */
    private class SimpleJson(private val text: String) {
        fun raw(key: String): String {
            val i = text.indexOf("\"$key\"")
            require(i >= 0) { "result JSON has no '$key': $text" }
            val c = text.indexOf(':', i) + 1
            var j = c
            var depth = 0
            while (j < text.length) {
                val ch = text[j]
                if (ch == '[') depth++
                if (ch == ']') depth--
                if (ch == ',' && depth == 0) break
                if (ch == '}' && depth == 0) break
                j++
            }
            return text.substring(c, j).trim()
        }

        fun bool(key: String): Boolean = raw(key) == "true"
        fun int(key: String): Int = raw(key).toInt()
        fun ints(key: String): List<Int> = raw(key).trim('[', ']').split(',').map { it.trim().toInt() }
        fun floats(key: String): List<Float> =
            raw(key).trim('[', ']').split(',').map { it.trim().toFloat() }
    }
}
