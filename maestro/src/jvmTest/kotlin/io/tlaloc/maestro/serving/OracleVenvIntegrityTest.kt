package io.tlaloc.maestro.serving

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.474 — **the oracle-venv canary.**
 *
 * `~/.local/venvs/iree` is not a convenience. It is the single place where
 * every cross-language claim in this repository gets its *other side*:
 *
 * | what runs there | what it certifies |
 * |---|---|
 * | `run_pytorch_llama.py`, `run_pytorch_llama_grad.py`, `run_pytorch*.py` | the PyTorch agreement harness — Tlaloc's forward and its gradients |
 * | `write_llama_safetensors.py` | H2's bit-exact safetensors parity (torch writes the bytes we then claim to read) |
 * | `tlaloc_serve.py` / `run_tlaloc_serve_check.py` | H3a's "a Python process with no JVM runs the artifact", at 1e-5 CPU / 1e-3 CUDA |
 * | the `vllm_tlaloc` package + `run_vllm_tlaloc_check.py` | H3b's plugin contract and the two-step decode runner |
 * | `jax_plugins/xla_cuda12/xla_cuda_plugin.so` | **every PJRT lane in the repo**, JVM-side included — `PjrtBinaries` resolves the plugin out of this venv's site-packages |
 * | `iree-base-compiler` / `iree-base-runtime` | the IREE compile-and-run lanes |
 *
 * Nothing in that list *fails loudly* when the venv is mutated. A `pip
 * install` that swapped `torch 2.11.0+cpu` for a CUDA build, or bumped
 * `jax` a minor version, or set CUDA-13 wheels down beside jax's CUDA-12
 * plugin, would leave the build green and quietly **move every number this
 * repo compares itself against**. That is the failure mode this test
 * exists to convert into a red line with an explanation attached.
 *
 * ## The incident that motivates it (§0.4.470)
 *
 * The H3b agent needed vLLM and dry-ran `pip install vllm` into this venv.
 * The plan came back with **186 packages**, including `torch 2.13`
 * replacing the oracle torch and **33 CUDA-13 wheels** landing beside
 * jax's CUDA-12 plugin. It refused, and that refusal is why the live-vLLM
 * certification is still recorded as a named deferral rather than as a
 * number nobody could reproduce. This test is that refusal, made automatic
 * and made cheap: the next agent that is tempted does not have to be
 * careful, it has to be *correct*.
 *
 * ## What is pinned, and what is merely required
 *
 * Exact pins below for the three packages whose *version* is load-bearing
 * in a certification: `jax`, `jaxlib`, and `torch`. `numpy`, `safetensors`
 * and the IREE pair are asserted **present**, not pinned — their versions
 * do not appear in any tolerance, and pinning them would buy a brittle
 * test rather than a claim. The CUDA plugin family is pinned to jax's own
 * version because a plugin/jax version skew is a PJRT load failure, not a
 * numerical drift.
 *
 * `torch 2.11.0+cpu` is pinned **including its local tag**, and the tag is
 * the point. A CUDA torch here would not be an upgrade, it would be a
 * silent swap of the reference implementation: the oracle's job is to
 * compute the answer a *deterministic host* computes, and a torch that
 * quietly dispatches to cuDNN is no longer that instrument. The
 * `torch.version.cuda is None` probe is asserted alongside the version
 * string because a local tag can be stripped and a version number cannot
 * be trusted to carry that fact.
 *
 * **Bumping any constant in this file is a deliberate act.** It means
 * re-certifying the oracles that depend on it — at minimum the PyTorch
 * agreement harness, the safetensors parity test and both serving lanes —
 * and saying so in the commit that moves the pin. It is not maintenance.
 *
 * ## Self-skip
 *
 * No venv at the path, or a venv that carries neither `jax` nor `torch`
 * (so: some other venv that happens to live there, not the oracle), skips.
 * A fresh clone on a machine that never built the oracle stays green. What
 * does *not* skip is a venv that has the oracle packages and has the wrong
 * ones — that is precisely the case worth being loud about.
 *
 * Deliberately **no `TLALOC_TORCH_PYTHON` override**, unlike every other
 * subprocess certification here: those ask "is there an interpreter that
 * can run my reference?", and any answer will do. This one asks "is *the*
 * oracle venv intact?", and an override would let the thing under test
 * point the test somewhere else.
 */
class OracleVenvIntegrityTest {

    private companion object {
        /** The oracle venv's path, relative to `$HOME`. Not configurable; see the KDoc. */
        val VENV_PYTHON = listOf(".local", "venvs", "iree", "bin", "python")

        // --- exact pins: bumping one of these requires re-certifying its oracles ---

        /** jax and jaxlib move together; every PJRT lane and `tlaloc_serve.py` ride on them. */
        const val JAX = "0.10.0"

        /**
         * CPU-only, and the `+cpu` local tag is load-bearing: this torch is a
         * *numerical reference*, not an accelerator. A CUDA build here is a
         * silent swap, not an upgrade.
         */
        const val TORCH = "2.11.0+cpu"

        /** Must match [JAX]: a plugin/jax skew is a PJRT load failure. */
        val CUDA12_PLUGIN_FAMILY = listOf("jax-cuda12-plugin", "jax-cuda12-pjrt")

        /** Present, not pinned — no tolerance in this repo quotes their versions. */
        val REQUIRED_PRESENT = listOf("numpy", "safetensors", "iree-base-compiler", "iree-base-runtime")

        /**
         * The §0.4.470 dry run's signature. CUDA-13 wheels beside a CUDA-12
         * jax plugin is the shape a `pip install vllm` (or a torch upgrade)
         * leaves behind, and it is the one that breaks the PJRT lanes in a
         * way that reads as a driver problem.
         */
        val FORBIDDEN_PREFIXES = listOf("jax-cuda13")
        const val FORBIDDEN_CUDA13_SUFFIX = "-cu13"
    }

    private val why = """
        |WHY THIS MATTERS: ~/.local/venvs/iree is FROZEN ORACLE INFRASTRUCTURE.
        |Every cross-language certification in this repo runs its reference side
        |there — the PyTorch agreement harness, H2's safetensors parity, both
        |serving lanes, every PJRT lane (the JVM resolves xla_cuda_plugin.so out
        |of its site-packages), and the IREE lanes. Mutating it does not break a
        |build; it silently moves every number this repo compares against.
        |
        |WHAT TO DO: do NOT pip install / upgrade / uninstall anything here.
        |If a slice needs packages, give it its OWN venv:
        |    python -m venv ~/.local/venvs/<slice> && . ~/.local/venvs/<slice>/bin/activate
        |and point the harness at it explicitly. See docs/SERVING_RUNBOOK.md §0.1.
        |
        |If a version here was bumped ON PURPOSE, that is a re-certification, not
        |a fix: re-run the oracles, then move the pin in OracleVenvIntegrityTest
        |in the same commit that says why. (§0.4.470 is the precedent: a
        |`pip install vllm` dry run showed 186 packages, torch 2.13 replacing the
        |oracle torch, and 33 CUDA-13 wheels. It was refused.)
    """.trimMargin()

    @Test
    fun theOracleVenvIsIntact() {
        val home = System.getProperty("user.home") ?: run {
            println("[skip] no user.home"); return
        }
        val python = VENV_PYTHON.fold(Path.of(home)) { p, s -> p.resolve(s) }
        if (!Files.isExecutable(python)) {
            println("[skip] no oracle venv python at ~/${VENV_PYTHON.joinToString("/")}"); return
        }

        val report = runReporter(python.toString())
        val dists = report.filter { it.startsWith("dist ") }
            .associate { val f = it.split(' '); f[1] to f[2] }
        val facts = report.filter { it.startsWith("fact ") }
            .associate { val f = it.split(' ', limit = 3); f[1] to (f.getOrNull(2) ?: "") }

        if ("jax" !in dists && "torch" !in dists) {
            println(
                "[skip] a venv lives at ~/${VENV_PYTHON.joinToString("/")} but carries neither " +
                    "jax nor torch — it is not the oracle venv, so there is nothing here to protect",
            )
            return
        }

        val problems = problemsWith(dists, facts)
        if (problems.isNotEmpty()) {
            fail(
                buildString {
                    appendLine("THE ORACLE VENV HAS BEEN MUTATED.")
                    appendLine()
                    problems.forEach { appendLine("  - $it") }
                    appendLine()
                    appendLine(why)
                    appendLine()
                    appendLine("installed: " + dists.entries.sortedBy { it.key }.joinToString { "${it.key}==${it.value}" })
                },
            )
        }
    }

    /**
     * The policy, as a pure function of the report, so that it can be
     * certified against a venv that is *wrong* — see
     * [theCanaryCatchesTheShapeItWasWrittenFor]. A canary that has only
     * ever been run against a healthy venv is an untested canary.
     */
    private fun problemsWith(dists: Map<String, String>, facts: Map<String, String>): List<String> {
        val problems = mutableListOf<String>()

        fun pin(name: String, expected: String) {
            when (val got = dists[name]) {
                null -> problems += "$name is MISSING (oracle pins $name==$expected)"
                expected -> Unit
                else -> problems += "$name is $got, oracle pins $expected"
            }
        }

        pin("jax", JAX)
        pin("jaxlib", JAX)
        pin("torch", TORCH)
        for (p in CUDA12_PLUGIN_FAMILY) pin(p, JAX)
        for (p in REQUIRED_PRESENT) if (p !in dists) problems += "$p is MISSING (oracle requires it present)"

        // The probe a version string cannot answer.
        when (val cuda = facts["torch.version.cuda"]) {
            null -> problems += "torch did not import: ${facts["torch.import_error"] ?: "(no reason reported)"}"
            "None" -> Unit
            else -> problems +=
                "torch.version.cuda is $cuda — a CUDA torch landed in the oracle venv. This is a SILENT " +
                    "SWAP of the reference implementation, not an upgrade: the oracle's job is to compute " +
                    "what a deterministic host computes."
        }

        val intruders = dists.keys.filter { name ->
            FORBIDDEN_PREFIXES.any { name.startsWith(it) } || name.endsWith(FORBIDDEN_CUDA13_SUFFIX)
        }
        if (intruders.isNotEmpty()) {
            problems += "CUDA-13 wheels beside the CUDA-12 jax plugin: ${intruders.sorted()} " +
                "(the §0.4.470 `pip install vllm` signature — this breaks the PJRT lanes in a way " +
                "that reads as a driver problem)"
        }
        return problems
    }

    /**
     * The negative case, with no venv in it: feed the policy the inventory
     * the §0.4.470 dry run predicted — `torch 2.13.0` (CUDA, no `+cpu`
     * tag) replacing the oracle torch, a `jax-cuda13` plugin family, and
     * CUDA-13 nvidia wheels beside the CUDA-12 ones — and require that it
     * names each one. This is the claim the canary actually makes; the
     * live test above only ever gets to say "and today it is fine".
     */
    @Test
    fun theCanaryCatchesTheShapeItWasWrittenFor() {
        val healthy = mapOf(
            "jax" to JAX, "jaxlib" to JAX,
            "jax-cuda12-plugin" to JAX, "jax-cuda12-pjrt" to JAX,
            "torch" to TORCH,
        ) + REQUIRED_PRESENT.associateWith { "1.0" }
        val facts = mapOf("torch.version.cuda" to "None")

        // Sanity: the policy is not simply always-unhappy.
        assertEquals(emptyList(), problemsWith(healthy, facts), "the recorded oracle inventory must pass")

        val mutated = healthy + mapOf(
            "torch" to "2.13.0",
            "jax-cuda13-plugin" to "0.10.0",
            "nvidia-cublas-cu13" to "13.0.0",
        )
        val found = problemsWith(mutated, mapOf("torch.version.cuda" to "13.0")).joinToString("\n")
        assertTrue("torch is 2.13.0" in found, "the torch swap must be named:\n$found")
        assertTrue("SILENT SWAP" in found, "and named as a swap, not an upgrade:\n$found")
        assertTrue("jax-cuda13-plugin" in found, "the CUDA-13 plugin must be named:\n$found")
        assertTrue("nvidia-cublas-cu13" in found, "the CUDA-13 wheels must be named:\n$found")

        // And a quiet single-package drift — the kind a `pip install -U` leaves — is caught too.
        assertTrue(
            "jax is 0.11.0, oracle pins 0.10.0" in problemsWith(healthy + ("jax" to "0.11.0"), facts).joinToString("\n"),
            "a one-version jax bump must be loud: every PJRT lane's plugin is version-matched to it",
        )
    }

    private fun runReporter(python: String): List<String> {
        val script = Path.of("..", "harness", "python", "check_oracle_venv.py").toAbsolutePath().normalize()
        val pb = ProcessBuilder(python, script.toString())
        // GB10 is unified-memory: never let anything on this path preallocate (§0.4.333).
        pb.environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val out = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(180, TimeUnit.SECONDS)) {
            proc.destroyForcibly()
            fail("check_oracle_venv.py timed out\n$out")
        }
        if (proc.exitValue() != 0) {
            fail("check_oracle_venv.py failed (exit ${proc.exitValue()})\n$out\n\n$why")
        }
        return out.lines().map { it.trim() }.filter { it.isNotEmpty() }
    }
}
