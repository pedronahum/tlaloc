package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.NpyWriter
import io.tlaloc.runtime.iree.runOnIree
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.289 — numerical-correctness evidence for the LlamaDecoder forward
 * dispatched via Tlaloc-IREE-CPU, vs an independent PyTorch reference.
 *
 * # Cross-language flow
 *
 *   1. Build LlamaDecoderPrimal at tiny config + run the CPU baseline lowering
 *      pipeline (`recognize → coarsen → decomposeCoarsened → toStablehlo`).
 *   2. Synthesize deterministic inputs (the same recipe §0.4.288 uses, seed
 *      held fixed so this test pins a stable cross-runtime comparison).
 *   3. Write the inputs as `.npy` files to a JVM temp dir (one per param,
 *      keyed by name) — see [NpyWriter].
 *   4. Run Tlaloc-IREE forward via [runOnIree] → loss_iree (single f32).
 *   5. Spawn `harness/python/run_pytorch_llama.py` as a subprocess: it loads
 *      the same `.npy` inputs into PyTorch tensors, runs a hand-translated
 *      forward that mirrors LlamaDecoderPrimal.kt op-for-op, and writes
 *      `{"loss": <float>}` to a JSON output file.
 *   6. Read loss_pytorch from the JSON and compare to loss_iree at a
 *      relative tolerance (1e-3 — generous enough to absorb FP32 reduction-
 *      order differences between the two backends, tight enough to catch
 *      any structural divergence).
 *
 * Self-skips when either toolchain is missing: the IREE binaries (same
 * resolution as IreeBinaries) or the venv'd `python` + `torch` install.
 *
 * # Why this is "real" numerical correctness, vs §0.4.288's self-consistency
 *
 * §0.4.288 only pinned that Tlaloc-IREE produces a finite, deterministic
 * scalar — a structural-reachability claim. This test pins that the scalar
 * Tlaloc produces *equals* (within 1e-3 relative tolerance) what PyTorch
 * produces on the *same inputs* with the *same forward math*. Discrepancies
 * here would point at: a missing op lowering, a dtype-promotion bug, an
 * axis convention mismatch, or a numerical-stability issue specific to one
 * runtime. The PyTorch reference lives at `harness/python/run_pytorch_llama.py`
 * and is op-for-op aligned with `LlamaDecoderPrimal.kt`'s forward.
 */
class LlamaDecoderIreeVsPytorchTest {

    private fun resolvePythonBinary(): String? {
        // Same resolution philosophy as IreeBinaries: env override → known venv → PATH.
        System.getenv("TLALOC_TORCH_PYTHON")?.let { p ->
            if (Files.isExecutable(Path.of(p))) return p
        }
        val home = System.getProperty("user.home")
        if (home != null) {
            val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
            if (Files.isExecutable(venv)) return venv.toString()
        }
        return null
    }

    private fun pythonHasTorch(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import torch")
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(15, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun resolveScriptPath(): Path {
        // Gradle runs `:benchmarks:jvmTest` with cwd = `<repo>/benchmarks/`. The
        // PyTorch reference lives at `<repo>/harness/python/run_pytorch_llama.py`,
        // so the relative path is `../harness/python/run_pytorch_llama.py`.
        return Path.of("..", "harness", "python", "run_pytorch_llama.py").toAbsolutePath().normalize()
    }

    @Test
    fun llamaDecoderLossAgreesWithPytorchReference() {
        // Skip ladder: IREE first (the test depends on Tlaloc-IREE producing the
        // baseline number), then python+torch (without which we can't compute the
        // reference). Each emits a clear assumeTrue message so partial-environment
        // hosts know exactly what's missing.
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree.",
        )
        val python = resolvePythonBinary()
        assumeTrue(
            python != null,
            "no python interpreter resolved — skipping. " +
                "Set TLALOC_TORCH_PYTHON to a python with torch installed, " +
                "or `pip install torch --index-url https://download.pytorch.org/whl/cpu` into ~/.local/venvs/iree.",
        )
        assumeTrue(
            pythonHasTorch(python!!),
            "python at $python cannot `import torch` — skipping.",
        )
        val script = resolveScriptPath()
        assumeTrue(
            Files.exists(script),
            "PyTorch reference script not found at $script — expected at harness/python/run_pytorch_llama.py.",
        )

        // ---- Tlaloc-IREE side ------------------------------------------------------
        val fn = llamaCpuBaselinePipeline()
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)
        val ireeOutputs = runOnIree(fn, inputs)
        val lossIree = ireeOutputs.single().single()

        // ---- PyTorch side ----------------------------------------------------------
        val workDir = Files.createTempDirectory("tlaloc-llama-vs-pytorch-")
        workDir.toFile().deleteOnExit()
        for ((i, p) in fn.params.withIndex()) {
            val target = workDir.resolve("${p.name}.npy")
            NpyWriter.writeFloat32(target, inputs[i], p.type.dims)
            target.toFile().deleteOnExit()
        }
        val outputJson = workDir.resolve("pytorch-loss.json")
        outputJson.toFile().deleteOnExit()

        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--inputs-dir", workDir.toString(),
            "--output", outputJson.toString(),
        )
        pb.redirectErrorStream(false)
        val proc = pb.start()
        val finished = proc.waitFor(60, TimeUnit.SECONDS)
        if (!finished) {
            proc.destroyForcibly()
            error("PyTorch reference subprocess timed out after 60s")
        }
        val stderr = proc.errorStream.bufferedReader().readText()
        if (proc.exitValue() != 0) {
            error("PyTorch reference exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}")
        }

        // Trivial JSON parse — the script writes exactly `{"loss": <float>}`.
        val raw = Files.readString(outputJson).trim()
        val lossPytorch = parseLossJson(raw)

        val absDiff = abs(lossIree - lossPytorch)
        val relDiff = absDiff / max(1.0f, max(abs(lossIree), abs(lossPytorch)))
        val tol = 1e-3f
        println(
            "[llama-decoder-iree-vs-pytorch] iree=$lossIree pytorch=$lossPytorch " +
                "absDiff=$absDiff relDiff=$relDiff (tol=$tol)",
        )
        assertTrue(
            relDiff < tol,
            "Tlaloc-IREE-CPU loss disagrees with PyTorch reference: " +
                "iree=$lossIree pytorch=$lossPytorch absDiff=$absDiff relDiff=$relDiff (tol=$tol).\n" +
                "PyTorch stderr (last 2 KB):\n${stderr.takeLast(2000)}",
        )
    }

    private fun parseLossJson(raw: String): Float {
        val key = "\"loss\""
        val ki = raw.indexOf(key)
        require(ki >= 0) { "loss JSON missing 'loss' key: $raw" }
        val colon = raw.indexOf(':', ki + key.length)
        require(colon >= 0) { "loss JSON missing ':' after 'loss': $raw" }
        var end = colon + 1
        while (end < raw.length && raw[end] != ',' && raw[end] != '}') end++
        val numToken = raw.substring(colon + 1, end).trim()
        return numToken.toFloatOrNull()
            ?: error("loss JSON: '$numToken' is not a parseable Float (raw=$raw)")
    }
}
