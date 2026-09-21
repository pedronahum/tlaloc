package io.tlaloc.benchmarks

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.step
import io.tlaloc.runtime.iree.NpyWriter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.444 — Phase F8 (3): PyTorch convergence parity for the F1→F3 training
 * loop, the §0.4.289 cross-language pattern (npy export → subprocess → JSON).
 * SAME initial weights (exported from the Kotlin side — threefry ≠ torch
 * RNG), same data, same MSE loss, same Adam hyperparameters (both sides are
 * Kingma–Ba bias-corrected; the F0 §4.0.2/§4.0.8 finding that DiffKT's own
 * Adam is a TODO placeholder is exactly why PyTorch is the oracle here), 50
 * identical full-batch steps.
 *
 * What is pinned, and what was OBSERVED (torch 2.x CPU, GB10 aarch64):
 *  - step-0 loss at 1e-5 RELATIVE (observed 9.8e-8): a pure forward on
 *    shared bits — only f32 matmul accumulation order separates the stacks.
 *  - the first 10 per-step losses at 1e-4 relative (observed max 5.9e-7).
 *  - the final loss at 1e-2 relative (observed 9.3e-8 — the two f32
 *    trajectories track each other far tighter than the slice's
 *    float-order-divergence contingency anticipated; at this size the
 *    matmuls are order-stable and Adam's `m/√v` never forks), PLUS both
 *    finals under the task's convergence bound 0.05. The pins sit 2–4
 *    orders above the observed diffs deliberately: they absorb
 *    torch-version/libm drift without ever admitting a structural
 *    divergence, and the observed numbers are recorded here so a future
 *    regression to "merely within tolerance" is visible in the log line.
 *
 * Self-skips when the venv python or torch is missing (the
 * LlamaDecoderIreeVsPytorchTest ladder).
 */
class NnMlpVsPytorchTrainingTest {

    private val n = 16
    private val d = 4
    private val steps = 50
    private val lr = 0.02f

    private fun resolvePythonBinary(): String? {
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

    @Test
    fun fiftyAdamStepsTrackPytorchOnSharedInitAndData() {
        val python = resolvePythonBinary()
        assumeTrue(
            python != null,
            "no python interpreter resolved — skipping. Set TLALOC_TORCH_PYTHON or install " +
                "torch into ~/.local/venvs/iree.",
        )
        assumeTrue(pythonHasTorch(python!!), "python at $python cannot `import torch` — skipping.")
        val script = Path.of("..", "harness", "python", "run_pytorch_nn_train.py")
            .toAbsolutePath().normalize()
        assumeTrue(Files.exists(script), "reference script not found at $script — skipping.")

        // ---- the shared task + shared init --------------------------------
        val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        val xv = FloatArray(u.size) { i -> 2f * u[i] - 1f }
        val yv = FloatArray(n) { i ->
            val r = i * d
            xv[r] * xv[r + 1] + 0.5f * xv[r + 2] - 0.25f * xv[r + 3]
        }
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        var model = Sequential(
            Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
            ReluLayer,
            Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
        )

        // ---- PyTorch side -------------------------------------------------
        val workDir = Files.createTempDirectory("tlaloc-nn-vs-pytorch-")
        workDir.toFile().deleteOnExit()
        NpyWriter.writeFloat32(workDir.resolve("x.npy"), xv, listOf(n, d))
        NpyWriter.writeFloat32(workDir.resolve("y.npy"), yv, listOf(n, 1))
        val init = model.parameters.associate { it.key to it.tensor }
        NpyWriter.writeFloat32(workDir.resolve("w1.npy"), init["0.w"]!!.hostF32(), listOf(d, 8))
        NpyWriter.writeFloat32(workDir.resolve("b1.npy"), init["0.b"]!!.hostF32(), listOf(8))
        NpyWriter.writeFloat32(workDir.resolve("w2.npy"), init["2.w"]!!.hostF32(), listOf(8, 1))
        NpyWriter.writeFloat32(workDir.resolve("b2.npy"), init["2.b"]!!.hostF32(), listOf(1))
        workDir.toFile().listFiles()?.forEach { it.deleteOnExit() }

        val outputJson = workDir.resolve("pytorch-train.json")
        outputJson.toFile().deleteOnExit()
        val pb = ProcessBuilder(
            python, script.toString(),
            "--inputs-dir", workDir.toString(),
            "--output", outputJson.toString(),
            "--steps", steps.toString(),
            "--lr", lr.toString(),
        )
        pb.redirectErrorStream(false)
        val proc = pb.start()
        check(proc.waitFor(120, TimeUnit.SECONDS)) {
            proc.destroyForcibly(); "PyTorch reference subprocess timed out after 120s"
        }
        val stderr = proc.errorStream.bufferedReader().readText()
        check(proc.exitValue() == 0) {
            "PyTorch reference exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}"
        }
        val raw = Files.readString(outputJson).trim()
        val torchLosses = parseFloatList(raw, "losses")
        val torchFinal = parseScalar(raw, "final_loss")
        assertEquals(steps, torchLosses.size, "torch loss-curve length")

        // ---- Tlaloc side: the same 50 steps through the captured graph ----
        val capturedStep = capture(model, listOf(x)) { y ->
            val t = y.constant<Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }
        val opt = Adam(learningRate = lr)
        var state = opt.initialState()
        val ourLosses = ArrayList<Float>(steps)
        repeat(steps) {
            val res = capturedStep.run(model, listOf(x))
            ourLosses += res.loss
            val (next, nextState) = opt.step(model, res.gradients, state)
            model = next
            state = nextState
        }
        val ourFinal = capturedStep.run(model, listOf(x)).loss

        // ---- the pins -----------------------------------------------------
        fun rel(a: Float, b: Float) = abs(a - b) / max(1e-6f, max(abs(a), abs(b)))

        println(
            "[nn-vs-pytorch] step0: tlaloc=${ourLosses[0]} torch=${torchLosses[0]} rel=${rel(ourLosses[0], torchLosses[0])}",
        )
        assertTrue(
            rel(ourLosses[0], torchLosses[0]) < 1e-5f,
            "step-0 forward on shared bits diverges: tlaloc=${ourLosses[0]} torch=${torchLosses[0]}",
        )
        var maxEarly = 0f
        for (i in 0 until 10) maxEarly = maxOf(maxEarly, rel(ourLosses[i], torchLosses[i]))
        println("[nn-vs-pytorch] steps 0-9 max rel diff=$maxEarly")
        assertTrue(maxEarly < 1e-4f, "early curve diverges: max rel diff over steps 0-9 = $maxEarly")

        val finalRel = rel(ourFinal, torchFinal)
        println(
            "[nn-vs-pytorch] final (after $steps steps): tlaloc=$ourFinal torch=$torchFinal rel=$finalRel",
        )
        assertTrue(
            finalRel < 1e-2f,
            "final losses diverge beyond the 1e-2 band: tlaloc=$ourFinal torch=$torchFinal",
        )
        assertTrue(ourFinal < 0.05f, "Tlaloc lane failed to converge: $ourFinal (curve $ourLosses)")
        assertTrue(torchFinal < 0.05f, "PyTorch lane failed to converge: $torchFinal")
    }

    // ---- trivial JSON extraction (the harness writes exactly one object) ----

    private fun parseFloatList(raw: String, key: String): List<Float> {
        val ki = raw.indexOf("\"$key\"")
        require(ki >= 0) { "JSON missing '$key': ${raw.take(200)}" }
        val open = raw.indexOf('[', ki)
        val close = raw.indexOf(']', open)
        require(open in 0 until close) { "JSON '$key' is not a list: ${raw.take(200)}" }
        return raw.substring(open + 1, close).split(',').map { it.trim().toFloat() }
    }

    private fun parseScalar(raw: String, key: String): Float {
        val ki = raw.indexOf("\"$key\"")
        require(ki >= 0) { "JSON missing '$key': ${raw.take(200)}" }
        val colon = raw.indexOf(':', ki)
        var end = colon + 1
        while (end < raw.length && raw[end] != ',' && raw[end] != '}') end++
        return raw.substring(colon + 1, end).trim().toFloat()
    }
}
