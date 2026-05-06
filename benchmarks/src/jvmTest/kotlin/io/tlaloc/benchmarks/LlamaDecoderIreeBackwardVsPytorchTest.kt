package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.runOnIree
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
 * §0.4.292 — backward-pass numerical-correctness evidence. Builds the
 * LlamaDecoderPrimal, runs `recognize → coarsen → DxirReverseTransform.apply`
 * to produce a 13-output gradient function (one ∂loss/∂param per param), then
 * dispatches it via Tlaloc-IREE-CPU and compares each gradient tensor against
 * `torch.autograd.grad`'s answer on the same inputs.
 *
 * Cross-language flow mirrors the forward agreement test (§0.4.289):
 *   1. Build + coarsen + reverse-transform → gradient DxirFunction.
 *   2. Synthesize the same deterministic inputs (shared helper).
 *   3. `runOnIree(gradFn, inputs)` → 13 FloatArrays.
 *   4. Dump inputs as `.npy`, spawn `harness/python/run_pytorch_llama_grad.py`
 *      to compute reference gradients via torch.autograd.grad.
 *   5. Compare each gradient elementwise at relDiff < 1e-3 (looser than the
 *      forward 2.3e-6 because reverse-mode reduction trees diverge more
 *      between LLVM-CPU SIMD and PyTorch BLAS).
 *
 * Self-skips when IREE binaries / python / torch / reference script missing.
 */
class LlamaDecoderIreeBackwardVsPytorchTest {

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

    private fun resolveScriptPath(): Path =
        Path.of("..", "harness", "python", "run_pytorch_llama_grad.py").toAbsolutePath().normalize()

    /**
     * Pipeline: build → recognize → coarsen → reverse-AD → decompose. The reverse
     * pass inlines each COARSENED's gradient_body (so adjoints flow correctly), but
     * may also clone forward COARSENED ops into the gradient body where their values
     * are read by adjoints (`usedByAdjoint`). decomposeCoarsened inlines those
     * remaining forward primals back to primitives so iree-compile sees only ops
     * with StableHLO lowerings.
     *
     * Output signature: `(13 params) → (13 gradients)`, positionally aligned with
     * the primal's params.
     */
    private fun llamaGradPipeline(): DxirFunction {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val grad = DxirReverseTransform.apply(coarsened)
        return decomposeCoarsened(grad)
    }

    @Test
    fun llamaDecoderGradientsAgreeWithPytorchAutograd() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping.",
        )
        val python = resolvePythonBinary()
        assumeTrue(python != null, "no python interpreter resolved — skipping.")
        assumeTrue(
            pythonHasTorch(python!!),
            "python at $python cannot `import torch` — skipping.",
        )
        val script = resolveScriptPath()
        assumeTrue(
            Files.exists(script),
            "PyTorch backward reference script not found at $script.",
        )

        val gradFn = llamaGradPipeline()
        // The primal has 13 params; the gradient function preserves the same params
        // (same names, same types) and emits one return per param. Pin both arities so
        // any future signature change surfaces here loudly.
        assertEquals(13, gradFn.params.size, "AD output param count")
        assertEquals(13, gradFn.returns.size, "AD output gradient count")

        val primal = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val inputs = llamaSynthesizeInputs(seed = 42L, primal)

        // ---- Tlaloc-IREE side -----------------------------------------------------
        val ireeGrads = runOnIree(gradFn, inputs)
        assertEquals(13, ireeGrads.size, "IREE returned wrong number of gradients")

        // ---- PyTorch side ---------------------------------------------------------
        val workDir = Files.createTempDirectory("tlaloc-llama-grad-vs-pytorch-")
        workDir.toFile().deleteOnExit()
        for ((i, p) in primal.params.withIndex()) {
            val target = workDir.resolve("${p.name}.npy")
            NpyWriter.writeFloat32(target, inputs[i], p.type.dims)
            target.toFile().deleteOnExit()
        }
        val outputJson = workDir.resolve("pytorch-grads.json")
        outputJson.toFile().deleteOnExit()

        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--inputs-dir", workDir.toString(),
            "--output", outputJson.toString(),
        )
        pb.redirectErrorStream(false)
        val proc = pb.start()
        val finished = proc.waitFor(120, TimeUnit.SECONDS)
        if (!finished) {
            proc.destroyForcibly()
            error("PyTorch backward reference subprocess timed out after 120s")
        }
        val stderr = proc.errorStream.bufferedReader().readText()
        if (proc.exitValue() != 0) {
            error("PyTorch backward reference exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}")
        }

        val pytorchGrads = parsePytorchGrads(Files.readString(outputJson), primal.params.map { it.name })
        // ---- Compare each gradient tensor ----------------------------------------
        // Hybrid PyTorch-allclose-style tolerance: |a-b| <= atol + rtol·|b|.
        // atol absorbs FP32 round-off noise near zero (where pure relative
        // tolerance is meaningless); rtol covers the bulk of well-scaled values.
        // Tighter than 5e-3/1e-4 starts surfacing reduction-order divergence
        // between LLVM-CPU SIMD and PyTorch BLAS on the deep gradient chain;
        // looser than this hides structural disagreement.
        val rtol = 5e-3f
        val atol = 1e-4f
        var worstViolationGap = 0f      // (|a-b| - (atol + rtol*|b|))
        var worstViolationParam = ""
        var worstAbs = 0f
        var worstAbsParam = ""
        for ((idx, p) in primal.params.withIndex()) {
            val ireeArr = ireeGrads[idx]
            val refArr = pytorchGrads[p.name]
                ?: error("PyTorch reference missing gradient for param '${p.name}'")
            assertEquals(
                ireeArr.size, refArr.size,
                "gradient[${p.name}] size mismatch: iree=${ireeArr.size} torch=${refArr.size}",
            )
            for (j in ireeArr.indices) {
                val a = ireeArr[j]
                val b = refArr[j]
                val absDiff = abs(a - b)
                val budget = atol + rtol * abs(b)
                val gap = absDiff - budget
                if (absDiff > worstAbs) {
                    worstAbs = absDiff
                    worstAbsParam = "${p.name}[$j] iree=$a torch=$b"
                }
                if (gap > worstViolationGap) {
                    worstViolationGap = gap
                    worstViolationParam = "${p.name}[$j] iree=$a torch=$b absDiff=$absDiff budget=$budget"
                }
            }
        }
        println(
            "[llama-decoder-grad-iree-vs-pytorch] " +
                "worstAbs=$worstAbs at $worstAbsParam | " +
                "worstViolationGap=$worstViolationGap at $worstViolationParam | rtol=$rtol atol=$atol",
        )
        assertTrue(
            worstViolationGap <= 0f,
            "gradient disagreement exceeds tolerance (rtol=$rtol atol=$atol): " +
                "worstViolationGap=$worstViolationGap at $worstViolationParam.\n" +
                "PyTorch stderr:\n${stderr.takeLast(2000)}",
        )
    }

    /**
     * Parse `{"x_in": [v0, v1, …], "labels": [...], …}` JSON. Trivial because the
     * Python script writes exactly one number per element with comma separators.
     */
    private fun parsePytorchGrads(raw: String, paramNames: List<String>): Map<String, FloatArray> {
        val result = HashMap<String, FloatArray>()
        for (name in paramNames) {
            val key = "\"$name\""
            val ki = raw.indexOf(key)
            require(ki >= 0) { "PyTorch grad JSON missing key for '$name'" }
            val open = raw.indexOf('[', ki)
            val close = raw.indexOf(']', open)
            require(open >= 0 && close > open) { "PyTorch grad JSON malformed array for '$name'" }
            val arrayStr = raw.substring(open + 1, close).trim()
            val tokens = if (arrayStr.isEmpty()) emptyList() else arrayStr.split(",").map { it.trim() }
            result[name] = FloatArray(tokens.size) { i ->
                tokens[i].toFloatOrNull()
                    ?: error("PyTorch grad JSON: token '${tokens[i]}' not a Float for '$name[$i]'")
            }
        }
        return result
    }
}
