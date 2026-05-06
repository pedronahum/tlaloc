package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.runOnIree
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.288 — first end-to-end native-CPU dispatch of the LlamaDecoderPrimal.
 * Builds the model at tiny config, runs the full CPU baseline pipeline
 * (`build → recognize → coarsen → decomposeCoarsened → toStablehlo`), generates
 * deterministic inputs, dispatches via [runOnIree] (which compiles via
 * `iree-compile` and invokes via `iree-run-module`), and pins:
 *
 *   - the forward produces a finite, non-NaN scalar loss,
 *   - the same inputs produce identical losses across two consecutive calls
 *     (determinism — no hidden state, no nondeterministic codegen tail).
 *
 * # Why this is "self-consistency", not "numerical correctness"
 *
 * The JVM-side DxirInterpreter cannot evaluate this primal — it deliberately
 * doesn't cover MEAN/RSQRT/SOFTMAX/SILU (the architectural boundary noted in
 * the dual-track plan). So this test cannot assert agreement with a JVM
 * reference. It pins **structural reachability**: the entire emit + dispatch
 * pipeline survives a llama-shaped workload end-to-end. Numerical-correctness
 * vs PyTorch/JAX is §0.4.289's concern (it'll need a Python harness reference
 * with matched weight initialization).
 *
 * Self-skips when the IREE binaries are not on host.
 */
class LlamaDecoderIreeRunSmokeTest {

    private fun requireIreeOrSkip() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree, " +
                "or set TLALOC_IREE_BIN to a directory containing the binaries.",
        )
    }

    @Test
    fun llamaDecoderForwardProducesFiniteScalarLoss() {
        requireIreeOrSkip()
        val fn = llamaCpuBaselinePipeline()
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)

        val outputs = runOnIree(fn, inputs)

        assertEquals(1, outputs.size, "LlamaDecoder primal returns single scalar loss")
        val loss = outputs.single()
        assertEquals(1, loss.size, "loss is a rank-0 scalar (one f32)")
        val v = loss[0]
        assertTrue(v.isFinite(), "loss must be finite (not NaN, not ±Inf); got $v")
    }

    @Test
    fun llamaDecoderForwardIsDeterministicAcrossCalls() {
        requireIreeOrSkip()
        val fn = llamaCpuBaselinePipeline()
        val inputs1 = llamaSynthesizeInputs(seed = 7L, fn)
        val inputs2 = llamaSynthesizeInputs(seed = 7L, fn)
        // Sanity: identical inputs from the same seed.
        for (i in inputs1.indices) {
            assertTrue(
                inputs1[i].contentEquals(inputs2[i]),
                "synthesizeInputs must be deterministic by seed (param ${fn.params[i].name})",
            )
        }

        val out1 = runOnIree(fn, inputs1).single().single()
        val out2 = runOnIree(fn, inputs2).single().single()

        assertEquals(
            out1, out2,
            "same inputs must produce bit-identical loss across calls — " +
                "any drift here points at nondeterministic codegen or hidden state in the bridge",
        )
    }
}
