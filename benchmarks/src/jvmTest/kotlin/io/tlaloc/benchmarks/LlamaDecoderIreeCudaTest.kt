package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.IreeTarget
import io.tlaloc.runtime.iree.runOnIree
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.291 — LlamaDecoder forward dispatched on IREE-CUDA. Two assertions:
 *
 *   1. CUDA produces a finite, deterministic scalar loss (no NaN, no nondeterministic
 *      atomic reductions in the lowered kernels).
 *   2. CUDA's loss agrees with the CPU baseline's loss within 1e-3 relative
 *      tolerance — looser than §0.4.289's CPU↔PyTorch tolerance to absorb the
 *      reduction-order differences between LLVM-CPU SIMD codegen and CUDA's
 *      warp-shuffle reductions, but tight enough to catch any structural
 *      divergence between the two backends.
 *
 * Self-skips when IREE binaries are not on host OR no CUDA device is detected
 * (`nvidia-smi -L` empty/non-zero).
 */
class LlamaDecoderIreeCudaTest {

    private fun requireIreeAndCudaOrSkip() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping CUDA agreement test.",
        )
    }

    @Test
    fun llamaDecoderForwardOnCudaProducesFiniteDeterministicLoss() {
        requireIreeAndCudaOrSkip()
        val fn = llamaCpuBaselinePipeline()
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)

        val outputs1 = runOnIree(fn, inputs, IreeTarget.Cuda)
        val outputs2 = runOnIree(fn, inputs, IreeTarget.Cuda)
        assertEquals(1, outputs1.size, "single scalar return")
        val loss1 = outputs1.single().single()
        val loss2 = outputs2.single().single()
        assertTrue(loss1.isFinite(), "CUDA loss must be finite; got $loss1")
        assertEquals(
            loss1, loss2,
            "CUDA forward should be deterministic across calls (no atomic-add reductions in the " +
                "decomposed primitives we emit). Drift here points at nondeterministic codegen.",
        )
    }

    @Test
    fun llamaDecoderForwardOnCudaAgreesWithCpu() {
        requireIreeAndCudaOrSkip()
        val fn = llamaCpuBaselinePipeline()
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)

        val lossCpu = runOnIree(fn, inputs, IreeTarget.LlvmCpu).single().single()
        val lossCuda = runOnIree(fn, inputs, IreeTarget.Cuda).single().single()

        val absDiff = abs(lossCpu - lossCuda)
        val relDiff = absDiff / max(1.0f, max(abs(lossCpu), abs(lossCuda)))
        val tol = 1e-3f
        println(
            "[llama-decoder-cpu-vs-cuda] cpu=$lossCpu cuda=$lossCuda " +
                "absDiff=$absDiff relDiff=$relDiff (tol=$tol)",
        )
        assertTrue(
            relDiff < tol,
            "CPU vs CUDA loss disagreement: cpu=$lossCpu cuda=$lossCuda " +
                "absDiff=$absDiff relDiff=$relDiff (tol=$tol)",
        )
    }
}
