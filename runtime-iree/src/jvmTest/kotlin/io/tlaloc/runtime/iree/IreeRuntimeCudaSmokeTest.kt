package io.tlaloc.runtime.iree

import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.290 — Phase 3b kickoff. Exercises [IreeTarget.Cuda] end-to-end:
 * hand-written StableHLO for `f(x) = x + 1` compiled via `iree-compile
 * --iree-hal-target-device=cuda` and dispatched via `iree-run-module
 * --device=cuda`. Asserts `2.0f → 3.0f` on the GPU codegen path.
 *
 * Self-skips when either the IREE binaries are not on host OR no CUDA device
 * is detected (`nvidia-smi -L` exits non-zero or empty). On dev hosts without
 * an NVIDIA GPU, the test stays green; on the GB10 Blackwell box, it runs
 * live and pins that sm_100 codegen works for the trivial elementwise op.
 */
class IreeRuntimeCudaSmokeTest {

    private fun requireIreeAndCudaOrSkip() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping CUDA smoke. " +
                "Test is green by design on hosts without an NVIDIA GPU.",
        )
    }

    @Test
    fun `f(x) = x + 1 round-trips through cuda target`() {
        requireIreeAndCudaOrSkip()
        val mlir = """
            module {
              func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                %c1 = stablehlo.constant dense<1.0> : tensor<f32>
                %0 = stablehlo.add %arg0, %c1 : tensor<f32>
                return %0 : tensor<f32>
              }
            }
        """.trimIndent()

        val module = IreeRuntime.compile(mlir, target = IreeTarget.Cuda)
        assertEquals(IreeTarget.Cuda, module.target, "compiled module must remember its target")
        val outputs = IreeRuntime.invoke(module, function = "main", inputs = listOf("f32=2.0"))

        assertEquals(listOf("f32=3"), outputs)
    }
}
