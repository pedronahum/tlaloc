package io.tlaloc.runtime.iree

import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Phase 1 first-slice from `docs/IREE_CPU_PORT_PLAN.md`: hand-written StableHLO for
 * `f(x) = x + 1`, compiled via `iree-compile`, dispatched via `iree-run-module`,
 * asserts `2.0f → 3.0f`. Self-skips if the IREE binaries are not resolvable.
 */
class IreeRuntimeSmokeTest {

    private fun requireIreeOrSkip() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree, " +
                "or set TLALOC_IREE_BIN to a directory containing the binaries.",
        )
    }

    @Test
    fun `f(x) = x + 1 round-trips through iree-compile + iree-run-module`() {
        requireIreeOrSkip()
        val mlir = """
            module {
              func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                %c1 = stablehlo.constant dense<1.0> : tensor<f32>
                %0 = stablehlo.add %arg0, %c1 : tensor<f32>
                return %0 : tensor<f32>
              }
            }
        """.trimIndent()

        val module = IreeRuntime.compile(mlir)
        val outputs = IreeRuntime.invoke(module, function = "main", inputs = listOf("f32=2.0"))

        assertEquals(listOf("f32=3"), outputs)
    }

    @Test
    fun `vector add round-trips through iree-compile + iree-run-module`() {
        requireIreeOrSkip()
        val mlir = """
            module {
              func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.add %a, %b : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        """.trimIndent()

        val module = IreeRuntime.compile(mlir)
        val outputs = IreeRuntime.invoke(
            module,
            function = "main",
            inputs = listOf("4xf32=1.0,2.0,3.0,4.0", "4xf32=10.0,20.0,30.0,40.0"),
        )

        assertEquals(1, outputs.size)
        assertEquals("4xf32=11 22 33 44", outputs.single())
    }

    @Test
    fun `compile failure surfaces as IreeCompileException with stderr`() {
        requireIreeOrSkip()
        val brokenMlir = "module { func.func @main() { not_a_real_op } }"

        val ex = runCatching { IreeRuntime.compile(brokenMlir) }.exceptionOrNull()
        assert(ex is IreeCompileException) { "expected IreeCompileException, got $ex" }
        ex as IreeCompileException
        assert(!ex.process.ok)
        assert(ex.process.stderr.isNotEmpty()) { "expected non-empty stderr from iree-compile failure" }
    }
}
