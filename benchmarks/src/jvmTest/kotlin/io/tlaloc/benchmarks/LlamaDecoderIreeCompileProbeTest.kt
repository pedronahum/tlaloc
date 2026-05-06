package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.IreeRuntime
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.286 — first-contact compile probe between the Tlaloc CPU baseline
 * pipeline and IREE. Builds the LlamaDecoderPrimal at tiny config, runs the
 * full lower chain (`build → recognize → coarsen → decomposeCoarsened →
 * toStablehlo`), and feeds the resulting MLIR through `iree-compile` (subprocess,
 * via [IreeRuntime.compile]). Asserts:
 *
 *   - the compile exits zero (no contracting-dim mismatches, no unsupported
 *     ops, no malformed attrs),
 *   - the produced VMFB exists on disk and is non-empty.
 *
 * This is **structural** evidence, not numerical: dispatching the VMFB with
 * realistic input tensors is §0.4.287's concern (needs FloatArray ↔ IREE
 * textual marshalling). The probe's value is binary — it pins "today's emit
 * pipeline produces valid IREE-CPU input for a llama-shaped model" so any
 * future emit regression that breaks `iree-compile` shows up here first.
 *
 * Self-skips when the IREE binaries are not on this host. The toolchain
 * lives at `~/.local/venvs/iree/bin/` per `pip install iree-base-compiler
 * iree-base-runtime`; see `iree_toolchain_dgx_spark.md` and the §0.4.284
 * commit message.
 */
class LlamaDecoderIreeCompileProbeTest {

    private fun cpuBaselineEmit(): String {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val decomposed = decomposeCoarsened(coarsened)
        return decomposed.toStablehlo("")
    }

    @Test
    fun llamaDecoderMlirCompilesViaIree() {
        assumeTrue(
            IreeBinaries.available,
            "iree-compile / iree-run-module not resolved — skipping. " +
                "Install via `pip install iree-base-compiler iree-base-runtime` into ~/.local/venvs/iree, " +
                "or set TLALOC_IREE_BIN to a directory containing the binaries.",
        )

        val mlir = cpuBaselineEmit()
        val module = IreeRuntime.compile(mlir)

        assertTrue(Files.exists(module.vmfbPath), "VMFB should exist on disk after compile")
        assertTrue(Files.size(module.vmfbPath) > 0, "VMFB should be non-empty")
    }
}
