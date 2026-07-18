package io.tlaloc.benchmarks

import io.tlaloc.kptx.KptxTranspiler
import io.tlaloc.kptx.emitPtx
import io.tlaloc.kptx.normalizePtx
import io.tlaloc.kptx.parsePtx
import io.tlaloc.kptx.validateIsaErrors
import io.tlaloc.runtime.cuda.CudaDriverFfm
import io.tlaloc.runtime.cuda.CudaDriverException
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.357 — the GQA bootstrap's first concrete step: **real pyptx
 * kernels ingest through the whole KPTX pipeline.** Three Apache-2.0
 * fixtures generated from pyptx 0.1.1 example builders (rms_norm with
 * v4 loads + shfl-butterfly reduction; softmax with the ex2 fold; a
 * bf16 mma.sync GEMM tile — see resources/pyptx/README.md) each:
 *
 *   normalize → ISA-clean → canonical byte-stable → transpile
 *   (self-verified byte-identical replay)
 *
 * and the normalized rms_norm additionally driver-JIT compiles on the
 * GB10 — proving the §0.4.352 normalizer's declaration canonicalization
 * (single-reg coalescing, custom array banks, mixed storage types)
 * produces driver-valid PTX. These kernels are the attention building
 * blocks (softmax, GEMM tile) the GQA assembly will edit and compose.
 */
class PyptxIngestionTest {

    private fun fixture(name: String): String =
        checkNotNull(javaClass.getResourceAsStream("/pyptx/$name")) { "missing fixture $name" }
            .bufferedReader().readText()

    private val fixtures = listOf("pyptx_rms_norm.ptx", "pyptx_softmax.ptx", "pyptx_gemm.ptx")

    @Test
    fun everyPyptxFixtureIngestsNormalizesAndTranspiles() {
        for (name in fixtures) {
            val module = normalizePtx(fixture(name))
            val isaErrors = module.validateIsaErrors()
            assertEquals(emptyList(), isaErrors, "$name must be ISA-clean after normalization")
            val canonical = module.emitPtx()
            assertEquals(canonical, parsePtx(canonical).emitPtx(), "$name canonical form must be byte-stable")
            val source = KptxTranspiler.transpile(module)
            assertTrue(source.contains("fun "), "$name transpile produced no source")
            println(
                "[pyptx-ingest] $name: ${module.kernels.single().body.size} stmts → " +
                    "${canonical.lines().size} canonical lines → ${source.lines().size} Kotlin lines (self-verified)",
            )
        }
    }

    @Test
    fun normalizedRmsNormDriverJitCompiles() {
        assumeTrue(cudaAvailable(), "no NVIDIA GPU/driver — skipping.")
        val canonical = normalizePtx(fixture("pyptx_rms_norm.ptx")).emitPtx()
        Arena.ofShared().use { arena ->
            val cuda = try {
                CudaDriverFfm.load(arena)
            } catch (e: CudaDriverException) {
                assumeTrue(false, "CUDA driver unusable (${e.message}) — skipping.")
                return
            }
            val device = cuda.deviceGet(0)
            cuda.primaryCtxRetainAndSetCurrent(device)
            try {
                val module = cuda.moduleLoadPtx(canonical)
                try {
                    cuda.moduleGetFunction(module, "rms_norm")
                    println("[pyptx-ingest] normalized pyptx rms_norm: driver-JIT accepts the canonicalized module")
                } finally {
                    cuda.moduleUnload(module)
                }
            } finally {
                cuda.primaryCtxRelease(device)
            }
        }
    }

    private fun cudaAvailable(): Boolean = runCatching {
        val p = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start()
        if (!p.waitFor(5, TimeUnit.SECONDS)) {
            p.destroyForcibly(); false
        } else {
            p.exitValue() == 0 && p.inputStream.bufferedReader().readText().isNotBlank()
        }
    }.getOrElse { false }
}
