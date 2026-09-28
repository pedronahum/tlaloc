package io.tlaloc.runtime.pjrt

import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import io.tlaloc.stablehlo.mosaic.MosaicRmsNorm
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Tlaloc's `tpu_custom_call` programs, checked by jaxlib on the CPU host:
 * `harness/python/check_tpu_custom_call.py` parses each one with the
 * StableHLO and TPU dialects, decodes the JSON backend_config and the Mosaic
 * body, converts the program to HLO, and compares the HLO custom-call with
 * the one `jax.export` produces for the same Pallas kernel. For the
 * Kotlin-emitted payloads it also re-serializes the emitter's text and
 * compares bytes. The same program lowered to its reference decomposition is
 * compiled and run on XLA's CPU client against the numpy outputs.
 *
 * Skips without a python that imports jax with the TPU dialect
 * (`TLALOC_JAX_PYTHON`, else `~/.local/venvs/iree`).
 */
class TpuCustomCallParseCheckTest {

    @Test
    fun jaxlibAcceptsEveryPayloadProgram() {
        val python = TpuKernelFixtures.jaxPython()
        assumeTrue(python != null, "no python with jax + jaxlib's TPU dialect — skipping")
        val script = TpuKernelFixtures.harnessScript("check_tpu_custom_call.py")
        assumeTrue(Files.exists(script), "missing $script — skipping")

        val work = Files.createTempDirectory("tlaloc-tpu-cc-")
        try {
            for (name in TpuKernelFixtures.NAMES) {
                val f = TpuKernelFixtures.load(name)
                val mlir = lowerMosaicKernels(f.program(referenceFallback = false), KernelTarget.GOOGLE_TPU_V5E)
                    .toStablehlo("")
                val mlirFile = work.resolve("$name.mlir").also { Files.writeString(it, mlir) }
                val reference = lowerMosaicKernels(f.program(referenceFallback = true), KernelTarget.CPU_GENERIC)
                    .toStablehlo("")
                val referenceFile = work.resolve("$name.reference.mlir").also { Files.writeString(it, reference) }
                val manifest = Path.of(
                    TpuKernelFixtures::class.java.getResource("/tpu-kernels/$name/manifest.json")!!.toURI(),
                )
                val out = work.resolve("$name.json")
                val cmd = mutableListOf(
                    python!!, script.toString(),
                    "--mlir", mlirFile.toString(),
                    "--manifest", manifest.toString(),
                    "--output", out.toString(),
                    "--reference-mlir", referenceFile.toString(),
                )
                if (f.source == "kmosaic") {
                    val dtype = if ("bf16" in name) MosaicRmsNorm.Dtype.BF16 else MosaicRmsNorm.Dtype.F32
                    val dims = f.inputs[0].shape
                    val text = MosaicRmsNorm.emit(rows = dims[0], hidden = dims[1], dtype = dtype, eps = f.eps)
                    cmd += listOf("--mosaic-text", work.resolve("$name.mosaic.mlir").also { Files.writeString(it, text) }.toString())
                }
                val pb = ProcessBuilder(cmd).redirectErrorStream(true)
                pb.environment()["JAX_PLATFORMS"] = "cpu"
                val proc = pb.start()
                val log = proc.inputStream.bufferedReader().readText()
                check(proc.waitFor(300, TimeUnit.SECONDS)) { "check_tpu_custom_call timed out for $name" }
                val result = Files.readString(out)
                println("[tpu-cc-check] $name exit=${proc.exitValue()}\n$result")
                assumeTrue(proc.exitValue() != 2, "python environment cannot run the check: $result")
                assertTrue(proc.exitValue() == 0 && "\"ok\": true" in result, "$name failed the jaxlib check:\n$result\n$log")
            }
        } finally {
            work.toFile().deleteRecursively()
        }
    }
}
