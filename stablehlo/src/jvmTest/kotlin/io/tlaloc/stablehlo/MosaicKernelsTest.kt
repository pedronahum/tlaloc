package io.tlaloc.stablehlo

import io.tlaloc.stablehlo.mosaic.MosaicKernels
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The [MosaicKernels], written in both forms to `build/mosaic-kernels/`, the shapes the twins in
 * `harness/python/check_mosaic_kernels.py` use. When a Python with JAX's Pallas is found
 * (`TLALOC_JAX_PYTHON`, else `~/.local/venvs/iree/bin/python`), the script runs: each module must
 * equal its Pallas twin's after `cse`, its serialized form mosaic-serde's, and each twin must
 * match numpy in TPU interpret mode.
 */
class MosaicKernelsTest {

    private val kernels: Map<String, (Boolean) -> String> = mapOf(
        "softmax_rows" to { s -> MosaicKernels.softmaxRows(64, 256, 16, s) },
        "rmsnorm_rows" to { s -> MosaicKernels.rmsNormRows(64, 512, 16, 1e-6f, s) },
        "flash_fwd" to { s -> MosaicKernels.flashForward(512, s) },
        "flash_bwd_dkdv" to { s -> MosaicKernels.flashBackwardDkDv(512, s) },
        "flash_bwd_dq" to { s -> MosaicKernels.flashBackwardDq(512, s) },
    )

    private fun python(): String? {
        System.getenv("TLALOC_JAX_PYTHON")?.let { return it }
        val venv = Path.of(System.getProperty("user.home"), ".local/venvs/iree/bin/python")
        return venv.toString().takeIf { Files.isExecutable(venv) }
    }

    @Test
    fun theKernelsEqualTheirPallasTwins() {
        val dir = Path.of("build", "mosaic-kernels").toAbsolutePath()
        Files.createDirectories(dir)
        for ((name, emit) in kernels) {
            val plain = emit(false)
            val serialized = emit(true)
            assertTrue(plain.startsWith("module @") && "stable_mosaic." in serialized, name)
            Files.writeString(dir.resolve("$name.mlir"), plain)
            Files.writeString(dir.resolve("$name.serialized.mlir"), serialized)
        }
        val py = python() ?: return println("MosaicKernelsTest: no Python with JAX found; wrote $dir only")
        val script = Path.of("..", "harness", "python", "check_mosaic_kernels.py").toAbsolutePath().normalize()
        val pb = ProcessBuilder(py, "-I", script.toString(), dir.toString()).redirectErrorStream(true)
        pb.environment()["JAX_PLATFORMS"] = "cpu"
        pb.environment()["PYTHONPATH"] = script.parent.toString()
        val proc = pb.start()
        val log = proc.inputStream.readBytes().decodeToString()
        assertTrue(proc.waitFor(1800, TimeUnit.SECONDS), "check_mosaic_kernels.py timed out\n$log")
        val report = log.lines().filter { it.startsWith("ok ") || it.startsWith("FAIL") || it.startsWith("-") || it.startsWith("+") }
        println(report.joinToString("\n"))
        assertEquals(0, proc.exitValue(), "check_mosaic_kernels.py failed:\n$log")
    }
}
