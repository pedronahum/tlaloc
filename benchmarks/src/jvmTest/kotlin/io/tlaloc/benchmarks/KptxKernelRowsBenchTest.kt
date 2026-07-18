package io.tlaloc.benchmarks

import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.emitPtx
import io.tlaloc.runtime.cuda.CudaDriverFfm
import io.tlaloc.runtime.cuda.CudaDriverException
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.356 — kernel-level comparison rows: the KPTX rms_norm against
 * JAX/XLA-fused and **JAX/Pallas** implementations of the identical
 * workload (256×512 f32, eps 1e-5, the §0.4.336-claimed unweighted
 * form), on the same GB10. Methodology cribbed from pyptx's benchmark
 * suite: warmup + per-iteration sync, medians over a fixed budget.
 *
 * The KPTX row is a direct `cuLaunchKernel` loop (no XLA, no FFI round
 * trip — this is the kernel itself); the Python rows come from
 * `harness/python/run_kernel_rmsnorm_bench.py` (Pallas lowered via the
 * Triton-IR path — `JAX_PALLAS_USE_MOSAIC_GPU=0`, this plugin wheel
 * ships no Mosaic dialect; the Triton row self-skips until a
 * CUDA-enabled torch lands in the venv, and prints its reason).
 *
 * Correctness pin: the KPTX kernel's output matches the same host
 * reference formula the Python rows validate against, so all rows are
 * measured on verified-equal math.
 */
class KptxKernelRowsBenchTest {

    private val rows = 256
    private val cols = 512
    private val eps = 1e-5f
    private val iters = 200

    private fun venvPython(): String? {
        val venv = Path.of(System.getProperty("user.home"), ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(venv)) venv.toString() else null
    }

    private fun cudaAvailable(): Boolean = runCatching {
        val p = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start()
        if (!p.waitFor(5, TimeUnit.SECONDS)) {
            p.destroyForcibly(); false
        } else {
            p.exitValue() == 0 && p.inputStream.bufferedReader().readText().isNotBlank()
        }
    }.getOrElse { false }

    @Test
    fun kernelRowsRmsNorm() {
        assumeTrue(cudaAvailable(), "no NVIDIA GPU/driver — skipping.")

        // ---- KPTX row: direct driver launches on the DSL-emitted kernel ----
        val ptx = KptxKernels.rmsNormEps.specialize(shapes = mapOf("block" to 256)).emitPtx()
        val rng = java.util.Random(42)
        val x = FloatArray(rows * cols) { (rng.nextGaussian() * 0.5).toFloat() }
        val epsArr = FloatArray(rows) { eps }

        var kptxMedianUs = -1L
        var maxAbs = 0f
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
                val module = cuda.moduleLoadPtx(ptx)
                try {
                    val function = cuda.moduleGetFunction(module, "kptx_rms_norm")
                    val xB = rows * cols * 4L
                    val host = arena.allocate(xB)
                    for (i in x.indices) host.set(JAVA_FLOAT, i * 4L, x[i])
                    val hostEps = arena.allocate(rows * 4L)
                    for (i in 0 until rows) hostEps.set(JAVA_FLOAT, i * 4L, epsArr[i])

                    val dX = cuda.memAlloc(xB)
                    val dEps = cuda.memAlloc(rows * 4L)
                    val dOut = cuda.memAlloc(xB)
                    try {
                        cuda.memcpyHtoD(dX, host, xB)
                        cuda.memcpyHtoD(dEps, hostEps, rows * 4L)
                        val pX = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dX) }
                        val pE = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dEps) }
                        val pO = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dOut) }
                        val pN = arena.allocate(JAVA_INT).also { it.set(JAVA_INT, 0L, cols) }
                        val params = arena.allocate(4 * 8L)
                        params.set(ADDRESS, 0L, pX); params.set(ADDRESS, 8L, pE)
                        params.set(ADDRESS, 16L, pO); params.set(ADDRESS, 24L, pN)

                        fun launch() {
                            cuda.launchKernel(
                                function, rows, 1, 1, 256, 1, 1,
                                sharedMemBytes = 0, stream = MemorySegment.NULL, kernelParams = params,
                            )
                        }
                        repeat(20) { launch() }
                        cuda.ctxSynchronize()
                        val times = LongArray(iters)
                        for (i in 0 until iters) {
                            val t0 = System.nanoTime()
                            launch()
                            cuda.ctxSynchronize()
                            times[i] = System.nanoTime() - t0
                        }
                        times.sort()
                        kptxMedianUs = times[iters / 2] / 1_000

                        val out = arena.allocate(xB)
                        cuda.memcpyDtoH(out, dOut, xB)
                        // Same host reference as the Python rows.
                        for (r in 0 until rows) {
                            var sumSq = 0.0
                            for (c in 0 until cols) {
                                val v = x[r * cols + c]; sumSq += (v * v).toDouble()
                            }
                            val inv = (1.0 / Math.sqrt(sumSq / cols + eps)).toFloat()
                            for (c in 0 until cols) {
                                val want = x[r * cols + c] * inv
                                val got = out.get(JAVA_FLOAT, (r * cols + c) * 4L)
                                maxAbs = maxOf(maxAbs, abs(got - want))
                            }
                        }
                    } finally {
                        cuda.memFree(dX); cuda.memFree(dEps); cuda.memFree(dOut)
                    }
                } finally {
                    cuda.moduleUnload(module)
                }
            } finally {
                cuda.primaryCtxRelease(device)
            }
        }
        assertTrue(maxAbs <= 1e-4f, "kptx kernel diverges from reference: max|diff|=$maxAbs")

        // ---- Python rows (jax-xla, pallas, triton-if-available) ----
        val python = venvPython()
        var pythonRows = "(python venv unavailable — Python rows skipped)"
        if (python != null) {
            val outJson = Files.createTempFile("rmsnorm-rows", ".json")
            val script = Path.of("..", "harness", "python", "run_kernel_rmsnorm_bench.py")
                .toAbsolutePath().normalize()
            val proc = ProcessBuilder(python, script.toString(), "--output", outJson.toString())
                .redirectErrorStream(true).start()
            val finished = proc.waitFor(300, TimeUnit.SECONDS)
            pythonRows = if (finished && proc.exitValue() == 0) {
                Files.readString(outJson).trim()
            } else {
                "(python bench failed: ${proc.inputStream.bufferedReader().readText().takeLast(300)})"
            }
            Files.deleteIfExists(outJson)
        }

        println(
            "[kptx-kernel-rows] rms_norm 256x512 f32 on GB10 | " +
                "kptx(direct cuLaunchKernel) median=${kptxMedianUs} us, max|diff|=$maxAbs | python rows:\n$pythonRows",
        )
        java.io.File("build").mkdirs()
        java.io.File("build/harness-results-kernel-rmsnorm.json").writeText(
            "{\n  \"kptx_median_us\": $kptxMedianUs,\n  \"python_rows\": $pythonRows\n}\n",
        )
        assertTrue(kptxMedianUs > 0)
    }
}
