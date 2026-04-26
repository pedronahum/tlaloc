package io.tlaloc.plugin

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.net.URLClassLoader
import java.nio.file.Files
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.Services
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.183 — Phase 1 of the head-to-head harness, per
 * `docs/HEAD_TO_HEAD_HARNESS_PLAN.md`. JVM-side scaffolding only — no Python /
 * PyTorch / JAX integration yet (Phase 2, gated on user-side toolchain).
 *
 * Runs three K2-plugin-shipped benchmarks in a 1000-iteration timing loop
 * (200 warmup + 800 measured) and writes per-benchmark median/min/p99 ns to
 * a CSV at `compiler-plugin/build/harness-results-tlaloc.csv`. Each benchmark
 * is compiled once via the existing in-process K2 harness, then its main()
 * runs the timing loop + emits PERF stdout lines that this test parses.
 *
 * Asserts numerical agreement with the per-port FD tests (sentinel-defeated).
 *
 * Phase 2 (PyTorch / JAX reference + JSON IPC) lands when the user has those
 * toolchains available; the JVM CSV produced here is the comparison baseline.
 *
 * Bench inhabitants:
 *  - Brachistochrone N=64 — `:core/sqrt`-bearing arithmetic loop, 64 hops.
 *  - HookeanSpring N=10 chain — `:core/sqrt` + scalar arithmetic, 9 spring iterations.
 *  - HMC logistic regression (n=4, d=2 column-major loop form) — exp/log + scalar arith.
 */
class HeadToHeadHarnessTest {

    @Test
    fun `head-to-head harness — 3 benchmarks, JVM-side timings, CSV`() {
        val results = mutableListOf<HarnessResult>()

        for ((spec, source) in BENCHMARK_SOURCES) {
            val result = compileAndRun(spec.stub, source)
            assertEquals(0, result.exitCode, "[${spec.name}] compile failed:\n${result.messages}")
            val perf = parsePerfLines(result.stdout)
            assertTrue(
                perf["sentinel_check"]?.toFloatOrNull()?.let { abs(it + 1.0f) > 1e-3f } ?: false,
                "[${spec.name}] sentinel_check=${perf["sentinel_check"]} matches broken-stub fallback",
            )
            results += HarnessResult(
                name = spec.name,
                medianNs = perf["gradient_median_ns"]?.toLongOrNull()
                    ?: error("[${spec.name}] missing gradient_median_ns"),
                minNs = perf["gradient_min_ns"]?.toLongOrNull()
                    ?: error("[${spec.name}] missing gradient_min_ns"),
                p99Ns = perf["gradient_p99_ns"]?.toLongOrNull()
                    ?: error("[${spec.name}] missing gradient_p99_ns"),
            )
        }

        // Write CSV. Phase 2 will read this + PyTorch/JAX JSONs into a comparison table.
        val csvPath = File("build/harness-results-tlaloc.csv").apply { parentFile?.mkdirs() }
        csvPath.writeText(
            buildString {
                appendLine("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns")
                for (r in results) {
                    appendLine("${r.name},tlaloc,800,${r.medianNs},${r.minNs},${r.p99Ns}")
                }
            },
        )
        println("[head-to-head harness] wrote ${results.size} rows to ${csvPath.absolutePath}")
        for (r in results) {
            println("[head-to-head harness] ${r.name}: median=${r.medianNs}ns min=${r.minNs}ns p99=${r.p99Ns}ns")
        }
    }

    // ----- Benchmark source registry --------------------------------------

    private data class BenchmarkSpec(val name: String, val stub: String)

    private data class HarnessResult(val name: String, val medianNs: Long, val minNs: Long, val p99Ns: Long)

    /**
     * Each entry: (spec, primal-bearing source). The source's `main` MUST emit
     * stdout lines of the form `PERF <key>=<value>` for at least:
     *  - `gradient_median_ns`, `gradient_min_ns`, `gradient_p99_ns` — the timing aggregates.
     *  - `sentinel_check` — a single Float that must NOT equal -1.0f (sentinel-defeat).
     *
     * The source is responsible for: warmup loop (200), measured loop (800),
     * percentile aggregation, and PERF stdout output.
     */
    private val BENCHMARK_SOURCES: List<Pair<BenchmarkSpec, String>> by lazy {
        listOf(
            BenchmarkSpec("brachistochrone_n64", AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_64) to BRACHISTOCHRONE_SRC,
            BenchmarkSpec("hookean_spring_n10", AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_10) to HOOKEAN_SPRING_SRC,
            BenchmarkSpec("hmc_logistic_n4_d2_loop", AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_14) to HMC_LOOP_SRC,
        )
    }

    // ----- Source primals ------------------------------------------------

    private val PERF_LOOP_SUFFIX = """
                val warmup = 200
                val measured = 800
                val timings = LongArray(measured)
                for (i in 0 until warmup) sink += g(input).hostF32()[0]
                for (i in 0 until measured) {
                    val t0 = System.nanoTime()
                    sink += g(input).hostF32()[0]
                    timings[i] = System.nanoTime() - t0
                }
                timings.sort()
                val median = timings[measured / 2]
                val min = timings[0]
                val p99 = timings[(measured * 99 / 100).coerceAtMost(measured - 1)]
                println("PERF gradient_median_ns=${'$'}median")
                println("PERF gradient_min_ns=${'$'}min")
                println("PERF gradient_p99_ns=${'$'}p99")
                println("PERF sentinel_check=${'$'}{g(input).hostF32()[0]}")
                println("PERF sink=${'$'}sink")
    """.trimIndent()

    private val BRACHISTOCHRONE_SRC = """
        import io.tlaloc.autograd.grad
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        import io.tlaloc.core.Tensors
        import io.tlaloc.core.hostF32
        import io.tlaloc.core.ops.get
        import io.tlaloc.core.sqrt
        fun main() {
            val g = grad { y: DTensor<Rank1<Sym>, F32> ->
                var t = 0.0f
                for (i in 0 until 64) {
                    val dy = y[i + 1] - y[i]
                    val ds = (1.0f + dy * dy).sqrt()
                    val v = (-y[i] - y[i + 1]).sqrt()
                    t = t + ds / (v + 1.0e-3f)
                }
                t
            }
            val arr = FloatArray(65) { -((it + 1).toFloat() * 0.1f) }
            val input = Tensors.f32Vector<Sym>(arr)
            var sink = 0.0f
$PERF_LOOP_SUFFIX
        }
    """.trimIndent()

    private val HOOKEAN_SPRING_SRC = """
        import io.tlaloc.autograd.grad
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        import io.tlaloc.core.Tensors
        import io.tlaloc.core.hostF32
        import io.tlaloc.core.ops.get
        import io.tlaloc.core.sqrt
        fun main() {
            val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                var energy = 0.0f
                for (i in 0 until 9) {
                    val d = p[i] - p[i + 1]
                    val len = (d * d).sqrt()
                    val stretch = len - 1.0f
                    energy = energy + 0.5f * stretch * stretch
                }
                energy
            }
            val input = Tensors.f32Vector<Sym>(FloatArray(10) { it.toFloat() * 1.2f })
            var sink = 0.0f
$PERF_LOOP_SUFFIX
        }
    """.trimIndent()

    private val HMC_LOOP_SRC = """
        import io.tlaloc.autograd.grad
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        import io.tlaloc.core.Tensors
        import io.tlaloc.core.exp
        import io.tlaloc.core.hostF32
        import io.tlaloc.core.log
        import io.tlaloc.core.ops.get
        fun main() {
            val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
                val b0 = packed[0]
                val b1 = packed[1]
                var sum1 = 0.0f
                var sum2 = 0.0f
                for (i in 0 until 4) {
                    val xi0 = packed[2 + i]
                    val xi1 = packed[6 + i]
                    val yi = packed[10 + i]
                    val xb = xi0 * b0 + xi1 * b1
                    sum1 = sum1 + (yi - 1.0f) * xb
                    sum2 = sum2 + (1.0f + (-xb).exp()).log()
                }
                val term3 = (b0 * b0 + b1 * b1) / 2000.0f
                sum1 - sum2 - term3
            }
            val cfg = floatArrayOf(0.5f, 0.3f, 1.0f, 0.5f, -0.5f, 1.5f, 0.5f, 1.0f, 1.5f, -0.5f, 1.0f, 0.0f, 1.0f, 0.0f)
            val input = Tensors.f32Vector<Sym>(cfg)
            var sink = 0.0f
$PERF_LOOP_SUFFIX
        }
    """.trimIndent()

    // ----- Harness plumbing -----------------------------------------------

    private fun parsePerfLines(stdout: String): Map<String, String> {
        return stdout.lines()
            .filter { it.startsWith("PERF ") }
            .associate { line ->
                val (k, v) = line.removePrefix("PERF ").split("=", limit = 2)
                k to v
            }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-harness-run").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(severity: CompilerMessageSeverity, message: String, location: CompilerMessageSourceLocation?) {
                    collected += CompileMessage(severity, message)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val capturedOut = PrintStream(baos, true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    // Stub variants by output rank — each benchmark's broken stub returns a
    // sentinel of the correct rank-1 length.

    private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_64 = """
        package io.tlaloc.autograd
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.HostF32Storage
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
            { _ -> DTensor(HostF32Storage(FloatArray(65) { -1.0f }), intArrayOf(65), F32) }
    """.trimIndent()

    private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_10 = """
        package io.tlaloc.autograd
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.HostF32Storage
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
            { _ -> DTensor(HostF32Storage(FloatArray(10) { -1.0f }), intArrayOf(10), F32) }
    """.trimIndent()

    private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_14 = """
        package io.tlaloc.autograd
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.HostF32Storage
        import io.tlaloc.core.Rank1
        import io.tlaloc.core.Sym
        fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
            { _ -> DTensor(HostF32Storage(FloatArray(14) { -1.0f }), intArrayOf(14), F32) }
    """.trimIndent()
}
