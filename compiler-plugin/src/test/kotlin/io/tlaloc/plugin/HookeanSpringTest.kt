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
 * Stage D.2 — HookeanSpring, the second OOPSLA 2021 benchmark port.
 *
 * The paper's HookeanSpring is a physical simulation: an N-vertex mass-spring system
 * where the primal computes total elastic energy and the gradient wrt vertex positions
 * drives the optimiser that settles the system to a minimum-energy configuration.
 * Paper §6.2 reports 4-11× end-to-end speedup, the highest of any benchmark — driven
 * by the SOI being "the entire energy calculation" so coarsening can REMOVE the primal
 * from the generated gradient code entirely (a benefit Brachistochrone doesn't get
 * because its SqrtRule + DivRule + MulRule all read primal values).
 *
 * Our port targets a 3-vertex triangle in 1D with 3 springs (0-1, 1-2, 0-2) of rest
 * lengths 1, 1, 2. 1D keeps the primal all-scalar post-gather — no rank-1 elementwise
 * operations needed. Still exercises the full path: rank-1 `DTensor<Rank1<N>, F32>`
 * input parameter, `arr[i]` GATHER, scalar arithmetic (subtract/multiply/sqrt), scalar
 * summation, rank-1 gradient output via GatherRule + gradAccum SCATTER_ADD chain.
 *
 * What this port does NOT do vs paper §6.2:
 * - 1D positions only (paper uses 2D or 3D per context; 1D keeps math tractable).
 * - 3 vertices (paper uses 10/20 per config; 3 is small enough for hand-computed
 *   gradient anchor).
 * - No ACTUAL primal-elimination measurement. Tlaloc's current pipeline always
 *   clones primal ops the adjoint rules read (SqrtRule reads its operand, etc.),
 *   so even though symbolically the gradient could be simpler, our compile path
 *   emits the clone. Closing this gap needs a dxir-level liveness-aware clone
 *   pruning pass — post-D.2 work.
 *
 * Correctness: hand-computed gradient + finite-difference cross-check.
 */
class HookeanSpringTest {

    @Test
    fun `hookean spring triangle at rest yields zero gradient`() {
        // At the rest configuration (stretches all zero), the gradient of total
        // energy wrt every vertex position is zero. Sanity anchor — no matter what
        // the symbolic differentiation does, the energy minimum has grad = 0.
        val src = """
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
                    // Spring 0-1, rest length 1.
                    val d01 = p[0] - p[1]
                    val len01 = (d01 * d01).sqrt()
                    val s01 = len01 - 1.0f
                    // Spring 1-2, rest length 1.
                    val d12 = p[1] - p[2]
                    val len12 = (d12 * d12).sqrt()
                    val s12 = len12 - 1.0f
                    // Spring 0-2, rest length 2.
                    val d02 = p[0] - p[2]
                    val len02 = (d02 * d02).sqrt()
                    val s02 = len02 - 2.0f
                    // Total energy (k = 1): 0.5 * sum of stretches^2.
                    0.5f * (s01 * s01 + s12 * s12 + s02 * s02)
                }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(0.0f, 1.0f, 2.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, grad.size)
        for ((k, v) in grad.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-3f,
                "slot $k = $v matches broken-stub sentinel; IR transform did not fire",
            )
            assertTrue(
                abs(v) < 1e-4f,
                "slot $k = $v: expected ~0 at rest configuration",
            )
        }
    }

    @Test
    fun `hookean spring triangle perturbed matches hand-computed gradient`() {
        // Perturb p[1] to 1.5 (stretches spring 0-1 by 0.5, compresses spring 1-2
        // by 0.5). Spring 0-2 stays at rest length.
        //
        // d01 = p[0] - p[1] = -1.5, len01 = 1.5, s01 = 0.5
        // d12 = p[1] - p[2] = -0.5, len12 = 0.5, s12 = -0.5
        // d02 = p[0] - p[2] = -2.0, len02 = 2.0, s02 = 0.0
        //
        // Hand-computed:
        //   dE/dp[0] = s01 · d(len01)/dp[0] + s02 · d(len02)/dp[0]
        //            = 0.5 · (-1) + 0 = -0.5
        //   dE/dp[1] = s01 · d(len01)/dp[1] + s12 · d(len12)/dp[1]
        //            = 0.5 · 1 + (-0.5) · (-1) = 1.0
        //   dE/dp[2] = s12 · d(len12)/dp[2] + s02 · d(len02)/dp[2]
        //            = -0.5 · 1 + 0 = -0.5
        //
        // Net force [-0.5, 1.0, -0.5] — mass conservation Σ = 0, as physics demands.
        val src = """
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
                    val d01 = p[0] - p[1]
                    val len01 = (d01 * d01).sqrt()
                    val s01 = len01 - 1.0f
                    val d12 = p[1] - p[2]
                    val len12 = (d12 * d12).sqrt()
                    val s12 = len12 - 1.0f
                    val d02 = p[0] - p[2]
                    val len02 = (d02 * d02).sqrt()
                    val s02 = len02 - 2.0f
                    0.5f * (s01 * s01 + s12 * s12 + s02 * s02)
                }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(0.0f, 1.5f, 2.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, grad.size)
        val expected = floatArrayOf(-0.5f, 1.0f, -0.5f)
        for (k in 0 until 3) {
            assertTrue(
                abs(grad[k] - expected[k]) < 1e-4f,
                "slot $k: expected ${expected[k]}, got ${grad[k]}",
            )
        }
        // Conservation of momentum: forces sum to zero.
        assertTrue(
            abs(grad.sum()) < 1e-4f,
            "expected Σ grad = 0 (conservation of momentum), got ${grad.sum()}",
        )
    }

    @Test
    fun `hookean spring gradient agrees with finite differences over a sweep`() {
        // Central-difference cross-check across 5 perturbed configurations. 5e-3
        // relative tolerance — sqrt-bearing primal accumulates modest f32 noise.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun primal(p: DTensor<Rank1<Sym>, F32>): Float {
                val d01 = p[0] - p[1]
                val len01 = (d01 * d01).sqrt()
                val s01 = len01 - 1.0f
                val d12 = p[1] - p[2]
                val len12 = (d12 * d12).sqrt()
                val s12 = len12 - 1.0f
                val d02 = p[0] - p[2]
                val len02 = (d02 * d02).sqrt()
                val s02 = len02 - 2.0f
                return 0.5f * (s01 * s01 + s12 * s12 + s02 * s02)
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val d01 = p[0] - p[1]
                    val len01 = (d01 * d01).sqrt()
                    val s01 = len01 - 1.0f
                    val d12 = p[1] - p[2]
                    val len12 = (d12 * d12).sqrt()
                    val s12 = len12 - 1.0f
                    val d02 = p[0] - p[2]
                    val len02 = (d02 * d02).sqrt()
                    val s02 = len02 - 2.0f
                    0.5f * (s01 * s01 + s12 * s12 + s02 * s02)
                }
                // Five perturbed configurations avoiding d = 0 (sqrt(0) = 0, but its
                // derivative is +∞ — the physics stays well-defined only for taut springs).
                val configs = listOf(
                    floatArrayOf(0.0f, 1.2f, 2.1f),
                    floatArrayOf(0.1f, 1.0f, 2.3f),
                    floatArrayOf(-0.5f, 1.0f, 2.5f),
                    floatArrayOf(0.0f, 0.8f, 1.7f),
                    floatArrayOf(0.2f, 1.4f, 2.0f),
                )
                val eps = 1.0e-3f
                for (cfg in configs) {
                    val input = Tensors.f32Vector<Sym>(cfg)
                    val analytic = g(input).hostF32()
                    for (k in 0 until 3) {
                        val plus = cfg.copyOf().also { it[k] += eps }
                        val minus = cfg.copyOf().also { it[k] -= eps }
                        val fd = (primal(Tensors.f32Vector<Sym>(plus)) - primal(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                        print("${'$'}{analytic[k]}=${'$'}fd;")
                    }
                    println()
                }
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val lines = result.stdout.trim().lines()
        assertEquals(5, lines.size, "expected 5 config lines, got:\n${result.stdout}")
        for ((lineIdx, line) in lines.withIndex()) {
            val entries = line.split(";").filter { it.isNotEmpty() }
            assertEquals(3, entries.size, "expected 3 slots per line; got: $line")
            for ((k, entry) in entries.withIndex()) {
                val (a, fd) = entry.split("=").map { it.toFloat() }
                assertTrue(
                    abs(a + 1.0f) > 1e-3f,
                    "config $lineIdx slot $k: analytic=$a matches broken-stub sentinel",
                )
                // Mixed absolute/relative tolerance — near-zero gradients (physics:
                // forces at a near-rest slot) blow up pure relative error. Combined
                // form passes iff at least ONE of the two bounds holds: either the
                // abs error < 1e-4, or the relative error < 5e-3.
                val absErr = abs(a - fd)
                val relErr = absErr / (abs(fd) + 1.0e-7f)
                assertTrue(
                    absErr < 1e-4f || relErr < 5e-3f,
                    "config $lineIdx slot $k: analytic=$a fd=$fd absErr=$absErr relErr=$relErr — both tolerances exceeded",
                )
            }
        }
    }

    @Test
    fun `hookean spring chain of 10 vertices loop-driven gradient is correct`() {
        // N=10 vertices connected by 9 rest-length-1 springs in a chain (paper's
        // smallest config is 10 vertices, though arrangement-in-chain is a simpler
        // topology than their 2D network — enough to exercise the loop-driven
        // per-spring-gather path).
        //
        // Initial config: vertex k at position k*1.2 → each spring is stretched by
        // 0.2 (stretch = 0.2). Energy = 0.5·9·0.04 = 0.18.
        //
        // Gradient:
        //   For interior vertex k (1 ≤ k ≤ 8): forces from springs (k-1,k) and
        //   (k,k+1) cancel (both stretched by same amount, opposite signs wrt p[k]).
        //   So interior grad[k] = stretch·1 + stretch·(-1) = 0.
        //   Endpoint k=0: only spring (0,1) contributes. grad[0] = stretch·(-1) = -0.2.
        //   Endpoint k=9: only spring (8,9) contributes. grad[9] = stretch·1 = 0.2.
        val src = """
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
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_10, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(10, grad.size)
        assertTrue(
            abs(grad[0] + 1.0f) > 1e-3f,
            "slot 0 = ${grad[0]} matches broken-stub sentinel",
        )
        assertTrue(abs(grad[0] - (-0.2f)) < 1e-4f, "grad[0] = ${grad[0]}, expected -0.2")
        assertTrue(abs(grad[9] - 0.2f) < 1e-4f, "grad[9] = ${grad[9]}, expected 0.2")
        for (k in 1..8) {
            assertTrue(abs(grad[k]) < 1e-4f, "grad[$k] = ${grad[k]}, expected 0 (interior cancellation)")
        }
        // Momentum conservation — closed system.
        assertTrue(abs(grad.sum()) < 1e-4f, "Σ grad = ${grad.sum()}, expected 0")
    }

    @Test
    fun `hookean spring N=10 chain — measured timings`() {
        // Perf anchor for HookeanSpring, mirroring BrachistochroneTest's N=64 perf
        // test methodology (§0.4.44). N=9 springs, loop-driven, 18 gathers total.
        // Reports PERF lines for human review — no hard assertion on ns values.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun primal(p: DTensor<Rank1<Sym>, F32>): Float {
                var energy = 0.0f
                for (i in 0 until 9) {
                    val d = p[i] - p[i + 1]
                    val len = (d * d).sqrt()
                    val stretch = len - 1.0f
                    energy = energy + 0.5f * stretch * stretch
                }
                return energy
            }
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
                val arr = FloatArray(10) { it.toFloat() * 1.2f }
                val input = Tensors.f32Vector<Sym>(arr)
                var sinkF = 0.0f
                var sinkG = 0.0f
                for (i in 0 until 1000) {
                    sinkF += primal(input)
                    sinkG += g(input).hostF32()[0]
                }
                val iters = 5000
                val tFwdStart = System.nanoTime()
                for (i in 0 until iters) sinkF += primal(input)
                val tFwdNs = (System.nanoTime() - tFwdStart) / iters
                val tGradStart = System.nanoTime()
                for (i in 0 until iters) sinkG += g(input).hostF32()[0]
                val tGradNs = (System.nanoTime() - tGradStart) / iters
                val checkSlot0 = g(input).hostF32()[0]
                println("PERF forward_ns_per_call=${'$'}tFwdNs")
                println("PERF gradient_ns_per_call=${'$'}tGradNs")
                println("PERF ratio_grad_over_fwd=${'$'}{tGradNs.toDouble() / tFwdNs}")
                println("PERF first_grad_slot0=${'$'}checkSlot0")
                println("PERF sink=${'$'}sinkF/${'$'}sinkG")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_10, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val perfLines = result.stdout.lines().filter { it.startsWith("PERF ") }
        assertEquals(5, perfLines.size, "expected 5 PERF lines, got:\n${result.stdout}")
        val metrics = perfLines.associate { line ->
            val (k, v) = line.removePrefix("PERF ").split("=", limit = 2)
            k to v
        }
        val fwdNs = metrics["forward_ns_per_call"]?.toLong() ?: error("missing forward_ns_per_call")
        val gradNs = metrics["gradient_ns_per_call"]?.toLong() ?: error("missing gradient_ns_per_call")
        val ratio = metrics["ratio_grad_over_fwd"]?.toDouble() ?: error("missing ratio_grad_over_fwd")
        val firstSlot = metrics["first_grad_slot0"]?.toFloat() ?: error("missing first_grad_slot0")
        assertTrue(
            abs(firstSlot + 1.0f) > 1e-3f,
            "first_grad_slot0=$firstSlot matches broken-stub sentinel",
        )
        assertTrue(ratio > 0.5, "gradient/forward ratio=$ratio implausibly low")
        assertTrue(ratio < 200.0, "gradient/forward ratio=$ratio exceeds 200×")
        println("[HookeanSpring N=10 perf] forward=${fwdNs}ns/call gradient=${gradNs}ns/call ratio=${ratio}")
    }

    // --------- Harness (file-local; mirrors BrachistochroneTest's helpers) ---------

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    )

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-hookean-run").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }

            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean =
                    collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
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
            val capturedOut = PrintStream(baos, /* autoFlush = */ true, Charsets.UTF_8)
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

    companion object {
        private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f, -1.0f)), intArrayOf(3), F32) }
        """.trimIndent()

        /** Sentinel stub sized for the N=10 chain config. */
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
    }
}
