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
 * §0.4.159 — HMC Phase 1 port (scalar/straight-line form).
 *
 * Ports the OOPSLA 2021 paper §7.2 HMC kernel's `U(β)` function for Bayesian logistic
 * regression at fixed n=4 records, d=2 features. With the §0.4.158 unblock (scalar
 * `Float.exp()` / `Float.log()` plugin lowering), every primitive in the U decomposition
 * lowers cleanly through the K2 plugin: scalar arithmetic + GATHER (β[i]) + EXP + LOG.
 *
 * The paper's full U is `β^T X^T (y - 1_n) - 1_n^T [log(1 + exp(-Xβ))] - β^T β / (2σ_β²)`.
 * MATMUL through the plugin is its own arc (per §0.4.158's plan amendment), so this
 * Phase-1 port hard-codes `X` (4×2) and `y` (4) as Float literals and decomposes
 * `Xβ` per record into scalar dot products. n=4, d=2, σ_β²=1000.
 *
 * Correctness is verified by finite-difference cross-check at one β value — the
 * straightforward way to check a compound gradient without hand-computing through
 * sigmoid chain rules per slot.
 *
 * Per `docs/HMC_PORT_PLAN.md`: this is Phase 1 (scalar/straight-line form). Phase 2
 * will widen to the loop form (`for (i in 0 until n)` with affine-recurrence
 * accumulators that hit C6/C7 closure). Phase 3 adds the numerical-stability mask
 * (`if (-Xβ_i > 80) -Xβ_i else log(1 + exp(-Xβ_i))`) and requires §0.4.156's Phase 4b
 * (region-recursive C5 into WHILE bodies) for nested-loop coarsening.
 */
class HmcLogisticRegressionTest {

    @Test
    fun `hmc U gradient matches finite difference at small fixed dataset`() {
        // Hard-coded dataset:
        //   X = [[1.0,  0.5],          y = [1, 0, 1, 0]
        //        [0.5,  1.0],
        //        [-0.5, 1.5],
        //        [1.5, -0.5]]
        //   σ_β² = 1000  (so 2·σ_β² = 2000)
        // Test β = [0.5, 0.3].
        //
        // U(β) = β^T X^T (y - 1_n) - Σ_i log(1 + exp(-X[i,:]·β)) - (β·β) / 2000
        //      where (y - 1_n) = [0, -1, 0, -1].
        //
        // Verification: compute analytic g(β) via grad; compute finite-difference
        // gradient via central differences; require agreement within 1e-3 absolute
        // OR 5e-3 relative tolerance (matches HookeanSpring's mixed-tolerance pattern
        // for sigmoid-bearing primals).
        val src = """
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
            fun primal(beta: DTensor<Rank1<Sym>, F32>): Float {
                val b0 = beta[0]
                val b1 = beta[1]
                // X·β per record (4 records, 2 features each).
                val xb0 = 1.0f * b0 + 0.5f * b1
                val xb1 = 0.5f * b0 + 1.0f * b1
                val xb2 = -0.5f * b0 + 1.5f * b1
                val xb3 = 1.5f * b0 + (-0.5f) * b1
                // term1 = β^T X^T (y - 1_n);  y - 1_n = [0, -1, 0, -1].
                val term1 = 0.0f * xb0 + (-1.0f) * xb1 + 0.0f * xb2 + (-1.0f) * xb3
                // term2 = Σ_i log(1 + exp(-Xβ_i)).
                val log0 = (1.0f + (-xb0).exp()).log()
                val log1 = (1.0f + (-xb1).exp()).log()
                val log2 = (1.0f + (-xb2).exp()).log()
                val log3 = (1.0f + (-xb3).exp()).log()
                val term2 = log0 + log1 + log2 + log3
                // term3 = β^T β / (2·σ_β²).
                val term3 = (b0 * b0 + b1 * b1) / 2000.0f
                return term1 - term2 - term3
            }
            fun main() {
                val g = grad { beta: DTensor<Rank1<Sym>, F32> ->
                    val b0 = beta[0]
                    val b1 = beta[1]
                    val xb0 = 1.0f * b0 + 0.5f * b1
                    val xb1 = 0.5f * b0 + 1.0f * b1
                    val xb2 = -0.5f * b0 + 1.5f * b1
                    val xb3 = 1.5f * b0 + (-0.5f) * b1
                    val term1 = 0.0f * xb0 + (-1.0f) * xb1 + 0.0f * xb2 + (-1.0f) * xb3
                    val log0 = (1.0f + (-xb0).exp()).log()
                    val log1 = (1.0f + (-xb1).exp()).log()
                    val log2 = (1.0f + (-xb2).exp()).log()
                    val log3 = (1.0f + (-xb3).exp()).log()
                    val term2 = log0 + log1 + log2 + log3
                    val term3 = (b0 * b0 + b1 * b1) / 2000.0f
                    term1 - term2 - term3
                }
                val cfg = floatArrayOf(0.5f, 0.3f)
                val input = Tensors.f32Vector<Sym>(cfg)
                val analytic = g(input).hostF32()
                val eps = 1.0e-3f
                for (k in 0 until 2) {
                    val plus = cfg.copyOf().also { it[k] += eps }
                    val minus = cfg.copyOf().also { it[k] -= eps }
                    val fd = (primal(Tensors.f32Vector<Sym>(plus)) - primal(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print("${'$'}{analytic[k]}=${'$'}fd;")
                }
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_2, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val line = result.stdout.trim()
        // Format: `<analytic>=<fd>;<analytic>=<fd>;`
        val parts = line.trimEnd(';').split(";")
        assertEquals(2, parts.size, "expected 2 slots, got: $line")
        for ((k, part) in parts.withIndex()) {
            val (analyticStr, fdStr) = part.split("=", limit = 2)
            val analytic = analyticStr.toFloat()
            val fd = fdStr.toFloat()
            // Sentinel check: any slot equal to -1.0f exactly suggests the broken-stub
            // fallback fired (the IR transform did not). One coincident -1.0 is improbable
            // but possible at this β; require both slots collectively to differ from the
            // sentinel pattern.
            assertTrue(
                abs(analytic + 1.0f) > 1e-6f || abs(fd + 1.0f) > 1e-6f,
                "slot $k: analytic=$analytic fd=$fd both equal -1.0 sentinel; IR transform didn't fire",
            )
            // Mixed absolute / relative tolerance — sigmoid-bearing primals like
            // HookeanSpring's sqrt-bearing primal accumulate modest f32 noise.
            val absErr = abs(analytic - fd)
            val relErr = absErr / (abs(fd) + 1.0e-7f)
            assertTrue(
                absErr < 1e-3f || relErr < 5e-3f,
                "slot $k: analytic=$analytic fd=$fd absErr=$absErr relErr=$relErr — both tolerances exceeded",
            )
        }
    }

    // --------- Harness (file-local; mirrors HookeanSpringTest's helpers) ---------

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
        val tempDir = Files.createTempDirectory("tlaloc-hmc-run").toFile()
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
        /** Sentinel stub for `(DTensor<Rank1<Sym>, F32>) -> Float` at d=2. */
        private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f)), intArrayOf(2), F32) }
        """.trimIndent()
    }
}
