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
 * §0.4.160 — HMC Phase 2 port (loop form).
 *
 * Refactors §0.4.159's straight-line port to use `for (i in 0 until 4)` accumulator
 * loops mirroring HookeanSpring's `N=10 chain` test pattern (§0.4.47) and BGDHyperOpt's
 * inner-for sub-kernel (§0.4.49). With concrete bound n=4, C5 (simple-loop direct
 * unroll) fires and produces a straight-line MUL/ADD chain post-coarsening — same
 * gradient as Phase 1, different IR shape pre-unroll.
 *
 * Packing (BGDHyperOpt convention, column-major over X to keep all GATHER indices
 * as `offset + i`):
 *   slots [0..1]   = β[0..1]
 *   slots [2..5]   = X[:,0] (4 records' first feature)
 *   slots [6..9]   = X[:,1] (4 records' second feature)
 *   slots [10..13] = y[0..3]
 *
 * Under this layout `xi0 = packed[2 + i]`, `xi1 = packed[6 + i]`, `yi = packed[10 + i]`
 * — all simple offset+counter indexing that the §0.4.42 GATHER lowering supports.
 *
 * Two accumulators (`sum1` for term1, `sum2` for term2) means the WHILE has 2 user-
 * carried slots + 1 counter — multi-carry shape that §0.4.39's C5 widening handles
 * via `findReferencedCarried` returning the set of live indices.
 *
 * Per `docs/HMC_PORT_PLAN.md`: this is Phase 2 (loop form). Phase 3 (numerical-
 * stability mask + region-recursive C5 into WHILE bodies) remains ahead.
 */
class HmcLogisticRegressionLoopTest {

    @Test
    fun `hmc U loop form gradient matches finite difference at small fixed dataset`() {
        // Test packed input (n=4, d=2), β = [0.5, 0.3]:
        //   X = [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]]
        //   y = [1, 0, 1, 0]
        // Same dataset as §0.4.159's straight-line test — same gradient at β.
        //
        // Verification: compute analytic g(packed) via grad; compute finite-difference
        // gradient by perturbing ONLY packed[0] / packed[1] (the β slots). Other slots
        // of the analytic gradient correspond to ∂U/∂X / ∂U/∂y which are non-zero but
        // not the test target; we only assert on the first 2 slots.
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
            fun primal(packed: DTensor<Rank1<Sym>, F32>): Float {
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
                return sum1 - sum2 - term3
            }
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
                // packed = [β[0], β[1], X[:,0]..., X[:,1]..., y...]
                val cfg = floatArrayOf(
                    0.5f, 0.3f,                          // β
                    1.0f, 0.5f, -0.5f, 1.5f,             // X[:,0]
                    0.5f, 1.0f, 1.5f, -0.5f,             // X[:,1]
                    1.0f, 0.0f, 1.0f, 0.0f,              // y
                )
                val input = Tensors.f32Vector<Sym>(cfg)
                val analytic = g(input).hostF32()
                val eps = 1.0e-3f
                // Only assert on the first 2 slots (β[0], β[1]) — that's the math target.
                for (k in 0 until 2) {
                    val plus = cfg.copyOf().also { it[k] += eps }
                    val minus = cfg.copyOf().also { it[k] -= eps }
                    val fd = (primal(Tensors.f32Vector<Sym>(plus)) - primal(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print("${'$'}{analytic[k]}=${'$'}fd;")
                }
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_14, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val line = result.stdout.trim()
        val parts = line.trimEnd(';').split(";")
        assertEquals(2, parts.size, "expected 2 β slots, got: $line")
        for ((k, part) in parts.withIndex()) {
            val (analyticStr, fdStr) = part.split("=", limit = 2)
            val analytic = analyticStr.toFloat()
            val fd = fdStr.toFloat()
            // Sentinel-defeat: at least one of (analytic, fd) per slot should differ
            // from -1.0f. Defends against the broken-stub fallback firing.
            assertTrue(
                abs(analytic + 1.0f) > 1e-6f || abs(fd + 1.0f) > 1e-6f,
                "slot $k: analytic=$analytic fd=$fd both equal -1.0 sentinel",
            )
            // Mixed absolute / relative tolerance — sigmoid-bearing primal noise.
            val absErr = abs(analytic - fd)
            val relErr = absErr / (abs(fd) + 1.0e-7f)
            assertTrue(
                absErr < 1e-3f || relErr < 5e-3f,
                "β slot $k: analytic=$analytic fd=$fd absErr=$absErr relErr=$relErr",
            )
        }
    }

    // --------- Harness (file-local; mirrors HmcLogisticRegressionTest's helpers) ---------

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
        val tempDir = Files.createTempDirectory("tlaloc-hmc-loop-run").toFile()
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
        /** Sentinel stub for `(DTensor<Rank1<Sym>, F32>) -> Float` at packed size 14. */
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
}
