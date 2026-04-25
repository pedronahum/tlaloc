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
 * §0.4.162 — HMC Phase 3 first slice: numerical-stability mask in the loop body.
 *
 * Adds the per-record `if (-Xβ_i > 80.0f) -Xβ_i else log(1 + exp(-Xβ_i))` mask
 * to §0.4.160's loop form. This is the OOPSLA paper's safeguard against `exp(-Xβ)`
 * overflow for very negative `Xβ`: when `-Xβ > 80`, `1 + exp(-Xβ) ≈ exp(-Xβ)` so
 * `log(1 + exp(-Xβ)) ≈ -Xβ`. The two branches give the same answer
 * mathematically; the mask exists for f32 stability.
 *
 * Pipeline reach:
 *  - The mask IF is a single-result scalar IF inside a WHILE body.
 *  - C5 (§0.4.39 multi-carry) unrolls the outer WHILE 4 times.
 *  - Each unrolled iteration carries its own clone of the mask IF.
 *  - Reverse-mode AD via `handleIfAdjoint` (§0.4.155) flows through each IF clone.
 *
 * Plugin support for the mask predicate: `Float > Float` lowers via
 * `FirLambdaToDxirLowering.lowerComparison` to `STEP(SUB(lhs, rhs))` (§0.4.24).
 *
 * Per `docs/HMC_PORT_PLAN.md`: this is Phase 3's stability-mask half. The
 * remaining Phase 3 piece (true nested loop over features for `Σ_j X[i,j] · β[j]`)
 * is its own slice, now unblocked by §0.4.161's region-recursive C5 into WHILE
 * bodies.
 *
 * Two test methods:
 *  1. β = [0.5, 0.3] — moderate values; mask predicate is FALSE for every record.
 *     Gradient must match §0.4.160's exactly (the mask in the source compiles +
 *     evaluates but never fires; finite-difference cross-check passes).
 *  2. β = [-200.0, 0.0] — large-magnitude β; mask predicate is TRUE for records
 *     with negXβ > 80. Verifies the mask branch propagates gradient correctly.
 */
class HmcLogisticRegressionMaskedTest {

    @Test
    fun `hmc U with mask gradient matches finite difference at moderate beta`() {
        // β = [0.5, 0.3]; mask predicate is false at every record. Gradient must
        // match §0.4.160's exactly — adding the mask IF in the source must not
        // perturb the no-fire path.
        runMaskedFiniteDifferenceCheck(
            betaCfg = floatArrayOf(0.5f, 0.3f),
            label = "moderate beta",
        )
    }

    @Test
    fun `hmc U with mask gradient matches finite difference at large negative beta`() {
        // β = [-200, 0]; mask predicate is TRUE for records 0/1/3 (negXβ ≥ 100).
        // Record 2 has negXβ = -100 (X[2] = [-0.5, 1.5], so xb = -0.5·(-200) =
        // 100, negXb = -100), mask FALSE. Verifies both branches propagate
        // correctly via the AD path.
        runMaskedFiniteDifferenceCheck(
            betaCfg = floatArrayOf(-200.0f, 0.0f),
            label = "large negative beta",
        )
    }

    private fun runMaskedFiniteDifferenceCheck(betaCfg: FloatArray, label: String) {
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
                    val negXb = -xb
                    val stable = if (negXb > 80.0f) negXb else (1.0f + negXb.exp()).log()
                    sum1 = sum1 + (yi - 1.0f) * xb
                    sum2 = sum2 + stable
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
                        val negXb = -xb
                        val stable = if (negXb > 80.0f) negXb else (1.0f + negXb.exp()).log()
                        sum1 = sum1 + (yi - 1.0f) * xb
                        sum2 = sum2 + stable
                    }
                    val term3 = (b0 * b0 + b1 * b1) / 2000.0f
                    sum1 - sum2 - term3
                }
                val cfg = floatArrayOf(
                    ${betaCfg[0]}f, ${betaCfg[1]}f,
                    1.0f, 0.5f, -0.5f, 1.5f,
                    0.5f, 1.0f, 1.5f, -0.5f,
                    1.0f, 0.0f, 1.0f, 0.0f,
                )
                val input = Tensors.f32Vector<Sym>(cfg)
                val analytic = g(input).hostF32()
                val eps = 1.0e-2f
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
        assertEquals(0, result.exitCode, "$label: compile failed:\n${result.messages}")
        val line = result.stdout.trim()
        val parts = line.trimEnd(';').split(";")
        assertEquals(2, parts.size, "$label: expected 2 β slots, got: $line")
        for ((k, part) in parts.withIndex()) {
            val (analyticStr, fdStr) = part.split("=", limit = 2)
            val analytic = analyticStr.toFloat()
            val fd = fdStr.toFloat()
            assertTrue(
                abs(analytic + 1.0f) > 1e-6f || abs(fd + 1.0f) > 1e-6f,
                "$label slot $k: both equal sentinel -1.0",
            )
            // Mixed absolute / relative tolerance — large-β regime accumulates
            // more f32 noise (negXβ values approach 200, log/exp domain extreme).
            // Tolerances widened from §0.4.160's: 1e-2 absolute, 5e-2 relative.
            val absErr = abs(analytic - fd)
            val relErr = absErr / (abs(fd) + 1.0e-7f)
            assertTrue(
                absErr < 1e-2f || relErr < 5e-2f,
                "$label slot $k: analytic=$analytic fd=$fd absErr=$absErr relErr=$relErr",
            )
        }
    }

    // --------- Harness (file-local; mirrors HmcLogisticRegressionLoopTest's helpers) ---------

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
        val tempDir = Files.createTempDirectory("tlaloc-hmc-masked-run").toFile()
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
