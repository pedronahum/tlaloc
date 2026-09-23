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
 * §0.4.176 — HMC Phase 3 nested-loop port. Closes the §0.4.163 deferred shape
 * (true nested for-loop over features `for (j) { xb += X[i,j] * β[j] }`), now
 * unblocked by §0.4.174 (lift pre-SCT) + §0.4.175 (deep-clone IF on empty regions).
 *
 * Mirrors HMC Phase 2 ([HmcLogisticRegressionLoopTest]) but uses a NESTED for-loop
 * for the per-record dot product, exercising:
 *  - §0.4.163's `collectMutatedTargets` local-decl exclusion (locally-declared
 *    `var xb = 0` inside the outer loop, mutated by the inner loop).
 *  - §0.4.161's region-recursive C5 (inner WHILE inside outer WHILE body).
 *  - §0.4.174's lift pass (post-coarsening IFs from C5/coarsening passes get
 *    their region body ops lifted to top level).
 *  - §0.4.175's deep-clone IF (empty-region IFs after lift get cloned into the
 *    grad body so downstream operand resolution doesn't leak primal ids).
 *
 * Same dataset as HMC Phase 2 (n=4, d=2). Compares analytic gradient against
 * finite-difference at the β slots only (slots 2-13 are ∂U/∂X / ∂U/∂y, not the
 * test target).
 */
class HmcNestedLoopTest {

    @Test
    fun `hmc nested loop gradient matches finite difference at small fixed dataset`() {
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
                    var xb = 0.0f
                    for (j in 0 until 2) {
                        val betaJ = packed[j]
                        val xij = packed[2 + j * 4 + i]
                        xb = xb + xij * betaJ
                    }
                    val yi = packed[10 + i]
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
                        var xb = 0.0f
                        for (j in 0 until 2) {
                            val betaJ = packed[j]
                            val xij = packed[2 + j * 4 + i]
                            xb = xb + xij * betaJ
                        }
                        val yi = packed[10 + i]
                        sum1 = sum1 + (yi - 1.0f) * xb
                        sum2 = sum2 + (1.0f + (-xb).exp()).log()
                    }
                    val term3 = (b0 * b0 + b1 * b1) / 2000.0f
                    sum1 - sum2 - term3
                }
                val cfg = floatArrayOf(
                    0.5f, 0.3f,                            // β
                    1.0f, 0.5f, -0.5f, 1.5f,               // X[:,0]
                    0.5f, 1.0f, 1.5f, -0.5f,               // X[:,1]
                    1.0f, 0.0f, 1.0f, 0.0f,                // y
                )
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
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_14, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // Sanity: confirm IR-side synthesis fired (no "kept original call" warning).
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "K2 plugin failed to synthesise HMC nested-loop — runtime tape fallback fired",
        )
        val parts = result.stdout.trim().trimEnd(';').split(";")
        assertEquals(2, parts.size, "expected 2 β slots, got: ${result.stdout}")
        for ((k, part) in parts.withIndex()) {
            val (analyticStr, fdStr) = part.split("=", limit = 2)
            val analytic = analyticStr.toFloat()
            val fd = fdStr.toFloat()
            assertTrue(
                abs(analytic + 1.0f) > 1e-6f || abs(fd + 1.0f) > 1e-6f,
                "slot $k: analytic=$analytic fd=$fd both equal -1.0 sentinel",
            )
            val absErr = abs(analytic - fd)
            val relErr = absErr / (abs(fd) + 1.0e-7f)
            // Mixed tolerance — sigmoid-bearing primal accumulates f32 noise.
            assertTrue(
                absErr < 1e-3f || relErr < 5e-3f,
                "β slot $k: analytic=$analytic fd=$fd absErr=$absErr relErr=$relErr",
            )
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
        val tempDir = Files.createTempDirectory("tlaloc-hmc-nested-run").toFile()
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
