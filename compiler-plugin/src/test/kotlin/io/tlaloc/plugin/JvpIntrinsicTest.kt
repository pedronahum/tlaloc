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
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.372 — Phase B1 E2E: the forward-mode user intrinsics `jvp` and
 * `valueAndJvp` lower through the K2 plugin (DxirForwardTransform +
 * synthesis), the missing user surface for the §0.4.361 forward transform.
 *
 *  j1 = jvp { x -> (x⊙x).sum() }         → (x, dx) ↦ dy = 2·⟨x, dx⟩
 *  j2 = valueAndJvp { x -> (x⊙x).sum() } → (x, dx) ↦ (y=Σx², dy=2⟨x,dx⟩)
 *  j3 = jvp { x -> exp(x).sum() }        → (x, dx) ↦ dy = Σ exp(x)·dx
 *
 * The scalar-output JVP is the directional derivative ⟨∇f(x), dx⟩, so it
 * also cross-checks the reverse mode: dy must equal grad·dx (verified here
 * against the closed forms).
 */
class JvpIntrinsicTest {

    @Test
    fun `jvp and valueAndJvp lower through the plugin`() {
        val src = """
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndJvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.exp
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val j1 = jvp { x: DTensor<Rank2<Sym, Lit<Int>>, F32> -> (x * x).sum().toFloat() }
                val j2 = valueAndJvp { x: DTensor<Rank2<Sym, Lit<Int>>, F32> -> (x * x).sum().toFloat() }
                val j3 = jvp { x: DTensor<Rank2<Sym, Lit<Int>>, F32> -> x.exp().sum().toFloat() }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                val V = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f))
                val dy1 = j1(X, V)
                val (y2, dy2) = j2(X, V)
                val dy3 = j3(X, V)
                println("dy1 " + dy1)
                println("y2 " + y2)
                println("dy2 " + dy2)
                println("dy3 " + dy3)
            }
        """.trimIndent()
        val result = compileAndRun(STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected jvp/valueAndJvp to lower forward-mode. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val x = floatArrayOf(1f, 2f, 3f, 4f)
        val v = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val wantY = x.sumOf { (it * it).toDouble() }.toFloat()                 // Σ x²
        val wantDy1 = 2f * x.indices.sumOf { (x[it] * v[it]).toDouble() }.toFloat()  // 2⟨x, dx⟩
        val wantDy3 = x.indices.sumOf { (exp(x[it]) * v[it]).toDouble() }.toFloat()  // Σ exp(x)·dx

        val vals = result.stdout.trim().lines().associate {
            val (k, num) = it.trim().split(" ", limit = 2)
            k to num.toFloat()
        }
        assertTrue(vals["dy1"] != -1f, "j1 sentinel returned — rewrite never fired. stdout:\n${result.stdout}")
        assertTrue(abs(vals.getValue("dy1") - wantDy1) < 1e-3f, "dy1=${vals["dy1"]} want $wantDy1")
        assertTrue(abs(vals.getValue("y2") - wantY) < 1e-3f, "y2=${vals["y2"]} want $wantY")
        assertTrue(abs(vals.getValue("dy2") - wantDy1) < 1e-3f, "dy2=${vals["dy2"]} want $wantDy1")
        assertTrue(abs(vals.getValue("dy3") - wantDy3) < 1e-2f, "dy3=${vals["dy3"]} want $wantDy3")
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jvp-test").toFile()
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
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: ${t.cause ?: t}",
                    ),
                    baos.toString(Charsets.UTF_8),
                )
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun jvp(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float =
                { _, _ -> -1.0f }
            fun valueAndJvp(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
        """.trimIndent()
    }
}
