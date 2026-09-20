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
 * §0.4.412 — the REVERSE-assembled (tall) Jacobian E2E: `jacobianReverse`
 * lowers through the K2 plugin as the §0.4.398 seeded reverse pullback
 * `vjp_f(x, ȳ) → x̄` plus the runtime output-basis assembly helper — one
 * Jacobian ROW per output-basis cotangent, m + 1 passes (one eager primal
 * to learn the output extent, then m pullbacks) versus `jacobian`'s n
 * forward passes. Like its forward sibling there is NO synthesis fallback
 * (the `concat` precedent), and no monomorphic stub is compiled: the user
 * source resolves the REAL generic `io.tlaloc.autograd.jacobianReverse`
 * declaration off the test classpath.
 *
 *  jr1 = jacobianReverse { x -> x ⊙ x }         → J = diag(2x)        [n, n]
 *  jf1 = jacobian        { x -> x ⊙ x }         → the SAME J, forward-assembled
 *  jr2 = jacobianReverse { x -> (x ⊙ x).sum() } → J = [2x] (grad row) [1, n]
 *  jr3 = jacobianReverse { x -> concat(x, x) }  → J = [I; I]          [2n, n]
 *
 * jr1 vs jf1 is the cross-assembly oracle — the two intrinsics run DIFFERENT
 * seeded transforms (reverse rows vs forward columns) over the same body, so
 * entrywise agreement pins both against each other. jr2 is the scalar-R
 * degenerate (the cotangent is a `Float` unit seed — the seeded reverse's
 * Float-typed upstream must synthesise). jr3 is the genuinely TALL case
 * (m = 2n > n): the stacked identity exercises row indexing through
 * ConcatRule's symbolic SLICE_LIKE adjoints.
 */
class JacobianReverseIntrinsicTest {

    @Test
    fun `jacobianReverse lowers through the plugin and assembles rows from seeded pullbacks`() {
        val src = """
            import io.tlaloc.autograd.jacobian
            import io.tlaloc.autograd.jacobianReverse
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.concat
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val jr1 = jacobianReverse { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val jf1 = jacobian { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val jr2 = jacobianReverse { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                val jr3 = jacobianReverse { x: DTensor<Rank1<Sym>, F32> -> concat(0, x, x) }
                val X = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                show("jr1", jr1(X))
                show("jf1", jf1(X))
                show("jr2", jr2(X))
                show("jr3", jr3(X))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; jacobianReverse has no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val rows = result.stdout.trim().lines().associate {
            val parts = it.trim().split(" ", limit = 3)
            parts[0] to (parts[1] to parts[2].split(",").map { s -> s.toFloat() })
        }

        fun check(name: String, wantDims: String, want: List<Float>, tol: Float = 1e-4f) {
            val (dims, got) = rows[name] ?: error("no '$name' row in stdout:\n${result.stdout}")
            assertEquals(wantDims, dims, "$name dims")
            assertEquals(want.size, got.size, "$name size")
            for (i in want.indices) {
                assertTrue(abs(got[i] - want[i]) < tol, "$name[$i]=${got[i]} want ${want[i]} (got $got)")
            }
        }

        // x = [1, 2, 3].
        check("jr1", "3x3", listOf(2f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 6f))
        // The cross-assembly oracle: forward-assembled columns must agree
        // ENTRYWISE with reverse-assembled rows over the same body.
        check("jf1", "3x3", listOf(2f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 6f))
        val jr1Row = rows["jr1"]!!.second
        val jf1Row = rows["jf1"]!!.second
        for (i in jr1Row.indices) {
            assertTrue(
                abs(jr1Row[i] - jf1Row[i]) < 1e-6f,
                "cross-assembly disagreement at [$i]: reverse=${jr1Row[i]} forward=${jf1Row[i]}",
            )
        }
        // Scalar-R degenerate: the [1, n] gradient row from a Float unit cotangent.
        check("jr2", "1x3", listOf(2f, 4f, 6f))
        // Tall: concat(x, x) has m = 2n = 6 > n = 3, J = [I; I] stacked.
        check(
            "jr3", "6x3",
            listOf(
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f,
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f,
            ),
        )
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jacobian-reverse-test").toFile()
        try {
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
}
