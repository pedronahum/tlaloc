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
 * §0.4.375 — Phase A4b E2E: `outerProduct(a, b)` differentiates end-to-end
 * through the K2 plugin. The host op + FIR lowering
 * (`MATMUL(reshape(a, [n,1]), reshape(b, [1,m]))`) + IR-level gradient all
 * landed §0.4.369, but `grad {}` fell back to the runtime tape: the
 * reshape-created unit axes carried no param-sourced shape atom, so
 * MatmulRule's adjoint `TRANSPOSE([n,1])` could not derive an IrType
 * (`irOpFor returned null for TRANSPOSE`).
 *
 * §0.4.375 closes it by synthesising a placeholder `Lit<Int>` atom for
 * reshape-created unit axes — forward through the `[n]→[n,1]` / `[m]→[1,m]`
 * reshapes (so the downstream TRANSPOSE types), and backward through the
 * `[n,1]→[n]` / `[1,m]→[m]` squeeze on the way out (so the MATMUL solver can
 * fill the `[n,m]` scalar-seed's IrType). Params carry DISTINCT shape atoms
 * (`n = Sym`, `m = Lit<Int>`) so the seed-broadcast axis-matcher resolves
 * `[n,m]` unambiguously.
 *
 * With `Σ outerProduct(a, b)` the upstream into the outer product is a matrix
 * of ones, so:
 *   da_i = Σ_j 1·b[j] = Σ_j b[j]   (each i gets the full column-sum of b)
 *   db_j = Σ_i 1·a[i] = Σ_i a[i]   (each j gets the full row-sum of a)
 */
class OuterProductGradientTest {

    @Test
    fun `grad through outerProduct`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.outerProduct
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Lit<Int>>, F32> ->
                    outerProduct(a, b).sum().toFloat()
                }
                val A = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val B = Tensors.f32Vector<Lit<Int>>(floatArrayOf(0.5f, -1.0f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected outerProduct to lower. Warnings:\n" +
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message },
        )

        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(0.5f, -1f)
        val sumB = b.sum()
        val sumA = a.sum()
        val want = mapOf(
            "da" to FloatArray(a.size) { sumB }.toList(),
            "db" to FloatArray(b.size) { sumA }.toList(),
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertEquals(expect.size, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f },
                "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            for (j in expect.indices) {
                assertTrue(
                    abs(values[j] - expect[j]) < 1e-4f,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                )
            }
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
        val tempDir = Files.createTempDirectory("tlaloc-outer-test").toFile()
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
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Lit<Int>>, F32>) ->
                        Pair<DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Lit<Int>>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32),
                    DTensor(HostF32Storage(FloatArray(2) { -1.0f }), intArrayOf(2), F32),
                ) }
        """.trimIndent()
    }
}
