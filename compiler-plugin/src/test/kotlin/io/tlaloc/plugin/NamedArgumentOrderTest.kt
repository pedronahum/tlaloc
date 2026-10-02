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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Kotlin accepts named arguments in any order, and FIR keeps the argument list in
 * source order. Every lowering arm must read an argument by the parameter it binds
 * to: each case below computes one gradient with positional arguments and again with
 * the same arguments named in another order, and the two must be bit-identical.
 * Every case is chosen so that reading by position gives another gradient.
 */
class NamedArgumentOrderTest {

    @Test
    fun `named arguments in any order lower to the parameters they name`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.grad3
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.avgPool2d
            import io.tlaloc.core.ops.batchNorm
            import io.tlaloc.core.ops.clip
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.convTranspose2d
            import io.tlaloc.core.ops.maxPool2d
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            typealias T4 = DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>
            typealias T1 = DTensor<Rank1<Sym>, F32>
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun data(n: Int, k: Int) = FloatArray(n) { kotlin.math.sin(0.37f * it + k).toFloat() }
            fun main() {
                val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 6, 5, data(60, 0))
                val w = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(3, 2, 3, 2, data(36, 1))
                val wt = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 3, 2, 2, data(24, 2))
                val g = Tensors.f32Vector<Sym>(floatArrayOf(1.5f, -0.25f))
                val b = Tensors.f32Vector<Sym>(floatArrayOf(0.3f, 2.0f))
                val xv = Tensors.f32Vector<Sym>(data(12, 3))

                show("conv.pos", grad2 { x: T4, w: T4 -> val y = x.conv2d(w, 2, 1, 1, 0, 2, 1); (y * y).sum().toFloat() }(x, w).first)
                show("conv.named", grad2 { x: T4, w: T4 ->
                    val y = x.conv2d(padRight = 1, padLeft = 2, padBottom = 0, padTop = 1, strideW = 1, strideH = 2, w = w)
                    (y * y).sum().toFloat()
                }(x, w).first)

                show("convT.pos", grad2 { x: T4, wt: T4 -> val y = x.convTranspose2d(wt, 2, 1, 0, 1, 1, 0); (y * y).sum().toFloat() }(x, wt).first)
                show("convT.named", grad2 { x: T4, wt: T4 ->
                    val y = x.convTranspose2d(wt, padRight = 0, padLeft = 1, padBottom = 1, padTop = 0, strideW = 1, strideH = 2)
                    (y * y).sum().toFloat()
                }(x, wt).first)

                show("avg.pos", grad { x: T4 -> val y = x.avgPool2d(3, 2, 2, 1, 0, 1, 1, 0); (y * y).sum().toFloat() }(x))
                show("avg.named", grad { x: T4 ->
                    val y = x.avgPool2d(padRight = 0, padLeft = 1, padBottom = 1, padTop = 0, strideW = 1, strideH = 2, windowW = 2, windowH = 3)
                    (y * y).sum().toFloat()
                }(x))

                show("max.pos", grad { x: T4 -> val y = x.maxPool2d(2, 3, 2, 3, 0, 0, 0, 0); (y * y).sum().toFloat() }(x))
                show("max.named", grad { x: T4 ->
                    val y = x.maxPool2d(windowW = 3, strideH = 2, windowH = 2, strideW = 3, padRight = 0, padLeft = 0, padBottom = 0, padTop = 0)
                    (y * y).sum().toFloat()
                }(x))

                show("clip.pos", grad { v: T1 -> val y = clip(v, -0.5f, 0.9f); (y * y).sum().toFloat() }(xv))
                show("clip.named", grad { v: T1 -> val y = clip(v, hi = 0.9f, lo = -0.5f); (y * y).sum().toFloat() }(xv))

                show("bn.pos", grad3 { x: T4, s: T1, o: T1 -> val y = x.batchNorm(s, o, 0.1f); (y * y * y).sum().toFloat() }(x, g, b).first)
                show("bn.named", grad3 { x: T4, s: T1, o: T1 -> val y = x.batchNorm(eps = 0.1f, offset = o, scale = s); (y * y * y).sum().toFloat() }(x, g, b).first)
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (key, value) = line.split(" ", limit = 2)
            key to value
        }
        for (op in listOf("conv", "convT", "avg", "max", "clip", "bn")) {
            assertEquals(rows.getValue("$op.pos"), rows.getValue("$op.named"), "$op: named arguments in another order")
        }
    }

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)

    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compileAndRun(program: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-named-args-test").toFile()
        try {
            File(tempDir, "Main.kt").writeText(program)
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
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(CompilerMessageSeverity.ERROR, "RUN FAILURE: ${t.cause ?: t}"),
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
