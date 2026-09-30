@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.plugin

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.DxirVmapTransform
import io.tlaloc.ir.render.toKotlinSource
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
 * Readable source for vmapped functions. A batched function with concrete extents
 * (per-example gradients: `vmap` of the reverse transform) is printed by `toKotlinSource`
 * as Kotlin over the `:core` host twins, compiled without the plugin, run, and its
 * outputs must be raw-bit identical to the interpreter evaluating the same function. And
 * `dumpGradSource` covers a `vmap {}` call: a plugin-lowered tensor lambda has `-1`
 * extents, which the printer refuses by name, as it does for tensor `grad {}` lambdas.
 */
class VmapReadableSourceTest {

    private fun t(vararg dims: Int) = DxirType(F32, dims.toList())

    private fun data(n: Int, seed: Int): FloatArray {
        var s = seed.toLong() * 7919L + 12345L
        return FloatArray(n) {
            s = (s * 1103515245L + 12345L) and 0x7fffffffL
            ((s % 20000L) - 10000L) / 7000.0f
        }
    }

    @Test
    fun `printed per-example gradients compile, run, and match the interpreter bit for bit`() {
        // loss(w, x) = sum(tanh(x · w + b)), x [1, 3] per example, w [3, 2], b [2].
        val loss = DxirBuilder.function("perExampleLoss") {
            val w = param("w", t(3, 2))
            val b = param("b", t(2))
            val x = param("x", t(1, 3))
            val h = op(OpKind.ADD, listOf(op(OpKind.MATMUL, listOf(x, w), t(1, 2)), b), t(1, 2))
            listOf(op(OpKind.SUM, listOf(op(OpKind.TANH, listOf(h), t(1, 2))), t()))
        }
        val batch = 4
        val perExample = DxirVmapTransform.apply(
            DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1), listOf(false, false, true), batch,
        )
        val source = perExample.toKotlinSource()
        assertTrue("@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)" in source, source)
        assertTrue("matmulBatched" in source, "the batched matmul prints as its host twin:\n$source")
        checkPrinted(perExample, source, listOf(data(6, 1), data(2, 2), data(batch * 3, 3)))
    }

    @Test
    fun `printed batched logDetSpd gradients use the batched linear-algebra twins`() {
        val n = 3
        val logdet = DxirBuilder.function("logdet") {
            val a = param("a", t(n, n))
            val l = op(OpKind.CHOLESKY, listOf(a), a.type)
            val diagM = op(OpKind.TRIANGLE, listOf(l), a.type, attrs = mapOf("lower" to 0.0, "diagonal" to 1.0, "upper" to 0.0))
            val diag = op(OpKind.SUM, listOf(diagM), t(n), attrs = mapOf("reduction_dims" to listOf(1)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.LOG, listOf(diag), t(n))), t()))
        }
        val batch = 3
        val g = DxirVmapTransform.apply(DxirReverseTransform.apply(logdet), listOf(true), batch)
        val source = g.toKotlinSource()
        for (twin in listOf("choleskyBatched", "triangularSolveBatched", "scaleTrianglesBatched")) {
            assertTrue(twin in source, "missing $twin:\n$source")
        }
        val raw = data(batch * n * n, 4)
        val spd = FloatArray(batch * n * n) { k ->
            val e = k / (n * n)
            val i = (k % (n * n)) / n
            val j = k % n
            var s = 0f
            for (q in 0 until n) s += raw[e * n * n + i * n + q] * raw[e * n * n + j * n + q]
            s + if (i == j) n.toFloat() else 0f
        }
        checkPrinted(g, source, listOf(spd))
    }

    @Test
    fun `dumpGradSource covers vmap and names why a tensor lambda is not printed`() {
        val src = """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.autograd.*
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val f = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum() }
                println(f(Tensors.f32Matrix<Named<Batch, Sym>, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))).hostF32().toList())
            }
        """.trimIndent()
        val r = F64TestHarness.compileAndRun(src, pluginOptions = arrayOf("plugin:io.tlaloc.plugin:dumpGradSource=true"))
        assertEquals(0, r.exitCode, r.describe())
        assertEquals("[5.0, 25.0]", r.stdout.trim())
        val dump = r.messages.singleOrNull { "Tlaloc vmap source dump SKIPPED for 'vmap'" in it.message }
            ?: error("no vmap dump message:\n${r.describe()}")
        assertTrue("sentinel" in dump.message, dump.message)
    }

    /** Compiles [source] with a driver that calls it on [inputs] and compares raw bits with the interpreter. */
    private fun checkPrinted(fn: DxirFunction, source: String, inputs: List<FloatArray>) {
        val expected = DxirInterpreter.evalFunction(fn, inputs)
        val name = Regex("""fun (\w+)\(""").find(source)!!.groupValues[1]
        fun rankType(dims: List<Int>): String = when (dims.size) {
            0 -> "ScalarShape"
            else -> "Rank${dims.size}<${dims.joinToString(", ") { "Sym" }}>"
        }
        val args = fn.params.mapIndexed { i, p ->
            "DTensor<${rankType(p.type.dims)}, F32>(HostF32Storage(floatArrayOf(${inputs[i].joinToString(", ") { "${it}f" }})), " +
                "intArrayOf(${p.type.dims.joinToString(", ")}), F32)"
        }
        val outs = if (fn.returns.size == 1) "listOf<DTensor<*, F32>>($name(${args.joinToString(", ")}))"
        else "$name(${args.joinToString(", ")}).toList() as List<DTensor<*, F32>>"
        val driver = """
            import io.tlaloc.core.*
            @Suppress("UNCHECKED_CAST")
            fun main() {
                for (t in $outs) println(t.hostF32().joinToString(" ") { it.toRawBits().toString() })
            }
        """.trimIndent()
        val result = compileAndRun(source, driver)
        assertEquals(0, result.exitCode, "printed source failed:\n${result.messages}\n--- source ---\n$source")
        val lines = result.stdout.trim().lines()
        assertEquals(expected.size, lines.size, result.stdout)
        for (i in expected.indices) {
            assertEquals(
                expected[i].map { it.toRawBits() },
                lines[i].trim().split(" ").map { it.toInt() },
                "output $i of the printed function must be bit-identical to the interpreter\n$source",
            )
        }
    }

    private data class RunResult(val exitCode: Int, val messages: List<String>, val stdout: String)

    /** Compiles and runs without the plugin: the printed source is plain Kotlin over `:core`. */
    private fun compileAndRun(printed: String, driver: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-printed-vmap").toFile()
        try {
            File(tempDir, "Printed.kt").writeText(printed)
            File(tempDir, "Main.kt").writeText(driver)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<String>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = false
                override fun report(severity: CompilerMessageSeverity, message: String, location: CompilerMessageSourceLocation?) {
                    if (severity == CompilerMessageSeverity.ERROR) collected += "${location?.line}:${location?.column} $message"
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(PrintStream(baos, true, Charsets.UTF_8))
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                RunResult(2, collected + "RUN FAILURE: ${t.cause ?: t}", baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
