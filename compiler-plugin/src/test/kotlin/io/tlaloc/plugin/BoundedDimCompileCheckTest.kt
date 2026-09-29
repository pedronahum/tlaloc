package io.tlaloc.plugin

import io.tlaloc.core.DimBound
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
 * A bound compiled into another module (here: this test source set, which the compiled
 * snippets see only as class files). Its `max` is not in FIR, so constant sizes against
 * it are not checked at compile time.
 */
object BinaryBoundForTest : DimBound(4)

/**
 * Compile-time checks for bounded dimensions (docs/design/bounded-dims.md): constant sizes
 * over a `Bounded<B>` axis's bound, invalid bounds, and mixing bounds, each an error at the
 * call's file, line and column.
 */
class BoundedDimCompileCheckTest {

    @Test
    fun `a constant size over the bound is an error at the call`() {
        val src = source(
            """
            object MaxSeq : DimBound(4096)
            fun main() {
                val ok = Tensors.f32Zeros<Named<SeqLen, Bounded<MaxSeq>>, Sym>(4096, 2)
                val t = Tensors.f32Zeros<Named<SeqLen, Bounded<MaxSeq>>, Sym>(5000, 2)
                println(ok.dims.size + t.dims.size)
            }
            """,
        )
        val r = compile(src)
        assertTrue(r.exitCode != 0, r.render())
        val err = r.errors().single()
        assertTrue("bounded dimension exceeded" in err.message, r.render())
        assertTrue("axis 0 is Bounded<MaxSeq>, of size 1..4096, but `rows` is 5000" in err.message, r.render())
        assertEquals("Main.kt", err.file)
        assertEquals(lineOf(src) { "(5000, 2)" in it }, err.line)
        assertEquals(columnOf(src, "Tensors.f32Zeros<Named<SeqLen, Bounded<MaxSeq>>, Sym>(5000"), err.column)
    }

    @Test
    fun `sizes at the bound, below it, and not constant compile`() {
        val src = source(
            """
            object MaxSeq : DimBound(8)
            fun main(args: Array<String>) {
                val n = args.size + 100
                val a = Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(8, 2)
                val b = Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(1, 2)
                val c = Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(n, 2)
                val d = Tensors.f32Zeros<Sym, Sym>(5000, 2)
                println(a.dims.size + b.dims.size + c.dims.size + d.dims.size)
            }
            """,
        )
        val r = compile(src)
        assertEquals(0, r.exitCode, r.render())
    }

    @Test
    fun `constant expressions, size zero, later axes, vectors and the DTensor constructor are checked`() {
        val src = source(
            """
            object MaxSeq : DimBound(8)
            const val TOO_MANY = 4 * 3
            fun main() {
                val a = Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(TOO_MANY, 2)
                val b = Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(0, 2)
                val c = Tensors.f32Tensor3<Sym, Sym, Named<SeqLen, Bounded<MaxSeq>>>(1, 1, 9, FloatArray(9))
                val d = Tensors.f32Vector<Bounded<MaxSeq>>(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f))
                val e = Tensors.f32Vector<Bounded<MaxSeq>>(FloatArray(10))
                val f: DTensor<Rank2<Bounded<MaxSeq>, Sym>, F32> =
                    DTensor(HostF32Storage(FloatArray(18)), intArrayOf(9, 2), F32)
                println(listOf(a, b, c, d, e, f).size)
            }
            """,
        )
        val r = compile(src)
        val messages = r.errors().map { it.message }
        assertEquals(6, messages.size, r.render())
        assertTrue(messages.any { "`rows` is 12" in it }, r.render())
        assertTrue(messages.any { "`rows` is 0" in it }, r.render())
        assertTrue(messages.any { "axis 2 is Bounded<MaxSeq>, of size 1..8, but `d2` is 9" in it }, r.render())
        assertTrue(messages.any { "`data` is 9" in it }, r.render())
        assertTrue(messages.any { "`data` is 10" in it }, r.render())
        assertTrue(messages.any { "`dims[0]` is 9" in it }, r.render())
    }

    @Test
    fun `the F64 factories are checked like the F32 ones`() {
        val src = source(
            """
            object MaxSeq : DimBound(8)
            fun main() {
                val a = Tensors.f64Zeros<Bounded<MaxSeq>, Sym>(12, 2)
                val c = Tensors.f64Tensor3<Sym, Sym, Named<SeqLen, Bounded<MaxSeq>>>(1, 1, 9, DoubleArray(9))
                val d = Tensors.f64Vector<Bounded<MaxSeq>>(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
                val e = Tensors.f64Vector<Bounded<MaxSeq>>(DoubleArray(10))
                val ok = Tensors.f64Vector<Bounded<MaxSeq>>(DoubleArray(8))
                println(listOf(a, c, d, e, ok).size)
            }
            """,
        )
        val r = compile(src)
        val messages = r.errors().map { it.message }
        assertEquals(4, messages.size, r.render())
        assertTrue(messages.any { "`rows` is 12" in it }, r.render())
        assertTrue(messages.any { "axis 2 is Bounded<MaxSeq>, of size 1..8, but `d2` is 9" in it }, r.render())
        assertTrue(messages.any { "`data` is 9" in it }, r.render())
        assertTrue(messages.any { "`data` is 10" in it }, r.render())
    }

    @Test
    fun `a bound from another compiled module is not checked at compile time`() {
        val src = source(
            """
            import io.tlaloc.plugin.BinaryBoundForTest
            fun main() {
                val t = Tensors.f32Zeros<Bounded<BinaryBoundForTest>, Sym>(5, 2)
                println(t.dims.size)
            }
            """,
        )
        assertEquals(0, compile(src).exitCode)
    }

    @Test
    fun `specOf with the wrong number of fixed sizes is an error at the call`() {
        val src = source(
            """
            import io.tlaloc.autograd.specOf
            object MaxSeq : DimBound(8)
            fun main() {
                val ok = specOf<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Named<Hidden, Sym>>>(F32, 16)
                val tooMany = specOf<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Sym>>(F32, 16, 4)
                val zero = specOf<Rank2<Bounded<MaxSeq>, Sym>>(F32, 0)
                val none = specOf<Rank2<Bounded<MaxSeq>, Sym>>(F32)
                val spread = specOf<Rank3<Bounded<MaxSeq>, Sym, Sym>>(F32, *intArrayOf(16, 32))
                val allBounded = specOf<Rank1<Bounded<MaxSeq>>>(F32)
                println(listOf(ok, tooMany, zero, none, spread, allBounded).size)
            }
            """,
        )
        val r = compile(src, SPEC_STUB)
        val errs = r.errors()
        assertEquals(3, errs.size, r.render())
        assertTrue(errs.any { it.line == lineOf(src) { "val none" in it } && "0 given" in it.message }, r.render())
        val tooMany = errs.single { it.line == lineOf(src) { "val tooMany" in it } }
        assertTrue(
            "bounded spec mismatch: the shape has 2 axes, 1 of them bounded, so it takes 1 fixed size(s), one " +
                "per unbounded axis in order; 2 given" in tooMany.message,
            r.render(),
        )
        assertTrue(errs.any { it.line == lineOf(src) { "val zero" in it } && "fixed size 1 is 0" in it.message }, r.render())
    }

    @Test
    fun `a bound below one is an error at its declaration`() {
        val src = source(
            """
            object Empty : DimBound(0)
            object Fine : DimBound(1)
            fun main() { println(Fine.max) }
            """,
        )
        val r = compile(src)
        val err = r.errors().single()
        assertTrue("invalid dimension bound: Empty passes 0 to DimBound; a bound must be at least 1" in err.message, r.render())
        assertEquals(lineOf(src) { "object Empty" in it }, err.line)
    }

    @Test
    fun `mixing two bounds in a broadcasting elementwise op is an error at the call`() {
        val src = source(
            """
            object MaxA : DimBound(8)
            object MaxB : DimBound(8)
            fun main() {
                val a = Tensors.f32Zeros<Bounded<MaxA>, Sym>(2, 2)
                val b = Tensors.f32Zeros<Bounded<MaxB>, Sym>(2, 2)
                val s = Tensors.f32Zeros<Sym, Sym>(2, 2)
                val t = Tensors.f32Zeros<Sym, Bounded<MaxA>>(2, 2)
                val v = Tensors.f32Vector<Named<Hidden, Bounded<MaxB>>>(floatArrayOf(1f, 2f))
                val same = a + a
                val withSym = a + s
                val symAgainstBound = a * v
                val mixed = a + b
                val trailing = t * v
                println(listOf(same, withSym, symAgainstBound, mixed, trailing).size)
            }
            """,
        )
        val r = compile(src)
        val errs = r.errors()
        assertEquals(2, errs.size, r.render())
        val mixed = errs.singleOrNull { it.line == lineOf(src) { "val mixed = a + b" in it } } ?: error(r.render())
        assertTrue(
            "bounded axis mismatch: `plus` aligns axis 0 of the left operand, Bounded<MaxA>, with axis 0 of the " +
                "right operand, Bounded<MaxB>" in mixed.message,
            r.render(),
        )
        assertEquals(columnOf(src, "a + b"), mixed.column)
        val trailing = errs.singleOrNull { it.line == lineOf(src) { "val trailing = t * v" in it } } ?: error(r.render())
        assertTrue(
            "`times` aligns axis 1 of the left operand, Bounded<MaxA>, with axis 0 of the right operand, " +
                "Bounded<MaxB>" in trailing.message,
            r.render(),
        )
    }

    @Test
    fun `mixing two bounds, or a bound and Sym, in matmul and contract is a type error at the call`() {
        val src = source(
            """
            object MaxA : DimBound(8)
            object MaxB : DimBound(8)
            fun main() {
                val a = Tensors.f32Zeros<Bounded<MaxA>, Sym>(2, 2)
                val b = Tensors.f32Zeros<Bounded<MaxB>, Sym>(2, 2)
                val s = Tensors.f32Zeros<Sym, Sym>(2, 2)
                val x = Tensors.f32Zeros<Named<SeqLen, Bounded<MaxA>>, Named<Hidden, Sym>>(2, 2)
                val y = Tensors.f32Zeros<Named<SeqLen, Bounded<MaxB>>, Named<Hidden, Sym>>(2, 2)
                val q = Tensors.f32Zeros<Named<Batch, Sym>, Named<SeqLen, Bounded<MaxA>>>(2, 2)
                val prod = q matmul b
                val prodSym = q matmul s
                val c = q contract y
                println(listOf(prod, prodSym, c, x, a).size)
            }
            """,
        )
        val r = compile(src)
        assertTrue(r.exitCode != 0, r.render())
        val lines = r.errors().map { it.line }.toSet()
        for (anchor in listOf("val prod = q matmul b", "val prodSym = q matmul s", "val c = q contract y")) {
            assertTrue(lineOf(src) { anchor in it } in lines, "no error on `$anchor`:\n${r.render()}")
        }
        assertTrue(r.errors().all { it.file == "Main.kt" && it.column != null }, r.render())
    }

    @Test
    fun `the bounded atoms require the experimental opt-in`() {
        val src = """
            import io.tlaloc.core.DimBound
            object MaxSeq : DimBound(8)
            fun main() { println(MaxSeq.max) }
        """.trimIndent()
        val r = compile(src)
        assertTrue(r.exitCode != 0, r.render())
        assertTrue(r.errors().any { "ExperimentalTlalocApi" in it.message || "opt-in" in it.message.lowercase() }, r.render())
    }

    @Test
    fun `grad over a bounded parameter lowers and runs at every size up to the bound`() {
        val src = source(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            import io.tlaloc.core.ops.times
            object MaxSeq : DimBound(6)
            fun main() {
                val g = grad { x: DTensor<Rank1<Named<SeqLen, Bounded<MaxSeq>>>, F32> -> (x * x).sum().toFloat() }
                for (n in 1..6) {
                    val x = Tensors.f32Vector<Named<SeqLen, Bounded<MaxSeq>>>(FloatArray(n) { it + 0.5f })
                    println(g(x).hostF32().joinToString(" "))
                }
            }
            """,
        )
        val r = compileAndRun(src, GRAD_STUB)
        assertEquals(0, r.exitCode, r.render())
        assertTrue(r.messages.none { "kept original call" in it.message }, r.render())
        val rows = r.stdout.trim().lines()
        assertEquals(6, rows.size, r.stdout)
        for ((i, row) in rows.withIndex()) {
            val got = row.trim().split(" ").map { it.toFloat() }
            val want = List(i + 1) { 2f * (it + 0.5f) }
            assertEquals(want, got, "size ${i + 1}")
        }
    }

    // ------------------------------------------------------------------ harness

    private data class Message(
        val severity: CompilerMessageSeverity,
        val message: String,
        val file: String?,
        val line: Int?,
        val column: Int?,
    )

    private data class Result(val exitCode: Int, val messages: List<Message>, val stdout: String = "") {
        fun errors() = messages.filter { it.severity == CompilerMessageSeverity.ERROR }
        fun render() = messages.filter { it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING }.joinToString("\n") { "[${it.severity}] ${it.file}:${it.line}:${it.column} ${it.message}" } +
            if (stdout.isNotEmpty()) "\nstdout:\n$stdout" else ""
    }

    private fun source(body: String): String =
        (PRELUDE + "\n" + body.trimIndent()).trimIndent()

    private fun lineOf(source: String, predicate: (String) -> Boolean): Int {
        val matches = source.lines().withIndex().filter { predicate(it.value) }
        require(matches.size == 1) { "anchor matched ${matches.size} lines" }
        return matches.single().index + 1
    }

    private fun columnOf(source: String, fragment: String): Int {
        val line = source.lines().single { fragment in it }
        return line.indexOf(fragment) + 1
    }

    private fun compile(user: String, stub: String? = null): Result = compileAndMaybeRun(user, stub, run = false)

    private fun compileAndRun(user: String, stub: String): Result = compileAndMaybeRun(user, stub, run = true)

    private fun compileAndMaybeRun(user: String, stub: String?, run: Boolean): Result {
        val tempDir = Files.createTempDirectory("tlaloc-bounded-dims").toFile()
        try {
            if (stub != null) File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<Message>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    collected += Message(severity, message, location?.path?.substringAfterLast('/'), location?.line, location?.column)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = arrayOf(
                    System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
                    System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
                    System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
                )
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0 || !run) return Result(exitCode, collected)
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(PrintStream(baos, true, Charsets.UTF_8))
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                Result(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                Result(2, collected + Message(CompilerMessageSeverity.ERROR, "RUN FAILURE: ${t.cause ?: t}", null, null, null), baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private companion object {
        val PRELUDE = """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.core.Batch
            import io.tlaloc.core.Bounded
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.DimBound
            import io.tlaloc.core.F32
            import io.tlaloc.core.Hidden
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Named
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Rank3
            import io.tlaloc.core.SeqLen
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.contract
            import io.tlaloc.core.ops.matmul
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.times
        """.trimIndent()

        val SPEC_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DType
            import io.tlaloc.core.Shape
            inline fun <reified S : Shape> specOf(dtype: DType, vararg fixedSizes: Int): Int = fixedSizes.size
        """.trimIndent()

        val GRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Shape
            fun <S : Shape> grad(f: (DTensor<S, F32>) -> Float): (DTensor<S, F32>) -> DTensor<S, F32> =
                { _ -> error("stub: the plugin did not rewrite this call") }
        """.trimIndent()
    }
}
