package io.tlaloc.plugin

import java.io.File
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
 * §0.4.505 (Tier 4, item 6) — WHERE the compiler points, pinned.
 *
 * The README and `docs/GETTING_STARTED.md` have promised, since §0.4.353, that a
 * shape or differentiability mistake is "a red squiggle in the IDE". Nothing tested
 * it, and by this repository's own house rule a claim ships with a test or it does
 * not ship. Two things are worth separating, because only one of them is testable
 * from here:
 *
 *  - **What this file certifies:** the diagnostic carries a SOURCE POSITION, and that
 *    position is the offending construct's own file, line and column — not the file
 *    header, not the `main` function, not `(no location)`. Eighteen test classes
 *    already assert that the right diagnostic is produced with the right text;
 *    **none of them looked at `location` at all**, which is why the stronger half of
 *    the claim was unsupported.
 *  - **What nothing here certifies:** IntelliJ. A squiggle is the IDE's rendering of
 *    a diagnostic its own K2 analysis produced by running these same FIR checkers in
 *    its own process, so the position below is the position a redline would land on
 *    — but that inference is not a test, no test in this repository drives an IDE,
 *    and the docs now say "expected, untested" instead of asserting it.
 *
 * The positions are asserted against lines located by their own text rather than by
 * a hardcoded number, so adding a line to a test source cannot silently turn this
 * green.
 */
class DiagnosticSourcePositionTest {

    @Test
    fun `an unlowerable body reports at the grad call, with a line and a column`() {
        val source = """
            import io.tlaloc.autograd.grad
            fun cube(x: Float): Float = x * x * x
            fun main() {
                val g = grad { x: Float -> cube(x) }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compile(SCALAR_STUB, source)
        assertTrue(result.exitCode != 0, "expected a refusal; got:\n${result.render()}")

        val err = result.messages.single {
            it.severity == CompilerMessageSeverity.ERROR && "could not lower this lambda" in it.message
        }
        val expectedLine = lineOf(source) { "val g = grad {" in it }
        assertEquals(
            expectedLine, err.line,
            "the error must point at the line the offending grad {} is written on, which is " +
                "what an IDE would underline. Got line ${err.line} for:\n${source.numbered()}",
        )
        assertEquals(
            "Main.kt", err.file,
            "the position must name the USER's file, not a stub or a synthesized one; got ${err.file}",
        )
        assertTrue(
            err.column != null && err.column > 0,
            "a line without a column underlines the whole line; got column ${err.column}",
        )
    }

    @Test
    fun `a named-index mismatch reports at the offending grad2 call`() {
        val source = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.Batch
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Hidden
            import io.tlaloc.core.Named
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.SeqLen
            import io.tlaloc.core.Sym
            import io.tlaloc.core.ops.contract
            import io.tlaloc.core.ops.sum
            fun main() {
                val g = grad2 {
                    a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
                    b: DTensor<Rank2<Named<Hidden, Sym>, Named<Hidden, Sym>>, F32> ->
                    (a contract b).sum()
                }
                println(g)
            }
        """.trimIndent()
        val result = compile(GRAD2_STUB, source)
        assertTrue(result.exitCode != 0, "expected a refusal; got:\n${result.render()}")

        val err = result.messages.single {
            it.severity == CompilerMessageSeverity.ERROR && "named-index mismatch" in it.message
        }
        val expectedLine = lineOf(source) { "val g = grad2 {" in it }
        assertEquals(
            expectedLine, err.line,
            "NAMED_INDEX_MISMATCH must land on the grad2 {} call — this is the exact claim the " +
                "README's named-axes section makes. Got line ${err.line} for:\n${source.numbered()}",
        )
        assertEquals("Main.kt", err.file, "got ${err.file}")
    }

    @Test
    fun `no Tlaloc diagnostic is ever emitted without a position`() {
        // A positionless diagnostic is the failure mode that makes "red squiggle"
        // false while every text assertion still passes: the IDE has nowhere to draw.
        val sources = listOf(
            SCALAR_STUB to """
                import io.tlaloc.autograd.grad
                fun cube(x: Float): Float = x * x * x
                fun main() { println(grad { x: Float -> cube(x) }(1.0f)) }
            """.trimIndent(),
            SCALAR_STUB to """
                import io.tlaloc.autograd.grad
                fun square(x: Float): Float = x * x
                fun main() { println(grad(::square)(2.0f)) }
            """.trimIndent(),
        )
        sources.forEach { (stub, user) ->
            val result = compile(stub, user)
            val tlalocDiagnostics = result.messages.filter {
                it.severity == CompilerMessageSeverity.ERROR && "Tlaloc" in it.message
            }
            assertTrue(
                tlalocDiagnostics.isNotEmpty(),
                "this source is supposed to be refused; got:\n${result.render()}",
            )
            tlalocDiagnostics.forEach {
                assertTrue(
                    it.line != null && it.line > 0 && it.file != null,
                    "a Tlaloc error with no source position cannot be rendered as a squiggle: " +
                        "${it.message}",
                )
            }
        }
    }

    // ---------------- harness ----------------

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
        val file: String?,
        val line: Int?,
        val column: Int?,
    )

    private data class CompileResult(val exitCode: Int, val messages: List<CompileMessage>) {
        fun render(): String = messages.joinToString("\n") {
            "[${it.severity}] ${it.file}:${it.line}:${it.column} ${it.message}"
        }
    }

    /** The 1-based line number of the single line matching [predicate]. */
    private fun lineOf(source: String, predicate: (String) -> Boolean): Int {
        val matches = source.lines().withIndex().filter { predicate(it.value) }
        require(matches.size == 1) {
            "the anchor must match exactly one line or the assertion means nothing; matched " +
                "${matches.size} in:\n${source.numbered()}"
        }
        return matches.single().index + 1
    }

    private fun String.numbered(): String =
        lines().withIndex().joinToString("\n") { (i, l) -> "${i + 1}: $l" }

    private fun compile(stub: String, user: String): CompileResult {
        val tempDir = Files.createTempDirectory("tlaloc-position-test").toFile()
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
                    collected += CompileMessage(
                        severity,
                        message,
                        location?.path?.substringAfterLast('/'),
                        location?.line,
                        location?.column,
                    )
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
            return CompileResult(exitCode, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private companion object {
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = f
        """.trimIndent()

        private val GRAD2_STUB = """
            package io.tlaloc.autograd
            fun <A, B, R> grad2(f: (A, B) -> R): (A, B) -> Pair<A, B> = { a, b -> a to b }
        """.trimIndent()
    }
}
