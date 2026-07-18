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
 * §0.4.353 — Meta-stage ergonomics: compile-time ERRORS (not warnings)
 * for named-index misuse and undifferentiable grad bodies, surfaced by
 * the K2 checker exactly where the IDE draws red squiggles. The Meta
 * Software-2.0 post's demo — shape errors before you run — is the
 * named-axis level in Tlaloc (dims are symbolic at lowering time; the
 * literal-dims validator is pinned in :ir's DxirShapeValidationTest and
 * wired here for when concrete-dim surfaces appear).
 */
class TlalocCompileTimeErrorTest {

    @Test
    fun `disjoint-name contract is a compile ERROR`() {
        val result = compile(
            stub = STUB_GRAD2_GENERIC,
            user = """
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
            """.trimIndent(),
        )
        assertTrue(result.exitCode != 0, "expected compilation to FAIL:\n${result.renderMessages()}")
        val errors = result.messages.filter { it.severity == CompilerMessageSeverity.ERROR }
        assertTrue(
            errors.any { "named-index mismatch" in it.message && "share no named axis" in it.message },
            "expected NAMED_INDEX_MISMATCH error, got:\n${result.renderMessages()}",
        )
    }

    @Test
    fun `valid contract still compiles with no Tlaloc errors`() {
        val result = compile(
            stub = STUB_GRAD2_GENERIC,
            user = """
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
                        b: DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> ->
                        (a contract b).sum()
                    }
                    println(g)
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.renderMessages()}")
        assertTrue(
            result.messages.none { it.severity == CompilerMessageSeverity.ERROR },
            "no errors expected:\n${result.renderMessages()}",
        )
    }

    // ---------------------------------------------------------------------
    // Harness (same shape as the sibling checker tests).
    // ---------------------------------------------------------------------

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>) {
        fun renderMessages(): String =
            messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun compile(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-compile-error").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
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
            return RunResult(exitCode, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private companion object {
        val STUB_GRAD2_GENERIC = """
            package io.tlaloc.autograd
            fun <A, B, R> grad2(f: (A, B) -> R): (A, B) -> Pair<A, B> = { a, b -> a to b }
        """.trimIndent()
    }
}
