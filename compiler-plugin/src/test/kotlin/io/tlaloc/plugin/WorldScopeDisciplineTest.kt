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
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — compile-fail tests for the four-worlds discipline.
 *
 * Each test compiles a Kotlin source snippet and asserts that the
 * compiler either accepts (legal in-scope call) or rejects (cross-scope
 * call) the program. The discipline is enforced by Kotlin's native
 * receiver-only resolution + DslMarker, *not* by any plugin extension —
 * so these tests don't need the Tlaloc plugin on the classpath. They run
 * the K2 compiler purely against `:core`.
 *
 * The compile-fail style mirrors `ContractInferenceTest.disjoint named
 * axes...` from Layer 1 — assert `exitCode != 0` and inspect the error
 * message for the expected "unresolved reference" / type-mismatch text.
 */
class WorldScopeDisciplineTest {

    @Test
    fun `kernel-only op called from kernel scope compiles`() {
        val result = compile(
            """
            import io.tlaloc.core.KernelScope
            import io.tlaloc.core.kernelMarker
            fun main() {
                val k = object : KernelScope {}
                println(k.kernelMarker())
            }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "expected legal compile; got:\n${result.renderMessages()}")
    }

    @Test
    fun `kernel-only op called outside kernel scope fails to compile`() {
        // Refined Option A's central guarantee for world discipline:
        // Kernel-only ops are extension functions on KernelScope. Calling
        // one without a KernelScope receiver in scope is an unresolved-
        // reference error from the Kotlin compiler. No plugin diagnostic
        // needed.
        val result = compile(
            """
            import io.tlaloc.core.kernelMarker
            fun main() {
                // No KernelScope receiver in scope — this should not compile.
                val s = kernelMarker()
                println(s)
            }
            """.trimIndent(),
        )
        assertNotEquals(0, result.exitCode, "expected compile failure; got success.\n${result.renderMessages()}")
        val errorText = result.messages
            .filter { it.severity == CompilerMessageSeverity.ERROR }
            .joinToString("\n") { it.message }
        assertTrue(
            "unresolved" in errorText.lowercase() || "kernelMarker" in errorText,
            "expected an unresolved-reference error mentioning kernelMarker, got:\n$errorText",
        )
    }

    @Test
    fun `orchestration op called from kernel scope fails to compile`() {
        // Inside `with(kernelScope)`, the implicit receiver is KernelScope
        // — which doesn't know about orchestrationMarker (an op declared
        // on OrchestrationScope). This is the "calling dispatch from
        // Kernel scope" adversarial case.
        val result = compile(
            """
            import io.tlaloc.core.KernelScope
            import io.tlaloc.core.orchestrationMarker
            fun main() {
                with(object : KernelScope {}) {
                    // OrchestrationScope is not the implicit receiver here.
                    println(orchestrationMarker())
                }
            }
            """.trimIndent(),
        )
        assertNotEquals(0, result.exitCode, "expected compile failure for cross-scope call")
        val errorText = result.messages
            .filter { it.severity == CompilerMessageSeverity.ERROR }
            .joinToString("\n") { it.message }
        assertTrue(
            "unresolved" in errorText.lowercase() || "orchestrationMarker" in errorText,
            "expected unresolved-reference error mentioning orchestrationMarker, got:\n$errorText",
        )
    }

    @Test
    fun `Tlaloc singleton opens both Orchestration and Program scopes`() {
        // The Tlaloc default singleton implements both OrchestrationScope
        // and ProgramScope, so within `with(Tlaloc) { … }` both
        // orchestrationMarker and programMarker resolve.
        val result = compile(
            """
            import io.tlaloc.core.Tlaloc
            import io.tlaloc.core.orchestrationMarker
            import io.tlaloc.core.programMarker
            fun main() {
                with(Tlaloc) {
                    println(orchestrationMarker())
                    println(programMarker())
                }
            }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "expected legal compile; got:\n${result.renderMessages()}")
    }

    @Test
    fun `cluster-only op outside cluster scope fails to compile`() {
        val result = compile(
            """
            import io.tlaloc.core.clusterMarker
            fun main() {
                println(clusterMarker())
            }
            """.trimIndent(),
        )
        assertNotEquals(0, result.exitCode, "expected compile failure for cluster-op without scope")
    }

    // ----- helpers (no plugin classpath; this is a pure :core compile test) ---

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>) {
        fun renderMessages(): String =
            messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun compile(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-worlds").toFile()
        try {
            File(tempDir, "Main.kt").writeText(OPT_IN + user)
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

    private companion object {
        /**
         * §0.4.505 — every source here opts in, and it matters that ALL of them do.
         *
         * The four-worlds taxonomy now carries `@ExperimentalTlalocApi`
         * (`@RequiresOptIn(ERROR)`), so without this line the two positive tests
         * would fail and — worse — the three NEGATIVE tests would keep passing for
         * the wrong reason: they assert `exitCode != 0`, and an opt-in error is a
         * non-zero exit that has nothing to do with scope discipline. Prepending it
         * in the harness rather than in each snippet is what makes "all of them"
         * checkable by reading one place.
         *
         * This harness runs `K2JVMCompiler` directly and therefore does NOT inherit
         * the `-opt-in=` flag the root build passes to Tlaloc's own compilations —
         * which is the same reason `ExperimentalApiOptInTest` can certify the
         * refusal at all.
         */
        private val OPT_IN = "@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)\n"
    }
}
