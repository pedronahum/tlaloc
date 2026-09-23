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
 * §0.4.505 (Tier 4, item 4) — `@ExperimentalTlalocApi` REFUSES A CONSUMER, and the
 * refusal is certified by compiling one.
 *
 * `:core`'s `ExperimentalTlalocApiTest` pins the marker's own properties by
 * reflection. That is not the claim that matters. The claim that matters is that a
 * stranger's file touching a provisional API stops compiling and is told why — and
 * nothing short of a second compilation can establish it, because Tlaloc's own
 * modules pass `-opt-in=io.tlaloc.core.ExperimentalTlalocApi` (root
 * `build.gradle.kts`) and therefore cannot observe the requirement at all.
 *
 * This harness runs a real `K2JVMCompiler` over source strings with **no** opt-in
 * flag — the configuration a consumer is in — and asserts:
 *
 *  - the four-worlds taxonomy and the kernel-choice surface refuse by name;
 *  - `@OptIn` makes the same source compile, unchanged otherwise;
 *  - a CERTIFIED surface (`DTensor`, `Tensors`, `MeshSpec`) compiles with no opt-in
 *    and no mention of the marker — the counter-assertion that keeps the annotation
 *    from being a blanket nobody reads.
 *
 * The Tlaloc K2 plugin is deliberately NOT on the classpath here: opt-in is
 * Kotlin's own checker, and a test that needed our plugin to see it would be
 * testing the wrong thing.
 */
class ExperimentalApiOptInTest {

    @Test
    fun `the four-worlds taxonomy refuses a consumer that has not opted in`() {
        val result = compile(FOUR_WORLDS_NO_OPT_IN)
        assertTrue(
            result.exitCode != 0,
            "touching @ExperimentalTlalocApi without opting in must FAIL the compile, not " +
                "warn; got exit ${result.exitCode}:\n${result.render()}",
        )
        val errors = result.messages.filter { it.severity == CompilerMessageSeverity.ERROR }
        assertTrue(
            errors.any { "ExperimentalTlalocApi" in it.message },
            "the error must name the marker so the reader knows what to opt into; got:\n" +
                result.render(),
        )
        assertTrue(
            errors.any { "docs/CAPABILITIES.md" in it.message },
            "the error must carry the marker's own message, which points at the file that " +
                "grades the capability; got:\n${result.render()}",
        )
    }

    @Test
    fun `OptIn makes the identical four-worlds program compile`() {
        val result = compile(FOUR_WORLDS_WITH_OPT_IN)
        assertEquals(
            0, result.exitCode,
            "@OptIn(ExperimentalTlalocApi::class) must be sufficient — nothing else changed " +
                "between this source and the refused one; got:\n${result.render()}",
        )
    }

    @Test
    fun `the kernel-choice surface refuses a consumer that has not opted in`() {
        val result = compile(KERNEL_CHOICE_NO_OPT_IN)
        assertTrue(
            result.exitCode != 0,
            "io.tlaloc.ir.recognizer.kernel is marked; got exit ${result.exitCode}:\n" +
                result.render(),
        )
        assertTrue(
            result.messages.any {
                it.severity == CompilerMessageSeverity.ERROR &&
                    "ExperimentalTlalocApi" in it.message
            },
            "the refusal must name the marker; got:\n${result.render()}",
        )
    }

    @Test
    fun `the collective attribute convention refuses a consumer that has not opted in`() {
        val result = compile(ALL_REDUCE_NO_OPT_IN)
        assertTrue(
            result.exitCode != 0,
            "AllReduceAttrs is marked because distributed execution is DESIGNED, never run; " +
                "got exit ${result.exitCode}:\n${result.render()}",
        )
    }

    @Test
    fun `a certified surface compiles with no opt-in and no mention of the marker`() {
        val result = compile(CERTIFIED_SURFACE)
        assertEquals(
            0, result.exitCode,
            "the tensor and op surface is a certified row in docs/CAPABILITIES.md and must " +
                "NOT require opt-in; got:\n${result.render()}",
        )
        assertTrue(
            result.messages.none { "ExperimentalTlalocApi" in it.message },
            "not one word about the marker may appear when a consumer uses the certified " +
                "surface — that is what stops the annotation becoming noise; got:\n" +
                result.render(),
        )
    }

    // ---------------- harness ----------------

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class CompileResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
    ) {
        fun render(): String = messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun compile(source: String): CompileResult {
        val tempDir = Files.createTempDirectory("tlaloc-optin-test").toFile()
        try {
            File(tempDir, "Main.kt").writeText(source)
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
                    collected += CompileMessage(severity, message)
                }
            }

            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                destination = outDir.absolutePath
                // :core and :ir are on the test JVM's classpath (both are
                // `implementation` dependencies of :compiler-plugin), which is exactly
                // what a consumer resolving io.github.pedronahum:tlaloc-core + io.github.pedronahum:tlaloc-ir would have.
                // NOTE what is absent: any `-opt-in=` argument.
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
        private val FOUR_WORLDS_NO_OPT_IN = """
            import io.tlaloc.core.Tlaloc
            import io.tlaloc.core.orchestrationMarker
            fun main() { println(Tlaloc.orchestrationMarker()) }
        """.trimIndent()

        private val FOUR_WORLDS_WITH_OPT_IN = """
            @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
            import io.tlaloc.core.Tlaloc
            import io.tlaloc.core.orchestrationMarker
            fun main() { println(Tlaloc.orchestrationMarker()) }
        """.trimIndent()

        private val KERNEL_CHOICE_NO_OPT_IN = """
            import io.tlaloc.ir.recognizer.kernel.KernelTarget
            fun main() { println(KernelTarget(vendor = "nvidia", arch = "gb10")) }
        """.trimIndent()

        private val ALL_REDUCE_NO_OPT_IN = """
            import io.tlaloc.ir.AllReduceAttrs
            fun main() { println(AllReduceAttrs) }
        """.trimIndent()

        /** `grad` itself needs the plugin, so the certified-surface control uses the
         * tensor/op/mesh types — three ✅ rows — which need nothing but the jars. */
        private val CERTIFIED_SURFACE = """
            import io.tlaloc.core.MeshSpec
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            fun main() {
                val t = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f))
                println(t.dims.toList())
                println(MeshSpec.of("dp" to 2))
            }
        """.trimIndent()
    }
}
