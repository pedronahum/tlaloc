package io.tlaloc.plugin

import io.tlaloc.ir.DxirFunction
import java.io.File
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
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.514 — the release-readiness review's compiler-plugin correctness items.
 *
 *  - R1: the FIR→IR handoff is keyed by file AND range, and is one table per
 *    compilation. Two `grad {}` calls at identical offsets in two files used to share
 *    a key: file A's call site got file B's gradient.
 *  - R2: every IR-phase "kept original call" follows `strictLowering` — an ERROR by
 *    default, a WARNING only when the build opts out.
 *  - R3: an unexpected exception inside the plugin is a named diagnostic with the
 *    issue-tracker address, not an internal compiler error.
 *  - R4: `dumpGradSource` parses strictly, like every other boolean option.
 */
class PluginRobustnessTest {

    // ---------------- R1: the handoff key ----------------

    /** Both bodies are 9 characters, and both files have identical text up to the
     * lambda, so the two `grad {}` calls sit at the SAME start and end offsets. */
    private val fileA = """
        package a
        import io.tlaloc.autograd.grad
        fun gradOf(): (Float) -> Float = grad { x: Float -> x * x + x }
    """.trimIndent()

    private val fileB = """
        package b
        import io.tlaloc.autograd.grad
        fun gradOf(): (Float) -> Float = grad { x: Float -> x * x * x }
    """.trimIndent()

    @Test
    fun `two grad calls at identical offsets in two files each get their own gradient`() {
        val gradCallA = fileA.indexOf("grad {")
        assertEquals(gradCallA, fileB.indexOf("grad {"), "the test needs identical offsets")
        assertEquals(fileA.length, fileB.length, "the test needs identical offsets")

        val result = compile(
            files = mapOf("stub.kt" to SCALAR_STUB, "A.kt" to fileA, "B.kt" to fileB),
            keepOutput = true,
        )
        try {
            assertEquals(0, result.exitCode, "compile failed:\n${result.render()}")
            URLClassLoader(arrayOf(result.outDir.toURI().toURL()), javaClass.classLoader).use { loader ->
                val x = 2.0f
                val da = gradAt(loader, "a.AKt", x)
                val db = gradAt(loader, "b.BKt", x)
                // d/dx (x² + x) = 2x + 1 = 5 ; d/dx x³ = 3x² = 12. The stub's fallback
                // (identity on the lambda) would return f(2) = 6 / 8 instead.
                assertEquals(5.0f, da, "file A's call site must get file A's gradient")
                assertEquals(12.0f, db, "file B's call site must get file B's gradient")
            }
        } finally {
            result.outDir.parentFile.deleteRecursively()
        }
    }

    @Test
    fun `the handoff table belongs to one compilation`() {
        val fn = DxirFunction("f", emptyList(), emptyList(), emptyList())
        val mine = TlalocLoweringHandoff()
        val other = TlalocLoweringHandoff()
        mine.record("/src/A.kt", 10, 20, fn)
        other.record("/src/A.kt", 10, 20, fn)
        other.clear()
        assertEquals(1, mine.size(), "clearing another compilation's table must not touch this one")
        assertNull(mine.take("/src/B.kt", 10, 20), "the same range in another file is another call")
        assertNotNull(mine.take("/src/A.kt", 10, 20))
        assertNull(mine.take("/src/A.kt", 10, 20), "take removes the entry")
    }

    /** A call the FIR phase lowered and the IR phase never matched (the two phases
     * disagreeing on its file path or offsets) would otherwise compile silently and
     * throw at its first call. */
    @Test
    fun `a lowered call the IR phase never matched is refused at compile time`() {
        val strict = withInjectedFault(TlalocInternalErrors.Phase.IR_HANDOFF_MISS) {
            compile(mapOf("stub.kt" to SCALAR_STUB, "Main.kt" to workingGrad))
        }
        assertEquals(1, strict.exitCode, "the build must fail; got ${strict.exitCode}:\n${strict.render()}")
        val err = strict.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.ERROR && "found no call at this position" in it.message
        } ?: error("expected one unclaimed-call ERROR; got:\n${strict.render()}")
        assertTrue("`grad`" in err.message, "names the intrinsic:\n${err.message}")
        assertTrue(TlalocInternalErrors.ISSUES_URL in err.message, "names the issue tracker:\n${err.message}")
        assertEquals(3, err.location?.line, "points at the grad call's line")

        val lenient = withInjectedFault(TlalocInternalErrors.Phase.IR_HANDOFF_MISS) {
            compile(
                mapOf("stub.kt" to SCALAR_STUB, "Main.kt" to workingGrad),
                options = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false"),
            )
        }
        assertEquals(0, lenient.exitCode, "strictLowering=false keeps the build green:\n${lenient.render()}")
        assertTrue(
            lenient.messages.any {
                it.severity == CompilerMessageSeverity.WARNING && "found no call at this position" in it.message
            },
            "under strictLowering=false it is a warning:\n${lenient.render()}",
        )
    }

    @Test
    fun `drainUnclaimed returns what was never taken and empties the table`() {
        val fn = DxirFunction("grad_body", emptyList(), emptyList(), emptyList())
        val h = TlalocLoweringHandoff()
        h.record("/src/A.kt", 30, 40, fn)
        h.record("/src/A.kt", 10, 20, fn)
        h.take("/src/A.kt", 30, 40)
        assertEquals(
            listOf(TlalocLoweringHandoff.Unclaimed("/src/A.kt", 10, 20, "grad_body")),
            h.drainUnclaimed(),
        )
        assertEquals(0, h.size())
    }

    // ---------------- R2: IR-phase refusals follow strictLowering ----------------

    /** `jacobian` whose assembly helper is missing from the classpath: the FIR phase
     * lowers the lambda, the IR phase cannot find `assembleJacobianForward` and keeps
     * the original call. That is the shape of a library/plugin version mismatch. */
    private val jacobianStubWithoutHelper = """
        package io.tlaloc.autograd
        fun <A, R> jacobian(f: (A) -> R): (A) -> A = { a -> a }
    """.trimIndent()

    private val jacobianUser = """
        import io.tlaloc.autograd.jacobian
        fun main() {
            val j = jacobian { x: Float -> x * x }
            println(j(2.0f))
        }
    """.trimIndent()

    @Test
    fun `an IR-phase kept original call is a compile ERROR by default`() {
        val result = compile(mapOf("stub.kt" to jacobianStubWithoutHelper, "Main.kt" to jacobianUser))
        assertTrue(result.exitCode != 0, "the build must fail; got:\n${result.render()}")
        val err = result.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.ERROR && "kept original call" in it.message
        } ?: error("expected one kept-original-call ERROR; got:\n${result.render()}")
        assertTrue("assembleJacobianForward" in err.message, "the error names the missing symbol:\n${err.message}")
        assertTrue("same version" in err.message, "a missing library symbol suggests a version mismatch:\n${err.message}")
        assertTrue("strictLowering=false" in err.message, "the error names the opt-out:\n${err.message}")
        assertNotNull(err.location, "the error points at the call site")
        assertEquals(3, err.location!!.line, "the error points at the jacobian call's line")
    }

    @Test
    fun `strictLowering=false turns the IR-phase refusal into a warning`() {
        val result = compile(
            mapOf("stub.kt" to jacobianStubWithoutHelper, "Main.kt" to jacobianUser),
            options = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false"),
        )
        assertEquals(0, result.exitCode, "the opt-out must compile; got:\n${result.render()}")
        val warn = result.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        } ?: error("expected one kept-original-call WARNING; got:\n${result.render()}")
        assertTrue("throws" in warn.message, "the warning says the call throws when it runs:\n${warn.message}")
    }

    @Test
    fun `a working jvp compiles under -Werror with no Tlaloc diagnostic`() {
        val result = compile(
            mapOf(
                "stub.kt" to """
                    package io.tlaloc.autograd
                    fun <A, R> jvp(f: (A) -> R): (A, A) -> R = { a, _ -> f(a) }
                """.trimIndent(),
                "Main.kt" to """
                    import io.tlaloc.autograd.jvp
                    fun main() {
                        val j = jvp { x: Float -> x * x }
                        println(j(2.0f, 1.0f))
                    }
                """.trimIndent(),
            ),
            werror = true,
        )
        assertEquals(0, result.exitCode, "a correct jvp {} must compile under -Werror; got:\n${result.render()}")
        assertTrue(result.messages.none { "Tlaloc" in it.message }, "no Tlaloc diagnostic:\n${result.render()}")
    }

    // ---------------- R3: unexpected exceptions ----------------

    private val workingGrad = """
        import io.tlaloc.autograd.grad
        fun main() {
            val g = grad { x: Float -> x * x }
            println(g(2.0f))
        }
    """.trimIndent()

    @Test
    fun `an unexpected exception in the FIR checker is a named error with the issue tracker`() {
        assertInternalErrorReported(TlalocInternalErrors.Phase.FIR)
    }

    @Test
    fun `an unexpected exception in the IR extension is a named error with the issue tracker`() {
        assertInternalErrorReported(TlalocInternalErrors.Phase.IR)
    }

    private fun assertInternalErrorReported(phase: TlalocInternalErrors.Phase) {
        val strict = withInjectedFault(phase) {
            compile(mapOf("stub.kt" to SCALAR_STUB, "Main.kt" to workingGrad))
        }
        assertEquals(
            1, strict.exitCode,
            "an internal error is a compilation ERROR (exit 1), not an internal compiler error (exit 3); " +
                "got ${strict.exitCode}:\n${strict.render()}",
        )
        val err = strict.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.ERROR && "Tlaloc internal error" in it.message
        } ?: error("expected one Tlaloc internal error; got:\n${strict.render()}")
        assertTrue(TlalocInternalErrors.ISSUES_URL in err.message, "names the issue tracker:\n${err.message}")
        assertTrue("injected fault" in err.message, "carries the exception's message:\n${err.message}")
        assertTrue("IllegalStateException" in err.message, "names the exception class:\n${err.message}")
        assertTrue(strict.messages.none { it.severity == CompilerMessageSeverity.EXCEPTION }, strict.render())

        val lenient = withInjectedFault(phase) {
            compile(
                mapOf("stub.kt" to SCALAR_STUB, "Main.kt" to workingGrad),
                options = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false"),
            )
        }
        assertEquals(0, lenient.exitCode, "strictLowering=false keeps the build green:\n${lenient.render()}")
        assertTrue(
            lenient.messages.any {
                it.severity == CompilerMessageSeverity.WARNING && "Tlaloc internal error" in it.message
            },
            "under strictLowering=false the internal error is a warning:\n${lenient.render()}",
        )
    }

    @Test
    fun `compiler control-flow exceptions and JVM errors are not swallowed`() {
        class ProcessCanceledException : RuntimeException()
        assertTrue(TlalocInternalErrors.mustRethrow(ProcessCanceledException()))
        assertTrue(TlalocInternalErrors.mustRethrow(StackOverflowError()))
        assertTrue(TlalocInternalErrors.mustRethrow(NoSuchMethodError()))
        assertTrue(TlalocInternalErrors.mustRethrow(java.util.concurrent.CancellationException()))
        assertTrue(!TlalocInternalErrors.mustRethrow(IllegalStateException()))
        assertTrue(!TlalocInternalErrors.mustRethrow(NullPointerException()))
    }

    // ---------------- R4: strict boolean parse ----------------

    @Test
    fun `an unknown value for dumpGradSource is refused by name`() {
        val result = compile(
            mapOf("stub.kt" to SCALAR_STUB, "Main.kt" to workingGrad),
            options = arrayOf("plugin:io.tlaloc.plugin:dumpGradSource=ture"),
        )
        assertTrue(result.exitCode != 0, "a misspelled value must fail, not be read as false")
        assertTrue(
            result.messages.any { "dumpGradSource" in it.message && "ture" in it.message },
            "the refusal names the option and the value; got:\n${result.render()}",
        )
    }

    @Test
    fun `the registrar and the command line processor share one plugin id`() {
        @OptIn(org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi::class)
        val ids = TlalocCompilerPluginRegistrar().pluginId to TlalocCommandLineProcessor().pluginId
        assertEquals(TlalocCommandLineProcessor.PLUGIN_ID, ids.first)
        assertEquals(TlalocCommandLineProcessor.PLUGIN_ID, ids.second)
    }

    // ---------------- harness ----------------

    private fun <T> withInjectedFault(phase: TlalocInternalErrors.Phase, block: () -> T): T {
        val previous = System.getProperty(TlalocInternalErrors.INJECT_FAULT_PROPERTY)
        System.setProperty(TlalocInternalErrors.INJECT_FAULT_PROPERTY, phase.propertyValue)
        try {
            return block()
        } finally {
            if (previous == null) {
                System.clearProperty(TlalocInternalErrors.INJECT_FAULT_PROPERTY)
            } else {
                System.setProperty(TlalocInternalErrors.INJECT_FAULT_PROPERTY, previous)
            }
        }
    }

    private fun gradAt(loader: ClassLoader, className: String, x: Float): Float {
        @Suppress("UNCHECKED_CAST")
        val g = loader.loadClass(className).getMethod("gradOf").invoke(null) as (Float) -> Float
        return g(x)
    }

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
        val location: CompilerMessageSourceLocation?,
    )

    private data class CompileResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val outDir: File,
    ) {
        fun render(): String = messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compile(
        files: Map<String, String>,
        options: Array<String> = emptyArray(),
        werror: Boolean = false,
        keepOutput: Boolean = false,
    ): CompileResult {
        val tempDir = Files.createTempDirectory("tlaloc-robustness-test").toFile()
        try {
            val srcDir = File(tempDir, "src").apply { mkdirs() }
            for ((name, text) in files) File(srcDir, name).writeText(text)
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
                    collected += CompileMessage(severity, message, location)
                }
            }

            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(srcDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                if (options.isNotEmpty()) pluginOptions = options
                destination = outDir.absolutePath
                // The real :autograd is on this test JVM's classpath; the stubs above
                // stand in for it, so it is left off the compiled program's classpath.
                classpath = System.getProperty("java.class.path")
                    .split(File.pathSeparator)
                    .filterNot { "autograd" in File(it).path }
                    .joinToString(File.pathSeparator)
                noStdlib = true
                noReflect = true
                allWarningsAsErrors = werror
            }

            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            return CompileResult(exitCode, collected, outDir)
        } finally {
            if (!keepOutput) tempDir.deleteRecursively()
        }
    }

    private companion object {
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = f
        """.trimIndent()
    }
}
