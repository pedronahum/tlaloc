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
 * §0.4.499 — the first-contact defects, pinned.
 *
 * Two things were true of every consumer project up to 0.1.0-alpha01, and both are
 * pinned here because a fix nobody tests comes back:
 *
 *  1. **The build log.** ONE `grad {}` call emitted TWO compiler WARNINGs, each
 *     dumping the lowered Tlaloc IR — the FIR checker's `LAMBDA_LOWERED` and the IR
 *     extension's "saw handoff". They were developer introspection sitting in a
 *     stranger's build output, and in a project compiling with
 *     `allWarningsAsErrors = true` (the `-Werror` flag; routine in Kotlin shops)
 *     they did not merely annoy — they FAILED THE BUILD, with
 *     `e: warnings found and -Werror specified`, on a program that was completely
 *     correct. [`a correct grad consumer compiles under -Werror`] is the test that
 *     matters: it is the exact configuration that could not compile before.
 *
 *  2. **The refusal.** A lambda the lowering could not handle was a WARNING; the
 *     call was left unrewritten, and `io.tlaloc.autograd`'s fallback body threw
 *     `IllegalStateException` at the FIRST CALL — at runtime — telling the user to
 *     add a compiler plugin that was already applied. It is now a compile-time
 *     ERROR carrying the lowering's own verbatim reason, with
 *     `strictLowering=false` as the named opt-out for anyone who wants the old
 *     late failure.
 *
 * The information is not lost, only moved: `dumpLoweredIr=true` brings both dumps
 * back, and the §0.4.450 `dumpGradSource` family is untouched (see
 * [DumpGradSourceTest]).
 */
class DiagnosticNoiseTest {

    // ---------------- 1. the build log ----------------

    @Test
    fun `a correct grad consumer compiles under -Werror and emits no Tlaloc diagnostic`() {
        val result = compile(
            stub = SCALAR_STUB,
            user = WORKING_GRAD,
            werror = true,
        )
        assertEquals(
            0, result.exitCode,
            "a correct grad {} program must compile with -Werror; got:\n${result.render()}",
        )
        assertTrue(
            result.messages.none { "Tlaloc" in it.message },
            "the default build must emit NO Tlaloc diagnostic for a working grad {}; got:\n" +
                result.render(),
        )
    }

    @Test
    fun `the default build emits neither the FIR dxir dump nor the IR handoff dump`() {
        val result = compile(stub = SCALAR_STUB, user = WORKING_GRAD)
        assertEquals(0, result.exitCode, "compile failed:\n${result.render()}")
        assertTrue(
            result.messages.none { "lowered lambda to dxir" in it.message },
            "the FIR LAMBDA_LOWERED dump must be off by default; got:\n${result.render()}",
        )
        assertTrue(
            result.messages.none { "saw handoff" in it.message },
            "the IR handoff dump must be off by default; got:\n${result.render()}",
        )
    }

    @Test
    fun `dumpLoweredIr brings both dumps back, and the IR half is an INFO not a warning`() {
        val result = compile(
            stub = SCALAR_STUB,
            user = WORKING_GRAD,
            options = arrayOf("plugin:io.tlaloc.plugin:dumpLoweredIr=true"),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.render()}")

        val firDump = result.messages.singleOrNull { "lowered lambda to dxir" in it.message }
            ?: error("dumpLoweredIr=true must restore the FIR dump; got:\n${result.render()}")
        assertTrue(
            "mul(%0, %0)" in firDump.message,
            "the restored FIR dump must carry the lowered dxir; got:\n${firDump.message}",
        )

        val irDump = result.messages.singleOrNull { "saw handoff" in it.message }
            ?: error("dumpLoweredIr=true must restore the IR dump; got:\n${result.render()}")
        assertEquals(
            CompilerMessageSeverity.INFO, irDump.severity,
            "the IR-phase dump is INFO now — it was an unconditional WARNING",
        )
        assertTrue(
            "mul(%0, %0)" in irDump.message,
            "the restored IR dump must carry the lowered dxir; got:\n${irDump.message}",
        )
    }

    @Test
    fun `an unknown value for a boolean plugin option is refused by name`() {
        val result = compile(
            stub = SCALAR_STUB,
            user = WORKING_GRAD,
            options = arrayOf("plugin:io.tlaloc.plugin:dumpLoweredIr=ture"),
        )
        assertTrue(result.exitCode != 0, "a misspelled option value must fail, not be read as false")
        assertTrue(
            result.messages.any { "dumpLoweredIr" in it.message && "ture" in it.message },
            "the refusal must name the option AND the value it did not understand; got:\n" +
                result.render(),
        )
    }

    // ---------------- 2. the refusal ----------------

    /** The lowering refuses a call to a user function: one of `FirLambdaToDxirLowering`'s
     * ~208 named `LoweringException` sites. Before §0.4.499 this compiled green and blew
     * up at the first call to `g`. */
    private val unlowerableGrad = """
        import io.tlaloc.autograd.grad
        fun cube(x: Float): Float = x * x * x
        fun main() {
            val g = grad { x: Float -> cube(x) }
            println(g(1.0f))
        }
    """.trimIndent()

    @Test
    fun `an unlowerable grad body is a compile-time ERROR naming the construct and the opt-out`() {
        val result = compile(stub = SCALAR_STUB, user = unlowerableGrad)
        assertTrue(
            result.exitCode != 0,
            "an unlowerable grad {} must fail the compilation; got:\n${result.render()}",
        )
        val err = result.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.ERROR && "could not lower this lambda" in it.message
        } ?: error("expected one LAMBDA_NOT_LOWERABLE error; got:\n${result.render()}")
        assertTrue(
            "cube" in err.message,
            "the error must carry the lowering's own verbatim reason (which names `cube`); " +
                "got:\n${err.message}",
        )
        assertTrue(
            "strictLowering=false" in err.message,
            "the error must name the opt-out flag; got:\n${err.message}",
        )
    }

    @Test
    fun `strictLowering=false restores the pre-alpha warning and the build stays green`() {
        val result = compile(
            stub = SCALAR_STUB,
            user = unlowerableGrad,
            options = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false"),
        )
        assertEquals(
            0, result.exitCode,
            "the opt-out must compile, deliberately deferring the failure to runtime; got:\n" +
                result.render(),
        )
        val warn = result.messages.singleOrNull {
            it.severity == CompilerMessageSeverity.WARNING && "could not lower lambda" in it.message
        } ?: error("expected one LAMBDA_UNSUPPORTED warning; got:\n${result.render()}")
        assertTrue("cube" in warn.message, "the warning still names the construct; got:\n${warn.message}")
    }

    @Test
    fun `a non-lambda argument is refused by name instead of failing at the first call`() {
        val result = compile(
            stub = SCALAR_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun square(x: Float): Float = x * x
                fun main() {
                    val g = grad(::square)
                    println(g(2.0f))
                }
            """.trimIndent(),
        )
        assertTrue(result.exitCode != 0, "a function reference cannot be lowered; got:\n${result.render()}")
        assertTrue(
            result.messages.any {
                it.severity == CompilerMessageSeverity.ERROR && "not a lambda literal" in it.message
            },
            "the refusal must say WHY (no lambda body at the call site); got:\n${result.render()}",
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

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compile(
        stub: String,
        user: String,
        options: Array<String> = emptyArray(),
        werror: Boolean = false,
    ): CompileResult {
        val tempDir = Files.createTempDirectory("tlaloc-noise-test").toFile()
        try {
            File(tempDir, "stub.kt").writeText(stub)
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
                    collected += CompileMessage(severity, message)
                }
            }

            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                if (options.isNotEmpty()) pluginOptions = options
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
                allWarningsAsErrors = werror
            }

            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            return CompileResult(exitCode, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private companion object {
        /** The `grad` surface the plugin recognises by FQN, as a source stub — the same
         * trick every other `:compiler-plugin` test uses so the harness needs no
         * `:autograd` artifact on the compiled program's classpath. */
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = f
        """.trimIndent()

        /** A program whose `grad {}` lowers cleanly: the case that must be SILENT. */
        private val WORKING_GRAD = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> x * x }
                println(g(2.0f))
            }
        """.trimIndent()
    }
}
