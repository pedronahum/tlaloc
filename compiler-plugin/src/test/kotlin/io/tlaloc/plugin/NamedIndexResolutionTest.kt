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
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Layer 1 §0.4.241+ — N.1 plugin name-resolution tests.
 *
 * Asserts that `FirLambdaToDxirLowering.resolveParamType` lifts `Named<N, A>`
 * shape atoms into [io.tlaloc.ir.DxirType.axisNames] when constructing the
 * DxirParam type. Tests are pure FIR-stage diagnostic checks — they read the
 * pretty-printed DXIR from the `LAMBDA_LOWERED` warning and assert on the
 * axis-name rendering (e.g. `f32[-1@Batch,-1@SeqLen]`).
 *
 * Backwards-compat is the load-bearing invariant: a purely positional shape
 * (`Rank1<Sym>`, `Rank2<Sym, Sym>`) must produce *no* axis-name annotation,
 * byte-for-byte equivalent to pre-Layer-1 output.
 */
class NamedIndexResolutionTest {

    @Test
    fun `Rank1 with Named lifts axis name into DxirType pretty-print`() {
        val result = compile(
            stub = AUTOGRAD_STUB_RANK1_NAMED,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.Batch
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank1<Named<Batch, Sym>>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        // Param type should carry the axis name. The pretty-printer renders
        // `dim@name` for named axes and bare `dim` for unnamed ones.
        assertContains(dxir, "fn grad_body(%0: f32[-1@Batch])")
    }

    @Test
    fun `Rank2 with two Named atoms lifts both axis names`() {
        val result = compile(
            stub = AUTOGRAD_STUB_RANK2_NAMED,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.Batch
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank2
                import io.tlaloc.core.SeqLen
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32[-1@Batch,-1@SeqLen])")
    }

    @Test
    fun `mixed Named and positional atoms lift only the named axes`() {
        // Half-named param: Named<Batch, Sym> on axis 0, raw Sym on axis 1.
        // The lifter should produce axisNames = ["Batch", null] and the
        // pretty-print should show `-1@Batch,-1` (no `@` for the unnamed axis).
        val result = compile(
            stub = AUTOGRAD_STUB_RANK2_NAMED,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.Batch
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank2
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank2<Named<Batch, Sym>, Sym>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32[-1@Batch,-1])")
    }

    @Test
    fun `purely positional Rank2 produces no axis-name annotation`() {
        // Backwards-compat: pre-Layer-1 user code should produce identical
        // pretty-printed DXIR. No `@` characters in the param type.
        val result = compile(
            stub = AUTOGRAD_STUB_RANK2_NAMED,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Rank2
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank2<Sym, Sym>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32[-1,-1])")
        // Crucially: no `@` in the param-type rendering. Bare positional shape.
        val paramLine = dxir.lines().first { "grad_body" in it }
        assertFalse(
            "@" in paramLine,
            "purely positional Rank2 must not emit any axis-name annotation; got: $paramLine",
        )
    }

    @Test
    fun `user-defined IndexName outside core resolves correctly`() {
        // The lifter keys off the FQN's short class name, not membership in
        // io.tlaloc.core.CommonNames. A user-defined IndexName in a different
        // package should lift its short class name as the axis name.
        val result = compile(
            stub = AUTOGRAD_STUB_RANK1_USER_NAMED,
            user = """
                package myapp
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.IndexName
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum

                object Vehicles : IndexName { override val name = "vehicles" }

                fun main() {
                    val g = grad { x: DTensor<Rank1<Named<Vehicles, Sym>>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        // v1 lifts the simple class name (Vehicles), not the override-name
        // string ("vehicles"). The audit at docs/audits/named_indices_audit.md
        // tracks this as a known v1 simplification.
        assertContains(dxir, "fn grad_body(%0: f32[-1@Vehicles])")
    }

    // ----- helpers (mirrors the per-file pattern in Rank2ParamLoweringTest) ---

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>) {
        fun loweredMessages(): List<String> =
            messages.filter { "Tlaloc lowered lambda to dxir:" in it.message }
                .map { it.message.substringAfter("dxir:\n").trimEnd() }
        fun renderMessages(): String =
            messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun compile(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-named-resolve").toFile()
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
                // §0.4.499 — this harness READS the lowered-dxir dump, which is off
                // by default now; and (where listed) it exercises the pre-alpha
                // tape-fallback path, which is a compile error by default.
                pluginOptions = arrayOf(
                    "plugin:io.tlaloc.plugin:dumpLoweredIr=true",
                )
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

    private fun assertContains(actual: String, expected: String) {
        assertTrue(
            expected in actual,
            "expected substring not found.\n  expected: $expected\n  actual:\n$actual",
        )
    }

    companion object {
        /**
         * grad over any rank-1 input that reduces to a scalar. Generic over `S`
         * so the stub can be reused across positional + named-index variants.
         * Body is a no-op (`{ x -> x }`); we never run, only inspect the
         * pretty-printed DXIR from the LAMBDA_LOWERED diagnostic.
         */
        private val AUTOGRAD_STUB_RANK1_NAMED = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.ScalarShape
            fun <S> grad(f: (S) -> DTensor<ScalarShape, F32>): (S) -> S = { x -> x }
        """.trimIndent()

        /** grad over rank-2 (variants of named/positional). Stub is identity. */
        private val AUTOGRAD_STUB_RANK2_NAMED = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.ScalarShape
            fun <S> grad(f: (S) -> DTensor<ScalarShape, F32>): (S) -> S = { x -> x }
        """.trimIndent()

        /** grad over a user-defined IndexName outside io.tlaloc.core. */
        private val AUTOGRAD_STUB_RANK1_USER_NAMED = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.ScalarShape
            fun <S> grad(f: (S) -> DTensor<ScalarShape, F32>): (S) -> S = { x -> x }
        """.trimIndent()
    }
}
