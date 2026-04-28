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
 * Layer 1 §0.4.241+ — N.2 contract-inference tests.
 *
 * Three concerns:
 *
 * 1. **Native Kotlin type-inference** drives the contraction-correctness
 *    enforcement (Refined Option A per the design audit). Disjoint named
 *    axes produce a compile-time *type-mismatch* error from Kotlin's own
 *    checker, no plugin diagnostic needed.
 * 2. **Plugin lowering** of `contract` calls inside `grad { }` lambdas
 *    emits `OpKind.MATMUL` (rank-2 case) or `OpKind.DOT` (rank-1 case)
 *    with `axisNames` populated on the result type and the four
 *    `*_contracting_dims` / `*_batching_dims` attrs the StableHLO emitter
 *    expects.
 * 3. **Result-type axisNames** propagate through to the DXIR pretty-print:
 *    a `Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>> contract Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>`
 *    produces an op typed `f32[-1@Batch,-1@Hidden]`.
 */
class ContractInferenceTest {

    @Test
    fun `Rank2 contract Rank2 emits MATMUL with named result axes`() {
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
        val lowered = result.loweredMessages()
        val unsupported = result.unsupportedMessages()
        assertEquals(
            1, lowered.size,
            "expected 1 LAMBDA_LOWERED, got ${lowered.size}.\n" +
                "unsupported reasons: $unsupported\n" +
                "all messages:\n${result.renderMessages()}",
        )
        val dxir = lowered.single()
        // Param `a` carries the Batch+SeqLen pair, surface'd via N.1.
        assertContains(dxir, "%0: f32[-1@Batch,-1@SeqLen]")
        // The contract-emitted MATMUL: result type carries the surviving axes
        // (Batch from lhs, Hidden from rhs); shared axis SeqLen is dropped.
        assertContains(dxir, "matmul(%")
        assertContains(dxir, "f32[-1@Batch,-1@Hidden]")
        // Op attrs populated for the StableHLO emitter explicit path.
        assertContains(dxir, "lhs_contracting_dims=[1]")
        assertContains(dxir, "rhs_contracting_dims=[0]")
        assertContains(dxir, "contracted_names=[SeqLen]")
    }

    @Test
    fun `Rank1 contract Rank1 emits DOT producing scalar`() {
        val result = compile(
            stub = STUB_GRAD_GENERIC,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.SeqLen
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.contract
                fun main() {
                    val g = grad { a: DTensor<Rank1<Named<SeqLen, Sym>>, F32> ->
                        a contract a
                    }
                    println(g)
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.renderMessages()}")
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}")
        val dxir = lowered.single()
        // Param `a` carries the SeqLen axis name (N.1 lift).
        assertContains(dxir, "%0: f32[-1@SeqLen]")
        // Rank-1 × rank-1 emits DOT (not MATMUL); result is scalar `f32`.
        assertContains(dxir, "dot(%0, %0)")
        assertContains(dxir, ": f32")
        // Contracted-name attr names the SeqLen axis.
        assertContains(dxir, "contracted_names=[SeqLen]")
    }

    @Test
    fun `disjoint named axes fail to compile with Kotlin type error`() {
        // Refined Option A's central guarantee: when both operands carry
        // named axes but with no shared name, Kotlin's overload resolution
        // cannot find a matching `contract` overload and surfaces a native
        // type-mismatch error at the call site. No plugin diagnostic; no
        // silent fallback to a positional path.
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
                import io.tlaloc.core.Vocab
                import io.tlaloc.core.ops.contract
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad2 {
                        a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
                        b: DTensor<Rank2<Named<Vocab, Sym>, Named<Hidden, Sym>>, F32> ->
                        // Disjoint names: Batch+SeqLen vs Vocab+Hidden — no overload matches.
                        (a contract b).sum()
                    }
                    println(g)
                }
            """.trimIndent(),
        )
        assertNotEquals(0, result.exitCode, "expected compile failure; got success.\n${result.renderMessages()}")
        // The error from Kotlin's overload resolution mentions either the
        // unresolved overload or a type-mismatch on the named-axis position.
        val errorText = result.messages
            .filter { it.severity == CompilerMessageSeverity.ERROR }
            .joinToString("\n") { it.message }
        assertTrue(
            "contract" in errorText || "type mismatch" in errorText.lowercase() ||
                "unresolved" in errorText.lowercase() || "cannot infer" in errorText.lowercase(),
            "expected a type-mismatch / overload-resolution / inference error mentioning contract, got:\n$errorText",
        )
    }

    @Test
    fun `mixed named-and-positional contract is a type error`() {
        // Refined Option A v1 only accepts overloads where both operands
        // carry named axes. A mix (one named, one positional Sym) does not
        // match either overload signature, so Kotlin reports a type error.
        val result = compile(
            stub = STUB_GRAD2_GENERIC,
            user = """
                import io.tlaloc.autograd.grad2
                import io.tlaloc.core.Batch
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Named
                import io.tlaloc.core.Rank2
                import io.tlaloc.core.SeqLen
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.contract
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad2 {
                        a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>,
                        b: DTensor<Rank2<Sym, Sym>, F32> ->
                        (a contract b).sum()
                    }
                    println(g)
                }
            """.trimIndent(),
        )
        assertNotEquals(0, result.exitCode, "expected compile failure for mixed named+positional; got success")
    }

    // ----- helpers -----------------------------------------------------------

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
        fun unsupportedMessages(): List<String> =
            messages.filter { "Tlaloc could not lower lambda" in it.message }
                .map { it.message }
        fun renderMessages(): String =
            messages.joinToString("\n") { "[${it.severity}] ${it.message}" }
    }

    private fun compile(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-contract").toFile()
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
        /** Generic single-param identity-grad stub; we never run, only inspect the dxir. */
        private val STUB_GRAD_GENERIC = """
            package io.tlaloc.autograd
            fun <S, R> grad(f: (S) -> R): (S) -> S = { x -> x }
        """.trimIndent()

        /**
         * Generic two-param identity-grad stub. Lets us pass `a` and `b` as
         * separate lambda parameters with different named-axis types so the
         * test doesn't need any cast-shenanigans to construct mismatched types.
         */
        private val STUB_GRAD2_GENERIC = """
            package io.tlaloc.autograd
            fun <A, B, R> grad2(f: (A, B) -> R): (A, B) -> Pair<A, B> = { a, b -> a to b }
        """.trimIndent()
    }
}
