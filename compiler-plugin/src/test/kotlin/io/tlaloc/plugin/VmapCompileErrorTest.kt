package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * `vmap` misuse is a compile error at the call's file, line and column: a batch of the
 * wrong batch axis (Kotlin's own type mismatch, from the batched type), an op without a
 * batching rule, and a batch axis whose name an example axis already has.
 */
class VmapCompileErrorTest {

    private val header = """
        @file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)
        import io.tlaloc.autograd.*
        import io.tlaloc.core.*
        import io.tlaloc.core.ops.*

        object Feat : IndexName { override val name = "feat" }
        object Time : IndexName { override val name = "time" }
    """.trimIndent()

    private fun errors(src: String): List<F64TestHarness.Message> {
        val r = F64TestHarness.compileAndRun(src)
        assertNotEquals(0, r.exitCode, "expected a compile error:\n${r.describe()}")
        return r.messages.filter { it.severity == CompilerMessageSeverity.ERROR }
    }

    @Test
    fun `a batch of another batch axis does not compile`() {
        val src = header + "\n" + """
            fun main() {
                val f = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Named<Feat, Sym>>, F32> -> x * x }
                val wrong = Tensors.f32Matrix<Named<Time, Sym>, Named<Feat, Sym>>(2, 3, FloatArray(6))
                println(f(wrong))
            }
        """.trimIndent()
        val e = errors(src)
        // The `wrong` argument of `f(wrong)`: line 11 (7 header lines, then the program), column 15.
        val mismatch = e.singleOrNull { "argument type mismatch" in it.message.lowercase() || "type mismatch" in it.message.lowercase() }
            ?: error("no type mismatch among:\n" + e.joinToString("\n") { "${it.line}:${it.column} ${it.message}" })
        assertEquals(11, mismatch.line, mismatch.message)
        assertEquals(15, mismatch.column, mismatch.message)
        assertTrue("Named<Time, Sym>" in mismatch.message && "Named<Batch, Sym>" in mismatch.message, mismatch.message)
    }

    @Test
    fun `an op without a batching rule is refused by name at the call`() {
        val src = header + "\n" + """
            fun main() {
                val f = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Named<Feat, Sym>>, F32> -> x[0] * 2f }
                println(f)
            }
        """.trimIndent()
        val e = errors(src)
        val refusal = e.singleOrNull { "vmap has no batching rule for GATHER" in it.message }
            ?: error("no batching-rule refusal among:\n" + e.joinToString("\n") { "${it.line}:${it.column} ${it.message}" })
        assertEquals(9, refusal.line, refusal.message)
        assertEquals(13, refusal.column, refusal.message)
    }

    @Test
    fun `a batch axis named like an example axis is refused`() {
        val src = header + "\n" + """
            fun main() {
                val f = vmap(batchAxis(Feat)) { x: DTensor<Rank1<Named<Feat, Sym>>, F32> -> x * x }
                println(f)
            }
        """.trimIndent()
        val e = errors(src)
        val clash = e.singleOrNull { "batch axis name clash" in it.message }
            ?: error("no clash among:\n" + e.joinToString("\n") { "${it.line}:${it.column} ${it.message}" })
        assertEquals(9, clash.line, clash.message)
        assertEquals(13, clash.column, clash.message)
        assertTrue("'Feat'" in clash.message, clash.message)
    }
}
