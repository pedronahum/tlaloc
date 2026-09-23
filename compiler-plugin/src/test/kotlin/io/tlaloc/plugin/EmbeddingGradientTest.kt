package io.tlaloc.plugin

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
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.400 — Phase A3b E2E: `embedding` differentiates end-to-end through the
 * K2 plugin, with the I32 index vector flowing through the `grad {}` lambda as
 * a NON-DIFFERENTIABLE param — the first integer tensor param the synthesis
 * accepts. The §0.4.370 EmbeddingRule below the surface emits the fused
 * EMBEDDING_GRAD scatter-add, which the synthesis lowers to the host twin
 * `embeddingGrad(upstream, indices, tableTemplate)`; the indices' gradient
 * slot is the §0.4.54 structural integer zero, materialised as
 * `intZerosLike(indices)`.
 *
 *  g1 = ∇_t Σ embedding(t, i)   → dT[v,:] = count(v) — collisions sum, and the
 *                                 unselected vocab row is EXACT zeros
 *  g2 = ∇_t Σ embedding(t, i)²  → dT[v,:] = 2·count(v)·t[v,:] — the gradient
 *                                 body must RECOMPUTE the embedding (irEmbedding)
 *  both: dI = I32 zeros shaped like i (never the stub's -1 sentinel)
 */
class EmbeddingGradientTest {

    @Test
    fun `grad through embedding with an I32 index param`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.hostI32
            import io.tlaloc.core.ops.embedding
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dumpF(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun dumpI(name: String, t: DTensor<*, I32>) {
                println(name)
                for (v in t.hostI32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank1<Sym>, I32> ->
                    embedding(t, i).sum().toFloat()
                }
                val g2 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank1<Sym>, I32> ->
                    val e = embedding(t, i)
                    (e * e).sum().toFloat()
                }
                val T = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
                    0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f,
                ))
                val I = Tensors.i32Vector<Sym>(intArrayOf(0, 2, 0, 1))
                val (d1t, d1i) = g1(T, I)
                dumpF("g1t", d1t); dumpI("g1i", d1i)
                val (d2t, d2i) = g2(T, I)
                dumpF("g2t", d2t); dumpI("g2i", d2i)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the embedding gradients to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // idx = [0, 2, 0, 1] over vocab 4: counts = [2, 1, 1, 0] — slot 0 is the
        // collision, slot 3 is never selected.
        val t = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f)
        val counts = floatArrayOf(2f, 1f, 1f, 0f)
        val wantG1t = FloatArray(8) { counts[it / 2] }
        val wantG2t = FloatArray(8) { 2f * counts[it / 2] * t[it] }
        val want = mapOf("g1t" to wantG1t.toList(), "g2t" to wantG2t.toList())

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            when (name) {
                "g1t", "g2t" -> {
                    val expect = want.getValue(name)
                    assertEquals(8, values.size, "$name size")
                    assertTrue(
                        values.any { it != -1.0f },
                        "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                            result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
                    )
                    for (j in values.indices) {
                        assertTrue(
                            abs(values[j] - expect[j]) < 1e-5f,
                            "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                        )
                    }
                    // The unselected vocab row must be EXACT zeros, not merely small.
                    assertTrue(
                        values[6] == 0f && values[7] == 0f,
                        "$name: unselected vocab row must be exactly zero, got " +
                            "[${values[6]}, ${values[7]}]",
                    )
                }
                "g1i", "g2i" -> {
                    assertEquals(4, values.size, "$name size")
                    assertTrue(
                        values.all { it == 0f },
                        "$name: I32 index gradient must be the structural zero, got $values " +
                            "(-1 would be the stub sentinel — the rewrite never fired)",
                    )
                }
                else -> error("unexpected section '$name'")
            }
        }
    }

    @Test
    fun `grad through embedding with a paddingIndex`() {
        // §0.4.409 — embedding(t, i, 1): rows equal to paddingIndex produce
        // zero output AND receive zero gradient. idx = [0, 1, 0, 2] over vocab
        // 4: counts = [2, 0(padded), 1, 0(unselected)] — both zero rows must be
        // EXACT zeros, the slot-0 collision still sums, and the nonlinear loss
        // must RECOMPUTE the PADDED embedding in its gradient body (the
        // 3-argument irEmbedding synthesis arm).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.hostI32
            import io.tlaloc.core.ops.embedding
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dumpF(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun dumpI(name: String, t: DTensor<*, I32>) {
                println(name)
                for (v in t.hostI32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank1<Sym>, I32> ->
                    embedding(t, i, 1).sum().toFloat()
                }
                val g2 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank1<Sym>, I32> ->
                    val e = embedding(t, i, 1)
                    (e * e).sum().toFloat()
                }
                val T = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
                    0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f,
                ))
                val I = Tensors.i32Vector<Sym>(intArrayOf(0, 1, 0, 2))
                val (d1t, d1i) = g1(T, I)
                dumpF("g1t", d1t); dumpI("g1i", d1i)
                val (d2t, d2i) = g2(T, I)
                dumpF("g2t", d2t); dumpI("g2i", d2i)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
        assertTrue(
            result.messages.none {
                "kept original call" in it.message
            },
            "synthesis fell back; expected the padded embedding gradients to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // counts = [2, 0 (padded), 1, 0 (unselected)]
        val t = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f)
        val counts = floatArrayOf(2f, 0f, 1f, 0f)
        val wantG1t = FloatArray(8) { counts[it / 2] }
        val wantG2t = FloatArray(8) { 2f * counts[it / 2] * t[it] }
        val want = mapOf("g1t" to wantG1t.toList(), "g2t" to wantG2t.toList())

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            when (name) {
                "g1t", "g2t" -> {
                    val expect = want.getValue(name)
                    assertEquals(8, values.size, "$name size")
                    assertTrue(
                        values.any { it != -1.0f },
                        "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                            result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
                    )
                    for (j in values.indices) {
                        assertTrue(
                            abs(values[j] - expect[j]) < 1e-5f,
                            "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                        )
                    }
                    // The PADDED vocab row must be EXACT zeros, not merely small.
                    assertTrue(
                        values[2] == 0f && values[3] == 0f,
                        "$name: padded vocab row must be exactly zero, got [${values[2]}, ${values[3]}]",
                    )
                }
                "g1i", "g2i" -> assertTrue(
                    values.size == 4 && values.all { it == 0f },
                    "$name: I32 index gradient must be the structural zero, got $values",
                )
                else -> error("unexpected section '$name'")
            }
        }
    }

    @Test
    fun `grad through embedding with a rank-2 index batch`() {
        // §0.4.409 — [B=2, N=2] index batch → [2, 2, D] embedding, with a
        // collision ACROSS batch rows (slot 0 in both). The nonlinear loss
        // recomputes the batched embedding (the 4-type-arg irEmbedding arm).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.hostI32
            import io.tlaloc.core.ops.embedding
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dumpF(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun dumpI(name: String, t: DTensor<*, I32>) {
                println(name)
                for (v in t.hostI32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank2<Sym, Sym>, I32> ->
                    embedding(t, i).sum().toFloat()
                }
                val g2 = grad { t: DTensor<Rank2<Sym, Sym>, F32>, i: DTensor<Rank2<Sym, Sym>, I32> ->
                    val e = embedding(t, i)
                    (e * e).sum().toFloat()
                }
                val T = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
                    0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f,
                ))
                val I = Tensors.i32Matrix<Sym, Sym>(2, 2, intArrayOf(0, 2, 0, 1))
                val (d1t, d1i) = g1(T, I)
                dumpF("g1t", d1t); dumpI("g1i", d1i)
                val (d2t, d2i) = g2(T, I)
                dumpF("g2t", d2t); dumpI("g2i", d2i)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_BATCH_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
        assertTrue(
            result.messages.none {
                "kept original call" in it.message
            },
            "synthesis fell back; expected the batched embedding gradients to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Flat idx = [0, 2, 0, 1]: the slot-0 collision spans batch rows.
        val t = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f)
        val counts = floatArrayOf(2f, 1f, 1f, 0f)
        val wantG1t = FloatArray(8) { counts[it / 2] }
        val wantG2t = FloatArray(8) { 2f * counts[it / 2] * t[it] }
        val want = mapOf("g1t" to wantG1t.toList(), "g2t" to wantG2t.toList())

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            when (name) {
                "g1t", "g2t" -> {
                    val expect = want.getValue(name)
                    assertEquals(8, values.size, "$name size")
                    assertTrue(
                        values.any { it != -1.0f },
                        "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                            result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
                    )
                    for (j in values.indices) {
                        assertTrue(
                            abs(values[j] - expect[j]) < 1e-5f,
                            "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                        )
                    }
                    assertTrue(
                        values[6] == 0f && values[7] == 0f,
                        "$name: unselected vocab row must be exactly zero, got [${values[6]}, ${values[7]}]",
                    )
                }
                "g1i", "g2i" -> assertTrue(
                    values.size == 4 && values.all { it == 0f },
                    "$name: rank-2 I32 index gradient must be the structural zero, got $values",
                )
                else -> error("unexpected section '$name'")
            }
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-embedding-test").toFile()
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
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val capturedOut = PrintStream(baos, true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: ${t.cause ?: t}",
                    ),
                    baos.toString(Charsets.UTF_8),
                )
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.HostI32Storage
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>) ->
                        Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                    DTensor(HostI32Storage(IntArray(4) { -1 }), intArrayOf(4), I32),
                ) }
        """.trimIndent()

        /** §0.4.409 — the rank-2 index-batch stub: `(Rank2 F32, Rank2 I32) → Float`. */
        private val AUTOGRAD_BATCH_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.HostI32Storage
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, I32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, I32>) ->
                        Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, I32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                    DTensor(HostI32Storage(IntArray(4) { -1 }), intArrayOf(2, 2), I32),
                ) }
        """.trimIndent()
    }
}
