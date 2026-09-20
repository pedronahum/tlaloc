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
 * §0.4.419 — Phase E1c-pre E2E: a `grad {}` lambda with TWO integer index
 * params synthesises with NO fallback. From §0.4.400 until this slice the
 * synthesis admitted exactly one integer-typed param per lambda — the
 * anonymous integer-zero const the reverse transform returned could not name
 * which param it zeroed under -1 sentinel dims, so several were rejected as
 * ambiguous. The reverse transform now emits the PARAM-ADDRESSED
 * `OpKind.ZEROS_LIKE` on the cloned param itself, and the synthesis
 * materialises it as `intZerosLike(<that param>)` through the env — so this
 * lambda, which previously fell back with "kept original call", lowers.
 *
 * The two index vectors have DIFFERENT lengths (4 vs 3) on purpose: each
 * structural zero must land at ITS OWN param's runtime extent, the
 * disambiguation the anonymous const structurally could not express. This is
 * the prerequisite Phase E1c (the `grad {}` sparse surface) needs — a CSR
 * operand carries colIdx AND rowPtr.
 *
 *  g1 = ∇ Σ emb(t,iA) + Σ emb(t,iB)   → dT[v,:] = countA(v) + countB(v)
 *  g2 = ∇ Σ emb(t,iA)² + Σ emb(t,iB)² → dT[v,:] = 2·(countA(v)+countB(v))·t[v,:]
 *                                        (both embeddings RECOMPUTE in the body)
 *  both: dIA = I32 zeros shaped like iA (4), dIB = I32 zeros shaped like iB (3)
 */
class MultiIndexParamGradientTest {

    @Test
    fun `grad through two I32 index params lowers with no fallback`() {
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
                val g1 = grad { t: DTensor<Rank2<Sym, Sym>, F32>,
                                iA: DTensor<Rank1<Sym>, I32>,
                                iB: DTensor<Rank1<Sym>, I32> ->
                    embedding(t, iA).sum().toFloat() + embedding(t, iB).sum().toFloat()
                }
                val g2 = grad { t: DTensor<Rank2<Sym, Sym>, F32>,
                                iA: DTensor<Rank1<Sym>, I32>,
                                iB: DTensor<Rank1<Sym>, I32> ->
                    val eA = embedding(t, iA)
                    val eB = embedding(t, iB)
                    (eA * eA).sum().toFloat() + (eB * eB).sum().toFloat()
                }
                val T = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
                    0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f,
                ))
                val IA = Tensors.i32Vector<Sym>(intArrayOf(0, 2, 0, 1))
                val IB = Tensors.i32Vector<Sym>(intArrayOf(3, 3, 1))
                val (d1t, d1a, d1b) = g1(T, IA, IB)
                dumpF("g1t", d1t); dumpI("g1a", d1a); dumpI("g1b", d1b)
                val (d2t, d2a, d2b) = g2(T, IA, IB)
                dumpF("g2t", d2t); dumpI("g2a", d2a); dumpI("g2b", d2b)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; a two-index-param lambda must lower post-§0.4.419. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // iA = [0,2,0,1] → countsA = [2,1,1,0]; iB = [3,3,1] → countsB = [0,1,0,2].
        val t = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f, -0.9f, 0.5f)
        val counts = floatArrayOf(2f, 2f, 1f, 2f)
        val wantG1t = FloatArray(8) { counts[it / 2] }
        val wantG2t = FloatArray(8) { 2f * counts[it / 2] * t[it] }
        val wantF = mapOf("g1t" to wantG1t, "g2t" to wantG2t)
        val wantZeroSize = mapOf("g1a" to 4, "g2a" to 4, "g1b" to 3, "g2b" to 3)

        val lines = result.stdout.trim().lines()
        assertEquals(12, lines.size, "expected 12 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            when (name) {
                "g1t", "g2t" -> {
                    val expect = wantF.getValue(name)
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
                }
                "g1a", "g2a", "g1b", "g2b" -> {
                    // Each structural zero at ITS OWN param's extent — 4 for iA,
                    // 3 for iB — never the stub's -1 sentinel, never the other
                    // param's size (the ambiguity §0.4.400 rejected on).
                    assertEquals(wantZeroSize.getValue(name), values.size, "$name size")
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

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-multi-index-test").toFile()
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
        /** The 3-param stub: `(Rank2 F32, Rank1 I32, Rank1 I32) → Float`. */
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
            fun grad(
                f: (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>) -> Float,
            ): (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>) ->
                    Triple<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>> =
                { _, _, _ -> Triple(
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                    DTensor(HostI32Storage(IntArray(4) { -1 }), intArrayOf(4), I32),
                    DTensor(HostI32Storage(IntArray(3) { -1 }), intArrayOf(3), I32),
                ) }
        """.trimIndent()
    }
}
