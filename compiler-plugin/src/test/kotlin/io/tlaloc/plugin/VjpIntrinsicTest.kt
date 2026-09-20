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
 * §0.4.398 — E2E for the seeded-cotangent intrinsics `vjp` / `valueAndVjp`
 * (DiffKT's `vjp` / `primalAndPullback`, audit item 10): the pullback of a
 * USER-SUPPLIED cotangent ȳ through f at x, in one reverse pass — `grad {}`
 * generalised to tensor-valued f. Like `jacobian` (§0.4.394), no monomorphic
 * stubs: the user source resolves the REAL generic `io.tlaloc.autograd.vjp` /
 * `valueAndVjp` declarations off the test classpath, and a synthesis fallback
 * is a hard failure (no tape path).
 *
 * Certification oracles:
 *  vf1 = vjp { x -> x ⊙ x } at non-uniform ȳ  → x̄ = 2·x ⊙ ȳ
 *  vf2 = vjp { x -> Σ(x ⊙ x) } at ȳ = 1       → x̄ = 2x = grad (the consistency identity)
 *  the JVP⇄VJP inner-product identity          → ⟨ȳ, jvp(x, v)⟩ == ⟨vjp(x, ȳ), v⟩
 *  pv  = valueAndVjp { x -> x ⊙ x }            → primal y = x ⊙ x alongside x̄
 */
class VjpIntrinsicTest {

    @Test
    fun `vjp and valueAndVjp lower through the plugin as seeded reverse pullbacks`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndVjp
            import io.tlaloc.autograd.vjp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun dot(a: DTensor<*, *>, b: DTensor<*, *>): Float {
                val ad = (a.storage as HostF32Storage).data
                val bd = (b.storage as HostF32Storage).data
                var s = 0.0f
                for (i in ad.indices) s += ad[i] * bd[i]
                return s
            }
            fun main() {
                val vf1 = vjp { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val vf2 = vjp { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                val gf2 = grad { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                val jf1 = jvp { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val pv1 = valueAndVjp { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val X = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val Ybar = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, -1.0f, 2.0f))
                val V = Tensors.f32Vector<Sym>(floatArrayOf(3.0f, 0.25f, -2.0f))
                show("vf1", vf1(X, Ybar))
                show("vf2", vf2(X, 1.0f))
                show("gf2", gf2(X))
                val (y1, xbar1) = pv1(X, Ybar)
                show("pv1y", y1)
                show("pv1x", xbar1)
                // The JVP⇄VJP inner-product identity, computed numerically:
                // ⟨ȳ, J·v⟩ (forward) must equal ⟨Jᵀ·ȳ, v⟩ (reverse).
                println("fwdip 1 " + dot(Ybar, jf1(X, V)))
                println("revip 1 " + dot(vf1(X, Ybar), V))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; vjp/valueAndVjp have no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val rows = result.stdout.trim().lines().associate {
            val parts = it.trim().split(" ", limit = 3)
            parts[0] to (parts[1] to parts[2].split(",").map { s -> s.toFloat() })
        }

        fun check(name: String, wantDims: String, want: List<Float>, tol: Float = 1e-4f) {
            val (dims, got) = rows[name] ?: error("no '$name' row in stdout:\n${result.stdout}")
            assertEquals(wantDims, dims, "$name dims")
            assertEquals(want.size, got.size, "$name size")
            for (i in want.indices) {
                assertTrue(abs(got[i] - want[i]) < tol, "$name[$i]=${got[i]} want ${want[i]} (got $got)")
            }
        }

        // x = [1, 2, 3], ȳ = [0.5, -1, 2]: x̄ = 2·x⊙ȳ = [1, -4, 12].
        check("vf1", "3", listOf(1f, -4f, 12f))
        // The grad-consistency identity: vjp at the unit seed IS grad.
        check("vf2", "3", listOf(2f, 4f, 6f))
        check("gf2", "3", listOf(2f, 4f, 6f))
        // valueAndVjp: the primal y = x⊙x rides along with the same pullback.
        check("pv1y", "3", listOf(1f, 4f, 9f))
        check("pv1x", "3", listOf(1f, -4f, 12f))
        // ⟨ȳ, J·v⟩ = ⟨Jᵀ·ȳ, v⟩ — both sides computed numerically in the user
        // program: J = diag(2x), so both equal Σ 2·xᵢ·ȳᵢ·vᵢ = 3 - 1 - 24 = -22.
        check("fwdip", "1", listOf(-22f))
        check("revip", "1", listOf(-22f))
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-vjp-test").toFile()
        try {
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
}
