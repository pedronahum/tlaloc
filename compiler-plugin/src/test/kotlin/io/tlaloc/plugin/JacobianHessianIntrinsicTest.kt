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
 * §0.4.394 — Phase B2 E2E: the `jacobian` / `hessian` assembly intrinsics
 * lower through the K2 plugin (a seeded single-pass transform + the runtime
 * basis-assembly helpers in `:autograd`) with NO synthesis fallback — like
 * `concat`, these have no runtime-tape path, so a rejection is a hard failure.
 *
 * Unlike the older intrinsic tests, no monomorphic stub is compiled: the user
 * source resolves the REAL generic `io.tlaloc.autograd.jacobian` / `hessian`
 * declarations off the test classpath, so this also certifies the shipped
 * surface (and the plugin's `assemble*Forward` helper resolution) end to end.
 *
 *  jf1 = jacobian { x -> x ⊙ x }              → J = diag(2x)            [n, n]
 *  jf2 = jacobian { x -> x · Σx }             → J = Σx·I + x·1ᵀ         [n, n]
 *  jf3 = jacobian { x -> (x ⊙ x).sum() }      → J = [2x] (gradient row) [1, n]
 *  hf1 = hessian  { x -> (x ⊙ x).sum() }      → H = 2·I                 [n, n]
 *  hf2 = hessian  { x -> (Σx)² }              → H = 2·1·1ᵀ              [n, n]
 *
 * jf2 exercises the A5a computed-scalar splat under the forward transform
 * (the tangent must flow through BOTH factors of the product rule); hf2's
 * forward-over-reverse threads a tangent through the reverse transform's
 * un-reduce broadcast.
 */
class JacobianHessianIntrinsicTest {

    @Test
    fun `jacobian and hessian lower through the plugin and assemble dense results`() {
        val src = """
            import io.tlaloc.autograd.hessian
            import io.tlaloc.autograd.jacobian
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.exp
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val jf1 = jacobian { x: DTensor<Rank1<Sym>, F32> -> x * x }
                val jf2 = jacobian { x: DTensor<Rank1<Sym>, F32> -> x * x.sum().toFloat() }
                val jf3 = jacobian { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                val hf1 = hessian { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                val hf2 = hessian { x: DTensor<Rank1<Sym>, F32> ->
                    val s = x.sum().toFloat()
                    s * s
                }
                val hf3 = hessian { x: DTensor<Rank1<Sym>, F32> -> x.exp().sum().toFloat() }
                val jf4 = jacobian { x: DTensor<Rank2<Sym, Lit<Int>>, F32> -> x * x }
                val X = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val M = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                show("jf1", jf1(X))
                show("jf2", jf2(X))
                show("jf3", jf3(X))
                show("hf1", hf1(X))
                show("hf2", hf2(X))
                show("hf3", hf3(X))
                show("jf4", jf4(M))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; jacobian/hessian have no tape path so this is a hard failure. " +
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

        // x = [1, 2, 3], Σx = 6.
        check("jf1", "3x3", listOf(2f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 6f))
        // J[i,j] = δij·Σx + x_i.
        check("jf2", "3x3", listOf(7f, 1f, 1f, 2f, 8f, 2f, 3f, 3f, 9f))
        check("jf3", "1x3", listOf(2f, 4f, 6f))
        check("hf1", "3x3", listOf(2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f))
        check("hf2", "3x3", List(9) { 2f })
        // H = diag(exp x) — the value-dependent Hessian: the forward tangent
        // must thread through the reverse body's exp recompute.
        val e = floatArrayOf(kotlin.math.exp(1f), kotlin.math.exp(2f), kotlin.math.exp(3f))
        check(
            "hf3", "3x3",
            listOf(e[0], 0f, 0f, 0f, e[1], 0f, 0f, 0f, e[2]),
            tol = 1e-3f,
        )
        // Rank-2 input flattens row-major: J = diag(2·[1,2,3,4]) over 4 flat elements.
        check(
            "jf4", "4x4",
            listOf(
                2f, 0f, 0f, 0f,
                0f, 4f, 0f, 0f,
                0f, 0f, 6f, 0f,
                0f, 0f, 0f, 8f,
            ),
        )
    }

    /**
     * §0.4.399 — second order THROUGH an in-place size-1 `broadcastTo` stretch,
     * E2E. The hvp body is forward-over-reverse, so its reverse half emits the
     * runtime-extent `SUM_TO` adjoint (§0.4.373) and the forward half threads a
     * tangent through it — the composition the §0.4.373 DEFERRED note flagged.
     * (The note blamed the missing SUM_TO VjpRule; that rule only gates
     * REVERSE-over-reverse, pinned at IR level in
     * `DxirRuntimeExtentClosureTest` — but this E2E surface was never pinned
     * either, and it is the user-visible face of the same closure.)
     *
     *   f(x:[1,3]) = Σ (x.broadcastTo(2,3) ⊙ x.broadcastTo(2,3)) = 2·Σ_j x_j²
     *   ∇f = 4x,  H = 4·I₃ over the flattened input.
     */
    @Test
    fun `hessian through an in-place broadcastTo stretch`() {
        val src = """
            import io.tlaloc.autograd.hessian
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.broadcastTo
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val hf = hessian { x: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (x.broadcastTo(2, 3) * x.broadcastTo(2, 3)).sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(1, 3, floatArrayOf(1.0f, -2.0f, 0.5f))
                show("hf", hf(X))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; hessian has no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
        val line = result.stdout.trim().lines().single { it.startsWith("hf ") }
        val parts = line.split(" ", limit = 3)
        assertEquals("3x3", parts[1], "H dims")
        val got = parts[2].split(",").map { it.toFloat() }
        val want = listOf(4f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 4f)
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < 1e-4f, "H[$i]=${got[i]} want ${want[i]} (got $got)")
        }
    }

    /**
     * §0.4.404 — second order THROUGH a symbolic concat window, E2E. Under
     * `grad {}`'s -1 sentinel dims the reverse half of the hvp body emits
     * SLICE_LIKE windows (ConcatRule's symbolic branch), and the forward half
     * threads a tangent through them — the user-visible face of the concat
     * closure. (The hessian intrinsic is forward-OVER-reverse, so it rides
     * SLICE_LIKE's forward tangent; the PAD_LIKE VjpRule this slice adds is
     * what closes the REVERSE-over-reverse route, pinned at IR level in
     * `DxirNestingMatrixTest`.)
     *
     *   f(x:[3]) = Σ concat(x, x)² = 2·Σx²  →  ∇f = 4x,  H = 4·I₃.
     */
    @Test
    fun `hessian through a symbolic concat window`() {
        val src = """
            import io.tlaloc.autograd.hessian
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.concat
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val hf = hessian { x: DTensor<Rank1<Sym>, F32> ->
                    val c = concat(0, x, x)
                    (c * c).sum().toFloat()
                }
                val X = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, -2.0f, 0.5f))
                show("hf", hf(X))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; hessian has no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
        val line = result.stdout.trim().lines().single { it.startsWith("hf ") }
        val parts = line.split(" ", limit = 3)
        assertEquals("3x3", parts[1], "H dims")
        val got = parts[2].split(",").map { it.toFloat() }
        val want = listOf(4f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 4f)
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < 1e-4f, "H[$i]=${got[i]} want ${want[i]} (got $got)")
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jacobian-test").toFile()
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
