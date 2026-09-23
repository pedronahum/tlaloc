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
 * §0.4.406 — E2E for the multi-argument seeded/assembled intrinsics `vjp2` /
 * `valueAndVjp2` / `jacobian2` / `hessian2`, closing the "multi-arg" tails
 * §0.4.394 and §0.4.398 recorded. The §0.4.394 pattern: real plugin, the REAL
 * generic `:autograd` declarations off the test classpath, no monomorphic
 * stubs, and a synthesis fallback ("kept original call") is a hard failure —
 * none of these has a runtime-tape path.
 *
 * Oracles (a = [1, 2, 3], b = [4, -1, 0.5] unless noted):
 *  - `vjp2 { Σ(a⊙b) }` at scalar ȳ = 2 → (2b, 2a); at ȳ = 1 == `grad2`
 *    (the consistency identity).
 *  - `vjp2 { a⊙b }` at NON-UNIFORM tensor ȳ → (b⊙ȳ, a⊙ȳ) — a unit-seed
 *    impostor cannot pass.
 *  - the JVP⇄VJP inner-product identity over BOTH slots, computed
 *    numerically in the user program:
 *    ⟨ȳ, jvp2(a, b, va, vb)⟩ == ⟨ā, va⟩ + ⟨b̄, vb⟩.
 *  - `valueAndVjp2` → Triple(a⊙b, b⊙ȳ, a⊙ȳ).
 *  - `jacobian2 { a⊙b }` → (J_a = diag(b), J_b = diag(a)); a scalar-valued
 *    body degenerates to the two [1, n] gradient rows; a RECTANGULAR case
 *    (na=2, nb=3) pins the per-input column indexing.
 *  - `hessian2 { Σ(a⊙b) }` → [[0, I], [I, 0]] (H_aa = 0, H_ab = I);
 *    `hessian2 { Σ(a⊙a⊙b) }` → value-DEPENDENT blocks (H_aa = diag(2b),
 *    H_ab = H_ba = diag(2a), H_bb = 0); a rectangular Σa·Σb case pins the
 *    [(na+nb), (na+nb)] block layout with unequal extents.
 */
class MultiArgSeededIntrinsicTest {

    @Test
    fun `vjp2 and valueAndVjp2 lower through the plugin as seeded reverse pullbacks`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.jvp2
            import io.tlaloc.autograd.valueAndVjp2
            import io.tlaloc.autograd.vjp2
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
                val vs = vjp2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    (a * b).sum().toFloat()
                }
                val gs = grad2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    (a * b).sum().toFloat()
                }
                val vt = vjp2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> a * b }
                val jt = jvp2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> a * b }
                val pv = valueAndVjp2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> a * b }
                val A = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val B = Tensors.f32Vector<Sym>(floatArrayOf(4.0f, -1.0f, 0.5f))
                val Ybar = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, -1.0f, 2.0f))
                val VA = Tensors.f32Vector<Sym>(floatArrayOf(3.0f, 0.25f, -2.0f))
                val VB = Tensors.f32Vector<Sym>(floatArrayOf(-1.0f, 2.0f, 0.5f))
                val (sa, sb) = vs(A, B, 2.0f)
                show("vsa", sa)
                show("vsb", sb)
                val (ua, ub) = vs(A, B, 1.0f)
                val (ga, gb) = gs(A, B)
                show("vua", ua)
                show("vub", ub)
                show("gsa", ga)
                show("gsb", gb)
                val (ta, tb) = vt(A, B, Ybar)
                show("vta", ta)
                show("vtb", tb)
                val (py, pa, pb) = pv(A, B, Ybar)
                show("pvy", py)
                show("pva", pa)
                show("pvb", pb)
                // The JVP⇄VJP inner-product identity over BOTH argument slots:
                // ⟨ȳ, J_a·va + J_b·vb⟩ (forward) == ⟨ā, va⟩ + ⟨b̄, vb⟩ (reverse).
                println("fwdip 1 " + dot(Ybar, jt(A, B, VA, VB)))
                println("revip 1 " + (dot(ta, VA) + dot(tb, VB)))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; vjp2/valueAndVjp2 have no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val rows = parseRows(result.stdout)
        fun check(name: String, wantDims: String, want: List<Float>, tol: Float = 1e-4f) =
            checkRow(rows, result.stdout, name, wantDims, want, tol)

        // Scalar-R vjp2 at ȳ = 2: (ā, b̄) = (2b, 2a).
        check("vsa", "3", listOf(8f, -2f, 1f))
        check("vsb", "3", listOf(2f, 4f, 6f))
        // The grad2-consistency identity: vjp2 at the unit seed IS grad2.
        check("vua", "3", listOf(4f, -1f, 0.5f))
        check("vub", "3", listOf(1f, 2f, 3f))
        check("gsa", "3", listOf(4f, -1f, 0.5f))
        check("gsb", "3", listOf(1f, 2f, 3f))
        // Tensor-R vjp2 at non-uniform ȳ = [0.5, -1, 2]: ā = b⊙ȳ, b̄ = a⊙ȳ.
        check("vta", "3", listOf(2f, 1f, 1f))
        check("vtb", "3", listOf(0.5f, -2f, 6f))
        // valueAndVjp2: the true primal a⊙b rides along with the same pullbacks.
        check("pvy", "3", listOf(4f, -2f, 1.5f))
        check("pva", "3", listOf(2f, 1f, 1f))
        check("pvb", "3", listOf(0.5f, -2f, 6f))
        // ⟨ȳ, J_a·va + J_b·vb⟩ = ⟨b⊙ȳ, va⟩ + ⟨a⊙ȳ, vb⟩ — both sides computed
        // numerically in the user program: 4.25 + (-1.5) = 2.75.
        check("fwdip", "1", listOf(2.75f))
        check("revip", "1", listOf(2.75f))
    }

    @Test
    fun `jacobian2 and hessian2 lower through the plugin and assemble dense blocks`() {
        val src = """
            import io.tlaloc.autograd.hessian2
            import io.tlaloc.autograd.jacobian2
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
            fun main() {
                val jt = jacobian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> a * b }
                val js = jacobian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    (a * b).sum().toFloat()
                }
                val jr = jacobian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    a * b.sum().toFloat()
                }
                val h1 = hessian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    (a * b).sum().toFloat()
                }
                val h2 = hessian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    (a * a * b).sum().toFloat()
                }
                val hr = hessian2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    a.sum().toFloat() * b.sum().toFloat()
                }
                val A = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val B = Tensors.f32Vector<Sym>(floatArrayOf(4.0f, -1.0f, 0.5f))
                val P = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f))
                val (jta, jtb) = jt(A, B)
                show("jta", jta)
                show("jtb", jtb)
                val (jsa, jsb) = js(A, B)
                show("jsa", jsa)
                show("jsb", jsb)
                val (jra, jrb) = jr(P, B)
                show("jra", jra)
                show("jrb", jrb)
                show("h1", h1(A, B))
                show("h2", h2(A, B))
                show("hr", hr(P, B))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; jacobian2/hessian2 have no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val rows = parseRows(result.stdout)
        fun check(name: String, wantDims: String, want: List<Float>, tol: Float = 1e-4f) =
            checkRow(rows, result.stdout, name, wantDims, want, tol)

        // f = a⊙b: J_a = diag(b), J_b = diag(a).
        check("jta", "3x3", listOf(4f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 0.5f))
        check("jtb", "3x3", listOf(1f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 3f))
        // Scalar-valued f degenerates to the two [1, n] gradient rows (b, a).
        check("jsa", "1x3", listOf(4f, -1f, 0.5f))
        check("jsb", "1x3", listOf(1f, 2f, 3f))
        // RECTANGULAR (na=2, nb=3): f = a·Σb → J_a = Σb·I₂ [2, 2],
        // J_b[i, j] = aᵢ [2, 3] — the per-input column indexing pin (Σb = 3.5).
        check("jra", "2x2", listOf(3.5f, 0f, 0f, 3.5f))
        check("jrb", "2x3", listOf(1f, 1f, 1f, 2f, 2f, 2f))
        // f = Σ(a⊙b): H = [[0, I], [I, 0]] over the concatenated flat input.
        check(
            "h1", "6x6",
            listOf(
                0f, 0f, 0f, 1f, 0f, 0f,
                0f, 0f, 0f, 0f, 1f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f,
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f, 0f, 0f,
                0f, 0f, 1f, 0f, 0f, 0f,
            ),
        )
        // f = Σ(a⊙a⊙b): value-DEPENDENT blocks — H_aa = diag(2b) uses b's
        // VALUES and H_ab = diag(2a) uses a's, so a value-blind impostor
        // cannot pass. H_bb = 0.
        check(
            "h2", "6x6",
            listOf(
                8f, 0f, 0f, 2f, 0f, 0f,
                0f, -2f, 0f, 0f, 4f, 0f,
                0f, 0f, 1f, 0f, 0f, 6f,
                2f, 0f, 0f, 0f, 0f, 0f,
                0f, 4f, 0f, 0f, 0f, 0f,
                0f, 0f, 6f, 0f, 0f, 0f,
            ),
        )
        // RECTANGULAR (na=2, nb=3): f = Σa·Σb → H_aa = 0 [2, 2],
        // H_ab = 1 [2, 3], H_ba = 1 [3, 2], H_bb = 0 [3, 3] — pins the
        // [(na+nb), (na+nb)] block layout with unequal extents.
        check(
            "hr", "5x5",
            listOf(
                0f, 0f, 1f, 1f, 1f,
                0f, 0f, 1f, 1f, 1f,
                1f, 1f, 0f, 0f, 0f,
                1f, 1f, 0f, 0f, 0f,
                1f, 1f, 0f, 0f, 0f,
            ),
        )
    }

    private fun parseRows(stdout: String): Map<String, Pair<String, List<Float>>> =
        stdout.trim().lines().associate {
            val parts = it.trim().split(" ", limit = 3)
            parts[0] to (parts[1] to parts[2].split(",").map { s -> s.toFloat() })
        }

    private fun checkRow(
        rows: Map<String, Pair<String, List<Float>>>,
        stdout: String,
        name: String,
        wantDims: String,
        want: List<Float>,
        tol: Float,
    ) {
        val (dims, got) = rows[name] ?: error("no '$name' row in stdout:\n$stdout")
        assertEquals(wantDims, dims, "$name dims")
        assertEquals(want.size, got.size, "$name size")
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < tol, "$name[$i]=${got[i]} want ${want[i]} (got $got)")
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
        val tempDir = Files.createTempDirectory("tlaloc-multiarg-test").toFile()
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
