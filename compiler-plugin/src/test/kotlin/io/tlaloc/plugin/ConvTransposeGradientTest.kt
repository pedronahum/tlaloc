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
 * §0.4.391 — differentiating THROUGH a transposed conv in `grad {}`:
 * `x.convTranspose2d(w, …)` was a loud "no VJP rule registered for
 * CONV_TRANSPOSE2D" before [io.tlaloc.ir.passes.VjpRegistry.ConvTranspose2dRule],
 * even though the primal's FIR arm and host twin shipped in §0.4.384 and forward
 * mode already worked.
 *
 *  g1 = valueAndGrad2 { x, w → Σ convTranspose2d(x, w) }         stride 1
 *  g2 = valueAndGrad2 { x, w → Σ convTranspose2d(x, w, 2, 2, …) } the UPSAMPLING
 *     spelling — the user-facing `stride` maps to `lhs_dilation`, which is the case
 *     whose adjoint index inversion has a non-trivial divisibility test
 *
 * Oracle: central differences from `valueAndGrad2`'s own primal value, at the
 * ±1e-3 step and 5e-2 tolerance `Rank2NNFiniteDifferenceTest` and
 * `CnnBlockGradientTest` use. FD rather than a hand-written adjoint reference on
 * purpose: the adjoint formulas are already pinned against FD and against the JVP
 * cross-identity at IR level (`DxirConvTransposeVjpTest`), so re-deriving them here
 * would only re-test my own algebra. What this test adds is the whole plugin path —
 * FIR literal-attr folding, the sentinel-dim result type, synthesis of both fused
 * adjoints, and gradient accumulation into two rank-4 params.
 */
class ConvTransposeGradientTest {

    @Test
    fun `convTranspose2d gradient lowers through the plugin and matches central differences`() {
        val dataDecl = listOf("X1" to X1, "W1" to W1, "X2" to X2, "W2" to W2).joinToString("\n") {
            (name, v) -> "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.valueAndGrad2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.convTranspose2d
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val EPS = 1e-3f
                val g1 = valueAndGrad2 { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                         w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.convTranspose2d(w).sum().toFloat()
                }
                val g2 = valueAndGrad2 { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                         w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.convTranspose2d(w, 2, 2, 0, 0, 0, 0).sum().toFloat()
                }
                fun t(d: FloatArray, n: Int, c: Int, h: Int, w: Int) =
                    Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(n, c, h, w, d)
                val (loss1, dx1, dw1) = g1(t(X1, 1, 2, 4, 4), t(W1, 2, 3, 2, 2))
                println("loss1 " + loss1)
                report("dx1", dx1); report("dw1", dw1)
                val (loss2, dx2, dw2) = g2(t(X2, 1, 1, 3, 3), t(W2, 1, 1, 2, 2))
                println("loss2 " + loss2)
                report("dx2", dx2); report("dw2", dw2)
                for (i in X1.indices) {
                    val p = X1.copyOf(); p[i] = p[i] + EPS
                    val m = X1.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = g1(t(p, 1, 2, 4, 4), t(W1, 2, 3, 2, 2))
                    val (lm, _, _) = g1(t(m, 1, 2, 4, 4), t(W1, 2, 3, 2, 2))
                    println("fdx1_${'$'}i " + ((lp - lm) / (2f * EPS)))
                }
                for (i in W1.indices) {
                    val p = W1.copyOf(); p[i] = p[i] + EPS
                    val m = W1.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = g1(t(X1, 1, 2, 4, 4), t(p, 2, 3, 2, 2))
                    val (lm, _, _) = g1(t(X1, 1, 2, 4, 4), t(m, 2, 3, 2, 2))
                    println("fdw1_${'$'}i " + ((lp - lm) / (2f * EPS)))
                }
                for (i in X2.indices) {
                    val p = X2.copyOf(); p[i] = p[i] + EPS
                    val m = X2.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = g2(t(p, 1, 1, 3, 3), t(W2, 1, 1, 2, 2))
                    val (lm, _, _) = g2(t(m, 1, 1, 3, 3), t(W2, 1, 1, 2, 2))
                    println("fdx2_${'$'}i " + ((lp - lm) / (2f * EPS)))
                }
                for (i in W2.indices) {
                    val p = W2.copyOf(); p[i] = p[i] + EPS
                    val m = W2.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = g2(t(X2, 1, 1, 3, 3), t(p, 1, 1, 2, 2))
                    val (lm, _, _) = g2(t(X2, 1, 1, 3, 3), t(m, 1, 1, 2, 2))
                    println("fdw2_${'$'}i " + ((lp - lm) / (2f * EPS)))
                }
            }
            fun report(tag: String, t: DTensor<*, F32>) {
                println("${'$'}{tag}Dims " + t.dims.joinToString(","))
                println("${'$'}{tag} " + t.hostF32().joinToString(","))
            }
            $dataDecl
        """.trimIndent()
        val result = compileAndRun(STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back to the runtime tape; expected the convT gradient to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = parse(result.stdout)
        // Stride 1: x [1,2,4,4] ⋆ IOHW w [2,3,2,2] → y [1,3,3,3]; gradients land back
        // on x's and w's shapes.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx1"], "dx1 shape\n${result.stdout}")
        assertEquals(listOf(2, 3, 2, 2), vals.dims["dw1"], "dw1 shape\n${result.stdout}")
        // Stride 2 (lhs_dilation): x [1,1,3,3] → y [1,1,4,4], the upsampling case.
        assertEquals(listOf(1, 1, 3, 3), vals.dims["dx2"], "dx2 shape\n${result.stdout}")
        assertEquals(listOf(1, 1, 2, 2), vals.dims["dw2"], "dw2 shape\n${result.stdout}")

        for (tag in listOf("dx1", "dw1", "dx2", "dw2")) {
            val got = vals.values.getValue(tag)
            assertTrue(got.any { abs(it) > 1e-4f }, "$tag is all zeros:\n${result.stdout}")
            for (i in got.indices) {
                val fd = vals.values["f${tag}_$i"]?.singleOrNull()
                    ?: error("no central difference for $tag[$i] in:\n${result.stdout}")
                val tol = 5e-2f * maxOf(1.0f, abs(fd))
                assertTrue(
                    abs(got[i] - fd) <= tol,
                    "$tag[$i] = ${got[i]} but central difference says $fd (tol $tol)\n" +
                        "stdout:\n${result.stdout}",
                )
            }
        }
    }

    private data class Reported(val dims: Map<String, List<Int>>, val values: Map<String, List<Float>>)

    private fun parse(stdout: String): Reported {
        val dims = HashMap<String, List<Int>>()
        val values = HashMap<String, List<Float>>()
        for (line in stdout.trim().lines()) {
            val parts = line.trim().split(" ", limit = 2)
            if (parts.size != 2 || parts[1].isEmpty()) continue
            if (parts[0].endsWith("Dims")) {
                dims[parts[0].removeSuffix("Dims")] = parts[1].split(",").map { it.trim().toInt() }
            } else {
                values[parts[0]] = parts[1].split(",").map { it.trim().toFloat() }
            }
        }
        return Reported(dims, values)
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-convt-test").toFile()
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
        private val X1 = FloatArray(1 * 2 * 4 * 4) { ((it * 37) % 23 - 11) / 8.0f }
        private val W1 = FloatArray(2 * 3 * 2 * 2) { ((it * 53) % 19 - 9) / 6.0f }
        private val X2 = FloatArray(1 * 1 * 3 * 3) { ((it * 29) % 17 - 8) / 5.0f }
        private val W2 = FloatArray(1 * 1 * 2 * 2) { ((it * 41) % 13 - 6) / 4.0f }

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun valueAndGrad2(
                f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float,
            ): (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) ->
                Triple<Float, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>> =
                { _, _ -> Triple(
                    -1.0f,
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32),
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32),
                ) }
        """.trimIndent()
    }
}
