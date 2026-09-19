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
 * §0.4.387 — the rank-4 layers COMPOSE: one `grad {}` body carrying
 * conv → relu → avgPool → sum, plus a skip connection, synthesises with no tape
 * fallback and its gradients survive central differences.
 *
 * The per-op E2E tests (`ConvGradientTest`, `AvgPoolGradientTest`) each certify one
 * layer against a hand-written reference. Nothing before this exercised them in
 * sequence, which is where composition bugs live: the gradient of the pooled relu'd
 * conv has to thread an AVGPOOL2D_GRAD into a CONV2D_DATA_ADJOINT and a
 * CONV2D_KERNEL_ADJOINT, with a RELU/STEP mask between them, and the skip term means
 * `dx` accumulates TWO rank-4 contributions rather than being a single call's result.
 *
 * Oracle: central differences at ±1e-3 computed from `valueAndGrad2`'s own primal
 * value, with the 5e-2 tolerance `Rank2NNFiniteDifferenceTest` uses (loose enough
 * for f32 noise and relu's kink, tight enough that a wrong window inversion or a
 * mis-solved padding cannot hide). The forward path is independently pinned by
 * `ConvForwardIntrinsicTest`, so borrowing its values here is not circular.
 */
class CnnBlockGradientTest {

    @Test
    fun `conv relu avgpool plus a skip connection compose in grad and match central differences`() {
        val dataDecl = listOf("XD" to XD, "WD" to WD).joinToString("\n") { (name, v) ->
            "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.valueAndGrad2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.avgPool2d
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val vg = valueAndGrad2 { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                         w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    // conv (pad 1, so 4×4 stays 4×4) → relu → 2×2 non-overlapping
                    // pool → sum, PLUS a skip term so `dx` accumulates two
                    // rank-4 contributions instead of being one call's result.
                    val h = x.conv2d(w, 1, 1, 1, 1, 1, 1).relu()
                    (h.avgPool2d(2, 2).sum() + x.sum()).toFloat()
                }
                fun mkX(d: FloatArray) = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, d)
                fun mkW(d: FloatArray) = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 2, 3, 3, d)
                val (loss, dx, dw) = vg(mkX(XD), mkW(WD))
                println("loss " + loss)
                report("dx", dx)
                report("dw", dw)
                val EPS = 1e-3f
                for (i in XD.indices) {
                    val p = XD.copyOf(); p[i] = p[i] + EPS
                    val m = XD.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = vg(mkX(p), mkW(WD))
                    val (lm, _, _) = vg(mkX(m), mkW(WD))
                    println("fdx${'$'}i " + ((lp - lm) / (2f * EPS)))
                }
                for (i in WD.indices) {
                    val p = WD.copyOf(); p[i] = p[i] + EPS
                    val m = WD.copyOf(); m[i] = m[i] - EPS
                    val (lp, _, _) = vg(mkX(XD), mkW(p))
                    val (lm, _, _) = vg(mkX(XD), mkW(m))
                    println("fdw${'$'}i " + ((lp - lm) / (2f * EPS)))
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
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back to the runtime tape; expected the CNN block to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = parse(result.stdout)
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx"], "dx shape\n${result.stdout}")
        assertEquals(listOf(2, 2, 3, 3), vals.dims["dw"], "dw shape\n${result.stdout}")

        val dx = vals.values.getValue("dx")
        val dw = vals.values.getValue("dw")
        // Not vacuous: a body that silently produced zeros would still "match" a
        // zero FD, so require the gradients to actually carry signal.
        assertTrue(dx.any { abs(it) > 1e-3f }, "dx is all zeros:\n${result.stdout}")
        assertTrue(dw.any { abs(it) > 1e-3f }, "dw is all zeros:\n${result.stdout}")

        assertMatchesFd("dx", dx, vals, result.stdout)
        assertMatchesFd("dw", dw, vals, result.stdout)
    }

    private fun assertMatchesFd(tag: String, got: List<Float>, vals: Reported, stdout: String) {
        for (i in got.indices) {
            // The generated program prints the central differences as `fdx<i>` /
            // `fdw<i>`, one value per line, alongside the gradient arrays `dx`/`dw`.
            val fd = vals.values["f$tag$i"]?.singleOrNull()
                ?: error("no central difference for $tag[$i] in:\n$stdout")
            val tol = 5e-2f * maxOf(1.0f, abs(fd))
            assertTrue(
                abs(got[i] - fd) <= tol,
                "$tag[$i] = ${got[i]} but central difference says $fd (tol $tol)\nstdout:\n$stdout",
            )
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
        val tempDir = Files.createTempDirectory("tlaloc-cnn-test").toFile()
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
        // x [1,2,4,4] against an OIHW kernel [Co=2, Ci=2, 3, 3].
        private val XD = FloatArray(1 * 2 * 4 * 4) { ((it * 37) % 23 - 11) / 8.0f }
        private val WD = FloatArray(2 * 2 * 3 * 3) { ((it * 53) % 19 - 9) / 6.0f }

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
