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
 * §0.4.387 — `jvp2` / `valueAndJvp2`: forward mode for a two-argument function,
 * the follow-up §0.4.375 scoped out ("multi-arg `jvp2` is a clean follow-up: same
 * pattern, more params").
 *
 * The point is not the extra plumbing — [io.tlaloc.ir.passes.DxirForwardTransform]
 * already emits all primals then all tangents for any arity, and the IR extension's
 * forward branch splits returns at `size / 2`, so both were arity-agnostic. The
 * point is what it UNBLOCKS: a single-argument `jvp` cannot express a convolution
 * against a separate kernel, so `ConvForwardIntrinsicTest` had to differentiate a
 * SELF-convolution `conv(x, x)`. With `jvp2` the tangent is the genuine bilinear
 * product rule over a real `(x, w)` pair, `∂f/∂x·dx + ∂f/∂w·dw`, in one pass.
 *
 *  j1 = jvp2 { a, b → Σ (a ⊙ b) }              elementwise: the plumbing, and a
 *                                               check that dy sums BOTH terms
 *  j2 = valueAndJvp2 { x, w → Σ conv(x, w) }    the conv case, primal and tangent
 */
class Jvp2IntrinsicTest {

    @Test
    fun `two-argument forward mode lowers through the plugin and matches an independent reference`() {
        val dataDecl = listOf("AD" to AD, "BD" to BD, "XD" to XD, "WD" to WD, "VA" to VA, "VB" to VB,
            "VX" to VX, "VW" to VW).joinToString("\n") { (name, v) ->
            "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.jvp2
            import io.tlaloc.autograd.valueAndJvp2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val j1 = jvp2 { a: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                b: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    (a * b).sum().toFloat()
                }
                val j2 = valueAndJvp2 { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                        w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(w, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                fun t3(d: FloatArray) = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 3, 3, d)
                fun t4(d: FloatArray) = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 4, 4, d)
                println("dy1 " + j1(t3(AD), t3(BD), t3(VA), t3(VB)))
                val (y2, dy2) = j2(t4(XD), t3(WD), t4(VX), t3(VW))
                println("y2 " + y2)
                println("dy2 " + dy2)
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
            "synthesis fell back to the runtime tape; expected jvp2/valueAndJvp2 to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = result.stdout.trim().lines().associate {
            val (k, num) = it.trim().split(" ", limit = 2)
            k to num.toFloat()
        }
        assertTrue(vals["dy1"] != -1f, "j1 sentinel returned — the rewrite never fired.\n${result.stdout}")

        val a = AD.map { it.toDouble() }.toDoubleArray()
        val b = BD.map { it.toDouble() }.toDoubleArray()
        val va = VA.map { it.toDouble() }.toDoubleArray()
        val vb = VB.map { it.toDouble() }.toDoubleArray()
        val x = XD.map { it.toDouble() }.toDoubleArray()
        val w = WD.map { it.toDouble() }.toDoubleArray()
        val vx = VX.map { it.toDouble() }.toDoubleArray()
        val vw = VW.map { it.toDouble() }.toDoubleArray()

        // Σ(a⊙b) is bilinear, so its directional derivative is both terms summed.
        var wantDy1 = 0.0
        for (i in a.indices) wantDy1 += va[i] * b[i] + a[i] * vb[i]
        assertTrue(
            abs(vals.getValue("dy1") - wantDy1) < 1e-2f,
            "dy1=${vals["dy1"]} want $wantDy1\n${result.stdout}",
        )

        // Σ conv(x, w): the product rule's two convolutions, each summed. Padded
        // 3×3 over a 4×4 input, so the output is 4×4 again.
        val wantY2 = refConv(x, 4, 4, w, 3, 3).sum()
        val wantDy2 = refConv(vx, 4, 4, w, 3, 3).sum() + refConv(x, 4, 4, vw, 3, 3).sum()
        assertTrue(
            abs(vals.getValue("y2") - wantY2) < 1e-3f,
            "y2=${vals["y2"]} want $wantY2\n${result.stdout}",
        )
        assertTrue(
            abs(vals.getValue("dy2") - wantDy2) < 1e-2f,
            "dy2=${vals["dy2"]} want $wantDy2\n${result.stdout}",
        )
    }

    /**
     * Naive single-channel NCHW 2-D conv in Double, stride 1 with symmetric pad 1,
     * written from the definition. Not the `:core/ops` host twin — the synthesised
     * code calls that, so comparing against it would prove nothing.
     */
    private fun refConv(a: DoubleArray, h: Int, w: Int, k: DoubleArray, kh: Int, kw: Int): DoubleArray {
        val out = DoubleArray(h * w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var acc = 0.0
                for (ky in 0 until kh) {
                    for (kx in 0 until kw) {
                        val iy = y + ky - 1
                        val ix = x + kx - 1
                        if (iy in 0 until h && ix in 0 until w) acc += a[iy * w + ix] * k[ky * kw + kx]
                    }
                }
                out[y * w + x] = acc
            }
        }
        return out
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jvp2-test").toFile()
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
        private val AD = FloatArray(9) { ((it * 13) % 11 - 5) / 4.0f }
        private val BD = FloatArray(9) { ((it * 29) % 13 - 6) / 5.0f }
        private val VA = FloatArray(9) { ((it * 7) % 9 - 4) / 3.0f }
        private val VB = FloatArray(9) { ((it * 17) % 11 - 5) / 6.0f }
        private val XD = FloatArray(16) { ((it * 37) % 23 - 11) / 8.0f }
        private val WD = FloatArray(9) { ((it * 53) % 19 - 9) / 6.0f }
        private val VX = FloatArray(16) { ((it * 11) % 17 - 8) / 7.0f }
        private val VW = FloatArray(9) { ((it * 19) % 13 - 6) / 5.0f }

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun jvp2(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                     DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float =
                { _, _, _, _ -> -1.0f }
            fun valueAndJvp2(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                     DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Pair<Float, Float> =
                { _, _, _, _ -> Pair(-1.0f, -1.0f) }
        """.trimIndent()
    }
}
