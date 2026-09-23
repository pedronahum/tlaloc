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
 * §0.4.386 — Phase A3b: `avgPool2d` reaches user code and differentiates in
 * `grad {}`, via the fused runtime-extent [io.tlaloc.ir.OpKind.AVGPOOL2D_GRAD].
 *
 * Three spellings, chosen to stress the window inversion rather than to look
 * pretty:
 *
 *  g1 = grad { x → Σ avgPool2d(x, 2×2) }            non-overlapping, exact tiling
 *  g2 = grad { x → Σ avgPool2d(x, 3×3, s=2, p=1) }  overlapping, padded, AND a
 *                                                   floor-division remainder
 *                                                   (5 rows at stride 2 leave the
 *                                                   last one uncovered)
 *  g3 = grad { x → Σ avgPool2d(relu(x), 2×2) }      chained primal, so the adjoint's
 *                                                   shape template is a
 *                                                   forward-derived node rather
 *                                                   than a param
 */
class AvgPoolGradientTest {

    @Test
    fun `avgPool2d gradient lowers through the plugin and matches an independent reference`() {
        val dataDecl = listOf("XA" to XA, "XB" to XB, "XC" to XC).joinToString("\n") {
            (name, v) -> "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.avgPool2d
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g1 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.avgPool2d(2, 2).sum().toFloat()
                }
                val g2 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.avgPool2d(3, 3, 2, 2, 1, 1, 1, 1).sum().toFloat()
                }
                val g3 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.relu().avgPool2d(2, 2).sum().toFloat()
                }
                // The loss READS the pooled value, so the primal AVGPOOL2D survives
                // into the gradient body's value stream and its own synthesis arm
                // is exercised (g1–g3 only ever reach AVGPOOL2D_GRAD).
                val g4 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    val p = x.avgPool2d(2, 2)
                    (p * p).sum().toFloat()
                }
                report("dx1", g1(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, XA)))
                report("dx2", g2(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 3, 5, 5, XB)))
                report("dx3", g3(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, XC)))
                report("dx4", g4(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, XA)))
            }
            fun report(tag: String, t: DTensor<*, F32>) {
                println("${"$"}{tag}Dims " + t.dims.joinToString(","))
                println("${"$"}{tag} " + t.hostF32().joinToString(","))
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
            "synthesis fell back to the runtime tape; expected the avgpool gradient to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = parse(result.stdout)
        // g1: non-overlapping 2×2 over [1,2,4,4] → every input covered exactly once.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx1"], "dx1 shape\n${result.stdout}")
        assertClose("dx1", vals.values.getValue("dx1"), refAvgPoolSumGrad(1, 2, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0), result.stdout)
        // g2: 3×3 window, stride 2, pad 1 over [2,3,5,5] → output 3×3, and the
        // stride-2 walk over 5 rows/cols leaves a remainder the inversion must crop.
        assertEquals(listOf(2, 3, 5, 5), vals.dims["dx2"], "dx2 shape\n${result.stdout}")
        assertClose("dx2", vals.values.getValue("dx2"), refAvgPoolSumGrad(2, 3, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1), result.stdout)
        // g3: the conv sees relu(x), so the pooled gradient is masked by step(x).
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx3"], "dx3 shape\n${result.stdout}")
        val pooled = refAvgPoolSumGrad(1, 2, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0)
        val wantDx3 = DoubleArray(pooled.size) { i -> if (XC[i] > 0f) pooled[i] else 0.0 }
        assertClose("dx3", vals.values.getValue("dx3"), wantDx3, result.stdout)
        // g4: f = Σ p² with p = avgPool2d(x), so the adjoint's upstream is 2p rather
        // than the all-ones of g1–g3 — and the primal AVGPOOL2D is recomputed into
        // the body to produce it.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx4"], "dx4 shape\n${result.stdout}")
        val p4 = refAvgPool(XA, 1, 2, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0)
        val wantDx4 = refAvgPoolGrad(
            DoubleArray(p4.size) { 2.0 * p4[it] },
            1, 2, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0,
        )
        assertClose("dx4", vals.values.getValue("dx4"), wantDx4, result.stdout)
    }

    private fun assertClose(tag: String, got: List<Float>, want: DoubleArray, stdout: String) {
        assertEquals(want.size, got.size, "$tag size (stdout:\n$stdout)")
        for (i in want.indices) {
            assertTrue(
                abs(got[i] - want[i]) <= 1e-5f * maxOf(1.0, abs(want[i])).toFloat(),
                "$tag[$i] = ${got[i]} want ${want[i]}\nstdout:\n$stdout",
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

    /**
     * Forward avgpool from the definition (count_include_pad: divide by the FULL
     * window). Not the host twin — that is what the synthesised code calls.
     */
    private fun refAvgPool(
        x: FloatArray,
        nB: Int, c: Int, h: Int, w: Int,
        kh: Int, kw: Int, sH: Int, sW: Int,
        pT: Int, pB: Int, pL: Int, pR: Int,
    ): DoubleArray {
        val hOut = (h + pT + pB - kh) / sH + 1
        val wOut = (w + pL + pR - kw) / sW + 1
        val out = DoubleArray(nB * c * hOut * wOut)
        var idx = 0
        for (n in 0 until nB) {
            for (ch in 0 until c) {
                val base = (n * c + ch) * h * w
                for (y in 0 until hOut) {
                    for (xo in 0 until wOut) {
                        var acc = 0.0
                        for (ky in 0 until kh) {
                            val iy = y * sH + ky - pT
                            if (iy !in 0 until h) continue
                            for (kx in 0 until kw) {
                                val ix = xo * sW + kx - pL
                                if (ix !in 0 until w) continue
                                acc += x[base + iy * w + ix].toDouble()
                            }
                        }
                        out[idx++] = acc / (kh * kw)
                    }
                }
            }
        }
        return out
    }

    /**
     * The avgpool adjoint by SCATTER — the transpose of the primal's own window
     * loop, for an arbitrary [up]. Each output spreads its upstream over the window
     * taps that were in range, divided by the FULL window size (count_include_pad,
     * the primal's convention). Written from the definition and structurally
     * different from the fused adjoint, which INVERTS the window per input element
     * instead: an off-by-one in either the stride divisibility test or the padding
     * offset shows up as a mismatch.
     */
    private fun refAvgPoolGrad(
        up: DoubleArray,
        nB: Int, c: Int, h: Int, w: Int,
        kh: Int, kw: Int, sH: Int, sW: Int,
        pT: Int, pB: Int, pL: Int, pR: Int,
    ): DoubleArray {
        val hOut = (h + pT + pB - kh) / sH + 1
        val wOut = (w + pL + pR - kw) / sW + 1
        val dx = DoubleArray(nB * c * h * w)
        for (n in 0 until nB) {
            for (ch in 0 until c) {
                val xBase = (n * c + ch) * h * w
                val upBase = (n * c + ch) * hOut * wOut
                for (y in 0 until hOut) {
                    for (xo in 0 until wOut) {
                        val u = up[upBase + y * wOut + xo] / (kh * kw)
                        for (ky in 0 until kh) {
                            val iy = y * sH + ky - pT
                            if (iy !in 0 until h) continue
                            for (kx in 0 until kw) {
                                val ix = xo * sW + kx - pL
                                if (ix !in 0 until w) continue
                                dx[xBase + iy * w + ix] += u
                            }
                        }
                    }
                }
            }
        }
        return dx
    }

    /** The adjoint of `f(x) = Σ avgPool2d(x)`, where every output's upstream is 1. */
    private fun refAvgPoolSumGrad(
        nB: Int, c: Int, h: Int, w: Int,
        kh: Int, kw: Int, sH: Int, sW: Int,
        pT: Int, pB: Int, pL: Int, pR: Int,
    ): DoubleArray {
        val hOut = (h + pT + pB - kh) / sH + 1
        val wOut = (w + pL + pR - kw) / sW + 1
        return refAvgPoolGrad(
            DoubleArray(nB * c * hOut * wOut) { 1.0 },
            nB, c, h, w, kh, kw, sH, sW, pT, pB, pL, pR,
        )
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-avgpool-test").toFile()
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
        private val XA = FloatArray(1 * 2 * 4 * 4) { ((it * 31) % 19 - 9) / 7.0f }
        private val XB = FloatArray(2 * 3 * 5 * 5) { ((it * 17) % 29 - 14) / 11.0f }
        private val XC = FloatArray(1 * 2 * 4 * 4) { ((it * 23) % 21 - 10) / 6.0f }

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32) }
        """.trimIndent()
    }
}
