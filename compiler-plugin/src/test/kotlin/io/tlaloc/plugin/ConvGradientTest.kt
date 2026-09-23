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
 * §0.4.385 — Phase A3b: REVERSE-mode conv reaches user code. `grad { x, w -> … }`
 * over a rank-4 NCHW convolution lowers through the K2 plugin, synthesises to host
 * calls with no fallback, and produces gradients that match an independent
 * reference — the first rank-4 gradient the synthesis has ever emitted.
 *
 * What makes this work is the fused adjoint ops (`CONV2D_DATA_ADJOINT` /
 * `CONV2D_KERNEL_ADJOINT`): [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule] used to
 * SOLVE the adjoint padding from the primal's extents at transform time, and every
 * `grad {}` param carries -1 sentinels, so it baked arithmetic garbage that every
 * consumer honoured faithfully. The fused ops carry the target tensor as a
 * shape-only template operand and solve at execution time.
 */
class ConvGradientTest {

    @Test
    fun `conv2d gradient lowers through the plugin and matches an independent reference`() {
        val dataDecl = listOf("XD" to XD, "WD" to WD, "XS" to XS, "WS" to WS).joinToString("\n") {
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
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                // The general-attr spelling: strides [2,1], asymmetric padding
                // [[1,0],[1,1]] — the case whose dW padding-high is NEGATIVE (the
                // crop that absorbs the stride-2 floor-division remainder).
                val g1 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(w, 2, 1, 1, 0, 1, 1).sum().toFloat()
                }
                // The plain spelling: stride 1, symmetric pad 1.
                val g2 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(w, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                val x1 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 3, 5, 4, XD)
                val w1 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 3, 3, 2, WD)
                val (dx1, dw1) = g1(x1, w1)
                report("dx1", dx1)
                report("dw1", dw1)
                val x2 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, XS)
                val w2 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 3, 3, WS)
                val (dx2, dw2) = g2(x2, w2)
                report("dx2", dx2)
                report("dw2", dw2)
                // A chained primal: the conv's input is a RELU result, so dX's
                // shape template is a forward-derived node rather than a param.
                val g3 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                w: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.relu().conv2d(w, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                val (dx3, dw3) = g3(x2, w2)
                report("dx3", dx3)
                report("dw3", dw3)
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
            "synthesis fell back to the runtime tape; expected the conv gradient to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = parse(result.stdout)
        // Case 1: x [2,3,5,4] ⋆ w [2,3,3,2], strides [2,1], padding [[1,0],[1,1]].
        assertEquals(listOf(2, 3, 5, 4), vals.dims["dx1"], "dx1 shape\n${result.stdout}")
        assertEquals(listOf(2, 3, 3, 2), vals.dims["dw1"], "dw1 shape\n${result.stdout}")
        val (wantDx1, wantDw1) = refConvSumGrad(
            XD, 2, 3, 5, 4, WD, 2, 3, 3, sH = 2, sW = 1, pT = 1, pB = 0, pL = 1, pR = 1,
        )
        assertClose("dx1", vals.values.getValue("dx1"), wantDx1, result.stdout)
        assertClose("dw1", vals.values.getValue("dw1"), wantDw1, result.stdout)

        // Case 2: x [1,2,4,4] ⋆ w [1,2,3,3], stride 1, padding 1 all round.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx2"], "dx2 shape\n${result.stdout}")
        assertEquals(listOf(1, 2, 3, 3), vals.dims["dw2"], "dw2 shape\n${result.stdout}")
        val (wantDx2, wantDw2) = refConvSumGrad(
            XS, 1, 2, 4, 4, WS, 1, 2, 3, sH = 1, sW = 1, pT = 1, pB = 1, pL = 1, pR = 1,
        )
        assertClose("dx2", vals.values.getValue("dx2"), wantDx2, result.stdout)
        assertClose("dw2", vals.values.getValue("dw2"), wantDw2, result.stdout)

        // Case 3: the conv sees relu(x), so its gradient scatters into d(relu(x))
        // and RELU's own adjoint masks that by step(x) — 0 at exactly 0, STEP's
        // documented convention. dw3 therefore differs from dw2: it accumulates the
        // relu'd input, not the raw one.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx3"], "dx3 shape\n${result.stdout}")
        assertEquals(listOf(1, 2, 3, 3), vals.dims["dw3"], "dw3 shape\n${result.stdout}")
        val xr = FloatArray(XS.size) { if (XS[it] > 0f) XS[it] else 0f }
        val (wantG3, wantDw3) = refConvSumGrad(
            xr, 1, 2, 4, 4, WS, 1, 2, 3, sH = 1, sW = 1, pT = 1, pB = 1, pL = 1, pR = 1,
        )
        val wantDx3 = DoubleArray(wantG3.size) { i -> if (XS[i] > 0f) wantG3[i] else 0.0 }
        assertClose("dx3", vals.values.getValue("dx3"), wantDx3, result.stdout)
        assertClose("dw3", vals.values.getValue("dw3"), wantDw3, result.stdout)
    }

    private fun assertClose(tag: String, got: List<Float>, want: DoubleArray, stdout: String) {
        assertEquals(want.size, got.size, "$tag size (stdout:\n$stdout)")
        for (i in want.indices) {
            assertTrue(
                abs(got[i] - want[i]) <= 1e-4f * maxOf(1.0, abs(want[i])).toFloat(),
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
     * The gradient of `f(x,w) = Σ conv(x,w)` by SCATTER — the transpose of the
     * primal's own loop. Every output element contributes 1 (the loss is a plain
     * sum), so each in-range tap adds the kernel weight into `dx` and the input
     * value into `dw`. Out-of-range taps are simply never visited, which is why
     * this needs NO padding solve at all: it is structurally different from both
     * the host engine and the interpreter (which gather through a solved padding),
     * so a mis-solved pad cannot make the two sides agree by sharing a bug.
     */
    private fun refConvSumGrad(
        x: FloatArray,
        nB: Int,
        cIn: Int,
        h: Int,
        w: Int,
        k: FloatArray,
        cOut: Int,
        cKIn: Int,
        kh: Int,
        sH: Int,
        sW: Int,
        pT: Int,
        pB: Int,
        pL: Int,
        pR: Int,
    ): Pair<DoubleArray, DoubleArray> {
        assertEquals(cKIn, cIn, "kernel input channels must match the input's")
        val kw = k.size / (cOut * cIn * kh)
        val hOut = (h + pT + pB - kh) / sH + 1
        val wOut = (w + pL + pR - kw) / sW + 1
        val dx = DoubleArray(nB * cIn * h * w)
        val dw = DoubleArray(cOut * cIn * kh * kw)
        for (n in 0 until nB) {
            for (o in 0 until cOut) {
                for (y in 0 until hOut) {
                    for (xo in 0 until wOut) {
                        for (i in 0 until cIn) {
                            for (ky in 0 until kh) {
                                for (kx in 0 until kw) {
                                    val iy = y * sH + ky - pT
                                    val ix = xo * sW + kx - pL
                                    if (iy !in 0 until h || ix !in 0 until w) continue
                                    val wIdx = ((o * cIn + i) * kh + ky) * kw + kx
                                    val xIdx = ((n * cIn + i) * h + iy) * w + ix
                                    dx[xIdx] += k[wIdx].toDouble()
                                    dw[wIdx] += x[xIdx].toDouble()
                                }
                            }
                        }
                    }
                }
            }
        }
        return dx to dw
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-convgrad-test").toFile()
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
        // x [2,3,5,4] ⋆ w [2,3,3,2] — the general-attr case.
        private val XD = FloatArray(2 * 3 * 5 * 4) { ((it * 37) % 23 - 11) / 8.0f }
        private val WD = FloatArray(2 * 3 * 3 * 2) { ((it * 53) % 19 - 9) / 6.0f }

        // x [1,2,4,4] ⋆ w [1,2,3,3] — the plain stride-1 padded case.
        private val XS = FloatArray(1 * 2 * 4 * 4) { ((it * 29) % 17 - 8) / 5.0f }
        private val WS = FloatArray(1 * 2 * 3 * 3) { ((it * 41) % 13 - 6) / 4.0f }

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) ->
                        Pair<DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32),
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32),
                ) }
        """.trimIndent()
    }
}
