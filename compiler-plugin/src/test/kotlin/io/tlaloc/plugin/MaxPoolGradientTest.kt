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
 * §0.4.389 — Phase A3b: `maxPool2d` reaches user code and differentiates in
 * `grad {}`, via the fused [io.tlaloc.ir.OpKind.MAXPOOL2D_GRAD]. This is the last
 * piece of the pooling/conv surface, and the one the plan had parked LAST behind
 * two blockers that turned out to be artefacts of the old spelling: the rank-6
 * upsample intermediates (whose types bake `n`/`c`/`Ho`/`Wo`, meaningless under
 * sentinels, and which the synthesis has no rank-6 witness for) and therefore the
 * single-representative `tensorIrType` generalisation. Inverting the window per
 * input element needs neither.
 *
 *  g1 = grad { x → Σ maxPool2d(x, 2×2) }          non-overlapping, exact tiling
 *  g2 = grad { x → Σ maxPool2d(relu(x), 2×2) }    chained primal: the adjoint's
 *                                                 shape template is a
 *                                                 forward-derived node, and relu's
 *                                                 step mask is exercised
 *  g3 = grad { x → Σ maxPool2d(x, 2×2) } over 5×5 a floor-division remainder: the
 *                                                 last row/column is covered by no
 *                                                 window and must get zero. The
 *                                                 host and interpreter handle it;
 *                                                 the StableHLO upsample expansion
 *                                                 deliberately does not.
 *
 * The all-ties convention (every within-window winner gets the full upstream, vs
 * XLA `select_and_scatter`'s single winner) is pinned at IR level on both engines by
 * `DxirHostConvParityTest.maxPoolAdjointRoutesFullUpstreamToEveryTie`, where the data
 * is hand-built to tie; asserting it from random-ish E2E data would be brittle.
 */
class MaxPoolGradientTest {

    @Test
    fun `maxPool2d gradient lowers through the plugin and matches an independent reference`() {
        val dataDecl = listOf("X1" to X1, "X2" to X2, "X3" to X3).joinToString("\n") { (name, v) ->
            "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.maxPool2d
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g1 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.maxPool2d(2, 2).sum().toFloat()
                }
                val g2 = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.relu().maxPool2d(2, 2).sum().toFloat()
                }
                report("dx1", g1(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, X1)))
                report("dx2", g2(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 4, 4, X2)))
                report("dx3", g1(Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 5, 5, X3)))
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
            "synthesis fell back to the runtime tape; expected the maxpool gradient to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = parse(result.stdout)
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx1"], "dx1 shape\n${result.stdout}")
        assertClose("dx1", vals.values.getValue("dx1"), refMaxPoolSumGrad(X1, 1, 2, 4, 4, 2, 2), result.stdout)

        // g2: the pooled gradient is masked by relu's derivative (step: 0 at exactly 0).
        // The all-ties convention itself is pinned deterministically at IR level by
        // `DxirHostConvParityTest.maxPoolAdjointRoutesFullUpstreamToEveryTie` — here
        // the relu'd data does tie, but the ties are at zero and the mask zeroes them
        // too, so asserting on tie counts from this data would be brittle rather than
        // meaningful.
        assertEquals(listOf(1, 2, 4, 4), vals.dims["dx2"], "dx2 shape\n${result.stdout}")
        val pooled2 = refMaxPoolSumGrad(relu(X2), 1, 2, 4, 4, 2, 2)
        val wantDx2 = DoubleArray(pooled2.size) { i -> if (X2[i] > 0f) pooled2[i] else 0.0 }
        assertClose("dx2", vals.values.getValue("dx2"), wantDx2, result.stdout)

        // g3: 5×5 with a 2×2 window tiles 2×2 and leaves the last row/column
        // uncovered — those inputs must get exactly zero.
        assertEquals(listOf(1, 1, 5, 5), vals.dims["dx3"], "dx3 shape\n${result.stdout}")
        val dx3 = vals.values.getValue("dx3")
        assertClose("dx3", dx3, refMaxPoolSumGrad(X3, 1, 1, 5, 5, 2, 2), result.stdout)
        for (i in 0 until 5) {
            assertEquals(0f, dx3[4 * 5 + i], "the uncovered last row must get no gradient:\n${result.stdout}")
            assertEquals(0f, dx3[i * 5 + 4], "the uncovered last column must get no gradient:\n${result.stdout}")
        }
    }

    private fun relu(v: FloatArray): FloatArray = FloatArray(v.size) { if (v[it] > 0f) v[it] else 0f }

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
     * The gradient of `f(x) = Σ maxPool2d(x)` by SCATTER — the transpose of the
     * primal's own window loop. Every output's upstream is 1, so: take each window's
     * max, then give 1 to EVERY in-range tap that equals it. Structurally different
     * from the fused adjoint (which inverts the window per input element) and from
     * the emitter's upsample-and-mask expansion, so an off-by-one or a
     * single-winner tie policy cannot agree with it by accident.
     */
    private fun refMaxPoolSumGrad(
        x: FloatArray, nB: Int, c: Int, h: Int, w: Int, kh: Int, kw: Int,
    ): DoubleArray {
        val sH = kh
        val sW = kw
        val hOut = (h - kh) / sH + 1
        val wOut = (w - kw) / sW + 1
        val dx = DoubleArray(nB * c * h * w)
        for (n in 0 until nB) {
            for (ch in 0 until c) {
                val base = (n * c + ch) * h * w
                for (y in 0 until hOut) {
                    for (xo in 0 until wOut) {
                        var max = Double.NEGATIVE_INFINITY
                        for (ky in 0 until kh) {
                            for (kx in 0 until kw) {
                                val v = x[base + (y * sH + ky) * w + (xo * sW + kx)].toDouble()
                                if (v > max) max = v
                            }
                        }
                        for (ky in 0 until kh) {
                            for (kx in 0 until kw) {
                                val idx = base + (y * sH + ky) * w + (xo * sW + kx)
                                // ALL within-window ties win, each getting the full upstream.
                                if (x[idx].toDouble() == max) dx[idx] += 1.0
                            }
                        }
                    }
                }
            }
        }
        return dx
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-maxpool-test").toFile()
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
        private val X1 = FloatArray(1 * 2 * 4 * 4) { ((it * 31) % 19 - 9) / 7.0f }
        // Deliberately coarse-quantised so relu produces exact zeros and 2×2 windows
        // tie: the all-ties convention is invisible on generic random data.
        private val X2 = FloatArray(1 * 2 * 4 * 4) { ((it * 7) % 5 - 3).toFloat() }
        private val X3 = FloatArray(1 * 1 * 5 * 5) { ((it * 13) % 11 - 5) / 3.0f }

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
