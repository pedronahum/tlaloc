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
 * §0.4.384 — Phase A3b slice 1 E2E: rank-4 NCHW conv reaches user code and
 * differentiates in FORWARD mode through the real K2 plugin.
 *
 * This is the first rank-4 tensor surface the synthesis accepts, so it exercises
 * the whole slice-1 substrate at once: the `isAcceptedTensorType` widen, the
 * `Rank4` shape-witness construction in `deriveResultIrTypeRank4`, the FIR arm
 * that folds the user's stride/padding literals into `window_strides` /
 * `padding` attrs (and turns symbolic extents into -1 sentinel result dims), and
 * the `conv2dGeneral` synthesis arm.
 *
 *  j1 = jvp { x → Σ conv(x, x, pad=1) }                 3×3 same-padded self-conv
 *  j2 = valueAndJvp { … }                               the same, with the primal
 *  j3 = jvp { x → Σ conv(x, x, stride=2, pad=[1,0,1,1]) }  4×4 kernel, asymmetric
 *
 * Reverse mode is deliberately NOT part of this test — see
 * [reverseModeConvIsRejectedLoudlyNotSilentlyWrong] below and Conv2dRule's
 * §0.4.384 guard.
 */
class ConvForwardIntrinsicTest {

    @Test
    fun `forward-mode conv lowers through the plugin and matches an independent reference`() {
        // The generated source and the reference below read the SAME arrays, so
        // they cannot drift apart.
        val dataDecl = listOf("X3" to X3, "V3" to V3, "X4" to X4, "V4" to V4).joinToString("\n") {
            (name, v) -> "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndJvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val j1 = jvp { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(x, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                val j2 = valueAndJvp { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(x, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                val j3 = jvp { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(x, 2, 2, 1, 0, 1, 1).sum().toFloat()
                }
                // The 1-argument arity: a valid conv (stride 1, no padding), which
                // over a 3×3 input with a 3×3 kernel collapses to one output.
                val j4 = jvp { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(x).sum().toFloat()
                }
                val x3 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 3, 3, X3)
                val v3 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 3, 3, V3)
                val x4 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 4, 4, X4)
                val v4 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 4, 4, V4)
                println("dy1 " + j1(x3, v3))
                val (y2, dy2) = j2(x3, v3)
                println("y2 " + y2)
                println("dy2 " + dy2)
                println("dy3 " + j3(x4, v4))
                println("dy4 " + j4(x3, v3))
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
            "synthesis fell back to the runtime tape; expected rank-4 conv to synthesise. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val vals = result.stdout.trim().lines().associate {
            val (k, num) = it.trim().split(" ", limit = 2)
            k to num.toFloat()
        }
        assertTrue(vals["dy1"] != -1f, "j1 sentinel returned — the rewrite never fired.\n${result.stdout}")

        // f(x) = Σ conv(x, x): quadratic in x, so the directional derivative is the
        // product rule's two terms — conv(dx, x) + conv(x, dx), each summed.
        val x3 = X3.map { it.toDouble() }.toDoubleArray()
        val v3 = V3.map { it.toDouble() }.toDoubleArray()
        val x4 = X4.map { it.toDouble() }.toDoubleArray()
        val v4 = V4.map { it.toDouble() }.toDoubleArray()
        val wantY2 = refConv(x3, 3, 3, x3, 3, 3, 1, 1, 1, 1, 1, 1).sum()
        val wantDy1 = refConv(v3, 3, 3, x3, 3, 3, 1, 1, 1, 1, 1, 1).sum() +
            refConv(x3, 3, 3, v3, 3, 3, 1, 1, 1, 1, 1, 1).sum()
        val wantDy3 = refConv(v4, 4, 4, x4, 4, 4, 2, 2, 1, 0, 1, 1).sum() +
            refConv(x4, 4, 4, v4, 4, 4, 2, 2, 1, 0, 1, 1).sum()
        // j4 is the valid conv: a single output, so f = ⟨x, x⟩ and dy = 2⟨x, dx⟩.
        val wantDy4 = refConv(v3, 3, 3, x3, 3, 3, 1, 1, 0, 0, 0, 0).sum() +
            refConv(x3, 3, 3, v3, 3, 3, 1, 1, 0, 0, 0, 0).sum()

        assertTrue(
            abs(vals.getValue("dy1") - wantDy1) < 1e-3f,
            "dy1=${vals["dy1"]} want $wantDy1 (all: $vals)\n${result.stdout}",
        )
        assertTrue(abs(vals.getValue("y2") - wantY2) < 1e-3f, "y2=${vals["y2"]} want $wantY2")
        assertTrue(abs(vals.getValue("dy2") - wantDy1) < 1e-3f, "dy2=${vals["dy2"]} want $wantDy1")
        assertTrue(abs(vals.getValue("dy3") - wantDy3) < 1e-2f, "dy3=${vals["dy3"]} want $wantDy3")
        assertTrue(abs(vals.getValue("dy4") - wantDy4) < 1e-3f, "dy4=${vals["dy4"]} want $wantDy4")
    }

    /**
     * Reverse-mode conv is a COMPILE ERROR, not a wrong gradient.
     *
     * Conv2dRule solves its adjoint `padding` from the primal's extents, and every
     * `grad {}` param carries -1 sentinels, so the solved padding is arithmetic
     * garbage (a stride-1 padding-1 conv comes out `[[-3,1],[-3,1]]` instead of
     * `[[1,1],[1,1]]`) — and the interpreter, the emitter and the host twins all
     * faithfully honour whatever attrs they are handed. §0.4.384 added the guard
     * that turns that into a loud failure; this pins that it surfaces at the call
     * site as `NOT_DIFFERENTIABLE` (error severity) rather than at run time, or
     * worse, not at all.
     */
    @Test
    fun reverseModeConvIsRejectedLoudlyNotSilentlyWrong() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.conv2d
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> ->
                    x.conv2d(x, 1, 1, 1, 1, 1, 1).sum().toFloat()
                }
                val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 3, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f))
                println("gx " + g(x).dims.toList())
            }
        """.trimIndent()
        val result = compileAndRun(STUB, src)
        val errors = result.messages.filter { it.severity == CompilerMessageSeverity.ERROR }
        assertTrue(
            errors.any { "Conv2dRule" in it.message && "symbolic dims" in it.message },
            "expected the Conv2dRule sentinel guard as a compile error; got:\n" +
                errors.joinToString("\n--\n") { it.message },
        )
    }

    /**
     * Naive single-channel NCHW 2-D convolution in Double, written straight from
     * the definition (map each tap back to an input coordinate, skip the taps that
     * land outside). Deliberately NOT the `:core/ops` host twin: the synthesised
     * code calls that twin, so comparing against it would prove nothing.
     */
    private fun refConv(
        a: DoubleArray,
        h: Int,
        w: Int,
        k: DoubleArray,
        kh: Int,
        kw: Int,
        sH: Int,
        sW: Int,
        pT: Int,
        pB: Int,
        pL: Int,
        pR: Int,
    ): DoubleArray {
        val hOut = (h + pT + pB - kh) / sH + 1
        val wOut = (w + pL + pR - kw) / sW + 1
        val out = DoubleArray(hOut * wOut)
        for (y in 0 until hOut) {
            for (x in 0 until wOut) {
                var acc = 0.0
                for (ky in 0 until kh) {
                    for (kx in 0 until kw) {
                        val iy = y * sH + ky - pT
                        val ix = x * sW + kx - pL
                        if (iy in 0 until h && ix in 0 until w) acc += a[iy * w + ix] * k[ky * kw + kx]
                    }
                }
                out[y * wOut + x] = acc
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
        val tempDir = Files.createTempDirectory("tlaloc-conv-test").toFile()
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
        private val X3 = floatArrayOf(0.5f, -1.25f, 0.75f, 2.0f, -0.5f, 1.5f, -1.75f, 0.25f, 1.0f)
        private val V3 = floatArrayOf(0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, 0.9f)
        private val X4 = floatArrayOf(
            1.0f, -0.5f, 0.25f, 2.0f,
            -1.5f, 0.75f, 1.25f, -0.25f,
            0.5f, 1.0f, -1.0f, 0.5f,
            -0.75f, 1.5f, 0.25f, -1.25f,
        )
        private val V4 = floatArrayOf(
            0.2f, -0.4f, 0.6f, 0.8f,
            -0.1f, 0.3f, -0.5f, 0.7f,
            0.9f, -0.2f, 0.4f, -0.6f,
            0.15f, -0.35f, 0.55f, 0.05f,
        )

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun jvp(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float =
                { _, _ -> -1.0f }
            fun valueAndJvp(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
            fun grad(f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>) -> DTensor<Rank4<Sym, Sym, Sym, Sym>, F32> =
                { x -> DTensor(HostF32Storage(FloatArray(9) { -1.0f }), intArrayOf(1, 1, 3, 3), F32) }
        """.trimIndent()
    }
}
