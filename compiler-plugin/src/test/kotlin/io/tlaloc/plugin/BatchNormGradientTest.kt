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
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.390 — Phase A3b: TRAINING-mode `batchNorm` differentiates in `grad {}` with
 * no new VjpRule, interpreter arm or synthesis arm. The FIR lowering desugars it
 * into `mean → sub → mul → mean → add(eps) → sqrt → div → mul(γ) → add(β)`, and
 * every one of those already has a sentinel-safe adjoint (MEAN via SUM_TO's runtime
 * template, the binaries via Phase A5c broadcasting, the `[C] → [1,C,1,1]` reshapes
 * since §0.4.375).
 *
 * This is also the first `grad {}` body that MIXES RANKS — a rank-4 `x` against
 * rank-1 `scale`/`offset` — so it is the empirical answer to whether the
 * single-representative `tensorIrType` blocks mixed-rank surfaces or whether
 * per-node derivation covers them.
 *
 * Two oracles, because one is not enough. Finite differences alone would be
 * circular: `valueAndGrad2`'s value and its gradients come from the SAME lowered
 * graph, so a mis-wiring that is consistent in both directions (say scale and
 * offset swapped) would agree with its own central differences perfectly. Hence the
 * primal value is also checked against an independent Double implementation of the
 * batchnorm formula — that pins the wiring, and FD then pins the derivative of the
 * thing that was verified to be the right function.
 */
class BatchNormGradientTest {

    @Test
    fun `training batchNorm lowers through the plugin and matches an independent reference`() {
        val dataDecl = listOf("XD" to XD, "SD" to SD).joinToString("\n") { (name, v) ->
            "val $name = floatArrayOf(${v.joinToString(", ") { "${it}f" }})"
        }
        val src = """
            import io.tlaloc.autograd.valueAndGrad2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.batchNorm
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val STEP = 1e-3f
                // scale = s, offset = relu(s): deliberately ASYMMETRIC, so a swapped
                // scale/offset in the lowering changes the primal value and the
                // independent reference catches it. `eps` is an inlined LITERAL and
                // not the `STEP` val: the FIR arm folds it into a splat const, so a
                // variable reference there is (correctly) not lowerable.
                val vg = valueAndGrad2 { x: DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>,
                                         s: DTensor<Rank1<Sym>, F32> ->
                    // Σ y², NOT Σ y: the sum of a batch-normalised tensor is exactly
                    // N·H·W·β per channel (the normalised values sum to zero by
                    // construction), so `Σ y` has an identically-zero dx and central
                    // differences would agree with it vacuously.
                    val y = x.batchNorm(s, s.relu(), 1e-2f)
                    (y * y).sum().toFloat()
                }
                fun mkX(d: FloatArray) = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 3, 4, 4, d)
                fun mkS(d: FloatArray) = Tensors.f32Vector<Sym>(d)
                val (loss, dx, ds) = vg(mkX(XD), mkS(SD))
                println("loss " + loss)
                report("dx", dx)
                report("ds", ds)
                for (i in XD.indices) {
                    val p = XD.copyOf(); p[i] = p[i] + STEP
                    val m = XD.copyOf(); m[i] = m[i] - STEP
                    val (lp, _, _) = vg(mkX(p), mkS(SD))
                    val (lm, _, _) = vg(mkX(m), mkS(SD))
                    println("fdx${'$'}i " + ((lp - lm) / (2f * STEP)))
                }
                for (i in SD.indices) {
                    val p = SD.copyOf(); p[i] = p[i] + STEP
                    val m = SD.copyOf(); m[i] = m[i] - STEP
                    // The LOSS at the perturbed scale/offset — component 1, not the
                    // gradients, which are what this difference approximates.
                    val (lp, _, _) = vg(mkX(XD), mkS(p))
                    val (lm, _, _) = vg(mkX(XD), mkS(m))
                    println("fds${'$'}i " + ((lp - lm) / (2f * STEP)))
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
            "synthesis fell back to the runtime tape; expected the mixed-rank batchNorm body " +
                "to synthesise. Warnings:\n" +
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message },
        )

        val vals = parse(result.stdout)
        // The stub returns -1.0f sentinels; catch a rewrite that never fired before
        // the shape assertions make it look like a shape bug.
        val gotLoss = vals.values.getValue("loss").single()
        assertTrue(gotLoss != -1.0f, "stub sentinel returned — the rewrite never fired.\n${result.stdout}")
        assertEquals(listOf(2, 3, 4, 4), vals.dims["dx"], "dx shape\n${result.stdout}")
        assertEquals(listOf(3), vals.dims["ds"], "ds shape (rank-1 gradient)\n${result.stdout}")

        // Oracle 1: the primal value against an independent implementation. This is
        // what makes the FD check below non-circular.
        val wantLoss = refBatchNormSqSum(XD, 2, 3, 4, 4, SD, relu(SD), 1e-2)
        assertTrue(
            abs(gotLoss - wantLoss) < 1e-2f,
            "loss=$gotLoss but the independent reference says $wantLoss — the " +
                "desugaring does not compute training batchNorm\n${result.stdout}",
        )

        // Oracle 2: both gradients against central differences of that verified function.
        val dx = vals.values.getValue("dx")
        val ds = vals.values.getValue("ds")
        assertTrue(dx.any { abs(it) > 1e-4f }, "dx is all zeros:\n${result.stdout}")
        assertTrue(ds.any { abs(it) > 1e-4f }, "ds is all zeros:\n${result.stdout}")
        assertMatchesFd("dx", dx, vals, result.stdout)
        assertMatchesFd("ds", ds, vals, result.stdout)
    }

    private fun relu(v: FloatArray): FloatArray = FloatArray(v.size) { if (v[it] > 0f) v[it] else 0f }

    /**
     * `Σ batchNorm(x)²` from the definition, in Double: per channel, μ and the BIASED
     * ν over the batch and spatial extents, then `((x−μ)/√(ν+eps)·γ + β)²`. Written
     * independently of both the host twin and the desugaring. Squared for the reason
     * given at the call site — the plain sum is constant in `x`.
     */
    private fun refBatchNormSqSum(
        x: FloatArray,
        nB: Int,
        c: Int,
        h: Int,
        w: Int,
        gamma: FloatArray,
        beta: FloatArray,
        eps: Double,
    ): Double {
        val spatial = h * w
        val count = (nB * spatial).toDouble()
        var total = 0.0
        for (ch in 0 until c) {
            var sum = 0.0
            for (n in 0 until nB) {
                val base = (n * c + ch) * spatial
                for (i in 0 until spatial) sum += x[base + i].toDouble()
            }
            val mean = sum / count
            var sq = 0.0
            for (n in 0 until nB) {
                val base = (n * c + ch) * spatial
                for (i in 0 until spatial) {
                    val e = x[base + i].toDouble() - mean
                    sq += e * e
                }
            }
            val invStd = 1.0 / sqrt(sq / count + eps)
            for (n in 0 until nB) {
                val base = (n * c + ch) * spatial
                for (i in 0 until spatial) {
                    val y = (x[base + i].toDouble() - mean) * invStd * gamma[ch] + beta[ch]
                    total += y * y
                }
            }
        }
        return total
    }

    private fun assertMatchesFd(tag: String, got: List<Float>, vals: Reported, stdout: String) {
        for (i in got.indices) {
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
        val tempDir = Files.createTempDirectory("tlaloc-batchnorm-test").toFile()
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
        // x [2,3,4,4] and per-channel scale/offset [3].
        private val XD = FloatArray(2 * 3 * 4 * 4) { ((it * 37) % 23 - 11) / 8.0f }
        private val SD = FloatArray(3) { ((it * 11) % 7 - 3) / 2.0f }

        private val STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank4
            import io.tlaloc.core.Sym
            fun valueAndGrad2(
                f: (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank1<Sym>, F32>) -> Float,
            ): (DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank1<Sym>, F32>) ->
                Triple<Float, DTensor<Rank4<Sym, Sym, Sym, Sym>, F32>, DTensor<Rank1<Sym>, F32>> =
                { _, _ -> Triple(
                    -1.0f,
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1, 1, 1, 1), F32),
                    DTensor(HostF32Storage(FloatArray(1) { -1.0f }), intArrayOf(1), F32),
                ) }
        """.trimIndent()
    }
}
