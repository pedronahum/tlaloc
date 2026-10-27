package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The two XLA:GPU faults [CompileOptions.withXlaGpuFixes] works around, reduced to one program
 * each and run through a default CUDA [PjrtSession]. Each test also prints what the same program
 * gives without the fix, so an XLA upgrade shows whether the override is still needed.
 */
class PjrtXlaGpuFixesTest {

    private val on = CompileOptions.OptionValue.Flag(true)

    /** max |got - want| / max |want|. */
    private fun relErr(got: FloatArray, want: DoubleArray): Double =
        got.indices.maxOf { kotlin.math.abs(got[it] - want[it]) } / want.maxOf { kotlin.math.abs(it) }

    /**
     * Four products `(Σ_g x_i[b, h, g, s, :]) transposed to rows (b, s), columns (h, d), times
     * Wᵀ`, the shape of the key-projection input gradients of four microbatches through
     * grouped-query attention. XLA's dot merger concatenates them (they share W); with v2 GEMM
     * fusions the layout copy that transposes each sum is lost, and products 1 to 3 are wrong.
     */
    @Test
    fun transposedOperandsOfAMergedDot() {
        assumeTrue(TestBackend.target == PjrtTarget.Cuda, "XLA:GPU only")
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val n = 4
        val args = (0 until n).joinToString(", ") { "%g$it: tensor<2x2x2x32x16xf32>" } + ", %w: tensor<64x32xf32>"
        val body = (0 until n).joinToString("\n") { i ->
            """
            |  %z$i = stablehlo.constant dense<0.0> : tensor<f32>
            |  %r$i = stablehlo.reduce(%g$i init: %z$i) applies stablehlo.add across dimensions = [2] : (tensor<2x2x2x32x16xf32>, tensor<f32>) -> tensor<2x2x32x16xf32>
            |  %t$i = stablehlo.transpose %r$i, dims = [0, 2, 1, 3] : (tensor<2x2x32x16xf32>) -> tensor<2x32x2x16xf32>
            |  %m$i = stablehlo.reshape %t$i : (tensor<2x32x2x16xf32>) -> tensor<64x32xf32>
            |  %d$i = stablehlo.dot_general %m$i, %wt, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x64xf32>) -> tensor<64x64xf32>
            """.trimMargin()
        }
        val results = List(n) { "tensor<64x64xf32>" }.joinToString(", ")
        val mlir = "func.func public @main($args) -> ($results) {\n" +
            "  %wt = stablehlo.transpose %w, dims = [1, 0] : (tensor<64x32xf32>) -> tensor<32x64xf32>\n" +
            body + "\n  return " + (0 until n).joinToString(", ") { "%d$it" } + " : $results\n}\n"
        val r = Random(4)
        val gs = List(n) { FloatArray(2 * 2 * 2 * 32 * 16) { r.nextFloat() - 0.5f } }
        val w = FloatArray(64 * 32) { r.nextFloat() - 0.5f }
        val want = gs.map { g ->
            // m[(b, s), (h, d)] = Σ_q g[b, h, q, s, d]; out = m · Wᵀ.
            val m = DoubleArray(64 * 32)
            for (b in 0 until 2) for (h in 0 until 2) for (q in 0 until 2) for (s in 0 until 32) for (d in 0 until 16) {
                m[(b * 32 + s) * 32 + h * 16 + d] += g[(((b * 2 + h) * 2 + q) * 32 + s) * 16 + d].toDouble()
            }
            DoubleArray(64 * 64) { k -> val row = k / 64; val col = k % 64; (0 until 32).sumOf { m[row * 32 + it] * w[col * 32 + it] } }
        }
        fun run(options: CompileOptions): List<Double> =
            PjrtSession(target = PjrtTarget.Cuda, diskCache = null, compileOptions = options).use { s ->
                val ins = gs.map { s.bufferFromHostF32(it, listOf(2, 2, 2, 32, 16)) } + s.bufferFromHostF32(w, listOf(64, 32))
                val outs = s.executeStablehlo(mlir, ins)
                val errs = outs.mapIndexed { i, o -> relErr(o.toFloatArray(64 * 64), want[i]) }
                (ins + outs).forEach { it.close() }
                errs
            }
        val unfixed = run(CompileOptions.DEFAULT.withOverride("xla_gpu_experimental_gemm_fusion_v2", on))
        println("PjrtXlaGpuFixesTest: without the fix the products are ${unfixed.joinToString { "%.1e".format(it) }} from the reference")
        val fixed = run(CompileOptions.DEFAULT)
        // TF32 dots: a few 1e-4.
        assertTrue(fixed.all { it < 2e-3 }, "relative errors $fixed")
    }

    /** An FP8 product with K = 16: a Triton FP8 GEMM tiled 16 deep along K returns twice the product. */
    @Test
    fun fp8ProductWithAShortContraction() {
        assumeTrue(TestBackend.target == PjrtTarget.Cuda, "XLA:GPU only")
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val m = 8
        val k = 16
        val n = 256
        val mlir = """
            |func.func public @main(%a: tensor<${m}x${k}xf8E4M3FN>, %b: tensor<${k}x${n}xf8E4M3FN>) -> tensor<${m}x${n}xf32> {
            |  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<${m}x${k}xf8E4M3FN>, tensor<${k}x${n}xf8E4M3FN>) -> tensor<${m}x${n}xf32>
            |  return %d : tensor<${m}x${n}xf32>
            |}
        """.trimMargin()
        val r = Random(8)
        // Finite e4m3fn codes (0x7F and 0xFF are NaN).
        fun codes(size: Int) = ByteArray(size) { (r.nextInt(0, 0x60) or if (r.nextBoolean()) 0x80 else 0).toByte() }
        val a = codes(m * k)
        val b = codes(k * n)
        val want = DoubleArray(m * n) { i ->
            (0 until k).sumOf { f8e4m3fnToFloat(a[(i / n) * k + it]).toDouble() * f8e4m3fnToFloat(b[it * n + i % n]) }
        }
        fun run(options: CompileOptions): Double =
            PjrtSession(target = PjrtTarget.Cuda, diskCache = null, compileOptions = options).use { s ->
                val ins = listOf(s.bufferFromHostBytes(a, listOf(m, k), F8E4M3FN), s.bufferFromHostBytes(b, listOf(k, n), F8E4M3FN))
                val out = s.executeStablehlo(mlir, ins).single()
                val err = relErr(out.toFloatArray(m * n), want)
                (ins + out).forEach { it.close() }
                err
            }
        val unfixed = run(CompileOptions.DEFAULT.withOverride("xla_gpu_enable_triton_gemm", on))
        println("PjrtXlaGpuFixesTest: without the fix the FP8 product is %.2f from the reference (1.00 is twice it)".format(unfixed))
        val fixed = run(CompileOptions.DEFAULT)
        assertTrue(fixed < 1e-5, "relative error $fixed")
    }
}
