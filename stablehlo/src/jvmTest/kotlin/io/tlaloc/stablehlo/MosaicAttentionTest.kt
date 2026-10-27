package io.tlaloc.stablehlo

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import io.tlaloc.stablehlo.mosaic.MosaicAttention.causalFlashAttention
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [causalFlashAttention] in the interpreter, where the kernels run as their references: the
 * output and, through the kernel's vjp (the two backward kernels), the gradients of
 * `Σ attention(q, k, v) ⊙ w` for q, k and v, against the same quantities in f64.
 */
class MosaicAttentionTest {

    private val seq = 256
    private val d = 128

    /** bf16 rounding (to nearest even) of [x], as the interpreter rounds a CAST to bf16. */
    private fun bf16(x: Float): Float {
        val bits = x.toRawBits()
        return Float.fromBits((bits + 0x7FFF + ((bits ushr 16) and 1)) and 0xFFFF0000.toInt())
    }

    @Test
    fun theOutputAndTheKernelGradientsMatchF64() {
        val io = DxirType(BF16, listOf(seq, d))
        val fn = DxirBuilder.function("attn_loss") {
            val q = param("q", io)
            val k = param("k", io)
            val v = param("v", io)
            val w = param("w", DxirType(F32, listOf(seq, d)))
            val o = causalFlashAttention(q, k, v)
            val of = op(OpKind.CAST, listOf(o), DxirType(F32, listOf(seq, d)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(of, w), of.type)), DxirType(F32, emptyList())))
        }
        assertTrue(fn.body.any { it is DxirOp && it.op == OpKind.MOSAIC_KERNEL })
        val grad = DxirReverseTransform.apply(fn, includeForward = true, inputOnlyTrailingParams = 1)
        assertEquals(2, grad.body.count { it is DxirOp && it.op == OpKind.MOSAIC_KERNEL && it.types.size <= 2 && (it.attrs["vjp"] == null) })

        val r = Random(3)
        val q = FloatArray(seq * d) { bf16(r.nextFloat() - 0.5f) }
        val k = FloatArray(seq * d) { bf16(r.nextFloat() - 0.5f) }
        val v = FloatArray(seq * d) { bf16(r.nextFloat() - 0.5f) }
        val w = FloatArray(seq * d) { r.nextFloat() - 0.5f }
        val out = DxirInterpreter.evalFunction(grad, listOf(q, k, v, w))

        // f64: o = softmax(q kᵀ/√d, causal) v; dO = w; the standard gradients.
        val scale = 1.0 / sqrt(d.toDouble())
        val p = Array(seq) { DoubleArray(seq) }
        for (i in 0 until seq) {
            val s = DoubleArray(i + 1) { j -> (0 until d).sumOf { q[i * d + it].toDouble() * k[j * d + it] } * scale }
            val m = s.max()
            val e = s.map { exp(it - m) }
            val l = e.sum()
            for (j in 0..i) p[i][j] = e[j] / l
        }
        val o = Array(seq) { i -> DoubleArray(d) { c -> (0..i).sumOf { j -> p[i][j] * v[j * d + c] } } }
        val loss = (0 until seq).sumOf { i -> (0 until d).sumOf { c -> o[i][c] * w[i * d + c] } }
        val dv = Array(seq) { j -> DoubleArray(d) { c -> (j until seq).sumOf { i -> p[i][j] * w[i * d + c] } } }
        val dp = Array(seq) { i -> DoubleArray(i + 1) { j -> (0 until d).sumOf { c -> w[i * d + c].toDouble() * v[j * d + c] } } }
        val ds = Array(seq) { i ->
            val di = (0 until d).sumOf { c -> w[i * d + c] * o[i][c] }
            DoubleArray(i + 1) { j -> p[i][j] * (dp[i][j] - di) * scale }
        }
        val dq = Array(seq) { i -> DoubleArray(d) { c -> (0..i).sumOf { j -> ds[i][j] * k[j * d + c] } } }
        val dk = Array(seq) { j -> DoubleArray(d) { c -> (j until seq).sumOf { i -> ds[i][j] * q[i * d + c] } } }

        fun check(what: String, got: FloatArray, want: Array<DoubleArray>, tol: Double) {
            var worst = 0.0
            var scaleOf = 0.0
            for (i in want.indices) for (c in 0 until d) {
                worst = maxOf(worst, abs(got[i * d + c] - want[i][c]))
                scaleOf = maxOf(scaleOf, abs(want[i][c]))
            }
            println("MosaicAttentionTest: $what max|err| ${"%.3g".format(worst)} of max|value| ${"%.3g".format(scaleOf)}")
            assertTrue(worst <= tol * scaleOf, "$what: max error $worst > $tol × $scaleOf")
        }
        assertEquals(loss, out[0][0].toDouble(), 2e-2 * maxOf(1.0, abs(loss)))

        // For a TPU the gradient program is three kernels: the forward, dK/dV and dQ.
        val tpu = lowerMosaicKernels(grad, KernelTarget.GOOGLE_TPU_V6E).toStablehlo()
        assertEquals(3, Regex("tpu_custom_call").findAll(tpu).count(), "custom calls in the TPU gradient program")
        // Anywhere else, the references.
        val gpu = lowerMosaicKernels(grad, KernelTarget("nvidia", null))
        assertTrue(gpu.body.none { it is DxirOp && it.op == OpKind.MOSAIC_KERNEL })
        check("dq", out[1], dq, 2e-2)
        check("dk", out[2], dk, 2e-2)
        check("dv", out[3], dv, 2e-2)
    }
}
