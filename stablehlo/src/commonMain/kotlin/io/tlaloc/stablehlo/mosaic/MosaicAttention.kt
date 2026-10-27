package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.BF16
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirEmitter
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.mosaicKernel
import io.tlaloc.ir.recognizer.kernel.MosaicKernel

/**
 * Causal flash attention as a differentiable op: [causalFlashAttention] emits the
 * [MosaicKernels.flashForward] kernel with a `vjp` that runs [MosaicKernels.flashBackwardDkDv]
 * and [MosaicKernels.flashBackwardDq]. On a TPU the three kernels run; elsewhere each falls back
 * to its reference, written in DXIR ops with the kernels' bf16 roundings, so a program that uses
 * it runs and differentiates on every target.
 */
@ExperimentalTlalocApi
object MosaicAttention {

    private val t = MosaicKernels.FLASH_BLOCK

    /**
     * Causal attention of one head: q, k, v bf16 `[seq, 128]`, softmax scale `1/√128`; returns
     * the output, bf16 `[seq, 128]`. [seq] is a multiple of 128.
     */
    fun DxirEmitter.causalFlashAttention(q: DxirNode, k: DxirNode, v: DxirNode): DxirNode {
        val seq = q.type.dims[0]
        val io = DxirType(BF16, listOf(seq, t))
        require(seq % t == 0 && listOf(q, k, v).all { it.type == io }) { "causalFlashAttention: q, k, v must be bf16 [seq, $t] with seq a multiple of $t, got ${q.type}, ${k.type}, ${v.type}" }
        val fwd = mosaicKernel(
            listOf(q, k, v),
            MosaicKernel.ofSerializedText("tlaloc_flash_fwd", MosaicKernels.flashForward(seq, serialized = true)),
            forwardReference(seq),
            referenceFallback = true,
            vjp = vjp(seq),
        )
        return fwd.result(0)
    }

    private fun scale() = (1.0 / kotlin.math.sqrt(t.toDouble())).toFloat()

    /** The helpers the references share. */
    private class Ops(val b: DxirBuilder, val seq: Int) {
        val f = DxirType(F32, listOf(seq, t))
        val square = DxirType(F32, listOf(seq, seq))
        val row = DxirType(F32, listOf(seq))
        fun f32(x: DxirNode) = b.op(OpKind.CAST, listOf(x), DxirType(F32, x.type.dims))
        fun bf16(x: DxirNode) = b.op(OpKind.CAST, listOf(x), DxirType(BF16, x.type.dims))
        /** The value rounded to bf16 and widened back, as the kernels feed bf16 to the MXU. */
        fun rounded(x: DxirNode) = f32(bf16(x))
        fun tr(x: DxirNode) = b.op(OpKind.TRANSPOSE, listOf(x), DxirType(F32, x.type.dims.reversed()), mapOf("permutation" to listOf(1, 0)))
        fun mm(x: DxirNode, y: DxirNode) = b.op(OpKind.MATMUL, listOf(x, y), DxirType(F32, listOf(x.type.dims[0], y.type.dims[1])))
        fun bin(kind: OpKind, x: DxirNode, y: DxirNode) = b.op(kind, listOf(x, y), x.type)
        fun scalar(v: Float, type: DxirType) =
            b.op(OpKind.BROADCAST, listOf(b.const(v, DxirType(F32, emptyList()))), type, mapOf("broadcast_dimensions" to emptyList<Int>()))
        fun rows(x: DxirNode, to: DxirType) = b.op(OpKind.BROADCAST, listOf(x), to, mapOf("broadcast_dimensions" to listOf(0)))
        fun reduce(kind: OpKind, x: DxirNode) = b.op(kind, listOf(x), row, mapOf("reduction_dims" to listOf(1)))

        /** Scaled scores with the causal mask: −∞ where the key follows the query. */
        fun scores(q: DxirNode, k: DxirNode): DxirNode {
            val s = bin(OpKind.MUL, mm(f32(q), tr(f32(k))), scalar(scale(), square))
            val mask = FloatArray(seq * seq) { val (r, c) = it / seq to it % seq; if (c <= r) 0f else Float.NEGATIVE_INFINITY }
            return bin(OpKind.ADD, s, b.const(mask, square))
        }

        /** The probabilities from the scores and the log-sum-exp (lanes equal). */
        fun probabilities(s: DxirNode, lse: DxirNode) = b.op(OpKind.EXP, listOf(bin(OpKind.SUB, s, rows(reduce(OpKind.MAX, lse), square))), square)
    }

    private fun forwardReference(seq: Int): DxirFunction = DxirBuilder.function("flash_fwd_reference") {
        val io = DxirType(BF16, listOf(seq, t))
        val q = param("q", io)
        val k = param("k", io)
        val v = param("v", io)
        with(Ops(this, seq)) {
            val s = scores(q, k)
            val m = reduce(OpKind.MAX, s)
            val p = op(OpKind.EXP, listOf(bin(OpKind.SUB, s, rows(m, square))), square)
            val l = reduce(OpKind.SUM, p)
            val o = bin(OpKind.DIV, mm(rounded(p), f32(v)), rows(l, f))
            val lse = rows(bin(OpKind.ADD, m, op(OpKind.LOG, listOf(l), row)), f)
            listOf(bf16(o), lse)
        }
    }

    private fun backwardReference(seq: Int, dq: Boolean): DxirFunction = DxirBuilder.function(if (dq) "flash_dq_reference" else "flash_dkdv_reference") {
        val io = DxirType(BF16, listOf(seq, t))
        val stat = DxirType(F32, listOf(seq, t))
        val q = param("q", io)
        val k = param("k", io)
        val v = param("v", io)
        val dO = param("do", io)
        val lse = param("lse", stat)
        val di = param("di", stat)
        with(Ops(this, seq)) {
            val p = probabilities(scores(q, k), lse)
            val dp = mm(f32(dO), tr(f32(v)))
            val ds = bin(OpKind.MUL, bin(OpKind.MUL, p, bin(OpKind.SUB, dp, rows(reduce(OpKind.MAX, di), square))), scalar(scale(), square))
            if (dq) listOf(bf16(mm(rounded(ds), f32(k))))
            else listOf(bf16(mm(tr(rounded(ds)), f32(q))), bf16(mm(tr(rounded(p)), f32(dO))))
        }
    }

    /** `(q, k, v, o, lse, dO, dlse) → (dq, dk, dv)`; lse is internal, so its cotangent is zero. */
    private fun vjp(seq: Int): DxirFunction = DxirBuilder.function("flash_vjp") {
        val io = DxirType(BF16, listOf(seq, t))
        val stat = DxirType(F32, listOf(seq, t))
        val q = param("q", io)
        val k = param("k", io)
        val v = param("v", io)
        val o = param("o", io)
        val lse = param("lse", stat)
        val dO = param("do", io)
        param("dlse", stat)
        with(Ops(this, seq)) {
            val di = rows(op(OpKind.SUM, listOf(bin(OpKind.MUL, f32(dO), f32(o))), row, mapOf("reduction_dims" to listOf(1))), f)
            val dkdv = mosaicKernel(
                listOf(q, k, v, dO, lse, di),
                MosaicKernel.ofSerializedText("tlaloc_flash_bwd_dkdv", MosaicKernels.flashBackwardDkDv(seq, serialized = true)),
                backwardReference(seq, dq = false),
                referenceFallback = true,
            )
            val dq = mosaicKernel(
                listOf(q, k, v, dO, lse, di),
                MosaicKernel.ofSerializedText("tlaloc_flash_bwd_dq", MosaicKernels.flashBackwardDq(seq, serialized = true)),
                backwardReference(seq, dq = true),
                referenceFallback = true,
            )
            listOf(dq.result(0), dkdv.result(0), dkdv.result(1))
        }
    }
}
