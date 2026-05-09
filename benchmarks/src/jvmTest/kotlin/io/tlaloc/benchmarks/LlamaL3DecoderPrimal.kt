package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.323 — Llama-3-shape decoder primal exercising GQA. Parallel to
 * [LlamaDecoderPrimal] (the §0.4.271 single-head MHA model) but with
 * grouped-query attention on the K and V path: K and V are projected
 * at a smaller dim and expanded via the canonical PyTorch `repeat_kv`
 * form (`RESHAPE → BROADCAST → RESHAPE`).
 *
 * Purpose: a structural exerciser for the §0.4.320 + §0.4.322
 * GroupedQueryAttention recognizer + coarsener arc on a model with
 * the rest of the Llama-3 primitives present (RmsNorm × 2, RoPE,
 * SwiGLU, CrossEntropy). Built primarily for the recognition coverage
 * sentinel and to give downstream lowering a real GQA-shape function
 * to consume once the kernel-template emit path lands.
 *
 * # Scope vs [LlamaDecoderPrimal]
 *
 * - **Rank-2 attention scoring** (same as parent). Score matrix is
 *   `[tokens, tokens]`, no head-splitting on the attention path. Real
 *   Llama-3 keeps a head axis (`[B, H, T, T]` scores); doing the
 *   head-split rewrite here would require batched-matmul realism that
 *   isn't load-bearing for the recognizer story. The GQA dim-shape
 *   structure (K and V at `kvDim < dModel`, expanded) is what the
 *   recognizer matches.
 *
 * - **GQA via `RESHAPE → BROADCAST → RESHAPE`** on K and V. This is
 *   the form the §0.4.322 coarsener handles when no TRANSPOSE follows
 *   the chain.
 *
 * - **TRANSPOSE on K^T** (same as parent). Tlaloc's `MATMUL` contracts
 *   `last(A) × first(B)`, so attention scores need an explicit
 *   pre-transposed K. With TRANSPOSE in the chain after the GQA
 *   expansion, the **recognizer fires** but the **§0.4.322 coarsener
 *   declines** (TRANSPOSE in `outerOps` is out of v1+v2 scope). v3 will
 *   add TRANSPOSE inversion to handle this case end-to-end.
 *
 * - **Other simplifications inherit from [LlamaDecoderPrimal]**:
 *   unweighted RmsNorm, RoPE on Q only, eps-stabilised RmsNorm.
 */

/**
 * Hyperparameters for [LlamaL3DecoderPrimal]. Adds a `nKvHeads` field
 * to the parent's [LlamaDecoderConfig] shape; everything else follows
 * the same dim arithmetic.
 *
 * @property nKvHeads number of KV heads. Must divide `nHeads`. The
 *   group ratio is `nHeads / nKvHeads` (≥ 2 for GQA, == nHeads for MQA).
 */
data class LlamaL3DecoderConfig(
    val batch: Int,
    val seq: Int,
    val dModel: Int,
    val nHeads: Int,
    val nKvHeads: Int,
    val headDim: Int,
    val ffnMult: Double,
    val vocab: Int,
) {
    val dFf: Int get() = (dModel * ffnMult).toInt()
    val tokens: Int get() = batch * seq
    val groupRatio: Int get() = nHeads / nKvHeads
    val kvDim: Int get() = nKvHeads * headDim

    init {
        require(dModel == nHeads * headDim) {
            "dModel ($dModel) must equal nHeads * headDim ($nHeads * $headDim = ${nHeads * headDim})"
        }
        require(nHeads % nKvHeads == 0) {
            "nHeads ($nHeads) must be divisible by nKvHeads ($nKvHeads)"
        }
        require(nHeads / nKvHeads >= 2) {
            "GQA requires group ratio nHeads / nKvHeads >= 2; got ${nHeads / nKvHeads}"
        }
        require(batch > 0 && seq > 0 && vocab > 0 && dModel > 0 && nHeads > 0 && nKvHeads > 0)
    }

    companion object {
        /** CI-friendly tiny config; group_ratio = 2 (4 Q-heads, 2 KV-heads). */
        val tiny = LlamaL3DecoderConfig(
            batch = 2, seq = 16, dModel = 64, nHeads = 4, nKvHeads = 2,
            headDim = 16, ffnMult = 4.0, vocab = 256,
        )

        /**
         * Medium config matching [LlamaDecoderConfig.medium]'s scale
         * (tokens=256, dModel=512, dFf=2048, vocab=2048) but with
         * GQA: 8 Q-heads, 2 KV-heads → group_ratio = 4.
         */
        val medium = LlamaL3DecoderConfig(
            batch = 1, seq = 256, dModel = 512, nHeads = 8, nKvHeads = 2,
            headDim = 64, ffnMult = 4.0, vocab = 2048,
        )

        /**
         * Llama-3-8B-shaped: 32 Q-heads, 8 KV-heads (group_ratio = 4),
         * dModel=4096, dFf=14336, vocab=128256, seq=2048. Defined for
         * Phase 4 measurement to import directly; not exercised by
         * the structural test (would push CI well past ms-scale).
         */
        val llama3_8b = LlamaL3DecoderConfig(
            batch = 1, seq = 2048, dModel = 4096, nHeads = 32, nKvHeads = 8,
            headDim = 128, ffnMult = 3.5, vocab = 128256,
        )
    }
}

/**
 * Builds a Llama-3-shape decoder primal with GQA. Param order:
 *
 * ```
 *  0  x_in       [tokens, D]      — input hidden states
 *  1  labels     [tokens, V]      — one-hot label distribution
 *  2  theta      [tokens, D]      — positional angles for RoPE
 *  3  q_w        [D, D]           — Q projection (full Q dim)
 *  4  k_w        [D, kvDim]       — K projection (smaller, GQA)
 *  5  v_w        [D, kvDim]       — V projection (smaller, GQA)
 *  6  out_w      [D, D]           — attention output projection
 *  7  gate_w     [D, D_ff]        — SwiGLU gate
 *  8  up_w       [D, D_ff]        — SwiGLU up
 *  9  down_w     [D_ff, D]        — SwiGLU down
 * 10  lm_head_w  [D, V]           — LM head
 * 11  eps_attn   [tokens, 1]      — RmsNorm eps (pre-attn)
 * 12  eps_mlp    [tokens, 1]      — RmsNorm eps (pre-MLP)
 * ```
 */
object LlamaL3DecoderPrimal {

    fun build(config: LlamaL3DecoderConfig): DxirFunction =
        DxirBuilder.function("llama_l3_decoder_layer_loss") {
            val tokens = config.tokens
            val d = config.dModel
            val kvDim = config.kvDim
            val groupRatio = config.groupRatio
            val dff = config.dFf
            val v = config.vocab

            val tH = DxirType(F32, listOf(tokens, d))
            val tHReduced = DxirType(F32, listOf(tokens, 1))
            val tScores = DxirType(F32, listOf(tokens, tokens))
            val tFf = DxirType(F32, listOf(tokens, dff))
            val tLogits = DxirType(F32, listOf(tokens, v))
            val tWqOut = DxirType(F32, listOf(d, d))
            val tWkv = DxirType(F32, listOf(d, kvDim))
            val tKv = DxirType(F32, listOf(tokens, kvDim))               // K/V proj output
            val tKvUnsq = DxirType(F32, listOf(tokens, kvDim, 1))         // insert size-1 axis
            val tKvBcast = DxirType(F32, listOf(tokens, kvDim, groupRatio))
            val tWGateUp = DxirType(F32, listOf(d, dff))
            val tWDown = DxirType(F32, listOf(dff, d))
            val tWLmHead = DxirType(F32, listOf(d, v))
            val tEps = DxirType(F32, listOf(tokens, 1))
            val tScalar = DxirType(F32, listOf())
            val tKt = DxirType(F32, listOf(d, tokens))

            // ---- Inputs + weights ------------------------------------
            val xIn = param("x_in", tH)
            val labels = param("labels", tLogits)
            val theta = param("theta", tH)
            val qW = param("q_w", tWqOut)
            val kW = param("k_w", tWkv)
            val vW = param("v_w", tWkv)
            val outW = param("out_w", tWqOut)
            val gateW = param("gate_w", tWGateUp)
            val upW = param("up_w", tWGateUp)
            val downW = param("down_w", tWDown)
            val lmHeadW = param("lm_head_w", tWLmHead)
            val epsAttn = param("eps_attn", tEps)
            val epsMlp = param("eps_mlp", tEps)

            // ---- Pre-attention RmsNorm -------------------------------
            val sq1 = op(OpKind.MUL, listOf(xIn, xIn), tH)
            val mean1 = op(
                OpKind.MEAN, listOf(sq1), tHReduced,
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            val mEps1 = op(OpKind.ADD, listOf(mean1, epsAttn), tHReduced)
            val rsq1 = op(OpKind.RSQRT, listOf(mEps1), tHReduced)
            val xNormAttn = op(OpKind.MUL, listOf(xIn, rsq1), tH)

            // ---- Q at full dim; K, V at kvDim ------------------------
            val q = op(OpKind.MATMUL, listOf(xNormAttn, qW), tH)
            val kProj = op(OpKind.MATMUL, listOf(xNormAttn, kW), tKv)
            val vProj = op(OpKind.MATMUL, listOf(xNormAttn, vW), tKv)

            // ---- GQA expansion: RESHAPE → BROADCAST → RESHAPE --------
            val kUnsq = op(OpKind.RESHAPE, listOf(kProj), tKvUnsq)
            val kBcast = op(
                OpKind.BROADCAST, listOf(kUnsq), tKvBcast,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
            )
            val kExpanded = op(OpKind.RESHAPE, listOf(kBcast), tH)

            val vUnsq = op(OpKind.RESHAPE, listOf(vProj), tKvUnsq)
            val vBcast = op(
                OpKind.BROADCAST, listOf(vUnsq), tKvBcast,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
            )
            val vExpanded = op(OpKind.RESHAPE, listOf(vBcast), tH)

            // ---- RoPE on Q -------------------------------------------
            val cosT = op(OpKind.COS, listOf(theta), tH)
            val sinT = op(OpKind.SIN, listOf(theta), tH)
            val qReal = op(OpKind.MUL, listOf(q, cosT), tH)
            val qImag = op(OpKind.MUL, listOf(theta, sinT), tH)
            val qRot = op(OpKind.SUB, listOf(qReal, qImag), tH)

            // ---- Attention: Q · K^T → SOFTMAX → · V ------------------
            val kT = op(
                OpKind.TRANSPOSE, listOf(kExpanded), tKt,
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            val s = op(OpKind.MATMUL, listOf(qRot, kT), tScores)
            val p = op(OpKind.SOFTMAX, listOf(s), tScores)
            val attnOut = op(OpKind.MATMUL, listOf(p, vExpanded), tH)

            // ---- Output projection + residual ------------------------
            val attnProj = op(OpKind.MATMUL, listOf(attnOut, outW), tH)
            val xAttn = op(OpKind.ADD, listOf(xIn, attnProj), tH)

            // ---- Pre-MLP RmsNorm -------------------------------------
            val sq2 = op(OpKind.MUL, listOf(xAttn, xAttn), tH)
            val mean2 = op(
                OpKind.MEAN, listOf(sq2), tHReduced,
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            val mEps2 = op(OpKind.ADD, listOf(mean2, epsMlp), tHReduced)
            val rsq2 = op(OpKind.RSQRT, listOf(mEps2), tHReduced)
            val xNormMlp = op(OpKind.MUL, listOf(xAttn, rsq2), tH)

            // ---- SwiGLU MLP ------------------------------------------
            val gateProj = op(OpKind.MATMUL, listOf(xNormMlp, gateW), tFf)
            val upProj = op(OpKind.MATMUL, listOf(xNormMlp, upW), tFf)
            val gateAct = op(OpKind.SILU, listOf(gateProj), tFf)
            val swiglu = op(OpKind.MUL, listOf(gateAct, upProj), tFf)

            // ---- Down projection + residual --------------------------
            val mlpOut = op(OpKind.MATMUL, listOf(swiglu, downW), tH)
            val xOut = op(OpKind.ADD, listOf(xAttn, mlpOut), tH)

            // ---- LM head + cross-entropy loss ------------------------
            val logits = op(OpKind.MATMUL, listOf(xOut, lmHeadW), tLogits)
            val probs = op(OpKind.SOFTMAX, listOf(logits), tLogits)
            val logp = op(OpKind.LOG, listOf(probs), tLogits)
            val pw = op(OpKind.MUL, listOf(labels, logp), tLogits)
            val loss = op(OpKind.SUM, listOf(pw), tScalar)

            listOf(loss)
        }
}
