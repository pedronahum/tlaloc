package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.271 — Llama-style decoder layer + LM-head loss as a DXIR primal.
 *
 * Phase 2 step 1 of the dual-track Llama-decoder benchmark plan
 * (`memory/llama_benchmark_dual_track.md`). The DXIR function this object
 * builds is the workload that exercises every L4 recognizer + coarsener
 * landed in Phase 1:
 *
 * - **2× RmsNorm** (pre-attention + pre-MLP — the third "final norm" is
 *   absorbed into the LM-head path)
 * - **1× FlashAttention** (`MATMUL → SOFTMAX → MATMUL`)
 * - **1× RoPE** (sin/cos rotation on Q's projection)
 * - **1× SwiGLU** (gate-MATMUL + up-MATMUL + SILU + gating MUL)
 * - **1× CrossEntropy** (LM-head SOFTMAX → LOG → MUL(labels, ·) → SUM)
 *
 * Plus the unrecognized-but-present residue: 4× regular MATMULs (Q/K/V/O
 * projections aren't part of any recognized pattern; flash-attention's
 * inner matmuls *are*), 1× MATMUL for the down projection, 1× MATMUL for
 * the LM head, and 2× residual ADDs.
 *
 * # Scope notes
 *
 * - **Rank-2 throughout.** Batch + sequence axes are flattened into a
 *   single `tokens = batch * seq` axis. This sidesteps batched-matmul
 *   semantics in the gradient-body transposes and matches the
 *   recognizer-test conventions (which all use rank-2). Real Llama
 *   keeps the batch axis, which the StableHLO matmul-VJP convention
 *   handles via leading-axis broadcast — but that's a Phase-3 backend
 *   concern, not a primal-builder concern.
 *
 * - **No head splitting.** Self-attention runs with shape
 *   `[tokens, tokens]` for the score matrix — i.e. one "head" over the
 *   full sequence. Real multi-head attention reshapes to
 *   `[tokens, n_heads, head_dim]`, which is a downstream realism step
 *   (Phase 4 measurement may want it). For Phase 2's "the patterns
 *   match" goal, single-head suffices.
 *
 * - **No grouped-query attention (GQA).** Llama-3 ships GQA (8 KV heads
 *   for 32 Q heads). Adding GQA would require K/V repeat ops that the
 *   FlashAttention recognizer doesn't yet match — separate phase if
 *   ever needed.
 *
 * - **Unweighted RmsNorm.** The recognized form is
 *   `x · rsqrt(mean(x²) [+ eps])`. Real Llama wraps this with a
 *   per-feature scale γ (`y = γ · normalised`); for v1 we omit the γ
 *   multiply since it lives outside the recognized region (no impact on
 *   coarsening; can be added as a trailing MUL when the benchmark wants
 *   numerical-fidelity comparisons).
 *
 * - **One RoPE only (on Q).** Real Llama applies RoPE to both Q and K.
 *   v1 applies it only to Q so the recognizer fires once with predictable
 *   structure. A future phase can extend.
 *
 * - **Eps-stabilised RmsNorm.** Both RmsNorms use the
 *   `ADD(MEAN, eps_const)` variant — matches what real Llama emits and
 *   exercises the eps-handling branch of the RmsNorm coarsener.
 *
 * # Two configs
 *
 * - [LlamaDecoderConfig.tiny] — small enough for CI smoke (~ms-scale on
 *   any backend). Used by the structural test.
 * - [LlamaDecoderConfig.llama3_8b] — Llama-3-8B-shaped (up to GQA). Not
 *   exercised by §0.4.271's tests; here for Phase 4's measurement
 *   pipeline to import directly.
 */

/**
 * Hyperparameters for [LlamaDecoderPrimal]. All shapes derive from these.
 *
 * @property batch batch size (B)
 * @property seq sequence length (S); the flattened-tokens axis is `B*S`
 * @property dModel hidden width (D); must equal `nHeads * headDim`
 * @property nHeads number of attention heads
 * @property headDim per-head width (D / nHeads)
 * @property ffnMult MLP expansion ratio (`dFf = ceil(dModel * ffnMult)`)
 * @property vocab vocabulary size (V) for the LM head
 */
data class LlamaDecoderConfig(
    val batch: Int,
    val seq: Int,
    val dModel: Int,
    val nHeads: Int,
    val headDim: Int,
    val ffnMult: Double,
    val vocab: Int,
) {
    val dFf: Int get() = (dModel * ffnMult).toInt()
    val tokens: Int get() = batch * seq

    init {
        require(dModel == nHeads * headDim) {
            "dModel ($dModel) must equal nHeads * headDim ($nHeads * $headDim = ${nHeads * headDim})"
        }
        require(batch > 0 && seq > 0 && vocab > 0 && dModel > 0 && nHeads > 0)
    }

    companion object {
        /** CI-friendly tiny config; ~ms-scale build + recognize on any backend. */
        val tiny = LlamaDecoderConfig(
            batch = 2, seq = 16, dModel = 64, nHeads = 4, headDim = 16,
            ffnMult = 4.0, vocab = 256,
        )

        /**
         * Llama-3-8B-shaped (modulo single-head + no-GQA simplifications).
         * Defined here for Phase 4 measurement to import directly; not
         * exercised by Phase 2 tests (would push CI beyond ms-scale).
         */
        val llama3_8b = LlamaDecoderConfig(
            batch = 1, seq = 2048, dModel = 4096, nHeads = 32, headDim = 128,
            ffnMult = 3.5, vocab = 128256,
        )
    }
}

/**
 * Builds the [DxirFunction] that one decoder layer + LM-head + loss
 * compiles to. The function's parameters (in order):
 *
 * ```
 *  0  x_in       [tokens, D]      — input hidden states
 *  1  labels     [tokens, V]      — one-hot label distribution
 *  2  theta      [tokens, D]      — positional angles for RoPE
 *  3  q_w        [D, D]           — Q projection weight
 *  4  k_w        [D, D]           — K projection weight
 *  5  v_w        [D, D]           — V projection weight
 *  6  out_w      [D, D]           — attention output projection weight
 *  7  gate_w     [D, D_ff]        — SwiGLU gate projection weight
 *  8  up_w       [D, D_ff]        — SwiGLU up projection weight
 *  9  down_w     [D_ff, D]        — SwiGLU down projection weight
 * 10  lm_head_w  [D, V]           — LM head projection weight
 * 11  eps_attn   []               — RmsNorm eps (pre-attn)
 * 12  eps_mlp    []               — RmsNorm eps (pre-MLP)
 * ```
 *
 * (No "final norm" — this v1 layer skips the gpt-style final RmsNorm and
 * goes from `x_attn + mlp_out` straight into the LM head. A second
 * RmsNorm before the LM head can be added later if needed; the
 * recognizers will pick it up automatically.)
 *
 * The function returns a single scalar — the cross-entropy loss over the
 * batch.
 */
object LlamaDecoderPrimal {

    fun build(config: LlamaDecoderConfig): DxirFunction =
        DxirBuilder.function("llama_decoder_layer_loss") {
            val tokens = config.tokens
            val d = config.dModel
            val dff = config.dFf
            val v = config.vocab

            val tH = DxirType(F32, listOf(tokens, d))           // [tokens, D]
            val tHReduced = DxirType(F32, listOf(tokens, 1))    // RmsNorm keepdims
            val tScores = DxirType(F32, listOf(tokens, tokens)) // attention scores
            val tFf = DxirType(F32, listOf(tokens, dff))        // [tokens, D_ff]
            val tLogits = DxirType(F32, listOf(tokens, v))      // LM head output
            val tWqkv = DxirType(F32, listOf(d, d))
            val tWGateUp = DxirType(F32, listOf(d, dff))
            val tWDown = DxirType(F32, listOf(dff, d))
            val tWLmHead = DxirType(F32, listOf(d, v))
            val tEps = DxirType(F32, listOf(tokens, 1))         // broadcast-compatible with mean
            val tScalar = DxirType(F32, listOf())

            // ---- Inputs + weights -------------------------------------------------
            val xIn = param("x_in", tH)
            val labels = param("labels", tLogits)
            val theta = param("theta", tH)
            val qW = param("q_w", tWqkv)
            val kW = param("k_w", tWqkv)
            val vW = param("v_w", tWqkv)
            val outW = param("out_w", tWqkv)
            val gateW = param("gate_w", tWGateUp)
            val upW = param("up_w", tWGateUp)
            val downW = param("down_w", tWDown)
            val lmHeadW = param("lm_head_w", tWLmHead)
            val epsAttn = param("eps_attn", tEps)
            val epsMlp = param("eps_mlp", tEps)

            // ---- Pre-attention RmsNorm (eps form) ---------------------------------
            val sq1 = op(OpKind.MUL, listOf(xIn, xIn), tH)
            val mean1 = op(OpKind.MEAN, listOf(sq1), tHReduced)
            val mEps1 = op(OpKind.ADD, listOf(mean1, epsAttn), tHReduced)
            val rsq1 = op(OpKind.RSQRT, listOf(mEps1), tHReduced)
            val xNormAttn = op(OpKind.MUL, listOf(xIn, rsq1), tH)

            // ---- Q, K, V projections ----------------------------------------------
            val q = op(OpKind.MATMUL, listOf(xNormAttn, qW), tH)
            val k = op(OpKind.MATMUL, listOf(xNormAttn, kW), tH)
            val vProj = op(OpKind.MATMUL, listOf(xNormAttn, vW), tH)

            // ---- RoPE on Q (single rotation; treats Q as the "real" half and
            //      a tied-shape projection of theta as the "imag" half — keeps
            //      the recognizer happy without introducing splitting ops).
            val cosT = op(OpKind.COS, listOf(theta), tH)
            val sinT = op(OpKind.SIN, listOf(theta), tH)
            val qReal = op(OpKind.MUL, listOf(q, cosT), tH)            // x_real · cos
            val qImag = op(OpKind.MUL, listOf(theta, sinT), tH)        // x_imag · sin (theta as proxy)
            val qRot = op(OpKind.SUB, listOf(qReal, qImag), tH)        // RoPE recombination

            // ---- Attention: MATMUL(Q, K) → SOFTMAX → MATMUL(P, V) ------------------
            val s = op(OpKind.MATMUL, listOf(qRot, k), tScores)
            val p = op(OpKind.SOFTMAX, listOf(s), tScores)
            val attnOut = op(OpKind.MATMUL, listOf(p, vProj), tH)

            // ---- Output projection + residual -------------------------------------
            val attnProj = op(OpKind.MATMUL, listOf(attnOut, outW), tH)
            val xAttn = op(OpKind.ADD, listOf(xIn, attnProj), tH)

            // ---- Pre-MLP RmsNorm (eps form) ---------------------------------------
            val sq2 = op(OpKind.MUL, listOf(xAttn, xAttn), tH)
            val mean2 = op(OpKind.MEAN, listOf(sq2), tHReduced)
            val mEps2 = op(OpKind.ADD, listOf(mean2, epsMlp), tHReduced)
            val rsq2 = op(OpKind.RSQRT, listOf(mEps2), tHReduced)
            val xNormMlp = op(OpKind.MUL, listOf(xAttn, rsq2), tH)

            // ---- SwiGLU MLP: SILU(MATMUL(x, gate_w)) · MATMUL(x, up_w) ------------
            val gateProj = op(OpKind.MATMUL, listOf(xNormMlp, gateW), tFf)
            val upProj = op(OpKind.MATMUL, listOf(xNormMlp, upW), tFf)
            val gateAct = op(OpKind.SILU, listOf(gateProj), tFf)
            val swiglu = op(OpKind.MUL, listOf(gateAct, upProj), tFf)

            // ---- Down projection + residual ---------------------------------------
            val mlpOut = op(OpKind.MATMUL, listOf(swiglu, downW), tH)
            val xOut = op(OpKind.ADD, listOf(xAttn, mlpOut), tH)

            // ---- LM head + cross-entropy loss -------------------------------------
            val logits = op(OpKind.MATMUL, listOf(xOut, lmHeadW), tLogits)
            val probs = op(OpKind.SOFTMAX, listOf(logits), tLogits)
            val logp = op(OpKind.LOG, listOf(probs), tLogits)
            val pw = op(OpKind.MUL, listOf(labels, logp), tLogits)
            val loss = op(OpKind.SUM, listOf(pw), tScalar)

            listOf(loss)
        }
}
