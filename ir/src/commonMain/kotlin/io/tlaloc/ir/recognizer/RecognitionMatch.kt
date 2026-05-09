package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType

/**
 * Layer 3 §0.4.250+ — typed match record produced by a [DxirFunction]
 * recognizer.
 *
 * Each recognized compound idiom is a sealed subtype carrying:
 *
 * - [ops] — the matched [DxirOp]s, in graph order. Used by downstream
 *   passes (VJP coarsener, kernel-template emitter, tile-fusion pass) to
 *   identify the region to substitute.
 * - Per-pattern structural metadata (named-axis structure, dtype,
 *   dimensions). Each subtype names the fields downstream needs.
 *
 * Recognizers are pure functions returning `List<Subtype>`. They never
 * mutate the IR; they only describe what's there.
 */
sealed class RecognitionMatch {
    /** All matched ops, in the order the recognizer encountered them. */
    abstract val ops: List<DxirOp>

    /** Stable tag for diagnostics + match-resolver ordering. */
    abstract val patternName: String

    /**
     * `MATMUL → SOFTMAX → MATMUL` compound form, the canonical
     * "FlashAttention" candidate. Layer 3 v1 matches the unfused shape
     * (the form a user-written attention forward produces); a future
     * recognizer can also match `OpKind.SCALED_DOT_PRODUCT_ATTENTION`
     * directly when users opt into the pre-fused op.
     *
     * @property qkMatmul the `Q · K^T` step (produces raw scores).
     * @property softmax the `SOFTMAX` over the score's last axis.
     * @property pvMatmul the `weights · V` step (produces the output).
     * @property qInput the Q tensor (Q operand of [qkMatmul]).
     * @property kInput the K tensor.
     * @property vInput the V tensor.
     * @property scoreType the type of [qkMatmul]'s result (raw scores).
     * @property outputType the type of [pvMatmul]'s result (attention out).
     */
    data class FlashAttention(
        val qkMatmul: DxirOp,
        val softmax: DxirOp,
        val pvMatmul: DxirOp,
        val qInput: io.tlaloc.ir.DxirNode,
        val kInput: io.tlaloc.ir.DxirNode,
        val vInput: io.tlaloc.ir.DxirNode,
        val scoreType: DxirType,
        val outputType: DxirType,
    ) : RecognitionMatch() {
        override val ops: List<DxirOp> = listOf(qkMatmul, softmax, pvMatmul)
        override val patternName: String = "FlashAttention"
    }

    /**
     * `square → mean(reduce) → rsqrt → multiply` form (RMS norm).
     * v1 stub — full structural match lands in L3.1.
     */
    data class RmsNorm(
        override val ops: List<DxirOp>,
        val input: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
    ) : RecognitionMatch() {
        override val patternName: String = "RmsNorm"
    }

    /**
     * §0.4.318 — Layer norm without affine. Canonical decomposition:
     *
     * ```
     * mean1 = MEAN(x)                  // keepdims, last axis
     * sub   = SUB(x, mean1)            // centered (broadcast on size-1 axis)
     * sq    = MUL(sub, sub)            // squared deviations
     * mean2 = MEAN(sq)                 // variance, keepdims
     * [eps  = ADD(mean2, eps_const)]   // optional stabiliser
     * std   = SQRT(mean2 or eps)
     * out   = DIV(sub, std)
     * ```
     *
     * v1 anchors on `OpKind.SQRT` consumed by `DIV(centered, std)`. This
     * doesn't overlap with [RmsNorm] (which anchors on RSQRT), so the
     * §0.4.282 [resolveLargestMatch] doesn't kick in for v1. If a future
     * RSQRT+MUL alternative form is added, RmsNorm's `MUL(x, x)` check
     * would pass on a LayerNorm region (with `x = sub`); the resolver
     * would then pick LayerNorm by op count (always larger by exactly
     * the {mean1, sub} pair).
     *
     * v1 also doesn't recognise the optional affine `* gamma + beta`
     * post-scale — same scope reasoning as [RmsNorm].
     */
    data class LayerNorm(
        override val ops: List<DxirOp>,
        val input: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
    ) : RecognitionMatch() {
        override val patternName: String = "LayerNorm"
    }

    /**
     * Rotary positional embedding (RoPE) — sin/cos rotation pair. v1
     * stub — full structural match lands in L3.1.
     */
    data class Rope(
        override val ops: List<DxirOp>,
        val input: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
    ) : RecognitionMatch() {
        override val patternName: String = "Rope"
    }

    /**
     * Cross-entropy loss compound form. v1 stub — full structural match
     * lands in L3.1.
     */
    data class CrossEntropy(
        override val ops: List<DxirOp>,
        val logits: io.tlaloc.ir.DxirNode,
        val labels: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
    ) : RecognitionMatch() {
        override val patternName: String = "CrossEntropy"
    }

    /**
     * SwiGLU gated MLP activation. Layer 4 §0.4.267 — the canonical
     * Llama / Mistral / PaLM MLP gate:
     *
     * ```
     * gate_proj = MATMUL(x, W_gate)
     * up_proj   = MATMUL(x, W_up)
     * out       = SILU(gate_proj) · up_proj
     * ```
     *
     * v1 matches the inner SwiGLU activation (two parallel matmuls + SILU
     * + elementwise gating), not the surrounding `down_proj` matmul. The
     * coarsener target is a fused `silu_mul_kernel` (one kernel launch +
     * no `silu(gate)` materialisation in HBM). A future "TransformerMLP"
     * recognizer can absorb the down-proj for cuBLASLt-style fused-MLP
     * kernels.
     *
     * @property xInput the shared input tensor (operand of both MATMULs).
     * @property wGate the gate-projection weight (the non-shared operand
     *   of [ops]\[1]).
     * @property wUp the up-projection weight (the non-shared operand of
     *   [ops]\[2]).
     * @property output the gating MUL's result (== [ops]\[3]).
     * @property xType convenience: type of [xInput].
     * @property outputType convenience: type of [output].
     */
    data class SwiGLU(
        override val ops: List<DxirOp>,
        val xInput: io.tlaloc.ir.DxirNode,
        val wGate: io.tlaloc.ir.DxirNode,
        val wUp: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
        val xType: DxirType,
        val outputType: DxirType,
    ) : RecognitionMatch() {
        override val patternName: String = "SwiGLU"
    }

    /**
     * Layer 4 §0.4.314 — the SwiGLU + down-projection fused MLP block,
     * the canonical Llama / Mistral / PaLM transformer MLP:
     *
     * ```
     * gate_proj = MATMUL(x, W_gate)
     * up_proj   = MATMUL(x, W_up)
     * silu_g    = SILU(gate_proj) · up_proj
     * out       = MATMUL(silu_g, W_down)
     * ```
     *
     * Strict superset of [SwiGLU] — same 4 ops plus the down-projection
     * MATMUL. The two patterns claim overlapping op ids (the SwiGLU four),
     * so [resolveLargestMatch] picks this one whenever it matches: the
     * v2 compound use-case the §0.4.282 resolver was plumbed for.
     *
     * Exists for cuBLASLt-style fused-MLP kernels: the down-proj's
     * `silu_g` materialisation is the heaviest tensor traffic in the
     * SwiGLU path, and a fused kernel can avoid it by streaming silu_g
     * straight into the down-proj's matmul accumulator.
     *
     * @property xInput the shared input tensor (operand of both gate
     *   and up MATMULs).
     * @property wGate the gate-projection weight.
     * @property wUp the up-projection weight.
     * @property wDown the down-projection weight (the non-silu_g operand
     *   of the trailing MATMUL).
     * @property output the down-projection MATMUL's result.
     * @property xType convenience: type of [xInput].
     * @property outputType convenience: type of [output].
     */
    data class TransformerMLP(
        override val ops: List<DxirOp>,
        val xInput: io.tlaloc.ir.DxirNode,
        val wGate: io.tlaloc.ir.DxirNode,
        val wUp: io.tlaloc.ir.DxirNode,
        val wDown: io.tlaloc.ir.DxirNode,
        val output: io.tlaloc.ir.DxirNode,
        val xType: DxirType,
        val outputType: DxirType,
    ) : RecognitionMatch() {
        override val patternName: String = "TransformerMLP"
    }
}

/**
 * Structured diagnostic for a near-miss: the recognizer's pre-filter
 * triggered (some op of the right kind appeared) but the full pattern
 * didn't match. Used for Tlaloc-equivalent of XATLib's
 * `emitResidualDiagnostics`.
 *
 * Recognizer functions take an optional `MutableList<RecognitionDiagnostic>?`
 * parameter; when non-null and a near-miss is detected, append a
 * structured record. Tooling (IDE plugin, debug printer) can surface
 * these to users to explain why a recognized-shape didn't match.
 */
data class RecognitionDiagnostic(
    /** Pattern name attempted (`"FlashAttention"`, `"RmsNorm"`, etc.). */
    val pattern: String,
    /** Short reason the match failed (`"reduce kind was SUM, expected SOFTMAX"`). */
    val reason: String,
    /** SSA id of the op that triggered the pre-filter. */
    val opId: Int,
)
