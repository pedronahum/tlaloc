package io.tlaloc.ir

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.KvQuantDtype
import io.tlaloc.ir.recognizer.quant.KvScaleStrategy

/**
 * The DEQUANTIZE_KV operand/attribute convention, shared
 * by every layer that touches the kind (the interpreter, the StableHLO
 * emitter, the renderer's refusal). ONE parser, so the layers cannot disagree
 * about what a legal dequantization looks like (the same pattern as
 * [PagedAttentionAttrs]).
 *
 * ```
 *   0 codes  [numBlocks, blockSize, numKvHeads, headDim]   integer
 *   1 scales [numKvHeads]  or  [1]                         float
 *   → out    [numBlocks, blockSize, numKvHeads, headDim]   float
 * ```
 * `out[b, s, h, d] = codes[b, s, h, d] * scales[h]` — the host codec
 * [io.tlaloc.ir.inference.KvQuantPool.dequantize] is the same formula, and the
 * oracle pins that they agree elementwise.
 *
 * Attributes: `kv_quant_dtype: String` — a [KvQuantDtype.nameTag], REQUIRED,
 * and the only attribute. It is not a shape-derived value (the sentinel-dims
 * rule's exemption for compile-time literals applies: it is a model-config
 * literal, the same class of thing as PAGED_ATTENTION's `scale`), and it is
 * load-bearing rather than decorative: it fixes the legal CODE RANGE, which is
 * the one thing about a quantized pool that the operand shapes cannot tell
 * anyone. A pool of int4 codes and a pool of int8 codes have identical types.
 *
 * THE SCALE STRATEGY IS DERIVED, never an attr: `scales.dims == [numKvHeads]`
 * is [KvScaleStrategy.PER_HEAD] and `scales.dims == [1]` is
 * [KvScaleStrategy.PER_TENSOR]. The two coincide when `numKvHeads == 1`, which
 * is not an ambiguity — the semantics are identical there — and deriving it
 * makes an attr-vs-operand disagreement unrepresentable, exactly as
 * `blockSize`/`numKvHeads` are derived for PAGED_ATTENTION.
 *
 * REJECTED alternative: folding the dequantization INTO PAGED_ATTENTION as two
 * extra optional operands (`kScales`, `vScales`) plus a dtype attr. It reads
 * attractive — one op, one fused kernel — and it is wrong for this IR: it
 * makes PAGED_ATTENTION's arity a mode flag (5 operands or 7), it duplicates
 * the same dequantization inside an op that already has the most
 * intricate emission of the inference ops, and it hides the quantization from every OTHER consumer
 * of a pool (KV_CACHE_WRITE's read-modify-write, a debug print, a CPU
 * fallback). As a separate op the dequantization is one node that any pass can
 * see, that CSE can share between the K and V paths of one layer, and that a
 * future fused kernel can CLAIM together with the attention it feeds — which
 * is precisely how the inference claiming registry works: it rewrites a
 * recognized op into a custom call, and a recognizer over the
 * DEQUANTIZE_KV → PAGED_ATTENTION pair fits that shape.
 */
object DequantizeKvAttrs {

    /** The attr key carrying the [KvQuantDtype.nameTag]. */
    const val DTYPE_ATTR: String = "kv_quant_dtype"

    /** The parsed, validated story. Every field is derived or checked. */
    data class Parsed(
        val numKvHeads: Int,
        val headDim: Int,
        val elementCount: Int,
        val config: KvQuantConfig,
    ) {
        /** The largest legal `|code|` — [KvQuantDtype.codeMax] of the parsed dtype. */
        val codeMax: Int get() = config.dtype.codeMax
    }

    /**
     * Parse and validate [op], refusing loudly with [layer] in the message.
     */
    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.DEQUANTIZE_KV) {
            "$layer: DequantizeKvAttrs.parse called on ${op.op} (a compiler bug)"
        }
        require(op.operands.size == 2) {
            "$layer: DEQUANTIZE_KV requires 2 operands (codes, scales), got ${op.operands.size}"
        }
        val codes = op.operands[0].type
        val scales = op.operands[1].type

        require(codes.rank == 4) {
            "$layer: DEQUANTIZE_KV codes must be a rank-4 KV pool " +
                "[numBlocks, blockSize, numKvHeads, headDim], got ${codes.dims}"
        }
        require(isIntegral(codes.dtype)) {
            "$layer: DEQUANTIZE_KV codes must be an integer tensor (they are small integer " +
                "codes, not values), got ${codes.dtype}"
        }
        require(scales.rank == 1) {
            "$layer: DEQUANTIZE_KV scales must be rank-1 — [numKvHeads] for per-head scaling " +
                "or [1] for per-tensor, got ${scales.dims}"
        }
        require(isFloating(scales.dtype)) {
            "$layer: DEQUANTIZE_KV scales must be a float tensor, got ${scales.dtype}"
        }
        require(op.type.dims == codes.dims) {
            "$layer: DEQUANTIZE_KV result shape ${op.type.dims} must match the codes shape " +
                "${codes.dims} (dequantization is elementwise)"
        }
        require(isFloating(op.type.dtype)) {
            "$layer: DEQUANTIZE_KV result must be a float tensor (it is the pool the attention " +
                "reads), got ${op.type.dtype}"
        }

        val numKvHeads = codes.dims[2]
        val headDim = codes.dims[3]
        require(numKvHeads >= 1 && headDim >= 1) {
            "$layer: DEQUANTIZE_KV needs positive numKvHeads/headDim, got $numKvHeads/$headDim"
        }
        val strategy = when (scales.dims[0]) {
            numKvHeads -> KvScaleStrategy.PER_HEAD
            1 -> KvScaleStrategy.PER_TENSOR
            else -> error(
                "$layer: DEQUANTIZE_KV scales has ${scales.dims[0]} entries, which is neither " +
                    "the pool's numKvHeads ($numKvHeads, per-head scaling) nor 1 (per-tensor) — " +
                    "the scale strategy is DERIVED from this extent, never an attr",
            )
        }

        val tag = op.attrs[DTYPE_ATTR]
        require(tag is String) {
            "$layer: DEQUANTIZE_KV requires a '$DTYPE_ATTR' attr naming the quantized format " +
                "(the operand shapes cannot tell an int4 pool from an int8 one); got $tag"
        }
        val dtype = KvQuantDtype.entries.firstOrNull { it.nameTag == tag }
            ?: error(
                "$layer: DEQUANTIZE_KV '$DTYPE_ATTR' = '$tag' is not a known KV-quant dtype " +
                    "(${KvQuantDtype.entries.joinToString(", ") { it.nameTag }})",
            )
        require(dtype.isIntegerCoded) {
            "$layer: DEQUANTIZE_KV carries '$tag', which is NOT an integer-coded format and is " +
                "refused BY NAME, not missing — an fp8 code is a bit pattern with its own " +
                "exponent, so `value = code * scale` is not its dequantization. fp8 KV-quant " +
                "needs a narrow DType of its own, as bf16 has; the integer-coded " +
                "formats this op implements are " +
                KvQuantDtype.entries.filter { it.isIntegerCoded }.joinToString(", ") { it.nameTag }
        }

        return Parsed(
            numKvHeads = numKvHeads,
            headDim = headDim,
            elementCount = codes.elementCount.toInt(),
            config = KvQuantConfig(dtype, strategy),
        )
    }

    private fun isIntegral(d: DType): Boolean = d == I32 || d == I64

    private fun isFloating(d: DType): Boolean = d == F32 || d == BF16
}
