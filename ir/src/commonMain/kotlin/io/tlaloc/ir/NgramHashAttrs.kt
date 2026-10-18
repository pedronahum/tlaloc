package io.tlaloc.ir

import io.tlaloc.core.I32
import io.tlaloc.core.I64

/**
 * [OpKind.NGRAM_HASH]'s operand and attributes, one parser for every layer
 * that touches the kind, and the arithmetic both interpreters run.
 * ```
 *   0 tokens [N, M]   integer: each row's token, then the M - 1 before it
 *   -> ids   [N, H]   I32
 * attrs: multipliers: List<Long> (M), orders: List<Int> (H, each in 2..M),
 *        moduli: List<Long> (H, each >= 1), offsets: List<Long> (H, each >= 0)
 * ```
 * Head `h` hashes the row's last `orders[h]` tokens:
 * `mixed = XOR over i < orders[h] of tokens[i] * multipliers[i]`, in
 * two's-complement 64-bit arithmetic (a product wraps), and
 * `ids[h] = floorMod(mixed, moduli[h]) + offsets[h]`. That is transformers'
 * `Qwen4ExpTextNGramEmbedding` for one row of token history, whose `torch.long`
 * products wrap the same way and whose `torch.remainder` takes the divisor's
 * sign. Every id must fit an I32.
 */
object NgramHashAttrs {

    data class Parsed(
        val rows: Int,
        /** Tokens per row: the row's own and the ones before it. */
        val width: Int,
        val multipliers: List<Long>,
        val orders: List<Int>,
        val moduli: List<Long>,
        val offsets: List<Long>,
    ) {
        val heads: Int get() = orders.size
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.NGRAM_HASH) { "$layer: NgramHashAttrs.parse called on ${op.op}" }
        require(op.operands.size == 1) { "$layer: NGRAM_HASH takes 1 operand (tokens), got ${op.operands.size}" }
        val t = op.operands[0].type
        require(t.rank == 2 && (t.dtype == I32 || t.dtype == I64)) {
            "$layer: NGRAM_HASH tokens must be an integer [N, M], got ${t.dtype}${t.dims}"
        }
        fun longs(key: String): List<Long> = (op.attrs[key] as? List<*>)?.map {
            (it as? Number)?.toLong() ?: error("$layer: NGRAM_HASH '$key' must hold integers, got $it")
        } ?: error("$layer: NGRAM_HASH needs a '$key' list, got ${op.attrs[key]}")
        val (n, m) = t.dims
        val multipliers = longs("multipliers")
        val orders = longs("orders").map { it.toInt() }
        val moduli = longs("moduli")
        val offsets = longs("offsets")
        require(multipliers.size == m) { "$layer: NGRAM_HASH has ${multipliers.size} multipliers for $m tokens per row" }
        require(orders.isNotEmpty() && moduli.size == orders.size && offsets.size == orders.size) {
            "$layer: NGRAM_HASH needs one order, modulus and offset per head, got ${orders.size}/${moduli.size}/${offsets.size}"
        }
        require(orders.all { it in 2..m }) { "$layer: NGRAM_HASH orders must be in 2..$m, got $orders" }
        require(moduli.all { it >= 1 } && offsets.all { it >= 0 }) {
            "$layer: NGRAM_HASH moduli must be >= 1 and offsets >= 0, got $moduli and $offsets"
        }
        require(moduli.indices.all { offsets[it] + moduli[it] - 1 <= Int.MAX_VALUE }) {
            "$layer: NGRAM_HASH ids must fit an I32: moduli $moduli, offsets $offsets"
        }
        require(op.type.dtype == I32 && op.type.dims == listOf(n, orders.size)) {
            "$layer: NGRAM_HASH result must be I32 [$n, ${orders.size}], got ${op.type.dtype}${op.type.dims}"
        }
        return Parsed(n, m, multipliers, orders, moduli, offsets)
    }

    /** The ids of [tokens] (`[N, M]` row-major), `[N, H]` row-major. */
    fun reference(p: Parsed, tokens: IntArray): IntArray {
        require(tokens.size == p.rows * p.width) { "NgramHashAttrs.reference: ${tokens.size} tokens for [${p.rows}, ${p.width}]" }
        val out = IntArray(p.rows * p.heads)
        for (r in 0 until p.rows) {
            for (h in 0 until p.heads) {
                var mixed = 0L
                for (i in 0 until p.orders[h]) mixed = mixed xor (tokens[r * p.width + i].toLong() * p.multipliers[i])
                out[r * p.heads + h] = (mixed.mod(p.moduli[h]) + p.offsets[h]).toInt()
            }
        }
        return out
    }
}
