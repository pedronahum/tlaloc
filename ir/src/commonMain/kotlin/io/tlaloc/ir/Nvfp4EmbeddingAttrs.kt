package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.core.U8

/**
 * [OpKind.NVFP4_EMBEDDING]'s operands, one parser for every layer that touches
 * the kind, and the lookup both interpreters run.
 * ```
 *   0 codes   [V, D / 2]   u8: two e2m1 codes per byte, the even column in the low nibble
 *   1 scales  [V, D / G]   u8: the e4m3fn scale of each group of G consecutive values of a row
 *   2 scale2  [1]          f32: the table's scale
 *   3 ids     [...]        integer, each in [0, V)
 *   -> rows   [..., D]     f32
 * ```
 * `rows[i, c] = e2m1(code[id_i, c]) * e4m3(scale[id_i, c / G]) * scale2`, with
 * `G = D / scales.dims[1]`: an NVFP4 table (G = 16, as NVIDIA's format groups
 * values) gathered by row and widened. Each value is exact in f32. The row
 * layout is the checkpoint's own (row-major codes and scales), not
 * [Nvfp4MatmulAttrs]'s fragment order: a lookup reads whole rows.
 */
object Nvfp4EmbeddingAttrs {

    data class Parsed(
        /** Rows of the table. */
        val rows: Int,
        /** Values per row. */
        val dim: Int,
        /** Values per group scale. */
        val group: Int,
        /** Ids looked up. */
        val lookups: Int,
    ) {
        val codeBytes: Int get() = dim / 2
        val groups: Int get() = dim / group
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.NVFP4_EMBEDDING) { "$layer: Nvfp4EmbeddingAttrs.parse called on ${op.op}" }
        require(op.operands.size == 4) {
            "$layer: NVFP4_EMBEDDING takes 4 operands (codes, scales, scale2, ids), got ${op.operands.size}"
        }
        val (codes, scales, s2, ids) = op.operands.map { it.type }
        require(codes.dtype == U8 && codes.rank == 2) { "$layer: NVFP4_EMBEDDING codes must be u8 [V, D / 2], got ${codes.dtype}${codes.dims}" }
        val (v, half) = codes.dims
        val d = 2 * half
        require(scales.dtype == U8 && scales.rank == 2 && scales.dims[0] == v && scales.dims[1] >= 1 && d % scales.dims[1] == 0) {
            "$layer: NVFP4_EMBEDDING scales must be u8 [$v, D / G] with G dividing D = $d, got ${scales.dtype}${scales.dims}"
        }
        val g = d / scales.dims[1]
        require(g % 2 == 0) { "$layer: NVFP4_EMBEDDING groups of $g values split a code byte" }
        require(s2.dtype == F32 && s2.dims == listOf(1)) { "$layer: NVFP4_EMBEDDING scale2 must be f32 [1], got ${s2.dtype}${s2.dims}" }
        require(ids.dtype == I32 || ids.dtype == I64) { "$layer: NVFP4_EMBEDDING ids must be integer, got ${ids.dtype}" }
        require(op.type.dtype == F32 && op.type.dims == ids.dims + d) {
            "$layer: NVFP4_EMBEDDING result must be f32 ${ids.dims + d}, got ${op.type.dtype}${op.type.dims}"
        }
        return Parsed(v, d, g, ids.dims.fold(1) { a, b -> a * b })
    }

    /** The rows of [ids], codes and scales given as byte values. */
    fun reference(p: Parsed, codes: FloatArray, scales: FloatArray, scale2: Float, ids: IntArray): FloatArray {
        val out = FloatArray(p.lookups * p.dim)
        for (i in 0 until p.lookups) {
            val id = ids[i]
            require(id in 0 until p.rows) { "NVFP4_EMBEDDING: id $id is outside [0, ${p.rows})" }
            for (c in 0 until p.dim) {
                val byte = codes[id * p.codeBytes + c / 2].toInt() and 0xFF
                val code = if (c % 2 == 0) byte and 0xF else byte ushr 4
                val s = scales[id * p.groups + c / p.group].toInt() and 0xFF
                out[i * p.dim + c] = io.tlaloc.core.f4e2m1ToFloat(code) * io.tlaloc.core.f8e4m3fnToFloat(s.toByte()) * scale2
            }
        }
        return out
    }
}
