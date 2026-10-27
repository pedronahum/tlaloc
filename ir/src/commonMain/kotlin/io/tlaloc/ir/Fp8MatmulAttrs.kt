package io.tlaloc.ir

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN

/**
 * [OpKind.FP8_MATMUL], one parser for every layer:
 * ```
 *   FP8_MATMUL(a [M, K] f8e4m3fn, b [K, N] f8e4m3fn, aScale [M] or [] f32, bScale [N] or [] f32)
 *     -> y [M, N] f32 or bf16,   y = (a · b) ⊙ aScale[:, None] ⊙ bScale[None, :]
 * ```
 * The product of the codes is summed in f32, then scaled per row of a and per column of b
 * (rowwise) or by one number each (tensorwise, an empty shape). Emitted as a `dot_general` on
 * the f8 operands with an f32 result, which XLA runs on FP8 units where the target has them
 * (TPU v7, Hopper and Blackwell GPUs) and widens elsewhere.
 */
object Fp8MatmulAttrs {

    data class Parsed(val m: Int, val k: Int, val n: Int, val rowwiseA: Boolean, val rowwiseB: Boolean)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.FP8_MATMUL) { "$layer: Fp8MatmulAttrs.parse called on ${op.op} (a compiler bug)" }
        require(op.attrs.isEmpty()) { "$layer: FP8_MATMUL takes no attributes, got ${op.attrs.keys}" }
        require(op.operands.size == 4 && op.types.size == 1) { "$layer: FP8_MATMUL takes a, b, aScale, bScale and returns one value" }
        val (a, b, sa, sb) = op.operands.map { it.type }
        require(a.dtype == F8E4M3FN && b.dtype == F8E4M3FN && a.rank == 2 && b.rank == 2 && a.dims[1] == b.dims[0]) {
            "$layer: FP8_MATMUL operands must be f8e4m3fn [M, K] and [K, N], got $a and $b"
        }
        val (m, k) = a.dims
        val n = b.dims[1]
        require(sa.dtype == F32 && (sa.dims.isEmpty() || sa.dims == listOf(m))) { "$layer: FP8_MATMUL aScale must be f32 [$m] or [], got $sa" }
        require(sb.dtype == F32 && (sb.dims.isEmpty() || sb.dims == listOf(n))) { "$layer: FP8_MATMUL bScale must be f32 [$n] or [], got $sb" }
        require(op.type.dims == listOf(m, n) && (op.type.dtype == F32 || op.type.dtype == BF16)) {
            "$layer: FP8_MATMUL returns f32 or bf16 [$m, $n], got ${op.type}"
        }
        return Parsed(m, k, n, sa.dims.isNotEmpty(), sb.dims.isNotEmpty())
    }

    /** The value, from the decoded codes [a] and [b] (row-major) and the scales, summed in double. */
    fun evaluate(op: DxirOp, a: DoubleArray, b: DoubleArray, aScale: DoubleArray, bScale: DoubleArray, layer: String): DoubleArray {
        val p = parse(op, layer)
        val out = DoubleArray(p.m * p.n)
        for (i in 0 until p.m) for (j in 0 until p.n) {
            var acc = 0.0
            for (l in 0 until p.k) acc += a[i * p.k + l] * b[l * p.n + j]
            out[i * p.n + j] = acc * aScale[if (p.rowwiseA) i else 0] * bScale[if (p.rowwiseB) j else 0]
        }
        return out
    }
}
