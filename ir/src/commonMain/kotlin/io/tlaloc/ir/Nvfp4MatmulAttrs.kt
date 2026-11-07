package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.core.U8

/**
 * [OpKind.NVFP4_MATMUL]'s operands, the packed weight layout, and the
 * reference arithmetic, one place for every layer that touches the kind.
 * ```
 *   0 x       [M, K]               f32
 *   1 codes   [T, K / 64, 512]     u8, T = ceil(N / 16)
 *   2 scales  [T, K / 64, 64]      u8 (e4m3fn bits)
 *   3 scale2  [N]                  f32, the tensor scale of each output row
 *   -> y      [M, N]               f32
 * attrs: fused_kernel: Boolean (optional) -- emit the CUDA kernel tlaloc_fp4_gemm
 *        (triton/kernels/fp4_gemm.cu) where it applies
 *        act_quant: "nvfp4" (optional, with fused_kernel) -- W4A4: x quantized per row to
 *        NVFP4 ([quantizeActivations]) and the products on the block-scaled FP4 tensor-core
 *        instruction (tlaloc_fp4_gemm_a4, triton/kernels/moe_fp4.cu); N a multiple of 256 and
 *        K of 128 ([actFp4Shape])
 * ```
 * The weight `W [N, K]` is NVFP4: e2m1 codes, an e4m3 scale per 16
 * consecutive values of a row, and a tensor scale, given per output row so
 * that weights stacked along N keep their own:
 * `W[n, k] = e2m1(code) * e4m3(scale[n, k / 16]) * scale2[n]`. `y = (bf16(x)
 * W'^T) * scale2`, `W'` without `scale2`, summed in f32: a code times its
 * group scale is exact in bf16, so this is what a bf16 dot of the widened
 * weight computes.
 *
 * The codes and scales are in the order the kernel's tensor-core fragments
 * read them ([pack]); a reshape and a transpose take them back to the
 * checkpoint's `[N, K / 2]` and `[N, K / 16]`:
 * code byte `[t, g, r, j, s, h, q]` (dims `[T, K/64, 8, 4, 4, 2, 2]`) is row
 * `16 t + 8 q + r`, byte `32 g + 8 s + 4 h + j` of the row; scale byte
 * `[t, g, r, s]` (dims `[T, K/64, 16, 4]`) is row `16 t + r`, group `4 g + s`.
 */
object Nvfp4MatmulAttrs {

    const val FUSED_KERNEL = "fused_kernel"

    /** `act_quant = "nvfp4"`: W4A4 ([referenceActFp4]). */
    const val ACT_QUANT = "act_quant"
    const val NVFP4 = "nvfp4"

    /** Shapes the W4A4 kernel takes: 16 output tiles a block and whole 128-column pipeline stages. */
    fun actFp4Shape(inFeatures: Int, outFeatures: Int): Boolean = outFeatures % 256 == 0 && inFeatures % 128 == 0

    /** The W4A4 kernel's scratch for [rows] rows of [inFeatures] (moe_fp4.cu's DenseFp4ScratchBytes). */
    fun actFp4ScratchBytes(rows: Int, inFeatures: Int): Long {
        fun a16(n: Long) = (n + 15) / 16 * 16
        val m = rows.toLong()
        return a16(8) + 2 * a16(4 * m) + a16(m * inFeatures / 2) + a16(m * inFeatures / 16) + a16(4 * m)
    }

    /** Whether [op] quantizes its activations (checked by [parse]). */
    fun actFp4(op: DxirOp): Boolean = op.attrs[ACT_QUANT] != null

    /** The most rows of x the fused kernel takes. */
    const val KERNEL_MAX_ROWS = 16

    /**
     * The most rows run as several calls of the fused kernel. Each reads the whole weight, and unpacking
     * it instead costs about 13 passes (Ripple-65B: 2 s against 160 ms for a step's 160 products).
     */
    const val KERNEL_PASSES_MAX_ROWS = 128

    data class Parsed(val rows: Int, val inFeatures: Int, val outFeatures: Int, val tiles: Int)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.NVFP4_MATMUL) { "$layer: expected NVFP4_MATMUL, got ${op.op}" }
        require(op.operands.size == 4) { "$layer: NVFP4_MATMUL takes x, codes, scales and scale2, got ${op.operands.size} operands" }
        val (x, codes, scales, s2) = op.operands
        require(x.type.dtype == F32 && x.type.rank == 2) { "$layer: NVFP4_MATMUL x must be f32 [M, K], got ${x.type}" }
        val m = x.type.dims[0]
        val k = x.type.dims[1]
        require(k % 64 == 0) { "$layer: NVFP4_MATMUL needs K a multiple of 64, got $k" }
        require(op.type.dtype == F32 && op.type.rank == 2 && op.type.dims[0] == m) {
            "$layer: NVFP4_MATMUL result must be f32 [$m, N], got ${op.type}"
        }
        val n = op.type.dims[1]
        val t = (n + 15) / 16
        require(codes.type.dtype == U8 && codes.type.dims == listOf(t, k / 64, 512)) {
            "$layer: NVFP4_MATMUL codes must be u8 [$t, ${k / 64}, 512], got ${codes.type}"
        }
        require(scales.type.dtype == U8 && scales.type.dims == listOf(t, k / 64, 64)) {
            "$layer: NVFP4_MATMUL scales must be u8 [$t, ${k / 64}, 64], got ${scales.type}"
        }
        require(s2.type.dtype == F32 && s2.type.dims == listOf(n)) { "$layer: NVFP4_MATMUL scale2 must be f32 [$n], got ${s2.type}" }
        when (val a = op.attrs[ACT_QUANT]) {
            null -> {}
            NVFP4 -> {
                require(op.attrs[FUSED_KERNEL] == true) { "$layer: NVFP4_MATMUL $ACT_QUANT '$NVFP4' runs only in the CUDA kernel ($FUSED_KERNEL)" }
                require(actFp4Shape(k, n)) { "$layer: NVFP4_MATMUL $ACT_QUANT '$NVFP4' needs N a multiple of 256 and K of 128, got [$n, $k]" }
            }
            else -> throw IllegalArgumentException("$layer: NVFP4_MATMUL $ACT_QUANT is '$NVFP4' or absent, got '$a'")
        }
        return Parsed(m, k, n, t)
    }

    /**
     * The checkpoint's codes `[n, k / 2]` (low nibble first) and group scales
     * `[n, k / 16]` in the packed order; rows past [n] up to a multiple of 16
     * are zero.
     */
    fun pack(codes: ByteArray, scales: ByteArray, n: Int, k: Int): Pair<ByteArray, ByteArray> {
        require(k % 64 == 0) { "Nvfp4MatmulAttrs.pack: K $k is not a multiple of 64" }
        require(codes.size.toLong() == n.toLong() * (k / 2) && scales.size.toLong() == n.toLong() * (k / 16)) {
            "Nvfp4MatmulAttrs.pack: ${codes.size} code bytes and ${scales.size} scales for [$n, $k]"
        }
        val g = k / 64
        val t = (n + 15) / 16
        val outCodes = ByteArray(t * g * 512)
        val outScales = ByteArray(t * g * 64)
        var o = 0
        for (ti in 0 until t) for (gi in 0 until g) {
            for (r in 0 until 8) for (j in 0 until 4) for (s in 0 until 4) for (h in 0 until 2) for (q in 0 until 2) {
                val row = 16 * ti + 8 * q + r
                outCodes[o++] = if (row < n) codes[row * (k / 2) + 32 * gi + 8 * s + 4 * h + j] else 0
            }
        }
        o = 0
        for (ti in 0 until t) for (gi in 0 until g) for (r in 0 until 16) for (s in 0 until 4) {
            val row = 16 * ti + r
            outScales[o++] = if (row < n) scales[row * (k / 16) + 4 * gi + s] else 0
        }
        return outCodes to outScales
    }

    /** `W'[n, k]` (without scale2) from packed codes and scales given as byte values. */
    fun weight(codes: FloatArray, scales: FloatArray, n: Int, k: Int, inFeatures: Int): Double {
        val g = inFeatures / 64
        val ti = n / 16
        val r = n % 16
        val gi = k / 64
        val kk = k % 64
        val s = kk / 16
        val w = kk % 16
        // Byte 32 gi + 8 s + 4 h + j of the row holds k = 64 gi + 16 s + 8 h + 2 j (+1, the high nibble).
        val h = w / 8
        val j = (w % 8) / 2
        val byte = codes[(((((ti * g + gi) * 8 + r % 8) * 4 + j) * 4 + s) * 2 + h) * 2 + r / 8].toInt() and 0xFF
        val code = if (w % 2 == 0) byte and 0xF else byte ushr 4
        val scale = scales[((ti * g + gi) * 16 + r) * 4 + s].toInt() and 0xFF
        return io.tlaloc.core.f4e2m1ToFloat(code).toDouble() * io.tlaloc.core.f8e4m3fnToFloat(scale.toByte()).toDouble()
    }

    /**
     * The weights `[count, n, k]` of [count] matrices packed one after
     * another (an expert stack), times their row scales `[count, n]`.
     */
    /**
     * A row of activations quantized to NVFP4 as triton/kernels/moe_fp4.cu's
     * `QuantizeRowsKernel` does it, every step in f32: the row's scale `g = amax / 2688`
     * (6 x 448); per 16 values `s = e4m3(amax16 / (6 g))` (nearest, saturating at 448); each
     * value's e2m1 code of `v / (s g)` (nearest, ties to the even code). Writes `code x s`
     * (without g) into [out] at [at] and returns g. A row of zeros has g = 0 and zero codes.
     */
    fun quantizeActivations(row: FloatArray, from: Int, k: Int, out: DoubleArray, at: Int): Float {
        require(k % 16 == 0) { "Nvfp4MatmulAttrs.quantizeActivations: a row of $k values is not whole groups of 16" }
        var amax = 0f
        for (i in 0 until k) amax = maxOf(amax, kotlin.math.abs(row[from + i]))
        val g = amax / 2688f
        val g6 = g * 6f
        for (q in 0 until k / 16) {
            var a = 0f
            for (i in 0 until 16) a = maxOf(a, kotlin.math.abs(row[from + 16 * q + i]))
            val sv = if (g6 > 0f) io.tlaloc.core.saturateToF8e4m3fn(a / g6) else 0f
            val sg = sv * g
            for (i in 0 until 16) {
                val c = if (sg > 0f) io.tlaloc.core.floatToF4e2m1(row[from + 16 * q + i] / sg) else 0
                out[at + 16 * q + i] = io.tlaloc.core.f4e2m1ToFloat(c).toDouble() * sv
            }
        }
        return g
    }

    fun dequantize(codes: FloatArray, scales: FloatArray, scale2: FloatArray, count: Int, n: Int, k: Int): DoubleArray {
        val t = (n + 15) / 16
        val perCodes = t * (k / 64) * 512
        val perScales = t * (k / 64) * 64
        val out = DoubleArray(count * n * k)
        for (e in 0 until count) {
            val c = codes.copyOfRange(e * perCodes, (e + 1) * perCodes)
            val s = scales.copyOfRange(e * perScales, (e + 1) * perScales)
            for (row in 0 until n) for (col in 0 until k) {
                out[(e * n + row) * k + col] = weight(c, s, row, col, k) * scale2[e * n + row]
            }
        }
        return out
    }

    /**
     * W4A4: each row of x quantized by [quantizeActivations] (its scale g, codes times their
     * group scales), `y = g (x' W'^T) * scale2`, the products summed in double.
     */
    fun referenceActFp4(p: Parsed, x: FloatArray, codes: FloatArray, scales: FloatArray, scale2: FloatArray): DoubleArray {
        val k = p.inFeatures
        val xq = DoubleArray(p.rows * k)
        val g = DoubleArray(p.rows) { m -> quantizeActivations(x, m * k, k, xq, m * k).toDouble() }
        val y = DoubleArray(p.rows * p.outFeatures)
        val wRow = DoubleArray(k)
        for (n in 0 until p.outFeatures) {
            for (c in 0 until k) wRow[c] = weight(codes, scales, n, c, k)
            for (m in 0 until p.rows) {
                var s = 0.0
                for (c in 0 until k) s += wRow[c] * xq[m * k + c]
                y[m * p.outFeatures + n] = s * g[m] * scale2[n]
            }
        }
        return y
    }

    /** y = (bf16(x) W'^T) * scale2, the products summed in double: the interpreters' arm. */
    fun reference(p: Parsed, x: FloatArray, codes: FloatArray, scales: FloatArray, scale2: FloatArray): DoubleArray {
        val y = DoubleArray(p.rows * p.outFeatures)
        val wRow = DoubleArray(p.inFeatures)
        for (n in 0 until p.outFeatures) {
            for (k in 0 until p.inFeatures) wRow[k] = weight(codes, scales, n, k, p.inFeatures)
            for (m in 0 until p.rows) {
                var s = 0.0
                for (k in 0 until p.inFeatures) {
                    val xb = io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(x[m * p.inFeatures + k]))
                    s += wRow[k] * xb
                }
                y[m * p.outFeatures + n] = s * scale2[n]
            }
        }
        return y
    }
}
