package io.tlaloc.ir.inference

import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.core.floatToF8e4m3fn
import java.io.OutputStream

/**
 * An n-gram table ([DecoderLayerPart.PLE_TABLE]) staged as NVFP4 for
 * [io.tlaloc.ir.OpKind.NVFP4_EMBEDDING] ([HfDecoderConfig.pleTableQuant]): codes
 * `[V, D / 2]`, group scales `[V, D / G]` and the tensor scale, row-major as stored.
 *
 * A table stored in FP8 shards with a tensor scale `s` (NVIDIA's) is rounded shard by
 * shard, never held whole: its tensor scale is `s / 6`, the bound [Nvfp4Quantizer]'s
 * `max |w| / (6 * 448)` takes for FP8 values, so a group's scale is the e4m3 value of its
 * largest code and every value's e2m1 code depends only on its own and that largest code
 * (two lookups, [Fp8Rows]). The result equals [Nvfp4Quantizer.quantize] of the widened
 * table with that tensor scale. Any other table is read whole and rounded by
 * [Nvfp4Quantizer.quantize] (`max |w| / (6 * 448)`).
 */
internal object PleTableNvfp4 {

    /** Values per group scale of a table of rows of [dim] values ([HfDecoderGraph.pleTableGroups]). */
    fun group(dim: Int): Int = dim / HfDecoderGraph.pleTableGroups(dim)

    /**
     * Rounds rows of FP8 codes of tensor scale [tableScale] to NVFP4 with tensor scale
     * `tableScale / 6`, as [Nvfp4Quantizer.quantize] rounds `f8(code) * tableScale` (the same
     * f32 operations, tabulated over the 128 magnitudes of a group's largest code and the 256
     * codes).
     */
    class Fp8Rows(val tableScale: Float, val dim: Int) {
        val group: Int = group(dim)
        val scale2: Float = tableScale / 6f
        private val scaleOf = ByteArray(128)
        private val codeOf = ByteArray(128 * 256)

        init {
            require(tableScale > 0f && tableScale.isFinite()) { "PleTableNvfp4: the table scale must be positive, got $tableScale" }
            for (g in 0 until 127) {
                val gmax = f8e4m3fnToFloat(g.toByte()) * tableScale
                val sb = floatToF8e4m3fn((gmax / 6f / scale2).coerceAtMost(io.tlaloc.core.F8E4M3FN_MAX))
                scaleOf[g] = sb
                val step = f8e4m3fnToFloat(sb) * scale2
                for (c in 0 until 256) {
                    codeOf[g * 256 + c] = if (step == 0f) 0 else Nvfp4Quantizer.nearest(f8e4m3fnToFloat(c.toByte()) * tableScale / step).toByte()
                }
            }
        }

        /** [rows] rows of [src] from [srcOff] rounded into [codes] (from [codesOff]) and [scales] (from [scalesOff]). */
        fun round(src: ByteArray, srcOff: Int, rows: Int, codes: ByteArray?, codesOff: Int, scales: ByteArray?, scalesOff: Int) {
            val groups = dim / group
            for (r in 0 until rows) {
                for (g in 0 until groups) {
                    val base = srcOff + r * dim + g * group
                    var m = 0
                    for (i in 0 until group) {
                        val mag = src[base + i].toInt() and 0x7F
                        require(mag != 0x7F) { "PleTableNvfp4: the table holds an FP8 NaN" }
                        if (mag > m) m = mag
                    }
                    scales?.set(scalesOff + r * groups + g, scaleOf[m])
                    if (codes != null) {
                        val row = m * 256
                        val at = codesOff + (r * dim + g * group) / 2
                        for (i in 0 until group step 2) {
                            val lo = codeOf[row + (src[base + i].toInt() and 0xFF)].toInt()
                            val hi = codeOf[row + (src[base + i + 1].toInt() and 0xFF)].toInt()
                            codes[at + i / 2] = (lo or (hi shl 4)).toByte()
                        }
                    }
                }
            }
        }
    }

    /** The table rounded whole: for a small table (a test model's). */
    fun whole(ckpt: HfCheckpoint, config: HfDecoderConfig, role: DecoderWeightRole): Nvfp4Quantizer.Quantized {
        val (rows, dim) = HfDecoderNames.expectedDims(role, config).let { it[0] to it[1] }
        val shards = ckpt.tableShards(role)
        val scale = ckpt.tableScale(role)
        if (shards != null && scale != null) {
            val q = Fp8Rows(scale, dim)
            require(rows.toLong() * dim <= Int.MAX_VALUE - 8) { "PleTableNvfp4.whole: $role has more values than one array holds" }
            val codes = ByteArray(rows * dim / 2)
            val scales = ByteArray(rows * (dim / q.group))
            var r0 = 0
            for (name in shards) {
                val e = ckpt.entryNamed(name)
                val n = e.dims[0].toInt()
                val src = ByteArray(n * dim)
                ckpt.readBytesNamed(name, 0L, src)
                q.round(src, 0, n, codes, r0 * dim / 2, scales, r0 * (dim / q.group))
                r0 += n
            }
            require(r0 == rows) { "PleTableNvfp4: the shards of $role hold $r0 rows, the config says $rows" }
            return Nvfp4Quantizer.Quantized(codes, scales, q.scale2)
        }
        val w = ckpt.load(role).toF32Array()
        return Nvfp4Quantizer.quantize(w, rows, dim, group(dim), null)
    }

    /**
     * Writes one of the table's three slots to [out] and returns its bytes. A table in FP8
     * shards is read in blocks of rows, a shard at a time; any other is rounded [whole].
     */
    fun write(ckpt: HfCheckpoint, config: HfDecoderConfig, role: DecoderWeightRole, kind: HfDecoderGraph.Fp4Slot, out: OutputStream): Long {
        val dim = HfDecoderNames.expectedDims(role, config)[1]
        val shards = ckpt.tableShards(role)
        val scale = ckpt.tableScale(role)
        if (shards == null || scale == null || shards.any { ckpt.entryNamed(it).wireDType != "F8_E4M3" }) {
            val q = whole(ckpt, config, role)
            return when (kind) {
                HfDecoderGraph.Fp4Slot.CODES -> q.codes.also { out.write(it) }.size.toLong()
                HfDecoderGraph.Fp4Slot.SCALES -> q.scales.also { out.write(it) }.size.toLong()
                HfDecoderGraph.Fp4Slot.SCALE2 -> writeScale2(q.scale2, out)
            }
        }
        val q = Fp8Rows(scale, dim)
        if (kind == HfDecoderGraph.Fp4Slot.SCALE2) return writeScale2(q.scale2, out)
        val groups = dim / q.group
        val block = maxOf(1, (64 shl 20) / dim)
        val src = ByteArray(block * dim)
        val codes = if (kind == HfDecoderGraph.Fp4Slot.CODES) ByteArray(block * dim / 2) else null
        val scales = if (kind == HfDecoderGraph.Fp4Slot.SCALES) ByteArray(block * groups) else null
        var written = 0L
        var rows = 0L
        for (name in shards) {
            val e = ckpt.entryNamed(name)
            require(e.dims.size == 2 && e.dims[1].toInt() == dim) { "PleTableNvfp4: '$name' is ${e.dims}, rows of $dim expected" }
            val n = e.dims[0].toLong()
            var r = 0L
            while (r < n) {
                val k = minOf(block.toLong(), n - r).toInt()
                ckpt.readBytesNamed(name, r * dim, src, 0, k * dim)
                q.round(src, 0, k, codes, 0, scales, 0)
                if (codes != null) out.write(codes, 0, k * dim / 2) else out.write(scales!!, 0, k * groups)
                written += if (codes != null) k.toLong() * dim / 2 else k.toLong() * groups
                r += k
            }
            rows += n
        }
        val want = HfDecoderNames.expectedDims(role, config)[0].toLong()
        require(rows == want) { "PleTableNvfp4: the shards of $role hold $rows rows, the config says $want" }
        return written
    }

    private fun writeScale2(v: Float, out: OutputStream): Long {
        val b = java.nio.ByteBuffer.allocate(4).order(java.nio.ByteOrder.LITTLE_ENDIAN).putFloat(v)
        out.write(b.array())
        return 4L
    }
}
