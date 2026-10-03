package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I8
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.io.JsonException
import java.io.OutputStream

// The bridge from "a tensor by role" to "the operand list a decode graph is
// called with". Its one job is the transpose of the Linears.

/**
 * Stage a checkpoint's weights into [HfDecoderGraph.weightSlots] order.
 *
 * ## The transpose, performed
 *
 * HF stores every
 * `nn.Linear` weight as `[out_features, in_features]` — because
 * `F.linear(x, W)` is `x @ W.T`. Tlaloc's [io.tlaloc.ir.OpKind.MATMUL]
 * contracts `last(A) × first(B)`, so a projection wants `[in, out]`. This
 * function is where the two meet, and it is deliberately the ONLY place: the
 * graph has no transpose node, and every downstream reader of a staged buffer
 * can assume math layout.
 *
 * WHY HOST-SIDE AND ONCE. The weight is a graph PARAMETER (see
 * [DecodeGraphSpec.weightSlots]), so an in-graph
 * [io.tlaloc.ir.OpKind.TRANSPOSE] has no constant to fold into and would
 * re-lay-out every projection matrix on every decode step — at TinyLlama's
 * shapes, ~1.1e9 element moves per token, to save a one-off pass at load.
 *
 * WHICH TENSORS. Exactly the ones [HfDecoderNames.isTransposedLinear] says: the
 * seven projections per layer and `lm_head`. A tied head has no slot: the
 * graph contracts the final hidden state against the embedding table's hidden
 * axis, so the table is on the device once (a separate transposed copy was
 * 594 MiB in f32 for Qwen3-0.6B). [HfDecoderConfig.tiedHeadCopy] stages the
 * copy anyway, as a control. The two RMSNorm gains are rank-1
 * and the embedding table is a LOOKUP, not a Linear — transposing either would
 * be a bug that a square model could not detect, which is why the inventory
 * check and this function both go through the same predicate rather than a
 * second list.
 *
 * ## Dtype
 *
 * A slot is staged in [HfDecoderConfig.weightDType]. For **f32** (Llama,
 * Qwen3) [io.tlaloc.core.io.WeightSource] widens bf16 bytes exactly
 * (bf16→f32 is lossless, it is a 16-bit left shift), and [stageAt] returns
 * the floats. For **bf16** (Muse Glimmer, 28 billion parameters, whose f32
 * copy would not fit the device) [writeSlot] copies the file's bf16 bytes to
 * the artifact a block at a time, transposing the Linears on the way, so no
 * tensor is ever widened and no JVM array holds more than [BLOCK_BYTES].
 *
 * Under [WeightQuant.INT8] a layer's Linear is staged as int8 codes and its
 * f32 per-output-channel scales, two slots ([quantizeInt8]); the codes are
 * the one array held whole, one byte per element.
 */
object HfStagedWeights {

    /**
     * Read every slot of [HfDecoderGraph.weightSlots] for [config] out of
     * [ckpt], transposing the Linears, and return them in call order.
     *
     * [config] may be a REDUCED copy of the checkpoint's own config (fewer
     * layers); the roles it implies are then a prefix-by-layer of the file's,
     * which is exactly what a reduced-layer certification wants. Everything
     * else — hidden size, head counts, vocab — must still match, and [load]
     * refuses per tensor if it does not because the dim check is
     * against the config it is handed.
     */
    fun stage(ckpt: HfCheckpoint, config: HfDecoderConfig = ckpt.config): List<FloatArray> {
        require(config.numLayers <= ckpt.config.numLayers) {
            "HfStagedWeights.stage: asked for ${config.numLayers} layers but the " +
                "checkpoint has ${ckpt.config.numLayers}"
        }
        val slots = HfDecoderGraph.weightSlots(config)
        val sources = HfDecoderGraph.weightSlotSources(config)
        check(slots.size == sources.size) {
            "HfStagedWeights: ${slots.size} slots vs ${sources.size} sources — " +
                "HfDecoderGraph.weightSlots and .weightSlotSources must stay in lockstep"
        }
        return slots.indices.map { i -> stageAt(ckpt, config, i) }
    }

    /**
     * One slot of [stage], by index, with nothing else resident.
     *
     * `ServingArtifactWriter` writes the staged weights
     * one file at a time, and a `List<FloatArray>` of TinyLlama-1.1B is 4.4 GB
     * of live heap for no reason — the exporter only ever looks at one tensor.
     * [stage] is now this function in a loop, so the two cannot disagree.
     */
    fun stageAt(ckpt: HfCheckpoint, config: HfDecoderConfig, index: Int): FloatArray {
        val slots = HfDecoderGraph.weightSlots(config)
        val sources = HfDecoderGraph.weightSlotSources(config)
        require(index in slots.indices) {
            "HfStagedWeights.stageAt: slot $index is outside 0..${slots.size - 1}"
        }
        val source = sources[index]
        if (HfDecoderNames.isQuantized(source.role, config)) {
            // The codes as floats (the interpreter's convention for every
            // dtype), or the scales.
            val q = quantizeInt8(ckpt, source.role, config)
            return if (source.scale) q.scales.copyOf() else FloatArray(q.codes.size) { q.codes[it].toFloat() }
        }
        return run {
            val i = index
            val role = source.role
            val slot = slots[i]
            if (readsHead(role, config)) ckpt.verifyTiedHead()
            val t = loadFor(ckpt, role, config)
            val data = t.toF32Array()
            val fileDims = t.dims
            val staged = if (HfDecoderNames.isTransposedLinear(role)) {
                require(fileDims.size == 2) {
                    "HfStagedWeights: $role is a Linear but its file tensor is " +
                        "${fileDims.toList()} — expected rank 2"
                }
                transpose(data, fileDims[0], fileDims[1])
            } else if (isConvKernel(role)) {
                // [C, 1, K] -> [K, C]: the middle axis is 1, so this is a transpose of [C, K].
                transpose(data, fileDims[0], fileDims[2])
            } else {
                data
            }
            val want = slot.type.dims.fold(1) { a, b -> a * b }
            check(staged.size == want) {
                "HfStagedWeights: slot '${slot.name}' wants $want elements " +
                    "(${slot.type.dims}) but $role staged ${staged.size} from " +
                    "${fileDims.toList()}"
            }
            staged
        }
    }

    /**
     * Load one role, verifying against [config] rather than the checkpoint's
     * own — so a reduced-layer config still gets the dim check, and a
     * reduced config that quietly disagrees about hidden size or head count
     * is caught at the tensor, by name, instead of inside a matmul.
     */
    private fun loadFor(
        ckpt: HfCheckpoint,
        role: DecoderWeightRole,
        config: HfDecoderConfig,
    ): io.tlaloc.core.io.LoadedTensor {
        val t = ckpt.load(role)
        val want = HfDecoderNames.expectedDims(role, config)
        if (!t.dims.contentEquals(want)) {
            throw JsonException(
                "HfStagedWeights: ${ckpt.resolveName(role)} is ${t.dims.toList()} but the " +
                    "config this graph is being built for says ${want.toList()} — the reduced " +
                    "config disagrees with the checkpoint about something other than layer count",
            )
        }
        return t
    }

    /**
     * Whether staging [role] stages the head: the head's own slot, or the
     * embedding table when a tied head reads it directly. A tied checkpoint
     * that also stores `lm_head.weight` is then checked to store the table.
     */
    private fun readsHead(role: DecoderWeightRole, config: HfDecoderConfig): Boolean =
        role == DecoderWeightRole.LmHead ||
            (role == DecoderWeightRole.EmbedTokens && HfDecoderGraph.headReadsEmbedding(config))

    private fun isConvKernel(role: DecoderWeightRole): Boolean =
        role is DecoderWeightRole.Layer && role.part == DecoderLayerPart.CONV1D

    /** The largest piece of a tensor [writeSlot] holds at once, in bytes. */
    const val BLOCK_BYTES: Int = 256 * 1024 * 1024

    /**
     * Write slot [index] to [out] as little-endian bytes of the slot's dtype
     * (math layout, the Linears transposed), and return the byte count.
     *
     * F32 slots are [stageAt]'s floats. BF16 slots whose file tensor is bf16
     * are copied without decoding, in blocks: a Linear stored `[rows, cols]`
     * is written `[cols, rows]` a band of output rows at a time, each band
     * filled by reading the source in chunks of input rows, so the embedding
     * table and the head (2.7 GB each for Muse Glimmer, more than a JVM array
     * holds) never sit in memory whole. A BF16 slot whose file tensor is not
     * bf16 is narrowed from [stageAt]'s floats, round to nearest even.
     *
     * @param blockBytes the largest piece held at once ([BLOCK_BYTES]; a test
     *   passes a small one to exercise the banded path on a small tensor).
     */
    fun writeSlot(
        ckpt: HfCheckpoint,
        config: HfDecoderConfig,
        index: Int,
        out: OutputStream,
        blockBytes: Int = BLOCK_BYTES,
    ): Long {
        require(blockBytes >= 2) { "HfStagedWeights.writeSlot: blockBytes must be >= 2, got $blockBytes" }
        val slots = HfDecoderGraph.weightSlots(config)
        val sources = HfDecoderGraph.weightSlotSources(config)
        require(index in slots.indices) {
            "HfStagedWeights.writeSlot: slot $index is outside 0..${slots.size - 1}"
        }
        val slot = slots[index]
        val role = sources[index].role
        when (slot.type.dtype) {
            I8 -> {
                val q = quantizeInt8(ckpt, role, config, blockBytes)
                out.write(q.codes)
                return q.codes.size.toLong()
            }
            F32 -> {
                val data = stageAt(ckpt, config, index)
                val buf = java.nio.ByteBuffer.allocate(4 * 65536).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                var i = 0
                while (i < data.size) {
                    buf.clear()
                    val n = minOf(65536, data.size - i)
                    for (k in 0 until n) buf.putFloat(data[i + k])
                    out.write(buf.array(), 0, 4 * n)
                    i += n
                }
                return 4L * data.size
            }
            BF16 -> {
                if (readsHead(role, config)) ckpt.verifyTiedHead()
                val e = ckpt.entry(role)
                val want = HfDecoderNames.expectedDims(role, config)
                if (e.dims != want.toList()) {
                    throw JsonException(
                        "HfStagedWeights: ${ckpt.resolveName(role)} is ${e.dims} but the config " +
                            "this graph is being built for says ${want.toList()}",
                    )
                }
                if (e.wireDType != "BF16") {
                    val data = stageAt(ckpt, config, index)
                    val bytes = ByteArray(2 * data.size)
                    for (k in data.indices) {
                        val b = floatToBf16Bits(data[k]).toInt()
                        bytes[2 * k] = b.toByte()
                        bytes[2 * k + 1] = (b shr 8).toByte()
                    }
                    out.write(bytes)
                    return bytes.size.toLong()
                }
                return if (HfDecoderNames.isTransposedLinear(role) || isConvKernel(role)) {
                    copyTransposedBf16(ckpt, role, e.dims[0], e.dims.last(), out, blockBytes)
                } else {
                    copyBf16(ckpt, role, e.byteLength, out, blockBytes)
                }
            }
            else -> throw JsonException(
                "HfStagedWeights.writeSlot: slot '${slot.name}' is ${slot.type.dtype}; weights are " +
                    "staged as F32, BF16 or (quantized) I8",
            )
        }
    }

    /**
     * A Linear quantized to int8 per output channel ([WeightQuant.INT8]):
     * [codes] in math layout `[in, out]` (row-major, the file's `[out, in]`
     * transposed) and one [scales] entry per output channel.
     */
    class Int8Linear(val codes: ByteArray, val scales: FloatArray)

    /**
     * Quantize [role]'s weight as [WeightQuant] describes: per output channel
     * (a row of the file's `[out, in]`), `scale = max|w| / 127` in f32 (1 for
     * an all-zero row) and `code = clamp(rint(w / scale), -127, 127)`, with
     * `w` the checkpoint's value widened to f32.
     *
     * The rows are read [blockBytes] at a time, so only the codes (one byte
     * per element) are held whole. The last result is kept, because the
     * exporter writes a weight's codes and then its scales as two slots.
     */
    fun quantizeInt8(
        ckpt: HfCheckpoint,
        role: DecoderWeightRole,
        config: HfDecoderConfig,
        blockBytes: Int = BLOCK_BYTES,
    ): Int8Linear {
        require(HfDecoderNames.isTransposedLinear(role)) {
            "HfStagedWeights.quantizeInt8: $role is not a Linear weight"
        }
        synchronized(this) {
            val hit = lastQuantized
            if (hit != null && hit.first === ckpt && hit.second == role) return hit.third
        }
        val e = ckpt.entry(role)
        val want = HfDecoderNames.expectedDims(role, config)
        if (e.dims != want.toList()) {
            throw JsonException(
                "HfStagedWeights: ${ckpt.resolveName(role)} is ${e.dims} but the config " +
                    "this graph is being built for says ${want.toList()}",
            )
        }
        val rows = e.dims[0]
        val cols = e.dims[1]
        require(rows.toLong() * cols <= Int.MAX_VALUE - 8) {
            "HfStagedWeights.quantizeInt8: ${ckpt.resolveName(role)} has ${rows.toLong() * cols} " +
                "elements, more than one array of codes holds"
        }
        val codes = ByteArray(rows * cols)
        val scales = FloatArray(rows)
        val row = FloatArray(cols)
        forEachRow(ckpt, role, e.wireDType, rows, cols, blockBytes, row) { r ->
            var max = 0f
            for (c in 0 until cols) max = maxOf(max, kotlin.math.abs(row[c]))
            val scale = if (max == 0f) 1f else max / 127f
            scales[r] = scale
            for (c in 0 until cols) {
                val q = Math.rint((row[c] / scale).toDouble()).coerceIn(-127.0, 127.0).toInt()
                codes[c * rows + r] = q.toByte()
            }
        }
        val result = Int8Linear(codes, scales)
        synchronized(this) { lastQuantized = Triple(ckpt, role, result) }
        return result
    }

    private var lastQuantized: Triple<HfCheckpoint, DecoderWeightRole, Int8Linear>? = null

    /**
     * Call [body] with each row of a `[rows, cols]` tensor widened to f32 in
     * [row], reading whole rows [blockBytes] at a time. BF16 and F32 files
     * are read in place; any other wire dtype is loaded whole.
     */
    private inline fun forEachRow(
        ckpt: HfCheckpoint,
        role: DecoderWeightRole,
        wire: String,
        rows: Int,
        cols: Int,
        blockBytes: Int,
        row: FloatArray,
        body: (Int) -> Unit,
    ) {
        val width = when (wire) {
            "BF16" -> 2
            "F32" -> 4
            else -> 0
        }
        if (width == 0) {
            val all = ckpt.load(role).toF32Array()
            for (r in 0 until rows) {
                System.arraycopy(all, r * cols, row, 0, cols)
                body(r)
            }
            return
        }
        val chunkRows = maxOf(1, minOf(rows.toLong(), blockBytes / (width.toLong() * cols)).toInt())
        val chunk = ByteArray(width * chunkRows * cols)
        val buf = java.nio.ByteBuffer.wrap(chunk).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var r0 = 0
        while (r0 < rows) {
            val rc = minOf(chunkRows, rows - r0)
            ckpt.readBytes(role, width.toLong() * r0 * cols, chunk, 0, width * rc * cols)
            for (r in 0 until rc) {
                val base = width * r * cols
                if (width == 2) {
                    for (c in 0 until cols) {
                        val bits = (buf.getShort(base + 2 * c).toInt() and 0xFFFF) shl 16
                        row[c] = Float.fromBits(bits)
                    }
                } else {
                    for (c in 0 until cols) row[c] = buf.getFloat(base + 4 * c)
                }
                body(r0 + r)
            }
            r0 += rc
        }
    }

    private fun copyBf16(
        ckpt: HfCheckpoint,
        role: DecoderWeightRole,
        length: Long,
        out: OutputStream,
        blockBytes: Int,
    ): Long {
        val buf = ByteArray(maxOf(1L, minOf(length, blockBytes.toLong())).toInt())
        var at = 0L
        while (at < length) {
            val n = minOf(buf.size.toLong(), length - at).toInt()
            ckpt.readBytes(role, at, buf, 0, n)
            out.write(buf, 0, n)
            at += n
        }
        return length
    }

    /**
     * `[rows, cols]` bf16 in the file → `[cols, rows]` in [out], in bands of
     * output rows. Each band reads the whole source once, in chunks of whole
     * input rows; a matrix up to `blockBytes` is one band and one read.
     */
    private fun copyTransposedBf16(
        ckpt: HfCheckpoint,
        role: DecoderWeightRole,
        rows: Int,
        cols: Int,
        out: OutputStream,
        blockBytes: Int,
    ): Long {
        val bandCols = maxOf(1, minOf(cols.toLong(), blockBytes / (2L * rows)).toInt())
        val chunkRows = maxOf(1, minOf(rows.toLong(), blockBytes / (2L * cols)).toInt())
        val band = ByteArray(2 * bandCols * rows)
        val chunk = ByteArray(2 * chunkRows * cols)
        var c0 = 0
        while (c0 < cols) {
            val bc = minOf(bandCols, cols - c0)
            var r0 = 0
            while (r0 < rows) {
                val rc = minOf(chunkRows, rows - r0)
                ckpt.readBytes(role, 2L * r0 * cols, chunk, 0, 2 * rc * cols)
                for (r in 0 until rc) {
                    val src = 2 * (r * cols + c0)
                    for (c in 0 until bc) {
                        val dst = 2 * (c * rows + r0 + r)
                        band[dst] = chunk[src + 2 * c]
                        band[dst + 1] = chunk[src + 2 * c + 1]
                    }
                }
                r0 += rc
            }
            out.write(band, 0, 2 * bc * rows)
            c0 += bc
        }
        return 2L * rows * cols
    }

    /** Row-major `[rows, cols]` → `[cols, rows]`. */
    fun transpose(src: FloatArray, rows: Int, cols: Int): FloatArray {
        require(src.size == rows * cols) {
            "HfStagedWeights.transpose: ${src.size} elements is not $rows x $cols"
        }
        val out = FloatArray(src.size)
        for (r in 0 until rows) {
            val base = r * cols
            for (c in 0 until cols) out[c * rows + r] = src[base + c]
        }
        return out
    }
}
