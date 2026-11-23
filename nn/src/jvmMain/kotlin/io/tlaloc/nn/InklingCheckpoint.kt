package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.pow

/**
 * An `inkling2_loop` checkpoint directory (`config.json` and its safetensors files), read into the
 * weights [InklingLoop.forward] takes. Linear weights are transposed once to `[in, out]`; the
 * routed experts are dequantized (NVFP4 or MXFP8, as the released files store them, or read as
 * stored when unquantized), their interleaved gate/up rows split into a gate block and an up block
 * per expert, and laid out as [InklingLoop.Weights.experts] reads them.
 *
 * NVFP4: two E2M1 codes per byte (even element in the low nibble), an E4M3 scale per 16 values
 * and a per-expert factor `amax / outer_scale_denominator`. MXFP8: E4M3 values with a scale
 * `2^(s - 127)` per 32 values. These are transformers 5.24.0 `Inkling2LoopDequantizeExperts`.
 */
class InklingCheckpoint(val dir: Path) : AutoCloseable {
    val config: InklingConfig
    private val files: List<SafetensorsFile>
    private val fileOf: Map<String, SafetensorsFile>

    init {
        val root = parseJson(Files.readString(dir.resolve("config.json"))) as JsonObject
        val text = (root["text_config"] as? JsonObject) ?: root
        config = InklingConfig.fromJson { k -> text[k]?.let(::kotlinOf) }
        val paths = Files.list(dir).use { s -> s.filter { it.fileName.toString().endsWith(".safetensors") && it.fileName.toString() != "mtp.safetensors" }.sorted().toList() }
        require(paths.isNotEmpty()) { "inkling2_loop: no safetensors files in $dir" }
        files = paths.map { SafetensorsFile.open(it) }
        fileOf = files.flatMap { f -> f.names.map { it to f } }.toMap()
    }

    override fun close() = files.forEach { it.close() }

    val names: Set<String> get() = fileOf.keys

    private fun file(name: String) = fileOf[name] ?: error("inkling2_loop: no tensor $name in $dir")

    fun dims(name: String): IntArray = file(name).entry(name).dims.toIntArray()

    /** A tensor as f32, any float dtype the files use. */
    fun f32(name: String): FloatArray = file(name).load(name).toF32Array()

    /** The names [InklingLoop.forward] reads through `linear` (stored `[out, in]`). */
    fun isLinear(name: String): Boolean =
        LINEAR_SUFFIXES.any { name.endsWith(it) } || Regex("""controllers\.\w+\.(down|forget|gain)_weight$""").containsMatchIn(name)

    /**
     * Every text-model weight the forward reads, as host tensors in a fixed order, keyed as
     * [Leaves] looks them up: checkpoint names (linears transposed to `[in, out]`) and
     * `experts.<l>.gate_up` `[H, E·2I]`, `experts.<l>.down` `[E, I, H]`. [bf16] keeps them in
     * bf16 (exact for the bf16 checkpoint tensors; the dequantized experts are rounded to it).
     */
    fun hostWeights(bf16: Boolean = false, log: (String) -> Unit = {}): LinkedHashMap<String, DTensor<*, *>> {
        val out = LinkedHashMap<String, DTensor<*, *>>()
        forEachWeight(bf16, log) { k, t -> out[k] = t }
        return out
    }

    /**
     * The weights of [hostWeights] one at a time, in the same order, each released by this reader
     * once [consume] returns: a caller that uploads each one holds a single layer's experts at most.
     */
    fun forEachWeight(bf16: Boolean = false, log: (String) -> Unit = {}, consume: (String, DTensor<*, *>) -> Unit) {
        fun put(key: String, values: FloatArray, dims: IntArray) = consume(
            key,
            if (bf16) DTensor<Shape, BF16>(HostBf16Storage(io.tlaloc.core.floatArrayToBf16Bits(values)), dims, BF16)
            else DTensor<Shape, F32>(HostF32Storage(values), dims, F32),
        )
        for (name in names.filter { it.startsWith(InklingNames.P) && !it.contains(".mlp.experts.") }.sorted()) {
            val d = dims(name)
            val v = f32(name)
            if (isLinear(name)) {
                require(d.size == 2) { "$name: a linear weight must be 2-D (got ${d.toList()})" }
                put(name, transpose(v, d[0], d[1]), intArrayOf(d[1], d[0]))
            } else put(name, v, d)
        }
        for (l in 0 until config.layers) {
            val (gu, down) = experts(l)
            put("experts.$l.gate_up", gu, intArrayOf(config.hidden, config.experts * 2 * config.expertDim))
            put("experts.$l.down", down, intArrayOf(config.experts, config.expertDim, config.hidden))
            log("inkling2_loop: layer $l experts read")
        }
    }

    /** One layer's experts: gate/up `[H, E·2I]` (per expert: I gate columns, then I up columns) and down `[E, I, H]`. */
    fun experts(l: Int): Pair<FloatArray, FloatArray> {
        val e = config.experts
        val i = config.expertDim
        val h = config.hidden
        val w13 = expertMatrix(InklingNames.layer(l, "mlp.experts.w13_weight"), 2 * i, h) // [E, 2I, H], rows interleaved gate/up
        val w2 = expertMatrix(InklingNames.layer(l, "mlp.experts.w2_weight"), h, i) // [E, H, I]
        val gu = FloatArray(h * e * 2 * i)
        // Checkpoint row r = 2·j + part (part 0 gate, 1 up) -> column x·2I + part·I + j. By output row k.
        java.util.stream.IntStream.range(0, h).parallel().forEach { k ->
            for (x in 0 until e) for (r in 0 until 2 * i) gu[k * e * 2 * i + x * 2 * i + (r % 2) * i + r / 2] = w13[(x * 2 * i + r) * h + k]
        }
        val down = FloatArray(e * i * h)
        java.util.stream.IntStream.range(0, e).parallel().forEach { x ->
            for (r in 0 until h) for (c in 0 until i) down[(x * i + c) * h + r] = w2[(x * h + r) * i + c]
        }
        return gu to down
    }

    /** An expert stack `[E, rows, cols]` in f32, dequantized when the files hold it quantized. */
    private fun expertMatrix(name: String, rows: Int, cols: Int): FloatArray =
        dequantizeExperts(name, config.experts, rows, cols) { n -> fileOf[n]?.load(n) }

    companion object {
        /**
         * The expert stack [name] `[E, rows, cols]` in f32: NVFP4 when `<name>_amax` is there, MXFP8
         * when only `<name>_scale` is, else as stored. [load] returns a tensor of the files or null.
         */
        fun dequantizeExperts(name: String, e: Int, rows: Int, cols: Int, load: (String) -> io.tlaloc.core.io.LoadedTensor?): FloatArray {
            val w = load(name) ?: error("inkling2_loop: no tensor $name")
            val scaleT = load(name + "_scale")
            if (scaleT == null) {
                val v = w.toF32Array()
                require(v.size == e * rows * cols) { "$name: ${w.dims.toList()} is not [$e, $rows, $cols]" }
                return v
            }
            val scales = scaleT.bytes()
            val out = FloatArray(e * rows * cols)
            val per = rows * cols
            val amaxT = load(name + "_amax")
            if (amaxT != null) {
                val codes = w.bytes()
                val amax = amaxT.toF32Array()
                val den = load(name + "_outer_scale_denominator")!!.toF32Array()[0]
                require(codes.size * 2 == out.size && scales.size * 16 == out.size) { "$name: NVFP4 sizes do not fit [$e, $rows, $cols]" }
                java.util.stream.IntStream.range(0, e).parallel().forEach { x ->
                    val factor = amax[x] / den
                    for (idx in x * per until (x + 1) * per) {
                        val b = codes[idx / 2].toInt() and 0xFF
                        val code = if (idx % 2 == 0) b and 0xF else b ushr 4
                        out[idx] = E2M1[code] * E4M3[scales[idx / 16].toInt() and 0xFF] * factor
                    }
                }
            } else {
                require(w.dtype == io.tlaloc.core.F8E4M3FN) { "$name: quantized as ${w.dtype.name}, expected F8_E4M3 or NVFP4 codes" }
                val values = w.bytes()
                require(values.size == out.size && scales.size * 32 == out.size) { "$name: MXFP8 sizes do not fit [$e, $rows, $cols]" }
                java.util.stream.IntStream.range(0, e).parallel().forEach { x ->
                    for (idx in x * per until (x + 1) * per) out[idx] = E4M3[values[idx].toInt() and 0xFF] * E8M0[scales[idx / 32].toInt() and 0xFF]
                }
            }
            return out
        }

        private val LINEAR_SUFFIXES = listOf(
            "attn.wq_du.weight", "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wo_ud.weight", "attn.gate_attn.weight",
            "attn.attn_sink_proj.weight", "attn.wr_du.weight", "sconv.weight_proj", "mlp.gate.weight",
        )

        /** E4M3FN and E8M0 (`2^(s - 127)`) by byte. */
        private val E4M3 = FloatArray(256) { e4m3(it.toByte()) }
        private val E8M0 = FloatArray(256) { 2.0.pow(it - 127).toFloat() }

        private val E2M1 = floatArrayOf(0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f)

        /** OCP float8 E4M3FN. */
        fun e4m3(b: Byte): Float {
            val v = b.toInt() and 0xFF
            val sign = if (v and 0x80 != 0) -1f else 1f
            val exp = (v ushr 3) and 0xF
            val man = v and 0x7
            if (exp == 15 && man == 7) return Float.NaN
            return sign * if (exp == 0) (man / 8f) * 2f.pow(-6) else (1f + man / 8f) * 2f.pow(exp - 7)
        }

        private fun transpose(v: FloatArray, rows: Int, cols: Int): FloatArray {
            val o = FloatArray(v.size)
            for (r in 0 until rows) for (c in 0 until cols) o[c * rows + r] = v[r * cols + c]
            return o
        }

        private fun kotlinOf(v: JsonValue): Any? = when (v) {
            is JsonNumber -> v.value
            is JsonString -> v.value
            is JsonBool -> v.value
            is JsonArray -> v.elements.map(::kotlinOf)
            is JsonObject -> v.fields.mapValues { kotlinOf(it.value) }
            is JsonNull -> null
            else -> null
        }
    }

    /** [InklingLoop.Weights] over traced leaves, looked up by the keys of [hostWeights]. */
    class Leaves(private val byKey: Map<String, Tracer<Shape>>) : InklingLoop.Weights {
        override fun tensor(name: String) = byKey[name] ?: error("inkling2_loop: no weight $name")
        override fun linear(name: String) = tensor(name)
        override fun experts(layer: Int) = tensor("experts.$layer.gate_up") to tensor("experts.$layer.down")
    }
}
