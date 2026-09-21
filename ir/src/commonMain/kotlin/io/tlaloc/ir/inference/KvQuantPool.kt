package io.tlaloc.ir.inference

import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.KvQuantDtype
import io.tlaloc.ir.recognizer.quant.KvScaleStrategy
import kotlin.math.abs
import kotlin.math.floor

/**
 * §0.4.472 — Phase H5: the KV-POOL QUANTIZATION CONTRACT — the layout, the
 * numerics, and the error bound that makes an int8 page pool checkable rather
 * than merely smaller.
 *
 * Until this slice, "KV-quant" in Tlaloc was a *directive*: §0.4.257's
 * [KvQuantConfig] rides on a COARSENED attention op and tells a vendor kernel
 * "materialize K and V narrow at runtime", with nothing in the IR, nothing in
 * the interpreter, and no number anyone could check. That is honest as a
 * kernel hint and useless as a capability. This file is the other half: the
 * exact map from an f32 page pool to (integer codes, per-head scales) and
 * back, with a bound on what the trip costs.
 *
 * # The layout
 *
 * A pool is `[numBlocks, blockSize, numKvHeads, headDim]` (the axis order the
 * whole arc uses — [DecodeModelShape], `ServingModelShape.kvPoolAxisOrder`,
 * `PAGED_ATTENTION`). Quantized, it becomes TWO tensors:
 *
 * ```
 *   codes  [numBlocks, blockSize, numKvHeads, headDim]   integer, |code| <= qmax
 *   scales [numKvHeads]   (PER_HEAD)  or  [1]  (PER_TENSOR)      float
 * ```
 *
 * and the dequantization is the elementwise product
 * `pool[b, s, h, d] = codes[b, s, h, d] * scales[h]` — which is exactly
 * [io.tlaloc.ir.OpKind.DEQUANTIZE_KV]'s semantics, so the host codec here and
 * the in-graph op are two spellings of one formula and the oracle test pins
 * that they agree.
 *
 * **Why the kv-HEAD axis carries the scales.** It is the only pool axis that
 * is neither page-allocator bookkeeping nor within-vector structure: `numBlocks`
 * and `blockSize` index *pages*, which the allocator reassigns between
 * sequences (a per-page scale would have to be rewritten every time a page is
 * recycled, and a decode step writing one slot would invalidate the scale of
 * every other token sharing its page); `headDim` is inside a single key
 * vector, where a per-element scale is just an f32 pool with extra steps.
 * Per-head is also what production serving does (vLLM's fp8 KV cache keys its
 * scales by layer and head), and it is what [KvScaleStrategy.PER_HEAD] has
 * meant since §0.4.257 — this file is the first place the meaning is executed.
 *
 * # The numerics: symmetric absmax
 *
 * Per group g (a kv head, or the whole tensor):
 * ```
 *   absmax = max |x|                 over the group
 *   scale  = absmax / qmax           (absmax == 0  =>  scale = 1, see below)
 *   code   = clamp(roundHalfAway(x / scale), -qmax, +qmax)
 *   x̂      = code * scale
 * ```
 * with `qmax` = 127 for int8 and 7 for int4 ([KvQuantDtype.codeMax]).
 *
 * **The error bound, hand-derived.** For any x in the group,
 * `|x| <= absmax = scale * qmax`, so `x / scale` lies in `[-qmax, +qmax]` and
 * the clamp NEVER fires for a value the scale was computed from (it is there
 * for a caller who supplies scales from elsewhere — a manifest, a previous
 * step's pool — and it is the only reason codes cannot go out of range).
 * Rounding therefore contributes the whole error:
 * ```
 *   |x - x̂| = scale * |x/scale - round(x/scale)| <= scale / 2 = absmax / (2*qmax)
 * ```
 * i.e. **absmax/254 for int8, absmax/14 for int4** — an ABSOLUTE bound per
 * element, and [maxRoundTripError] recomputes it so a test can assert the
 * measured error against the derivation rather than against a magic number.
 *
 * **Ties round AWAY FROM ZERO, explicitly.** `kotlin.math.round` breaks ties
 * towards positive infinity (`round(-0.5) = -0.0`), which makes the code set
 * asymmetric in a way that shows up as a small systematic negative bias on
 * real weights. [roundHalfAway] is spelled out rather than inherited so the
 * convention is a decision in this file and not a property of a stdlib
 * function: `q(0.5) = 1`, `q(-0.5) = -1`, `q(-1.5) = -2`.
 *
 * # What is NOT here, and why (the named deferrals)
 *
 * - **FP8 is REFUSED, not missing.** int8/int4 are INTEGER-CODE formats: a
 *   code is a small integer, so it rides today's integer tensors losslessly
 *   and the dequantization is one multiply. fp8 (e4m3/e5m2) is a FLOAT format
 *   whose "code" is a bit pattern with its own exponent field; representing it
 *   as an integer code would either store the bit pattern (making the multiply
 *   meaningless) or round twice (fp8 then f32, a different and worse error
 *   bound than the one derived above). fp8 wants a narrow [io.tlaloc.core.DType]
 *   the way bf16 got one in §0.4.455, and [KvQuantDtype.isIntegerCoded] is the
 *   predicate that says so by name.
 * - **The codes ride an I32 tensor, so v1 saves no BYTES yet.** There is no
 *   I8 DType (the sealed [io.tlaloc.core.DType] is F32/F64/BF16/I32/I64/Bool),
 *   and adding one is bf16's whole §0.4.455–458 tour — host storage, PJRT
 *   buffer types, emitter type names, every exhaustive `when`. What v1 buys is
 *   the CONTRACT: the layout, the scales, the bound, the in-graph dequant, and
 *   the manifest field that states the format. The byte-narrow pool is the
 *   named tail, and the artifact says so out loud
 *   (`ServingKvQuant.codeDtype` = "i32" today).
 * - **No calibration.** Scales come from the pool's own absmax, computed when
 *   the pool is quantized. Activation-aware or percentile-clipped scales are a
 *   model-prep concern, and the codec takes scales as an argument
 *   ([quantizeWithScales]) precisely so a caller can supply better ones.
 */
object KvQuantPool {

    /**
     * The per-group scales of [pool], by the symmetric-absmax rule.
     *
     * @param pool row-major `[numBlocks, blockSize, numKvHeads, headDim]`.
     * @return `[numKvHeads]` for [KvScaleStrategy.PER_HEAD], `[1]` for
     *   [KvScaleStrategy.PER_TENSOR].
     */
    fun scalesOf(
        pool: FloatArray,
        numKvHeads: Int,
        headDim: Int,
        config: KvQuantConfig,
    ): FloatArray {
        requireIntegerCoded(config, "KvQuantPool.scalesOf")
        checkPool(pool, numKvHeads, headDim, "KvQuantPool.scalesOf")
        val groups = groupCount(numKvHeads, config.scaleStrategy)
        val absmax = FloatArray(groups)
        forEachElement(pool.size, numKvHeads, headDim) { i, head ->
            val g = if (config.scaleStrategy == KvScaleStrategy.PER_HEAD) head else 0
            val a = abs(pool[i])
            require(a.isFinite()) {
                "KvQuantPool.scalesOf: pool[$i] = ${pool[i]} is not finite — a KV page pool " +
                    "with a NaN or an Inf in it has no absmax, and quantizing it would hide " +
                    "the bug behind a scale"
            }
            if (a > absmax[g]) absmax[g] = a
        }
        val qmax = config.dtype.codeMax.toFloat()
        return FloatArray(groups) { g ->
            // An all-zero group has no scale; 1 keeps the round trip exact
            // (every code is 0, every value is 0) where a 0 scale would read
            // like a bug downstream and make any later rescale divide by zero.
            if (absmax[g] == 0f) 1f else absmax[g] / qmax
        }
    }

    /** [scalesOf] followed by [quantizeWithScales] — the usual entry point. */
    fun quantize(
        pool: FloatArray,
        numKvHeads: Int,
        headDim: Int,
        config: KvQuantConfig,
    ): QuantizedPool {
        val scales = scalesOf(pool, numKvHeads, headDim, config)
        return QuantizedPool(quantizeWithScales(pool, scales, numKvHeads, headDim, config), scales, config)
    }

    /**
     * Quantize [pool] against scales the caller already has — a manifest's, a
     * calibration pass's, or a previous step's, so a decode loop never
     * re-derives a scale mid-sequence and silently changes the meaning of
     * codes already in the pool.
     */
    fun quantizeWithScales(
        pool: FloatArray,
        scales: FloatArray,
        numKvHeads: Int,
        headDim: Int,
        config: KvQuantConfig,
    ): IntArray {
        requireIntegerCoded(config, "KvQuantPool.quantizeWithScales")
        checkPool(pool, numKvHeads, headDim, "KvQuantPool.quantizeWithScales")
        checkScales(scales, numKvHeads, config, "KvQuantPool.quantizeWithScales")
        val qmax = config.dtype.codeMax
        val codes = IntArray(pool.size)
        forEachElement(pool.size, numKvHeads, headDim) { i, head ->
            val s = scales[if (scales.size == 1) 0 else head]
            val q = roundHalfAway(pool[i] / s)
            codes[i] = if (q > qmax) qmax else if (q < -qmax) -qmax else q
        }
        return codes
    }

    /**
     * The host twin of [io.tlaloc.ir.OpKind.DEQUANTIZE_KV]:
     * `out[i] = codes[i] * scales[head(i)]`. The in-graph op and this function
     * are pinned to agree.
     */
    fun dequantize(
        codes: IntArray,
        scales: FloatArray,
        numKvHeads: Int,
        headDim: Int,
        config: KvQuantConfig,
    ): FloatArray {
        requireIntegerCoded(config, "KvQuantPool.dequantize")
        checkScales(scales, numKvHeads, config, "KvQuantPool.dequantize")
        val qmax = config.dtype.codeMax
        val out = FloatArray(codes.size)
        forEachElement(codes.size, numKvHeads, headDim) { i, head ->
            require(codes[i] in -qmax..qmax) {
                "KvQuantPool.dequantize: code[$i] = ${codes[i]} is outside the " +
                    "${config.dtype.nameTag} code range [${-qmax}, $qmax] — a pool that " +
                    "carries out-of-range codes was not produced by this contract"
            }
            out[i] = codes[i] * scales[if (scales.size == 1) 0 else head]
        }
        return out
    }

    /**
     * The hand-derived per-element bound `scale/2`, at the widest scale in
     * [scales]. A test asserts the MEASURED round-trip error against this,
     * which is the derivation in §"The numerics" recomputed — not a tolerance
     * someone tuned until the test passed.
     */
    fun maxRoundTripError(scales: FloatArray): Float = (scales.max()) / 2f

    /**
     * Ties away from zero, spelled out (see the file comment): `0.5 -> 1`,
     * `-0.5 -> -1`, `-1.5 -> -2`.
     */
    fun roundHalfAway(v: Float): Int {
        val a = floor(abs(v) + 0.5f).toInt()
        return if (v < 0f) -a else a
    }

    private fun groupCount(numKvHeads: Int, strategy: KvScaleStrategy): Int =
        if (strategy == KvScaleStrategy.PER_HEAD) numKvHeads else 1

    /** The number of scale entries a pool of [numKvHeads] heads needs under [config]. */
    fun scaleCount(numKvHeads: Int, config: KvQuantConfig): Int =
        groupCount(numKvHeads, config.scaleStrategy)

    private inline fun forEachElement(n: Int, numKvHeads: Int, headDim: Int, body: (Int, Int) -> Unit) {
        val headStride = headDim
        for (i in 0 until n) body(i, (i / headStride) % numKvHeads)
    }

    private fun checkPool(pool: FloatArray, numKvHeads: Int, headDim: Int, layer: String) {
        require(numKvHeads >= 1 && headDim >= 1) {
            "$layer: numKvHeads/headDim must be >= 1, got $numKvHeads/$headDim"
        }
        require(pool.isNotEmpty() && pool.size % (numKvHeads * headDim) == 0) {
            "$layer: pool of ${pool.size} elements is not a whole number of " +
                "[.., numKvHeads=$numKvHeads, headDim=$headDim] rows"
        }
    }

    private fun checkScales(scales: FloatArray, numKvHeads: Int, config: KvQuantConfig, layer: String) {
        val want = scaleCount(numKvHeads, config)
        require(scales.size == want) {
            "$layer: ${config.scaleStrategy} over $numKvHeads kv heads needs $want scale(s), " +
                "got ${scales.size}"
        }
        for ((g, s) in scales.withIndex()) {
            require(s.isFinite() && s > 0f) {
                "$layer: scale[$g] = $s must be finite and strictly positive (an all-zero " +
                    "group takes scale 1 by this contract, never 0)"
            }
        }
    }

    private fun requireIntegerCoded(config: KvQuantConfig, layer: String) {
        require(config.dtype.isIntegerCoded) {
            "$layer: ${config.dtype.nameTag} is REFUSED BY NAME, not missing — this codec is " +
                "the symmetric-absmax INTEGER-CODE contract (a code is a small integer and the " +
                "dequantization is one multiply), and ${config.dtype.nameTag} is a float format " +
                "whose code is a bit pattern with its own exponent. It wants a narrow DType the " +
                "way bf16 got one in §0.4.455, not an integer code path that would round twice. " +
                "Integer-coded dtypes: " +
                KvQuantDtype.entries.filter { it.isIntegerCoded }.joinToString(", ") { it.nameTag }
        }
    }
}

/**
 * A quantized page pool: the codes, the scales they are read against, and the
 * config that produced them, kept together so the three cannot drift apart on
 * their way to a graph input or a manifest.
 */
data class QuantizedPool(
    val codes: IntArray,
    val scales: FloatArray,
    val config: KvQuantConfig,
) {
    /** The host-side dequantization, for oracles and for a CPU fallback. */
    fun dequantize(numKvHeads: Int, headDim: Int): FloatArray =
        KvQuantPool.dequantize(codes, scales, numKvHeads, headDim, config)

    override fun equals(other: Any?): Boolean =
        other is QuantizedPool && codes.contentEquals(other.codes) &&
            scales.contentEquals(other.scales) && config == other.config

    override fun hashCode(): Int =
        (codes.contentHashCode() * 31 + scales.contentHashCode()) * 31 + config.hashCode()
}
