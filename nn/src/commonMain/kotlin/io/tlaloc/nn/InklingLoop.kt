/**
 * Inkling 2 with looped middle layers (`inkling2_loop`, Thinking Machines' Inkling-2-Mini):
 * the text model's forward over a whole sequence, as a trace over `:autograd` ops.
 *
 * Written from transformers 5.24.0 `modeling_inkling2.py` and `modular_inkling2_loop.py`, and
 * checked against that code on a tiny checkpoint (`InklingLoopTest`, the oracle is
 * `harness/python/inkling_modeling_fixture.py`). Weights are addressed by the checkpoint's own
 * names (`model.llm.*`), so the released files load without renaming.
 *
 * What a layer does, in the modeling code's order:
 *
 *  - attention residuals: the input of the attention (three inputs, query/key/value, in the
 *    full-attention layers) and of the MLP is a softmax mix of the earlier blocks' outputs and the
 *    running state, all RMS-normalized, with a learned query per input;
 *  - attention without RoPE: q is an L2-clipped scaled projection, k is a short causal
 *    convolution of its projection then L2-clipped, v the same then RMS-normalized without gain.
 *    The full-attention layers' k/v convolutions add per-token taps projected from their input.
 *    Positions enter through a relative bias (per head, a mix of a learned bank of bias-by-distance
 *    profiles, zero past the extent) and, in the full layers, a log scale of q and of the bias
 *    states past `log_scaling_n_floor`. Each head has a sink logit with a zero value. The output
 *    is gated by a sigmoid of the layer input;
 *  - short causal convolutions after the attention and after the MLP, with a residual;
 *  - a sigmoid-routed mixture of experts (top-k on score + bias, weights renormalized and
 *    scaled), with a softcapped SwiGLU;
 *  - the looped layers run `num_passes` times; passes after the first restart from the state
 *    before the loop and mix in messages read from the previous pass through gates.
 *
 * Batch 1, `[T, H]` activations in f32. Weights may be bf16: a projection then rounds its input to
 * bf16 and multiplies in bf16, as the bf16 modeling code does, and every other weight is widened
 * to f32 where it is used. The routed experts are evaluated densely (every expert on every
 * token, weighted by the routing weights, zero for unselected experts), which is the same sum.
 */
package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.bmm
import io.tlaloc.autograd.cast
import io.tlaloc.autograd.broadcastTo
import io.tlaloc.autograd.concat
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.div
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.log
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.neg
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.raggedDot
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.sigmoid
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.softmax
import io.tlaloc.autograd.sqrt
import io.tlaloc.autograd.step
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.takeAlong
import io.tlaloc.autograd.tanh
import io.tlaloc.autograd.times
import io.tlaloc.autograd.topK
import io.tlaloc.autograd.transpose
import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/** The text config of an `inkling2_loop` checkpoint (`config.json` `text_config`, or the file itself). */
data class InklingConfig(
    val vocabSize: Int,
    val unpaddedVocabSize: Int,
    val hidden: Int,
    val layers: Int,
    /** True for a sliding-window layer, false for full attention. */
    val sliding: List<Boolean>,
    val heads: Int,
    val kvHeads: Int,
    val headDim: Int,
    val swaHeads: Int,
    val swaKvHeads: Int,
    val swaHeadDim: Int,
    val window: Int,
    val dRel: Int,
    val relExtent: Int,
    val logScalingFloor: Int,
    val logScalingAlpha: Double,
    val eps: Double,
    val convKernel: Int,
    val experts: Int,
    val expertsPerToken: Int,
    val expertDim: Int,
    val routeScale: Double,
    val qkClip: Double,
    val relClip: Double,
    val layersPerBlock: Int,
    val gateSoftcap: Double,
    val upSoftcap: Double,
    val logitSoftcap: Double,
    val loopStart: Int,
    val loopEnd: Int,
    val passes: Int,
    val gateRank: Int,
) {
    init {
        require(sliding.size == layers) { "inkling2_loop: ${sliding.size} layer types for $layers layers" }
        require(loopStart > 0 && loopStart % layersPerBlock == 0 && loopEnd % layersPerBlock == 0 && loopEnd in loopStart..layers) {
            "inkling2_loop: loop layers $loopStart..$loopEnd must start after layer 0 on block boundaries of $layersPerBlock"
        }
        require(passes >= 1) { "inkling2_loop: $passes passes" }
    }

    val loopLayers: Int get() = loopEnd - loopStart

    fun isBoundary(layer: Int): Boolean = layer % layersPerBlock == 0

    fun isLoop(layer: Int): Boolean = layer in loopStart until loopEnd

    companion object {
        /** Reads the fields from a parsed text config ([get] returns a JSON value as Kotlin: Number, String, List, Map, null). */
        fun fromJson(get: (String) -> Any?): InklingConfig {
            fun int(k: String, default: Int? = null): Int = (get(k) as? Number)?.toInt() ?: default ?: error("inkling2_loop config: no $k")
            fun dbl(k: String, default: Double? = null): Double = (get(k) as? Number)?.toDouble() ?: default ?: error("inkling2_loop config: no $k")
            val layers = int("num_hidden_layers")
            val perBlock = int("n_layers_per_attn_res_block", 6)

            @Suppress("UNCHECKED_CAST")
            val sliding: List<Boolean> = when {
                get("local_layer_ids") != null -> (get("local_layer_ids") as List<Number>).map { it.toInt() }.toSet()
                    .let { local -> List(layers) { it in local } }
                get("layer_types") != null -> (get("layer_types") as List<String>).map { it == "hybrid_sliding" }
                else -> List(layers) { (it + 1) % 6 != 0 }
            }

            @Suppress("UNCHECKED_CAST")
            val loop = get("loop_config") as? Map<String, Any?>
            fun loopInt(k: String, flat: String, default: Int) = (loop?.get(k) as? Number)?.toInt() ?: (get(flat) as? Number)?.toInt() ?: default
            val heads = int("num_attention_heads")
            val kv = int("num_key_value_heads")
            val hd = int("head_dim")
            return InklingConfig(
                vocabSize = int("vocab_size"),
                unpaddedVocabSize = int("unpadded_vocab_size", int("vocab_size")),
                hidden = int("hidden_size"),
                layers = layers,
                sliding = sliding,
                heads = heads, kvHeads = kv, headDim = hd,
                swaHeads = int("swa_num_attention_heads", heads),
                swaKvHeads = int("swa_num_key_value_heads", kv),
                swaHeadDim = int("swa_head_dim", hd),
                window = int("sliding_window_size", 1024),
                dRel = int("d_rel", 16),
                relExtent = int("rel_extent", 1024),
                logScalingFloor = int("log_scaling_n_floor", 128000),
                logScalingAlpha = dbl("log_scaling_alpha", 0.1),
                eps = dbl("rms_norm_eps", 1e-6),
                convKernel = (get("conv_kernel_size") ?: get("sconv_kernel_size") as? Number)?.let { (it as Number).toInt() } ?: 4,
                experts = int("n_routed_experts"),
                expertsPerToken = int("num_experts_per_tok"),
                expertDim = (get("moe_intermediate_size") ?: get("intermediate_size") as? Number)?.let { (it as Number).toInt() } ?: error("inkling2_loop config: no moe_intermediate_size"),
                routeScale = dbl("route_scale", 1.0),
                qkClip = dbl("qk_l2_clip_threshold"),
                relClip = dbl("rel_l2_clip_threshold"),
                layersPerBlock = perBlock,
                gateSoftcap = dbl("mlp_gate_softcap"),
                upSoftcap = dbl("mlp_up_softcap"),
                logitSoftcap = dbl("final_logit_softcapping"),
                loopStart = loopInt("core_start", "loop_layer_index_start", 6),
                loopEnd = loopInt("core_end", "loop_layer_index_end", 18),
                passes = loopInt("num_passes", "loop_num_passes", 3),
                gateRank = loopInt("gate_rank", "loop_gate_rank", 64),
            )
        }
    }
}

/**
 * The checkpoint names of the text model. Linear weights are `[out, in]` as stored; the forward
 * reads them through [InklingLoop.Weights], which a loader fills (transposed or not, see
 * [InklingLoop.Weights.linear]).
 */
object InklingNames {
    const val P = "model.llm."
    fun layer(l: Int, rest: String) = "${P}layers.$l.$rest"
    const val EMBED = "${P}embed.weight"
    const val EMBED_NORM = "${P}embed_norm.weight"
    const val NORM = "${P}norm.weight"
    const val UNEMBED = "${P}unembed.weight"
    const val OUTPUT_QUERY = "${P}attn_res.output_query"
    const val MESSAGE_QUERIES = "${P}middle_cycle_cross_pass_queries"
    const val MESSAGE_KV_QUERIES = "${P}middle_cycle_cross_pass_kv_queries"
    /** The attention-input query of layer [l] (layer 0 has none; stored at index l - 1). */
    fun attnQuery(l: Int) = "${P}attn_res.attn_queries.${l - 1}"
    fun mlpQuery(l: Int) = "${P}attn_res.mlp_queries.$l"
}

object InklingLoop {

    /**
     * The weights a forward reads, by checkpoint name. [tensor] returns the stored tensor as a
     * tracer; [linear] returns an `nn.Linear` weight as `[in, out]` (a loader may hand over the
     * transpose it made once); [experts] the routed experts of a layer as `[E, H, 2I]` gate/up
     * (in each expert the gate columns, then the up columns: the checkpoint's interleaved rows
     * undone) and `[E, I, H]` down.
     */
    interface Weights {
        fun tensor(name: String): Tracer<Shape>
        fun linear(name: String): Tracer<Shape>
        fun experts(layer: Int): Pair<Tracer<Shape>, Tracer<Shape>>
    }

    /**
     * The logits `[T, unpadded vocab]` of [ids] (`[T]` I32 tracer) at every position, after the
     * final softcap. Positions are `0 until T`.
     */
    fun forward(c: InklingConfig, ids: Tracer<Shape>, w: Weights, routed: Boolean = false): Tracer<Shape> =
        run(c, ids, w, Ops(c, ids.dims[0], ids, routed = routed))

    private fun run(c: InklingConfig, ids: Tracer<Shape>, w: Weights, ops: Ops): Tracer<Shape> {
        var h = ops.rmsNorm(ops.f32(w.tensor(InklingNames.EMBED).embedding<Shape>(ids)), ops.t(w, InklingNames.EMBED_NORM))
        val blocks = ArrayList<Tracer<Shape>>()

        fun runLayer(l: Int, x: Tracer<Shape>, blk: MutableList<Tracer<Shape>>, messages: List<Tracer<Shape>>?): Tracer<Shape> {
            val out = ops.layer(l, x, blk, messages, w)
            if (c.isBoundary(l)) blk += x
            return out
        }

        for (l in 0 until c.loopStart) h = runLayer(l, h, blocks, null)
        val preH = h
        val preBlocks = blocks.toList()
        for (l in c.loopStart until c.loopEnd) h = runLayer(l, h, blocks, null)
        var cur: MutableList<Tracer<Shape>> = blocks
        for (pass in 1 until c.passes) {
            // The blocks the previous pass appended after its first (which is the pre-loop state).
            val prevBlocks = cur.drop(preBlocks.size + 1)
            val prevH = h
            h = preH
            cur = preBlocks.toMutableList()
            for (l in c.loopStart until c.loopEnd) {
                val msgs = ops.messages(prevBlocks, prevH, l - c.loopStart, w)
                h = runLayer(l, h, cur, msgs)
            }
        }
        for (l in c.loopEnd until c.layers) h = runLayer(l, h, cur, null)
        h = ops.blockMix(cur, h, ops.t(w, InklingNames.OUTPUT_QUERY), 1)[0]
        h = ops.rmsNorm(h, ops.t(w, InklingNames.NORM))
        val unembed = w.tensor(InklingNames.UNEMBED).let { if (c.unpaddedVocabSize < it.dims[0]) it.slice<Shape>(0, c.unpaddedVocabSize, 0) else it }
        val logits = ops.mmW(h, unembed.transpose<Shape>(1, 0))
        return ops.softcap(logits, c.logitSoftcap)
    }

    /**
     * A decode step's inputs: the position (an f32 `[1]`), the cache length, and the state of every
     * layer application in execution order ([stateDims]). [layer] calls [take] once per application
     * and appends the new state to [out] in the same order.
     */
    private class Step(val pos: Tracer<Shape>, val ctx: Int, val state: List<Tracer<Shape>>) {
        private var next = 0
        val out = ArrayList<Tracer<Shape>>()
        fun take(): List<Tracer<Shape>> = state.subList(next * STATE_PER_APPLICATION, (next + 1) * STATE_PER_APPLICATION).also { next++ }
        val done: Boolean get() = next * STATE_PER_APPLICATION == state.size
    }

    /** Per layer application: K and V caches, then the K, V, attention-output and MLP-output convolution histories. */
    const val STATE_PER_APPLICATION = 6

    /** The layers in execution order: before the loop, the loop [InklingConfig.passes] times, after it. */
    fun applications(c: InklingConfig): List<Int> =
        (0 until c.loopStart) + List(c.passes) { c.loopStart until c.loopEnd }.flatten() + (c.loopEnd until c.layers)

    /** The dims of a decode step's state at cache length [ctx], in [decodeStep]'s order. */
    fun stateDims(c: InklingConfig, ctx: Int): List<IntArray> = applications(c).flatMap { l ->
        val kv = if (c.sliding[l]) c.swaKvHeads * c.swaHeadDim else c.kvHeads * c.headDim
        val k1 = c.convKernel - 1
        listOf(intArrayOf(ctx, kv), intArrayOf(ctx, kv), intArrayOf(k1, kv), intArrayOf(k1, kv), intArrayOf(k1, c.hidden), intArrayOf(k1, c.hidden))
    }

    /**
     * One token through the model with caches of [ctx] positions: [id] `[1]` I32 at [pos] (`[1]`
     * f32, below [ctx]) with [state] (zeros before the first token, see [stateDims]). Returns the
     * logits `[1, V]`, then the new state. Equal to row `pos` of [forward] over the tokens so far.
     * The routed experts are gathered (only the chosen ones are read).
     */
    fun decodeStep(c: InklingConfig, ctx: Int, id: Tracer<Shape>, pos: Tracer<Shape>, state: List<Tracer<Shape>>, w: Weights): List<Tracer<Shape>> =
        chunk(c, ctx, id, pos, state, w)

    /**
     * [ids] (`[T]` I32) at positions `pos until pos + T` ([pos] `[1]` f32; `pos + T` ≤ [ctx]),
     * continuing [state] (what [prefill], [decodeStep] or an earlier chunk returned; zeros at
     * position 0). Returns the logits `[T, V]`, then the new state. Equal to rows `pos until pos + T`
     * of [forward] over all the tokens so far, so a prompt runs as consecutive chunks of any sizes,
     * and a later turn continues where the last one ended. One token is [decodeStep]. The routed
     * experts are gathered for one token; for more they run as ragged dots (each expert on its own
     * tokens) when [routed], otherwise densely (every expert on every token, for a backend without
     * `ragged_dot`, such as XLA:CPU).
     */
    fun chunk(
        c: InklingConfig,
        ctx: Int,
        ids: Tracer<Shape>,
        pos: Tracer<Shape>,
        state: List<Tracer<Shape>>,
        w: Weights,
        routed: Boolean = true,
    ): List<Tracer<Shape>> {
        require(ids.dims[0] <= ctx) { "chunk: ${ids.dims[0]} tokens for caches of $ctx" }
        val step = Step(pos, ctx, state)
        val logits = run(c, ids, w, Ops(c, ids.dims[0], ids, step, routed = routed))
        check(step.done) { "chunk: ${state.size} state tensors, ${step.out.size} used" }
        return listOf(logits) + step.out
    }

    /**
     * The prompt [ids] (`[T]`, positions `0 until T`, T ≤ [ctx]) in one forward: its logits
     * `[T, V]`, then the state [decodeStep] continues from at position T (caches of [ctx] rows).
     */
    fun prefill(c: InklingConfig, ctx: Int, ids: Tracer<Shape>, w: Weights): List<Tracer<Shape>> {
        require(ids.dims[0] <= ctx) { "prefill: ${ids.dims[0]} tokens for caches of $ctx" }
        val ops = Ops(c, ids.dims[0], ids, collectCtx = ctx)
        val logits = run(c, ids, w, ops)
        return listOf(logits) + ops.collected
    }

    private class Ops(
        val c: InklingConfig, val t: Int, val anchor: Tracer<*>, val step: Step? = null, val collectCtx: Int? = null,
        /** The experts on their own tokens only (ragged dots) instead of every expert on every token. */
        val routed: Boolean = false,
    ) {
        /** The state a [prefill] hands to the decode step, in [stateDims] order. */
        val collected = ArrayList<Tracer<Shape>>()

        /** [x] `[T, C]` as `[collectCtx, C]` cache rows, zeros after row T. */
        fun cacheRows(x: Tracer<Shape>): Tracer<Shape> {
            val ctx = collectCtx!!
            return if (ctx == t) x else concat(listOf(x, const(0f, intArrayOf(ctx - t, x.dims[1]))), 0)
        }

        @Suppress("UNCHECKED_CAST")
        fun mm(a: Tracer<*>, b: Tracer<*>): Tracer<Shape> =
            ((a as Tracer<Rank2<Sym, Sym>>) matmul (b as Tracer<Rank2<Sym, Sym>>)) as Tracer<Shape>

        @Suppress("UNCHECKED_CAST")
        fun bmm(a: Tracer<*>, b: Tracer<*>): Tracer<Shape> =
            ((a as Tracer<Rank3<Sym, Sym, Sym>>) bmm (b as Tracer<Rank3<Sym, Sym, Sym>>)) as Tracer<Shape>

        /** [x] in f32 (a bf16 tensor widened). */
        fun f32(x: Tracer<Shape>): Tracer<Shape> = if (x.dtype == BF16) x.cast(F32) else x

        /** The weight [name] in f32. */
        fun t(w: Weights, name: String): Tracer<Shape> = f32(w.tensor(name))

        /** `x [.., K] @ W [K, N]`, in bf16 when W is: x rounded to bf16, the product widened back. */
        fun mmW(x: Tracer<Shape>, wt: Tracer<Shape>): Tracer<Shape> =
            if (wt.dtype == BF16) f32(mm(x.cast(BF16), wt)) else mm(x, wt)

        fun bmmW(x: Tracer<Shape>, wt: Tracer<Shape>): Tracer<Shape> =
            if (wt.dtype == BF16) f32(bmm(x.cast(BF16), wt)) else bmm(x, wt)

        /** [concat], or the one part itself (the emitter takes two operands or more). */
        fun cat(parts: List<Tracer<Shape>>, axis: Int): Tracer<Shape> = if (parts.size == 1) parts[0] else concat(parts, axis)

        fun const(v: Float, dims: IntArray): Tracer<Shape> = anchor.constant<Shape>(FloatArray(dims.fold(1) { a, b -> a * b }) { v }, dims)

        fun bc(x: Tracer<*>, dims: IntArray): Tracer<Shape> = x.broadcastTo(dims)

        /** [x] `[…, D]` times the row [r] `[D]`. */
        fun rowTimes(x: Tracer<Shape>, r: Tracer<*>): Tracer<Shape> = x * bc(r, x.dims)

        fun meanKeep(x: Tracer<Shape>): Tracer<Shape> =
            x.mean<Shape>(intArrayOf(x.rank - 1)).reshape(x.dims.copyOf().also { it[x.rank - 1] = 1 })

        fun sumKeep(x: Tracer<Shape>): Tracer<Shape> =
            x.sum<Shape>(intArrayOf(x.rank - 1)).reshape(x.dims.copyOf().also { it[x.rank - 1] = 1 })

        /** RMS over the last axis, without a gain when [gain] is null. */
        fun rmsNorm(x: Tracer<Shape>, gain: Tracer<Shape>?, eps: Double = c.eps): Tracer<Shape> {
            val ms = meanKeep(x * x)
            val inv = bc((ms + eps.toFloat()).sqrt(), x.dims)
            val n = x / inv
            return if (gain == null) n else rowTimes(n, gain)
        }

        fun softcap(x: Tracer<Shape>, cap: Double): Tracer<Shape> = (x / cap.toFloat()).tanh() * cap.toFloat()

        /** `x · min(threshold / (‖x‖ + 1e-6), 1)` over the last axis. */
        fun l2Clip(x: Tracer<Shape>, threshold: Double): Tracer<Shape> {
            val norm = sumKeep(x * x).sqrt()
            val ratio = const(threshold.toFloat(), norm.dims) / (norm + 1e-6f)
            val clipped = const(1f, norm.dims) - (const(1f, norm.dims) - ratio).relu()
            return x * bc(clipped, x.dims)
        }

        /** `1 + α · ln(max((p + 1) / floor, 1))` per position, as `[T, 1, 1]` broadcast to [dims]. */
        fun logScale(x: Tracer<Shape>): Tracer<Shape> {
            if (step != null) {
                val ratio = (posRows + 1f) / c.logScalingFloor.toFloat()
                val tau = ((ratio - 1f).relu() + 1f).log() * c.logScalingAlpha.toFloat() + 1f
                return x * bc(tau.reshape<Shape>(intArrayOf(t, 1, 1)), x.dims)
            }
            val tau = FloatArray(t) { p -> (1.0 + c.logScalingAlpha * ln(max((p + 1).toDouble() / c.logScalingFloor, 1.0))).toFloat() }
            return x * bc(anchor.constant<Shape>(tau, intArrayOf(t, 1, 1)), x.dims)
        }

        // ---- chunk position helpers (step != null; query rows i = 0 until t at positions pos + i,
        // cache rows j = 0 until ctx)

        /** The rows' positions, `[t]`. */
        private val posRows: Tracer<Shape> by lazy {
            val s = step!!
            val p = bc(s.pos, intArrayOf(t))
            if (t == 1) p else p + anchor.constant<Shape>(FloatArray(t) { it.toFloat() }, intArrayOf(t))
        }

        /** `pos + i − j`, `[t, ctx]`. */
        private val dist: Tracer<Shape> by lazy {
            val ctx = step!!.ctx
            bc(posRows.reshape<Shape>(intArrayOf(t, 1)), intArrayOf(t, ctx)) -
                bc(anchor.constant<Shape>(FloatArray(ctx) { it.toFloat() }, intArrayOf(1, ctx)), intArrayOf(t, ctx))
        }

        /** 1 where `pos + i − j == d`, `[t, extent, ctx]`. */
        private val selectors = HashMap<Int, Tracer<Shape>>()
        fun selector(extent: Int): Tracer<Shape> = selectors.getOrPut(extent) {
            val ctx = step!!.ctx
            val dims = intArrayOf(t, extent, ctx)
            val diff = bc(dist.reshape<Shape>(intArrayOf(t, 1, ctx)), dims) -
                bc(anchor.constant<Shape>(FloatArray(extent) { it.toFloat() }, intArrayOf(1, extent, 1)), dims)
            const(1f, dims) - (diff * diff - 0.25f).step()
        }

        /** 1 at cache row `pos + i` in column i, `[ctx, t]`. */
        private val atPos: Tracer<Shape> by lazy {
            (const(1f, intArrayOf(t, step!!.ctx)) - (dist * dist - 0.25f).step()).transpose<Shape>(1, 0)
        }

        /** 1 at the cache rows the chunk writes, `[ctx, 1]`. */
        private val written: Tracer<Shape> by lazy {
            if (t == 1) atPos else mm(atPos, const(1f, intArrayOf(t, 1)))
        }

        /** 0 where row j is attended by query row i (`0 ≤ pos + i − j`, and `< window` when sliding), −1e30 elsewhere; `[1, t, ctx]`. */
        private val masks = HashMap<Boolean, Tracer<Shape>>()
        fun stepMask(sliding: Boolean): Tracer<Shape> = masks.getOrPut(sliding) {
            val dims = intArrayOf(t, step!!.ctx)
            var valid = (dist + 0.5f).step()
            if (sliding) valid = valid * (const(c.window - 0.5f, dims) - dist).step()
            ((const(1f, dims) - valid) * -1e30f).reshape(intArrayOf(1, t, step.ctx))
        }

        /** [cache] `[ctx, C]` with rows `pos until pos + t` replaced by [rows] `[t, C]`. */
        fun write(cache: Tracer<Shape>, rows: Tracer<Shape>): Tracer<Shape> {
            val keep = const(1f, cache.dims) - bc(written, cache.dims)
            val put = if (t == 1) bc(atPos, cache.dims) * bc(rows, cache.dims) else mm(atPos, rows)
            return cache * keep + put
        }

        /** The rows a convolution sees: its history (decode) or K − 1 zero rows, then [x]. Also the history after [x]: its last K − 1 rows. */
        fun convInput(x: Tracer<Shape>, history: Tracer<Shape>?): Pair<Tracer<Shape>, Tracer<Shape>> {
            val k = c.convKernel
            val padded = concat<Shape>(listOf(history ?: const(0f, intArrayOf(k - 1, x.dims[1])), x), 0)
            return padded to padded.slice<Shape>(t, t + k - 1, 0)
        }

        /** `x [T, in] @ W` for the `nn.Linear` weight [name]. */
        fun linear(x: Tracer<Shape>, w: Weights, name: String): Tracer<Shape> = mmW(x, w.linear(name))

        /** Short causal convolution of `x [T, C]` with the `[C, 1, K]` weight (`w[c, K-1]` is the current token), plus the input, over √(K + 1). */
        fun shortConv(x: Tracer<Shape>, weight: Tracer<Shape>, history: Tracer<Shape>? = null): Pair<Tracer<Shape>, Tracer<Shape>> {
            val k = c.convKernel
            val ch = x.dims[1]
            val wk = weight.reshape<Shape>(intArrayOf(ch, k)).transpose<Shape>(1, 0) // [K, C]
            val (padded, next) = convInput(x, history)
            var acc: Tracer<Shape>? = null
            for (j in 0 until k) {
                val term = rowTimes(padded.slice(j, j + t, 0), wk.slice<Shape>(j, j + 1, 0).reshape<Shape>(intArrayOf(ch)))
                acc = acc?.let { it + term } ?: term
            }
            return (acc!! + x) / sqrt((k + 1).toDouble()).toFloat() to next
        }

        /**
         * The full layers' key/value convolution: taps `static[lag, c]` plus per-token taps
         * projected from [dyn] `[T, H]` (one set per KV head), `lag` 0 the current token; plus the
         * input, over √(2K + 1).
         */
        fun dynamicConv(x: Tracer<Shape>, dyn: Tracer<Shape>, static: Tracer<Shape>, proj: String, kvHeads: Int, w: Weights, history: Tracer<Shape>? = null): Pair<Tracer<Shape>, Tracer<Shape>> {
            val k = c.convKernel
            val ch = x.dims[1]
            val per = ch / kvHeads
            val taps = linear(dyn, w, proj).reshape<Shape>(intArrayOf(t, kvHeads, k)) // [T, heads, K]
            val (padded, next) = convInput(x, history)
            var acc: Tracer<Shape>? = null
            for (lag in 0 until k) {
                val shifted = padded.slice<Shape>(k - 1 - lag, k - 1 - lag + t, 0)
                val s = rowTimes(shifted, static.slice<Shape>(lag, lag + 1, 0).reshape<Shape>(intArrayOf(ch)))
                val d = taps.slice<Shape>(lag, lag + 1, 2) // [T, heads, 1]
                val dFull = bc(d, intArrayOf(t, kvHeads, per)).reshape<Shape>(intArrayOf(t, ch))
                val term = s + shifted * dFull
                acc = acc?.let { it + term } ?: term
            }
            return (acc!! + x) / sqrt((2 * k + 1).toDouble()).toFloat() to next
        }

        /**
         * The attention residual: `softmax_n(q · norm(s_n) / √H) · norm(s_n)` over the blocks and
         * [partial], for each of the [nq] rows of [query] (`[H]` or `[nq, H]`). Returns nq `[T, H]`.
         */
        fun blockMix(blocks: List<Tracer<Shape>>, partial: Tracer<Shape>, query: Tracer<Shape>, nq: Int): List<Tracer<Shape>> {
            val hd = c.hidden
            val all = blocks + partial
            val n = all.size
            val stacked = rmsNorm(cat(all.map { it.reshape<Shape>(intArrayOf(t, 1, hd)) }, 1), null, 1e-6) // [T, n, H]
            val q = query.reshape<Shape>(intArrayOf(nq, hd))
            val logits = mm(stacked.reshape<Shape>(intArrayOf(t * n, hd)), q.transpose<Shape>(1, 0))
                .reshape<Shape>(intArrayOf(t, n, nq)) * (1.0 / sqrt(hd.toDouble())).toFloat()
            val p = logits.softmax(1).transpose<Shape>(0, 2, 1) // [T, nq, n]
            val out = bmm(p, stacked) // [T, nq, H]
            return List(nq) { i -> out.slice<Shape>(i, i + 1, 1).reshape(intArrayOf(t, hd)) }
        }

        /** The four messages (query, key, value, MLP) loop layer [i] reads from the previous pass. */
        fun messages(prevBlocks: List<Tracer<Shape>>, prevH: Tracer<Shape>, i: Int, w: Weights): List<Tracer<Shape>> {
            val qm = t(w, InklingNames.MESSAGE_QUERIES) // [2L, H]: query, mlp per loop layer
            val kvm = t(w, InklingNames.MESSAGE_KV_QUERIES) // [2L, H]: key, value per loop layer
            val query = concat<Shape>(
                listOf(qm.slice<Shape>(2 * i, 2 * i + 1, 0), kvm.slice<Shape>(2 * i, 2 * i + 2, 0), qm.slice<Shape>(2 * i + 1, 2 * i + 2, 0)), 0,
            )
            return blockMix(prevBlocks, prevH, query, 4)
        }

        /** `(1 − tanh(f)) · h + tanh(g) · m` with `f, g` from a rank-[gateRank] projection of both, normalized. */
        fun messageGate(h: Tracer<Shape>, m: Tracer<Shape>, l: Int, which: String, w: Weights): Tracer<Shape> {
            val base = "controllers.$which."
            val z = concat<Shape>(listOf(rmsNorm(h, null), rmsNorm(m, null)), 1)
            val down = linear(z, w, InklingNames.layer(l, base + "down_weight"))
            val forget = linear(down, w, InklingNames.layer(l, base + "forget_weight")).tanh()
            val gain = linear(down, w, InklingNames.layer(l, base + "gain_weight")).tanh()
            return (const(1f, forget.dims) - forget) * h + gain * m
        }

        /** The attention output; in a decode step also the new K/V caches and K/V convolution histories ([st]: the application's state). */
        fun attention(l: Int, xq: Tracer<Shape>, xk: Tracer<Shape>, xv: Tracer<Shape>, w: Weights, st: List<Tracer<Shape>>? = null): Pair<Tracer<Shape>, List<Tracer<Shape>>> {
            val sliding = c.sliding[l]
            val nh = if (sliding) c.swaHeads else c.heads
            val nkv = if (sliding) c.swaKvHeads else c.kvHeads
            val hd = if (sliding) c.swaHeadDim else c.headDim
            val extent = if (sliding) c.window else c.relExtent
            fun a(rest: String) = InklingNames.layer(l, "attn.$rest")

            // q: [T, nh, hd], scaled, clipped (and log-scaled in the full layers)
            var q = rowTimes(linear(xq, w, a("wq_du.weight")).reshape(intArrayOf(t, nh, hd)), t(w, a("q_l2_clip.scale")))
            q = l2Clip(q, c.qkClip)
            if (!sliding) q = logScale(q)
            // k, v: [T, nkv, hd]
            val kLin = linear(xk, w, a("wk_dv.weight"))
            val vLin = linear(xv, w, a("wv_dv.weight"))
            val (kConv, kHist) = if (sliding) shortConv(kLin, t(w, a("k_sconv.weight")), st?.get(2))
            else dynamicConv(kLin, xk, t(w, a("k_sconv.static_w")), a("k_sconv.weight_proj"), nkv, w, st?.get(2))
            val (vConv, vHist) = if (sliding) shortConv(vLin, t(w, a("v_sconv.weight")), st?.get(3))
            else dynamicConv(vLin, xv, t(w, a("v_sconv.static_w")), a("v_sconv.weight_proj"), nkv, w, st?.get(3))
            val k = l2Clip(kConv.reshape(intArrayOf(t, nkv, hd)), c.qkClip)
            val v = rmsNorm(vConv.reshape(intArrayOf(t, nkv, hd)), null)
            if (st != null) return stepAttention(l, xq, q, k, v, w, st) to listOf(stepCaches!!.first, stepCaches!!.second, kHist, vHist)
            val prefillState = if (collectCtx == null) emptyList()
            else listOf(cacheRows(k.reshape(intArrayOf(t, nkv * hd))), cacheRows(v.reshape(intArrayOf(t, nkv * hd))), kHist, vHist)

            // Relative bias [nh, T, T]: rel[i, h, d] at d = i - j inside 0 ≤ d < extent, else 0.
            var rs = rowTimes(linear(xq, w, a("wr_du.weight")).reshape(intArrayOf(t, nh, c.dRel)), t(w, a("rel_l2_clip.scale")))
            rs = l2Clip(rs, c.relClip)
            if (!sliding) rs = logScale(rs)
            val relLogits = mm(rs.reshape<Shape>(intArrayOf(t * nh, c.dRel)), t(w, a("rel_logits_proj.proj")))
                .reshape<Shape>(intArrayOf(t, nh, extent))
            val sel = FloatArray(t * extent * t)
            for (i in 0 until t) for (j in 0..i) {
                val d = i - j
                if (d < extent) sel[(i * extent + d) * t + j] = 1f
            }
            val bias = bmm(relLogits, anchor.constant<Shape>(sel, intArrayOf(t, extent, t))).transpose<Shape>(1, 0, 2) // [nh, T, T]
            val maskV = FloatArray(t * t) { ij ->
                val d = ij / t - ij % t
                if (d >= 0 && (!sliding || d < c.window)) 0f else -1e30f
            }
            val mask = bc(anchor.constant<Shape>(maskV, intArrayOf(1, t, t)), intArrayOf(nh, t, t))

            val qh = q.transpose<Shape>(1, 0, 2) // [nh, T, hd]
            val g = nh / nkv
            fun heads(x: Tracer<Shape>) = bc(x.transpose<Shape>(1, 0, 2).reshape<Shape>(intArrayOf(nkv, 1, t, hd)), intArrayOf(nkv, g, t, hd))
                .reshape<Shape>(intArrayOf(nh, t, hd))
            val kh = heads(k)
            val vh = heads(v)
            val scores = bmm(qh, kh.transpose<Shape>(0, 2, 1)) * (1.0 / sqrt(hd.toDouble())).toFloat() + bias + mask
            val sink = linear(xq, w, a("attn_sink_proj.weight")).transpose<Shape>(1, 0).reshape<Shape>(intArrayOf(nh, t, 1))
            val probs = concat<Shape>(listOf(scores, sink), 2).softmax(2).slice<Shape>(0, t, 2)
            val o = bmm(probs, vh).transpose<Shape>(1, 0, 2).reshape<Shape>(intArrayOf(t, nh * hd))
            val gate = linear(xq, w, a("gate_attn.weight")).sigmoid()
            return linear(o * gate, w, a("wo_ud.weight")) to prefillState
        }

        /** The new caches of the last [stepAttention]. */
        private var stepCaches: Pair<Tracer<Shape>, Tracer<Shape>>? = null

        /** Attention of the one query row over the caches (the new row written at `pos`). */
        private fun stepAttention(l: Int, xq: Tracer<Shape>, q: Tracer<Shape>, k: Tracer<Shape>, v: Tracer<Shape>, w: Weights, st: List<Tracer<Shape>>): Tracer<Shape> {
            val sliding = c.sliding[l]
            val nh = if (sliding) c.swaHeads else c.heads
            val nkv = if (sliding) c.swaKvHeads else c.kvHeads
            val hd = if (sliding) c.swaHeadDim else c.headDim
            val extent = if (sliding) c.window else c.relExtent
            val ctx = step!!.ctx
            fun a(rest: String) = InklingNames.layer(l, "attn.$rest")
            val kc = write(st[0], k.reshape(intArrayOf(t, nkv * hd)))
            val vc = write(st[1], v.reshape(intArrayOf(t, nkv * hd)))
            stepCaches = kc to vc

            var rs = rowTimes(linear(xq, w, a("wr_du.weight")).reshape(intArrayOf(t, nh, c.dRel)), t(w, a("rel_l2_clip.scale")))
            rs = l2Clip(rs, c.relClip)
            if (!sliding) rs = logScale(rs)
            val rel = mm(rs.reshape<Shape>(intArrayOf(t * nh, c.dRel)), t(w, a("rel_logits_proj.proj"))) // [t·nh, extent]
            val bias = if (t == 1) mm(rel, selector(extent).reshape<Shape>(intArrayOf(extent, ctx))).reshape<Shape>(intArrayOf(nh, 1, ctx))
            else bmm(rel.reshape<Shape>(intArrayOf(t, nh, extent)), selector(extent)).transpose<Shape>(1, 0, 2) // [nh, t, ctx]

            val g = nh / nkv
            fun heads(x: Tracer<Shape>) = bc(x.reshape<Shape>(intArrayOf(ctx, nkv, hd)).transpose<Shape>(1, 0, 2).reshape<Shape>(intArrayOf(nkv, 1, ctx, hd)), intArrayOf(nkv, g, ctx, hd))
                .reshape<Shape>(intArrayOf(nh, ctx, hd))
            val qh = if (t == 1) q.reshape<Shape>(intArrayOf(nh, 1, hd)) else q.transpose<Shape>(1, 0, 2) // [nh, t, hd]
            val scores = bmm(qh, heads(kc).transpose<Shape>(0, 2, 1)) * (1.0 / sqrt(hd.toDouble())).toFloat() + bias +
                bc(stepMask(sliding), intArrayOf(nh, t, ctx))
            val sinkRows = linear(xq, w, a("attn_sink_proj.weight")) // [t, nh]
            val sink = if (t == 1) sinkRows.reshape<Shape>(intArrayOf(nh, 1, 1)) else sinkRows.transpose<Shape>(1, 0).reshape<Shape>(intArrayOf(nh, t, 1))
            val probs = concat<Shape>(listOf(scores, sink), 2).softmax(2).slice<Shape>(0, ctx, 2)
            val ob = bmm(probs, heads(vc)) // [nh, t, hd]
            val o = if (t == 1) ob.reshape<Shape>(intArrayOf(1, nh * hd)) else ob.transpose<Shape>(1, 0, 2).reshape<Shape>(intArrayOf(t, nh * hd))
            val gate = linear(xq, w, a("gate_attn.weight")).sigmoid()
            return linear(o * gate, w, a("wo_ud.weight"))
        }

        fun moe(l: Int, x: Tracer<Shape>, w: Weights): Tracer<Shape> {
            val e = c.experts
            val i = c.expertDim
            val hd = c.hidden
            val scores = linear(x, w, InklingNames.layer(l, "mlp.gate.weight")).sigmoid() // [T, E]
            val biased = scores + bc(t(w, InklingNames.layer(l, "mlp.gate.bias")), intArrayOf(t, e))
            // The chosen experts as a 0/1 mask from the top-k positions (integers, so the mask is exact;
            // comparing scores with the k-th score instead depends on both being rounded alike).
            val k = c.expertsPerToken
            val top = f32(biased.topK(k).cast(F32)).reshape<Shape>(intArrayOf(t, k, 1))
            val iota = anchor.constant<Shape>(FloatArray(e) { it.toFloat() }, intArrayOf(1, 1, e))
            val dist = bc(top, intArrayOf(t, k, e)) - bc(iota, intArrayOf(t, k, e))
            val hit = const(1f, intArrayOf(t, k, e)) - (dist * dist - 0.25f).step() // 1 where the position is e
            val chosen = hit.sum<Shape>(intArrayOf(1)) // [T, E]
            val picked = scores * chosen
            val weights = picked / bc(sumKeep(picked) + 1e-20f, intArrayOf(t, e)) * c.routeScale.toFloat() // [T, E]

            val (gateUp, down) = w.experts(l)
            fun swiglu(gu: Tracer<Shape>): Tracer<Shape> {
                val gate = gu.slice<Shape>(0, i, 2)
                val up = gu.slice<Shape>(i, 2 * i, 2)
                return softcap(gate, c.gateSoftcap) * gate.sigmoid() * softcap(up, c.upSoftcap)
            }
            if (routed && t > 1) return raggedExperts(x, biased.topK(k), weights, gateUp, down)
            if (step != null && t == 1) {
                // One token: only the chosen experts, gathered from the stacks.
                val idx = biased.topK(k) // [1, k]
                val sel = weights.takeAlong(idx).reshape<Shape>(intArrayOf(k, 1, 1))
                val gu = gateUp.reshape<Shape>(intArrayOf(1, e, hd * 2 * i)).takeAlong(idx).reshape<Shape>(intArrayOf(k, hd, 2 * i))
                val dn = down.reshape<Shape>(intArrayOf(1, e, i * hd)).takeAlong(idx).reshape<Shape>(intArrayOf(k, i, hd))
                val xb = bc(x.reshape<Shape>(intArrayOf(1, 1, hd)), intArrayOf(k, 1, hd))
                val po = bmmW(swiglu(bmmW(xb, gu)), dn) // [k, 1, H]
                return (po * bc(sel, intArrayOf(k, 1, hd))).sum<Shape>(intArrayOf(0))
            }
            val xb = bc(x.reshape<Shape>(intArrayOf(1, t, hd)), intArrayOf(e, t, hd))
            val perExpert = bmmW(swiglu(bmmW(xb, gateUp)), down) // [E, T, H]
            val weighted = perExpert * bc(weights.transpose<Shape>(1, 0).reshape<Shape>(intArrayOf(e, t, 1)), intArrayOf(e, t, hd))
            return weighted.sum<Shape>(intArrayOf(0))
        }

        /**
         * The routed experts of `x [t, H]` on their own tokens: the (token, expert) pairs sorted by
         * expert, each expert's rows through `gateUp[e]` and `down[e]` as two ragged dots, back in
         * pair order and summed with the routing weights. Equal to the dense sum (zero weight on the
         * others) up to the order of additions.
         */
        fun raggedExperts(x: Tracer<Shape>, idx: Tracer<Shape>, weights: Tracer<Shape>, gateUp: Tracer<Shape>, down: Tracer<Shape>): Tracer<Shape> {
            val e = c.experts
            val i = c.expertDim
            val hd = c.hidden
            val k = c.expertsPerToken
            val pairs = t * k
            require(e.toLong() * pairs + pairs < (1L shl 24)) { "raggedExperts: $t tokens × $k experts over $e is past the f32 sort keys' 2^24" }
            val sel = weights.takeAlong(idx) // [t, k]
            val ids = idx.cast(F32).reshape<Shape>(intArrayOf(pairs, 1))
            val key = ids * pairs.toFloat() + anchor.constant<Shape>(FloatArray(pairs) { it.toFloat() }, intArrayOf(pairs, 1))
            val order = key.neg().reshape<Shape>(intArrayOf(1, pairs)).topK(pairs)
            val back = order.cast(F32).neg().topK(pairs)
            val d = bc(ids, intArrayOf(pairs, e)) - bc(anchor.constant<Shape>(FloatArray(e) { it.toFloat() }, intArrayOf(1, e)), intArrayOf(pairs, e))
            val isExpert = const(1f, intArrayOf(pairs, e)) - (d * d - 0.25f).step()
            val sizes = mm(const(1f, intArrayOf(1, pairs)), isExpert).reshape<Shape>(intArrayOf(e)).cast(I32)
            val xs = bc(x.reshape<Shape>(intArrayOf(t, 1, hd)), intArrayOf(t, k, hd)).reshape<Shape>(intArrayOf(1, pairs, hd))
                .takeAlong(order).reshape<Shape>(intArrayOf(pairs, hd))
            fun rdot(a: Tracer<Shape>, b: Tracer<Shape>): Tracer<Shape> =
                if (b.dtype == BF16) f32(a.cast(BF16).raggedDot(b, sizes)) else a.raggedDot(b, sizes)
            val gu = rdot(xs, gateUp) // [pairs, 2I]
            val gate = gu.slice<Shape>(0, i, 1)
            val up = gu.slice<Shape>(i, 2 * i, 1)
            val y = rdot(softcap(gate, c.gateSoftcap) * gate.sigmoid() * softcap(up, c.upSoftcap), down) // [pairs, H]
            val yPairs = y.reshape<Shape>(intArrayOf(1, pairs, hd)).takeAlong(back).reshape<Shape>(intArrayOf(t, k, hd))
            return bmm(sel.reshape<Shape>(intArrayOf(t, 1, k)), yPairs).reshape(intArrayOf(t, hd))
        }

        fun layer(l: Int, x: Tracer<Shape>, blocks: List<Tracer<Shape>>, messages: List<Tracer<Shape>>?, w: Weights): Tracer<Shape> {
            fun n(rest: String) = t(w, InklingNames.layer(l, rest))
            val st = step?.take()
            val blockResidual = x
            val (attnOut, attnState) = if (c.sliding[l]) {
                val h = if (l == 0) x else blockMix(blocks, x, t(w, InklingNames.attnQuery(l)), 1)[0]
                if (messages == null) {
                    val xn = rmsNorm(h, n("attn_norm.weight"))
                    attention(l, xn, xn, xn, w, st)
                } else {
                    val xq = rmsNorm(messageGate(h, messages[0], l, "attn", w), n("attn_norm.weight"))
                    val xk = rmsNorm(messageGate(h, messages[1], l, "k", w), n("attn_norm.weight"))
                    val xv = rmsNorm(messageGate(h, messages[2], l, "v", w), n("attn_norm.weight"))
                    attention(l, xq, xk, xv, w, st)
                }
            } else {
                var (hq, hk, hv) = if (l == 0) listOf(x, x, x) else blockMix(blocks, x, t(w, InklingNames.attnQuery(l)), 3)
                if (messages != null) {
                    hq = messageGate(hq, messages[0], l, "attn", w)
                    hk = messageGate(hk, messages[1], l, "k", w)
                    hv = messageGate(hv, messages[2], l, "v", w)
                }
                attention(l, rmsNorm(hq, n("q_input_norm.weight")), rmsNorm(hk, n("k_input_norm.weight")), rmsNorm(hv, n("v_input_norm.weight")), w, st)
            }
            val (a, aHist) = shortConv(attnOut, n("attn_sconv.weight"), st?.get(4))
            val mlpBlocks: List<Tracer<Shape>>
            val residual: Tracer<Shape>
            if (c.isBoundary(l)) {
                mlpBlocks = blocks + blockResidual
                residual = a
            } else {
                mlpBlocks = blocks
                residual = blockResidual + a
            }
            var m = blockMix(mlpBlocks, residual, t(w, InklingNames.mlpQuery(l)), 1)[0]
            if (messages != null) m = messageGate(m, messages[3], l, "mlp", w)
            val mo = moe(l, rmsNorm(m, n("mlp_norm.weight")), w)
            val (mc, mHist) = shortConv(mo, n("mlp_sconv.weight"), st?.get(5))
            if (step != null) step.out += attnState + listOf(aHist, mHist)
            if (collectCtx != null) collected += attnState + listOf(aHist, mHist)
            return residual + mc
        }
    }
}
