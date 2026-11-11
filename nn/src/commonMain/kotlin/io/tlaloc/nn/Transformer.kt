/**
 * Transformer layers: normalization, rotary position embeddings, causal
 * multi-head attention with grouped key/value heads, the SwiGLU and plain MLP
 * feed-forwards, a pre-norm block, and a decoder-only language model.
 *
 * Every forward is a trace spelling over `:autograd` ops, and the gradients
 * come from `DxirReverseTransform`, as for every other `:nn` layer. The
 * semantics follow PyTorch and Hugging Face transformers (`LlamaRMSNorm`,
 * `LlamaAttention` with `rotate_half` RoPE and `repeat_kv`, `LlamaMLP`), and
 * `TransformerOracleTest` checks gradients against PyTorch.
 *
 * Activations are `[batch, seq, dModel]`. `Dense` weights are `[in, out]`;
 * Hugging Face stores `nn.Linear` weights `[out, in]` (see `HfCausalLm.kt`).
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.checkpoint
import io.tlaloc.autograd.bmm
import io.tlaloc.autograd.broadcastTo
import io.tlaloc.autograd.concat
import io.tlaloc.autograd.constantMatching
import io.tlaloc.autograd.div
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.neg
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.sigmoid
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.softmax
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.sqrt
import io.tlaloc.autograd.times
import io.tlaloc.autograd.transpose
import kotlin.math.pow

// ---------------------------------------------------------------------------
// Named children. A composite layer's parameter keys are "<child>.<childKey>",
// and its forward hands each child a Params scoped to that prefix.
// ---------------------------------------------------------------------------

internal fun childParameters(children: List<Pair<String, Trainable<*>?>>): List<NamedParameter> =
    children.flatMap { (name, child) ->
        child?.parameters?.map { NamedParameter("$name.${it.key}", it.tensor) } ?: emptyList()
    }

/** Splits [updated] by child name; a key that names no child is an error. */
internal fun routeUpdates(
    owner: String,
    updated: Map<String, DTensor<*, F32>>,
    names: Collection<String>,
): Map<String, Map<String, DTensor<*, F32>>> {
    val out = LinkedHashMap<String, MutableMap<String, DTensor<*, F32>>>()
    for ((key, tensor) in updated) {
        val name = names.filter { key.startsWith("$it.") }.maxByOrNull { it.length }
            ?: throw IllegalArgumentException("$owner.withParameters: unknown key '$key' (children: $names)")
        out.getOrPut(name) { LinkedHashMap() }[key.substring(name.length + 1)] = tensor
    }
    return out
}

@Suppress("UNCHECKED_CAST")
internal fun <T : Trainable<*>> T.updatedWith(updates: Map<String, DTensor<*, F32>>?): T =
    if (updates == null) this else withParameters(updates) as T

private fun filled(n: Int, value: Float): DTensor<*, F32> =
    DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { value }), intArrayOf(n), F32)

/** The mean over the last axis, kept as a size-1 axis. */
private fun Tracer<Shape>.meanLastAxisKeep(): Tracer<Shape> {
    val last = rank - 1
    return mean<Shape>(intArrayOf(last)).reshape(dims.copyOf().also { it[last] = 1 })
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/**
 * PyTorch's `LayerNorm(dim, eps, elementwise_affine=True)` over the last axis:
 * `(x − mean) / √(var + eps) · weight + bias`, with the biased variance.
 * Parameters `weight` `[dim]` (ones) and `bias` `[dim]` (zeros); `bias` is
 * absent when constructed with `bias = false`.
 */
class LayerNorm(
    val weight: DTensor<*, F32>,
    val bias: DTensor<*, F32>?,
    val eps: Float = 1e-5f,
) : TrainableLayer<LayerNorm> {

    init {
        require(weight.dims.size == 1) { "LayerNorm: weight must be rank-1 [dim] (got ${weight.dims.toList()})" }
        require(bias == null || bias.dims.contentEquals(weight.dims)) {
            "LayerNorm: bias dims ${bias?.dims?.toList()} must equal weight dims ${weight.dims.toList()}"
        }
    }

    override val parameters: List<NamedParameter> =
        listOfNotNull(NamedParameter("weight", weight), bias?.let { NamedParameter("bias", it) })

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): LayerNorm {
        val known = parameters.map { it.key }.toSet()
        require((updated.keys - known).isEmpty()) { "LayerNorm.withParameters: unknown keys ${updated.keys - known}" }
        return LayerNorm(updated["weight"] ?: weight, if (bias != null) updated["bias"] ?: bias else null, eps)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 1 && x.dims[x.rank - 1] == weight.dims[0]) {
            "LayerNorm: last axis of ${x.dims.toList()} must be ${weight.dims[0]}"
        }
        val centered = x - x.meanLastAxisKeep().broadcastTo(x.dims)
        val variance = (centered * centered).meanLastAxisKeep()
        val std: Tracer<Shape> = (variance + variance.splat(eps)).sqrt().broadcastTo(x.dims)
        val scaled = centered / std * params["weight"].broadcastTo(x.dims)
        return if (bias != null) scaled + params["bias"].broadcastTo(x.dims) else scaled
    }

    companion object {
        operator fun invoke(dim: Int, eps: Float = 1e-5f, bias: Boolean = true): LayerNorm {
            require(dim > 0) { "LayerNorm: dim must be positive (got $dim)" }
            return LayerNorm(filled(dim, 1f), if (bias) filled(dim, 0f) else null, eps)
        }
    }
}

/**
 * Root-mean-square normalization over the last axis, as `LlamaRMSNorm`:
 * `x / √(mean(x²) + eps) · weight`. Parameter `weight` `[dim]` (ones).
 */
class RMSNorm(
    val weight: DTensor<*, F32>,
    val eps: Float = 1e-6f,
) : TrainableLayer<RMSNorm> {

    init {
        require(weight.dims.size == 1) { "RMSNorm: weight must be rank-1 [dim] (got ${weight.dims.toList()})" }
    }

    override val parameters: List<NamedParameter> = listOf(NamedParameter("weight", weight))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): RMSNorm {
        require((updated.keys - setOf("weight")).isEmpty()) {
            "RMSNorm.withParameters: unknown keys ${updated.keys - setOf("weight")}"
        }
        return RMSNorm(updated["weight"] ?: weight, eps)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 1 && x.dims[x.rank - 1] == weight.dims[0]) {
            "RMSNorm: last axis of ${x.dims.toList()} must be ${weight.dims[0]}"
        }
        val meanSquare = (x * x).meanLastAxisKeep()
        val rms: Tracer<Shape> = (meanSquare + meanSquare.splat(eps)).sqrt().broadcastTo(x.dims)
        return x / rms * params["weight"].broadcastTo(x.dims)
    }

    companion object {
        operator fun invoke(dim: Int, eps: Float = 1e-6f): RMSNorm {
            require(dim > 0) { "RMSNorm: dim must be positive (got $dim)" }
            return RMSNorm(filled(dim, 1f), eps)
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/**
 * Rotary position embeddings in the Hugging Face layout: frequency
 * `θ_i = base^(−2i/headDim)` for `i < headDim/2`, the angle table
 * `[pos·θ, pos·θ]` (the two halves repeated, not interleaved), and
 * `x·cos + rotate_half(x)·sin` with `rotate_half(x) = [−x₂, x₁]`.
 * Not trainable; positions run from 0.
 */
class RotaryEmbedding(val headDim: Int, val base: Float = 10000f) {

    init {
        require(headDim > 0 && headDim % 2 == 0) { "RotaryEmbedding: headDim must be positive and even (got $headDim)" }
    }

    /** `cos` and `sin` tables, each `[seqLen, headDim]` row-major, computed in f32 as transformers does. */
    fun tables(seqLen: Int): Pair<FloatArray, FloatArray> {
        val half = headDim / 2
        val invFreq = FloatArray(half) { i -> 1f / base.pow((2 * i).toFloat() / headDim) }
        val cos = FloatArray(seqLen * headDim)
        val sin = FloatArray(seqLen * headDim)
        for (t in 0 until seqLen) {
            for (i in 0 until headDim) {
                val angle = t.toFloat() * invFreq[i % half]
                cos[t * headDim + i] = kotlin.math.cos(angle)
                sin[t * headDim + i] = kotlin.math.sin(angle)
            }
        }
        return cos to sin
    }

    /** Rotates [x] `[batch, heads, seq, headDim]`. */
    fun apply(x: Tracer<Shape>): Tracer<Shape> {
        require(x.rank == 4 && x.dims[3] == headDim) {
            "RotaryEmbedding: input must be [batch, heads, seq, $headDim] (got ${x.dims.toList()})"
        }
        val seq = x.dims[2]
        val (cos, sin) = tables(seq)
        val cosT: Tracer<Shape> = x.constantMatching<Shape>(cos, intArrayOf(seq, headDim)).broadcastTo(x.dims)
        val sinT: Tracer<Shape> = x.constantMatching<Shape>(sin, intArrayOf(seq, headDim)).broadcastTo(x.dims)
        val half = headDim / 2
        val x1: Tracer<Shape> = x.slice(0, half, 3)
        val x2: Tracer<Shape> = x.slice(half, headDim, 3)
        val rotated: Tracer<Shape> = concat(listOf(x2.neg(), x1), 3)
        return x * cosT + rotated * sinT
    }
}

/**
 * Multi-head scaled dot-product attention with separate q/k/v/o projections,
 * as `LlamaAttention`. Queries have [numHeads] heads and keys/values
 * [numKvHeads] (grouped-query attention; each key/value head serves
 * `numHeads / numKvHeads` consecutive query heads, as `repeat_kv`). With
 * [causal], position `t` attends to positions `≤ t`. With [qNorm] and
 * [kNorm] (Qwen3), each query and key head is RMS-normalized over `headDim`;
 * then, with [rope], queries and keys are rotated before the dot product.
 * With [window] (a sliding layer), position `t` attends to the `window`
 * positions up to and including `t`. With [gate] (Ripple, Muse Glimmer), the
 * attention output is multiplied by `sigmoid(gate(x))` before [o]. [scale] is
 * the softmax's scale when it is not `1 / sqrt(headDim)` (Granite).
 *
 * Input and output `[batch, seq, dModel]`. Parameter keys `q.w`, `k.w`,
 * `v.w`, `o.w` (and `.b` when the projections have biases), `g.w` with a
 * gate, and `qNorm.weight`, `kNorm.weight` when present.
 */
class MultiHeadAttention(
    val q: Dense,
    val k: Dense,
    val v: Dense,
    val o: Dense,
    val numHeads: Int,
    val numKvHeads: Int = numHeads,
    val causal: Boolean = true,
    val rope: RotaryEmbedding? = null,
    val qNorm: RMSNorm? = null,
    val kNorm: RMSNorm? = null,
    val window: Int? = null,
    val gate: Dense? = null,
    val scale: Float? = null,
) : TrainableLayer<MultiHeadAttention> {

    val headDim: Int = q.w.dims[1] / numHeads

    init {
        require(numHeads > 0 && numKvHeads > 0 && numHeads % numKvHeads == 0) {
            "MultiHeadAttention: numHeads=$numHeads must be a positive multiple of numKvHeads=$numKvHeads"
        }
        require(q.w.dims[1] == numHeads * headDim) {
            "MultiHeadAttention: q projects to ${q.w.dims[1]}, not a multiple of numHeads=$numHeads"
        }
        for ((name, proj) in listOf("k" to k, "v" to v)) {
            require(proj.w.dims[1] == numKvHeads * headDim) {
                "MultiHeadAttention: $name projects to ${proj.w.dims[1]}, expected numKvHeads*headDim = ${numKvHeads * headDim}"
            }
        }
        require(o.w.dims[0] == numHeads * headDim) {
            "MultiHeadAttention: o takes ${o.w.dims[0]} inputs, expected numHeads*headDim = ${numHeads * headDim}"
        }
        require(rope == null || rope.headDim == headDim) {
            "MultiHeadAttention: rope.headDim=${rope?.headDim} != headDim=$headDim"
        }
        require((qNorm == null) == (kNorm == null)) { "MultiHeadAttention: qNorm and kNorm come together" }
        require(qNorm == null || (qNorm.weight.dims[0] == headDim && kNorm!!.weight.dims[0] == headDim)) {
            "MultiHeadAttention: qNorm/kNorm must normalize headDim=$headDim"
        }
        require(q.activation == Activation.Identity && k.activation == Activation.Identity &&
            v.activation == Activation.Identity && o.activation == Activation.Identity &&
            gate?.activation.let { it == null || it == Activation.Identity }) {
            "MultiHeadAttention: the projections must have no activation"
        }
        require(window == null || (window >= 1 && causal)) { "MultiHeadAttention: a window ($window) is a positive span of a causal layer" }
        require(gate == null || (gate.w.dims[0] == q.w.dims[0] && gate.w.dims[1] == numHeads * headDim)) {
            "MultiHeadAttention: the gate must project dModel to numHeads*headDim = ${numHeads * headDim}"
        }
    }

    private val children: List<Pair<String, Trainable<*>?>>
        get() = listOf("q" to q, "k" to k, "v" to v, "qNorm" to qNorm, "kNorm" to kNorm, "g" to gate, "o" to o)

    override val parameters: List<NamedParameter> = childParameters(children)

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): MultiHeadAttention {
        val u = routeUpdates("MultiHeadAttention", updated, children.filter { it.second != null }.map { it.first })
        return MultiHeadAttention(
            q.updatedWith(u["q"]), k.updatedWith(u["k"]), v.updatedWith(u["v"]), o.updatedWith(u["o"]),
            numHeads, numKvHeads, causal, rope, qNorm?.updatedWith(u["qNorm"]), kNorm?.updatedWith(u["kNorm"]),
            window, gate?.updatedWith(u["g"]), scale,
        )
    }

    /** This layer with another causal flag and everything else unchanged. */
    fun withCausal(causal: Boolean): MultiHeadAttention =
        MultiHeadAttention(q, k, v, o, numHeads, numKvHeads, causal, rope, qNorm, kNorm, window, gate, scale)

    /** This layer with other projections (LoRA, sharding) and everything else unchanged. */
    fun withProjections(q: Dense, k: Dense, v: Dense, o: Dense, gate: Dense? = this.gate): MultiHeadAttention =
        MultiHeadAttention(q, k, v, o, numHeads, numKvHeads, causal, rope, qNorm, kNorm, window, gate, scale)

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = forward(x, params, null)

    /** As [forward], with [v] reading [valueInput] instead of [x] (Ripple's value residual). */
    @Suppress("UNCHECKED_CAST")
    fun forward(x: Tracer<Shape>, params: Params, valueInput: Tracer<Shape>?): Tracer<Shape> {
        require(x.rank == 3) { "MultiHeadAttention: input must be [batch, seq, dModel] (got ${x.dims.toList()})" }
        require(valueInput == null || valueInput.dims.contentEquals(x.dims)) {
            "MultiHeadAttention: the value input ${valueInput?.dims?.toList()} must be shaped like x ${x.dims.toList()}"
        }
        val batch = x.dims[0]
        val seq = x.dims[1]

        // [batch, seq, heads*headDim] -> [batch, heads, seq, headDim]
        fun heads(t: Tracer<Shape>, n: Int): Tracer<Shape> =
            t.reshape<Shape>(intArrayOf(batch, seq, n, headDim)).transpose(0, 2, 1, 3)

        var qh = heads(q.forward(x, params.scoped("q")), numHeads)
        var kh = heads(k.forward(x, params.scoped("k")), numKvHeads)
        var vh = heads(v.forward(valueInput ?: x, params.scoped("v")), numKvHeads)
        if (qNorm != null && kNorm != null) {
            qh = qNorm.forward(qh, params.scoped("qNorm"))
            kh = kNorm.forward(kh, params.scoped("kNorm"))
        }
        if (rope != null) {
            qh = rope.apply(qh)
            kh = rope.apply(kh)
        }
        val groups = numHeads / numKvHeads
        if (groups > 1) {
            fun repeat(t: Tracer<Shape>): Tracer<Shape> =
                t.reshape<Shape>(intArrayOf(batch, numKvHeads, 1, seq, headDim))
                    .broadcastTo<Shape>(intArrayOf(batch, numKvHeads, groups, seq, headDim))
                    .reshape(intArrayOf(batch, numHeads, seq, headDim))
            kh = repeat(kh)
            vh = repeat(vh)
        }
        val bh = batch * numHeads
        val q3 = qh.reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(bh, seq, headDim))
        val kT = kh.reshape<Shape>(intArrayOf(bh, seq, headDim)).transpose<Rank3<Sym, Sym, Sym>>(0, 2, 1)
        val v3 = vh.reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(bh, seq, headDim))

        var scores = (q3 bmm kT) as Tracer<Shape>
        scores = scores * scores.splat(scale ?: (1f / kotlin.math.sqrt(headDim.toFloat())))
        if (causal) {
            // Row t (the query) attends to column s when s <= t, and with a window when t - s < window.
            val span = window ?: Int.MAX_VALUE
            val mask = FloatArray(seq * seq) { i ->
                val t = i / seq
                val s = i % seq
                if (s > t || t - s >= span) MASKED else 0f
            }
            scores = scores + scores.constantMatching<Shape>(mask, intArrayOf(seq, seq)).broadcastTo(scores.dims)
        }
        val probs = scores.softmax(-1) as Tracer<Rank3<Sym, Sym, Sym>>
        val attended = (probs bmm v3) as Tracer<Shape>
        var merged = attended.reshape<Shape>(intArrayOf(batch, numHeads, seq, headDim))
            .transpose<Shape>(0, 2, 1, 3)
            .reshape<Shape>(intArrayOf(batch, seq, numHeads * headDim))
        if (gate != null) merged = merged * gate.forward(x, params.scoped("g")).sigmoid()
        return o.forward(merged, params.scoped("o"))
    }

    companion object {
        /**
         * The score added at masked positions. Finite so that the StableHLO
         * literal is a plain number; `exp(−1e9 − max)` is exactly 0 in f32
         * and bf16, so the probabilities equal those of a `−∞` mask.
         */
        const val MASKED: Float = -1e9f

        /** Projections drawn from `N(0, initStd²)`, the Llama initialization. */
        operator fun invoke(
            dModel: Int,
            numHeads: Int,
            key: RandomKey,
            numKvHeads: Int = numHeads,
            headDim: Int = dModel / numHeads,
            bias: Boolean = false,
            causal: Boolean = true,
            rope: RotaryEmbedding? = null,
            qkNormEps: Float? = null,
            initStd: Float = 0.02f,
        ): MultiHeadAttention {
            val keys = key.split(4)
            fun proj(i: Int, inp: Int, out: Int) = normalDense(keys[i], inp, out, bias, initStd)
            return MultiHeadAttention(
                proj(0, dModel, numHeads * headDim),
                proj(1, dModel, numKvHeads * headDim),
                proj(2, dModel, numKvHeads * headDim),
                proj(3, numHeads * headDim, dModel),
                numHeads, numKvHeads, causal, rope,
                qkNormEps?.let { RMSNorm(headDim, it) }, qkNormEps?.let { RMSNorm(headDim, it) },
            )
        }
    }
}

internal fun normalDense(key: RandomKey, inputs: Int, outputs: Int, bias: Boolean, std: Float): Dense {
    val keys = key.split(2)
    val w = gaussianInit(keys[0], intArrayOf(inputs, outputs), variance = std * std)
    return Dense(w, if (bias) filled(outputs, 0f) else null)
}

// ---------------------------------------------------------------------------
// Feed-forward
// ---------------------------------------------------------------------------

/**
 * `down(silu(gate(x)) · up(x))`, as `LlamaMLP`. With [clamp] (Ripple), the
 * gate is clamped to at most `clamp` and the up projection to `[-clamp, clamp]`
 * before the product, with PyTorch's `clamp` gradient (1 inside the range and
 * on its bounds, 0 outside). Parameter keys `gate.w`, `up.w`, `down.w`.
 */
class SwiGLU(val gate: Dense, val up: Dense, val down: Dense, val clamp: Float? = null) : TrainableLayer<SwiGLU> {

    init {
        require(clamp == null || clamp > 0f) { "SwiGLU: clamp must be positive (got $clamp)" }
    }

    override val parameters: List<NamedParameter> =
        childParameters(listOf("gate" to gate, "up" to up, "down" to down))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): SwiGLU {
        val u = routeUpdates("SwiGLU", updated, listOf("gate", "up", "down"))
        return SwiGLU(gate.updatedWith(u["gate"]), up.updatedWith(u["up"]), down.updatedWith(u["down"]), clamp)
    }

    /** This MLP with other projections (LoRA, sharding) and the same clamp. */
    fun withProjections(gate: Dense, up: Dense, down: Dense): SwiGLU = SwiGLU(gate, up, down, clamp)

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        var g = gate.forward(x, params.scoped("gate"))
        var u = up.forward(x, params.scoped("up"))
        if (clamp != null) {
            // min(g, c) = g - relu(g - c); clamp(u, -c, c) = u - relu(u - c) + relu(-c - u).
            val c = g.splat(clamp)
            g = g - (g - c).relu()
            u = u - (u - c).relu() + (u.splat(-clamp) - u).relu()
        }
        return down.forward(Activation.Silu.apply(g) * u, params.scoped("down"))
    }

    companion object {
        operator fun invoke(dModel: Int, hidden: Int, key: RandomKey, initStd: Float = 0.02f): SwiGLU {
            val keys = key.split(3)
            return SwiGLU(
                normalDense(keys[0], dModel, hidden, false, initStd),
                normalDense(keys[1], dModel, hidden, false, initStd),
                normalDense(keys[2], hidden, dModel, false, initStd),
            )
        }
    }
}

/** `fc2(activation(fc1(x)))`, the GPT-2 MLP. Parameter keys `fc1.*`, `fc2.*`. */
class Mlp(val fc1: Dense, val fc2: Dense) : TrainableLayer<Mlp> {

    override val parameters: List<NamedParameter> = childParameters(listOf("fc1" to fc1, "fc2" to fc2))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Mlp {
        val u = routeUpdates("Mlp", updated, listOf("fc1", "fc2"))
        return Mlp(fc1.updatedWith(u["fc1"]), fc2.updatedWith(u["fc2"]))
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        fc2.forward(fc1.forward(x, params.scoped("fc1")), params.scoped("fc2"))

    companion object {
        operator fun invoke(
            dModel: Int,
            hidden: Int,
            key: RandomKey,
            activation: Activation = Activation.GeluTanh,
            initStd: Float = 0.02f,
        ): Mlp {
            val keys = key.split(2)
            val fc1 = normalDense(keys[0], dModel, hidden, true, initStd)
            return Mlp(Dense(fc1.w, fc1.b, activation), normalDense(keys[1], hidden, dModel, true, initStd))
        }
    }
}

// ---------------------------------------------------------------------------
// Blocks and the language model
// ---------------------------------------------------------------------------

/**
 * A pre-norm residual block: `h = x + attn(attnNorm(x))`, then
 * `h + mlp(mlpNorm(h))`. With [attnOutNorm] and [mlpOutNorm] (Gemma-style
 * sandwich norms, Ripple) each sublayer's output is normalized before its
 * residual add. With [valueResLambda] (Ripple's value residual, a `[1]`
 * tensor `w`), the attention's value projection reads
 * `x0 + sigmoid(w) (attnNorm(x) - x0)`, `x0` the model's scaled embeddings
 * ([forward] with `x0`). With [residualMultiplier] (Granite) each sublayer's
 * output is multiplied by it before its residual add. Parameter keys are prefixed `attnNorm.`, `attn.`,
 * `attnOutNorm.`, `mlpNorm.`, `mlp.`, `mlpOutNorm.`, plus `valueResLambda`.
 */
class TransformerBlock(
    val attnNorm: TrainableLayer<*>,
    val attn: MultiHeadAttention,
    val mlpNorm: TrainableLayer<*>,
    val mlp: TrainableLayer<*>,
    val attnOutNorm: TrainableLayer<*>? = null,
    val mlpOutNorm: TrainableLayer<*>? = null,
    val valueResLambda: DTensor<*, F32>? = null,
    val residualMultiplier: Float = 1f,
) : TrainableLayer<TransformerBlock> {

    init {
        require(valueResLambda == null || valueResLambda.dims.contentEquals(intArrayOf(1))) {
            "TransformerBlock: valueResLambda is one scalar [1] (got ${valueResLambda?.dims?.toList()})"
        }
    }

    private val children: List<Pair<String, Trainable<*>?>>
        get() = listOf(
            "attnNorm" to attnNorm, "attn" to attn, "attnOutNorm" to attnOutNorm,
            "mlpNorm" to mlpNorm, "mlp" to mlp, "mlpOutNorm" to mlpOutNorm,
        )

    override val parameters: List<NamedParameter> =
        childParameters(children) + listOfNotNull(valueResLambda?.let { NamedParameter(LAMBDA, it) })

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): TransformerBlock {
        require(valueResLambda != null || LAMBDA !in updated) { "TransformerBlock.withParameters: unknown key '$LAMBDA'" }
        val u = routeUpdates("TransformerBlock", updated - LAMBDA, children.filter { it.second != null }.map { it.first })
        return TransformerBlock(
            attnNorm.updatedWith(u["attnNorm"]),
            attn.updatedWith(u["attn"]),
            mlpNorm.updatedWith(u["mlpNorm"]),
            mlp.updatedWith(u["mlp"]),
            attnOutNorm?.updatedWith(u["attnOutNorm"]),
            mlpOutNorm?.updatedWith(u["mlpOutNorm"]),
            updated[LAMBDA] ?: valueResLambda,
            residualMultiplier,
        )
    }

    /** This block with another attention and MLP (LoRA, sharding) and everything else unchanged. */
    fun withSublayers(attn: MultiHeadAttention, mlp: TrainableLayer<*>): TransformerBlock =
        TransformerBlock(attnNorm, attn, mlpNorm, mlp, attnOutNorm, mlpOutNorm, valueResLambda, residualMultiplier)

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = forward(x, null, params)

    /** The block on [x]; [x0], the model's scaled embeddings, is what a value residual blends in. */
    fun forward(x: Tracer<Shape>, x0: Tracer<Shape>?, params: Params): Tracer<Shape> {
        val xn = attnNorm.forward(x, params.scoped("attnNorm"))
        val valueInput = if (valueResLambda != null) {
            requireNotNull(x0) { "TransformerBlock: a value residual needs the model's embeddings (forward with x0)" }
            val blend: Tracer<Shape> = params[LAMBDA].sigmoid().broadcastTo(xn.dims)
            x0 + blend * (xn - x0)
        } else {
            null
        }
        var a = attn.forward(xn, params.scoped("attn"), valueInput)
        if (attnOutNorm != null) a = attnOutNorm.forward(a, params.scoped("attnOutNorm"))
        if (residualMultiplier != 1f) a = a * a.splat(residualMultiplier)
        val h = x + a
        var m = mlp.forward(mlpNorm.forward(h, params.scoped("mlpNorm")), params.scoped("mlp"))
        if (mlpOutNorm != null) m = mlpOutNorm.forward(m, params.scoped("mlpOutNorm"))
        if (residualMultiplier != 1f) m = m * m.splat(residualMultiplier)
        return h + m
    }

    companion object {
        private const val LAMBDA = "valueResLambda"

        /**
         * RMSNorm, RoPE attention (grouped when `numKvHeads < numHeads`, with
         * per-head q/k RMSNorm when [CausalLmConfig.qkNorm]) and a SwiGLU MLP, or
         * [CausalLmConfig.numExperts] SwiGLU experts ([MixtureOfExperts]) of width `ffHidden`.
         */
        fun llama(config: CausalLmConfig, key: RandomKey): TransformerBlock {
            val keys = key.split(2)
            return TransformerBlock(
                RMSNorm(config.dModel, config.normEps),
                MultiHeadAttention(
                    config.dModel, config.numHeads, keys[0],
                    numKvHeads = config.numKvHeads,
                    headDim = config.headDim,
                    rope = RotaryEmbedding(config.headDim, config.ropeTheta),
                    qkNormEps = if (config.qkNorm) config.normEps else null,
                    initStd = config.initStd,
                ),
                RMSNorm(config.dModel, config.normEps),
                if (config.numExperts > 0) {
                    MixtureOfExperts(config.dModel, config.ffHidden, config.numExperts, config.expertsPerToken, keys[1], config.initStd)
                } else {
                    SwiGLU(config.dModel, config.ffHidden, keys[1], config.initStd)
                },
            )
        }
    }
}

/** The shape of a Llama-architecture [CausalLM] (Qwen3 with [qkNorm]). */
data class CausalLmConfig(
    val vocabSize: Int,
    val dModel: Int,
    val numLayers: Int,
    val numHeads: Int,
    val ffHidden: Int,
    val numKvHeads: Int = numHeads,
    val headDim: Int = dModel / numHeads,
    val ropeTheta: Float = 10000f,
    val normEps: Float = 1e-5f,
    val tiedEmbeddings: Boolean = false,
    /** Per-head RMSNorm on queries and keys before RoPE, as Qwen3. */
    val qkNorm: Boolean = false,
    val initStd: Float = 0.02f,
    /** With more than 0, each block's MLP is a [MixtureOfExperts] of this many experts. */
    val numExperts: Int = 0,
    /** The experts each token is routed to, with [numExperts]. */
    val expertsPerToken: Int = 2,
)

/**
 * A decoder-only language model: token embedding, [blocks], a final norm and
 * an output projection. Input is I32 token ids `[batch, seq]`; output is
 * logits `[batch, seq, vocab]`. With [head] null the output projection is
 * the embedding table transposed (tied embeddings).
 *
 * The embeddings are multiplied by [embedScale] (Ripple: `sqrt(hidden)`)
 * before the first block; blocks with a value residual blend those scaled
 * embeddings into their value projections. The final norm's output is
 * multiplied by [logitScale] (Granite: `1 / logits_scaling`), which scales the
 * logits.
 *
 * Parameter keys: `embed.table`, `blocks.<i>.…`, `norm.…`, and `head.w`
 * unless tied.
 */
class CausalLM(
    val embed: Embedding,
    val blocks: List<TransformerBlock>,
    val norm: TrainableLayer<*>,
    val head: Dense?,
    val embedScale: Float = 1f,
    val logitScale: Float = 1f,
) : TrainableLayer<CausalLM> {

    init {
        require(head == null || head.b == null) { "CausalLM: the output projection has no bias" }
    }

    /** This model with other blocks and head (LoRA, sharding) and everything else unchanged. */
    fun withLayers(blocks: List<TransformerBlock>, head: Dense?): CausalLM = CausalLM(embed, blocks, norm, head, embedScale, logitScale)

    private val children: List<Pair<String, Trainable<*>?>>
        get() = listOf("embed" to embed) +
            blocks.mapIndexed { i, b -> "blocks.$i" to b } +
            listOf("norm" to norm, "head" to head)

    override val parameters: List<NamedParameter> = childParameters(children)

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): CausalLM {
        val u = routeUpdates("CausalLM", updated, children.filter { it.second != null }.map { it.first })
        return CausalLM(
            embed.updatedWith(u["embed"]),
            blocks.mapIndexed { i, b -> b.updatedWith(u["blocks.$i"]) },
            norm.updatedWith(u["norm"]),
            head?.updatedWith(u["head"]),
            embedScale,
            logitScale,
        )
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        val h = hidden(x, params)
        if (head != null) return head.forward(h, params.scoped("head"))
        val rows = h.size / h.dims[2]
        val flat = h.reshape<Rank2<Sym, Sym>>(intArrayOf(rows, h.dims[2]))
        val logits = flat matmul (headWeight(params) as Tracer<Rank2<Sym, Sym>>)
        return logits.reshape(intArrayOf(h.dims[0], h.dims[1], logits.dims[1]))
    }

    /** The final norm's output `[batch, seq, dModel]` times [logitScale]: [forward] before the output projection. */
    fun hidden(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.dtype == I32 && x.rank == 2) {
            "CausalLM: input must be I32 token ids [batch, seq] (got ${x.dtype.name} ${x.dims.toList()})"
        }
        var h: Tracer<Shape> = params["embed.table"].embedding(x)
        if (embedScale != 1f) h = h * h.splat(embedScale)
        val x0 = if (blocks.any { it.valueResLambda != null }) h else null
        // Block outputs are checkpoints for a rematerialized step (DxirRemat.SAVE_MARKED); the
        // mark changes nothing otherwise.
        for ((i, block) in blocks.withIndex()) h = block.forward(h, x0, params.scoped("blocks.$i")).checkpoint()
        val out = norm.forward(h, params.scoped("norm"))
        return if (logitScale == 1f) out else out * out.splat(logitScale)
    }

    /** The output projection `[dModel, vocab]`: `head.w`, or the embedding table transposed. */
    fun headWeight(params: Params): Tracer<Shape> {
        require(head?.lora == null) { "CausalLM.headWeight: the output projection has a LoRA adapter" }
        return if (head != null) params.scoped("head")["w"] else params["embed.table"].transpose(1, 0)
    }

    companion object {
        /** A randomly initialized Llama-architecture model, weights from `N(0, initStd²)`. */
        fun llama(config: CausalLmConfig, key: RandomKey): CausalLM {
            val keys = key.split(config.numLayers + 2)
            val table = gaussianInit(keys[0], intArrayOf(config.vocabSize, config.dModel), variance = config.initStd * config.initStd)
            return CausalLM(
                Embedding(table),
                List(config.numLayers) { TransformerBlock.llama(config, keys[1 + it]) },
                RMSNorm(config.dModel, config.normEps),
                if (config.tiedEmbeddings) null
                else normalDense(keys[config.numLayers + 1], config.dModel, config.vocabSize, false, config.initStd),
            )
        }
    }
}
