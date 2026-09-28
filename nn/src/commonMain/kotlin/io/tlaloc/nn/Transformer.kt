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
import io.tlaloc.autograd.reshape
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

internal fun Params.scoped(prefix: String): Params = Params { key -> this["$prefix.$key"] }

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
 *
 * Input and output `[batch, seq, dModel]`. Parameter keys `q.w`, `k.w`,
 * `v.w`, `o.w` (and `.b` when the projections have biases), and
 * `qNorm.weight`, `kNorm.weight` when present.
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
            v.activation == Activation.Identity && o.activation == Activation.Identity) {
            "MultiHeadAttention: the projections must have no activation"
        }
    }

    private val children: List<Pair<String, Trainable<*>?>>
        get() = listOf("q" to q, "k" to k, "v" to v, "qNorm" to qNorm, "kNorm" to kNorm, "o" to o)

    override val parameters: List<NamedParameter> = childParameters(children)

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): MultiHeadAttention {
        val u = routeUpdates("MultiHeadAttention", updated, children.filter { it.second != null }.map { it.first })
        return MultiHeadAttention(
            q.updatedWith(u["q"]), k.updatedWith(u["k"]), v.updatedWith(u["v"]), o.updatedWith(u["o"]),
            numHeads, numKvHeads, causal, rope, qNorm?.updatedWith(u["qNorm"]), kNorm?.updatedWith(u["kNorm"]),
        )
    }

    /** This layer with another causal flag and everything else unchanged. */
    fun withCausal(causal: Boolean): MultiHeadAttention =
        MultiHeadAttention(q, k, v, o, numHeads, numKvHeads, causal, rope, qNorm, kNorm)

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank == 3) { "MultiHeadAttention: input must be [batch, seq, dModel] (got ${x.dims.toList()})" }
        val batch = x.dims[0]
        val seq = x.dims[1]

        // [batch, seq, heads*headDim] -> [batch, heads, seq, headDim]
        fun heads(t: Tracer<Shape>, n: Int): Tracer<Shape> =
            t.reshape<Shape>(intArrayOf(batch, seq, n, headDim)).transpose(0, 2, 1, 3)

        var qh = heads(q.forward(x, params.scoped("q")), numHeads)
        var kh = heads(k.forward(x, params.scoped("k")), numKvHeads)
        var vh = heads(v.forward(x, params.scoped("v")), numKvHeads)
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
        scores = scores * scores.splat(1f / kotlin.math.sqrt(headDim.toFloat()))
        if (causal) {
            val mask = FloatArray(seq * seq) { i -> if (i % seq > i / seq) MASKED else 0f }
            scores = scores + scores.constantMatching<Shape>(mask, intArrayOf(seq, seq)).broadcastTo(scores.dims)
        }
        val probs = scores.softmax(-1) as Tracer<Rank3<Sym, Sym, Sym>>
        val attended = (probs bmm v3) as Tracer<Shape>
        val merged = attended.reshape<Shape>(intArrayOf(batch, numHeads, seq, headDim))
            .transpose<Shape>(0, 2, 1, 3)
            .reshape<Shape>(intArrayOf(batch, seq, numHeads * headDim))
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
 * `down(silu(gate(x)) · up(x))`, as `LlamaMLP`. Parameter keys `gate.w`,
 * `up.w`, `down.w`.
 */
class SwiGLU(val gate: Dense, val up: Dense, val down: Dense) : TrainableLayer<SwiGLU> {

    override val parameters: List<NamedParameter> =
        childParameters(listOf("gate" to gate, "up" to up, "down" to down))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): SwiGLU {
        val u = routeUpdates("SwiGLU", updated, listOf("gate", "up", "down"))
        return SwiGLU(gate.updatedWith(u["gate"]), up.updatedWith(u["up"]), down.updatedWith(u["down"]))
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        val g = Activation.Silu.apply(gate.forward(x, params.scoped("gate")))
        return down.forward(g * up.forward(x, params.scoped("up")), params.scoped("down"))
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
 * `h + mlp(mlpNorm(h))`. Parameter keys are prefixed `attnNorm.`, `attn.`,
 * `mlpNorm.`, `mlp.`.
 */
class TransformerBlock(
    val attnNorm: TrainableLayer<*>,
    val attn: MultiHeadAttention,
    val mlpNorm: TrainableLayer<*>,
    val mlp: TrainableLayer<*>,
) : TrainableLayer<TransformerBlock> {

    private val children: List<Pair<String, Trainable<*>>>
        get() = listOf("attnNorm" to attnNorm, "attn" to attn, "mlpNorm" to mlpNorm, "mlp" to mlp)

    override val parameters: List<NamedParameter> = childParameters(children)

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): TransformerBlock {
        val u = routeUpdates("TransformerBlock", updated, children.map { it.first })
        return TransformerBlock(
            attnNorm.updatedWith(u["attnNorm"]),
            attn.updatedWith(u["attn"]),
            mlpNorm.updatedWith(u["mlpNorm"]),
            mlp.updatedWith(u["mlp"]),
        )
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        val h = x + attn.forward(attnNorm.forward(x, params.scoped("attnNorm")), params.scoped("attn"))
        return h + mlp.forward(mlpNorm.forward(h, params.scoped("mlpNorm")), params.scoped("mlp"))
    }

    companion object {
        /**
         * RMSNorm, RoPE attention (grouped when `numKvHeads < numHeads`, with
         * per-head q/k RMSNorm when [CausalLmConfig.qkNorm]) and a SwiGLU MLP.
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
                SwiGLU(config.dModel, config.ffHidden, keys[1], config.initStd),
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
)

/**
 * A decoder-only language model: token embedding, [blocks], a final norm and
 * an output projection. Input is I32 token ids `[batch, seq]`; output is
 * logits `[batch, seq, vocab]`. With [head] null the output projection is
 * the embedding table transposed (tied embeddings).
 *
 * Parameter keys: `embed.table`, `blocks.<i>.…`, `norm.…`, and `head.w`
 * unless tied.
 */
class CausalLM(
    val embed: Embedding,
    val blocks: List<TransformerBlock>,
    val norm: TrainableLayer<*>,
    val head: Dense?,
) : TrainableLayer<CausalLM> {

    init {
        require(head == null || head.b == null) { "CausalLM: the output projection has no bias" }
    }

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
        )
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.dtype == I32 && x.rank == 2) {
            "CausalLM: input must be I32 token ids [batch, seq] (got ${x.dtype.name} ${x.dims.toList()})"
        }
        var h: Tracer<Shape> = params["embed.table"].embedding(x)
        for ((i, block) in blocks.withIndex()) h = block.forward(h, params.scoped("blocks.$i"))
        h = norm.forward(h, params.scoped("norm"))
        if (head != null) return head.forward(h, params.scoped("head"))
        val table = params["embed.table"]
        val rows = h.size / h.dims[2]
        val flat = h.reshape<Rank2<Sym, Sym>>(intArrayOf(rows, h.dims[2]))
        val logits = flat matmul table.transpose<Rank2<Sym, Sym>>(1, 0)
        return logits.reshape(intArrayOf(h.dims[0], h.dims[1], table.dims[0]))
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
