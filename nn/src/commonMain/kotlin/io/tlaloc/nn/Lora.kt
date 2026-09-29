/**
 * LoRA (Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language
 * Models"): a frozen `Dense` weight `W` `[in, out]` plus a trained low-rank
 * update, `y = x·W + b + scale·(dropout(x)·A)·B`, with `A` `[in, r]`,
 * `B` `[r, out]` and `scale = alpha / r` (`alpha / √r` with rsLoRA). This is
 * the computation of Hugging Face PEFT's `LoraLayer` for `nn.Linear`, with
 * PEFT's initialization: `A` kaiming-uniform (`U(±1/√in)`), `B` zero, so an
 * adapted model computes exactly what the base model does until `B` trains.
 *
 * [Lora.apply] adds adapters to the `Dense` layers a [LoraConfig] names,
 * [Lora.frozen] freezes everything else for [capture] and
 * [Optimizer.step], and [Lora.merge] folds the adapters into the base
 * weights (`W + scale·A·B`) so the result is an ordinary model that the
 * checkpoint writers and the serving exporter take as is.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.times
import kotlin.jvm.JvmOverloads
import kotlin.math.sqrt

/**
 * One adapter: `A` `[in, r]`, `B` `[r, out]` (the [Dense] `[in, out]`
 * layout; PEFT's files store their transposes), [alpha], and dropout on the
 * adapter's input.
 *
 * Dropout follows [Dropout]: the mask is a constant drawn from
 * [dropoutKey], so a new mask needs a new key and a new capture
 * ([Lora.withDropoutKey]). With [dropout] 0, or with no [dropoutKey] (the
 * inference form, as PEFT's `eval()`), the adapter input is `x` itself.
 */
class LoraAdapter @JvmOverloads constructor(
    val a: DTensor<*, F32>,
    val b: DTensor<*, F32>,
    val alpha: Float,
    val dropout: Float = 0f,
    val dropoutKey: RandomKey? = null,
    val useRslora: Boolean = false,
) {
    init {
        require(a.dims.size == 2 && b.dims.size == 2 && a.dims[1] == b.dims[0] && a.dims[1] > 0) {
            "LoraAdapter: A must be [in, r] and B [r, out] with r > 0 (got ${a.dims.toList()} and ${b.dims.toList()})"
        }
        require(dropout >= 0f && dropout < 1f) { "LoraAdapter: dropout=$dropout must lie in [0, 1)" }
    }

    val rank: Int get() = a.dims[1]

    /** `alpha / r`, or `alpha / √r` with rsLoRA. */
    val scale: Float get() = if (useRslora) alpha / sqrt(rank.toFloat()) else alpha / rank

    /** This adapter with other A and B tensors. */
    fun withTensors(a: DTensor<*, F32>, b: DTensor<*, F32>): LoraAdapter =
        LoraAdapter(a, b, alpha, dropout, dropoutKey, useRslora)

    /** This adapter with another dropout key (null turns dropout off). */
    fun withDropoutKey(key: RandomKey?): LoraAdapter = LoraAdapter(a, b, alpha, dropout, key, useRslora)

    /** `scale·A·B` `[in, out]`, accumulated in f64. */
    fun delta(): FloatArray {
        val (n, r) = a.dims[0] to a.dims[1]
        val m = b.dims[1]
        val av = a.hostF32()
        val bv = b.hostF32()
        val out = FloatArray(n * m)
        val acc = DoubleArray(m)
        for (i in 0 until n) {
            acc.fill(0.0)
            for (k in 0 until r) {
                val aik = av[i * r + k].toDouble()
                if (aik == 0.0) continue
                val row = k * m
                for (j in 0 until m) acc[j] += aik * bv[row + j]
            }
            for (j in 0 until m) out[i * m + j] = (acc[j] * scale).toFloat()
        }
        return out
    }

    /** `scale·(dropout(x)·A)·B` for rows [x] `[rows, in]`. */
    @Suppress("UNCHECKED_CAST")
    internal fun forward(x: Tracer<Rank2<Sym, Sym>>, params: Params): Tracer<Shape> {
        val input: Tracer<Rank2<Sym, Sym>> =
            if (dropout > 0f && dropoutKey != null) {
                val u = uniformFloats(dropoutKey, x.size)
                val keep = 1f / (1f - dropout)
                val mask: Tracer<Rank2<Sym, Sym>> = x.constant(FloatArray(x.size) { if (u[it] > dropout) keep else 0f }, x.dims)
                x * mask
            } else x
        val h = input matmul (params[A_KEY] as Tracer<Rank2<Sym, Sym>>)
        val d = h matmul (params[B_KEY] as Tracer<Rank2<Sym, Sym>>)
        return (d * d.splat(scale)) as Tracer<Shape>
    }

    companion object {
        /** The adapter's parameter keys inside its [Dense]. */
        const val A_KEY: String = "lora_A"
        const val B_KEY: String = "lora_B"
    }
}

/**
 * Which `Dense` layers get adapters, and their shape.
 *
 * [targetModules] are matched as PEFT matches `target_modules`: a layer is a
 * target if its name equals an entry or ends with `.` and the entry. Names
 * are the layer's parameter-key path (`blocks.3.attn.q`, `0`) and, inside a
 * [CausalLM], its Hugging Face module name
 * (`model.layers.3.self_attn.q_proj`, `lm_head`), so `"q_proj"` selects every
 * query projection.
 */
data class LoraConfig @JvmOverloads constructor(
    val rank: Int,
    val alpha: Float,
    val targetModules: List<String>,
    val dropout: Float = 0f,
    val useRslora: Boolean = false,
) {
    init {
        require(rank > 0) { "LoraConfig: rank must be positive (got $rank)" }
        require(targetModules.isNotEmpty()) { "LoraConfig: targetModules is empty" }
        require(dropout >= 0f && dropout < 1f) { "LoraConfig: dropout=$dropout must lie in [0, 1)" }
    }

    companion object {
        /** The attention projections of a Llama or Qwen3 model. */
        @JvmField
        val ATTENTION: List<String> = listOf("q_proj", "k_proj", "v_proj", "o_proj")

        /** Every linear layer of a Llama or Qwen3 decoder block. */
        @JvmField
        val ALL_LINEAR: List<String> = ATTENTION + listOf("gate_proj", "up_proj", "down_proj")
    }
}

/** Adding, freezing around, keying and merging LoRA adapters. */
object Lora {

    /** True for an adapter parameter key (`blocks.0.attn.q.lora_A`). */
    @JvmStatic
    fun isAdapterKey(key: String): Boolean {
        val last = key.substringAfterLast('.')
        return last == LoraAdapter.A_KEY || last == LoraAdapter.B_KEY
    }

    /** Everything except the adapters: pass it to [capture] and [Optimizer.step]. */
    @JvmField
    val frozen: Frozen = Frozen.allExcept(::isAdapterKey)

    /** The adapter parameters of [model], in model order. */
    @JvmStatic
    fun adapterParameters(model: Trainable<*>): List<NamedParameter> = model.parameters.filter { isAdapterKey(it.key) }

    /** True when [model] holds at least one adapter. */
    @JvmStatic
    fun hasAdapters(model: Trainable<*>): Boolean = model.parameters.any { isAdapterKey(it.key) }

    /**
     * The names of the `Dense` layers of [model] that [targets] select, in
     * model order, as parameter-key paths (`blocks.0.attn.q`).
     */
    @JvmStatic
    fun matchingLayers(model: Layer, targets: List<String>): List<String> =
        denseLayers(model).filter { (path, hf) -> matches(path, hf, targets) }.map { it.first }

    /**
     * [model] with an adapter on every `Dense` that [config] selects: `A`
     * drawn from `U(±1/√in)` off [key] (one child key per layer, in model
     * order), `B` zero. Refuses a target that selects no layer and a layer
     * that already has an adapter.
     */
    @JvmStatic
    @Suppress("UNCHECKED_CAST")
    fun <M : Layer> apply(model: M, config: LoraConfig, key: RandomKey): M {
        val layers = denseLayers(model)
        for (t in config.targetModules) {
            require(layers.any { (path, hf) -> matches(path, hf, listOf(t)) }) {
                val names = layers.map { (p, hf) -> if (hf != null) "$p ($hf)" else p }
                "Lora.apply: target module '$t' matches no Dense layer of ${model::class.simpleName} " +
                    "(layers: ${names.take(12).joinToString()}${if (names.size > 12) ", ..." else ""})"
            }
        }
        val selected = layers.filter { (path, hf) -> matches(path, hf, config.targetModules) }.map { it.first }
        val keys = key.split(maxOf(selected.size, 1))
        val index = selected.withIndex().associate { (i, p) -> p to i }
        val keySplit = config.dropout > 0f
        return rewriteDense(model, "", hfRoot(model)) { dense, path, _ ->
            val i = index[path] ?: return@rewriteDense dense
            require(dense.lora == null) { "Lora.apply: layer '$path' already has an adapter" }
            val inputs = dense.w.dims[0]
            val outputs = dense.w.dims[1]
            val children = keys[i].split(2)
            val bound = 1f / sqrt(inputs.toFloat())
            val a = uniformInit(children[0], intArrayOf(inputs, config.rank), -bound, bound)
            val b = DTensor<Shape, F32>(HostF32Storage(FloatArray(config.rank * outputs)), intArrayOf(config.rank, outputs), F32)
            dense.withLora(LoraAdapter(a, b, config.alpha, config.dropout, if (keySplit) children[1] else null, config.useRslora))
        } as M
    }

    /**
     * [model] with every adapter's dropout re-keyed from [key] (one child
     * per adapter, in model order), for a new mask each step. A new key
     * changes the traced graph, so the step must be captured again. Null
     * turns dropout off.
     */
    @JvmStatic
    fun <M : Layer> withDropoutKey(model: M, key: RandomKey?): M {
        val paths = ArrayList<String>()
        rewriteDense(model, "", null) { d, path, _ -> if (d.lora != null) paths += path; d }
        val keys = key?.split(maxOf(paths.size, 1))
        val index = paths.withIndex().associate { (i, p) -> p to i }
        @Suppress("UNCHECKED_CAST")
        return rewriteDense(model, "", null) { d, path, _ ->
            val lora = d.lora ?: return@rewriteDense d
            d.withLora(lora.withDropoutKey(keys?.get(index.getValue(path))))
        } as M
    }

    /**
     * [model] with every adapter folded into its base weight,
     * `W' = W + scale·A·B` (accumulated in f64, rounded to f32 once), and no
     * adapters left. The result computes what the adapted model computes
     * without dropout, and is an ordinary model for `HfCausalLm.save`, the
     * checkpoint writers and the serving exporter.
     */
    @JvmStatic
    fun <M : Layer> merge(model: M): M {
        @Suppress("UNCHECKED_CAST")
        return rewriteDense(model, "", null) { d, _, _ -> d.merged() } as M
    }

    // ------------------------------------------------------------------
    // Traversal. Rebuilds the containers this module defines around their
    // Dense children; any other layer is returned unchanged (its Dense
    // layers, if it has any, are not reachable and so never selected).
    // ------------------------------------------------------------------

    private fun matches(path: String, hf: String?, targets: List<String>): Boolean =
        targets.any { t -> path == t || path.endsWith(".$t") || (hf != null && (hf == t || hf.endsWith(".$t"))) }

    private fun hfRoot(model: Layer): String? = if (model is CausalLM) "" else null

    /** Every reachable Dense layer: its key path and, inside a [CausalLM], its Hugging Face module name. */
    private fun denseLayers(model: Layer): List<Pair<String, String?>> {
        val out = ArrayList<Pair<String, String?>>()
        rewriteDense(model, "", hfRoot(model)) { d, p, hf -> out += p to hf; d }
        return out
    }

    private fun join(prefix: String, name: String) = if (prefix.isEmpty()) name else "$prefix.$name"

    private fun hfJoin(prefix: String?, name: String?): String? =
        if (prefix == null || name == null) null else join(prefix, name)

    internal fun rewriteDense(
        layer: Layer,
        path: String,
        hf: String?,
        f: (Dense, String, String?) -> Dense,
    ): Layer = when (layer) {
        is Dense -> f(layer, path, hf)
        is Sequential -> Sequential(layer.layers.mapIndexed { i, l -> rewriteDense(l, join(path, "$i"), null, f) })
        is SwiGLU -> SwiGLU(
            rewriteDense(layer.gate, join(path, "gate"), hfJoin(hf, "gate_proj"), f) as Dense,
            rewriteDense(layer.up, join(path, "up"), hfJoin(hf, "up_proj"), f) as Dense,
            rewriteDense(layer.down, join(path, "down"), hfJoin(hf, "down_proj"), f) as Dense,
        )
        is Mlp -> Mlp(
            rewriteDense(layer.fc1, join(path, "fc1"), null, f) as Dense,
            rewriteDense(layer.fc2, join(path, "fc2"), null, f) as Dense,
        )
        is MultiHeadAttention -> MultiHeadAttention(
            rewriteDense(layer.q, join(path, "q"), hfJoin(hf, "q_proj"), f) as Dense,
            rewriteDense(layer.k, join(path, "k"), hfJoin(hf, "k_proj"), f) as Dense,
            rewriteDense(layer.v, join(path, "v"), hfJoin(hf, "v_proj"), f) as Dense,
            rewriteDense(layer.o, join(path, "o"), hfJoin(hf, "o_proj"), f) as Dense,
            layer.numHeads, layer.numKvHeads, layer.causal, layer.rope, layer.qNorm, layer.kNorm,
        )
        is TransformerBlock -> TransformerBlock(
            layer.attnNorm,
            rewriteDense(layer.attn, join(path, "attn"), hfJoin(hf, "self_attn"), f) as MultiHeadAttention,
            layer.mlpNorm,
            rewriteDense(layer.mlp, join(path, "mlp"), hfJoin(hf, "mlp"), f) as TrainableLayer<*>,
        )
        is CausalLM -> CausalLM(
            layer.embed,
            layer.blocks.mapIndexed { i, b ->
                rewriteDense(b, join(path, "blocks.$i"), hfJoin(hf, "model.layers.$i"), f) as TransformerBlock
            },
            layer.norm,
            layer.head?.let { rewriteDense(it, join(path, "head"), hfJoin(hf, "lm_head"), f) as Dense },
        )
        else -> layer
    }
}
