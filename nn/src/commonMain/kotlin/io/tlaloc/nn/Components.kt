/**
 * The `:nn` component substrate. Immutable, functional components in
 * DiffKT's own convention: a layer is a value holding its `DTensor`
 * parameters, `forward` is pure, and a training step returns a NEW instance.
 *
 * The AD route is the COMPILER stack: a layer's `forward`
 * is written once, in Tracer land, and [Training.kt]'s capture step turns the
 * whole model into a `DxirFunction` that `DxirReverseTransform` differentiates.
 * No gradient math lives in this module — the transform owns it.
 *
 * The Tracer surface here is the ERASED `Tracer<Shape>` view: a model layer is
 * shape-heterogeneous by nature (DiffKT's own surface is untyped `DTensor`),
 * so the phantom-shape branding stops at the module boundary and shapes are
 * checked at trace time by the traced ops themselves. REJECTED: threading the
 * phantom types through the layer interfaces — Dense-style shape changes make
 * a general `Layer` interface inexpressible without an explosion of arity- and
 * rank-specific variants, for zero runtime safety gain over the tracer's own
 * dims checks.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Shape
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.times

/** One parameter tensor with its STABLE path-like key ("m", "0.b", …). */
class NamedParameter(val key: String, val tensor: DTensor<*, F32>)

/**
 * Resolves a trainable component's parameter tensors to their leaf tracers
 * during a capture. Keys are the component's OWN keys — a container scopes the
 * resolver before handing it to a child (Sequential prefixes `"<index>."`).
 */
fun interface Params {
    operator fun get(key: String): Tracer<Shape>
}

/**
 * A layer: a pure traced forward, single input (DiffKT's `LayerSingleInput` —
 * DiffKT's multi-input `invoke(vararg)` has no counterpart here; DiffKT's own
 * Sequential is single-input too).
 * Non-trainable layers ignore [params].
 */
interface Layer {
    fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape>
}

/**
 * The training-capable node (DiffKT's `Trainable`, minus the
 * `extractTangent` extractor protocol — under the compiler route gradients
 * come back addressed by parameter KEY from the transformed graph, so no
 * extraction hook is needed).
 *
 * [parameters] and [withParameters] are exactly the two operations a
 * checkpoint needs, and [ModelCheckpoint] is written against them and nothing
 * else. Persistent state that is NOT trainable — BatchNorm's running
 * statistics — is [Stateful]'s business, for the reason recorded there.
 *
 * Contract: [parameters] is in stable declaration order with stable keys, and
 * [withParameters] returns a NEW instance with the keyed tensors replaced
 * (absent keys keep their current tensor; unknown keys are an error).
 */
interface Trainable<T : Trainable<T>> {
    val parameters: List<NamedParameter>
    fun withParameters(updated: Map<String, DTensor<*, F32>>): T
}

/** DiffKT's `TrainableLayer`: a layer that is also a trainable component. */
interface TrainableLayer<T : TrainableLayer<T>> : Layer, Trainable<T>

/**
 * The NON-TRAINABLE persistent state of a
 * component: tensors that are part of what the model IS, that a checkpoint
 * must carry, and that no optimizer ever touches. Today there is exactly one
 * such component, [BatchNorm], whose running statistics are the difference
 * between a reloaded model that can be served and one that emits NaNs the
 * first time [BatchNorm.inferenceMode] is called.
 *
 * This is a SEPARATE interface from [Trainable] on purpose, and the reason is
 * the gradient contract: [Trainable.parameters] is the list the reverse
 * transform returns one gradient per, keyed by [NamedParameter.key], and an
 * optimizer refuses a key it has no gradient for. Putting running statistics
 * in that list would make every optimizer step demand a gradient for a
 * quantity that has none. Naming them separately keeps both contracts exact.
 *
 * Contract, mirroring [Trainable] line for line: [buffers] is in stable
 * declaration order with stable keys, and [withBuffers] returns a NEW instance
 * with the keyed tensors replaced (absent keys keep their current tensor;
 * unknown keys are an error). A scalar buffer is a RANK-0 tensor, because
 * safetensors has a rank-0 shape (`[]`) and inventing a second scalar channel
 * in the checkpoint format would have been a second thing to keep in sync.
 *
 * **Container contract.** A container layer that can hold a [Stateful] child
 * must itself be [Stateful] and must prefix its children's keys exactly as it
 * prefixes their parameter keys, or a save silently drops their state. In this
 * module [Sequential] is the only such container, and that is a TYPE fact, not
 * a review finding: `GRU` holds `Dense`s, `Conv2dWithSamePadding` holds a
 * `Conv2d` and `EmbeddingBag.WithOffsets` holds an `EmbeddingBag` — none of
 * those is `Stateful`, so none of them can lose state it does not have.
 * `ModelCheckpoint` states the same contract at the point a user meets it.
 */
interface Stateful<T : Stateful<T>> {
    val buffers: List<NamedParameter>
    fun withBuffers(updated: Map<String, DTensor<*, F32>>): T
}

/**
 * `AffineTransform(m, b)`: elementwise `m * x + b`, both parameters trainable
 * (DiffKT's BatchNorm frozen form). Same-shape elementwise at any
 * rank, exactly like the source.
 */
class AffineTransform(
    val m: DTensor<*, F32>,
    val b: DTensor<*, F32>,
) : TrainableLayer<AffineTransform> {

    override val parameters: List<NamedParameter> =
        listOf(NamedParameter("m", m), NamedParameter("b", b))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): AffineTransform {
        val unknown = updated.keys - setOf("m", "b")
        require(unknown.isEmpty()) { "AffineTransform.withParameters: unknown keys $unknown" }
        return AffineTransform(updated["m"] ?: m, updated["b"] ?: b)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        params["m"] * x + params["b"]
}

/**
 * `Sequential(layers)`: the fold of single-input layers (as in DiffKT).
 * Parameter keys are path-like, prefixed by the child's LIST INDEX (`"0.m"`,
 * `"1.b"`) — positional like DiffKT's `withTrainables` splice, so two children
 * of the same class never collide and non-trainable layers hold their index
 * without contributing keys. REJECTED: user-supplied layer names — DiffKT has
 * none, and optional names can layer on later without breaking the positional
 * scheme.
 */
class Sequential(val layers: List<Layer>) : TrainableLayer<Sequential>, Stateful<Sequential> {

    constructor(vararg layers: Layer) : this(layers.toList())

    override val parameters: List<NamedParameter> =
        layers.flatMapIndexed { i, layer ->
            if (layer is Trainable<*>) {
                layer.parameters.map { NamedParameter("$i.${it.key}", it.tensor) }
            } else emptyList()
        }

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Sequential {
        val perChild = HashMap<Int, MutableMap<String, DTensor<*, F32>>>()
        for ((key, tensor) in updated) {
            val dot = key.indexOf('.')
            val index = if (dot > 0) key.substring(0, dot).toIntOrNull() else null
            require(index != null && dot < key.length - 1 && index in layers.indices) {
                "Sequential.withParameters: key '$key' is not '<layerIndex>.<childKey>'"
            }
            perChild.getOrPut(index) { HashMap() }[key.substring(dot + 1)] = tensor
        }
        val rebuilt = layers.mapIndexed { i, layer ->
            val childUpdates = perChild[i] ?: return@mapIndexed layer
            require(layer is Trainable<*>) {
                "Sequential.withParameters: layer $i is not trainable but keys ${childUpdates.keys} target it"
            }
            layer.withParameters(childUpdates) as Layer
        }
        return Sequential(rebuilt)
    }

    /**
     * The [Stateful] half, keyed with the SAME `"<index>."` prefix
     * the parameters use, so a checkpoint's `param.3.gamma` and
     * `buffer.3.runningSum` name one layer.
     */
    override val buffers: List<NamedParameter> =
        layers.flatMapIndexed { i, layer ->
            if (layer is Stateful<*>) {
                layer.buffers.map { NamedParameter("$i.${it.key}", it.tensor) }
            } else emptyList()
        }

    override fun withBuffers(updated: Map<String, DTensor<*, F32>>): Sequential {
        val perChild = HashMap<Int, MutableMap<String, DTensor<*, F32>>>()
        for ((key, tensor) in updated) {
            val dot = key.indexOf('.')
            val index = if (dot > 0) key.substring(0, dot).toIntOrNull() else null
            require(index != null && dot < key.length - 1 && index in layers.indices) {
                "Sequential.withBuffers: key '$key' is not '<layerIndex>.<childKey>'"
            }
            perChild.getOrPut(index) { HashMap() }[key.substring(dot + 1)] = tensor
        }
        val rebuilt = layers.mapIndexed { i, layer ->
            val childUpdates = perChild[i] ?: return@mapIndexed layer
            require(layer is Stateful<*>) {
                "Sequential.withBuffers: layer $i is not stateful but keys ${childUpdates.keys} target it"
            }
            layer.withBuffers(childUpdates) as Layer
        }
        return Sequential(rebuilt)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        layers.foldIndexed(x) { i, acc, layer ->
            layer.forward(acc, Params { key -> params["$i.$key"] })
        }
}
