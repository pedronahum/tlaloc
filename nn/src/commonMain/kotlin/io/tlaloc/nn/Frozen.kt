/**
 * Frozen parameters: parameters that a training step reads but does not
 * train. A [Frozen] value selects them by key; [capture] with `frozen` emits
 * no gradient for them (the reverse transform drops their adjoint work), and
 * [Optimizer.step] with `frozen` neither updates them nor creates optimizer
 * state for them.
 *
 * Keys are the model's [Trainable.parameters] keys (`"blocks.3.attn.q.w"`).
 * A submodule is frozen by its key prefix: [Frozen.prefixes]`("embed",
 * "blocks.0")` freezes `embed.table` and everything under `blocks.0.`.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32

/** A selection of parameter keys that training leaves unchanged. */
class Frozen private constructor(
    private val predicate: (String) -> Boolean,
    private val description: String,
) {
    /** True when the parameter with [key] is frozen. */
    fun isFrozen(key: String): Boolean = predicate(key)

    /** The parameters of [model] that stay trainable, in model order. */
    fun trainable(model: Trainable<*>): List<NamedParameter> = model.parameters.filter { !isFrozen(it.key) }

    /** The parameters of [model] that are frozen, in model order. */
    fun frozen(model: Trainable<*>): List<NamedParameter> = model.parameters.filter { isFrozen(it.key) }

    /** Frozen if either selection freezes it. */
    operator fun plus(other: Frozen): Frozen =
        Frozen({ isFrozen(it) || other.isFrozen(it) }, "$description + ${other.description}")

    override fun toString(): String = "Frozen($description)"

    companion object {
        /** Nothing frozen: training behaves as without a selection. */
        @JvmField
        val NONE: Frozen = Frozen({ false }, "none")

        /** Exactly these keys. */
        @JvmStatic
        fun keys(vararg keys: String): Frozen {
            val set = keys.toSet()
            return Frozen({ it in set }, "keys ${set.sorted()}")
        }

        /**
         * Every key equal to one of [prefixes] or under it: `"blocks.0"` freezes
         * `blocks.0.attn.q.w` but not `blocks.10.attn.q.w`.
         */
        @JvmStatic
        fun prefixes(vararg prefixes: String): Frozen {
            val list = prefixes.map { it.trimEnd('.') }
            require(list.none { it.isEmpty() }) { "Frozen.prefixes: an empty prefix would freeze every parameter; use allExcept" }
            return Frozen({ key -> list.any { key == it || key.startsWith("$it.") } }, "prefixes $list")
        }

        /** Every key [predicate] accepts. */
        @JvmStatic
        fun matching(predicate: (String) -> Boolean): Frozen = Frozen(predicate, "matching")

        /** Every key except those [trainable] accepts. */
        @JvmStatic
        fun allExcept(trainable: (String) -> Boolean): Frozen = Frozen({ !trainable(it) }, "all except")
    }
}

/**
 * [step] for a model with [frozen] parameters: only the trainable parameters
 * are stepped, so the optimizer state holds slots for them alone, and the
 * frozen tensors in the returned model are the same objects as in [model].
 *
 * [grads] must hold a gradient for every trainable parameter (as the
 * optimizer requires) and none for a frozen one: a gradient for a frozen key
 * means the caller and the capture disagree on what is frozen.
 */
fun <M, S> Optimizer<S>.step(
    model: M,
    grads: Map<String, DTensor<*, F32>>,
    state: S,
    frozen: Frozen,
): Pair<M, S> where M : Trainable<M> {
    val stray = grads.keys.filter { frozen.isFrozen(it) }
    require(stray.isEmpty()) {
        "optimizer step: gradients given for frozen parameters $stray ($frozen)"
    }
    val out = step(frozen.trainable(model), grads, state)
    return model.withParameters(out.params) to out.state
}
