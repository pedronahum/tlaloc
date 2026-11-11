/**
 * Dropout in DiffKT's exact convention:
 * inverted dropout, scaled at TRAIN time — `mask[i] = rand() > p ? 1/(1−p) :
 * 0f`, `train: x·mask`, `inferenceMode: identity`. Randomness is threefry-keyed
 * and explicit: the mask is a pure function of
 * `(key, p, element count)` over the [uniformFloats] stream — bit-
 * deterministic, no global RNG anywhere.
 *
 * The mask enters the trace either as a CONSTANT leaf (`isConstant = true`: random draws
 * differentiate as constants) or, under `capture`, as a data input the reverse transform treats
 * as a constant parameter. The forward is one existing MUL; the transform hands back
 * `upstream · mask` through `MulRule` with zero new rules.
 */
package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.cast
import io.tlaloc.autograd.constantMatching
import io.tlaloc.autograd.times

/** The identity layer — Dropout's frozen inference form (DiffKT freezes to identity too). */
object IdentityLayer : Layer {
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = x
}

/**
 * `Dropout(p, key)`: [forward] is the TRAINING forward — each element is kept
 * with probability `1 − p` (`u > p` on the uniform draw, DiffKT's own
 * comparison) and scaled by `1/(1 − p)`, dropped to exact zero otherwise.
 * DiffKT's `Dropout(p)` takes its `random` per CALL; our [Layer.forward] has
 * no key slot, so the key is part of the layer VALUE and a training loop
 * re-keys per step with [withKey] (split a step key off the loop's root). Under `capture`
 * the mask is an input of the captured step, drawn from the model's current key at each
 * [CapturedStep.run], so re-keying needs no new capture; with `fixedDropoutMask = true`, and in
 * a container other than [Sequential], it is a constant of the graph. Not trainable: no
 * parameters, and the mask contributes no adjoint work.
 */
class Dropout(
    val p: Float,
    val key: RandomKey,
) : Layer {

    init {
        require(p >= 0f && p < 1f) { "Dropout: p=$p must lie in [0, 1)" }
    }

    /** The per-step re-key: same rate, new mask stream. */
    fun withKey(newKey: RandomKey): Dropout = Dropout(p, newKey)

    /** DiffKT's `inferenceMode`: dropout freezes to the identity. */
    fun inferenceMode(): Layer = IdentityLayer

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 1) {
            "Dropout: input must have rank >= 1 (got a scalar) — the mask constant is a tensor leaf"
        }
        val supplied = if (p > 0f) params.dropoutMask(MASK_KEY, x.dims) else null
        // In x's dtype, so the mask also multiplies a bf16 activation (Precision.MIXED_BF16).
        val mask: Tracer<Shape> = when {
            supplied == null -> x.constantMatching(maskValues(x.size), x.dims)
            supplied.dtype != x.dtype -> supplied.cast(x.dtype)
            else -> supplied
        }
        return x * mask
    }

    /** The mask for an input of [size] elements: `1/(1 − p)` where an element is kept, 0 where dropped. */
    fun maskValues(size: Int): FloatArray {
        val u = uniformFloats(key, size)
        val scale = 1f / (1f - p)
        return FloatArray(size) { if (u[it] > p) scale else 0f }
    }

    companion object {
        /** The key under which a capture supplies this layer's mask ([Params.dropoutMask]). */
        const val MASK_KEY: String = "dropoutMask"
    }
}
