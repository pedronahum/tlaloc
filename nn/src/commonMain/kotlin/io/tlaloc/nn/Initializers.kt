/**
 * The initializer family over the `:core` threefry
 * streams. Pure host math, bit-deterministic per key (no global
 * RNG anywhere), zero tape involvement — an initializer's output is an
 * ordinary parameter tensor the capture step later traces as a leaf.
 *
 * DiffKT ships: `uniform(min, max)`, `gaussian(mean, variance)`,
 * and `kaimingUniform(fanMode, activationGainFactor)` with
 * `bound = sqrt(3/fan) · gain` over `fan = fanSize(shape) · Π shape[2:]`,
 * `FanIn.fanSize = shape[1]`, `FanOut.fanSize = shape[0]`. Those three are the
 * parity surface. The Kaiming-normal and Xavier/Glorot pair are the standard
 * He/Glorot formulas (PyTorch's `kaiming_normal_` / `xavier_{uniform,normal}_`)
 * — NOT in DiffKT, recorded as Tlaloc extensions, oracled against the formulas
 * themselves, never against DiffKT.
 *
 * Draw substrate: `uniformFloats` / `normalFloats` (`:core`'s bit-pinned
 * threefry streams). Every affine rescale below is spelled once and pinned bit-exact in
 * the tests by replaying the identical expression over the same key.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.normalFloats
import io.tlaloc.core.uniformFloats
import kotlin.math.sqrt

/** DiffKT's `FanMode`: which axis counts as the fan for the Kaiming family. */
enum class FanMode { FanIn, FanOut }

/**
 * The fan, exactly as DiffKT computes it: `fanSize(dims) · Π dims[2:]` with
 * `FanIn.fanSize = dims[1]` and `FanOut.fanSize = dims[0]` — the PyTorch
 * formula on the layout at hand. Rank ≥ 2 required (a fan needs both an input
 * and an output axis).
 */
fun fanOf(dims: IntArray, mode: FanMode): Int {
    require(dims.size >= 2) { "fanOf: rank >= 2 required (got dims ${dims.toList()})" }
    val fanSize = if (mode == FanMode.FanIn) dims[1] else dims[0]
    var receptive = 1
    for (i in 2 until dims.size) receptive *= dims[i]
    return fanSize * receptive
}

/**
 * DiffKT's activation gain constants: Linear/Conv/Sigmoid `1`,
 * Tanh `5/3`, Relu `√2`, LeakyRelu(slope) `√(2/(1+slope²))` — so
 * `LeakyRelu(0)` = Relu's gain and `LeakyRelu(1)` = 1 exactly.
 */
sealed class ActivationGain(val gain: Float) {
    object Linear : ActivationGain(1f)
    object Conv : ActivationGain(1f)
    object Sigmoid : ActivationGain(1f)
    object Tanh : ActivationGain(5f / 3f)
    object Relu : ActivationGain(sqrt(2f))
    class LeakyRelu(val negativeSlope: Float) :
        ActivationGain(sqrt(2f / (1f + negativeSlope * negativeSlope)))
}

private fun checkedSize(dims: IntArray): Int {
    require(dims.isNotEmpty() && dims.all { it > 0 }) {
        "initializer dims must be non-empty and positive (got ${dims.toList()})"
    }
    return dims.fold(1) { acc, d -> acc * d }
}

private fun tensorOf(data: FloatArray, dims: IntArray): DTensor<*, F32> =
    DTensor<Shape, F32>(HostF32Storage(data), dims.copyOf(), F32)

/**
 * DiffKT `uniform(min, max)`: one uniform draw per element, rescaled
 * `u · (max − min) + min` over `u = uniformFloats(key, n)` — bit-deterministic
 * given [key]. The rescale spelling here is THE spelling: the Kaiming/Xavier
 * uniform variants delegate so every uniform initializer shares one bit
 * pattern per (key, bound).
 */
fun uniformInit(key: RandomKey, dims: IntArray, min: Float = 0f, max: Float = 1f): DTensor<*, F32> {
    require(min < max) { "uniformInit: min=$min must be < max=$max" }
    val n = checkedSize(dims)
    val u = uniformFloats(key, n)
    return tensorOf(FloatArray(n) { u[it] * (max - min) + min }, dims)
}

/**
 * DiffKT `gaussian(mean, variance)`: `z · sqrt(variance) + mean` per element
 * over `z = normalFloats(key, n)` — the same affine spelling as the source
 * (which scales `nextGaussian()` by `sqrt(variance)`), on the `:core` Box-Muller
 * stream instead of `java.util.Random`.
 */
fun gaussianInit(key: RandomKey, dims: IntArray, mean: Float = 0f, variance: Float = 1f): DTensor<*, F32> {
    require(variance >= 0f) { "gaussianInit: variance=$variance must be >= 0" }
    val n = checkedSize(dims)
    val z = normalFloats(key, n)
    val scale = sqrt(variance)
    return tensorOf(FloatArray(n) { z[it] * scale + mean }, dims)
}

/**
 * DiffKT `kaimingUniform(fanMode, activationGainFactor)` — He-uniform:
 * `bound = sqrt(3/fan) · gain`, uniform in `[−bound, bound)`. The DiffKT conv
 * default is `kaimingUniform(FanIn, LeakyRelu(sqrt(5)))`; the defaults here
 * (`FanIn`, `Linear`) are the neutral spelling — callers wanting the conv
 * default pass it explicitly, as DiffKT's Conv2d does.
 */
fun kaimingUniformInit(
    key: RandomKey,
    dims: IntArray,
    fanMode: FanMode = FanMode.FanIn,
    gain: ActivationGain = ActivationGain.Linear,
): DTensor<*, F32> {
    val bound = sqrt(3f / fanOf(dims, fanMode)) * gain.gain
    return uniformInit(key, dims, -bound, bound)
}

/**
 * He-normal (NOT in DiffKT — the PyTorch `kaiming_normal_` formula, a recorded
 * Tlaloc extension): `std = gain / sqrt(fan)`, `z · std` per element.
 */
fun kaimingNormalInit(
    key: RandomKey,
    dims: IntArray,
    fanMode: FanMode = FanMode.FanIn,
    gain: ActivationGain = ActivationGain.Linear,
): DTensor<*, F32> {
    val n = checkedSize(dims)
    val std = gain.gain / sqrt(fanOf(dims, fanMode).toFloat())
    val z = normalFloats(key, n)
    return tensorOf(FloatArray(n) { z[it] * std }, dims)
}

/**
 * Xavier/Glorot uniform (NOT in DiffKT — the standard Glorot formula, a
 * recorded Tlaloc extension): `bound = gain · sqrt(6/(fanIn + fanOut))`,
 * uniform in `[−bound, bound)`.
 */
fun xavierUniformInit(
    key: RandomKey,
    dims: IntArray,
    gain: ActivationGain = ActivationGain.Linear,
): DTensor<*, F32> {
    val fanSum = fanOf(dims, FanMode.FanIn) + fanOf(dims, FanMode.FanOut)
    val bound = gain.gain * sqrt(6f / fanSum)
    return uniformInit(key, dims, -bound, bound)
}

/**
 * Xavier/Glorot normal (NOT in DiffKT): `std = gain · sqrt(2/(fanIn + fanOut))`,
 * `z · std` per element.
 */
fun xavierNormalInit(
    key: RandomKey,
    dims: IntArray,
    gain: ActivationGain = ActivationGain.Linear,
): DTensor<*, F32> {
    val n = checkedSize(dims)
    val fanSum = fanOf(dims, FanMode.FanIn) + fanOf(dims, FanMode.FanOut)
    val std = gain.gain * sqrt(2f / fanSum)
    val z = normalFloats(key, n)
    return tensorOf(FloatArray(n) { z[it] * std }, dims)
}
