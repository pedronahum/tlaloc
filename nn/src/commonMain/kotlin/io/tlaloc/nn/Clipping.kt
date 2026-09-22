/**
 * §0.4.502 (Tier 2 item 7) — **gradient clipping**, by global norm and by
 * value. A review named it as missing and it was: nothing in `:nn` stood
 * between a gradient and an optimizer.
 *
 * Clipping is a pure function on the gradient MAP, applied between
 * `CapturedStep.run` and `Optimizer.step`:
 *
 * ```kotlin
 * val r = step.run(model, listOf(x))
 * val g = GradientClipping.byGlobalNorm(r.gradients, maxNorm = 1f)
 * val (next, s) = opt.step(model, g, state)
 * ```
 *
 * REJECTED: clipping inside the optimizers. It is not an optimizer property —
 * the same clip applies to all four, PyTorch keeps it outside for the same
 * reason, and putting it in would have meant four copies and a constructor
 * argument on each. REJECTED: clipping inside the captured graph. The
 * gradients are already back on the host by the time they are keyed (the F3
 * contract: "pure host math on DTensor/FloatArray"), and a graph-level clip
 * would need a global reduction across every parameter tensor inside the
 * traced function — real work for the GPU lane, and a named deferral, not a
 * v1 requirement.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import kotlin.math.sqrt

object GradientClipping {

    /**
     * The 2-norm of every gradient entry taken together:
     * `√(Σ_over_all_keys Σ_over_elements g²)`.
     *
     * **Accumulated in Double.** The returned value is therefore the correctly
     * rounded norm of the f32 inputs. PyTorch's `clip_grad_norm_` computes a
     * per-tensor f32 norm and then the norm of those, which is a different
     * rounding of the same quantity; the two agree to about 1e-7 relative and
     * the cross-check against torch pins that. Summing in f32 to "match" would
     * have matched one particular accumulation order and lost precision on a
     * model with many small tensors.
     */
    fun globalNorm(gradients: Map<String, DTensor<*, F32>>): Float {
        var acc = 0.0
        for (g in gradients.values) {
            for (v in g.hostF32()) acc += v.toDouble() * v.toDouble()
        }
        return sqrt(acc).toFloat()
    }

    /**
     * Scale every gradient by `maxNorm / ‖g‖` when the global norm exceeds
     * [maxNorm], leaving it untouched otherwise. The DIRECTION of the combined
     * gradient is preserved exactly — that is what distinguishes this from
     * [byValue], which does not.
     *
     * The scale is the plain `maxNorm / norm`. PyTorch uses
     * `maxNorm / (norm + 1e-6)`, a guard against a zero norm that also shrinks
     * every clipped step by a factor of `norm / (norm + 1e-6)`. We take the
     * analytic ratio and handle the zero case by not clipping at all (a zero
     * norm cannot exceed a positive [maxNorm], so the branch is never reached),
     * which is exact rather than nearly exact. The divergence is ~1e-6 relative
     * and is recorded here because a user comparing trajectories with PyTorch
     * will see it.
     *
     * A NON-FINITE norm refuses BY NAME rather than scaling every gradient to
     * NaN: `maxNorm / inf` is 0 and `maxNorm / NaN` is NaN, so both silently
     * destroy the step, and a run whose gradients have gone non-finite needs to
     * know that, not to be clipped past it. (PyTorch offers the same behaviour
     * behind an opt-in flag, `error_if_nonfinite`; here it is the only
     * behaviour.)
     */
    fun byGlobalNorm(
        gradients: Map<String, DTensor<*, F32>>,
        maxNorm: Float,
    ): Map<String, DTensor<*, F32>> {
        require(maxNorm > 0f) {
            "GradientClipping.byGlobalNorm: maxNorm must be > 0 (got $maxNorm) — clipping to 0 " +
                "is spelled by not stepping"
        }
        val norm = globalNorm(gradients)
        require(norm.isFinite()) {
            "GradientClipping.byGlobalNorm: the global gradient norm is $norm. Scaling by " +
                "maxNorm/$norm would silently turn every gradient into " +
                "${if (norm.isNaN()) "NaN" else "zero"} and hide a diverged run; the loss and the " +
                "inputs of this step are what to look at (${gradients.size} gradient tensors)"
        }
        if (norm <= maxNorm) return gradients
        val scale = maxNorm / norm
        return gradients.mapValues { (_, g) ->
            val d = g.hostF32()
            tensorLike(g, FloatArray(d.size) { i -> d[i] * scale })
        }
    }

    /**
     * Clamp every gradient ELEMENT into `[minValue, maxValue]`.
     *
     * PyTorch's `clip_grad_value_`. Unlike [byGlobalNorm] this CHANGES THE
     * DIRECTION of the update — it is a per-coordinate operation — which is
     * why both exist and why a caller has to choose. Non-finite entries are
     * not refused here: `NaN.coerceIn` is NaN and an infinity clamps to the
     * bound, so this function cannot turn a good gradient into a bad one, and
     * a caller who wants the diagnosis calls [globalNorm] (or [byGlobalNorm])
     * for it.
     */
    fun byValue(
        gradients: Map<String, DTensor<*, F32>>,
        minValue: Float,
        maxValue: Float,
    ): Map<String, DTensor<*, F32>> {
        require(minValue <= maxValue) {
            "GradientClipping.byValue: minValue $minValue must be <= maxValue $maxValue"
        }
        require(minValue.isFinite() && maxValue.isFinite()) {
            "GradientClipping.byValue: the bounds must be finite (got [$minValue, $maxValue])"
        }
        return gradients.mapValues { (_, g) ->
            val d = g.hostF32()
            tensorLike(
                g,
                FloatArray(d.size) { i ->
                    val v = d[i]
                    // Written out rather than `coerceIn` so a NaN passes
                    // through unchanged instead of becoming a bound: a NaN is
                    // neither above nor below anything, and turning it into
                    // `maxValue` would be inventing a gradient.
                    if (v < minValue) minValue else if (v > maxValue) maxValue else v
                },
            )
        }
    }

    /** The symmetric spelling: clamp into `[-limit, limit]`. */
    fun byValue(
        gradients: Map<String, DTensor<*, F32>>,
        limit: Float,
    ): Map<String, DTensor<*, F32>> {
        require(limit > 0f) { "GradientClipping.byValue: limit must be > 0 (got $limit)" }
        return byValue(gradients, -limit, limit)
    }

    private fun tensorLike(g: DTensor<*, F32>, data: FloatArray): DTensor<*, F32> =
        DTensor<Shape, F32>(HostF32Storage(data), g.dims.copyOf(), F32)
}
