package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF64
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random

// The F64 form of the customVjp oracle. Its own file because it erases to the JVM
// signature of the F32 `checkCustomVjp`.

/** The result of the F64 [checkCustomVjp]: [CustomVjpCheck] in Double. */
data class CustomVjpCheckF64(
    /** `⟨ȳ, J·v⟩` — the forward side, from central differences of `f`. */
    val forwardInner: Double,
    /** `⟨vjpFn(ȳ, at), v⟩` — the reverse side, from the user's adjoint. */
    val vjpInner: Double,
    /** `|forwardInner − vjpInner| / max(1, |forwardInner|, |vjpInner|)`. */
    val relativeError: Double,
    /** `relativeError <= tolerance`. */
    val passed: Boolean,
)

/**
 * [checkCustomVjp] for a pair over F64 tensors, computed in Double. The defaults are set
 * for double precision: the central difference with `eps = 1e-5` is good to about 1e-10
 * relative for O(1) inputs, so `tolerance = 1e-8` separates an exact adjoint from one that
 * is off by 1e-6 relative, which the F32 check (tolerance 1e-2) cannot.
 */
fun <SA : Shape, SR : Shape> checkCustomVjp(
    f: (DTensor<SA, F64>) -> DTensor<SR, F64>,
    vjpFn: (DTensor<SR, F64>, DTensor<SA, F64>) -> DTensor<SA, F64>,
    at: DTensor<SA, F64>,
    seed: Int = 0,
    eps: Double = 1e-5,
    tolerance: Double = 1e-8,
): CustomVjpCheckF64 {
    val rnd = Random(seed)
    val x = at.hostF64()
    val v = DoubleArray(x.size) { rnd.nextDouble() * 2 - 1 }

    fun perturbed(sign: Double): DTensor<SA, F64> =
        DTensor(HostF64Storage(DoubleArray(x.size) { x[it] + sign * eps * v[it] }), at.dims.copyOf(), F64)

    val yPlus = f(perturbed(+1.0))
    val yp = yPlus.hostF64()
    val ym = f(perturbed(-1.0)).hostF64()
    require(yp.size == ym.size) {
        "checkCustomVjp: f returned differing sizes (${yp.size} vs ${ym.size}) at x ± eps·v"
    }
    val jv = DoubleArray(yp.size) { (yp[it] - ym[it]) / (2 * eps) }
    val yBarData = DoubleArray(yp.size) { rnd.nextDouble() * 2 - 1 }
    var forwardInner = 0.0
    for (i in jv.indices) forwardInner += yBarData[i] * jv[i]

    val yBar = DTensor<SR, F64>(HostF64Storage(yBarData.copyOf()), yPlus.dims.copyOf(), F64)
    val xBar = vjpFn(yBar, at).hostF64()
    require(xBar.size == x.size) {
        "checkCustomVjp: vjpFn returned size ${xBar.size} for an input of size ${x.size} — " +
            "the VJP shape contract (d_x must match x's shape) is violated"
    }
    var vjpInner = 0.0
    for (i in xBar.indices) vjpInner += xBar[i] * v[i]

    val relativeError = abs(forwardInner - vjpInner) / max(1.0, max(abs(forwardInner), abs(vjpInner)))
    return CustomVjpCheckF64(forwardInner, vjpInner, relativeError, relativeError <= tolerance)
}
