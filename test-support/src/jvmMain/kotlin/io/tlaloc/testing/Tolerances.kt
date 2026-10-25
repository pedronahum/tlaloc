package io.tlaloc.testing

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.OpKind
import kotlin.math.abs

/**
 * The tolerances comparisons use, by surface and dtype, with per-op overrides. Each
 * value is a relative error scaled by `max(1, max |expected|)` ([scaledError]).
 * docs/NUMERICAL_PRECISION.md gives the reasons for the defaults.
 *
 * Every value is positive (checked at load). Setting the system property or
 * environment variable `TLALOC_STRICT_TOLERANCES=1` ignores the overrides, to see
 * which ops still need theirs.
 */
object Tolerances {

    /** The default per (surface, dtype). */
    private val defaults: Map<Pair<Surface, DType>, Double> = mapOf(
        // One F32 op against the same op in F64 on F32-rounded inputs: a few ulps of
        // the result, more where the op sums or factorizes.
        (Surface.INTERPRETER_F64 to F32) to 2e-5,
        // A PJRT GPU against the interpreter: XLA reassociates sums and fuses, ~4e-5
        // on the GB10 for elementwise and reductions.
        (Surface.DEVICE to F32) to 2e-4,
        (Surface.DEVICE to F64) to 1e-10,
    )

    /**
     * Matrix products on a GPU run at TF32 (10-bit mantissa) for F32: plain `MATMUL`
     * emits no precision, and XLA's default is TF32 on the GB10.
     */
    const val TF32_DOT: Double = 1e-2

    /** Per-op overrides, each with its reason. */
    data class Override(val tolerance: Double, val reason: String)

    private val overrides: Map<Triple<OpKind, Surface, DType>, Override> = mapOf(
        Triple(OpKind.POLYGAMMA, Surface.INTERPRETER_F64, F32) to
            Override(2e-4, "the series loses digits at higher orders in F32"),
        Triple(OpKind.TRIGAMMA, Surface.INTERPRETER_F64, F32) to
            Override(1e-4, "the recurrence and series sum many F32 terms"),
        Triple(OpKind.EIGH_V, Surface.INTERPRETER_F64, F32) to
            Override(1e-3, "eigenvectors are as accurate as the eigenvalue gaps allow"),
        Triple(OpKind.EIGH_V, Surface.DEVICE, F32) to
            Override(1e-3, "eigenvectors are as accurate as the eigenvalue gaps allow"),
    )

    init {
        for ((k, v) in defaults) require(v > 0) { "Tolerances: default for $k is $v" }
        for ((k, v) in overrides) require(v.tolerance > 0 && v.reason.isNotBlank()) { "Tolerances: override for $k" }
    }

    val strict: Boolean
        get() = (System.getProperty("TLALOC_STRICT_TOLERANCES") ?: System.getenv("TLALOC_STRICT_TOLERANCES")) == "1"

    /** The tolerance for [kind] on [surface] at [dtype]; [usesDot] for graphs with matrix products. */
    fun relative(kind: OpKind?, dtype: DType, surface: Surface, usesDot: Boolean = false): Double {
        val base = defaults[surface to dtype] ?: error("Tolerances: no default for $surface at $dtype")
        val dot = if (usesDot && surface == Surface.DEVICE && dtype == F32) TF32_DOT else 0.0
        val override = if (strict || kind == null) null else overrides[Triple(kind, surface, dtype)]?.tolerance
        return maxOf(base, dot, override ?: 0.0)
    }

    fun relative(sample: OpSample, dtype: DType, surface: Surface): Double =
        relative(OpCatalogue.kindOf(sample), dtype, surface, sample.usesDot)

    /**
     * The tightest absolute tolerance (over entries with `|expected| ≤ 1`) and relative
     * tolerance (over the rest) under which [got] matches [want], times 1.2: what a
     * failure message suggests.
     */
    fun suggest(got: DoubleArray, want: DoubleArray): Pair<Double, Double> {
        var atol = 0.0
        var rtol = 0.0
        for (i in got.indices) {
            val d = abs(got[i] - want[i])
            if (abs(want[i]) <= 1.0) atol = maxOf(atol, d) else rtol = maxOf(rtol, d / abs(want[i]))
        }
        return 1.2 * atol to 1.2 * rtol
    }

    /**
     * Throws when [got] differs from [want] by more than [tolerance] (scaled as
     * [scaledError]); the message names the worst entry and the tolerances that would pass.
     */
    fun assertClose(got: DoubleArray, want: DoubleArray, tolerance: Double, what: String) {
        val err = scaledError(got, want)
        if (err <= tolerance) return
        val worst = got.indices.maxBy { abs(got[it] - want[it]) }
        val (atol, rtol) = suggest(got, want)
        throw SweepFailure(
            "$what: scaled error ${"%.3g".format(err)} > ${"%.3g".format(tolerance)}; worst at [$worst] " +
                "${got[worst]} against ${want[worst]}; would pass with atol ${"%.3g".format(atol)} " +
                "(|expected| ≤ 1) and rtol ${"%.3g".format(rtol)} (the rest)",
        )
    }

    fun assertClose(got: FloatArray, want: FloatArray, tolerance: Double, what: String) =
        assertClose(DoubleArray(got.size) { got[it].toDouble() }, DoubleArray(want.size) { want[it].toDouble() }, tolerance, what)

    /** The dtypes [relative] has defaults for. */
    val dtypes: Set<DType> get() = defaults.keys.map { it.second }.toSet() + setOf(F32, F64)
}
