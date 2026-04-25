package io.tlaloc.core

sealed interface DScalar {
    fun toFloat(): Float
    fun toDouble(): Double
}

@JvmInline
value class FloatScalar(val v: Float) : DScalar {
    override fun toFloat(): Float = v
    override fun toDouble(): Double = v.toDouble()
}

@JvmInline
value class DoubleScalar(val v: Double) : DScalar {
    override fun toFloat(): Float = v.toFloat()
    override fun toDouble(): Double = v
}

operator fun FloatScalar.plus(other: FloatScalar): FloatScalar = FloatScalar(v + other.v)
operator fun FloatScalar.minus(other: FloatScalar): FloatScalar = FloatScalar(v - other.v)
operator fun FloatScalar.times(other: FloatScalar): FloatScalar = FloatScalar(v * other.v)
operator fun FloatScalar.div(other: FloatScalar): FloatScalar = FloatScalar(v / other.v)
operator fun FloatScalar.unaryMinus(): FloatScalar = FloatScalar(-v)

operator fun DoubleScalar.plus(other: DoubleScalar): DoubleScalar = DoubleScalar(v + other.v)
operator fun DoubleScalar.minus(other: DoubleScalar): DoubleScalar = DoubleScalar(v - other.v)
operator fun DoubleScalar.times(other: DoubleScalar): DoubleScalar = DoubleScalar(v * other.v)
operator fun DoubleScalar.div(other: DoubleScalar): DoubleScalar = DoubleScalar(v / other.v)
operator fun DoubleScalar.unaryMinus(): DoubleScalar = DoubleScalar(-v)

private fun DScalar.promote(other: DScalar): Pair<Double, Double> {
    val a = if (this is DoubleScalar) v else toDouble()
    val b = if (other is DoubleScalar) other.v else other.toDouble()
    return a to b
}

private fun pickResult(a: DScalar, b: DScalar, result: Double): DScalar =
    if (a is DoubleScalar || b is DoubleScalar) DoubleScalar(result) else FloatScalar(result.toFloat())

operator fun DScalar.plus(other: DScalar): DScalar {
    val (a, b) = promote(other); return pickResult(this, other, a + b)
}

operator fun DScalar.minus(other: DScalar): DScalar {
    val (a, b) = promote(other); return pickResult(this, other, a - b)
}

operator fun DScalar.times(other: DScalar): DScalar {
    val (a, b) = promote(other); return pickResult(this, other, a * b)
}

operator fun DScalar.div(other: DScalar): DScalar {
    val (a, b) = promote(other); return pickResult(this, other, a / b)
}

operator fun DScalar.unaryMinus(): DScalar = when (this) {
    is FloatScalar -> FloatScalar(-v)
    is DoubleScalar -> DoubleScalar(-v)
}

// --- Scalar relu entries ---
//
// Primitive-scalar ReLU extensions live here rather than being synthesised from
// `if (x > 0) x else 0` on the FIR side. Compiling `grad { x -> x.relu() }` maps the
// receiver-only call at `io.tlaloc.core.relu` to [OpKind.RELU], which lets the
// registered [ReluRule] pick it up and emit `upstream * STEP(x)` on the gradient side.
fun Float.relu(): Float = if (this > 0f) this else 0f
fun Double.relu(): Double = if (this > 0.0) this else 0.0
fun FloatScalar.relu(): FloatScalar = FloatScalar(if (v > 0f) v else 0f)
fun DoubleScalar.relu(): DoubleScalar = DoubleScalar(if (v > 0.0) v else 0.0)
fun DScalar.relu(): DScalar = when (this) {
    is FloatScalar -> relu()
    is DoubleScalar -> relu()
}

// --- Scalar sqrt entries ---
//
// Scalar sqrt extensions mirror the relu pattern. `Float.sqrt()` / `Double.sqrt()`
// resolve at FQN `io.tlaloc.core.sqrt`; the FIR lowering maps that to [OpKind.SQRT],
// the registered [SqrtRule] picks it up on the gradient side (emitting `upstream /
// (2 * sqrt(x))`), and [DxirToIrSynthesis.irSqrt] emits an IrCall back to this
// extension to close the compile path end-to-end. Needed by scalar brachistochrone-
// style benchmarks (Stage D.1b) that use `sqrt(...)` inside a for-loop body.
fun Float.sqrt(): Float = kotlin.math.sqrt(this.toDouble()).toFloat()
fun Double.sqrt(): Double = kotlin.math.sqrt(this)
fun FloatScalar.sqrt(): FloatScalar = FloatScalar(v.sqrt())
fun DoubleScalar.sqrt(): DoubleScalar = DoubleScalar(v.sqrt())
fun DScalar.sqrt(): DScalar = when (this) {
    is FloatScalar -> sqrt()
    is DoubleScalar -> sqrt()
}

// --- Scalar exp / log entries (§0.4.158) ---
//
// Scalar exp/log extensions mirror the sqrt pattern. `Float.exp()` / `Float.log()` /
// `Double.exp()` / `Double.log()` resolve at FQN `io.tlaloc.core.exp` /
// `io.tlaloc.core.log`; the FIR lowering maps each to `OpKind.EXP` / `OpKind.LOG`,
// and the registered `ExpRule` / `LogRule` cover the gradient side. Synthesis was
// already wired (`DxirToIrSynthesis.irExp` / `irLog` ship since §0.4.53). Needed by
// HMC's logistic-regression port (`docs/HMC_PORT_PLAN.md` Phase 1) which writes
// `(1.0f + (-xb).exp()).log()` element-by-element inside a scalar/loop body.
fun Float.exp(): Float = kotlin.math.exp(this.toDouble()).toFloat()
fun Double.exp(): Double = kotlin.math.exp(this)
fun FloatScalar.exp(): FloatScalar = FloatScalar(v.exp())
fun DoubleScalar.exp(): DoubleScalar = DoubleScalar(v.exp())
fun DScalar.exp(): DScalar = when (this) {
    is FloatScalar -> exp()
    is DoubleScalar -> exp()
}
fun Float.log(): Float = kotlin.math.ln(this.toDouble()).toFloat()
fun Double.log(): Double = kotlin.math.ln(this)
fun FloatScalar.log(): FloatScalar = FloatScalar(v.log())
fun DoubleScalar.log(): DoubleScalar = DoubleScalar(v.log())
fun DScalar.log(): DScalar = when (this) {
    is FloatScalar -> log()
    is DoubleScalar -> log()
}
