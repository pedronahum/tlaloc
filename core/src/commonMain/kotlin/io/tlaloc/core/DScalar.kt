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
