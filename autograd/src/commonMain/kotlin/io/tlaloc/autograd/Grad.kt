package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32

private fun <S : Shape> gradTensor(tape: Tape, grads: Gradients, id: Int): DTensor<S, F32> {
    val entry = tape.entries[id]
    val data = grads.get(id) ?: FloatArray(entry.size)
    return DTensor(HostF32Storage(data), entry.dims.copyOf(), F32)
}

fun <S : Shape> valueAndGrad(
    f: (Tracer<S>) -> Tracer<ScalarShape>,
): (DTensor<S, F32>) -> Pair<Float, DTensor<S, F32>> = { input ->
    val tape = Tape()
    val x = tape.traceLeaf<S>(input)
    val out = f(x)
    require(out.rank == 0) { "valueAndGrad expects a scalar output, got rank ${out.rank}" }
    val value = out.entry.value[0]
    val grads = backward(tape, out.id, floatArrayOf(1f))
    value to gradTensor<S>(tape, grads, x.id)
}

fun <S : Shape> grad(
    f: (Tracer<S>) -> Tracer<ScalarShape>,
): (DTensor<S, F32>) -> DTensor<S, F32> {
    val vg = valueAndGrad(f)
    return { input -> vg(input).second }
}

fun <S1 : Shape, S2 : Shape> valueAndGrad2(
    f: (Tracer<S1>, Tracer<S2>) -> Tracer<ScalarShape>,
): (DTensor<S1, F32>, DTensor<S2, F32>) -> Triple<Float, DTensor<S1, F32>, DTensor<S2, F32>> =
    { a, b ->
        val tape = Tape()
        val ta = tape.traceLeaf<S1>(a)
        val tb = tape.traceLeaf<S2>(b)
        val out = f(ta, tb)
        require(out.rank == 0) { "valueAndGrad2 expects scalar output, got rank ${out.rank}" }
        val value = out.entry.value[0]
        val grads = backward(tape, out.id, floatArrayOf(1f))
        Triple(value, gradTensor<S1>(tape, grads, ta.id), gradTensor<S2>(tape, grads, tb.id))
    }

fun <S1 : Shape, S2 : Shape> grad2(
    f: (Tracer<S1>, Tracer<S2>) -> Tracer<ScalarShape>,
): (DTensor<S1, F32>, DTensor<S2, F32>) -> Pair<DTensor<S1, F32>, DTensor<S2, F32>> {
    val vg = valueAndGrad2(f)
    return { a, b -> val t = vg(a, b); t.second to t.third }
}

/**
 * §0.4.134 — value-and-gradient for a 3-tensor function `f: (S1, S2, S3) → scalar`.
 * Returns a 4-tuple `(value, dA, dB, dC)` via [Quadruple] — Kotlin's stdlib stops
 * at [Triple], so the 3-input variant introduces a small named 4-tuple parallel to
 * how [valueAndGrad2] reused stdlib [Triple] for its 3-tuple result. Mechanics
 * mirror [valueAndGrad2]: trace each input as a tape leaf, evaluate the lambda,
 * require a scalar output, run reverse mode with seed `1f`, and unpack the
 * gradients per leaf.
 */
fun <S1 : Shape, S2 : Shape, S3 : Shape> valueAndGrad3(
    f: (Tracer<S1>, Tracer<S2>, Tracer<S3>) -> Tracer<ScalarShape>,
): (DTensor<S1, F32>, DTensor<S2, F32>, DTensor<S3, F32>) -> Quadruple<Float, DTensor<S1, F32>, DTensor<S2, F32>, DTensor<S3, F32>> =
    { a, b, c ->
        val tape = Tape()
        val ta = tape.traceLeaf<S1>(a)
        val tb = tape.traceLeaf<S2>(b)
        val tc = tape.traceLeaf<S3>(c)
        val out = f(ta, tb, tc)
        require(out.rank == 0) { "valueAndGrad3 expects scalar output, got rank ${out.rank}" }
        val value = out.entry.value[0]
        val grads = backward(tape, out.id, floatArrayOf(1f))
        Quadruple(
            value,
            gradTensor<S1>(tape, grads, ta.id),
            gradTensor<S2>(tape, grads, tb.id),
            gradTensor<S3>(tape, grads, tc.id),
        )
    }

/**
 * §0.4.134 — gradient-only convenience for a 3-tensor scalar-valued function.
 * Returns a [Triple] of the per-input gradients (drops the primal value). For the
 * primal value alongside the gradients use [valueAndGrad3].
 */
fun <S1 : Shape, S2 : Shape, S3 : Shape> grad3(
    f: (Tracer<S1>, Tracer<S2>, Tracer<S3>) -> Tracer<ScalarShape>,
): (DTensor<S1, F32>, DTensor<S2, F32>, DTensor<S3, F32>) -> Triple<DTensor<S1, F32>, DTensor<S2, F32>, DTensor<S3, F32>> {
    val vg = valueAndGrad3(f)
    return { a, b, c -> val q = vg(a, b, c); Triple(q.second, q.third, q.fourth) }
}

// §0.4.81 — (DTensor, Float) convenience overloads. Internally wrap the Float
// scalar as a `Tensors.f32Scalar(f)`-style DTensor, route through the existing
// `valueAndGrad2` / `grad2`, and unwrap the scalar gradient back to Float at
// the return boundary. Lets users write
//
//   val g = gradWithScalar { x: Tracer<Rank1<Sym>>, lr: Tracer<ScalarShape> -> ... }
//   val (dx, dLr) = g(vectorInput, 0.01f)
//
// without reaching for `Tensors.f32Scalar(0.01f)` every call. The underlying
// mechanics are unchanged: §0.4.77's BroadcastRule handles the reverse for
// scalar operands that got broadcast to the first operand's shape.
//
// Distinct Kotlin name (`gradWithScalar`) rather than an overload on `grad2`
// because Kotlin's type inference would see `(Tracer<S>, Tracer<ScalarShape>)
// -> Tracer<ScalarShape>` as compatible with both the generic
// `(Tracer<S1>, Tracer<S2>) -> Tracer<ScalarShape>` (with S2 = ScalarShape)
// and the new mixed variant — producing overload ambiguity. A different name
// keeps each surface unambiguous.

@Suppress("UNCHECKED_CAST")
fun <S : Shape> valueAndGradWithScalar(
    f: (Tracer<S>, Tracer<ScalarShape>) -> Tracer<ScalarShape>,
): (DTensor<S, F32>, Float) -> Triple<Float, DTensor<S, F32>, Float> = { a, bFloat ->
    val tape = Tape()
    val ta = tape.traceLeaf<S>(a)
    val bTensor = DTensor<ScalarShape, F32>(
        HostF32Storage(floatArrayOf(bFloat)),
        IntArray(0),
        F32,
    )
    val tb = tape.traceLeaf<ScalarShape>(bTensor)
    val out = f(ta, tb)
    require(out.rank == 0) { "valueAndGradWithScalar expects scalar output, got rank ${out.rank}" }
    val value = out.entry.value[0]
    val grads = backward(tape, out.id, floatArrayOf(1f))
    val dScalar = gradTensor<ScalarShape>(tape, grads, tb.id).hostF32()[0]
    Triple(value, gradTensor<S>(tape, grads, ta.id), dScalar)
}

fun <S : Shape> gradWithScalar(
    f: (Tracer<S>, Tracer<ScalarShape>) -> Tracer<ScalarShape>,
): (DTensor<S, F32>, Float) -> Pair<DTensor<S, F32>, Float> {
    val vg = valueAndGradWithScalar(f)
    return { a, b -> val t = vg(a, b); t.second to t.third }
}

// §0.4.82 — pure-scalar convenience. Both operands are Floats; both gradients
// are Floats. Wraps each as a `Tensors.f32Scalar`-style DTensor internally.
// Target use: scalar calculus playgrounds / tight numeric experiments where
// DTensor wrapping/unwrapping at the boundary is noise.

fun valueAndGradWithScalars(
    f: (Tracer<ScalarShape>, Tracer<ScalarShape>) -> Tracer<ScalarShape>,
): (Float, Float) -> Triple<Float, Float, Float> = { aFloat, bFloat ->
    val tape = Tape()
    val aTensor = DTensor<ScalarShape, F32>(HostF32Storage(floatArrayOf(aFloat)), IntArray(0), F32)
    val bTensor = DTensor<ScalarShape, F32>(HostF32Storage(floatArrayOf(bFloat)), IntArray(0), F32)
    val ta = tape.traceLeaf<ScalarShape>(aTensor)
    val tb = tape.traceLeaf<ScalarShape>(bTensor)
    val out = f(ta, tb)
    require(out.rank == 0) { "valueAndGradWithScalars expects scalar output, got rank ${out.rank}" }
    val value = out.entry.value[0]
    val grads = backward(tape, out.id, floatArrayOf(1f))
    val da = gradTensor<ScalarShape>(tape, grads, ta.id).hostF32()[0]
    val db = gradTensor<ScalarShape>(tape, grads, tb.id).hostF32()[0]
    Triple(value, da, db)
}

fun gradWithScalars(
    f: (Tracer<ScalarShape>, Tracer<ScalarShape>) -> Tracer<ScalarShape>,
): (Float, Float) -> Pair<Float, Float> {
    val vg = valueAndGradWithScalars(f)
    return { a, b -> val t = vg(a, b); t.second to t.third }
}

/**
 * §0.4.134 — generic 4-tuple. Kotlin's stdlib stops at [Triple]; [valueAndGrad3]
 * needs a 4-slot return type for `(value, dA, dB, dC)`. Equivalent to [Pair]
 * and [Triple] in shape — destructurable, and componentN-returning.
 */
data class Quadruple<out A, out B, out C, out D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
) {
    override fun toString(): String = "($first, $second, $third, $fourth)"
}
