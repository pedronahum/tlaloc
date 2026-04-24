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
