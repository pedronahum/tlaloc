package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform

/*
 * §0.4.446 — ONE AD ENGINE. The Tracer-convenience API below used to run its own
 * runtime value-tape reverse walk (`Backward.kt`, deleted in the same commit). Now
 * every entry point routes through the COMPILER's AD, exactly like `grad {}`
 * intrinsics and the `:nn` model layer (§0.4.437):
 *
 *   trace (Tape) → [Tape.toDxirFunction] → [DxirReverseTransform] (includeForward)
 *                → [DxirInterpreter.evalFunction]
 *
 * The signatures are unchanged; only the engine underneath moved. The tape remains
 * purely a TRACING structure (eager forward values + op recording) — it no longer
 * walks itself backwards. Gradient math lives in one place, the IR-side
 * `VjpRegistry`, consumed by one transform.
 */

/**
 * The shared spine: reproduce the tape as a [io.tlaloc.ir.DxirFunction] whose
 * params are [leafIds] (in order) and whose single return is [outId], apply the
 * reverse transform with `includeForward = true` — the gradient function's
 * outputs are `(value, *grads)`, one grad per param positionally — and evaluate
 * it on the tape's cached leaf values via the interpreter.
 */
private fun reverseThroughCompiler(
    tape: Tape,
    leafIds: List<Int>,
    outId: Int,
    name: String,
): List<FloatArray> {
    val primal = tape.toDxirFunction(name, paramIds = leafIds, returnIds = listOf(outId))
    val gradient = DxirReverseTransform.apply(primal, includeForward = true)
    val inputs = leafIds.map { tape.entries[it].value }
    return DxirInterpreter.evalFunction(gradient, inputs)
}

private fun <S : Shape> gradTensor(tape: Tape, id: Int, data: FloatArray): DTensor<S, F32> =
    DTensor(HostF32Storage(data), tape.entries[id].dims.copyOf(), F32)

fun <S : Shape> valueAndGrad(
    f: (Tracer<S>) -> Tracer<ScalarShape>,
): (DTensor<S, F32>) -> Pair<Float, DTensor<S, F32>> = { input ->
    val tape = Tape()
    val x = tape.traceLeaf<S>(input)
    val out = f(x)
    require(out.rank == 0) { "valueAndGrad expects a scalar output, got rank ${out.rank}" }
    val outs = reverseThroughCompiler(tape, listOf(x.id), out.id, "valueAndGrad")
    outs[0][0] to gradTensor<S>(tape, x.id, outs[1])
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
        val outs = reverseThroughCompiler(tape, listOf(ta.id, tb.id), out.id, "valueAndGrad2")
        Triple(outs[0][0], gradTensor<S1>(tape, ta.id, outs[1]), gradTensor<S2>(tape, tb.id, outs[2]))
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
 * require a scalar output, and run the captured function through the compiler's
 * reverse transform (§0.4.446).
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
        val outs = reverseThroughCompiler(tape, listOf(ta.id, tb.id, tc.id), out.id, "valueAndGrad3")
        Quadruple(
            outs[0][0],
            gradTensor<S1>(tape, ta.id, outs[1]),
            gradTensor<S2>(tape, tb.id, outs[2]),
            gradTensor<S3>(tape, tc.id, outs[3]),
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
    val outs = reverseThroughCompiler(tape, listOf(ta.id, tb.id), out.id, "valueAndGradWithScalar")
    Triple(outs[0][0], gradTensor<S>(tape, ta.id, outs[1]), outs[2][0])
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
    val outs = reverseThroughCompiler(tape, listOf(ta.id, tb.id), out.id, "valueAndGradWithScalars")
    Triple(outs[0][0], outs[1][0], outs[2][0])
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
