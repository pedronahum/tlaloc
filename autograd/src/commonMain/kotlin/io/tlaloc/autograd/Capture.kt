package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Shape
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType

fun Tape.toDxirFunction(
    name: String,
    paramIds: List<Int>,
    returnIds: List<Int>,
): DxirFunction {
    val paramIdSet = paramIds.toHashSet()
    val returnIdSet = returnIds.toHashSet()
    val allIds = entries.map { it.id }.toHashSet()
    val unknownParams = paramIds - allIds
    val unknownReturns = returnIds - allIds
    require(unknownParams.isEmpty()) { "paramIds not on tape: $unknownParams" }
    require(unknownReturns.isEmpty()) { "returnIds not on tape: $unknownReturns" }

    return DxirBuilder.function(name) {
        val idToNode = HashMap<Int, DxirNode>(entries.size)
        val returns = ArrayList<DxirNode>(returnIds.size)

        for (e in entries) {
            // §0.4.442 — the entry's dtype (F32 everywhere pre-F6; I32 for an
            // embedding-index leaf) stamps the reproduced node's DxirType, so
            // an integer leaf comes out an integer-typed param and the reverse
            // transform's §0.4.419 arm hands back its ZEROS_LIKE.
            val type = DxirType(e.dtype, e.dims.toList())
            val node: DxirNode = when {
                e.id in paramIdSet -> param("p${e.id}", type)
                // §0.4.71 — non-param leaves are constants. Pre-§0.4.71 this
                // branch passed the SSA placeholder name ("leaf${id}") as the
                // const's value, producing a malformed DxirConst holding a
                // String. The branch was dead until §0.4.65's `Tracer.constant`
                // introduced user-facing non-param leaves; fix is to pass the
                // actual cached float value from the tape entry. For scalars
                // that's `value[0]`; for rank-N leaves the const value is the
                // whole FloatArray (interpreter + synthesis already know how
                // to splat a rank-N const from a FloatArray).
                e.op == null -> {
                    val constValue: Any = if (e.dims.isEmpty()) e.value[0] else e.value.copyOf()
                    const(constValue, type)
                }
                else -> op(
                    kind = e.op,
                    operands = e.inputs.map { id ->
                        idToNode[id] ?: error("tape entry ${e.id} references unknown input $id")
                    },
                    type = type,
                    // §0.4.80 — propagate the tape entry's attrs onto the dxir
                    // op. BROADCAST needs `broadcast_dimensions` here; emitter
                    // and interpreter both read attrs and error out when
                    // required keys are missing.
                    attrs = e.attrs,
                )
            }
            idToNode[e.id] = node
            if (e.id in returnIdSet) returns += node
        }

        // preserve the requested return order
        returnIds.map { id ->
            idToNode[id] ?: error("missing return $id")
        }
    }
}

fun <S : Shape> capture(
    f: (Tracer<S>) -> Tracer<*>,
    input: DTensor<S, F32>,
    name: String = "traced",
): DxirFunction {
    val tape = Tape()
    val x = tape.traceLeaf<S>(input)
    val out = f(x)
    return tape.toDxirFunction(name, paramIds = listOf(x.id), returnIds = listOf(out.id))
}

/**
 * §0.4.437 — the N-ary capture, the Phase F model-layer entry point. [capture] and
 * [capture2] are the fixed-arity conveniences the intrinsic surface grew up on; a
 * model's forward has one tape leaf per (input tensor + parameter tensor) and that
 * count is arbitrary — the 1–4 ceiling is a `grad {}` lambda-intrinsic property,
 * never the IR's ([Tape.toDxirFunction] takes any [paramIds] list, `DxirFunction`
 * any param count, `DxirReverseTransform` any arity).
 *
 * Traces every tensor in [inputs] as a differentiable tape leaf (in list order —
 * the resulting `DxirFunction`'s positional param order), runs [f] over the leaf
 * tracers, and captures the single result as the function's return. [f] receives
 * the leaves erased to `Tracer<Shape>` — the N-ary surface is untyped by nature
 * (a heterogeneous list of phantom shapes has no useful common spelling); the
 * same-shape generic operators apply directly and shapes are checked at trace time
 * as always.
 *
 * The caller owns the semantics of the position list (which slots are model inputs
 * vs. parameters); `:nn`'s capture step builds exactly that bookkeeping on top.
 *
 * §0.4.442 — the input list is dtype-heterogeneous: an [io.tlaloc.core.I32]
 * tensor (an embedding-index batch) traces through [traceLeafI32] into an
 * I32-typed leaf/param, whose gradient the reverse transform returns as the
 * §0.4.419 ZEROS_LIKE structural zero. Every other dtype refuses loudly.
 */
@Suppress("UNCHECKED_CAST")
fun captureN(
    inputs: List<DTensor<*, *>>,
    name: String = "traced",
    f: (List<Tracer<Shape>>) -> Tracer<*>,
): DxirFunction {
    require(inputs.isNotEmpty()) { "captureN: at least one input tensor is required" }
    val tape = Tape()
    val leaves = inputs.map {
        when (it.dtype) {
            F32 -> tape.traceLeaf(it as DTensor<Shape, F32>)
            io.tlaloc.core.I32 -> tape.traceLeafI32(it as DTensor<Shape, io.tlaloc.core.I32>)
            else -> error("captureN: unsupported leaf dtype ${it.dtype.name} (F32 and I32 only)")
        }
    }
    val out = f(leaves)
    require(out.tape === tape) {
        "captureN: the result tracer does not belong to this capture's tape — " +
            "the lambda must derive its result from the supplied leaf tracers"
    }
    return tape.toDxirFunction(name, paramIds = leaves.map { it.id }, returnIds = listOf(out.id))
}

fun <S1 : Shape, S2 : Shape> capture2(
    f: (Tracer<S1>, Tracer<S2>) -> Tracer<*>,
    a: DTensor<S1, F32>,
    b: DTensor<S2, F32>,
    name: String = "traced",
): DxirFunction {
    val tape = Tape()
    val ta = tape.traceLeaf<S1>(a)
    val tb = tape.traceLeaf<S2>(b)
    val out = f(ta, tb)
    return tape.toDxirFunction(name, paramIds = listOf(ta.id, tb.id), returnIds = listOf(out.id))
}
