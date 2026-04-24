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
            val type = DxirType(F32, e.dims.toList())
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
