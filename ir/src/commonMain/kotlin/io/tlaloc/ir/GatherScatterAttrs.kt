package io.tlaloc.ir

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * ALL_GATHER and REDUCE_SCATTER, one parser for every layer.
 *
 * Attributes:
 * - `replica_groups` as [AllReduceAttrs] takes them (absent: `[[0]]`);
 * - `dimension: Int`, the axis gathered (ALL_GATHER: the result's axis is |group| times the
 *   operand's) or scattered (REDUCE_SCATTER: the operand's axis is |group| times the
 *   result's; the reduction is a sum).
 *
 * Single-process semantics, the replicated-value view [AllReduceAttrs] uses: this process is
 * replica 0, and every member of its group holds the same value. ALL_GATHER is then |group|
 * copies of the value along `dimension`; REDUCE_SCATTER is |group| times the chunk at replica
 * 0's place in its group. With one replica both are the identity. Both are linear (the tangent
 * is the same op); their adjoints (each other's, per replica) are not built: the
 * single-process view could not check them.
 */
@ExperimentalTlalocApi
object GatherScatterAttrs {

    data class Parsed(val groups: List<List<Int>>, val dimension: Int) {
        /** Size of replica 0's group. */
        val groupSize: Int = groups.first { 0 in it }.size

        /** Replica 0's place in its group: its chunk under REDUCE_SCATTER. */
        val placeOfReplicaZero: Int = groups.first { 0 in it }.indexOf(0)
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.ALL_GATHER || op.op == OpKind.REDUCE_SCATTER) {
            "$layer: GatherScatterAttrs.parse called on ${op.op} (a compiler bug)"
        }
        val kind = op.op.name
        require(op.attrs.keys.all { it == "replica_groups" || it == "dimension" }) {
            "$layer: $kind takes replica_groups and dimension, got ${op.attrs.keys}"
        }
        val d = (op.attrs["dimension"] as? Number)?.toInt() ?: error("$layer: $kind needs an Int dimension")
        val groups = AllReduceAttrs.replicaGroups(op, kind, layer)
        val p = Parsed(groups, d)
        require(op.operands.size == 1 && op.types.size == 1) { "$layer: $kind takes one operand and returns one result" }
        val x = op.operands[0].type
        require(d in 0 until x.rank && op.type.dtype == x.dtype && op.type.rank == x.rank) {
            "$layer: $kind dimension $d must be an axis of the operand ${x}, result ${op.type}"
        }
        val want = x.dims.toMutableList()
        if (op.op == OpKind.ALL_GATHER) {
            want[d] = x.dims[d] * p.groupSize
        } else {
            require(x.dims[d] % p.groupSize == 0) { "$layer: REDUCE_SCATTER axis $d of $x is not a multiple of the group size ${p.groupSize}" }
            want[d] = x.dims[d] / p.groupSize
        }
        require(op.type.dims == want) { "$layer: $kind of $x over groups of ${p.groupSize} along $d is ${want}, got ${op.type.dims}" }
        return p
    }

    /** The single-process value of [op] on [x] (row-major, the operand's dims). */
    fun evaluate(op: DxirOp, x: DoubleArray, layer: String): DoubleArray {
        val p = parse(op, layer)
        val inDims = op.operands[0].type.dims
        val outDims = op.type.dims
        val d = p.dimension
        val inner = inDims.drop(d + 1).fold(1) { a, c -> a * c }
        val outer = inDims.take(d).fold(1) { a, c -> a * c }
        val out = DoubleArray(outDims.fold(1) { a, c -> a * c })
        if (op.op == OpKind.ALL_GATHER) {
            val n = inDims[d]
            for (o in 0 until outer) for (c in 0 until p.groupSize) for (i in 0 until n) for (j in 0 until inner) {
                out[((o * p.groupSize * n) + c * n + i) * inner + j] = x[(o * n + i) * inner + j]
            }
        } else {
            val m = outDims[d]
            val at = p.placeOfReplicaZero * m
            for (o in 0 until outer) for (i in 0 until m) for (j in 0 until inner) {
                out[(o * m + i) * inner + j] = p.groupSize * x[(o * inDims[d] + at + i) * inner + j]
            }
        }
        return out
    }
}
