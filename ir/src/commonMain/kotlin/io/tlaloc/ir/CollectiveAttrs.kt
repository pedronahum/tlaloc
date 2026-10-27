package io.tlaloc.ir

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * ALL_TO_ALL and COLLECTIVE_PERMUTE, one parser for every layer.
 *
 * ALL_TO_ALL attributes: `replica_groups` as [AllReduceAttrs] takes them (absent: `[[0]]`),
 * `split_dimension: Int` and `concat_dimension: Int`. Each member of a group splits its value
 * into |group| chunks along `split_dimension`, sends chunk j to the group's member j, and
 * concatenates the chunks it receives, in member order, along `concat_dimension`.
 *
 * COLLECTIVE_PERMUTE attribute: `source_target_pairs: List<List<Int>>`, each `[source, target]`;
 * no replica is a source twice or a target twice. A replica that is a target receives its
 * source's value; any other replica gets zeros. The shape is unchanged.
 *
 * Single-process semantics, the replicated-value view [AllReduceAttrs] uses: this process is
 * replica 0 and every replica holds the same value. ALL_TO_ALL is then |group| copies of the
 * chunk at replica 0's place, along `concat_dimension`; COLLECTIVE_PERMUTE is the value when
 * replica 0 is a target and zeros otherwise. Both are linear: the tangent is the same op, and
 * the adjoint is ALL_TO_ALL with the two dimensions swapped and COLLECTIVE_PERMUTE with each
 * pair reversed.
 */
@ExperimentalTlalocApi
object CollectiveAttrs {

    data class AllToAll(val groups: List<List<Int>>, val splitDimension: Int, val concatDimension: Int) {
        val groupSize: Int = groups.first { 0 in it }.size
        val placeOfReplicaZero: Int = groups.first { 0 in it }.indexOf(0)
    }

    data class Permute(val pairs: List<Pair<Int, Int>>)

    fun parseAllToAll(op: DxirOp, layer: String): AllToAll {
        require(op.op == OpKind.ALL_TO_ALL) { "$layer: CollectiveAttrs.parseAllToAll called on ${op.op} (a compiler bug)" }
        require(op.attrs.keys.all { it == "replica_groups" || it == "split_dimension" || it == "concat_dimension" }) {
            "$layer: ALL_TO_ALL takes replica_groups, split_dimension and concat_dimension, got ${op.attrs.keys}"
        }
        val split = (op.attrs["split_dimension"] as? Number)?.toInt() ?: error("$layer: ALL_TO_ALL needs an Int split_dimension")
        val concat = (op.attrs["concat_dimension"] as? Number)?.toInt() ?: error("$layer: ALL_TO_ALL needs an Int concat_dimension")
        val p = AllToAll(AllReduceAttrs.replicaGroups(op, "ALL_TO_ALL", layer), split, concat)
        require(op.operands.size == 1 && op.types.size == 1) { "$layer: ALL_TO_ALL takes one operand and returns one result" }
        val x = op.operands[0].type
        require(split in 0 until x.rank && concat in 0 until x.rank && op.type.dtype == x.dtype && op.type.rank == x.rank) {
            "$layer: ALL_TO_ALL dimensions $split and $concat must be axes of the operand $x, result ${op.type}"
        }
        require(x.dims[split] % p.groupSize == 0) {
            "$layer: ALL_TO_ALL split axis $split of $x is not a multiple of the group size ${p.groupSize}"
        }
        require(op.type.dims == resultDims(x.dims, p)) {
            "$layer: ALL_TO_ALL of $x over groups of ${p.groupSize} (split $split, concat $concat) is ${resultDims(x.dims, p)}, got ${op.type.dims}"
        }
        return p
    }

    /** The result dims of an all-to-all of [dims]. */
    fun resultDims(dims: List<Int>, p: AllToAll): List<Int> {
        val out = dims.toMutableList()
        out[p.splitDimension] = out[p.splitDimension] / p.groupSize
        out[p.concatDimension] = out[p.concatDimension] * p.groupSize
        return out
    }

    fun parsePermute(op: DxirOp, layer: String): Permute {
        require(op.op == OpKind.COLLECTIVE_PERMUTE) { "$layer: CollectiveAttrs.parsePermute called on ${op.op} (a compiler bug)" }
        require(op.attrs.keys == setOf("source_target_pairs")) {
            "$layer: COLLECTIVE_PERMUTE takes source_target_pairs, got ${op.attrs.keys}"
        }
        val raw = op.attrs.getValue("source_target_pairs") as? List<*> ?: error("$layer: source_target_pairs must be List<List<Int>>")
        val pairs = raw.map { r ->
            val l = (r as? List<*>)?.map { (it as Number).toInt() } ?: error("$layer: a source_target_pairs row must be List<Int>, got $r")
            require(l.size == 2 && l.all { it >= 0 }) { "$layer: a source_target_pairs row is [source, target] of replica ids, got $l" }
            l[0] to l[1]
        }
        require(pairs.isNotEmpty()) { "$layer: COLLECTIVE_PERMUTE needs at least one pair" }
        require(pairs.map { it.first }.toSet().size == pairs.size && pairs.map { it.second }.toSet().size == pairs.size) {
            "$layer: COLLECTIVE_PERMUTE sends from each replica once and to each replica once, got $pairs"
        }
        require(op.operands.size == 1 && op.types.size == 1 && op.type == op.operands[0].type) {
            "$layer: COLLECTIVE_PERMUTE takes one operand and returns its type"
        }
        return Permute(pairs)
    }

    /** The attributes of [op]'s adjoint: the dimensions swapped, or each pair reversed. */
    fun adjointAttrs(op: DxirOp, layer: String): Map<String, Any> = when (op.op) {
        OpKind.ALL_TO_ALL -> {
            val p = parseAllToAll(op, layer)
            op.attrs + mapOf("split_dimension" to p.concatDimension, "concat_dimension" to p.splitDimension)
        }
        OpKind.COLLECTIVE_PERMUTE -> mapOf("source_target_pairs" to parsePermute(op, layer).pairs.map { (s, t) -> listOf(t, s) })
        else -> error("$layer: CollectiveAttrs.adjointAttrs called on ${op.op} (a compiler bug)")
    }

    /** The single-process value of [op] on [x] (row-major, the operand's dims). */
    fun evaluate(op: DxirOp, x: DoubleArray, layer: String): DoubleArray = when (op.op) {
        OpKind.COLLECTIVE_PERMUTE ->
            if (parsePermute(op, layer).pairs.any { it.second == 0 }) x.copyOf() else DoubleArray(x.size)
        OpKind.ALL_TO_ALL -> {
            val p = parseAllToAll(op, layer)
            val inDims = op.operands[0].type.dims
            val outDims = op.type.dims
            val out = DoubleArray(outDims.fold(1) { a, c -> a * c })
            val chunk = inDims[p.splitDimension] / p.groupSize
            // A chunk's dims: the operand's, with the split axis divided by the group size.
            val chunkDims = inDims.toMutableList().also { it[p.splitDimension] = chunk }
            val inStrides = strides(inDims)
            val outIdx = IntArray(outDims.size)
            for (flat in out.indices) {
                var rem = flat
                for (k in outDims.indices.reversed()) { outIdx[k] = rem % outDims[k]; rem /= outDims[k] }
                // The output's concat axis runs over the senders' chunks; each sender sent its
                // chunk at replica 0's place.
                var src = 0
                for (k in inDims.indices) {
                    var i = outIdx[k]
                    if (k == p.concatDimension) i %= chunkDims[k]
                    if (k == p.splitDimension) i += p.placeOfReplicaZero * chunk
                    src += i * inStrides[k]
                }
                out[flat] = x[src]
            }
            out
        }
        else -> error("$layer: CollectiveAttrs.evaluate called on ${op.op} (a compiler bug)")
    }

    private fun strides(dims: List<Int>): IntArray {
        val s = IntArray(dims.size)
        var acc = 1
        for (k in dims.indices.reversed()) { s[k] = acc; acc *= dims[k] }
        return s
    }
}
