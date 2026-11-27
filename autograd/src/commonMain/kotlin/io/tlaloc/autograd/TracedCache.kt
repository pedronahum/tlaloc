package io.tlaloc.autograd

import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.OpKind

/**
 * This cache `[numBlocks, blockSize, kvHeads, headDim]` with the rows of [newKv]
 * `[numTokens, kvHeads, headDim]` written at the flat slots [slots] (I32 `[numTokens]`, slot
 * `b * blockSize + o` is `cache[b, o]`); a negative slot writes nothing (a padding row). The
 * non-negative slots must be distinct. A dense cache of `ctx` rows is the pool with block size 1
 * (`[ctx, 1, kvHeads, headDim]`, slot = position). Records [OpKind.KV_CACHE_WRITE], which XLA emits
 * as one scatter, so with the cache donated a write costs its rows, not the cache. Not
 * differentiable (an inference op).
 */
fun Tracer<*>.kvCacheWrite(newKv: Tracer<*>, slots: Tracer<*>): Tracer<Shape> {
    require(rank == 4) { "kvCacheWrite: the cache must be [numBlocks, blockSize, kvHeads, headDim] (got ${dims.toList()})" }
    require(newKv.rank == 3 && newKv.dims[1] == dims[2] && newKv.dims[2] == dims[3]) {
        "kvCacheWrite: newKv must be [numTokens, ${dims[2]}, ${dims[3]}] (got ${newKv.dims.toList()})"
    }
    require(newKv.dtype == dtype) { "kvCacheWrite: newKv is ${newKv.dtype.name}, the cache ${dtype.name}" }
    require(slots.dtype == I32 && slots.rank == 1 && slots.dims[0] == newKv.dims[0]) {
        "kvCacheWrite: slots must be i32 [${newKv.dims[0]}] (got ${slots.dtype.name} ${slots.dims.toList()})"
    }
    val tape = sameTape(this, newKv).also { sameTape(this, slots) }
    val numSlots = dims[0] * dims[1]
    val stride = dims[2] * dims[3]
    val out = tape.value {
        val o = entry.value.copyOf()
        val seen = HashSet<Int>()
        for (i in 0 until newKv.dims[0]) {
            val s = slots.entry.value[i].toInt()
            if (s < 0) continue
            require(s < numSlots) { "kvCacheWrite: slot $s outside the $numSlots slots" }
            require(seen.add(s)) { "kvCacheWrite: slot $s written twice" }
            newKv.entry.value.copyInto(o, s * stride, i * stride, (i + 1) * stride)
        }
        o
    }
    val e = tape.op(OpKind.KV_CACHE_WRITE, intArrayOf(id, newKv.id, slots.id), dims.copyOf(), out, dtype = dtype)
    return Tracer(tape, e)
}
