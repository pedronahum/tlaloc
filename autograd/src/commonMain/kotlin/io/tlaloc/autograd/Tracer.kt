package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32

class Tracer<S : Shape> internal constructor(
    internal val tape: Tape,
    internal val entry: TapeEntry,
) {
    val id: Int get() = entry.id
    val dims: IntArray get() = entry.dims
    val rank: Int get() = entry.dims.size
    val size: Int get() = entry.size

    fun toDTensor(): DTensor<S, F32> =
        DTensor(HostF32Storage(entry.value.copyOf()), dims.copyOf(), F32)

    /**
     * §0.4.59 — raw read of the tape-recorded forward value at [index], without
     * allocating a copy. Intended for loop predicates that need to decide whether
     * to emit more tape ops based on the current forward state (e.g.
     * `while (d.peek() <= threshold) { d = d + d }`). Reading is a plain array
     * access — the tape records nothing — so this is the right way to branch
     * the host-side control flow on forward state without polluting the reverse
     * walk with irrelevant ops.
     *
     * For a scalar ([rank] `== 0`) omit the index; it defaults to 0.
     *
     * Does NOT defensively copy — callers MUST NOT mutate any FloatArray obtained
     * through this path. (Today no public surface hands out the underlying array;
     * this stays true unless a future API change breaks encapsulation.)
     */
    fun peek(index: Int = 0): Float {
        require(index in 0 until size) {
            "peek index $index out of bounds for tracer with size $size (dims=${dims.toList()})"
        }
        return entry.value[index]
    }
}

/**
 * §0.4.62 — type-safe shortcut for `Tracer<ScalarShape>.peek()`. Compiles only on
 * scalar tracers (rank 0), so a user writing `arr.scalar` on a `Tracer<Rank1<N>>`
 * gets a compile error at the call site instead of a silent `peek(0)` that returns
 * only the first element. Follow-up to §0.4.59 where this convenience was noted as
 * "could still be added later without breaking `peek()`".
 *
 * Implemented as an extension property (not a member) because member-on-generic-
 * with-specific-type-parameter can't express "only on `S = ScalarShape`". An
 * extension where the receiver type is `Tracer<ScalarShape>` does exactly this.
 */
val Tracer<io.tlaloc.core.ScalarShape>.scalar: Float
    get() = peek()

internal fun <S : Shape> Tape.traceLeaf(value: DTensor<S, F32>): Tracer<S> {
    val entry = leaf(dims = value.dims.copyOf(), value = value.hostF32().copyOf())
    return Tracer(this, entry)
}

internal fun sameTape(a: Tracer<*>, b: Tracer<*>): Tape {
    require(a.tape === b.tape) { "operands come from different tapes" }
    return a.tape
}
