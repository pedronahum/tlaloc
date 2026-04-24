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

/**
 * §0.4.65 — creates a scalar constant leaf on the same tape as [this]. The leaf is
 * flagged `isConstant`, so the reverse walk short-circuits any gradient flowing
 * into it — the exp position of `x.pow(x.constant(2f))` doesn't spend compute
 * materialising PowRule's `x^e · ln(x)` path for a value the user can't retrieve.
 *
 * Typical use: `x.pow(x.constant(2f))` for a squaring kernel whose exponent is
 * known at trace time, `y - x.constant(threshold)` to bias a Tracer against a
 * constant offset, etc. The scalar shape is always `ScalarShape`; a rank-1
 * overload can be added when a use case surfaces.
 */
fun Tracer<*>.constant(value: Float): Tracer<io.tlaloc.core.ScalarShape> {
    val entry = tape.leaf(dims = IntArray(0), value = floatArrayOf(value), isConstant = true)
    return Tracer(tape, entry)
}

/**
 * §0.4.67 — rank-1 form of [constant]. Creates a `Rank1<Sym>` leaf on the same
 * tape as [this], flagged non-differentiable. Defensively copies [values] so a
 * later caller mutation of the input array doesn't desync from the tape's cache.
 *
 * Useful for `x.pow(x.constant(floatArrayOf(2f, 3f, 4f)))` (per-element exponent
 * schedule), or as a fixed bias in an `x + x.constant(...)` pattern once the
 * broadcasting story lands. The `Sym` symbolic axis is unbranded — any concrete
 * Rank1 type the caller prefers can be used at the call site via `as`, but the
 * canonical shape here is `Rank1<Sym>` to match `Tensors.f32Vector`'s default.
 */
fun Tracer<*>.constant(values: FloatArray): Tracer<io.tlaloc.core.Rank1<io.tlaloc.core.Sym>> {
    require(values.isNotEmpty()) { "constant(FloatArray): empty array is not a valid rank-1 tracer" }
    val entry = tape.leaf(
        dims = intArrayOf(values.size),
        value = values.copyOf(),
        isConstant = true,
    )
    return Tracer(tape, entry)
}
