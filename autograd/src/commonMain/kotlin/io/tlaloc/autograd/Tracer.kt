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

/**
 * §0.4.69 — same-shape constant convenience. Creates a leaf on [this]'s tape
 * with the SAME shape as [this], every element filled with [value], flagged
 * non-differentiable. Enables `x + x.constantLike(5f)` on rank-N Tracers
 * without going through scalar-rank broadcasting: the constant and [this]
 * already share a shape so the existing same-shape `plus`/`minus`/`times`/`div`
 * overloads apply directly.
 *
 * Generalises §0.4.65's scalar `constant(Float)` (which only returns a scalar
 * Tracer) by projecting the fill value into [this]'s shape. For callers who
 * want an arbitrary per-element rank-1 constant, §0.4.67's `constant(FloatArray)`
 * remains the path.
 */
/**
 * §0.4.70 — rank-N form of [constant]. Caller supplies both a flat `FloatArray`
 * of values (row-major) and the target `IntArray` of dims. Validates that
 * `values.size` matches the product of `dims`, and that each dim is positive.
 * Values are defensively copied.
 *
 * The caller picks the phantom shape via type inference or explicit type
 * argument, e.g.:
 *
 * ```kotlin
 * val m: Tracer<Rank2<Sym, Sym>> = x.constant(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
 * ```
 *
 * Kotlin infers `S = Rank2<Sym, Sym>` from the assignment target. The phantom
 * type has no runtime enforcement — mismatched `dims` vs. target `S` would
 * compile fine and fail only at first same-shape op use. This is consistent
 * with `Tensors.f32Vector` / `f32Matrix` (the rank-N constructors in :core).
 *
 * For scalar + rank-1 uses, prefer the existing overloads ([constant]`(Float)`
 * and [constant]`(FloatArray)`) which pin their phantom shape explicitly.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.constant(values: FloatArray, dims: IntArray): Tracer<S> {
    require(dims.isNotEmpty()) { "constant(values, dims): empty dims — use constant(Float) for a scalar leaf" }
    require(dims.all { it > 0 }) {
        "constant(values, dims): all dims must be positive (got ${dims.toList()})"
    }
    val expected = dims.fold(1) { acc, d -> acc * d }
    require(values.size == expected) {
        "constant(values, dims): values.size=${values.size} doesn't match product of dims=${dims.toList()} ($expected)"
    }
    val entry = tape.leaf(
        dims = dims.copyOf(),
        value = values.copyOf(),
        isConstant = true,
    )
    return Tracer<Shape>(tape, entry) as Tracer<S>
}

fun <S : Shape> Tracer<S>.constantLike(value: Float): Tracer<S> {
    val entry = tape.leaf(
        dims = dims.copyOf(),
        value = FloatArray(size) { value },
        isConstant = true,
    )
    return Tracer(tape, entry)
}
