package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.hostF32

private fun <S : Shape> elementwise(
    a: DTensor<S, F32>,
    b: DTensor<S, F32>,
    f: (Float, Float) -> Float,
): DTensor<S, F32> {
    require(a.dims.contentEquals(b.dims)) {
        "elementwise shape mismatch: ${a.dims.toList()} vs ${b.dims.toList()}"
    }
    val av = a.hostF32()
    val bv = b.hostF32()
    val out = FloatArray(av.size)
    for (i in av.indices) out[i] = f(av[i], bv[i])
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
}

operator fun <S : Shape> DTensor<S, F32>.plus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> x + y }

operator fun <S : Shape> DTensor<S, F32>.minus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> x - y }

operator fun <S : Shape> DTensor<S, F32>.times(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> x * y }

operator fun <S : Shape> DTensor<S, F32>.div(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> x / y }

fun <S : Shape> DTensor<S, F32>.relu(): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0f) v[i] else 0f
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.neg(): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = -v[i]
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

private fun <S : Shape> DTensor<S, F32>.unary(f: (Float) -> Float): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = f(v[i])
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.sigmoid(): DTensor<S, F32> =
    unary { x -> 1f / (1f + kotlin.math.exp(-x)) }

fun <S : Shape> DTensor<S, F32>.tanh(): DTensor<S, F32> =
    unary { x -> kotlin.math.tanh(x) }

fun <S : Shape> DTensor<S, F32>.exp(): DTensor<S, F32> =
    unary { x -> kotlin.math.exp(x) }

fun <S : Shape> DTensor<S, F32>.log(): DTensor<S, F32> =
    unary { x -> kotlin.math.ln(x) }

fun <S : Shape> DTensor<S, F32>.sqrt(): DTensor<S, F32> =
    unary { x -> kotlin.math.sqrt(x) }

/**
 * Scalar → rank-N uniform broadcast: produce a fresh `DTensor<S, F32>` shaped like
 * [template] whose every element equals [v]. Used by the IR-rewrite synthesis path to
 * lower `OpKind.BROADCAST` in gradient bodies emitted by [io.tlaloc.ir.passes.VjpRegistry.SumRule]
 * (and later MeanRule). Keeping this as a `:core/ops` helper rather than inlining an
 * IR-level loop mirrors every other op's synthesis shape (`IrCall` into HostOps).
 *
 * The caller is responsible for ensuring [template]'s shape matches the intended target
 * shape; the helper copies [template.dims] rather than trusting the phantom type parameter,
 * since generic erasure strips `S` at runtime.
 */
/**
 * Scalar-index-into-rank-1 read (`arr[i]`). The `operator` form lets user code write
 * `arr[i]` inside a `grad` lambda; FIR sees `io.tlaloc.core.ops.get` and emits
 * `OpKind.GATHER(arr, idx)` (§0.4.42's FIR mapping). Generic in shape so we don't
 * fight Kotlin's operator resolver, but the compile path validates `rank == 1` —
 * calls on rank-2 tensors surface as `LoweringException` at FIR time. Runtime falls
 * back to `HostF32Storage` linear indexing, which is correct for rank-1 and
 * surprising for rank ≥ 2 (returns the `i`th element of the flat buffer, not a
 * row). Users who need multi-dim indexing should use tensor-slice helpers (not
 * yet shipped).
 *
 * Out-of-bounds indices propagate as standard `ArrayIndexOutOfBoundsException` from
 * the underlying `FloatArray` — matches Kotlin stdlib conventions.
 */
operator fun <S : Shape> DTensor<S, F32>.get(i: Int): Float = hostF32()[i]

/**
 * Non-destructive scalar-index-into-rank-1 write (`scatter(arr, i, v)`). Returns a
 * fresh `DTensor` with the same shape as [base] and slot `[i]` replaced by [value].
 * Emitted by `DxirToIrSynthesis.irScatter` for gradient bodies — specifically the
 * `SCATTER(BROADCAST(0, arr.type), idx, upstream)` one-hot construction that
 * `GatherRule` (§0.4.41) uses to propagate a scalar adjoint back to a rank-1
 * primal array.
 *
 * Non-destructive by design: gradient accumulation in [io.tlaloc.ir.passes.DxirReverseTransform]
 * relies on value-semantics (each SCATTER produces a fresh vector; outer `ADD`s
 * combine them). Sharing `base`'s buffer and mutating in place would break the
 * accumulator's correctness.
 */
fun <S : Shape> scatter(base: DTensor<S, F32>, i: Int, value: Float): DTensor<S, F32> {
    val src = base.hostF32()
    val out = src.copyOf()
    out[i] = value
    return DTensor(HostF32Storage(out), base.dims.copyOf(), F32)
}

/**
 * §0.4.45 — fused `base[i] += value` with fresh buffer. Returns a new `DTensor`
 * equal to [base] with slot `[i]` incremented by [value]; the other slots are
 * preserved from [base]. Emitted by `DxirToIrSynthesis.irScatterAdd` for the
 * `OpKind.SCATTER_ADD` op, which `GatherRule` emits instead of the 3-op
 * `BROADCAST(0) + SCATTER + ADD` chain it used pre-§0.4.45.
 *
 * One allocation (output buffer) + one `copyOf` + one slot update. Down from
 * three allocations per gradient contribution in the old chain. For N gathers
 * in a gradient body, total transient allocations drop from ~3N to ~N (plus
 * one initial zero-broadcast for the first gather's accumulator seed).
 */
fun <S : Shape> scatterAddInto(base: DTensor<S, F32>, i: Int, value: Float): DTensor<S, F32> {
    val src = base.hostF32()
    val out = src.copyOf()
    out[i] = out[i] + value
    return DTensor(HostF32Storage(out), base.dims.copyOf(), F32)
}

/**
 * §0.4.46 — destructive `base[i] += value`. **Mutates [base]'s internal buffer
 * in place** and returns the SAME `DTensor` wrapper.
 *
 * **ONLY call this when you have exclusive ownership of [base].** The intended
 * caller is synthesised gradient code emitted by `DxirToIrSynthesis.irScatterAdd`
 * for `OpKind.SCATTER_ADD` ops whose `operand[0]` has been proven single-use by
 * `DxirReverseTransform.tagSingleUseScatterAdds`'s SSA use-graph analysis.
 * Invoking this on a DTensor that's aliased elsewhere silently corrupts shared
 * state — no runtime check catches it.
 *
 * Perf win over [scatterAddInto]: saves the per-call `FloatArray.copyOf` +
 * DTensor allocation. For a rank-1 tensor of size N, that's O(N) memory copies
 * + one JVM allocation eliminated per gradient-accumulator step.
 */
fun <S : Shape> scatterAddInPlace(base: DTensor<S, F32>, i: Int, value: Float): DTensor<S, F32> {
    val buf = base.hostF32()
    buf[i] = buf[i] + value
    return base
}

fun <S : Shape> broadcastLike(v: Float, template: DTensor<S, F32>): DTensor<S, F32> {
    val n = template.size
    val out = FloatArray(n) { v }
    return DTensor(HostF32Storage(out), template.dims.copyOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.sum(): DTensor<ScalarShape, F32> {
    val v = hostF32()
    var acc = 0f
    for (x in v) acc += x
    return DTensor(HostF32Storage(floatArrayOf(acc)), intArrayOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.mean(): DTensor<ScalarShape, F32> {
    val v = hostF32()
    if (v.isEmpty()) return DTensor(HostF32Storage(floatArrayOf(0f)), intArrayOf(), F32)
    var acc = 0f
    for (x in v) acc += x
    return DTensor(HostF32Storage(floatArrayOf(acc / v.size)), intArrayOf(), F32)
}

infix fun <R : ShapeAtom, K : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, K>, F32>.matmul(
    other: DTensor<Rank2<K, C>, F32>,
): DTensor<Rank2<R, C>, F32> {
    require(rank == 2 && other.rank == 2) { "matmul requires rank-2 tensors" }
    val m = dims[0]
    val k = dims[1]
    val kb = other.dims[0]
    val n = other.dims[1]
    require(k == kb) { "matmul inner dim mismatch: ${dims.toList()} x ${other.dims.toList()}" }

    val a = hostF32()
    val b = other.hostF32()
    val out = FloatArray(m * n)
    for (i in 0 until m) {
        for (p in 0 until k) {
            val aip = a[i * k + p]
            if (aip == 0f) continue
            val rowOff = i * n
            val bOff = p * n
            for (j in 0 until n) {
                out[rowOff + j] += aip * b[bOff + j]
            }
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(m, n), F32)
}
