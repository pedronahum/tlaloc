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

/**
 * §0.4.206 — Scalar-multiply on a DTensor: `tensor * scalar` returns a fresh
 * `DTensor<S, F32>` with the same shape and each element multiplied by [scalar].
 * Required for vanilla gradient-descent updates (`W = W - lr * dW`) used by
 * CartPole's outer training loop. Avoids the `broadcastLike(scalar, W) * dW`
 * roundabout that would otherwise be needed for `lr * dW`.
 */
operator fun <S : Shape> DTensor<S, F32>.times(scalar: Float): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = v[i] * scalar
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.relu(): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0f) v[i] else 0f
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.198 — Phase 3 first slice. Elementwise step (Heaviside) on a tensor:
 * `1.0` where the element is strictly positive, `0.0` elsewhere (including
 * exactly zero — matches the convention `ReluRule` / `AbsRule` use for STEP
 * adjoints). Emitted by the K2 plugin's [DxirToIrSynthesis.irStep] when the
 * dxir `OpKind.STEP` op has rank-1/2/3 F32 type, which happens in the gradient
 * body of any `relu`-bearing rank-2/3 surface (CartPole NN forward's
 * `relu(X · W)` chain produces RELU on a rank-2 tensor; ReluRule's adjoint
 * emits STEP on the same shape).
 */
fun <S : Shape> DTensor<S, F32>.step(): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0f) 1f else 0f
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.204 — CartPole Phase 3 sixth slice. Elementwise sign (signum) on a tensor:
 * `+1` where x > 0, `-1` where x < 0, `0` at x = 0. Required by CartPole's NN
 * forward `a = sign(tanh(...) - ε)` which discretises the action to {-1, +1}.
 *
 * The K2 plugin's [DxirToIrSynthesis.irSign] emits a call to this helper for
 * `OpKind.SIGN` ops. `SignRule`'s gradient is identically 0 (sign is non-
 * differentiable at the origin and constant elsewhere) — the policy / loss
 * gradient correctly stops at the discretisation boundary.
 */
fun <S : Shape> DTensor<S, F32>.sign(): DTensor<S, F32> {
    val v = hostF32()
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = when {
        v[i] > 0f -> 1f
        v[i] < 0f -> -1f
        else -> 0f
    }
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

/**
 * §0.4.364 — elementwise comparisons producing a 0/1 F32 mask (the
 * DiffKT-gap user surface for `grad {}` lambdas). The host surface stays
 * all-F32 — there is no `DTensor<S, Bool>` host type; masks are 1f/0f,
 * matching the IR's Bool encoding. The K2 plugin lowers these to
 * `COMPARE(direction)` + `CAST` so the same lambda runs on XLA with a
 * genuine `tensor<xi1>` intermediate. Non-differentiable (piecewise
 * constant — CompareRule's zero adjoint), but fully differentiable
 * *through* when routed with [where].
 */
infix fun <S : Shape> DTensor<S, F32>.gt(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x > y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.ge(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x >= y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.lt(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x < y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.le(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x <= y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.eq(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x == y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.ne(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, other) { x, y -> if (x != y) 1f else 0f }

/**
 * §0.4.364 — elementwise select: `where(pred, a, b)[i] = if (pred[i] != 0)
 * a[i] else b[i]`. [pred] is a 0/1 F32 mask (usually from [gt] and
 * friends). The differentiable routing primitive: gradients flow to [a]
 * where the mask holds and to [b] elsewhere (WhereRule); [pred] gets
 * none. Lowered by the K2 plugin to `stablehlo.select` via
 * `OpKind.WHERE`.
 */
fun <S : Shape> where(
    pred: DTensor<S, F32>,
    a: DTensor<S, F32>,
    b: DTensor<S, F32>,
): DTensor<S, F32> {
    require(pred.dims.contentEquals(a.dims) && a.dims.contentEquals(b.dims)) {
        "where shape mismatch: pred=${pred.dims.toList()} a=${a.dims.toList()} b=${b.dims.toList()}"
    }
    val p = pred.hostF32()
    val av = a.hostF32()
    val bv = b.hostF32()
    val out = FloatArray(av.size)
    for (i in out.indices) out[i] = if (p[i] != 0f) av[i] else bv[i]
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
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

/**
 * §0.4.195 — Phase 0c-rectangular slice 3b-1. Scalar → rank-N uniform broadcast keyed
 * by an explicit [dims] array rather than a runtime template DTensor. Mirrors
 * [broadcastLike]'s output structurally — `DTensor<S, F32>` with `dims = dims.copyOf()`
 * and `storage = FloatArray(prod(dims)) { v }` — but lifts the shape source out of the
 * template-DTensor channel.
 *
 * **Why a separate helper.** [broadcastLike] requires the caller to pass a template
 * DTensor whose `dims` already match the desired broadcast shape. For square-matrix
 * gradient bodies the lambda's first tensor param consistently has the right shape
 * (e.g., `Rank2<Sym, Sym>` → all axes share one dim), so [broadcastLike]'s template
 * channel works. For rectangular gradient bodies (e.g., `Rank2<R, K> matmul Rank2<K, C>`
 * yields BROADCAST target `Rank2<R, C>`), no single param has matching dims — the
 * dims must be assembled from multiple params (`a.dims[0]`, `b.dims[1]`). Slice 3b-2
 * will wire the K2 plugin to synthesize an `intArrayOf(a.dims[0], b.dims[1])` IR
 * expression at the BROADCAST site and emit a call to this helper.
 *
 * Defensive: copies `dims` so callers retain ownership of their array. Empty dims
 * produces a 1-element scalar-shaped DTensor (size = 1 by convention; `dims.fold(1)` →
 * 1 for empty).
 */
fun <S : Shape> broadcastDims(v: Float, dims: IntArray): DTensor<S, F32> {
    val n = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
    val out = FloatArray(n) { v }
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.197 — Phase 0c-rectangular slice 3b-2b. Rank-specific delegates for
 * [broadcastDims]. The K2 plugin's `irBroadcast` path emits a call to
 * `broadcastDimsRank{1, 2, 3}` with individual `Int` args (read from existing
 * tensor params' dims via property-getter + IntArray.get IR calls) instead of
 * synthesising an `IntArray` vararg expression. Each delegate just reassembles
 * the IntArray and forwards to [broadcastDims], so the runtime semantics are
 * identical.
 *
 * **Why per-rank delegates instead of a vararg helper.** Synthesising an
 * `intArrayOf(vararg)` IR expression requires building an [IrVararg] node,
 * whose plugin-facing constructor is gated behind an `IrElementConstructorIndicator`
 * marker (the public surface of `IrSimpleTypeImplKt` doesn't expose a clean
 * factory for it). Per-rank delegates sidestep the vararg synthesis entirely —
 * each call is a regular `IrCall` with one `Float` + N `Int` arguments, which
 * `IrCallImpl.fromSymbolOwner` handles natively.
 */
fun <S : Shape> broadcastDimsRank1(v: Float, d0: Int): DTensor<S, F32> =
    broadcastDims(v, intArrayOf(d0))

fun <S : Shape> broadcastDimsRank2(v: Float, d0: Int, d1: Int): DTensor<S, F32> =
    broadcastDims(v, intArrayOf(d0, d1))

fun <S : Shape> broadcastDimsRank3(v: Float, d0: Int, d1: Int, d2: Int): DTensor<S, F32> =
    broadcastDims(v, intArrayOf(d0, d1, d2))

/**
 * §0.4.188 — DTensor → Float bridge for grad lambdas. The K2 plugin recognises
 * this call site (via FirLambdaToDxirLowering) as a no-op at the dxir level —
 * `DxirType(F32, [])` is the same whether the value flows through a DTensor
 * wrapper or a primitive Float. Closes the last gap for end-to-end MATMUL /
 * SUM-bearing gradient lambdas where the body must terminate in a Float.
 */
fun DTensor<ScalarShape, F32>.toFloat(): Float = hostF32()[0]

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

/**
 * §0.4.366 — axis-wise reduction engine (DiffKT parity, Phase A1: DiffKT's
 * `sum(vararg axes: Int, keepDims: Boolean)` family). Reduces [x] over the
 * axes in [dims] (negative axes count from the back), accumulating with [acc]
 * from [init]; [finish] maps (accumulated, reducedElementCount) → output
 * element (mean divides, sum/max/min pass through). `keepDims = true` keeps
 * the reduced axes as size-1; `false` drops them (rank shrinks). Row-major
 * stride walk, matching the dxir interpreter's SUM/MEAN/MAX/MIN evals so the
 * host path and the IR path agree bit-for-bit on iteration order.
 */
private fun <S : Shape> reduceOver(
    x: DTensor<S, F32>,
    dims: IntArray,
    keepDims: Boolean,
    init: Float,
    acc: (Float, Float) -> Float,
    finish: (Float, Int) -> Float = { a, _ -> a },
): DTensor<Shape, F32> {
    val r = x.dims.size
    require(dims.isNotEmpty()) { "reduceOver: empty axis list (use the no-arg full reduction)" }
    val reduced = BooleanArray(r)
    for (d in dims) {
        val a = if (d < 0) d + r else d
        require(a in 0 until r) { "reduceOver: axis $d out of range for rank $r" }
        require(!reduced[a]) { "reduceOver: duplicate axis $d" }
        reduced[a] = true
    }
    val keepShape = IntArray(r) { if (reduced[it]) 1 else x.dims[it] }
    var n = 1
    for (i in 0 until r) if (reduced[i]) n *= x.dims[i]
    var outSize = 1
    for (d in keepShape) outSize *= d
    val inStrides = IntArray(r)
    val outStrides = IntArray(r)
    var si = 1
    var so = 1
    for (i in r - 1 downTo 0) {
        inStrides[i] = si
        si *= x.dims[i]
        outStrides[i] = so
        so *= keepShape[i]
    }
    val v = x.hostF32()
    val out = FloatArray(outSize) { init }
    for (li in v.indices) {
        var rem = li
        var oi = 0
        for (i in 0 until r) {
            val idx = rem / inStrides[i]
            rem %= inStrides[i]
            if (!reduced[i]) oi += idx * outStrides[i]
        }
        out[oi] = acc(out[oi], v[li])
    }
    for (i in out.indices) out[i] = finish(out[i], n)
    val outDims = if (keepDims) keepShape else {
        var k = 0
        val squeezed = IntArray(r - dims.size)
        for (i in 0 until r) if (!reduced[i]) squeezed[k++] = x.dims[i]
        squeezed
    }
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * §0.4.366 — axis-wise reductions, the DiffKT `sum(axes, keepDims)` user
 * surface (Phase A1). The result's shape type is erased to [Shape]: the
 * output dims depend on the runtime axis list, which Kotlin's phantom shape
 * typing cannot express per-overload. Inside `grad {}` the K2 plugin computes
 * the exact `DxirType` from the operand's dims and the constant axis
 * arguments, so the IR stays precisely shaped; only the host-side Kotlin
 * static type widens. The no-arg overloads above keep their `ScalarShape`.
 */
fun <S : Shape> DTensor<S, F32>.sum(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F32> =
    reduceOver(this, dims, keepDims, 0f, { a, b -> a + b })

fun <S : Shape> DTensor<S, F32>.mean(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F32> =
    reduceOver(this, dims, keepDims, 0f, { a, b -> a + b }, { a, n -> a / n })

fun <S : Shape> DTensor<S, F32>.max(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F32> =
    reduceOver(this, dims, keepDims, Float.NEGATIVE_INFINITY, { a, b -> if (b > a) b else a })

fun <S : Shape> DTensor<S, F32>.min(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F32> =
    reduceOver(this, dims, keepDims, Float.POSITIVE_INFINITY, { a, b -> if (b < a) b else a })

/** §0.4.366 — full-reduce extremum companions to [sum]/[mean] (DiffKT defaults `axes = allAxes`). */
fun <S : Shape> DTensor<S, F32>.max(): DTensor<ScalarShape, F32> {
    val v = hostF32()
    require(v.isNotEmpty()) { "max: empty tensor" }
    var m = Float.NEGATIVE_INFINITY
    for (x in v) if (x > m) m = x
    return DTensor(HostF32Storage(floatArrayOf(m)), intArrayOf(), F32)
}

fun <S : Shape> DTensor<S, F32>.min(): DTensor<ScalarShape, F32> {
    val v = hostF32()
    require(v.isNotEmpty()) { "min: empty tensor" }
    var m = Float.POSITIVE_INFINITY
    for (x in v) if (x < m) m = x
    return DTensor(HostF32Storage(floatArrayOf(m)), intArrayOf(), F32)
}

/**
 * §0.4.366 — fixed-arity synthesis delegates for the axis reductions, one per
 * (kind, axis-count) pair. Same reason as [broadcastDimsRank1] (§0.4.197):
 * the K2 synthesis cannot build `IrVararg` nodes, so `DxirToIrSynthesis`
 * emits calls to these with plain `Int` + `Boolean` const arguments read off
 * the dxir op's `reduction_dims` attr and result type. Runtime semantics are
 * identical to the vararg user surface above.
 */
fun <S : Shape> sumOver1(x: DTensor<S, F32>, d0: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.sum(d0, keepDims = keepDims)

fun <S : Shape> sumOver2(x: DTensor<S, F32>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.sum(d0, d1, keepDims = keepDims)

fun <S : Shape> meanOver1(x: DTensor<S, F32>, d0: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.mean(d0, keepDims = keepDims)

fun <S : Shape> meanOver2(x: DTensor<S, F32>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.mean(d0, d1, keepDims = keepDims)

fun <S : Shape> maxOver1(x: DTensor<S, F32>, d0: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.max(d0, keepDims = keepDims)

fun <S : Shape> maxOver2(x: DTensor<S, F32>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.max(d0, d1, keepDims = keepDims)

fun <S : Shape> minOver1(x: DTensor<S, F32>, d0: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.min(d0, keepDims = keepDims)

fun <S : Shape> minOver2(x: DTensor<S, F32>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F32> =
    x.min(d0, d1, keepDims = keepDims)

/**
 * §0.4.366 — stretch broadcast: tile [x] (whose dims must each be 1 or equal
 * the target) up to the target dims. This is the host twin of the dxir
 * interpreter's keepdims-stretch BROADCAST arm — the shape the reduction
 * VJP rules emit when un-reducing an upstream back over the reduced axes
 * (`RESHAPE to keepdims` → `BROADCAST stretch to x.dims`). The splat helper
 * [broadcastDims] fills a constant; this tiles a tensor — different op.
 * Fixed-arity rank delegates below for the same IrVararg reason as
 * [broadcastDimsRank1].
 */
private fun <S : Shape> stretchTo(x: DTensor<*, F32>, target: IntArray): DTensor<S, F32> {
    val r = target.size
    require(x.dims.size == r) {
        "stretchTo: rank mismatch ${x.dims.toList()} vs ${target.toList()} (reshape to keepdims first)"
    }
    for (i in 0 until r) require(x.dims[i] == 1 || x.dims[i] == target[i]) {
        "stretchTo: dim $i is ${x.dims[i]}, target ${target[i]} (must be 1 or equal)"
    }
    val inStrides = IntArray(r)
    val outStrides = IntArray(r)
    var si = 1
    var so = 1
    for (i in r - 1 downTo 0) {
        inStrides[i] = si
        si *= x.dims[i]
        outStrides[i] = so
        so *= target[i]
    }
    var outSize = 1
    for (d in target) outSize *= d
    val v = x.hostF32()
    val out = FloatArray(outSize)
    for (li in out.indices) {
        var rem = li
        var ii = 0
        for (i in 0 until r) {
            val idx = rem / outStrides[i]
            rem %= outStrides[i]
            ii += (if (x.dims[i] == 1) 0 else idx) * inStrides[i]
        }
        out[li] = v[ii]
    }
    return DTensor(HostF32Storage(out), target.copyOf(), F32)
}

fun <S : Shape> stretchToRank1(x: DTensor<*, F32>, d0: Int): DTensor<S, F32> =
    stretchTo(x, intArrayOf(d0))

fun <S : Shape> stretchToRank2(x: DTensor<*, F32>, d0: Int, d1: Int): DTensor<S, F32> =
    stretchTo(x, intArrayOf(d0, d1))

fun <S : Shape> stretchToRank3(x: DTensor<*, F32>, d0: Int, d1: Int, d2: Int): DTensor<S, F32> =
    stretchTo(x, intArrayOf(d0, d1, d2))

/**
 * §0.4.366 — template-shaped stretch: tile [x] up to [template]'s runtime
 * dims. The synthesis fallback when structural axis-matching against the
 * function params fails (mirrors `broadcastLike(v, template)` for splats).
 */
fun <S : Shape> stretchLike(x: DTensor<*, F32>, template: DTensor<S, F32>): DTensor<S, F32> =
    stretchTo(x, template.dims)

/**
 * §0.4.366 — insert size-1 axes at the given (result-indexed, ascending)
 * positions. The host twin of the keepdims RESHAPE the reduction VJP rules
 * emit (`upstream` at the squeezed shape → the keepdims spelling): axis
 * POSITIONS are compile-time constants from `reduction_dims`, so the
 * synthesis can bake them as Int consts without touching runtime dims —
 * dims themselves may be symbolic sentinels at compile time. Fixed-arity
 * variants for the usual IrVararg reason.
 */
private fun unsqueezeAxes(x: DTensor<*, F32>, axes: IntArray): IntArray {
    val outRank = x.dims.size + axes.size
    val out = IntArray(outRank)
    var prev = -1
    for (a in axes) {
        require(a in 0 until outRank) { "unsqueezeAxes: axis $a out of range for result rank $outRank" }
        require(a > prev) { "unsqueezeAxes: axes must be strictly ascending, got ${axes.toList()}" }
        prev = a
    }
    var src = 0
    for (i in 0 until outRank) {
        out[i] = if (i in axes) 1 else x.dims[src++]
    }
    return out
}

fun <S : Shape> unsqueezeAxes1(x: DTensor<*, F32>, a0: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(x.hostF32().copyOf()), unsqueezeAxes(x, intArrayOf(a0)), F32)

fun <S : Shape> unsqueezeAxes2(x: DTensor<*, F32>, a0: Int, a1: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(x.hostF32().copyOf()), unsqueezeAxes(x, intArrayOf(a0, a1)), F32)

/**
 * §0.4.367 — Phase A2a (DiffKT parity): the RESHAPE-family user surface.
 * `squeeze(axis)` drops a size-1 axis, `unsqueeze(axis)` inserts one,
 * `flatten()` collapses to rank-1, `reshape(vararg dims)` is the general
 * element-count-preserving relayout (row-major; data is shared semantics —
 * we copy for host-value simplicity). Result shape types erase to [Shape]
 * for the same reason as the axis reductions (§0.4.366): the result dims
 * depend on runtime arguments; inside `grad {}` the K2 plugin computes the
 * exact `DxirType` from the literal arguments instead.
 */
fun <S : Shape> DTensor<S, F32>.squeeze(axis: Int): DTensor<Shape, F32> {
    val a = if (axis < 0) axis + dims.size else axis
    require(a in dims.indices) { "squeeze: axis $axis out of range for rank ${dims.size}" }
    require(dims[a] == 1) { "squeeze: axis $axis has size ${dims[a]} (must be 1)" }
    val out = IntArray(dims.size - 1)
    var k = 0
    for (i in dims.indices) if (i != a) out[k++] = dims[i]
    return DTensor(HostF32Storage(hostF32().copyOf()), out, F32)
}

fun <S : Shape> DTensor<S, F32>.unsqueeze(axis: Int): DTensor<Shape, F32> {
    val outRank = dims.size + 1
    val a = if (axis < 0) axis + outRank else axis
    require(a in 0 until outRank) { "unsqueeze: axis $axis out of range for result rank $outRank" }
    return DTensor(HostF32Storage(hostF32().copyOf()), unsqueezeAxes(this, intArrayOf(a)), F32)
}

fun <S : Shape> DTensor<S, F32>.flatten(): DTensor<Shape, F32> {
    val v = hostF32()
    return DTensor(HostF32Storage(v.copyOf()), intArrayOf(v.size), F32)
}

fun <S : Shape> DTensor<S, F32>.reshape(vararg newDims: Int): DTensor<Shape, F32> {
    val v = hostF32()
    var n = 1
    for (d in newDims) {
        require(d > 0) { "reshape: dims must be positive, got ${newDims.toList()}" }
        n *= d
    }
    require(n == v.size) {
        "reshape: element count mismatch — ${dims.toList()} (${v.size}) vs ${newDims.toList()} ($n)"
    }
    return DTensor(HostF32Storage(v.copyOf()), newDims.copyOf(), F32)
}

/**
 * §0.4.367 — general permutation transpose (DiffKT `transpose(axes)`).
 * The no-arg rank-2 [transpose] above keeps its precise `Rank2<C, R>`
 * shape typing; this vararg form handles any rank 1..3 with an erased
 * result type. Row-major stride walk mirroring the dxir interpreter's
 * TRANSPOSE arm.
 */
fun <S : Shape> DTensor<S, F32>.transpose(vararg perm: Int): DTensor<Shape, F32> {
    val r = dims.size
    require(perm.size == r) { "transpose: perm ${perm.toList()} must have length $r" }
    val norm = IntArray(r) { i ->
        val p = if (perm[i] < 0) perm[i] + r else perm[i]
        require(p in 0 until r) { "transpose: axis ${perm[i]} out of range for rank $r" }
        p
    }
    require(norm.toSet().size == r) { "transpose: ${perm.toList()} is not a permutation" }
    val v = hostF32()
    val outDims = IntArray(r) { dims[norm[it]] }
    val inStrides = IntArray(r)
    var st = 1
    for (i in r - 1 downTo 0) { inStrides[i] = st; st *= dims[i] }
    val outStrides = IntArray(r)
    st = 1
    for (i in r - 1 downTo 0) { outStrides[i] = st; st *= outDims[i] }
    val out = FloatArray(v.size)
    for (li in out.indices) {
        var rem = li
        var src = 0
        for (i in 0 until r) {
            val coord = rem / outStrides[i]
            rem %= outStrides[i]
            src += coord * inStrides[norm[i]]
        }
        out[li] = v[src]
    }
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * §0.4.367 — fixed-arity synthesis delegates (the usual IrVararg reason).
 * `squeezeAxes{N}` drops size-1 axes at result-computed positions (the
 * adjoint of an unsqueeze); `reshapeToRank{N}` relayouts to explicit dims —
 * the synthesis feeds each dim either as a baked const (concrete dxir dim)
 * or a `param.dims[i]` runtime read (sentinel dim); `transposePerm{N}`
 * carries the permutation attr's compile-time constants.
 */
fun <S : Shape> squeezeAxes1(x: DTensor<*, F32>, a0: Int): DTensor<S, F32> {
    require(x.dims[a0] == 1) { "squeezeAxes1: axis $a0 has size ${x.dims[a0]}" }
    val out = IntArray(x.dims.size - 1)
    var k = 0
    for (i in x.dims.indices) if (i != a0) out[k++] = x.dims[i]
    return DTensor(HostF32Storage(x.hostF32().copyOf()), out, F32)
}

fun <S : Shape> squeezeAxes2(x: DTensor<*, F32>, a0: Int, a1: Int): DTensor<S, F32> {
    require(a0 < a1) { "squeezeAxes2: axes must be ascending" }
    require(x.dims[a0] == 1 && x.dims[a1] == 1) { "squeezeAxes2: axes must have size 1" }
    val out = IntArray(x.dims.size - 2)
    var k = 0
    for (i in x.dims.indices) if (i != a0 && i != a1) out[k++] = x.dims[i]
    return DTensor(HostF32Storage(x.hostF32().copyOf()), out, F32)
}

private fun reshapeTo(x: DTensor<*, F32>, target: IntArray): FloatArray {
    val v = x.hostF32()
    var n = 1
    for (d in target) n *= d
    require(n == v.size) {
        "reshapeToRank: element count mismatch ${x.dims.toList()} vs ${target.toList()}"
    }
    return v.copyOf()
}

fun <S : Shape> reshapeToRank1(x: DTensor<*, F32>, d0: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(reshapeTo(x, intArrayOf(d0))), intArrayOf(d0), F32)

fun <S : Shape> reshapeToRank2(x: DTensor<*, F32>, d0: Int, d1: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(reshapeTo(x, intArrayOf(d0, d1))), intArrayOf(d0, d1), F32)

fun <S : Shape> reshapeToRank3(x: DTensor<*, F32>, d0: Int, d1: Int, d2: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(reshapeTo(x, intArrayOf(d0, d1, d2))), intArrayOf(d0, d1, d2), F32)

fun <S : Shape> transposePerm2(x: DTensor<*, F32>, p0: Int, p1: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).transpose(p0, p1) as DTensor<S, F32>
}

fun <S : Shape> transposePerm3(x: DTensor<*, F32>, p0: Int, p1: Int, p2: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).transpose(p0, p1, p2) as DTensor<S, F32>
}

/**
 * §0.4.189 — rank-2 transpose. Used by the K2 plugin's synthesis-side lowering
 * of [OpKind.TRANSPOSE] emitted by [io.tlaloc.ir.passes.VjpRegistry.MatmulRule].
 * The signature flips R and C in the shape type so the result is correctly
 * typed for downstream matmul chains.
 */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F32>.transpose(): DTensor<Rank2<C, R>, F32> {
    require(rank == 2) { "transpose requires rank-2 tensor" }
    val rows = dims[0]
    val cols = dims[1]
    val a = hostF32()
    val out = FloatArray(cols * rows)
    for (i in 0 until rows) {
        for (j in 0 until cols) {
            out[j * rows + i] = a[i * cols + j]
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(cols, rows), F32)
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
