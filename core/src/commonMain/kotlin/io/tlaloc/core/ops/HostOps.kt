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
