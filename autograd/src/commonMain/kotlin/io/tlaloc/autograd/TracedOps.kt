package io.tlaloc.autograd

import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.ir.OpKind
import kotlin.math.pow

private fun requireSameShape(a: Tracer<*>, b: Tracer<*>) {
    require(a.dims.contentEquals(b.dims)) {
        "shape mismatch: ${a.dims.toList()} vs ${b.dims.toList()}"
    }
}

private fun elementwise(a: FloatArray, b: FloatArray, f: (Float, Float) -> Float): FloatArray {
    val out = FloatArray(a.size)
    for (i in a.indices) out[i] = f(a[i], b[i])
    return out
}

operator fun <S : Shape> Tracer<S>.plus(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = elementwise(entry.value, other.entry.value) { x, y -> x + y }
    val e = tape.op(OpKind.ADD, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.minus(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = elementwise(entry.value, other.entry.value) { x, y -> x - y }
    val e = tape.op(OpKind.SUB, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.times(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = elementwise(entry.value, other.entry.value) { x, y -> x * y }
    val e = tape.op(OpKind.MUL, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.relu(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0f) v[i] else 0f
    val e = tape.op(OpKind.RELU, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.step(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0f) 1f else 0f
    val e = tape.op(OpKind.STEP, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.div(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = elementwise(entry.value, other.entry.value) { x, y -> x / y }
    val e = tape.op(OpKind.DIV, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.neg(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = -v[i]
    val e = tape.op(OpKind.NEG, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

/**
 * §0.4.96 — Kotlin unary-minus operator. Lets users write `-x` instead of
 * `x.neg()`. Delegates straight to [neg]; no new tape op or rule. Closes a
 * cosmetic gap where `+`, `-`, `*`, `/` were operators but the negation form
 * needed an explicit method call.
 */
operator fun <S : Shape> Tracer<S>.unaryMinus(): Tracer<S> = neg()

// §0.4.63 — elementwise unary math. The IR side (VjpRegistry) has had rules for
// SQRT / EXP / LOG / TANH / SIGMOID / POW since §0.4.22, but the Tracer surface
// exposed none of them — so a user writing `grad { x -> x.sqrt() }` would get
// an unresolved-reference compile error. These are the tracer-side wrappers.
// Each wrapper (a) computes the forward value directly (no tape) so `entry.value`
// is populated for downstream predicate reads via `peek()` / `.scalar`, and
// (b) records the op on the tape for the reverse walk.
//
// Forward math uses `kotlin.math.*` for F32 — matches what DxirInterpreter
// would compute through the VjpRegistry bridge, avoiding drift between the
// tape's cached forward value and the value the grad rule will re-evaluate.

fun <S : Shape> Tracer<S>.sqrt(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = kotlin.math.sqrt(v[i])
    val e = tape.op(OpKind.SQRT, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.exp(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = kotlin.math.exp(v[i])
    val e = tape.op(OpKind.EXP, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.log(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = kotlin.math.ln(v[i])
    val e = tape.op(OpKind.LOG, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.tanh(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = kotlin.math.tanh(v[i])
    val e = tape.op(OpKind.TANH, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.sigmoid(): Tracer<S> {
    val v = entry.value
    val out = FloatArray(v.size)
    for (i in v.indices) out[i] = 1f / (1f + kotlin.math.exp(-v[i]))
    val e = tape.op(OpKind.SIGMOID, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

/**
 * §0.4.64 — elementwise `base^exp`. Both operands must be same-shape F32 Tracers
 * sharing one tape. VjpRegistry's `PowRule` (§0.4.22; §0.4.53 widened for Int exp
 * inside C6's closed form) computes grad_base = upstream · exp · base^(exp-1)
 * and grad_exp = upstream · base^exp · ln(base). Both flow back through the
 * tape when [Backward.kt:49] routes `OpKind.POW` into the registry.
 *
 * For `x ↦ x^k` where `k` is a runtime-known scalar, wrap `k` as a scalar leaf
 * (e.g. `tape.traceLeaf(f32Scalar(k))`) — it participates in the reverse walk,
 * but if you don't seed `k` with anything in the lambda it contributes no grad.
 * For constant-exp cases where `k` need NOT be differentiable, a future
 * lightweight "constant tracer" wrapper would avoid the leaf dance; tracked as
 * a potential convenience but not shipping here (out of scope for §0.4.64).
 */
fun <S : Shape> Tracer<S>.pow(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = elementwise(entry.value, other.entry.value) { x, y -> x.pow(y) }
    val e = tape.op(OpKind.POW, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

// §0.4.75 — scalar-literal operator overloads. The "broadcasting story" for
// `Tracer<S> <op> Float` composes cleanly through `.constantLike(scalar)` +
// the existing same-shape operators: the scalar is promoted to a rank-S
// constant leaf (flagged non-differentiable via §0.4.65's isConstant path),
// and the existing plus/minus/times/div/pow operators apply unchanged. The
// constant-skip short-circuit in `Backward.applyRegistryRule` means no
// gradient work is spent on the promoted leaf. Callers can write idiomatic
// `x + 5f`, `x * 0.5f` on any rank without reaching for `x.constantLike(...)`
// explicitly.

operator fun <S : Shape> Tracer<S>.plus(scalar: Float): Tracer<S> = this + constantLike(scalar)
operator fun <S : Shape> Tracer<S>.minus(scalar: Float): Tracer<S> = this - constantLike(scalar)
operator fun <S : Shape> Tracer<S>.times(scalar: Float): Tracer<S> = this * constantLike(scalar)
operator fun <S : Shape> Tracer<S>.div(scalar: Float): Tracer<S> = this / constantLike(scalar)

/**
 * §0.4.76 — scalar-literal `pow` overload. Follows the §0.4.75 pattern: promote
 * the Float literal to a same-shape constant leaf and route through the
 * existing `Tracer<S>.pow(Tracer<S>)` surface. PowRule still builds its
 * `grad_exp = upstream · x^e · ln(x)` tree — the §0.4.65 constant-skip cuts
 * the DxirInterpreter evaluation step but not the rule's dxir construction.
 * For callers who want to save that construction cost on a hot loop, a
 * dedicated `OpKind.SCALAR_POW` with a one-sided VjpRule is filed as a future
 * optimisation (§0.4.75's decision note).
 */
fun <S : Shape> Tracer<S>.pow(scalar: Float): Tracer<S> = this.pow(constantLike(scalar))

// §0.4.77 — differentiable scalar-to-rank1 broadcasting. Unlike §0.4.75's
// scalar-literal overloads (which promote a Float constant), these overloads
// accept a Tracer<ScalarShape> that participates in the reverse walk. The
// scalar gets a SUM-reduced gradient, while the rank-1 operand sees a
// per-element gradient matching the expression's elementwise derivative.
//
// Forward: broadcast scalar to rank-1 (records OpKind.BROADCAST on the tape
// with the rank-1 operand's dims as target), then run the regular same-shape
// op. Backward routes BROADCAST through VjpRegistry.BroadcastRule which
// emits SUM(upstream) as the reverse.
//
// Scope for §0.4.77: rank-1 receiver only. Rank-2+ overloads are a
// straight-line extension using the same machinery; deferred until a use case
// surfaces.

// §0.4.77 helper generalised to arbitrary non-scalar shape by §0.4.78 so the
// rank-1 and rank-2 overloads share one implementation. The reverse rule is
// already rank-agnostic: BroadcastRule's SUM-to-scalar collapses any shape to
// rank 0, so adding a rank-2 receiver overload just needs a new operator wrap
// with distinct @JvmName.
private fun <S : Shape> Tracer<S>.broadcastScalar(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<S> {
    val tape = sameTape(this, scalar)
    val scalarValue = scalar.entry.value[0]
    val broadcasted = FloatArray(size) { scalarValue }
    // §0.4.80 — carry `broadcast_dimensions` in the tape op's attrs so both the
    // capture → dxir bridge (emitter requires this attr) and the DxirInterpreter
    // BROADCAST arm see the same shape metadata the IR expects. Scalar input
    // has rank 0, so the canonical attr value is an empty list.
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(scalar.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
    )
    return Tracer<S>(tape, e)
}

// @JvmName distinguishes these from the same-shape Tracer<S>.plus/minus/times/div
// above — after JVM type erasure all four methods would have signature
// `plus(Tracer, Tracer)`. Each gets a distinct JVM name so both overloads
// coexist without "Platform declaration clash" errors.

@kotlin.jvm.JvmName("plusScalarTracerRank1")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.Rank1<A>>.plus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank1<A>> = this + broadcastScalar(scalar)

@kotlin.jvm.JvmName("minusScalarTracerRank1")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.Rank1<A>>.minus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank1<A>> = this - broadcastScalar(scalar)

@kotlin.jvm.JvmName("timesScalarTracerRank1")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.Rank1<A>>.times(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank1<A>> = this * broadcastScalar(scalar)

@kotlin.jvm.JvmName("divScalarTracerRank1")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.Rank1<A>>.div(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank1<A>> = this / broadcastScalar(scalar)

// §0.4.78 — rank-2 receiver overloads. Same pattern as §0.4.77's rank-1 set.
// BroadcastRule's reverse (SUM-to-scalar) handles rank-2 input without changes.

@kotlin.jvm.JvmName("plusScalarTracerRank2")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.plus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<Rank2<A, B>> = this + broadcastScalar(scalar)

@kotlin.jvm.JvmName("minusScalarTracerRank2")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.minus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<Rank2<A, B>> = this - broadcastScalar(scalar)

@kotlin.jvm.JvmName("timesScalarTracerRank2")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.times(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<Rank2<A, B>> = this * broadcastScalar(scalar)

@kotlin.jvm.JvmName("divScalarTracerRank2")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.div(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<Rank2<A, B>> = this / broadcastScalar(scalar)

// §0.4.85 — rank-1-to-rank-2 row-vector broadcasting. `matrix + row` replicates
// `row` (a rank-1 tracer of size N) across the M rows of `matrix` (rank-2
// [M, N]), then applies the same-shape op. The reverse uses §0.4.84's
// axis-aware BroadcastRule: grad wrt row is SUM(upstream) over axis 0 (the
// broadcast-inserted axis), producing a rank-1 gradient.
//
// Records BROADCAST with `broadcast_dimensions = [1]`: input dim 0 → output
// dim 1, so output dim 0 is broadcast-inserted. BroadcastRule's reverse then
// emits `SUM(upstream, reduction_dims = [0])` — exactly the row-wise sum
// users expect as the gradient of a bias term.

/**
 * §0.4.89 — public companion to the §0.4.85 `Tracer<Rank2<A, B>>.plus(Tracer<Rank1<B>>)`
 * operator. Same machinery (records BROADCAST with `broadcast_dimensions = [1]`),
 * exposed as a named builder so users who want the broadcast direction to be
 * explicit in their code can call it directly — e.g. inside a larger expression
 * where the implicit operator overload would be harder to read.
 *
 * Mirrors §0.4.87's `broadcastCol` in both signature and intent. The §0.4.85
 * implicit operator continues to delegate here; no behavioural change.
 */
fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.broadcastRow(
    row: Tracer<io.tlaloc.core.Rank1<B>>,
): Tracer<Rank2<A, B>> {
    require(dims[1] == row.dims[0]) {
        "broadcastRow: row size ${row.dims[0]} doesn't match matrix col size ${dims[1]}"
    }
    val tape = sameTape(this, row)
    val m = dims[0]
    val n = dims[1]
    val rowValues = row.entry.value
    val broadcasted = FloatArray(m * n) { idx -> rowValues[idx % n] }
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(row.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(1)),
    )
    return Tracer<Rank2<A, B>>(tape, e)
}

@kotlin.jvm.JvmName("plusRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.plus(
    row: Tracer<io.tlaloc.core.Rank1<B>>,
): Tracer<Rank2<A, B>> = this + broadcastRow(row)

@kotlin.jvm.JvmName("minusRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.minus(
    row: Tracer<io.tlaloc.core.Rank1<B>>,
): Tracer<Rank2<A, B>> = this - broadcastRow(row)

@kotlin.jvm.JvmName("timesRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.times(
    row: Tracer<io.tlaloc.core.Rank1<B>>,
): Tracer<Rank2<A, B>> = this * broadcastRow(row)

@kotlin.jvm.JvmName("divRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.div(
    row: Tracer<io.tlaloc.core.Rank1<B>>,
): Tracer<Rank2<A, B>> = this / broadcastRow(row)

/**
 * §0.4.87 — rank-1-to-rank-2 column-vector broadcast. Produces a rank-2 tracer
 * where each column-entry gets the corresponding [col] value replicated across
 * all columns. Shape contract: `col.dims[0] == this.dims[0]` (matrix row
 * count); output shape matches [this].
 *
 * Not an operator overload: `Tracer<Rank2<A, B>>.plus(Tracer<Rank1<A>>)` would
 * clash source-level with §0.4.85's `Tracer<Rank2<A, B>>.plus(Tracer<Rank1<B>>)`
 * when the phantom types A and B are both `Sym` (the default for `Tensors.
 * f32Matrix<Sym, Sym>` callers). Kotlin's overload resolution picks "more
 * specific", and both candidates are equally specific for `Rank1<Sym>` — so the
 * call site is ambiguous. Exposing the col-broadcast as a named builder avoids
 * the collision; users write `matrix + matrix.broadcastCol(col)` explicitly.
 *
 * Records `OpKind.BROADCAST` with `broadcast_dimensions = listOf(0)` (input dim
 * 0 → output dim 0; output dim 1 is broadcast-inserted). §0.4.84's axis-aware
 * BroadcastRule reverses it as `SUM(upstream, reduction_dims = [1])` →
 * rank-1 grad back to [col].
 */
// §0.4.97 — rank-3 scalar-broadcast overloads. Same pattern as §0.4.78 but on
// Tracer<Rank3<A, B, C>> receivers. Now that Tensors.f32Tensor3 exists, users
// can construct rank-3 inputs and apply scalar operators directly.

@kotlin.jvm.JvmName("plusScalarTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.plus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this + broadcastScalar(scalar)

@kotlin.jvm.JvmName("minusScalarTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.minus(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this - broadcastScalar(scalar)

@kotlin.jvm.JvmName("timesScalarTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.times(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this * broadcastScalar(scalar)

@kotlin.jvm.JvmName("divScalarTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.div(
    scalar: Tracer<io.tlaloc.core.ScalarShape>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this / broadcastScalar(scalar)

// §0.4.91 — reverse-order scalar-to-rank-1 broadcast operators. `scalar op row`
// where scalar is LHS. Complements §0.4.77's rank-1-on-LHS overloads. Matters
// for non-commutative ops (minus, div). Each overload uses `broadcastScalar`
// on the rank-1 receiver to lift the scalar to the rank-1 shape, then applies
// the existing same-shape operator.

@kotlin.jvm.JvmName("plusRank1ScalarLhs")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.plus(
    row: Tracer<io.tlaloc.core.Rank1<A>>,
): Tracer<io.tlaloc.core.Rank1<A>> = row.broadcastScalar(this) + row

@kotlin.jvm.JvmName("minusRank1ScalarLhs")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.minus(
    row: Tracer<io.tlaloc.core.Rank1<A>>,
): Tracer<io.tlaloc.core.Rank1<A>> = row.broadcastScalar(this) - row

@kotlin.jvm.JvmName("timesRank1ScalarLhs")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.times(
    row: Tracer<io.tlaloc.core.Rank1<A>>,
): Tracer<io.tlaloc.core.Rank1<A>> = row.broadcastScalar(this) * row

@kotlin.jvm.JvmName("divRank1ScalarLhs")
operator fun <A : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.div(
    row: Tracer<io.tlaloc.core.Rank1<A>>,
): Tracer<io.tlaloc.core.Rank1<A>> = row.broadcastScalar(this) / row

// §0.4.92 — reverse-order scalar-to-rank-2 broadcast operators. `scalar op matrix`
// where scalar is LHS. Complements §0.4.78's rank-2-on-LHS scalar-broadcast
// overloads. Same pattern as §0.4.91 but for rank-2 receivers.

@kotlin.jvm.JvmName("plusRank2ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.plus(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastScalar(this) + matrix

@kotlin.jvm.JvmName("minusRank2ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.minus(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastScalar(this) - matrix

@kotlin.jvm.JvmName("timesRank2ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.times(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastScalar(this) * matrix

@kotlin.jvm.JvmName("divRank2ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.div(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastScalar(this) / matrix

// §0.4.98 — reverse-order scalar-to-rank-3 broadcast operators. Same pattern
// as §0.4.91/§0.4.92 but for rank-3 receivers. Lifts the scalar via the rank-3
// receiver's broadcastScalar (rank-agnostic since §0.4.78) and applies the
// existing same-shape operator.

@kotlin.jvm.JvmName("plusRank3ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.plus(
    tensor: Tracer<io.tlaloc.core.Rank3<A, B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = tensor.broadcastScalar(this) + tensor

@kotlin.jvm.JvmName("minusRank3ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.minus(
    tensor: Tracer<io.tlaloc.core.Rank3<A, B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = tensor.broadcastScalar(this) - tensor

@kotlin.jvm.JvmName("timesRank3ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.times(
    tensor: Tracer<io.tlaloc.core.Rank3<A, B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = tensor.broadcastScalar(this) * tensor

@kotlin.jvm.JvmName("divRank3ScalarLhs")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.ScalarShape>.div(
    tensor: Tracer<io.tlaloc.core.Rank3<A, B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = tensor.broadcastScalar(this) / tensor

// §0.4.93 — Float-literal LHS broadcast. `0.5f * matrix`, `1f - row`, etc.
// Closes the last corner of the scalar-op-tensor surface: §0.4.75 gave us
// `tracer <op> Float`; these four give us `Float <op> tracer`. Generic over
// receiver shape via `constantLike` — works for scalar, rank-1, rank-2 (and
// anything else with a shape-aware `constantLike`). Matters for non-
// commutative directions (`5f - tracer`, `10f / tracer`).

operator fun <S : Shape> Float.plus(tracer: Tracer<S>): Tracer<S>  = tracer.constantLike(this) + tracer
operator fun <S : Shape> Float.minus(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this) - tracer
operator fun <S : Shape> Float.times(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this) * tracer
operator fun <S : Shape> Float.div(tracer: Tracer<S>): Tracer<S>   = tracer.constantLike(this) / tracer

// §0.4.95 — Double and Int literal broadcast overloads. Same compositional path
// as §0.4.75 / §0.4.93's Float overloads — the literal is converted to Float
// (the tape's native dtype) at the boundary, then routes through `constantLike`
// + the existing same-shape operator. Lets users write `0.5 * matrix`,
// `2 * row`, `x - 1` without an explicit `f` suffix on every literal.
//
// Both LHS and RHS forms ship together; otherwise the surface is asymmetric
// (e.g. `matrix * 0.5` works but `0.5 * matrix` doesn't) and users hit
// type-mismatch errors mid-expression.

// --- Double ---

operator fun <S : Shape> Tracer<S>.plus(scalar: Double): Tracer<S>  = this + constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.minus(scalar: Double): Tracer<S> = this - constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.times(scalar: Double): Tracer<S> = this * constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.div(scalar: Double): Tracer<S>   = this / constantLike(scalar.toFloat())

operator fun <S : Shape> Double.plus(tracer: Tracer<S>): Tracer<S>  = tracer.constantLike(this.toFloat()) + tracer
operator fun <S : Shape> Double.minus(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) - tracer
operator fun <S : Shape> Double.times(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) * tracer
operator fun <S : Shape> Double.div(tracer: Tracer<S>): Tracer<S>   = tracer.constantLike(this.toFloat()) / tracer

// --- Int ---

operator fun <S : Shape> Tracer<S>.plus(scalar: Int): Tracer<S>  = this + constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.minus(scalar: Int): Tracer<S> = this - constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.times(scalar: Int): Tracer<S> = this * constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.div(scalar: Int): Tracer<S>   = this / constantLike(scalar.toFloat())

operator fun <S : Shape> Int.plus(tracer: Tracer<S>): Tracer<S>  = tracer.constantLike(this.toFloat()) + tracer
operator fun <S : Shape> Int.minus(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) - tracer
operator fun <S : Shape> Int.times(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) * tracer
operator fun <S : Shape> Int.div(tracer: Tracer<S>): Tracer<S>   = tracer.constantLike(this.toFloat()) / tracer

// §0.4.90 — reverse-order row-broadcast operators. `row + matrix`, `row - matrix`,
// etc. where the rank-1 tracer is the LHS. Matters specifically for non-
// commutative ops (minus, div) where argument order changes the math. Each
// overload promotes `this` (the row) to the matrix's shape via `broadcastRow`
// and forwards to the existing same-shape operator. @JvmName disambiguates
// from §0.4.77's `Tracer<Rank1<A>>.plus(Tracer<ScalarShape>)` after erasure.

@kotlin.jvm.JvmName("plusMatrixRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.Rank1<B>>.plus(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastRow(this) + matrix

@kotlin.jvm.JvmName("minusMatrixRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.Rank1<B>>.minus(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastRow(this) - matrix

@kotlin.jvm.JvmName("timesMatrixRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.Rank1<B>>.times(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastRow(this) * matrix

@kotlin.jvm.JvmName("divMatrixRank1Row")
operator fun <A : ShapeAtom, B : ShapeAtom> Tracer<io.tlaloc.core.Rank1<B>>.div(
    matrix: Tracer<Rank2<A, B>>,
): Tracer<Rank2<A, B>> = matrix.broadcastRow(this) / matrix

fun <A : ShapeAtom, B : ShapeAtom> Tracer<Rank2<A, B>>.broadcastCol(
    col: Tracer<io.tlaloc.core.Rank1<A>>,
): Tracer<Rank2<A, B>> {
    require(dims[0] == col.dims[0]) {
        "broadcastCol: col size ${col.dims[0]} doesn't match matrix row size ${dims[0]}"
    }
    val tape = sameTape(this, col)
    val m = dims[0]
    val n = dims[1]
    val colValues = col.entry.value
    val broadcasted = FloatArray(m * n) { idx -> colValues[idx / n] }
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(col.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(0)),
    )
    return Tracer<Rank2<A, B>>(tape, e)
}

fun <S : Shape> Tracer<S>.sum(): Tracer<ScalarShape> {
    val v = entry.value
    var acc = 0f
    for (x in v) acc += x
    val e = tape.op(OpKind.SUM, intArrayOf(id), IntArray(0), floatArrayOf(acc))
    return Tracer<ScalarShape>(tape, e)
}

fun <S : Shape> Tracer<S>.mean(): Tracer<ScalarShape> {
    val v = entry.value
    if (v.isEmpty()) {
        val e = tape.op(OpKind.MEAN, intArrayOf(id), IntArray(0), floatArrayOf(0f))
        return Tracer<ScalarShape>(tape, e)
    }
    var acc = 0f
    for (x in v) acc += x
    val e = tape.op(OpKind.MEAN, intArrayOf(id), IntArray(0), floatArrayOf(acc / v.size))
    return Tracer<ScalarShape>(tape, e)
}

infix fun <R : ShapeAtom, K : ShapeAtom, C : ShapeAtom> Tracer<Rank2<R, K>>.matmul(
    other: Tracer<Rank2<K, C>>,
): Tracer<Rank2<R, C>> {
    require(rank == 2 && other.rank == 2) { "matmul requires rank-2 tensors" }
    val tape = sameTape(this, other)
    val m = dims[0]
    val k = dims[1]
    val kb = other.dims[0]
    val n = other.dims[1]
    require(k == kb) { "matmul inner dim mismatch: ${dims.toList()} x ${other.dims.toList()}" }

    val a = entry.value
    val b = other.entry.value
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
    val e = tape.op(OpKind.MATMUL, intArrayOf(id, other.id), intArrayOf(m, n), out)
    return Tracer<Rank2<R, C>>(tape, e)
}
