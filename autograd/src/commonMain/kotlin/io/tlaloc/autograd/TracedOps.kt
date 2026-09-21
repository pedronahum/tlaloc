package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.broadcastToLike
import io.tlaloc.core.ops.div
import io.tlaloc.core.ops.exp
import io.tlaloc.core.ops.log
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.mean
import io.tlaloc.core.ops.minus
import io.tlaloc.core.ops.neg
import io.tlaloc.core.ops.plus
import io.tlaloc.core.ops.pow
import io.tlaloc.core.ops.relu
import io.tlaloc.core.ops.sigmoid
import io.tlaloc.core.ops.sqrt
import io.tlaloc.core.ops.step
import io.tlaloc.core.ops.sum
import io.tlaloc.core.ops.tanh
import io.tlaloc.core.ops.times
import io.tlaloc.ir.OpKind

// §0.4.447 — audit finding B (docs/AD_SINGLE_ENGINE_AUDIT.md): the pre-F4
// spellings below used to carry their own private FloatArray forward loops — a
// third implementation of the same math alongside the DxirInterpreter arms and
// the `:core` host twins. They now route through the certified
// `io.tlaloc.core.ops` host twins (the same engines the F4–F6 TRACE spellings
// and the K2 plugin's synthesis already call), so the traced forward value and
// the host path are one implementation by construction. Bit-equality with the
// old loops is pinned in TracedOpsHostTwinParityTest. The ONLY private forward
// loop left in this file is `bmm` (no rank-3 batched-matmul host twin exists —
// the named twin-gap, recorded in the audit doc); SLICE / CONCAT / RESHAPE and
// the pool/conv/embedding F4–F6 spellings carry no elementwise math (copy walks
// or already-twinned calls).

private fun requireSameShape(a: Tracer<*>, b: Tracer<*>) {
    require(a.dims.contentEquals(b.dims)) {
        "shape mismatch: ${a.dims.toList()} vs ${b.dims.toList()}"
    }
}

operator fun <S : Shape> Tracer<S>.plus(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = (toDTensor() + other.toDTensor()).hostF32()
    val e = tape.op(OpKind.ADD, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.minus(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = (toDTensor() - other.toDTensor()).hostF32()
    val e = tape.op(OpKind.SUB, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.times(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = (toDTensor() * other.toDTensor()).hostF32()
    val e = tape.op(OpKind.MUL, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.relu(): Tracer<S> {
    val out = toDTensor().relu().hostF32()
    val e = tape.op(OpKind.RELU, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.step(): Tracer<S> {
    val out = toDTensor().step().hostF32()
    val e = tape.op(OpKind.STEP, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

operator fun <S : Shape> Tracer<S>.div(other: Tracer<S>): Tracer<S> {
    requireSameShape(this, other)
    val tape = sameTape(this, other)
    val out = (toDTensor() / other.toDTensor()).hostF32()
    val e = tape.op(OpKind.DIV, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.neg(): Tracer<S> {
    val out = toDTensor().neg().hostF32()
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
// Forward math routes through the `:core` host twins (§0.4.447, audit finding
// B) — the same engines the DxirInterpreter's arms and the K2 plugin's
// synthesis certify against, so there is no drift between the tape's cached
// forward value and the value the grad rule will re-evaluate.

fun <S : Shape> Tracer<S>.sqrt(): Tracer<S> {
    val out = toDTensor().sqrt().hostF32()
    val e = tape.op(OpKind.SQRT, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.exp(): Tracer<S> {
    val out = toDTensor().exp().hostF32()
    val e = tape.op(OpKind.EXP, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.log(): Tracer<S> {
    val out = toDTensor().log().hostF32()
    val e = tape.op(OpKind.LOG, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.tanh(): Tracer<S> {
    val out = toDTensor().tanh().hostF32()
    val e = tape.op(OpKind.TANH, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

fun <S : Shape> Tracer<S>.sigmoid(): Tracer<S> {
    val out = toDTensor().sigmoid().hostF32()
    val e = tape.op(OpKind.SIGMOID, intArrayOf(id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

/**
 * §0.4.64 — elementwise `base^exp`. Both operands must be same-shape F32 Tracers
 * sharing one tape. VjpRegistry's `PowRule` (§0.4.22; §0.4.53 widened for Int exp
 * inside C6's closed form) computes grad_base = upstream · exp · base^(exp-1)
 * and grad_exp = upstream · base^exp · ln(base). Both flow back when the
 * captured function's `OpKind.POW` meets `DxirReverseTransform` (§0.4.446).
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
    val out = toDTensor().pow(other.toDTensor()).hostF32()
    val e = tape.op(OpKind.POW, intArrayOf(id, other.id), dims.copyOf(), out)
    return Tracer<S>(tape, e)
}

// §0.4.75 — scalar-literal operator overloads. The "broadcasting story" for
// `Tracer<S> <op> Float` composes cleanly through `.constantLike(scalar)` +
// the existing same-shape operators: the scalar is promoted to a rank-S
// constant leaf (flagged non-differentiable via §0.4.65's isConstant path),
// and the existing plus/minus/times/div/pow operators apply unchanged. In the
// captured `DxirFunction` (§0.4.446) the promoted leaf becomes a `DxirConst`,
// not a param, so no gradient is ever produced for it. Callers can write idiomatic
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
// op. The reverse transform routes BROADCAST through VjpRegistry.BroadcastRule
// which emits SUM(upstream) as the reverse.
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
    val broadcasted =
        io.tlaloc.core.ops.broadcastLike(scalar.entry.value[0], toDTensor()).hostF32()
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
    // Right-aligned broadcastToLike: rank-1 [N] against the rank-2 [M, N]
    // template replicates the missing leading axis — exactly `broadcast_
    // dimensions = [1]` (§0.4.447: the host twin owns the walk).
    val broadcasted = broadcastToLike(row.toDTensor(), toDTensor()).hostF32()
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

// §0.4.116 — rank-3 + rank-1 cross-rank broadcast on the inner axis.
// `broadcastInner(rank1)` lifts a rank-1 [C] tracer to a rank-3 [A, B, C]
// tracer where each [a, b, :] slice equals the rank-1 input. Records BROADCAST
// with `broadcast_dimensions = [2]`: input dim 0 → output dim 2; output
// dims 0, 1 are broadcast-inserted. §0.4.84's axis-aware BroadcastRule
// reverses it as `SUM(upstream, reduction_dims = [0, 1])` → rank-1 grad
// back to the inner-vector input. Mirrors §0.4.89's `broadcastRow` pattern,
// generalised one rank up.

fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.broadcastInner(
    inner: Tracer<io.tlaloc.core.Rank1<C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> {
    require(dims[2] == inner.dims[0]) {
        "broadcastInner: inner size ${inner.dims[0]} doesn't match tensor dim 2 ${dims[2]}"
    }
    val tape = sameTape(this, inner)
    // Right-aligned broadcastToLike: rank-1 [C] against the rank-3 [A, B, C]
    // template replicates both missing leading axes — `broadcast_dimensions =
    // [2]` (§0.4.447: the host twin owns the walk).
    val broadcasted = broadcastToLike(inner.toDTensor(), toDTensor()).hostF32()
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(inner.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(2)),
    )
    return Tracer<io.tlaloc.core.Rank3<A, B, C>>(tape, e)
}

@kotlin.jvm.JvmName("plusRank1InnerTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.plus(
    inner: Tracer<io.tlaloc.core.Rank1<C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this + broadcastInner(inner)

@kotlin.jvm.JvmName("minusRank1InnerTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.minus(
    inner: Tracer<io.tlaloc.core.Rank1<C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this - broadcastInner(inner)

@kotlin.jvm.JvmName("timesRank1InnerTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.times(
    inner: Tracer<io.tlaloc.core.Rank1<C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this * broadcastInner(inner)

@kotlin.jvm.JvmName("divRank1InnerTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.div(
    inner: Tracer<io.tlaloc.core.Rank1<C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this / broadcastInner(inner)

// §0.4.117 — rank-3 + rank-2 cross-rank broadcast on the leading (batch) axis.
// `broadcastBatch(matrix)` lifts a rank-2 [B, C] tracer to a rank-3 [A, B, C]
// tracer where every [a, :, :] slice equals the matrix. Records BROADCAST with
// `broadcast_dimensions = [1, 2]`: input dim 0 → output dim 1, input dim 1 →
// output dim 2; output dim 0 is broadcast-inserted. §0.4.84's axis-aware
// BroadcastRule reverses it as `SUM(upstream, reduction_dims = [0])` →
// rank-2 grad back to the matrix input. Mirrors §0.4.116's `broadcastInner`
// pattern, broadcasting over the outer axis instead of the inner.

fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.broadcastBatch(
    matrix: Tracer<Rank2<B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> {
    require(dims[1] == matrix.dims[0]) {
        "broadcastBatch: matrix row count ${matrix.dims[0]} doesn't match tensor dim 1 ${dims[1]}"
    }
    require(dims[2] == matrix.dims[1]) {
        "broadcastBatch: matrix col count ${matrix.dims[1]} doesn't match tensor dim 2 ${dims[2]}"
    }
    val tape = sameTape(this, matrix)
    // Right-aligned broadcastToLike: rank-2 [B, C] against the rank-3
    // [A, B, C] template replicates the missing batch axis — `broadcast_
    // dimensions = [1, 2]` (§0.4.447: the host twin owns the walk).
    val broadcasted = broadcastToLike(matrix.toDTensor(), toDTensor()).hostF32()
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(matrix.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(1, 2)),
    )
    return Tracer<io.tlaloc.core.Rank3<A, B, C>>(tape, e)
}

@kotlin.jvm.JvmName("plusRank2BatchTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.plus(
    matrix: Tracer<Rank2<B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this + broadcastBatch(matrix)

@kotlin.jvm.JvmName("minusRank2BatchTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.minus(
    matrix: Tracer<Rank2<B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this - broadcastBatch(matrix)

@kotlin.jvm.JvmName("timesRank2BatchTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.times(
    matrix: Tracer<Rank2<B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this * broadcastBatch(matrix)

@kotlin.jvm.JvmName("divRank2BatchTracerRank3")
operator fun <A : ShapeAtom, B : ShapeAtom, C : ShapeAtom> Tracer<io.tlaloc.core.Rank3<A, B, C>>.div(
    matrix: Tracer<Rank2<B, C>>,
): Tracer<io.tlaloc.core.Rank3<A, B, C>> = this / broadcastBatch(matrix)

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

// --- Long --- (§0.4.110)
// Same compositional path as Int / Double / Float. Long → Float at the boundary
// loses precision past 2^24 (matching the Int and Double overloads' boundary
// behaviour); the typical scalar-literal use case (`5L`, `1000L`) stays exact.

operator fun <S : Shape> Tracer<S>.plus(scalar: Long): Tracer<S>  = this + constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.minus(scalar: Long): Tracer<S> = this - constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.times(scalar: Long): Tracer<S> = this * constantLike(scalar.toFloat())
operator fun <S : Shape> Tracer<S>.div(scalar: Long): Tracer<S>   = this / constantLike(scalar.toFloat())

operator fun <S : Shape> Long.plus(tracer: Tracer<S>): Tracer<S>  = tracer.constantLike(this.toFloat()) + tracer
operator fun <S : Shape> Long.minus(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) - tracer
operator fun <S : Shape> Long.times(tracer: Tracer<S>): Tracer<S> = tracer.constantLike(this.toFloat()) * tracer
operator fun <S : Shape> Long.div(tracer: Tracer<S>): Tracer<S>   = tracer.constantLike(this.toFloat()) / tracer

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
    // The column direction is NOT right-aligned ([M] against [M, N] would pair
    // M with N), so the rank-1 value is reviewed as an [M, 1] column first —
    // a dims-only relabel of the same storage — and broadcastToLike stretches
    // the size-1 axis: `broadcast_dimensions = [0]` (§0.4.447).
    val colAsMatrix = DTensor<Shape, F32>(
        HostF32Storage(col.entry.value.copyOf()), intArrayOf(dims[0], 1), F32,
    )
    val broadcasted = broadcastToLike(colAsMatrix, toDTensor()).hostF32()
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(col.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(0)),
    )
    return Tracer<Rank2<A, B>>(tape, e)
}

/**
 * §0.4.438 — F2: element-count-preserving relayout, the Flatten substrate.
 * Records [OpKind.RESHAPE] with the RESULT dims on the entry and NO attrs —
 * exactly the spelling the interpreter's §0.4.359 arm and `ReshapeRule` read
 * (both work off the operand/result types alone; `Tape.toDxirFunction` puts
 * the entry dims into the op's `DxirType`, so the captured graph carries
 * everything the transform needs). Forward is a pure row-major copy of the
 * cached value; the reverse is `ReshapeRule`'s reshape-the-upstream-back
 * (identity Jacobian under the flat view) — no gradient math here.
 *
 * The result dims are read off the receiver at TRACE time, which is the
 * contract everywhere on the tape: a captured graph is valid for one
 * structure and a structure change retraces (the F1 caching contract).
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.reshape(newDims: IntArray): Tracer<S> {
    require(newDims.all { it > 0 }) {
        "reshape: all dims must be positive (got ${newDims.toList()})"
    }
    val expected = if (newDims.isEmpty()) 1 else newDims.fold(1) { acc, d -> acc * d }
    require(size == expected) {
        "reshape: element count $size does not match product of dims ${newDims.toList()} ($expected)"
    }
    // §0.4.442 — dtype-preserving: a reshape of an I32 index leaf stays
    // I32-typed in the captured graph (EmbeddingBag's rank-2 index flatten).
    val e = tape.op(OpKind.RESHAPE, intArrayOf(id), newDims.copyOf(), entry.value.copyOf(), dtype = entry.dtype)
    return Tracer<Shape>(tape, e) as Tracer<S>
}

// §0.4.440 — F4: the conv-stack TRACE spellings. Forward values come from the
// certified `:core` host twins ([io.tlaloc.core.ops.conv2dGeneral] /
// [io.tlaloc.core.ops.maxPool2dGeneral] / [io.tlaloc.core.ops.avgPool2dGeneral]
// — the same engines the VjpRule gradient bodies execute through), and the
// recorded attrs are EXACTLY the DxirInterpreter's spellings, so
// `Tape.toDxirFunction` reproduces the ops in the captured graph and
// `DxirReverseTransform` differentiates them through the §0.4.385/386/389 fused
// adjoints. No gradient math here — the transform owns it.

/**
 * §0.4.440 — 2-D convolution: NCHW `[N, Ci, H, W]` input against an OIHW
 * `[Co, Ci, kh, kw]` kernel, output NCHW (Tlaloc's native layouts — the F4
 * layout decision recorded in MODEL_LAYER_PLAN.md; DiffKT's NHWC/[Co,kh,kw,Ci]
 * is a layout transpose of the same maths). Records [OpKind.CONV2D] with attrs
 * `window_strides = [strideH, strideW]` and `padding = [[top, bottom], [left,
 * right]]` — `lhs_dilation`/`rhs_dilation`/`window_reversal`/
 * `feature_group_count` are left to their defaults ([1,1]/[1,1]/[false,false]/1),
 * which is both the interpreter's and `Conv2dRule`'s reading. Groups stay 1 at
 * the trace level: the IR supports them but the host twin this forward routes
 * through does not (§0.4.429's named deferral).
 *
 * Result shape `[N, Co, (H + top + bottom − kh)/strideH + 1, (W + left + right
 * − kw)/strideW + 1]`, read off the host twin's own result.
 */
@Suppress("UNCHECKED_CAST", "LongParameterList")
fun <S : Shape> Tracer<*>.conv2d(
    w: Tracer<*>,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): Tracer<S> {
    require(rank == 4 && w.rank == 4) {
        "conv2d: rank-4 NCHW input and OIHW kernel required; got " +
            "${dims.toList()} / ${w.dims.toList()}"
    }
    require(w.dims[1] == dims[1]) {
        "conv2d: kernel input channels ${w.dims[1]} ≠ input channels ${dims[1]} " +
            "(groups stay 1 at the trace level — §0.4.429)"
    }
    require(strideH > 0 && strideW > 0) {
        "conv2d: strides must be positive; got [$strideH, $strideW]"
    }
    val tape = sameTape(this, w)
    val out = io.tlaloc.core.ops.conv2dGeneral<Shape>(
        toDTensor(), w.toDTensor(), strideH, strideW, 1, 1, 1, 1,
        padTop, padBottom, padLeft, padRight, false, false,
    )
    val e = tape.op(
        OpKind.CONV2D,
        intArrayOf(id, w.id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf(
            "window_strides" to listOf(strideH, strideW),
            "padding" to listOf(listOf(padTop, padBottom), listOf(padLeft, padRight)),
        ),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * §0.4.440 — 2-D max pooling, NCHW, the classic non-overlapping pool (window =
 * stride, zero padding — DiffKT's `MaxPool2d(poolH, poolW)` shape and the ONLY
 * form `MaxPool2dRule` v1 differentiates). Records [OpKind.MAXPOOL2D] with
 * attrs `window = [windowH, windowW]`, `window_strides = window`, `padding =
 * [[0, 0], [0, 0]]` — the interpreter's exact spelling, all three explicit so
 * the captured graph never leans on a default.
 *
 * DiffKT requires the spatial dims to divide by the pool (`H %% poolH == 0`);
 * F4 keeps the same requires (F0 landmine 7). Tie convention downstream:
 * MAXPOOL2D_GRAD routes the upstream to ALL within-window ties (§0.4.389) —
 * pin oracles on tie-free grids.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.maxPool2d(windowH: Int, windowW: Int): Tracer<S> =
    tracePool2d(OpKind.MAXPOOL2D, windowH, windowW) as Tracer<S>

/**
 * §0.4.440 — 2-D average pooling, NCHW, non-overlapping (window = stride, zero
 * padding), dividing by the FULL window `kh·kw` (count_include_pad — the
 * interpreter's and PyTorch's convention; padding is zero here so the two
 * conventions coincide anyway). Same attr spelling and divisibility contract
 * as [maxPool2d]; `AvgPool2dRule` reverses through the fused AVGPOOL2D_GRAD.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.avgPool2d(windowH: Int, windowW: Int): Tracer<S> =
    tracePool2d(OpKind.AVGPOOL2D, windowH, windowW) as Tracer<S>

private fun Tracer<*>.tracePool2d(kind: OpKind, windowH: Int, windowW: Int): Tracer<Shape> {
    val opName = if (kind == OpKind.MAXPOOL2D) "maxPool2d" else "avgPool2d"
    require(rank == 4) { "$opName: rank-4 NCHW input required; got ${dims.toList()}" }
    require(windowH > 0 && windowW > 0) {
        "$opName: window must be positive; got [$windowH, $windowW]"
    }
    require(dims[2] % windowH == 0 && dims[3] % windowW == 0) {
        "$opName: spatial dims [${dims[2]}, ${dims[3]}] must divide by the window " +
            "[$windowH, $windowW] (DiffKT's own require — F0 landmine 7)"
    }
    val out =
        if (kind == OpKind.MAXPOOL2D) {
            io.tlaloc.core.ops.maxPool2dGeneral<Shape>(
                toDTensor(), windowH, windowW, windowH, windowW, 0, 0, 0, 0,
            )
        } else {
            io.tlaloc.core.ops.avgPool2dGeneral<Shape>(
                toDTensor(), windowH, windowW, windowH, windowW, 0, 0, 0, 0,
            )
        }
    val e = tape.op(
        kind,
        intArrayOf(id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf(
            "window" to listOf(windowH, windowW),
            "window_strides" to listOf(windowH, windowW),
            "padding" to listOf(listOf(0, 0), listOf(0, 0)),
        ),
    )
    return Tracer(tape, e)
}

// §0.4.441 — F5: axis reductions + the axis broadcast, the BatchNorm substrate
// (MODEL_LAYER_PLAN.md gap-table row F5). The forwards route through the
// `:core` host twins (`sum`/`mean(vararg dims)`, `broadcastToLike` — §0.4.447),
// whose linear-iteration walk the DxirInterpreter's SUM/MEAN projection
// mirrors EXACTLY, so the trace-cached value and the transform's
// `includeForward` re-evaluation are bit-identical. The attrs are
// the interpreter's spellings: `reduction_dims` on SUM/MEAN (§0.4.366 Phase A1
// arms; SumRule/MeanRule read the same attr for the keepdims-reshape reverse),
// `broadcast_dimensions` on BROADCAST (§0.4.371 general form; BroadcastRule
// reverses with SUM over the complementary axes). No gradient math here — the
// transform owns it.

private fun Tracer<*>.checkReductionAxes(axes: IntArray, opName: String): List<Int> {
    val sorted = axes.distinct().sorted()
    require(sorted.isNotEmpty()) {
        "$opName: empty axis list — use the full-reduce ${opName.removeSuffix("(axes)")}() form"
    }
    require(sorted.all { it in 0 until rank }) {
        "$opName: axes ${sorted} out of range for rank $rank (dims ${dims.toList()})"
    }
    return sorted
}

/**
 * §0.4.441 — axis-aware sum. Records [OpKind.SUM] with
 * `reduction_dims = axes` (sorted, deduplicated) and the KEPT dims on the
 * entry — the reduced axes are dropped, the interpreter's own projection.
 * `SumRule` reverses via the keepdims RESHAPE + equal-rank stretch BROADCAST
 * (§0.4.366). `sum(0, 2, 3)` on NCHW is the per-channel batch statistic F5's
 * BatchNorm desugars through.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.sum(axes: IntArray): Tracer<S> {
    val sorted = checkReductionAxes(axes, "sum(axes)")
    val out = toDTensor().sum(*sorted.toIntArray())
    val e = tape.op(
        OpKind.SUM,
        intArrayOf(id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf("reduction_dims" to sorted),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * §0.4.441 — axis-aware mean: the [sum] projection divided per cell by the
 * product of the REDUCED extents (the interpreter's §0.4.366 MEAN arm computes
 * the same sum-then-divide, so the cached value matches bit-for-bit).
 * `MeanRule`'s reverse bakes the same 1/n for concrete dims.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.mean(axes: IntArray): Tracer<S> {
    val sorted = checkReductionAxes(axes, "mean(axes)")
    val out = toDTensor().mean(*sorted.toIntArray())
    val e = tape.op(
        OpKind.MEAN,
        intArrayOf(id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf("reduction_dims" to sorted),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * §0.4.441 — rank-1 → rank-N broadcast along one axis of the RECEIVER's shape:
 * the receiver contributes dims + tape only (it is NOT an operand — no gradient
 * flows to it through this op, matching [broadcastRow]'s convention), and [vec]
 * (rank-1, `vec.dims[0] == dims[axis]`) is replicated over every other axis.
 * Records [OpKind.BROADCAST] with `broadcast_dimensions = [axis]`;
 * `BroadcastRule` reverses it as `SUM(upstream, reduction_dims = <all other
 * axes>)` — for the NCHW channel axis (`axis = 1`) exactly the per-channel
 * gradient every BatchNorm/bias term wants. Generalises §0.4.85's
 * `broadcastRow` (rank-2, axis 1) to any receiver rank; F5's BatchNorm uses it
 * at rank 4.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.broadcastAlong(vec: Tracer<*>, axis: Int): Tracer<S> {
    require(vec.rank == 1) {
        "broadcastAlong: vec must be rank-1 (got dims ${vec.dims.toList()})"
    }
    require(rank >= 1 && axis in 0 until rank) {
        "broadcastAlong: axis $axis out of range for receiver rank $rank"
    }
    require(dims[axis] == vec.dims[0]) {
        "broadcastAlong: vec size ${vec.dims[0]} doesn't match receiver dim $axis = ${dims[axis]}"
    }
    val tape = sameTape(this, vec)
    // Reviewed as a rank-N shape that is 1 everywhere except [axis] — a
    // dims-only relabel of the same storage — so broadcastToLike's equal-rank
    // stretch replicates every other axis: `broadcast_dimensions = [axis]`
    // (§0.4.447: the host twin owns the walk).
    val vecAligned = DTensor<Shape, F32>(
        HostF32Storage(vec.entry.value.copyOf()),
        IntArray(rank) { if (it == axis) dims[axis] else 1 },
        F32,
    )
    val broadcasted = broadcastToLike(vecAligned, toDTensor()).hostF32()
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(vec.id),
        dims.copyOf(),
        broadcasted,
        attrs = mapOf("broadcast_dimensions" to listOf(axis)),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

fun <S : Shape> Tracer<S>.sum(): Tracer<ScalarShape> {
    val out = toDTensor().sum().hostF32()
    val e = tape.op(OpKind.SUM, intArrayOf(id), IntArray(0), out)
    return Tracer<ScalarShape>(tape, e)
}

fun <S : Shape> Tracer<S>.mean(): Tracer<ScalarShape> {
    // The host twin's empty-input convention (0f, not NaN) matches the tape's.
    val out = toDTensor().mean().hostF32()
    val e = tape.op(OpKind.MEAN, intArrayOf(id), IntArray(0), out)
    return Tracer<ScalarShape>(tape, e)
}

infix fun <R : ShapeAtom, K : ShapeAtom, C : ShapeAtom> Tracer<Rank2<R, K>>.matmul(
    other: Tracer<Rank2<K, C>>,
): Tracer<Rank2<R, C>> {
    require(rank == 2 && other.rank == 2) { "matmul requires rank-2 tensors" }
    val tape = sameTape(this, other)
    require(dims[1] == other.dims[0]) {
        "matmul inner dim mismatch: ${dims.toList()} x ${other.dims.toList()}"
    }
    val out = toDTensor() matmul other.toDTensor()
    val e = tape.op(OpKind.MATMUL, intArrayOf(id, other.id), out.dims.copyOf(), out.hostF32())
    return Tracer<Rank2<R, C>>(tape, e)
}

/**
 * §0.4.137 — batched matrix multiply for rank-3 inputs. `(B, M, K) × (B, K, N) → (B, M, N)`.
 * Records as [OpKind.MATMUL] in the tape (the same op kind that the rank-2 matmul uses);
 * §0.4.135's substrate handles both ranks, and §0.4.137's [MatmulRule] extension
 * recognises the rank-3 shape on the reverse pass and emits batched-TRANSPOSE +
 * batched-MATMUL contributions for each operand.
 */
infix fun <B : ShapeAtom, R : ShapeAtom, K : ShapeAtom, C : ShapeAtom>
    Tracer<Rank3<B, R, K>>.bmm(
    other: Tracer<Rank3<B, K, C>>,
): Tracer<Rank3<B, R, C>> {
    require(rank == 3 && other.rank == 3) { "bmm requires rank-3 tensors" }
    val tape = sameTape(this, other)
    val batch = dims[0]
    val m = dims[1]
    val k = dims[2]
    require(other.dims[0] == batch) {
        "bmm batch dim mismatch: ${dims.toList()} x ${other.dims.toList()}"
    }
    require(other.dims[1] == k) {
        "bmm inner dim mismatch: ${dims.toList()} x ${other.dims.toList()}"
    }
    val n = other.dims[2]

    // §0.4.447 TWIN-GAP (audit finding B, recorded in AD_SINGLE_ENGINE_AUDIT.md):
    // `:core` has no rank-3 batched-matmul host twin — its `matmul` is rank-2
    // only — so this is the one private forward loop left in this file. The
    // per-batch inner loop is byte-for-byte the rank-2 twin's skip-zero walk;
    // when a `bmmGeneral` host twin lands, route through it and extend
    // TracedOpsHostTwinParityTest's bit-equality pin.
    val a = entry.value
    val bArr = other.entry.value
    val out = FloatArray(batch * m * n)
    for (bb in 0 until batch) {
        val aBase = bb * m * k
        val bBase = bb * k * n
        val outBase = bb * m * n
        for (i in 0 until m) {
            for (p in 0 until k) {
                val aip = a[aBase + i * k + p]
                if (aip == 0f) continue
                val rowOff = outBase + i * n
                val bOff = bBase + p * n
                for (j in 0 until n) {
                    out[rowOff + j] += aip * bArr[bOff + j]
                }
            }
        }
    }
    val e = tape.op(OpKind.MATMUL, intArrayOf(id, other.id), intArrayOf(batch, m, n), out)
    return Tracer<Rank3<B, R, C>>(tape, e)
}

// §0.4.442 — F6: the embedding-family TRACE spellings (MODEL_LAYER_PLAN.md
// gap-table rows F6). EMBEDDING's forward routes through the certified `:core`
// host twin (the §0.4.409 paddingIndex arity-disambiguated form — the same
// engine the EmbeddingRule gradient body's runtime twin mirrors); SLICE and
// CONCAT are host copy walks mirroring the DxirInterpreter's arms bit-for-bit.
// The attrs are EXACTLY the interpreter's spellings, so `Tape.toDxirFunction`
// reproduces the ops in the captured graph and `DxirReverseTransform` routes
// them through EmbeddingRule (fused EMBEDDING_GRAD dense scatter — the ratified
// §2.7 dense v1), SliceRule and ConcatRule. No gradient math here — the
// transform owns it.

/**
 * §0.4.442 — embedding lookup: the receiver is the rank-2 `[V, D]` F32 table,
 * [indices] a rank-1 `[N]` or rank-2 `[B, N]` I32 leaf (traced through
 * `captureN`'s dtype dispatch), result `[N, D]` / `[B, N, D]`. Records
 * [OpKind.EMBEDDING] with operands (table, indices) and the optional
 * `padding_index` attr (recorded only when ≥ 0 — absent = none, the
 * interpreter's own reading); positions whose index equals [paddingIndex]
 * produce EXACT-zero output rows and, through the fused EMBEDDING_GRAD the
 * EmbeddingRule emits (primal attrs riding along, §0.4.409), contribute
 * exactly zero gradient to the table.
 *
 * The indices operand is non-differentiable by DTYPE: its captured param is
 * I32-typed, so the reverse transform returns the §0.4.419 ZEROS_LIKE
 * structural zero for it — expected, not a bug.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.embedding(indices: Tracer<*>, paddingIndex: Int = -1): Tracer<S> {
    require(rank == 2) {
        "embedding: table must be rank-2 [V, D]; got ${dims.toList()}"
    }
    require(dtype == io.tlaloc.core.F32) {
        "embedding: table must be F32; got ${dtype.name}"
    }
    require(indices.dtype == io.tlaloc.core.I32) {
        "embedding: indices must be an I32 tracer (an integer leaf — trace the index " +
            "tensor as DTensor<*, I32>); got ${indices.dtype.name}"
    }
    require(indices.rank == 1 || indices.rank == 2) {
        "embedding: indices must be rank-1 [N] or rank-2 [B, N]; got ${indices.dims.toList()}"
    }
    val tape = sameTape(this, indices)
    val idxInts = IntArray(indices.size) { indices.entry.value[it].toInt() }
    val tableD = toDTensor() as io.tlaloc.core.DTensor<Rank2<io.tlaloc.core.Sym, io.tlaloc.core.Sym>, io.tlaloc.core.F32>
    val out: io.tlaloc.core.DTensor<*, io.tlaloc.core.F32> =
        if (indices.rank == 1) {
            io.tlaloc.core.ops.embedding(
                tableD,
                io.tlaloc.core.DTensor<io.tlaloc.core.Rank1<io.tlaloc.core.Sym>, io.tlaloc.core.I32>(
                    io.tlaloc.core.HostI32Storage(idxInts), indices.dims.copyOf(), io.tlaloc.core.I32,
                ),
                paddingIndex,
            )
        } else {
            io.tlaloc.core.ops.embedding(
                tableD,
                io.tlaloc.core.DTensor<Rank2<io.tlaloc.core.Sym, io.tlaloc.core.Sym>, io.tlaloc.core.I32>(
                    io.tlaloc.core.HostI32Storage(idxInts), indices.dims.copyOf(), io.tlaloc.core.I32,
                ),
                paddingIndex,
            )
        }
    val e = tape.op(
        OpKind.EMBEDDING,
        intArrayOf(id, indices.id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = if (paddingIndex >= 0) mapOf("padding_index" to paddingIndex) else emptyMap(),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * §0.4.442 — contiguous slice `[start, end)` along [axis] (stride 1 — the only
 * form `SliceRule` v1 differentiates; its reverse is the zero-PAD back to the
 * operand's extent). Records [OpKind.SLICE] with the interpreter's full-rank
 * attr spelling: `start_indices` / `limit_indices` / `strides`, one entry per
 * axis, the untouched axes running `[0, dim)` at stride 1. Forward is the
 * interpreter's own strided copy walk. Dtype-preserving, like [reshape].
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.slice(start: Int, end: Int, axis: Int): Tracer<S> {
    require(rank >= 1 && axis in 0 until rank) {
        "slice: axis $axis out of range for rank $rank (dims ${dims.toList()})"
    }
    require(start in 0 until dims[axis] && end in (start + 1)..dims[axis]) {
        "slice: [start=$start, end=$end) invalid for axis $axis of extent ${dims[axis]}"
    }
    val outDims = dims.copyOf().also { it[axis] = end - start }
    var inner = 1
    for (k in axis + 1 until rank) inner *= dims[k]
    val axisLen = dims[axis]
    val sliceLen = end - start
    var outer = 1
    for (k in 0 until axis) outer *= dims[k]
    val v = entry.value
    val out = FloatArray(outer * sliceLen * inner)
    var dst = 0
    for (o in 0 until outer) {
        val base = (o * axisLen + start) * inner
        v.copyInto(out, dst, base, base + sliceLen * inner)
        dst += sliceLen * inner
    }
    val e = tape.op(
        OpKind.SLICE,
        intArrayOf(id),
        outDims,
        out,
        attrs = mapOf(
            "start_indices" to List(rank) { if (it == axis) start else 0 },
            "limit_indices" to List(rank) { if (it == axis) end else dims[it] },
            "strides" to List(rank) { 1 },
        ),
        dtype = entry.dtype,
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * §0.4.442 — concatenation of [parts] along [axis] (every part the same rank
 * and extents outside [axis]). Records [OpKind.CONCAT] with the interpreter's
 * `dimension` attr and one operand per part; `ConcatRule` reverses it as one
 * SLICE per operand. Forward is the interpreter's own outer-major copy walk.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> concat(parts: List<Tracer<*>>, axis: Int): Tracer<S> {
    require(parts.isNotEmpty()) { "concat: at least one part required" }
    val first = parts[0]
    val tape = parts.fold(first.tape) { t, p ->
        require(p.tape === t) { "concat: parts come from different tapes" }
        t
    }
    val rank = first.rank
    require(rank >= 1 && axis in 0 until rank) {
        "concat: axis $axis out of range for rank $rank (dims ${first.dims.toList()})"
    }
    for (p in parts) {
        require(p.rank == rank) { "concat: rank mismatch ${p.dims.toList()} vs ${first.dims.toList()}" }
        for (k in 0 until rank) {
            require(k == axis || p.dims[k] == first.dims[k]) {
                "concat: dim $k mismatch ${p.dims.toList()} vs ${first.dims.toList()} (only axis $axis may differ)"
            }
        }
    }
    val outDims = first.dims.copyOf().also { it[axis] = parts.sumOf { p -> p.dims[axis] } }
    var outer = 1
    for (k in 0 until axis) outer *= first.dims[k]
    var inner = 1
    for (k in axis + 1 until rank) inner *= first.dims[k]
    val out = FloatArray(outer * outDims[axis] * inner)
    var dst = 0
    for (o in 0 until outer) {
        for (p in parts) {
            val len = p.dims[axis] * inner
            val src = o * len
            p.entry.value.copyInto(out, dst, src, src + len)
            dst += len
        }
    }
    val e = tape.op(
        OpKind.CONCAT,
        IntArray(parts.size) { parts[it].id },
        outDims,
        out,
        attrs = mapOf("dimension" to axis),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}
