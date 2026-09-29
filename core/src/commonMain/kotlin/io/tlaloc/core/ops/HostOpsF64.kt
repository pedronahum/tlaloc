package io.tlaloc.core.ops

import io.tlaloc.core.DoubleScalar
import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.Sym
import io.tlaloc.core.digamma
import io.tlaloc.core.hostF64
import io.tlaloc.core.hostI32
import io.tlaloc.core.lgamma
import io.tlaloc.core.polygamma
import io.tlaloc.core.trigamma
import kotlin.math.pow
import kotlin.math.sqrt

// F64 twins of the host ops in HostOps.kt, generated from it at Double width, for
// `grad {}` over F64 tensors: the synthesized gradient calls these, and they are the
// plain-Kotlin F64 tensor surface. Convolution, pooling and batchNorm are in ConvOpsF64.kt;
// embedding, the RNG draws and the sparse products exist for F32 only. Their own file
// because each erases to its F32 twin's JVM signature.

private fun <S : Shape> elementwise(
    a: DTensor<S, F64>,
    b: DTensor<S, F64>,
    f: (Double, Double) -> Double,
): DTensor<S, F64> {
    require(a.dims.contentEquals(b.dims)) {
        "elementwise shape mismatch: ${a.dims.toList()} vs ${b.dims.toList()}"
    }
    val av = a.hostF64()
    val bv = b.hostF64()
    val out = DoubleArray(av.size)
    for (i in av.indices) out[i] = f(av[i], bv[i])
    return DTensor(HostF64Storage(out), a.dims.copyOf(), F64)
}

/**
 * The shape-preserving arithmetic operators. Kotlin resolves these (not the
 * broadcasting `<S1, S2>` overloads in BroadcastOps.kt) whenever the two operands
 * share a static shape type — but a shared static type does NOT imply shared
 * runtime dims: `[N,1]` and `[N,C]` are both `Rank2<Sym, Lit<Int>>`. So these
 * delegate to the same broadcasting walk and keep only the precise return witness;
 * equal dims take its flat-zip fast path, and a genuinely mixed pair broadcasts
 * instead of throwing. `elementwise`'s strict `require` survives for the
 * comparisons below, which have no broadcasting surface yet.
 */
operator fun <S : Shape> DTensor<S, F64>.plus(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseBroadcast<S>(this, other) { x, y -> x + y }

operator fun <S : Shape> DTensor<S, F64>.minus(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseBroadcast<S>(this, other) { x, y -> x - y }

operator fun <S : Shape> DTensor<S, F64>.times(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseBroadcast<S>(this, other) { x, y -> x * y }

operator fun <S : Shape> DTensor<S, F64>.div(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseBroadcast<S>(this, other) { x, y -> x / y }

/**
 * Scalar-multiply on a DTensor: `tensor * scalar` returns a fresh
 * `DTensor<S, F64>` with the same shape and each element multiplied by [scalar].
 * Required for vanilla gradient-descent updates (`W = W - lr * dW`) used by
 * CartPole's outer training loop. Avoids the `broadcastLike(scalar, W) * dW`
 * roundabout that would otherwise be needed for `lr * dW`.
 *
 * The scalar-mixing overloads below complete the set: DiffKT mixes scalars into every
 * binary op (`timesScalar` + `broadcast(S1, S2)`), so `a + 1.0`, `a / 2.0`
 * and the scalar-on-the-left spellings `3.0 - a` / `2.0 * a` are all
 * writable. The scalar-on-left forms matter for the non-commutative ops —
 * `3.0 - a` is not `a - 3.0` — and the K2 plugin preserves source operand
 * order when it splats them (see `FirLambdaToDxirLowering`'s mixed-rank arm).
 */
private fun <S : Shape> elementwiseScalar(
    a: DTensor<S, F64>,
    b: Double,
    f: (Double, Double) -> Double,
): DTensor<S, F64> {
    val av = a.hostF64()
    val out = DoubleArray(av.size)
    for (i in av.indices) out[i] = f(av[i], b)
    return DTensor(HostF64Storage(out), a.dims.copyOf(), F64)
}

operator fun <S : Shape> DTensor<S, F64>.plus(scalar: Double): DTensor<S, F64> =
    elementwiseScalar(this, scalar) { x, y -> x + y }

operator fun <S : Shape> DTensor<S, F64>.minus(scalar: Double): DTensor<S, F64> =
    elementwiseScalar(this, scalar) { x, y -> x - y }

operator fun <S : Shape> DTensor<S, F64>.times(scalar: Double): DTensor<S, F64> =
    elementwiseScalar(this, scalar) { x, y -> x * y }

operator fun <S : Shape> DTensor<S, F64>.div(scalar: Double): DTensor<S, F64> =
    elementwiseScalar(this, scalar) { x, y -> x / y }

operator fun <S : Shape> Double.plus(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseScalar(other, this) { x, y -> y + x }

operator fun <S : Shape> Double.minus(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseScalar(other, this) { x, y -> y - x }

operator fun <S : Shape> Double.times(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseScalar(other, this) { x, y -> y * x }

operator fun <S : Shape> Double.div(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseScalar(other, this) { x, y -> y / x }

/**
 * `DoubleScalar × DTensor` mixing, the F64 form of the `DScalar` overloads in
 * HostOps.kt. They take [DoubleScalar], not the `DScalar` interface: a `FloatScalar`
 * next to an F64 tensor would be an implicit F32→F64 promotion, so it does not resolve.
 */
operator fun <S : Shape> DTensor<S, F64>.times(scalar: DoubleScalar): DTensor<S, F64> =
    elementwiseScalar(this, scalar.v) { x, y -> x * y }

operator fun <S : Shape> DoubleScalar.times(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwiseScalar(other, this.v) { x, y -> y * x }

fun <S : Shape> DTensor<S, F64>.relu(): DTensor<S, F64> {
    val v = hostF64()
    val out = DoubleArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0.0) v[i] else 0.0
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * Elementwise step (Heaviside) on a tensor:
 * `1.0` where the element is strictly positive, `0.0` elsewhere (including
 * exactly zero — matches the convention `ReluRule` / `AbsRule` use for STEP
 * adjoints). Emitted by the K2 plugin's [DxirToIrSynthesis.irStep] when the
 * dxir `OpKind.STEP` op has rank-1/2/3 F64 type, which happens in the gradient
 * body of any `relu`-bearing rank-2/3 surface (CartPole NN forward's
 * `relu(X · W)` chain produces RELU on a rank-2 tensor; ReluRule's adjoint
 * emits STEP on the same shape).
 */
fun <S : Shape> DTensor<S, F64>.step(): DTensor<S, F64> {
    val v = hostF64()
    val out = DoubleArray(v.size)
    for (i in v.indices) out[i] = if (v[i] > 0.0) 1.0 else 0.0
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * Elementwise sign (signum) on a tensor:
 * `+1` where x > 0, `-1` where x < 0, `0` at x = 0. Required by CartPole's NN
 * forward `a = sign(tanh(...) - ε)` which discretises the action to {-1, +1}.
 *
 * The K2 plugin's [DxirToIrSynthesis.irSign] emits a call to this helper for
 * `OpKind.SIGN` ops. `SignRule`'s gradient is identically 0 (sign is non-
 * differentiable at the origin and constant elsewhere) — the policy / loss
 * gradient correctly stops at the discretisation boundary.
 */
fun <S : Shape> DTensor<S, F64>.sign(): DTensor<S, F64> {
    val v = hostF64()
    val out = DoubleArray(v.size)
    for (i in v.indices) out[i] = when {
        v[i] > 0.0 -> 1.0
        v[i] < 0.0 -> -1.0
        else -> 0.0
    }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

fun <S : Shape> DTensor<S, F64>.neg(): DTensor<S, F64> {
    val v = hostF64()
    val out = DoubleArray(v.size)
    for (i in v.indices) out[i] = -v[i]
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

private fun <S : Shape> DTensor<S, F64>.unary(f: (Double) -> Double): DTensor<S, F64> {
    val v = hostF64()
    val out = DoubleArray(v.size)
    for (i in v.indices) out[i] = f(v[i])
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * Elementwise comparisons producing a 0/1 F64 mask (the
 * DiffKT-gap user surface for `grad {}` lambdas). The host surface stays
 * all-F64 — there is no `DTensor<S, Bool>` host type; masks are 1.0/0.0,
 * matching the IR's Bool encoding. The K2 plugin lowers these to
 * `COMPARE(direction)` + `CAST` so the same lambda runs on XLA with a
 * genuine `tensor<xi1>` intermediate. Non-differentiable (piecewise
 * constant — CompareRule's zero adjoint), but fully differentiable
 * *through* when routed with [where].
 */
infix fun <S : Shape> DTensor<S, F64>.gt(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x > y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.ge(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x >= y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.lt(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x < y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.le(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x <= y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.eq(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x == y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.ne(other: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, other) { x, y -> if (x != y) 1.0 else 0.0 }

/**
 * Comparisons against a Double scalar (`a gt 1.0`),
 * the last everyday DiffKT comparison spelling Tlaloc rejected. Same 0/1 F64
 * mask contract as the tensor⊙tensor forms above; the scalar side is compared
 * against every element. Inside `grad {}` the K2 plugin splats the scalar side
 * over the tensor operand's shape (the same literal splat the arithmetic
 * overloads use), so the IR sees the uniform two-tensor COMPARE.
 */
infix fun <S : Shape> DTensor<S, F64>.gt(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x > y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.ge(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x >= y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.lt(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x < y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.le(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x <= y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.eq(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x == y) 1.0 else 0.0 }

infix fun <S : Shape> DTensor<S, F64>.ne(other: Double): DTensor<S, F64> =
    elementwiseScalar(this, other) { x, y -> if (x != y) 1.0 else 0.0 }

/**
 * Elementwise select: `where(pred, a, b)[i] = if (pred[i] != 0)
 * a[i] else b[i]`. [pred] is a 0/1 F64 mask (usually from [gt] and
 * friends). The differentiable routing primitive: gradients flow to [a]
 * where the mask holds and to [b] elsewhere (WhereRule); [pred] gets
 * none. Lowered by the K2 plugin to `stablehlo.select` via
 * `OpKind.WHERE`.
 */
fun <S : Shape> where(
    pred: DTensor<S, F64>,
    a: DTensor<S, F64>,
    b: DTensor<S, F64>,
): DTensor<S, F64> {
    require(pred.dims.contentEquals(a.dims) && a.dims.contentEquals(b.dims)) {
        "where shape mismatch: pred=${pred.dims.toList()} a=${a.dims.toList()} b=${b.dims.toList()}"
    }
    val p = pred.hostF64()
    val av = a.hostF64()
    val bv = b.hostF64()
    val out = DoubleArray(av.size)
    for (i in out.indices) out[i] = if (p[i] != 0.0) av[i] else bv[i]
    return DTensor(HostF64Storage(out), a.dims.copyOf(), F64)
}

/**
 * Elementwise `maximum(a, b)` /
 * `minimum(a, b)` — first-class in DiffKT, sugar in Tlaloc. Inside `grad {}`
 * the K2 plugin lowers `maximum` to `WHERE(COMPARE(a, b, GE), a, b)` and
 * `minimum` to `WHERE(COMPARE(a, b, LE), a, b)` — the [where]/compare
 * surface — so the gradient flows through WhereRule with no new AD math:
 * full upstream to the larger (resp. smaller) operand, ties to the first
 * (the `>=` / `<=` mask keeps `a`). This host body is the runtime twin. The
 * shared phantom shape [S] requires same-shape operands (elementwise).
 */
fun <S : Shape> maximum(a: DTensor<S, F64>, b: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(a, b) { x, y -> if (x >= y) x else y }

fun <S : Shape> minimum(a: DTensor<S, F64>, b: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(a, b) { x, y -> if (x <= y) x else y }

/**
 * `clip(x, lo, hi)` with compile-time Double scalar bounds =
 * `minimum(maximum(x, lo), hi)`. Inside `grad {}` this composes as two
 * COMPARE+WHERE pairs against `lo`/`hi` splat consts of `x`'s shape, so the
 * gradient is exactly 1 where `lo ≤ x ≤ hi` and 0 outside (WhereRule routes
 * upstream to `x` on the in-bounds mask and to the const bound — zero
 * gradient — outside). Requires `lo ≤ hi`.
 */
fun <S : Shape> clip(x: DTensor<S, F64>, lo: Double, hi: Double): DTensor<S, F64> {
    require(lo <= hi) { "clip: lo ($lo) must be ≤ hi ($hi)" }
    return x.unary { v -> if (v < lo) lo else if (v > hi) hi else v }
}

/**
 * `outerProduct(a, b)` for rank-1 operands: `out[i, j] = a[i]·b[j]`,
 * result shape `[n, m]` (DiffKT's `concat(shapeA, shapeB)` specialised to
 * rank-1 ⊗ rank-1). Inside `grad {}` the K2 plugin lowers this to
 * `MATMUL(reshape(a, [n, 1]), reshape(b, [1, m]))` — the outer product IS a
 * `[n,1]×[1,m]` matmul — so the gradient flows through MatmulRule ∘ ReshapeRule
 * with no new AD math (`da_i = Σ_j upstream[i,j]·b[j]`, `db_j = Σ_i
 * upstream[i,j]·a[i]`). v1 scope: rank-1 ⊗ rank-1 (the only ranks that compose
 * cleanly through the existing rank-2 MATMUL synthesis arm). This host body is
 * the runtime twin.
 */
fun <SA : Shape, SB : Shape> outerProduct(
    a: DTensor<SA, F64>,
    b: DTensor<SB, F64>,
): DTensor<Shape, F64> {
    require(a.dims.size == 1 && b.dims.size == 1) {
        "outerProduct v1 requires rank-1 operands, got ${a.dims.toList()} ⊗ ${b.dims.toList()}"
    }
    val av = a.hostF64()
    val bv = b.hostF64()
    val n = av.size
    val m = bv.size
    val out = DoubleArray(n * m)
    for (i in 0 until n) {
        val ai = av[i]
        val rowOff = i * m
        for (j in 0 until m) out[rowOff + j] = ai * bv[j]
    }
    return DTensor(HostF64Storage(out), intArrayOf(n, m), F64)
}

fun <S : Shape> DTensor<S, F64>.sigmoid(): DTensor<S, F64> =
    unary { x -> 1.0 / (1.0 + kotlin.math.exp(-x)) }

fun <S : Shape> DTensor<S, F64>.tanh(): DTensor<S, F64> =
    unary { x -> kotlin.math.tanh(x) }

fun <S : Shape> DTensor<S, F64>.exp(): DTensor<S, F64> =
    unary { x -> kotlin.math.exp(x) }

fun <S : Shape> DTensor<S, F64>.log(): DTensor<S, F64> =
    unary { x -> kotlin.math.ln(x) }

fun <S : Shape> DTensor<S, F64>.sqrt(): DTensor<S, F64> =
    unary { x -> kotlin.math.sqrt(x) }

// Trigonometric functions, evaluated in Double (`DxirInterpreterF64`'s arms do the same).
fun <S : Shape> DTensor<S, F64>.tan(): DTensor<S, F64> =
    unary { x -> kotlin.math.tan(x) }

fun <S : Shape> DTensor<S, F64>.sin(): DTensor<S, F64> =
    unary { x -> kotlin.math.sin(x) }

fun <S : Shape> DTensor<S, F64>.cos(): DTensor<S, F64> =
    unary { x -> kotlin.math.cos(x) }

fun <S : Shape> DTensor<S, F64>.atan(): DTensor<S, F64> =
    unary { x -> kotlin.math.atan(x) }

// Special functions, through the shared Double kernels in `:core/SpecialFunctions.kt`
// (the same functions the interpreters' LGAMMA/DIGAMMA/TRIGAMMA arms call).
fun <S : Shape> DTensor<S, F64>.lgamma(): DTensor<S, F64> =
    unary { x -> x.lgamma() }

fun <S : Shape> DTensor<S, F64>.digamma(): DTensor<S, F64> =
    unary { x -> x.digamma() }

fun <S : Shape> DTensor<S, F64>.trigamma(): DTensor<S, F64> =
    unary { x -> x.trigamma() }

// §0.4.405 — polygamma(n), C1's recorded deferral. The order is a value
// parameter here but a compile-time Int literal inside `grad {}` (the FIR
// folds it: 0 → DIGAMMA, 1 → TRIGAMMA, n ≥ 2 → POLYGAMMA + `order` attr).
// One positional parameter, no defaults — the K2 named-arg landmine.
fun <S : Shape> DTensor<S, F64>.polygamma(n: Int): DTensor<S, F64> =
    unary { x -> x.polygamma(n) }

/**
 * Elementwise power, the user-facing entry to POW (PowRule, the interpreter
 * arm, `stablehlo.power` emission, the forward-mode tangent, and synthesis's
 * scalar `kotlin.math.pow` arm).
 * Three spellings, matching DiffKT's `pow(Double/Int/tensor-exponent)`: a tensor
 * exponent (elementwise, same shape) and Double / Int exponents — the K2 plugin
 * splats a literal exponent to the operand's shape, so the IR always sees the
 * uniform two-operand POW that PowRule expects.
 *
 * Computed in Double, as `DxirInterpreterF64`'s POW arm does.
 */
fun <S : Shape> DTensor<S, F64>.pow(exp: DTensor<S, F64>): DTensor<S, F64> =
    elementwise(this, exp) { a, b -> a.pow(b) }

fun <S : Shape> DTensor<S, F64>.pow(exp: Double): DTensor<S, F64> =
    unary { x -> x.pow(exp) }

fun <S : Shape> DTensor<S, F64>.pow(exp: Int): DTensor<S, F64> = pow(exp.toDouble())

/**
 * `softmax(axis)` over a single axis
 * (default last; negative axes count from the back). Numerically stable
 * (subtract the per-slice max before exponentiating), row-major stride walk
 * matching the dxir interpreter's SOFTMAX arm bit-for-bit so the host path
 * and the IR path agree. Shape-preserving — the phantom shape witness [S]
 * survives (unlike the axis reductions, which erase to [Shape]).
 */
fun <S : Shape> DTensor<S, F64>.softmax(axis: Int = -1): DTensor<S, F64> {
    val r = dims.size
    val a = if (axis < 0) axis + r else axis
    require(a in 0 until r) { "softmax: axis $axis out of range for rank $r" }
    val v = hostF64()
    val axisLen = dims[a]
    var inner = 1
    for (k in a + 1 until r) inner *= dims[k]
    var outer = 1
    for (k in 0 until a) outer *= dims[k]
    val out = DoubleArray(v.size)
    for (o in 0 until outer) {
        for (i in 0 until inner) {
            val base = o * axisLen * inner + i
            var mx = Double.NEGATIVE_INFINITY
            for (j in 0 until axisLen) mx = maxOf(mx, v[base + j * inner])
            var sum = 0.0
            for (j in 0 until axisLen) {
                val e = kotlin.math.exp(v[base + j * inner] - mx)
                out[base + j * inner] = e
                sum += e
            }
            for (j in 0 until axisLen) out[base + j * inner] /= sum
        }
    }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * `logSoftmax(axis)` = log(softmax(x, axis)), computed in the
 * stable `x - max - log(Σ exp(x - max))` form (never materialises the
 * softmax then logs it, which would lose precision in the tail). Inside
 * `grad {}` the K2 plugin lowers this to `LOG(SOFTMAX(x, axis))` — both
 * ops carry full VJP/JVP rules — so the gradient flows through the existing
 * LogRule ∘ SoftmaxRule chain; this host body is the runtime twin.
 */
fun <S : Shape> DTensor<S, F64>.logSoftmax(axis: Int = -1): DTensor<S, F64> {
    val r = dims.size
    val a = if (axis < 0) axis + r else axis
    require(a in 0 until r) { "logSoftmax: axis $axis out of range for rank $r" }
    val v = hostF64()
    val axisLen = dims[a]
    var inner = 1
    for (k in a + 1 until r) inner *= dims[k]
    var outer = 1
    for (k in 0 until a) outer *= dims[k]
    val out = DoubleArray(v.size)
    for (o in 0 until outer) {
        for (i in 0 until inner) {
            val base = o * axisLen * inner + i
            var mx = Double.NEGATIVE_INFINITY
            for (j in 0 until axisLen) mx = maxOf(mx, v[base + j * inner])
            var sum = 0.0
            for (j in 0 until axisLen) sum += kotlin.math.exp(v[base + j * inner] - mx)
            val logSum = kotlin.math.ln(sum)
            for (j in 0 until axisLen) out[base + j * inner] = v[base + j * inner] - mx - logSum
        }
    }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * `crossEntropyLoss(logits, oneHot)` =
 * the sum-reduced softmax cross entropy. Equal to `-Σ oneHot ⊙ logSoftmax(logits)`
 * over the last (class) axis, then summed over every position (the sum-reduction
 * convention — the total of the per-sample cross-entropies). `oneHot` is a float
 * one-hot (or soft) label tensor the same shape as `logits`. Inside `grad {}` the
 * K2 plugin lowers this to `NEG(SUM(MUL(oneHot, LOG(SOFTMAX(logits, -1)))))` —
 * every op fully-ruled — so the gradient flows with no new AD math; this host body
 * is the runtime twin. Returns a scalar so it composes with `.toDouble()`.
 */
fun <S : Shape> crossEntropyLoss(
    logits: DTensor<S, F64>,
    oneHot: DTensor<S, F64>,
): DTensor<ScalarShape, F64> {
    val ls = logits.logSoftmax(-1).hostF64()
    val oh = oneHot.hostF64()
    require(ls.size == oh.size) {
        "crossEntropyLoss: logits and oneHot must have the same element count (${ls.size} vs ${oh.size})"
    }
    var acc = 0.0
    for (i in ls.indices) acc -= oh[i] * ls[i]
    return DTensor(HostF64Storage(doubleArrayOf(acc)), intArrayOf(), F64)
}

/**
 * `nllLoss(logProbs, oneHot)` = the negative-log-likelihood loss on
 * already-log-normalised probabilities: `-Σ oneHot ⊙ logProbs`, summed over every
 * position. The companion to [crossEntropyLoss] for when the caller has already
 * applied `logSoftmax`. Inside `grad {}` it lowers to `NEG(SUM(MUL(oneHot, logProbs)))`.
 */
fun <S : Shape> nllLoss(
    logProbs: DTensor<S, F64>,
    oneHot: DTensor<S, F64>,
): DTensor<ScalarShape, F64> {
    val lp = logProbs.hostF64()
    val oh = oneHot.hostF64()
    require(lp.size == oh.size) {
        "nllLoss: logProbs and oneHot must have the same element count (${lp.size} vs ${oh.size})"
    }
    var acc = 0.0
    for (i in lp.indices) acc -= oh[i] * lp[i]
    return DTensor(HostF64Storage(doubleArrayOf(acc)), intArrayOf(), F64)
}

/**
 * Scalar → rank-N uniform broadcast: produce a fresh `DTensor<S, F64>` shaped like [template]
 * whose every element equals [v]. Used by the IR-rewrite synthesis path to lower
 * `OpKind.BROADCAST` in gradient bodies emitted by [io.tlaloc.ir.passes.VjpRegistry.SumRule]
 * (and later MeanRule). Keeping this as a `:core/ops` helper rather than inlining an IR-level
 * loop mirrors every other op's synthesis shape (`IrCall` into HostOps).
 *
 * The caller is responsible for ensuring [template]'s shape matches the intended target
 * shape; the helper copies [template.dims] rather than trusting the phantom type parameter,
 * since generic erasure strips `S` at runtime.
 */
/**
 * Scalar-index-into-rank-1 read (`arr[i]`). The `operator` form lets user code write
 * `arr[i]` inside a `grad` lambda; FIR sees `io.tlaloc.core.ops.get` and emits
 * `OpKind.GATHER(arr, idx)`. Generic in shape so we don't
 * fight Kotlin's operator resolver, but the compile path validates `rank == 1` —
 * calls on rank-2 tensors surface as `LoweringException` at FIR time. Runtime falls
 * back to `HostF64Storage` linear indexing, which is correct for rank-1 and
 * surprising for rank ≥ 2 (returns the `i`th element of the flat buffer, not a
 * row). Users who need multi-dim indexing should use tensor-slice helpers (not
 * yet shipped).
 *
 * Out-of-bounds indices propagate as standard `ArrayIndexOutOfBoundsException` from
 * the underlying `DoubleArray` — matches Kotlin stdlib conventions.
 */
operator fun <S : Shape> DTensor<S, F64>.get(i: Int): Double = hostF64()[i]

/**
 * Non-destructive scalar-index-into-rank-1 write (`scatter(arr, i, v)`). Returns a
 * fresh `DTensor` with the same shape as [base] and slot `[i]` replaced by [value].
 * Emitted by `DxirToIrSynthesis.irScatter` for gradient bodies — specifically the
 * `SCATTER(BROADCAST(0, arr.type), idx, upstream)` one-hot construction that
 * `GatherRule` uses to propagate a scalar adjoint back to a rank-1
 * primal array.
 *
 * Non-destructive by design: gradient accumulation in
 * [io.tlaloc.ir.passes.DxirReverseTransform] relies on value-semantics (each SCATTER produces a
 * fresh vector; outer `ADD`s combine them). Sharing `base`'s buffer and mutating in place would
 * break the accumulator's correctness.
 */
fun <S : Shape> scatter(base: DTensor<S, F64>, i: Int, value: Double): DTensor<S, F64> {
    val src = base.hostF64()
    val out = src.copyOf()
    out[i] = value
    return DTensor(HostF64Storage(out), base.dims.copyOf(), F64)
}

/**
 * Fused `base[i] += value` with fresh buffer. Returns a new `DTensor`
 * equal to [base] with slot `[i]` incremented by [value]; the other slots are
 * preserved from [base]. Emitted by `DxirToIrSynthesis.irScatterAdd` for the
 * `OpKind.SCATTER_ADD` op, which `GatherRule` emits instead of the 3-op
 * `BROADCAST(0) + SCATTER + ADD` chain.
 *
 * One allocation (output buffer) + one `copyOf` + one slot update. Down from
 * three allocations per gradient contribution in the unfused chain. For N gathers
 * in a gradient body, total transient allocations drop from ~3N to ~N (plus
 * one initial zero-broadcast for the first gather's accumulator seed).
 */
fun <S : Shape> scatterAddInto(base: DTensor<S, F64>, i: Int, value: Double): DTensor<S, F64> {
    val src = base.hostF64()
    val out = src.copyOf()
    out[i] = out[i] + value
    return DTensor(HostF64Storage(out), base.dims.copyOf(), F64)
}

/**
 * Destructive `base[i] += value`. **Mutates [base]'s internal buffer
 * in place** and returns the SAME `DTensor` wrapper.
 *
 * **ONLY call this when you have exclusive ownership of [base].** The intended
 * caller is synthesised gradient code emitted by `DxirToIrSynthesis.irScatterAdd`
 * for `OpKind.SCATTER_ADD` ops whose `operand[0]` has been proven single-use by
 * `DxirReverseTransform.tagSingleUseScatterAdds`'s SSA use-graph analysis.
 * Invoking this on a DTensor that's aliased elsewhere silently corrupts shared
 * state — no runtime check catches it.
 *
 * Perf win over [scatterAddInto]: saves the per-call `DoubleArray.copyOf` +
 * DTensor allocation. For a rank-1 tensor of size N, that's O(N) memory copies
 * + one JVM allocation eliminated per gradient-accumulator step.
 */
fun <S : Shape> scatterAddInPlace(base: DTensor<S, F64>, i: Int, value: Double): DTensor<S, F64> {
    val buf = base.hostF64()
    buf[i] = buf[i] + value
    return base
}

fun <S : Shape> broadcastLike(v: Double, template: DTensor<S, F64>): DTensor<S, F64> {
    val n = template.size
    val out = DoubleArray(n) { v }
    return DTensor(HostF64Storage(out), template.dims.copyOf(), F64)
}

/**
 * Scalar → rank-N uniform broadcast keyed
 * by an explicit [dims] array rather than a runtime template DTensor. Mirrors
 * [broadcastLike]'s output structurally — `DTensor<S, F64>` with `dims = dims.copyOf()`
 * and `storage = DoubleArray(prod(dims)) { v }` — but lifts the shape source out of the
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
fun <S : Shape> broadcastDims(v: Double, dims: IntArray): DTensor<S, F64> {
    val n = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
    val out = DoubleArray(n) { v }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * Rank-specific delegates for
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
 * each call is a regular `IrCall` with one `Double` + N `Int` arguments, which
 * `IrCallImpl.fromSymbolOwner` handles natively.
 */
fun <S : Shape> broadcastDimsRank1(v: Double, d0: Int): DTensor<S, F64> =
    broadcastDims(v, intArrayOf(d0))

fun <S : Shape> broadcastDimsRank2(v: Double, d0: Int, d1: Int): DTensor<S, F64> =
    broadcastDims(v, intArrayOf(d0, d1))

fun <S : Shape> broadcastDimsRank3(v: Double, d0: Int, d1: Int, d2: Int): DTensor<S, F64> =
    broadcastDims(v, intArrayOf(d0, d1, d2))

/**
 * DTensor → Double bridge for grad lambdas. The K2 plugin recognises
 * this call site (via FirLambdaToDxirLowering) as a no-op at the dxir level —
 * `DxirType(F64, [])` is the same whether the value flows through a DTensor
 * wrapper or a primitive Double. Closes the last gap for end-to-end MATMUL /
 * SUM-bearing gradient lambdas where the body must terminate in a Double.
 */
fun DTensor<ScalarShape, F64>.toDouble(): Double = hostF64()[0]

fun <S : Shape> DTensor<S, F64>.sum(): DTensor<ScalarShape, F64> {
    val v = hostF64()
    var acc = 0.0
    for (x in v) acc += x
    return DTensor(HostF64Storage(doubleArrayOf(acc)), intArrayOf(), F64)
}

fun <S : Shape> DTensor<S, F64>.mean(): DTensor<ScalarShape, F64> {
    val v = hostF64()
    if (v.isEmpty()) return DTensor(HostF64Storage(doubleArrayOf(0.0)), intArrayOf(), F64)
    var acc = 0.0
    for (x in v) acc += x
    return DTensor(HostF64Storage(doubleArrayOf(acc / v.size)), intArrayOf(), F64)
}

/**
 * DiffKT's `stats()` — the `(mean, variance)`
 * pair over all elements, thin sugar over the full reductions. Variance is
 * BIASED (divide by N, not N−1), matching both DiffKT's convention and the
 * per-channel statistic `batchNormGeneral` takes. Host-level only
 * by design: DiffKT's `stats` is a convenience accessor, not a
 * differentiation surface — a Pair-returning body has no `grad {}` lowering
 * (the loss contract is scalar), and a loss that needs the pieces writes
 * `x.mean()` and the squared-deviation mean directly, both of which
 * differentiate today.
 */
fun <S : Shape> DTensor<S, F64>.stats(): Pair<DTensor<ScalarShape, F64>, DTensor<ScalarShape, F64>> {
    val v = hostF64()
    if (v.isEmpty()) {
        val zero = { DTensor<ScalarShape, F64>(HostF64Storage(doubleArrayOf(0.0)), intArrayOf(), F64) }
        return Pair(zero(), zero())
    }
    var acc = 0.0
    for (x in v) acc += x
    val mu = acc / v.size
    var sq = 0.0
    for (x in v) {
        val d = x - mu
        sq += d * d
    }
    return Pair(
        DTensor(HostF64Storage(doubleArrayOf(mu)), intArrayOf(), F64),
        DTensor(HostF64Storage(doubleArrayOf(sq / v.size)), intArrayOf(), F64),
    )
}

/**
 * Axis-wise reduction engine (DiffKT's
 * `sum(vararg axes: Int, keepDims: Boolean)` family). Reduces [x] over the
 * axes in [dims] (negative axes count from the back), accumulating with [acc]
 * from [init]; [finish] maps (accumulated, reducedElementCount) → output
 * element (mean divides, sum/max/min pass through). `keepDims = true` keeps
 * the reduced axes as size-1; `false` drops them (rank shrinks). Row-major
 * stride walk, matching the dxir interpreter's SUM/MEAN/MAX/MIN evals so the
 * host path and the IR path agree bit-for-bit on iteration order.
 */
private fun <S : Shape> reduceOver(
    x: DTensor<S, F64>,
    dims: IntArray,
    keepDims: Boolean,
    init: Double,
    acc: (Double, Double) -> Double,
    finish: (Double, Int) -> Double = { a, _ -> a },
): DTensor<Shape, F64> {
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
    val v = x.hostF64()
    val out = DoubleArray(outSize) { init }
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
    return DTensor(HostF64Storage(out), outDims, F64)
}

/**
 * Axis-wise reductions, the DiffKT `sum(axes, keepDims)` user
 * surface. The result's shape type is erased to [Shape]: the
 * output dims depend on the runtime axis list, which Kotlin's phantom shape
 * typing cannot express per-overload. Inside `grad {}` the K2 plugin computes
 * the exact `DxirType` from the operand's dims and the constant axis
 * arguments, so the IR stays precisely shaped; only the host-side Kotlin
 * static type widens. The no-arg overloads above keep their `ScalarShape`.
 */
fun <S : Shape> DTensor<S, F64>.sum(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F64> =
    reduceOver(this, dims, keepDims, 0.0, { a, b -> a + b })

fun <S : Shape> DTensor<S, F64>.mean(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F64> =
    reduceOver(this, dims, keepDims, 0.0, { a, b -> a + b }, { a, n -> a / n })

fun <S : Shape> DTensor<S, F64>.max(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F64> =
    reduceOver(this, dims, keepDims, Double.NEGATIVE_INFINITY, { a, b -> if (b > a) b else a })

fun <S : Shape> DTensor<S, F64>.min(vararg dims: Int, keepDims: Boolean = false): DTensor<Shape, F64> =
    reduceOver(this, dims, keepDims, Double.POSITIVE_INFINITY, { a, b -> if (b < a) b else a })

/** Full-reduce extremum companions to [sum]/[mean] (DiffKT defaults `axes = allAxes`). */
fun <S : Shape> DTensor<S, F64>.max(): DTensor<ScalarShape, F64> {
    val v = hostF64()
    require(v.isNotEmpty()) { "max: empty tensor" }
    var m = Double.NEGATIVE_INFINITY
    for (x in v) if (x > m) m = x
    return DTensor(HostF64Storage(doubleArrayOf(m)), intArrayOf(), F64)
}

fun <S : Shape> DTensor<S, F64>.min(): DTensor<ScalarShape, F64> {
    val v = hostF64()
    require(v.isNotEmpty()) { "min: empty tensor" }
    var m = Double.POSITIVE_INFINITY
    for (x in v) if (x < m) m = x
    return DTensor(HostF64Storage(doubleArrayOf(m)), intArrayOf(), F64)
}

/**
 * Fixed-arity synthesis delegates for the axis reductions, one per
 * (kind, axis-count) pair. Same reason as [broadcastDimsRank1]:
 * the K2 synthesis cannot build `IrVararg` nodes, so `DxirToIrSynthesis`
 * emits calls to these with plain `Int` + `Boolean` const arguments read off
 * the dxir op's `reduction_dims` attr and result type. Runtime semantics are
 * identical to the vararg user surface above.
 */
fun <S : Shape> sumOver1(x: DTensor<S, F64>, d0: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.sum(d0, keepDims = keepDims)

fun <S : Shape> sumOver2(x: DTensor<S, F64>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.sum(d0, d1, keepDims = keepDims)

fun <S : Shape> meanOver1(x: DTensor<S, F64>, d0: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.mean(d0, keepDims = keepDims)

fun <S : Shape> meanOver2(x: DTensor<S, F64>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.mean(d0, d1, keepDims = keepDims)

fun <S : Shape> maxOver1(x: DTensor<S, F64>, d0: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.max(d0, keepDims = keepDims)

fun <S : Shape> maxOver2(x: DTensor<S, F64>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.max(d0, d1, keepDims = keepDims)

fun <S : Shape> minOver1(x: DTensor<S, F64>, d0: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.min(d0, keepDims = keepDims)

fun <S : Shape> minOver2(x: DTensor<S, F64>, d0: Int, d1: Int, keepDims: Boolean): DTensor<Shape, F64> =
    x.min(d0, d1, keepDims = keepDims)

/**
 * The three-axis shims. A rank-4 NCHW surface needs them: reducing over
 * the batch and both spatial axes (`mean(0, 2, 3)`, which is how training batchNorm
 * takes its per-channel statistics) is a three-axis reduction, and the fixed-arity
 * family stopped at two, so such a body fell out of synthesis scope.
 */
fun <S : Shape> sumOver3(
    x: DTensor<S, F64>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F64> = x.sum(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> meanOver3(
    x: DTensor<S, F64>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F64> = x.mean(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> maxOver3(
    x: DTensor<S, F64>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F64> = x.max(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> minOver3(
    x: DTensor<S, F64>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F64> = x.min(d0, d1, d2, keepDims = keepDims)

/**
 * Stretch broadcast: tile [x] (whose dims must each be 1 or equal
 * the target) up to the target dims. This is the host twin of the dxir
 * interpreter's keepdims-stretch BROADCAST arm — the shape the reduction
 * VJP rules emit when un-reducing an upstream back over the reduced axes
 * (`RESHAPE to keepdims` → `BROADCAST stretch to x.dims`). The splat helper
 * [broadcastDims] fills a constant; this tiles a tensor — different op.
 * Fixed-arity rank delegates below for the same IrVararg reason as
 * [broadcastDimsRank1].
 */
private fun <S : Shape> stretchTo(x: DTensor<*, F64>, target: IntArray): DTensor<S, F64> {
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
    val v = x.hostF64()
    val out = DoubleArray(outSize)
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
    return DTensor(HostF64Storage(out), target.copyOf(), F64)
}

fun <S : Shape> stretchToRank1(x: DTensor<*, F64>, d0: Int): DTensor<S, F64> =
    stretchTo(x, intArrayOf(d0))

fun <S : Shape> stretchToRank2(x: DTensor<*, F64>, d0: Int, d1: Int): DTensor<S, F64> =
    stretchTo(x, intArrayOf(d0, d1))

fun <S : Shape> stretchToRank3(x: DTensor<*, F64>, d0: Int, d1: Int, d2: Int): DTensor<S, F64> =
    stretchTo(x, intArrayOf(d0, d1, d2))

/**
 * Template-shaped stretch: tile [x] up to [template]'s runtime
 * dims. The synthesis fallback when structural axis-matching against the
 * function params fails (mirrors `broadcastLike(v, template)` for splats).
 */
fun <S : Shape> stretchLike(x: DTensor<*, F64>, template: DTensor<S, F64>): DTensor<S, F64> =
    stretchTo(x, template.dims)

/**
 * Numpy unbroadcast: reduce [value] down to [template]'s RUNTIME
 * dims — the reverse mirror of [stretchLike]. This is the synthesis/plugin
 * twin of the dxir interpreter's SUM_TO arm: BroadcastRule's adjoint for the
 * in-place size-1 stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`) sums `value` over the
 * leading (value.rank − template.rank) axes AND over every aligned axis where
 * the template extent is 1 but `value`'s is > 1, keeping those axes size-1.
 * Which axes were size-1-stretched is unknowable at compile time under the -1
 * sentinel dims of `grad {}`, so the extent is read from [template]'s ACTUAL
 * runtime shape here. [template] contributes SHAPE ONLY — its values are never
 * read.
 */
/**
 * The customVjp host twin of the dxir
 * `CHECK_SHAPE_LIKE` op — a value-identity that ASSERTS the user vjpFn's
 * returned gradient has its operand's runtime shape before it is accumulated.
 * [template] is the customVjp operand the contribution belongs to and
 * contributes SHAPE ONLY (its values are never read). Under `grad {}`'s -1
 * sentinel dims the contract is undecidable at compile time, so this is where
 * a wrong-shaped user adjoint fails LOUDLY instead of silently corrupting the
 * gradient (the same template-assert pattern `conv2dDataAdjoint`
 * uses). The value passes through as a re-wrapped view — no copy, its
 * storage is the user body's freshly computed tensor.
 */
fun <S : Shape> checkShapeLike(value: DTensor<*, F64>, template: DTensor<S, F64>): DTensor<S, F64> {
    require(value.dims.contentEquals(template.dims)) {
        "checkShapeLike: customVjp gradient body returned shape ${value.dims.toList()} for an " +
            "operand of shape ${template.dims.toList()} — the user vjpFn violates the VJP shape " +
            "contract (each d_operand must match its operand's shape)"
    }
    return DTensor(value.storage, value.dims, F64)
}

fun <S : Shape> sumToLike(value: DTensor<*, F64>, template: DTensor<S, F64>): DTensor<S, F64> {
    val u = value.dims
    val t = template.dims
    // Phase A5c — identity fast path. The elementwise binary VjpRules now wrap
    // every contribution whose shape is not PROVABLY its operand's in a SUM_TO
    // (that is what makes them correct under `grad {}`'s -1 sentinel dims, where
    // "provably" is never available), so at RUNTIME the shapes usually do match and
    // there is nothing to reduce. Copy instead of walking the stride arithmetic —
    // and copy rather than alias, because a contribution may feed an in-place
    // accumulator downstream.
    if (u.contentEquals(t)) {
        return DTensor(HostF64Storage(value.hostF64().copyOf()), t.copyOf(), F64)
    }
    val ru = u.size
    val rt = t.size
    require(rt <= ru) { "sumToLike: template rank $rt exceeds value rank $ru" }
    val offset = ru - rt
    for (i in 0 until rt) require(t[i] == u[offset + i] || t[i] == 1) {
        "sumToLike: template dim $i = ${t[i]} incompatible with value axis ${offset + i} = ${u[offset + i]} (must be equal or 1)"
    }
    val inStrides = IntArray(ru)
    run { var s = 1; for (i in ru - 1 downTo 0) { inStrides[i] = s; s *= u[i] } }
    val outStrides = IntArray(rt)
    run { var s = 1; for (i in rt - 1 downTo 0) { outStrides[i] = s; s *= t[i] } }
    var outSize = 1
    for (d in t) outSize *= d
    val v = value.hostF64()
    val out = DoubleArray(outSize)
    for (flat in v.indices) {
        var rem = flat
        var outIdx = 0
        for (k in 0 until ru) {
            val coord = rem / inStrides[k]
            rem -= coord * inStrides[k]
            val tAxis = k - offset
            if (tAxis >= 0 && t[tAxis] != 1) outIdx += coord * outStrides[tAxis]
        }
        out[outIdx] += v[flat]
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/**
 * Broadcast-to-template: stretch [value] up to [template]'s RUNTIME
 * dims under NumPy right-alignment — the forward twin (and VJP) of
 * [sumToLike], and the synthesis/plugin twin of the dxir interpreter's
 * BROADCAST_LIKE arm. Each aligned axis of [value] must equal the template's
 * or be size-1; missing leading axes are replicated. This is SumToRule's
 * adjoint: the upstream (shaped like SUM_TO's template) is broadcast back up
 * to the value operand's shape, which is a -1 sentinel at compile time under
 * `grad {}`, so the target extents are read from [template]'s ACTUAL runtime
 * shape here. [template] contributes SHAPE ONLY — its values are never read.
 * Rank-polymorphic like [sumToLike] (no `…RankN` shims needed: no attrs to
 * bake), unlike the equal-rank-only [stretchLike].
 */
fun <S : Shape> broadcastToLike(value: DTensor<*, F64>, template: DTensor<S, F64>): DTensor<S, F64> {
    val u = value.dims
    val t = template.dims
    // Identity fast path — copy rather than alias (a contribution may feed an
    // in-place accumulator downstream), the sumToLike convention.
    if (u.contentEquals(t)) {
        return DTensor(HostF64Storage(value.hostF64().copyOf()), t.copyOf(), F64)
    }
    val ru = u.size
    val rt = t.size
    require(ru <= rt) { "broadcastToLike: value rank $ru exceeds template rank $rt" }
    val offset = rt - ru
    for (i in 0 until ru) require(u[i] == t[offset + i] || u[i] == 1) {
        "broadcastToLike: value dim $i = ${u[i]} incompatible with template axis ${offset + i} = ${t[offset + i]} (must be equal or 1)"
    }
    val inStrides = IntArray(ru)
    run { var s = 1; for (i in ru - 1 downTo 0) { inStrides[i] = s; s *= u[i] } }
    val outStrides = IntArray(rt)
    run { var s = 1; for (i in rt - 1 downTo 0) { outStrides[i] = s; s *= t[i] } }
    var outSize = 1
    for (d in t) outSize *= d
    val v = value.hostF64()
    val out = DoubleArray(outSize)
    for (flat in out.indices) {
        var rem = flat
        var src = 0
        for (k in 0 until rt) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            val uAxis = k - offset
            if (uAxis >= 0 && u[uAxis] != 1) src += coord * inStrides[uAxis]
        }
        out[flat] = v[src]
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/**
 * Zero-pad-to-template: place [value] into a zero tensor of
 * [template]'s RUNTIME dims at offset [low] per axis — the reverse mirror of
 * [slice] and the synthesis/plugin twin of the dxir interpreter's PAD_TO arm.
 * This is SliceRule's adjoint: the upstream gradient is zero-padded back into
 * the sliced operand's window. The trailing pad per axis (`high[i] =
 * template.dim[i] − low[i] − value.dim[i]`) reads [template]'s extent, which is
 * a -1 sentinel at compile time under `grad {}`, so it is derived from the
 * template's ACTUAL runtime shape here. [low] is a compile-time literal (the
 * user's `slice` start offsets, 0 on the non-sliced axes). [template]
 * contributes SHAPE ONLY — its values are never read.
 */
fun <S : Shape> padToLike(value: DTensor<*, F64>, template: DTensor<S, F64>, low: IntArray): DTensor<S, F64> {
    val u = value.dims
    val t = template.dims
    val r = t.size
    require(u.size == r) { "padToLike: value rank ${u.size} != template rank $r" }
    require(low.size == r) { "padToLike: low size ${low.size} != rank $r" }
    for (i in 0 until r) require(low[i] >= 0 && low[i] + u[i] <= t[i]) {
        "padToLike: axis $i: low ${low[i]} + value ${u[i]} exceeds template ${t[i]}"
    }
    val inStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= u[i] } }
    val outStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { outStrides[i] = s; s *= t[i] } }
    var outSize = 1
    for (d in t) outSize *= d
    val v = value.hostF64()
    val out = DoubleArray(outSize)
    for (flat in v.indices) {
        var rem = flat
        var dst = 0
        for (k in 0 until r) {
            val coord = rem / inStrides[k]
            rem -= coord * inStrides[k]
            dst += (coord + low[k]) * outStrides[k]
        }
        out[dst] = v[flat]
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/** Fixed-arity `padToLike` shims (synthesis bakes the `low`
 * offsets as Int consts, one per axis; mirror of the `stretchToRankN` family). */
fun <S : Shape> padToLikeRank1(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int): DTensor<S, F64> =
    padToLike(value, template, intArrayOf(l0))

fun <S : Shape> padToLikeRank2(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int, l1: Int): DTensor<S, F64> =
    padToLike(value, template, intArrayOf(l0, l1))

fun <S : Shape> padToLikeRank3(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int, l1: Int, l2: Int): DTensor<S, F64> =
    padToLike(value, template, intArrayOf(l0, l1, l2))

/**
 * Window-at-literal-offset: cut out of [value] the window of
 * [template]'s RUNTIME dims starting at [low] per axis — the reverse mirror
 * (and VJP) of [padToLike], and the synthesis/plugin twin of the dxir
 * interpreter's SLICE_AT arm. This is PadToRule's adjoint: the upstream
 * (shaped like PAD_TO's template) is sliced back down to the value operand's
 * window, whose extents are -1 sentinels at compile time under `grad {}`, so
 * they are read from [template]'s ACTUAL runtime shape here; [low] is the
 * PAD_TO node's own literal offset, carried verbatim. Differs from
 * [sliceLikeStart]/[sliceLikeAfter1], whose offset is a runtime SUM of prior
 * templates' extents along one axis. [template] contributes SHAPE ONLY — its
 * values are never read.
 */
fun <S : Shape> sliceAtLike(value: DTensor<*, F64>, template: DTensor<S, F64>, low: IntArray): DTensor<S, F64> {
    val u = value.dims
    val t = template.dims
    val r = t.size
    require(u.size == r) { "sliceAtLike: value rank ${u.size} != template rank $r" }
    require(low.size == r) { "sliceAtLike: low size ${low.size} != rank $r" }
    for (i in 0 until r) require(low[i] >= 0 && low[i] + t[i] <= u[i]) {
        "sliceAtLike: axis $i: low ${low[i]} + template ${t[i]} exceeds value ${u[i]}"
    }
    val inStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= u[i] } }
    val outStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { outStrides[i] = s; s *= t[i] } }
    var outSize = 1
    for (d in t) outSize *= d
    val v = value.hostF64()
    val out = DoubleArray(outSize)
    for (flat in out.indices) {
        var rem = flat
        var src = 0
        for (k in 0 until r) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            src += (coord + low[k]) * inStrides[k]
        }
        out[flat] = v[src]
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/** Fixed-arity `sliceAtLike` shims (synthesis bakes the `low`
 * offsets as Int consts, one per axis; mirror of the `padToLikeRankN` family). */
fun <S : Shape> sliceAtLikeRank1(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int): DTensor<S, F64> =
    sliceAtLike(value, template, intArrayOf(l0))

fun <S : Shape> sliceAtLikeRank2(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int, l1: Int): DTensor<S, F64> =
    sliceAtLike(value, template, intArrayOf(l0, l1))

fun <S : Shape> sliceAtLikeRank3(value: DTensor<*, F64>, template: DTensor<S, F64>, l0: Int, l1: Int, l2: Int): DTensor<S, F64> =
    sliceAtLike(value, template, intArrayOf(l0, l1, l2))

/**
 * The host twin of dxir `SLICE_LIKE`, i.e. CONCAT's adjoint: cut out
 * of [value] the window along [axis] that starts after every one of [priors] and
 * runs for [thisTemplate]'s extent, taking every other axis whole.
 *
 * Both bounds are read from the templates' ACTUAL runtime dims. That is the whole
 * point: a concat operand's window offset is the cumulative sum of the PRIOR
 * operands' runtime axis extents, which under `grad {}`'s -1 sentinel dims does not
 * exist at compile time, so it cannot ride as an attr the way `slice`'s literal
 * start/end do (cf. [padToLike], which bakes `low` because a slice offset IS a
 * literal). The templates contribute SHAPE ONLY — their values are never read.
 *
 * [padToLike] puts a window back INTO a shape; this cuts one OUT of it.
 */
private fun <S : Shape> sliceWindow(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    axis: Int,
    priors: List<DTensor<*, F64>>,
): DTensor<S, F64> {
    val v = value.dims
    val t = thisTemplate.dims
    val r = v.size
    require(t.size == r) { "sliceWindow: template rank ${t.size} != value rank $r" }
    require(axis in 0 until r) { "sliceWindow: axis $axis outside rank $r" }
    val start = priors.sumOf { it.dims[axis] }
    val len = t[axis]
    require(start >= 0 && start + len <= v[axis]) {
        "sliceWindow: window [$start, ${start + len}) exceeds the value's axis-$axis extent ${v[axis]}"
    }
    for (i in 0 until r) {
        require(i == axis || t[i] == v[i]) {
            "sliceWindow: non-axis $i template extent ${t[i]} != value extent ${v[i]}"
        }
    }
    var outer = 1
    for (k in 0 until axis) outer *= v[k]
    var inner = 1
    for (k in axis + 1 until r) inner *= v[k]
    val src = value.hostF64()
    val out = DoubleArray(outer * len * inner)
    var dst = 0
    for (o in 0 until outer) {
        val from = o * (v[axis] * inner) + start * inner
        src.copyInto(out, dst, from, from + len * inner)
        dst += len * inner
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/** Fixed-arity `SLICE_LIKE` twins, one per PRIOR-template count — the usual
 * IrVararg reason (see [broadcastDimsRank1]): synthesis builds positional
 * `IrCall` arguments, so the operand count has to be in the callee's name.
 * The family covers up to 7 priors (an 8-operand IR-level
 * CONCAT); the user-facing [concat]'s fold-to-binary never needs more than
 * ONE prior, so the wider twins serve only hand-built variadic CONCAT nodes
 * differentiated through the plugin. */
fun <S : Shape> sliceLikeStart(value: DTensor<*, F64>, thisTemplate: DTensor<S, F64>, axis: Int): DTensor<S, F64> =
    sliceWindow(value, thisTemplate, axis, emptyList())

fun <S : Shape> sliceLikeAfter1(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = sliceWindow(value, thisTemplate, axis, listOf(prior0))

fun <S : Shape> sliceLikeAfter2(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1))

fun <S : Shape> sliceLikeAfter3(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2))

fun <S : Shape> sliceLikeAfter4(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3))

fun <S : Shape> sliceLikeAfter5(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4))

fun <S : Shape> sliceLikeAfter6(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    prior5: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> =
    sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5))

fun <S : Shape> sliceLikeAfter7(
    value: DTensor<*, F64>,
    thisTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    prior5: DTensor<*, F64>,
    prior6: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> =
    sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5, prior6))

/**
 * The host twin of dxir `PAD_LIKE`, i.e. SLICE_LIKE's transpose and
 * VJP: place [value] into a zero tensor of [outTemplate]'s RUNTIME dims at the
 * window along [axis] that starts after every one of [priors] (every other
 * axis at 0).
 *
 * Both the offset and the target extent are read from the templates' ACTUAL
 * runtime dims, for the same reason [sliceWindow] reads its bounds there: a
 * concat window's offset is the cumulative sum of the PRIOR operands' runtime
 * axis extents, which under `grad {}`'s -1 sentinel dims does not exist at
 * compile time — so it cannot ride as a literal the way [padToLike]'s `low`
 * does. The templates contribute SHAPE ONLY — their values are never read.
 *
 * [sliceWindow] cuts a window OUT of a shape; this puts one back INTO it.
 */
private fun <S : Shape> padWindow(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    axis: Int,
    priors: List<DTensor<*, F64>>,
): DTensor<S, F64> {
    val v = value.dims
    val t = outTemplate.dims
    val r = v.size
    require(t.size == r) { "padWindow: outTemplate rank ${t.size} != value rank $r" }
    require(axis in 0 until r) { "padWindow: axis $axis outside rank $r" }
    val start = priors.sumOf { it.dims[axis] }
    val len = v[axis]
    require(start >= 0 && start + len <= t[axis]) {
        "padWindow: window [$start, ${start + len}) exceeds the outTemplate's axis-$axis extent ${t[axis]}"
    }
    for (i in 0 until r) {
        require(i == axis || t[i] == v[i]) {
            "padWindow: non-axis $i outTemplate extent ${t[i]} != value extent ${v[i]}"
        }
    }
    var outer = 1
    for (k in 0 until axis) outer *= t[k]
    var inner = 1
    for (k in axis + 1 until r) inner *= t[k]
    var outSize = 1
    for (d in t) outSize *= d
    val src = value.hostF64()
    val out = DoubleArray(outSize)
    var from = 0
    for (o in 0 until outer) {
        val dst = o * (t[axis] * inner) + start * inner
        src.copyInto(out, dst, from, from + len * inner)
        from += len * inner
    }
    return DTensor(HostF64Storage(out), t.copyOf(), F64)
}

/** Fixed-arity `PAD_LIKE` twins, one per PRIOR-template count — the usual
 * IrVararg reason (see [sliceLikeStart]): synthesis builds positional
 * `IrCall` arguments, so the operand count has to be in the callee's name.
 * The family covers up to 7 priors, mirroring the `SLICE_LIKE`
 * twins — the pair must stay closed under differentiation (SLICE_LIKE's VJP
 * is PAD_LIKE with the SAME priors), so the two bounds move together. */
fun <S : Shape> padLikeStart(value: DTensor<*, F64>, outTemplate: DTensor<S, F64>, axis: Int): DTensor<S, F64> =
    padWindow(value, outTemplate, axis, emptyList())

fun <S : Shape> padLikeAfter1(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = padWindow(value, outTemplate, axis, listOf(prior0))

fun <S : Shape> padLikeAfter2(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = padWindow(value, outTemplate, axis, listOf(prior0, prior1))

fun <S : Shape> padLikeAfter3(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2))

fun <S : Shape> padLikeAfter4(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3))

fun <S : Shape> padLikeAfter5(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4))

fun <S : Shape> padLikeAfter6(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    prior5: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> =
    padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5))

fun <S : Shape> padLikeAfter7(
    value: DTensor<*, F64>,
    outTemplate: DTensor<S, F64>,
    prior0: DTensor<*, F64>,
    prior1: DTensor<*, F64>,
    prior2: DTensor<*, F64>,
    prior3: DTensor<*, F64>,
    prior4: DTensor<*, F64>,
    prior5: DTensor<*, F64>,
    prior6: DTensor<*, F64>,
    axis: Int,
): DTensor<S, F64> =
    padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5, prior6))

/**
 * The two-operand concat the K2 plugin synthesises with.
 *
 * Fixed arity on purpose: synthesis builds positional `IrCall` arguments and cannot
 * construct an `IrVararg` (the documented reason the whole `…RankN` shim family
 * exists), so the user-facing [concat] below folds n operands into a right-fold of
 * these. Concat is associative along the axis, so the fold is semantics-preserving;
 * it costs one extra pass per intermediate, which is the price of never needing a
 * variadic call.
 *
 * Non-axis extents must agree and every operand must share the rank — the same
 * contract the dxir interpreter's CONCAT arm and `stablehlo.concatenate` have.
 */
fun <R : Shape> concatPair(axis: Int, a: DTensor<*, F64>, b: DTensor<*, F64>): DTensor<R, F64> {
    val ad = a.dims
    val bd = b.dims
    require(ad.size == bd.size) {
        "concatPair: ranks differ (${ad.toList()} vs ${bd.toList()}); concat does not broadcast"
    }
    require(axis in ad.indices) { "concatPair: axis $axis outside rank ${ad.size}" }
    for (i in ad.indices) {
        require(i == axis || ad[i] == bd[i]) {
            "concatPair: non-axis $i extents differ (${ad[i]} vs ${bd[i]})"
        }
    }
    var outer = 1
    for (k in 0 until axis) outer *= ad[k]
    var inner = 1
    for (k in axis + 1 until ad.size) inner *= ad[k]
    val aRun = ad[axis] * inner
    val bRun = bd[axis] * inner
    val av = a.hostF64()
    val bv = b.hostF64()
    val out = DoubleArray(outer * (aRun + bRun))
    var dst = 0
    for (o in 0 until outer) {
        av.copyInto(out, dst, o * aRun, o * aRun + aRun)
        dst += aRun
        bv.copyInto(out, dst, o * bRun, o * bRun + bRun)
        dst += bRun
    }
    val outDims = ad.copyOf().also { it[axis] = ad[axis] + bd[axis] }
    return DTensor(HostF64Storage(out), outDims, F64)
}

/**
 * Concatenate [tensors] along [axis]. The result erases
 * to `DTensor<Shape, F64>`: the concat axis's extent is a runtime SUM of the
 * operands' extents, which no static shape witness can carry — the same convention
 * `slice`, `reshape` and `broadcastTo` follow. Differentiable in `grad {}`: the
 * adjoint gives each operand its window of the upstream via `SLICE_LIKE`, whose
 * bounds are read off the operands' runtime shapes.
 */
fun concat(axis: Int, vararg tensors: DTensor<*, F64>): DTensor<Shape, F64> {
    require(tensors.size >= 2) { "concat: needs at least 2 tensors, got ${tensors.size}" }
    var acc: DTensor<Shape, F64> = concatPair(axis, tensors[0], tensors[1])
    for (i in 2 until tensors.size) acc = concatPair(axis, acc, tensors[i])
    return acc
}

/**
 * `stack(axis, tensors)`: give each operand a new
 * size-1 axis at [axis], then concatenate along it, so the result has rank
 * `operand.rank + 1`. Pure sugar over [unsqueeze] + [concat], which is exactly how
 * the K2 plugin lowers it.
 */
fun stack(axis: Int, vararg tensors: DTensor<*, F64>): DTensor<Shape, F64> {
    require(tensors.size >= 2) { "stack: needs at least 2 tensors, got ${tensors.size}" }
    val lifted = tensors.map { it.unsqueeze(axis) }
    var acc: DTensor<Shape, F64> = concatPair(axis, lifted[0], lifted[1])
    for (i in 2 until lifted.size) acc = concatPair(axis, acc, lifted[i])
    return acc
}

/**
 * Insert size-1 axes at the given (result-indexed, ascending)
 * positions. The host twin of the keepdims RESHAPE the reduction VJP rules
 * emit (`upstream` at the squeezed shape → the keepdims spelling): axis
 * POSITIONS are compile-time constants from `reduction_dims`, so the
 * synthesis can bake them as Int consts without touching runtime dims —
 * dims themselves may be symbolic sentinels at compile time. Fixed-arity
 * variants for the usual IrVararg reason.
 */
private fun unsqueezeAxes(x: DTensor<*, F64>, axes: IntArray): IntArray {
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

fun <S : Shape> unsqueezeAxes1(x: DTensor<*, F64>, a0: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(x.hostF64().copyOf()), unsqueezeAxes(x, intArrayOf(a0)), F64)

fun <S : Shape> unsqueezeAxes2(x: DTensor<*, F64>, a0: Int, a1: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(x.hostF64().copyOf()), unsqueezeAxes(x, intArrayOf(a0, a1)), F64)

/**
 * Three inserted unit axes: what `[C] → [1,C,1,1]` needs, i.e. how a
 * per-channel parameter becomes broadcastable against an NCHW tensor. Without it a
 * rank-4 body containing that reshape falls out of synthesis scope.
 */
fun <S : Shape> unsqueezeAxes3(x: DTensor<*, F64>, a0: Int, a1: Int, a2: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(x.hostF64().copyOf()), unsqueezeAxes(x, intArrayOf(a0, a1, a2)), F64)

/**
 * The RESHAPE-family user surface.
 * `squeeze(axis)` drops a size-1 axis, `unsqueeze(axis)` inserts one,
 * `flatten()` collapses to rank-1, `reshape(vararg dims)` is the general
 * element-count-preserving relayout (row-major; data is shared semantics —
 * we copy for host-value simplicity). Result shape types erase to [Shape]
 * for the same reason as the axis reductions: the result dims
 * depend on runtime arguments; inside `grad {}` the K2 plugin computes the
 * exact `DxirType` from the literal arguments instead.
 */
fun <S : Shape> DTensor<S, F64>.squeeze(axis: Int): DTensor<Shape, F64> {
    val a = if (axis < 0) axis + dims.size else axis
    require(a in dims.indices) { "squeeze: axis $axis out of range for rank ${dims.size}" }
    require(dims[a] == 1) { "squeeze: axis $axis has size ${dims[a]} (must be 1)" }
    val out = IntArray(dims.size - 1)
    var k = 0
    for (i in dims.indices) if (i != a) out[k++] = dims[i]
    return DTensor(HostF64Storage(hostF64().copyOf()), out, F64)
}

fun <S : Shape> DTensor<S, F64>.unsqueeze(axis: Int): DTensor<Shape, F64> {
    val outRank = dims.size + 1
    val a = if (axis < 0) axis + outRank else axis
    require(a in 0 until outRank) { "unsqueeze: axis $axis out of range for result rank $outRank" }
    return DTensor(HostF64Storage(hostF64().copyOf()), unsqueezeAxes(this, intArrayOf(a)), F64)
}

fun <S : Shape> DTensor<S, F64>.flatten(): DTensor<Shape, F64> {
    val v = hostF64()
    return DTensor(HostF64Storage(v.copyOf()), intArrayOf(v.size), F64)
}

fun <S : Shape> DTensor<S, F64>.reshape(vararg newDims: Int): DTensor<Shape, F64> {
    val v = hostF64()
    var n = 1
    for (d in newDims) {
        require(d > 0) { "reshape: dims must be positive, got ${newDims.toList()}" }
        n *= d
    }
    require(n == v.size) {
        "reshape: element count mismatch — ${dims.toList()} (${v.size}) vs ${newDims.toList()} ($n)"
    }
    return DTensor(HostF64Storage(v.copyOf()), newDims.copyOf(), F64)
}

/**
 * General permutation transpose (DiffKT `transpose(axes)`).
 * The no-arg rank-2 [transpose] above keeps its precise `Rank2<C, R>`
 * shape typing; this vararg form handles any rank 1..3 with an erased
 * result type. Row-major stride walk mirroring the dxir interpreter's
 * TRANSPOSE arm.
 */
fun <S : Shape> DTensor<S, F64>.transpose(vararg perm: Int): DTensor<Shape, F64> {
    val r = dims.size
    require(perm.size == r) { "transpose: perm ${perm.toList()} must have length $r" }
    val norm = IntArray(r) { i ->
        val p = if (perm[i] < 0) perm[i] + r else perm[i]
        require(p in 0 until r) { "transpose: axis ${perm[i]} out of range for rank $r" }
        p
    }
    require(norm.toSet().size == r) { "transpose: ${perm.toList()} is not a permutation" }
    val v = hostF64()
    val outDims = IntArray(r) { dims[norm[it]] }
    val inStrides = IntArray(r)
    var st = 1
    for (i in r - 1 downTo 0) { inStrides[i] = st; st *= dims[i] }
    val outStrides = IntArray(r)
    st = 1
    for (i in r - 1 downTo 0) { outStrides[i] = st; st *= outDims[i] }
    val out = DoubleArray(v.size)
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
    return DTensor(HostF64Storage(out), outDims, F64)
}

/**
 * Axis flip (DiffKT `flip`): reverse the element order
 * along each listed axis, all other axes untouched. Shape-preserving, so the
 * receiver's precise shape type survives. Negative axes count from the end;
 * axes must be distinct. Row-major stride walk mirroring the dxir
 * interpreter's REVERSE arm: the output element at multi-index (i_0, …,
 * i_{N-1}) reads the input at `j_k = dims[k] − 1 − i_k` on flipped axes.
 * Its `grad {}` adjoint is the same flip of the upstream (REVERSE is
 * self-adjoint), so no runtime-extent template is involved anywhere.
 */
fun <S : Shape> DTensor<S, F64>.flip(vararg axes: Int): DTensor<S, F64> {
    val r = dims.size
    require(axes.isNotEmpty()) { "flip: needs at least one axis" }
    val norm = BooleanArray(r)
    for (ax in axes) {
        val a = if (ax < 0) ax + r else ax
        require(a in 0 until r) { "flip: axis $ax out of range for rank $r" }
        require(!norm[a]) { "flip: axes ${axes.toList()} must be distinct" }
        norm[a] = true
    }
    val v = hostF64()
    val strides = IntArray(r)
    var st = 1
    for (i in r - 1 downTo 0) { strides[i] = st; st *= dims[i] }
    val out = DoubleArray(v.size) { flat ->
        var rem = flat
        var src = 0
        for (k in 0 until r) {
            val coord = rem / strides[k]
            rem -= coord * strides[k]
            src += (if (norm[k]) dims[k] - 1 - coord else coord) * strides[k]
        }
        v[src]
    }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * Fixed-arity synthesis delegates for [flip] (the usual IrVararg
 * reason: synthesis cannot build an `IrVararg`, so each REVERSE node's literal
 * `dimensions` attr rides as positional Int constants, exactly as for
 * `transposePerm{N}`).
 */
fun <S : Shape> flipAxes1(x: DTensor<*, F64>, a0: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).flip(a0) as DTensor<S, F64>
}

fun <S : Shape> flipAxes2(x: DTensor<*, F64>, a0: Int, a1: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).flip(a0, a1) as DTensor<S, F64>
}

fun <S : Shape> flipAxes3(x: DTensor<*, F64>, a0: Int, a1: Int, a2: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).flip(a0, a1, a2) as DTensor<S, F64>
}

/**
 * Rank-increasing broadcast (DiffKT `broadcastTo`/`expand`). NumPy right-alignment: the
 * receiver's axes map to the TRAILING axes of [newDims]; the new leading axes are replicated.
 * In-place size-1 stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`) is supported too: an operand axis of
 * extent 1 replicates across its (equal-position) target extent (full
 * `stablehlo.broadcast_in_dim` semantics, matching the dxir interpreter's BROADCAST arm). A
 * non-1 operand axis must match its target exactly.
 */
fun <S : Shape> DTensor<S, F64>.broadcastTo(vararg newDims: Int): DTensor<Shape, F64> {
    val r = dims.size
    val outRank = newDims.size
    require(outRank >= r) { "broadcastTo: target rank $outRank < operand rank $r (only new leading axes)" }
    for (d in newDims) require(d > 0) { "broadcastTo: dims must be positive, got ${newDims.toList()}" }
    val offset = outRank - r
    for (j in 0 until r) require(dims[j] == newDims[offset + j] || dims[j] == 1) {
        "broadcastTo: operand dim $j = ${dims[j]} must match target ${newDims[offset + j]} or be a size-1 stretch"
    }
    val v = hostF64()
    // General broadcast_in_dim eval: operand axis j maps to output axis
    // (offset + j); a size-1 operand axis contributes index 0 (replicates).
    val inStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= dims[i] } }
    val outStrides = IntArray(outRank)
    run { var s = 1; for (i in outRank - 1 downTo 0) { outStrides[i] = s; s *= newDims[i] } }
    var outSize = 1
    for (d in newDims) outSize *= d
    val out = DoubleArray(outSize) { flat ->
        var rem = flat
        var src = 0
        for (k in 0 until outRank) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            val j = k - offset
            if (j >= 0 && dims[j] != 1) src += coord * inStrides[j]
        }
        v[src]
    }
    return DTensor(HostF64Storage(out), newDims.copyOf(), F64)
}

/**
 * Single-axis slice (DiffKT `slice`): take elements
 * `[start, end)` along [axis], all other axes full. `start`/`end`/`axis` are
 * compile-time literals; the result shape equals the receiver's with `axis`'s
 * extent replaced by `end − start`. This is the differentiable `slice`: its
 * `grad {}` adjoint is [padToLike] (the upstream zero-padded back into the
 * sliced window, reading the receiver's extent at runtime). Unit stride only.
 */
fun <S : Shape> DTensor<S, F64>.slice(start: Int, end: Int, axis: Int): DTensor<Shape, F64> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "slice: axis $axis out of range for rank $r" }
    require(start in 0..end && end <= dims[ax]) {
        "slice: [$start, $end) out of range for axis $ax extent ${dims[ax]}"
    }
    val outDims = IntArray(r) { if (it == ax) end - start else dims[it] }
    val inStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= dims[i] } }
    val outStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { outStrides[i] = s; s *= outDims[i] } }
    var outSize = 1
    for (d in outDims) outSize *= d
    val v = hostF64()
    val out = DoubleArray(outSize) { flat ->
        var rem = flat
        var src = 0
        for (k in 0 until r) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            src += (coord + if (k == ax) start else 0) * inStrides[k]
        }
        v[src]
    }
    return DTensor(HostF64Storage(out), outDims, F64)
}

/**
 * `view(range, axis)` (DiffKT indexing sugar):
 * the contiguous-range view of one axis, all other axes full. Pure sugar over
 * [slice] — `view(a..b, axis) == slice(a, b + 1, axis)` — with DiffKT's
 * inclusive-range spelling. The axis survives (rank is preserved); the
 * single-index overload below drops it. Differentiable in `grad {}` through
 * the same SLICE lowering and PAD_TO adjoint as `slice`.
 */
fun <S : Shape> DTensor<S, F64>.view(range: IntRange, axis: Int): DTensor<Shape, F64> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "view: axis $axis out of range for rank $r" }
    require(!range.isEmpty()) { "view: empty range $range" }
    return slice(range.first, range.last + 1, ax)
}

/**
 * `view(index, axis)`: pick one index along [axis] and DROP the
 * axis (DiffKT's indexing view — the result has rank `r − 1`). Sugar over
 * [slice] + [squeeze]. Its `grad {}` adjoint routes the upstream into the
 * indexed window and zeros elsewhere (the slice adjoint after the unit-axis
 * reshape restores the receiver's rank).
 */
fun <S : Shape> DTensor<S, F64>.view(index: Int, axis: Int): DTensor<Shape, F64> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "view: axis $axis out of range for rank $r" }
    require(index in 0 until dims[ax]) {
        "view: index $index out of range for axis $ax extent ${dims[ax]}"
    }
    return slice(index, index + 1, ax).squeeze(ax)
}

/**
 * `withChange(range, axis, replacement)`: the FUNCTIONAL update
 * (DiffKT `withChange`) — a copy of the receiver with the contiguous window
 * `[range.first, range.last]` along [axis] replaced by [replacement] (same
 * rank; the window's shape). The receiver is untouched. Inside `grad {}` the
 * K2 plugin lowers this as `x + PAD_TO(replacement − slice(x), template = x)`
 * — every piece an existing fully-ruled op, so the adjoint routes the
 * upstream's window to `replacement` and zeros that window in `d_x` with no
 * new IR: the PAD_TO ⇄ SLICE_AT pair does the bookkeeping.
 */
fun <S : Shape> DTensor<S, F64>.withChange(
    range: IntRange,
    axis: Int,
    replacement: DTensor<*, F64>,
): DTensor<Shape, F64> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "withChange: axis $axis out of range for rank $r" }
    require(!range.isEmpty()) { "withChange: empty range $range" }
    val start = range.first
    val end = range.last + 1
    require(start >= 0 && end <= dims[ax]) {
        "withChange: window [$start, $end) out of range for axis $ax extent ${dims[ax]}"
    }
    val rd = replacement.dims
    require(rd.size == r) { "withChange: replacement rank ${rd.size} != receiver rank $r" }
    for (i in 0 until r) {
        val want = if (i == ax) end - start else dims[i]
        require(rd[i] == want) {
            "withChange: replacement shape ${rd.toList()} != window shape at axis $i (want $want)"
        }
    }
    val out = hostF64().copyOf()
    val rv = replacement.hostF64()
    var run = 1
    for (i in ax + 1 until r) run *= dims[i]
    var outer = 1
    for (i in 0 until ax) outer *= dims[i]
    var src = 0
    for (o in 0 until outer) {
        for (j in start until end) {
            val dst = (o * dims[ax] + j) * run
            rv.copyInto(out, dst, src, src + run)
            src += run
        }
    }
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/**
 * `withChange(index, axis, replacement)`: replace the single
 * `view(index, axis)` slice — [replacement] has rank `r − 1` (the view's
 * shape). Sugar over [unsqueeze] + the range overload above.
 */
fun <S : Shape> DTensor<S, F64>.withChange(
    index: Int,
    axis: Int,
    replacement: DTensor<*, F64>,
): DTensor<Shape, F64> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "withChange: axis $axis out of range for rank $r" }
    require(index in 0 until dims[ax]) {
        "withChange: index $index out of range for axis $ax extent ${dims[ax]}"
    }
    require(replacement.dims.size == r - 1) {
        "withChange: replacement rank ${replacement.dims.size} != ${r - 1} (the view(index) shape)"
    }
    return withChange(index..index, ax, replacement.unsqueeze(ax))
}

/**
 * `meld(tensors…)` (DiffKT `meld`): flatten every operand
 * row-major and concatenate the flats into one rank-1 tensor of the total
 * element count. The inverse of [split]. Inside `grad {}` this is pure sugar
 * — RESHAPE-to-rank-1 per operand + the binary-CONCAT fold — so each
 * operand's gradient is its window of the upstream reshaped back to its own
 * shape (SLICE_LIKE + the reshape adjoint, both existing rules).
 */
fun meld(vararg tensors: DTensor<*, F64>): DTensor<Shape, F64> {
    require(tensors.isNotEmpty()) { "meld: needs at least 1 tensor" }
    var n = 0
    val flats = tensors.map { it.hostF64() }
    for (v in flats) n += v.size
    val out = DoubleArray(n)
    var k = 0
    for (v in flats) {
        v.copyInto(out, k)
        k += v.size
    }
    return DTensor(HostF64Storage(out), intArrayOf(n), F64)
}

/**
 * `split(shapes)` (DiffKT `split`): cut the receiver's
 * row-major data into consecutive tensors of the given shapes, which must
 * consume every element exactly. The inverse of [meld]. HOST-LEVEL ONLY:
 * a `List<DTensor>`-valued expression has no value model in the `grad {}`
 * lambda lowering, so the differentiable spelling stays `view`/`slice`
 * per piece.
 */
fun DTensor<*, F64>.split(shapes: List<IntArray>): List<DTensor<Shape, F64>> {
    require(shapes.isNotEmpty()) { "split: needs at least 1 shape" }
    val v = hostF64()
    var total = 0
    val sizes = shapes.map { s ->
        var n = 1
        for (d in s) {
            require(d > 0) { "split: dims must be positive, got ${s.toList()}" }
            n *= d
        }
        total += n
        n
    }
    require(total == v.size) {
        "split: shapes consume $total elements, tensor has ${v.size}"
    }
    var k = 0
    return shapes.mapIndexed { i, s ->
        val piece = v.copyOfRange(k, k + sizes[i])
        k += sizes[i]
        DTensor(HostF64Storage(piece), s.copyOf(), F64)
    }
}

/**
 * Fixed-arity synthesis delegates (the usual IrVararg reason).
 * `squeezeAxes{N}` drops size-1 axes at result-computed positions (the
 * adjoint of an unsqueeze); `reshapeToRank{N}` relayouts to explicit dims —
 * the synthesis feeds each dim either as a baked const (concrete dxir dim)
 * or a `param.dims[i]` runtime read (sentinel dim); `transposePerm{N}`
 * carries the permutation attr's compile-time constants.
 */
fun <S : Shape> squeezeAxes1(x: DTensor<*, F64>, a0: Int): DTensor<S, F64> {
    require(x.dims[a0] == 1) { "squeezeAxes1: axis $a0 has size ${x.dims[a0]}" }
    val out = IntArray(x.dims.size - 1)
    var k = 0
    for (i in x.dims.indices) if (i != a0) out[k++] = x.dims[i]
    return DTensor(HostF64Storage(x.hostF64().copyOf()), out, F64)
}

fun <S : Shape> squeezeAxes2(x: DTensor<*, F64>, a0: Int, a1: Int): DTensor<S, F64> {
    require(a0 < a1) { "squeezeAxes2: axes must be ascending" }
    require(x.dims[a0] == 1 && x.dims[a1] == 1) { "squeezeAxes2: axes must have size 1" }
    val out = IntArray(x.dims.size - 2)
    var k = 0
    for (i in x.dims.indices) if (i != a0 && i != a1) out[k++] = x.dims[i]
    return DTensor(HostF64Storage(x.hostF64().copyOf()), out, F64)
}

/**
 * The adjoint of [unsqueezeAxes3]: `[1,C,1,1] → [C]`, which is what the
 * gradient of a per-channel parameter reshape needs.
 */
fun <S : Shape> squeezeAxes3(x: DTensor<*, F64>, a0: Int, a1: Int, a2: Int): DTensor<S, F64> {
    require(a0 < a1 && a1 < a2) { "squeezeAxes3: axes must be ascending" }
    require(x.dims[a0] == 1 && x.dims[a1] == 1 && x.dims[a2] == 1) {
        "squeezeAxes3: axes must have size 1"
    }
    val out = IntArray(x.dims.size - 3)
    var k = 0
    for (i in x.dims.indices) if (i != a0 && i != a1 && i != a2) out[k++] = x.dims[i]
    return DTensor(HostF64Storage(x.hostF64().copyOf()), out, F64)
}

private fun reshapeTo(x: DTensor<*, F64>, target: IntArray): DoubleArray {
    val v = x.hostF64()
    var n = 1
    for (d in target) n *= d
    require(n == v.size) {
        "reshapeToRank: element count mismatch ${x.dims.toList()} vs ${target.toList()}"
    }
    return v.copyOf()
}

fun <S : Shape> reshapeToRank1(x: DTensor<*, F64>, d0: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(reshapeTo(x, intArrayOf(d0))), intArrayOf(d0), F64)

fun <S : Shape> reshapeToRank2(x: DTensor<*, F64>, d0: Int, d1: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(reshapeTo(x, intArrayOf(d0, d1))), intArrayOf(d0, d1), F64)

fun <S : Shape> reshapeToRank3(x: DTensor<*, F64>, d0: Int, d1: Int, d2: Int): DTensor<S, F64> =
    DTensor(HostF64Storage(reshapeTo(x, intArrayOf(d0, d1, d2))), intArrayOf(d0, d1, d2), F64)

/** Rank-4 relayout, for the NCHW surfaces (conv/pool/batchNorm). */
fun <S : Shape> reshapeToRank4(
    x: DTensor<*, F64>, d0: Int, d1: Int, d2: Int, d3: Int,
): DTensor<S, F64> =
    DTensor(
        HostF64Storage(reshapeTo(x, intArrayOf(d0, d1, d2, d3))),
        intArrayOf(d0, d1, d2, d3),
        F64,
    )

fun <S : Shape> transposePerm2(x: DTensor<*, F64>, p0: Int, p1: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).transpose(p0, p1) as DTensor<S, F64>
}

fun <S : Shape> transposePerm3(x: DTensor<*, F64>, p0: Int, p1: Int, p2: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).transpose(p0, p1, p2) as DTensor<S, F64>
}

/**
 * The rank-4 permutation transpose, for the batch↔feature swap
 * `[0,1,2,3] → [1,0,2,3]` that [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule]'s
 * `dW = conv(Xᵀ, dYᵀ)` trick emits on both sides. The vararg [transpose] above
 * already walks any rank; this delegate exists for the usual IrVararg reason and
 * to keep the permutation's compile-time constants positional.
 */
fun <S : Shape> transposePerm4(x: DTensor<*, F64>, p0: Int, p1: Int, p2: Int, p3: Int): DTensor<S, F64> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F64>).transpose(p0, p1, p2, p3) as DTensor<S, F64>
}

/**
 * Rank-2 transpose. Used by the K2 plugin's synthesis-side lowering
 * of [OpKind.TRANSPOSE] emitted by [io.tlaloc.ir.passes.VjpRegistry.MatmulRule].
 * The signature flips R and C in the shape type so the result is correctly
 * typed for downstream matmul chains.
 */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F64>.transpose(): DTensor<Rank2<C, R>, F64> {
    require(rank == 2) { "transpose requires rank-2 tensor" }
    val rows = dims[0]
    val cols = dims[1]
    val a = hostF64()
    val out = DoubleArray(cols * rows)
    for (i in 0 until rows) {
        for (j in 0 until cols) {
            out[j * rows + i] = a[i * cols + j]
        }
    }
    return DTensor(HostF64Storage(out), intArrayOf(cols, rows), F64)
}

infix fun <R : ShapeAtom, K : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, K>, F64>.matmul(
    other: DTensor<Rank2<K, C>, F64>,
): DTensor<Rank2<R, C>, F64> {
    require(rank == 2 && other.rank == 2) { "matmul requires rank-2 tensors" }
    val m = dims[0]
    val k = dims[1]
    val kb = other.dims[0]
    val n = other.dims[1]
    require(k == kb) { "matmul inner dim mismatch: ${dims.toList()} x ${other.dims.toList()}" }

    val a = hostF64()
    val b = other.hostF64()
    val out = DoubleArray(m * n)
    for (i in 0 until m) {
        for (p in 0 until k) {
            val aip = a[i * k + p]
            if (aip == 0.0) continue
            val rowOff = i * n
            val bOff = p * n
            for (j in 0 until n) {
                out[rowOff + j] += aip * b[bOff + j]
            }
        }
    }
    return DTensor(HostF64Storage(out), intArrayOf(m, n), F64)
}
