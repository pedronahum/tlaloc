package io.tlaloc.core.ops

import io.tlaloc.core.DScalar
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.RandomKey
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.Sym
import io.tlaloc.core.digamma
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import io.tlaloc.core.lgamma
import io.tlaloc.core.normalFloats
import io.tlaloc.core.polygamma
import io.tlaloc.core.trigamma
import io.tlaloc.core.uniformFloats
import kotlin.math.pow
import kotlin.math.sqrt

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
operator fun <S : Shape> DTensor<S, F32>.plus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseBroadcast<S>(this, other) { x, y -> x + y }

operator fun <S : Shape> DTensor<S, F32>.minus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseBroadcast<S>(this, other) { x, y -> x - y }

operator fun <S : Shape> DTensor<S, F32>.times(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseBroadcast<S>(this, other) { x, y -> x * y }

operator fun <S : Shape> DTensor<S, F32>.div(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseBroadcast<S>(this, other) { x, y -> x / y }

/**
 * §0.4.206 — Scalar-multiply on a DTensor: `tensor * scalar` returns a fresh
 * `DTensor<S, F32>` with the same shape and each element multiplied by [scalar].
 * Required for vanilla gradient-descent updates (`W = W - lr * dW`) used by
 * CartPole's outer training loop. Avoids the `broadcastLike(scalar, W) * dW`
 * roundabout that would otherwise be needed for `lr * dW`.
 *
 * Phase A5 (DiffKT parity) completes the set: DiffKT mixes scalars into every
 * binary op (`timesScalar` + `broadcast(S1, S2)`), so `a + 1.0f`, `a / 2.0f`
 * and the scalar-on-the-left spellings `3.0f - a` / `2.0f * a` are all
 * writable. The scalar-on-left forms matter for the non-commutative ops —
 * `3.0f - a` is not `a - 3.0f` — and the K2 plugin preserves source operand
 * order when it splats them (see `FirLambdaToDxirLowering`'s mixed-rank arm).
 */
private fun <S : Shape> elementwiseScalar(
    a: DTensor<S, F32>,
    b: Float,
    f: (Float, Float) -> Float,
): DTensor<S, F32> {
    val av = a.hostF32()
    val out = FloatArray(av.size)
    for (i in av.indices) out[i] = f(av[i], b)
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
}

operator fun <S : Shape> DTensor<S, F32>.plus(scalar: Float): DTensor<S, F32> =
    elementwiseScalar(this, scalar) { x, y -> x + y }

operator fun <S : Shape> DTensor<S, F32>.minus(scalar: Float): DTensor<S, F32> =
    elementwiseScalar(this, scalar) { x, y -> x - y }

operator fun <S : Shape> DTensor<S, F32>.times(scalar: Float): DTensor<S, F32> =
    elementwiseScalar(this, scalar) { x, y -> x * y }

operator fun <S : Shape> DTensor<S, F32>.div(scalar: Float): DTensor<S, F32> =
    elementwiseScalar(this, scalar) { x, y -> x / y }

operator fun <S : Shape> Float.plus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseScalar(other, this) { x, y -> y + x }

operator fun <S : Shape> Float.minus(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseScalar(other, this) { x, y -> y - x }

operator fun <S : Shape> Float.times(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseScalar(other, this) { x, y -> y * x }

operator fun <S : Shape> Float.div(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseScalar(other, this) { x, y -> y / x }

/**
 * §0.4.397 — Phase A5c-3(iv): `DScalar × DTensor` mixing, DiffKT's
 * `timesScalar`. DiffKT's `Operations` interface carries exactly ONE
 * scalar-mixing primitive — `timesScalar(left: DScalar, right: DTensor)` —
 * surfaced as `DScalar * DTensor` and `DTensor * DScalar`; the other binaries
 * mix through `Float` (which Tlaloc ships since Phase A5a above). These two
 * overloads close that parity point at host level: the scalar side unwraps to
 * its Float value (F32 host storage — DoubleScalar narrows, matching
 * `DScalar.toFloat()`), and the walk is the same [elementwiseScalar]. Inside
 * `grad {}` the K2 plugin's mixed-rank arm splats the rank-0 DScalar operand
 * through the templated BROADCAST (the A5a computed-scalar path), so a
 * DIFFERENTIABLE scalar factor gets BroadcastRule's full-reduce adjoint for
 * free — `d s = Σ (∂loss/∂prod ⊙ a)`.
 */
operator fun <S : Shape> DTensor<S, F32>.times(scalar: DScalar): DTensor<S, F32> =
    elementwiseScalar(this, scalar.toFloat()) { x, y -> x * y }

operator fun <S : Shape> DScalar.times(other: DTensor<S, F32>): DTensor<S, F32> =
    elementwiseScalar(other, this.toFloat()) { x, y -> y * x }

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
 * §0.4.397 — Phase A5c-3(iv): comparisons against a Float scalar (`a gt 1.0f`),
 * the last everyday DiffKT comparison spelling Tlaloc rejected. Same 0/1 F32
 * mask contract as the tensor⊙tensor forms above; the scalar side is compared
 * against every element. Inside `grad {}` the K2 plugin splats the scalar side
 * over the tensor operand's shape (the Phase A5a literal-splat pattern), so the
 * IR sees the uniform two-tensor COMPARE the §0.4.364 arm already lowers.
 */
infix fun <S : Shape> DTensor<S, F32>.gt(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x > y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.ge(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x >= y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.lt(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x < y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.le(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x <= y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.eq(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x == y) 1f else 0f }

infix fun <S : Shape> DTensor<S, F32>.ne(other: Float): DTensor<S, F32> =
    elementwiseScalar(this, other) { x, y -> if (x != y) 1f else 0f }

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

/**
 * §0.4.369 — Phase A4 (DiffKT parity): elementwise `maximum(a, b)` /
 * `minimum(a, b)` — first-class in DiffKT, sugar in Tlaloc. Inside `grad {}`
 * the K2 plugin lowers `maximum` to `WHERE(COMPARE(a, b, GE), a, b)` and
 * `minimum` to `WHERE(COMPARE(a, b, LE), a, b)` — the §0.4.364 where/compare
 * surface — so the gradient flows through WhereRule with no new AD math:
 * full upstream to the larger (resp. smaller) operand, ties to the first
 * (the `>=` / `<=` mask keeps `a`). This host body is the runtime twin. The
 * shared phantom shape [S] requires same-shape operands (elementwise).
 */
fun <S : Shape> maximum(a: DTensor<S, F32>, b: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(a, b) { x, y -> if (x >= y) x else y }

fun <S : Shape> minimum(a: DTensor<S, F32>, b: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(a, b) { x, y -> if (x <= y) x else y }

/**
 * §0.4.369 — `clip(x, lo, hi)` with compile-time Float scalar bounds =
 * `minimum(maximum(x, lo), hi)`. Inside `grad {}` this composes as two
 * COMPARE+WHERE pairs against `lo`/`hi` splat consts of `x`'s shape, so the
 * gradient is exactly 1 where `lo ≤ x ≤ hi` and 0 outside (WhereRule routes
 * upstream to `x` on the in-bounds mask and to the const bound — zero
 * gradient — outside). Requires `lo ≤ hi`.
 */
fun <S : Shape> clip(x: DTensor<S, F32>, lo: Float, hi: Float): DTensor<S, F32> {
    require(lo <= hi) { "clip: lo ($lo) must be ≤ hi ($hi)" }
    return x.unary { v -> if (v < lo) lo else if (v > hi) hi else v }
}

/**
 * §0.4.369 — `outerProduct(a, b)` for rank-1 operands: `out[i, j] = a[i]·b[j]`,
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
    a: DTensor<SA, F32>,
    b: DTensor<SB, F32>,
): DTensor<Shape, F32> {
    require(a.dims.size == 1 && b.dims.size == 1) {
        "outerProduct v1 requires rank-1 operands, got ${a.dims.toList()} ⊗ ${b.dims.toList()}"
    }
    val av = a.hostF32()
    val bv = b.hostF32()
    val n = av.size
    val m = bv.size
    val out = FloatArray(n * m)
    for (i in 0 until n) {
        val ai = av[i]
        val rowOff = i * m
        for (j in 0 until m) out[rowOff + j] = ai * bv[j]
    }
    return DTensor(HostF32Storage(out), intArrayOf(n, m), F32)
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

// §0.4.395 — Phase C2 trig tails (DiffKT parity: `tan`/`atan` are the only trig
// ops DiffKT has that Tlaloc lacked). Double-precision evaluation before the F32
// narrow, bit-for-bit the interpreter's TAN/ATAN arm convention.
fun <S : Shape> DTensor<S, F32>.tan(): DTensor<S, F32> =
    unary { x -> kotlin.math.tan(x.toDouble()).toFloat() }

fun <S : Shape> DTensor<S, F32>.atan(): DTensor<S, F32> =
    unary { x -> kotlin.math.atan(x.toDouble()).toFloat() }

// §0.4.402 — Phase C1 special functions (DiffKT parity: its Dirichlet example
// depends on these). Evaluation routes through the shared Double kernels in
// `:core/SpecialFunctions.kt` (Lanczos g=7 lgamma; recurrence-to-asymptotic
// digamma/trigamma) — the same functions the interpreter's LGAMMA/DIGAMMA/
// TRIGAMMA arms call, so host and interpreter agree bit-for-bit. `trigamma`
// is gradient machinery (∇ digamma bodies call it), not user parity surface.
fun <S : Shape> DTensor<S, F32>.lgamma(): DTensor<S, F32> =
    unary { x -> x.toDouble().lgamma().toFloat() }

fun <S : Shape> DTensor<S, F32>.digamma(): DTensor<S, F32> =
    unary { x -> x.toDouble().digamma().toFloat() }

fun <S : Shape> DTensor<S, F32>.trigamma(): DTensor<S, F32> =
    unary { x -> x.toDouble().trigamma().toFloat() }

// §0.4.405 — polygamma(n), C1's recorded deferral. The order is a value
// parameter here but a compile-time Int literal inside `grad {}` (the FIR
// folds it: 0 → DIGAMMA, 1 → TRIGAMMA, n ≥ 2 → POLYGAMMA + `order` attr).
// One positional parameter, no defaults — the K2 named-arg landmine.
fun <S : Shape> DTensor<S, F32>.polygamma(n: Int): DTensor<S, F32> =
    unary { x -> x.toDouble().polygamma(n).toFloat() }

/**
 * Phase A5b (DiffKT parity) — elementwise power. POW has been fully ruled below
 * the surface since Stage B.3 (PowRule, the interpreter arm, `stablehlo.power`
 * emission, the forward-mode tangent, and synthesis's scalar `kotlin.math.pow`
 * arm) but had no host op and no FIR entry, so user code could never reach it.
 * Three spellings, matching DiffKT's `pow(Float/Int/tensor-exponent)`: a tensor
 * exponent (elementwise, same shape) and Float / Int exponents — the K2 plugin
 * splats a literal exponent to the operand's shape, so the IR always sees the
 * uniform two-operand POW that PowRule expects.
 *
 * Arithmetic goes through Double and back to F32, bit-for-bit the convention the
 * dxir interpreter's POW arm uses, so the host and IR paths agree.
 */
fun <S : Shape> DTensor<S, F32>.pow(exp: DTensor<S, F32>): DTensor<S, F32> =
    elementwise(this, exp) { a, b -> a.toDouble().pow(b.toDouble()).toFloat() }

fun <S : Shape> DTensor<S, F32>.pow(exp: Float): DTensor<S, F32> =
    unary { x -> x.toDouble().pow(exp.toDouble()).toFloat() }

fun <S : Shape> DTensor<S, F32>.pow(exp: Int): DTensor<S, F32> = pow(exp.toFloat())

/**
 * §0.4.368 — Phase A3 (DiffKT parity): `softmax(axis)` over a single axis
 * (default last; negative axes count from the back). Numerically stable
 * (subtract the per-slice max before exponentiating), row-major stride walk
 * matching the dxir interpreter's SOFTMAX arm bit-for-bit so the host path
 * and the IR path agree. Shape-preserving — the phantom shape witness [S]
 * survives (unlike the axis reductions, which erase to [Shape]).
 */
fun <S : Shape> DTensor<S, F32>.softmax(axis: Int = -1): DTensor<S, F32> {
    val r = dims.size
    val a = if (axis < 0) axis + r else axis
    require(a in 0 until r) { "softmax: axis $axis out of range for rank $r" }
    val v = hostF32()
    val axisLen = dims[a]
    var inner = 1
    for (k in a + 1 until r) inner *= dims[k]
    var outer = 1
    for (k in 0 until a) outer *= dims[k]
    val out = FloatArray(v.size)
    for (o in 0 until outer) {
        for (i in 0 until inner) {
            val base = o * axisLen * inner + i
            var mx = Float.NEGATIVE_INFINITY
            for (j in 0 until axisLen) mx = maxOf(mx, v[base + j * inner])
            var sum = 0f
            for (j in 0 until axisLen) {
                val e = kotlin.math.exp(v[base + j * inner] - mx)
                out[base + j * inner] = e
                sum += e
            }
            for (j in 0 until axisLen) out[base + j * inner] /= sum
        }
    }
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.368 — `logSoftmax(axis)` = log(softmax(x, axis)), computed in the
 * stable `x - max - log(Σ exp(x - max))` form (never materialises the
 * softmax then logs it, which would lose precision in the tail). Inside
 * `grad {}` the K2 plugin lowers this to `LOG(SOFTMAX(x, axis))` — both
 * ops carry full VJP/JVP rules — so the gradient flows through the existing
 * LogRule ∘ SoftmaxRule chain; this host body is the runtime twin.
 */
fun <S : Shape> DTensor<S, F32>.logSoftmax(axis: Int = -1): DTensor<S, F32> {
    val r = dims.size
    val a = if (axis < 0) axis + r else axis
    require(a in 0 until r) { "logSoftmax: axis $axis out of range for rank $r" }
    val v = hostF32()
    val axisLen = dims[a]
    var inner = 1
    for (k in a + 1 until r) inner *= dims[k]
    var outer = 1
    for (k in 0 until a) outer *= dims[k]
    val out = FloatArray(v.size)
    for (o in 0 until outer) {
        for (i in 0 until inner) {
            val base = o * axisLen * inner + i
            var mx = Float.NEGATIVE_INFINITY
            for (j in 0 until axisLen) mx = maxOf(mx, v[base + j * inner])
            var sum = 0f
            for (j in 0 until axisLen) sum += kotlin.math.exp(v[base + j * inner] - mx)
            val logSum = kotlin.math.ln(sum)
            for (j in 0 until axisLen) out[base + j * inner] = v[base + j * inner] - mx - logSum
        }
    }
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.370 — Phase A3b (DiffKT parity): `crossEntropyLoss(logits, oneHot)` =
 * the sum-reduced softmax cross entropy. Equal to `-Σ oneHot ⊙ logSoftmax(logits)`
 * over the last (class) axis, then summed over every position (the sum-reduction
 * convention — the total of the per-sample cross-entropies). `oneHot` is a float
 * one-hot (or soft) label tensor the same shape as `logits`. Inside `grad {}` the
 * K2 plugin lowers this to `NEG(SUM(MUL(oneHot, LOG(SOFTMAX(logits, -1)))))` —
 * every op fully-ruled — so the gradient flows with no new AD math; this host body
 * is the runtime twin. Returns a scalar so it composes with `.toFloat()`.
 */
fun <S : Shape> crossEntropyLoss(
    logits: DTensor<S, F32>,
    oneHot: DTensor<S, F32>,
): DTensor<ScalarShape, F32> {
    val ls = logits.logSoftmax(-1).hostF32()
    val oh = oneHot.hostF32()
    require(ls.size == oh.size) {
        "crossEntropyLoss: logits and oneHot must have the same element count (${ls.size} vs ${oh.size})"
    }
    var acc = 0f
    for (i in ls.indices) acc -= oh[i] * ls[i]
    return DTensor(HostF32Storage(floatArrayOf(acc)), intArrayOf(), F32)
}

/**
 * §0.4.370 — `nllLoss(logProbs, oneHot)` = the negative-log-likelihood loss on
 * already-log-normalised probabilities: `-Σ oneHot ⊙ logProbs`, summed over every
 * position. The companion to [crossEntropyLoss] for when the caller has already
 * applied `logSoftmax`. Inside `grad {}` it lowers to `NEG(SUM(MUL(oneHot, logProbs)))`.
 */
fun <S : Shape> nllLoss(
    logProbs: DTensor<S, F32>,
    oneHot: DTensor<S, F32>,
): DTensor<ScalarShape, F32> {
    val lp = logProbs.hostF32()
    val oh = oneHot.hostF32()
    require(lp.size == oh.size) {
        "nllLoss: logProbs and oneHot must have the same element count (${lp.size} vs ${oh.size})"
    }
    var acc = 0f
    for (i in lp.indices) acc -= oh[i] * lp[i]
    return DTensor(HostF32Storage(floatArrayOf(acc)), intArrayOf(), F32)
}

/**
 * §0.4.400 — Phase A3b (DiffKT parity): `embedding(table, indices)` gathers
 * `table[indices[p], :]` for each index position — the rank-2 `[V, D]` table
 * against a rank-1 `[N]` I32 index vector, producing `[N, D]`. Row-major walk
 * matching the dxir interpreter's EMBEDDING arm bit-for-bit (same bounds check,
 * same gather order) so the host path and the IR path agree. The phantom result
 * shape is `Rank2<N, D>`: the position atom from the indices, the feature atom
 * from the table. Inside `grad {}` the K2 plugin lowers this to
 * [io.tlaloc.ir.OpKind.EMBEDDING]; the §0.4.370 EmbeddingRule provides the
 * reverse (a fused scatter-add), for which [embeddingGrad] is the runtime twin.
 */
fun <V : ShapeAtom, D : ShapeAtom, N : ShapeAtom> embedding(
    table: DTensor<Rank2<V, D>, F32>,
    indices: DTensor<Rank1<N>, I32>,
): DTensor<Rank2<N, D>, F32> {
    require(indices.rank == 1) { "embedding: indices must be rank-1; got ${indices.dims.toList()}" }
    val out = embeddingGather(table, indices, paddingIndex = -1)
    return DTensor(HostF32Storage(out), intArrayOf(indices.dims[0], table.dims[1]), F32)
}

/**
 * §0.4.409 — DiffKT's `embedding(table, indices, paddingIndex)`: positions whose
 * index equals [paddingIndex] produce EXACT-zero output rows (and, through the
 * padded `embeddingGrad` twin, contribute zero gradient to the table). A
 * negative [paddingIndex] means "none" — the -1 sentinel the IR's optional
 * `padding_index` attr uses. Padded positions skip the bounds check too, so a
 * paddingIndex outside the vocab is legal (it can never gather).
 *
 * Positional-arity disambiguation on purpose: NO default parameter value on
 * [paddingIndex] — K2 unwraps named args to bare literals without reordering,
 * so attr-bearing host ops must disambiguate by arity alone.
 */
fun <V : ShapeAtom, D : ShapeAtom, N : ShapeAtom> embedding(
    table: DTensor<Rank2<V, D>, F32>,
    indices: DTensor<Rank1<N>, I32>,
    paddingIndex: Int,
): DTensor<Rank2<N, D>, F32> {
    require(indices.rank == 1) { "embedding: indices must be rank-1; got ${indices.dims.toList()}" }
    val out = embeddingGather(table, indices, paddingIndex)
    return DTensor(HostF32Storage(out), intArrayOf(indices.dims[0], table.dims[1]), F32)
}

/**
 * §0.4.409 — the rank-2 index batch `[B, N]` → `[B, N, D]`. Same row-major
 * gather walk as the rank-1 spelling (the flat position order IS the batch
 * order), so the dxir interpreter's rank-agnostic EMBEDDING arm stays the
 * bit-exact oracle. `@JvmName` dodges the erasure clash with the rank-1
 * overload (both erase to `embedding(DTensor, DTensor)`); Kotlin-side the name
 * stays `embedding`, which is what the FIR arm and user code see.
 */
@JvmName("embeddingBatch")
fun <V : ShapeAtom, D : ShapeAtom, B : ShapeAtom, N : ShapeAtom> embedding(
    table: DTensor<Rank2<V, D>, F32>,
    indices: DTensor<Rank2<B, N>, I32>,
): DTensor<Rank3<B, N, D>, F32> {
    require(indices.rank == 2) { "embedding: indices must be rank-2 (B, N); got ${indices.dims.toList()}" }
    val out = embeddingGather(table, indices, paddingIndex = -1)
    return DTensor(HostF32Storage(out), intArrayOf(indices.dims[0], indices.dims[1], table.dims[1]), F32)
}

/** §0.4.409 — rank-2 index batch with [paddingIndex]; see the rank-1 padded overload. */
@JvmName("embeddingBatchPadded")
fun <V : ShapeAtom, D : ShapeAtom, B : ShapeAtom, N : ShapeAtom> embedding(
    table: DTensor<Rank2<V, D>, F32>,
    indices: DTensor<Rank2<B, N>, I32>,
    paddingIndex: Int,
): DTensor<Rank3<B, N, D>, F32> {
    require(indices.rank == 2) { "embedding: indices must be rank-2 (B, N); got ${indices.dims.toList()}" }
    val out = embeddingGather(table, indices, paddingIndex)
    return DTensor(HostF32Storage(out), intArrayOf(indices.dims[0], indices.dims[1], table.dims[1]), F32)
}

/** Shared flat gather walk — the dxir interpreter's EMBEDDING arm, bit-for-bit. */
private fun embeddingGather(
    table: DTensor<*, F32>,
    indices: DTensor<*, I32>,
    paddingIndex: Int,
): FloatArray {
    require(table.rank == 2) { "embedding: table must be rank-2 (V, D); got ${table.dims.toList()}" }
    val t = table.hostF32()
    val idx = indices.hostI32()
    val vocab = table.dims[0]
    val embedDim = table.dims[1]
    val positions = idx.size
    val out = FloatArray(positions * embedDim)
    for (p in 0 until positions) {
        val v = idx[p]
        if (paddingIndex >= 0 && v == paddingIndex) continue // exact-zero row
        require(v in 0 until vocab) {
            "embedding: index $v at position $p out of bounds for vocab $vocab"
        }
        for (d in 0 until embedDim) out[p * embedDim + d] = t[v * embedDim + d]
    }
    return out
}

/**
 * §0.4.400 — [embedding]'s reverse twin: scatter-ADD each upstream row back to
 * the vocab slot its index selected, `dTable[indices[p], :] += upstream[p, :]`,
 * collisions summing when the same vocab row was embedded at multiple
 * positions. The runtime twin of the dxir interpreter's EMBEDDING_GRAD arm,
 * bit-for-bit. [tableTemplate] contributes SHAPE ONLY — its values are never
 * read (the SUM_TO / conv2dDataAdjoint template convention): under `grad {}`'s
 * -1 sentinel dims the vocab extent is unknowable at compile time, so the
 * template's ACTUAL runtime dims size the result. Unselected vocab rows stay
 * exactly zero.
 */
fun <S : Shape> embeddingGrad(
    upstream: DTensor<*, F32>,
    indices: DTensor<*, I32>,
    tableTemplate: DTensor<S, F32>,
): DTensor<S, F32> = embeddingGrad(upstream, indices, tableTemplate, -1)

/**
 * §0.4.409 — the padded reverse twin: positions whose index equals
 * [paddingIndex] are SKIPPED by the scatter walk entirely, so the padded vocab
 * row's gradient stays exactly zero (and a paddingIndex outside the vocab is
 * legal — nothing is ever scattered there). Negative [paddingIndex] = none.
 * Arity-disambiguated from the 3-arg twin — no default parameter values.
 */
fun <S : Shape> embeddingGrad(
    upstream: DTensor<*, F32>,
    indices: DTensor<*, I32>,
    tableTemplate: DTensor<S, F32>,
    paddingIndex: Int,
): DTensor<S, F32> {
    require(tableTemplate.rank == 2) {
        "embeddingGrad: tableTemplate must be rank-2 (V, D); got ${tableTemplate.dims.toList()}"
    }
    val vocab = tableTemplate.dims[0]
    val embedDim = tableTemplate.dims[1]
    val idx = indices.hostI32()
    val up = upstream.hostF32()
    val positions = idx.size
    require(up.size == positions * embedDim) {
        "embeddingGrad: upstream size ${up.size} != positions $positions * embedDim $embedDim"
    }
    val out = FloatArray(vocab * embedDim)
    for (p in 0 until positions) {
        val v = idx[p]
        if (paddingIndex >= 0 && v == paddingIndex) continue // padded rows carry no gradient
        require(v in 0 until vocab) {
            "embeddingGrad: index $v at position $p out of bounds for vocab $vocab"
        }
        for (d in 0 until embedDim) out[v * embedDim + d] += up[p * embedDim + d]
    }
    return DTensor(HostF32Storage(out), tableTemplate.dims.copyOf(), F32)
}

/**
 * §0.4.400 — the zero "gradient" of an integer tensor param. `grad {}` on a
 * lambda with a non-differentiable I32 param (embedding indices) still returns
 * one gradient per param; the reverse transform types the integer slot as a
 * structural zero (§0.4.54), and the synthesis materialises it with this —
 * shaped like the param at RUNTIME, since its static dims are -1 sentinels.
 */
fun <S : Shape> intZerosLike(t: DTensor<S, I32>): DTensor<S, I32> =
    DTensor(HostI32Storage(IntArray(t.size)), t.dims.copyOf(), I32)

// ---------------------------------------------------------------------------
// §0.4.421 — Phase D2 tail: the synthesis twins of the zero-operand RNG ops
// (RNG_UNIFORM / RNG_NORMAL). The `grad {}` FIR front-end bakes the draw's
// literal key words and dims onto the op as attrs (the §0.4.408 design), and
// the synthesis replays them here as plain Int arguments — same
// `:core/Random.kt` kernels the host tensor surface and the interpreter call,
// so all three paths agree bit-for-bit by construction. The phantom shape S
// is caller-asserted from literal dims, exactly like `Tensors.f32Vector<A>`.
// ---------------------------------------------------------------------------

/** Uniform [0, 1) rank-1 draw of [n] floats from the literal key words. */
fun <S : Shape> rngUniformVector(key0: Int, key1: Int, n: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(uniformFloats(RandomKey(key0, key1), n)), intArrayOf(n), F32)

/** Uniform [0, 1) rank-2 draw over the flat index space. */
fun <S : Shape> rngUniformMatrix(key0: Int, key1: Int, rows: Int, cols: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(uniformFloats(RandomKey(key0, key1), rows * cols)), intArrayOf(rows, cols), F32)

/** Standard-normal rank-1 draw of [n] floats from the literal key words. */
fun <S : Shape> rngNormalVector(key0: Int, key1: Int, n: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(normalFloats(RandomKey(key0, key1), n)), intArrayOf(n), F32)

/** Standard-normal rank-2 draw over the flat index space. */
fun <S : Shape> rngNormalMatrix(key0: Int, key1: Int, rows: Int, cols: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(normalFloats(RandomKey(key0, key1), rows * cols)), intArrayOf(rows, cols), F32)

/**
 * §0.4.418 — Phase E1b: loud validation of a CSR component triple, the E1a
 * `SparseTensor` constructor's invariant minus the strictly-increasing-columns
 * check (the SpMM/SDDMM walks are order-independent in the math; canonical
 * inputs additionally get bit-for-bit parity with `SparseTensor.matmul`).
 * Shared by the three sparse host twins below — the dxir interpreter's
 * `validateCsrComponents` mirror, message-for-message.
 */
private fun validateCsrComponents(
    opName: String,
    nnz: Int,
    colIdx: IntArray,
    rowPtr: IntArray,
    colBound: Int,
) {
    require(rowPtr.isNotEmpty()) { "$opName: rowPtr must have N+1 ≥ 1 entries" }
    require(colIdx.size == nnz) {
        "$opName: values ($nnz) and colIdx (${colIdx.size}) must be parallel"
    }
    require(rowPtr[0] == 0) { "$opName: rowPtr[0] must be 0, got ${rowPtr[0]}" }
    require(rowPtr[rowPtr.size - 1] == nnz) {
        "$opName: rowPtr[${rowPtr.size - 1}] must equal nnz=$nnz, got ${rowPtr[rowPtr.size - 1]}"
    }
    for (i in 1 until rowPtr.size) {
        require(rowPtr[i] >= rowPtr[i - 1]) {
            "$opName: rowPtr must be monotone non-decreasing: " +
                "rowPtr[${i - 1}]=${rowPtr[i - 1]} > rowPtr[$i]=${rowPtr[i]}"
        }
    }
    for (k in 0 until nnz) {
        require(colIdx[k] in 0 until colBound) {
            "$opName: colIdx[$k]=${colIdx[k]} out of range [0, $colBound)"
        }
    }
}

/**
 * §0.4.418 — Phase E1b: the host twin of the dxir interpreter's SPARSE_MATMUL
 * arm — sparse `[N, C]` (as CSR components) × dense `[C, D]` → dense `[N, D]`.
 * `N` is read off [rowPtr]'s ACTUAL runtime extent (N+1 entries — the
 * runtime-extent house pattern, which is what makes the eventual `grad {}`
 * synthesis path sound under -1 sentinel dims), `C`/`D` off the dense
 * operand's runtime dims. The walk is `SparseTensor.matmul(dense)` — and the
 * interpreter arm — bit-for-bit: per output row a Double accumulator collects
 * `values[k] · dense[colIdx[k], :]` in increasing-k order, then narrows.
 */
fun sparseMatmul(
    values: DTensor<*, F32>,
    colIdx: DTensor<*, I32>,
    rowPtr: DTensor<*, I32>,
    dense: DTensor<*, F32>,
): DTensor<Rank2<Sym, Sym>, F32> {
    require(dense.rank == 2) {
        "sparseMatmul: dense operand must be rank-2 (C, D); got ${dense.dims.toList()}"
    }
    val v = values.hostF32()
    val ci = colIdx.hostI32()
    val rp = rowPtr.hostI32()
    val b = dense.hostF32()
    val c = dense.dims[0]
    val d = dense.dims[1]
    validateCsrComponents("sparseMatmul", v.size, ci, rp, c)
    val n = rp.size - 1
    val out = FloatArray(n * d)
    val acc = DoubleArray(d)
    for (i in 0 until n) {
        acc.fill(0.0)
        for (k in rp[i] until rp[i + 1]) {
            val vk = v[k].toDouble()
            val bOff = ci[k] * d
            for (j in 0 until d) acc[j] += vk * b[bOff + j]
        }
        val rowOff = i * d
        for (j in 0 until d) out[rowOff + j] = acc[j].toFloat()
    }
    return DTensor(HostF32Storage(out), intArrayOf(n, d), F32)
}

/**
 * §0.4.418 — the `transposed = true` form's host twin: `Aᵀ · dense` over the
 * SAME CSR components, `dense` here `[N, D]` (the upstream, in the adjoint
 * use). [denseTemplate] contributes SHAPE ONLY — its values are never read
 * (the SUM_TO / embeddingGrad template convention): its runtime leading dim
 * is the output row extent `C`, which no component tensor's shape carries
 * (rowPtr gives N, colIdx gives nnz) and which is a -1 sentinel under
 * `grad {}`. The scatter walk (source rows in order) reproduces
 * `SparseTensor.transpose().matmul(dense)` — and the interpreter's transposed
 * arm — bit-for-bit: per output element the Double-add sequence is identical,
 * because the canonical counting-sort transpose orders each output row's
 * entries by source row.
 */
fun <S : Shape> sparseMatmulTransposed(
    values: DTensor<*, F32>,
    colIdx: DTensor<*, I32>,
    rowPtr: DTensor<*, I32>,
    dense: DTensor<*, F32>,
    denseTemplate: DTensor<S, F32>,
): DTensor<Rank2<Sym, Sym>, F32> {
    require(dense.rank == 2) {
        "sparseMatmulTransposed: dense operand must be rank-2 (N, D); got ${dense.dims.toList()}"
    }
    require(denseTemplate.rank == 2) {
        "sparseMatmulTransposed: denseTemplate must be rank-2 (C, ·); got ${denseTemplate.dims.toList()}"
    }
    val v = values.hostF32()
    val ci = colIdx.hostI32()
    val rp = rowPtr.hostI32()
    val u = dense.hostF32()
    val d = dense.dims[1]
    val n = rp.size - 1
    val cOut = denseTemplate.dims[0]
    require(dense.dims[0] == n) {
        "sparseMatmulTransposed: dense operand has ${dense.dims[0]} rows but rowPtr implies N=$n"
    }
    validateCsrComponents("sparseMatmulTransposed", v.size, ci, rp, cOut)
    val acc = DoubleArray(cOut * d)
    for (i in 0 until n) {
        val dOff = i * d
        for (k in rp[i] until rp[i + 1]) {
            val vk = v[k].toDouble()
            val outOff = ci[k] * d
            for (j in 0 until d) acc[outOff + j] += vk * u[dOff + j]
        }
    }
    val out = FloatArray(cOut * d) { acc[it].toFloat() }
    return DTensor(HostF32Storage(out), intArrayOf(cOut, d), F32)
}

/**
 * §0.4.418 — the fused SDDMM values-adjoint's host twin, the dxir
 * interpreter's SPARSE_MATMUL_VALUES_ADJOINT arm bit-for-bit:
 * `d_values[k] = Σ_j upstream[row(k), j] · dense[colIdx[k], j]`, one
 * Double-accumulated length-D dot per STORED entry — the structural zeros
 * are not inputs, so no gradient exists for them and the result is exactly
 * `[nnz]` (read off [colIdx]'s runtime extent, never baked).
 */
fun sparseMatmulValuesAdjoint(
    upstream: DTensor<*, F32>,
    dense: DTensor<*, F32>,
    colIdx: DTensor<*, I32>,
    rowPtr: DTensor<*, I32>,
): DTensor<Rank1<Sym>, F32> {
    require(dense.rank == 2) {
        "sparseMatmulValuesAdjoint: dense operand must be rank-2 (C, D); got ${dense.dims.toList()}"
    }
    val up = upstream.hostF32()
    val bv = dense.hostF32()
    val ci = colIdx.hostI32()
    val rp = rowPtr.hostI32()
    val c = dense.dims[0]
    val d = dense.dims[1]
    validateCsrComponents("sparseMatmulValuesAdjoint", ci.size, ci, rp, c)
    val n = rp.size - 1
    require(up.size == n * d) {
        "sparseMatmulValuesAdjoint: upstream size ${up.size} != N $n * D $d"
    }
    val out = FloatArray(ci.size)
    for (i in 0 until n) {
        val upOff = i * d
        for (k in rp[i] until rp[i + 1]) {
            val dOff = ci[k] * d
            var acc = 0.0
            for (j in 0 until d) acc += up[upOff + j].toDouble() * bv[dOff + j]
            out[k] = acc.toFloat()
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(ci.size), F32)
}

/**
 * §0.4.384 — Phase A3b slice 1, the rank-4 substrate: the shared conv engine.
 * NCHW input; the kernel reads OIHW `[Co, Ci, kh, kw]` when [transpose] is false
 * and IOHW `[Ci, Co, kh, kw]` when true — the layouts the dxir interpreter's
 * `evalConv2d` and the StableHLO emitter fix. This is a port of that eval, kept
 * deliberately literal: same output-extent formula, same tap→input coordinate
 * mapping, same tap skipping (padding edges, lhs-dilation holes), and the same
 * `Double` accumulator walked in the same `i → ky → kx` order. Bit-exactness
 * against the interpreter is the point — the K2 plugin synthesises `grad {}` conv
 * gradients into calls on these twins, so the interpreted dxir stays a usable
 * oracle for the generated host code.
 *
 * Attrs mirror the IR's one-for-one: `window_strides` [sH, sW], `padding`
 * [[top, bottom], [left, right]] (negative values CROP, as in StableHLO),
 * `lhs_dilation` (interior-dilates the input — the transposed-conv mechanism),
 * `rhs_dilation` (à-trous kernel), `window_reversal` (spatially flips the kernel
 * taps — what [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule]'s `dX` needs).
 * `feature_group_count` / `batch_group_count` are 1 here: the interpreter grew a
 * grouped arm in §0.4.429, but the host twins (and with them the `grad {}` /
 * `jvp {}` synthesis surface) are part of that section's named deferral — the K2
 * synthesis rejects grouped conv ops loudly rather than convolving the wrong way.
 */
private fun conv2dEngine(
    lhs: DTensor<*, F32>,
    rhs: DTensor<*, F32>,
    transpose: Boolean,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<Shape, F32> {
    val ld = lhs.dims
    val rd = rhs.dims
    val opName = if (transpose) "convTranspose2d" else "conv2d"
    require(ld.size == 4 && rd.size == 4) {
        "$opName: rank-4 NCHW lhs and rank-4 kernel required; got ${ld.toList()} / ${rd.toList()}"
    }
    require(
        strideH > 0 && strideW > 0 && lhsDilH > 0 && lhsDilW > 0 && rhsDilH > 0 && rhsDilW > 0
    ) {
        "$opName: strides and dilations must be positive; got strides [$strideH, $strideW], " +
            "lhs_dilation [$lhsDilH, $lhsDilW], rhs_dilation [$rhsDilH, $rhsDilW]"
    }
    val nB = ld[0]
    val cIn = ld[1]
    val h = ld[2]
    val w = ld[3]
    val kh = rd[2]
    val kw = rd[3]
    val cOut: Int
    val cKIn: Int
    if (transpose) {
        cKIn = rd[0]
        cOut = rd[1]
    } else {
        cOut = rd[0]
        cKIn = rd[1]
    }
    require(cKIn == cIn) { "$opName: kernel input channels $cKIn ≠ lhs channels $cIn" }

    val hDil = (h - 1) * lhsDilH + 1
    val wDil = (w - 1) * lhsDilW + 1
    val kEffH = (kh - 1) * rhsDilH + 1
    val kEffW = (kw - 1) * rhsDilW + 1
    val hOut = (hDil + padTop + padBottom - kEffH) / strideH + 1
    val wOut = (wDil + padLeft + padRight - kEffW) / strideW + 1
    require(hOut > 0 && wOut > 0) {
        "$opName: derived output extents [$hOut, $wOut] are empty — window [$kh, $kw] with " +
            "rhs_dilation [$rhsDilH, $rhsDilW] does not fit the padded, dilated input [$hDil, $wDil]"
    }

    val lv = lhs.hostF32()
    val rv = rhs.hostF32()
    val out = FloatArray(nB * cOut * hOut * wOut)
    var outIdx = 0
    for (n in 0 until nB) {
        for (o in 0 until cOut) {
            for (y in 0 until hOut) {
                for (x in 0 until wOut) {
                    var acc = 0.0
                    for (i in 0 until cIn) {
                        for (ky in 0 until kh) {
                            // Tap position in the dilated+padded input space.
                            val yDil = y * strideH + ky * rhsDilH - padTop
                            if (yDil < 0 || yDil % lhsDilH != 0) continue
                            val inY = yDil / lhsDilH
                            if (inY >= h) continue
                            val wKy = if (revH) kh - 1 - ky else ky
                            for (kx in 0 until kw) {
                                val xDil = x * strideW + kx * rhsDilW - padLeft
                                if (xDil < 0 || xDil % lhsDilW != 0) continue
                                val inX = xDil / lhsDilW
                                if (inX >= w) continue
                                val wKx = if (revW) kw - 1 - kx else kx
                                val wIdx = if (transpose) {
                                    ((i * cOut + o) * kh + wKy) * kw + wKx
                                } else {
                                    ((o * cIn + i) * kh + wKy) * kw + wKx
                                }
                                acc += lv[((n * cIn + i) * h + inY) * w + inX].toDouble() * rv[wIdx]
                            }
                        }
                    }
                    out[outIdx++] = acc.toFloat()
                }
            }
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(nB, cOut, hOut, wOut), F32)
}

/**
 * §0.4.384 — 2-D convolution, NCHW input against an OIHW `[Co, Ci, kh, kw]`
 * kernel. The result erases to `DTensor<Shape, F32>`: its spatial extents are a
 * runtime function of the input's and of stride/padding, which no static shape
 * witness can carry — the convention `reshape`, `slice`, `concat` and
 * `broadcastTo` follow.
 *
 * Two arities, and deliberately NO default parameter values: `conv2d(w)` is the
 * plain valid conv (stride 1, no padding) and the 7-argument form takes the attrs
 * POSITIONALLY in a fixed order — `(w, strideH, strideW, padTop, padBottom,
 * padLeft, padRight)`, the same spelling as `slice(start, end, axis)`. Defaults
 * would be a silent-wrongness trap here: K2 unwraps a named argument like
 * `padTop = 1` to its bare literal BEFORE the plugin's FIR lowering sees the
 * call, without reordering it into its parameter's position, so the lowering
 * cannot tell `padTop = 1` from `strideH = 1` and would fold the padding into the
 * stride. Arity is unambiguous, so an unwritable spelling is a compile error
 * instead of a mis-folded attr.
 */
fun <S : Shape> DTensor<S, F32>.conv2d(w: DTensor<*, F32>): DTensor<Shape, F32> =
    conv2dGeneral(this, w, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, false)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F32>.conv2d(
    w: DTensor<*, F32>,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F32> = conv2dGeneral(
    this, w, strideH, strideW, 1, 1, 1, 1, padTop, padBottom, padLeft, padRight, false, false,
)

/**
 * §0.4.384 — transposed 2-D convolution (the "deconvolution" / fractionally-strided
 * conv), NCHW input against an IOHW `[Ci, Co, kh, kw]` kernel — note the channel
 * order is the reverse of [conv2d]'s, because the op contracts over the kernel's
 * FIRST axis and emits the second.
 *
 * [strideH]/[strideW] are the UPSAMPLING factor: they map onto the IR's
 * `lhs_dilation` with `window_strides` left at `[1, 1]`, which is exactly how
 * [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule] spells `dX` and how StableHLO
 * expresses a transposed conv. `padTop`/… are the IR's `padding` attr verbatim —
 * applied to the dilated input, and NEGATIVE values crop (this is how the conv
 * adjoint lands back on the primal input's exact extents). Same two-arity,
 * no-defaults contract as [conv2d], for the same reason.
 */
fun <S : Shape> DTensor<S, F32>.convTranspose2d(w: DTensor<*, F32>): DTensor<Shape, F32> =
    convTranspose2dGeneral(this, w, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, false, false)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F32>.convTranspose2d(
    w: DTensor<*, F32>,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F32> = convTranspose2dGeneral(
    this, w, 1, 1, strideH, strideW, 1, 1, padTop, padBottom, padLeft, padRight, false, false,
)

/**
 * §0.4.384 — fixed-arity synthesis delegates: every attr the dxir CONV2D /
 * CONV_TRANSPOSE2D ops carry, as positional `Int`/`Boolean` arguments, with the
 * result's shape witness `S` supplied by the caller (synthesis passes the derived
 * IrType's shape argument). All attrs explicit and none defaulted because these
 * are the calls [io.tlaloc.plugin.DxirToIrSynthesis] builds when it replays a conv
 * gradient body — the attrs come straight off the dxir op, and a delegate that
 * omitted one (say `lhs_dilation`, or `window_reversal`) would silently evaluate a
 * different convolution. The usual IrVararg reason applies: synthesis builds
 * positional `IrCall` arguments and cannot construct an array literal, so the
 * attr lists arrive as scalars.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dGeneral(
    x: DTensor<*, F32>,
    w: DTensor<*, F32>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return conv2dEngine(
        x, w, false, strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padBottom, padLeft, padRight, revH, revW,
    ) as DTensor<S, F32>
}

@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dGeneral(
    x: DTensor<*, F32>,
    w: DTensor<*, F32>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return conv2dEngine(
        x, w, true, strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padBottom, padLeft, padRight, revH, revW,
    ) as DTensor<S, F32>
}

/**
 * §0.4.385 — the host twin of `OpKind.CONV2D_DATA_ADJOINT`: a 2-D conv's gradient
 * w.r.t. its INPUT, fused into one op.
 *
 * Mathematically it is the lhs-dilated, tap-reversed transposed convolution of
 * [upstream] against [kernel], padded so the result lands exactly on the primal
 * input's extents. That padding is why the op exists at all: it is SOLVED from
 * those extents, and under `grad {}` every extent in the graph is a -1 sentinel,
 * so padding baked at transform time is arithmetic garbage that every consumer
 * would faithfully honour. [xTemplate] therefore contributes SHAPE ONLY — its
 * values are never read — and the solve happens here, at execution, against
 * `dims` that are real:
 *
 *     kEff = (k−1)·d + 1        dilSize = (hOut−1)·s + 1
 *     low  = kEff − 1 − p_low   high    = p_low + H − dilSize
 *
 * with `k` the kernel's spatial extent, `hOut` the upstream's, `H` the target's,
 * `s` [strideH]/[strideW], `d` [rhsDilH]/[rhsDilW] and `p_low` [padTop]/[padLeft]
 * — all read off the primal conv. Note only the primal's LOW padding is needed:
 * its high side is already implicit in the upstream's runtime shape, since that
 * shape is what the primal's floor-division produced.
 *
 * The result's shape witness is [xTemplate]'s `S` — not a guess, because the
 * template IS the shape this op produces.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dDataAdjoint(
    upstream: DTensor<*, F32>,
    kernel: DTensor<*, F32>,
    xTemplate: DTensor<S, F32>,
    strideH: Int,
    strideW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F32> {
    val up = upstream.dims
    val k = kernel.dims
    val target = xTemplate.dims
    val strides = intArrayOf(strideH, strideW)
    val dil = intArrayOf(rhsDilH, rhsDilW)
    val pLow = intArrayOf(padTop, padLeft)
    fun solve(axis: Int): IntArray {
        val kEff = (k[2 + axis] - 1) * dil[axis] + 1
        val dilSize = (up[2 + axis] - 1) * strides[axis] + 1
        val low = kEff - 1 - pLow[axis]
        return intArrayOf(low, pLow[axis] + target[2 + axis] - dilSize)
    }
    val h = solve(0)
    val w = solve(1)
    val dx = conv2dEngine(
        upstream, kernel, true,
        1, 1, strideH, strideW, rhsDilH, rhsDilW,
        h[0], h[1], w[0], w[1], true, true,
    )
    require(dx.dims.contentEquals(target)) {
        "conv2dDataAdjoint: solved padding landed on ${dx.dims.toList()} but the template " +
            "is ${target.toList()} (upstream ${up.toList()}, kernel ${k.toList()}, " +
            "strides [$strideH, $strideW], rhs_dilation [$rhsDilH, $rhsDilW], " +
            "primal padding low [$padTop, $padLeft])"
    }
    @Suppress("UNCHECKED_CAST")
    return dx as DTensor<S, F32>
}

/**
 * §0.4.385 — the host twin of `OpKind.CONV2D_KERNEL_ADJOINT`: a 2-D conv's gradient
 * w.r.t. its KERNEL, fused into one op.
 *
 * This is the batch↔feature transposed trick — `dW = (Xᵀ ⋆ dYᵀ)ᵀ`, where X reads
 * as `[Ci, N, H, W]` and dY as an OIHW kernel `[Co, N, Ho, Wo]` with `o = Co`,
 * `i = N`, and the primal's stride and rhs_dilation SWAP roles — padded so the
 * inner conv's result is exactly `[Ci, Co, kh, kw]`, then transposed back to
 * OIHW. All three transposes are folded in here, so a gradient body needs no
 * rank-4 TRANSPOSE nodes and the result's shape witness is simply
 * [wTemplate]'s `S`. The padding solve, likewise read from runtime dims:
 *
 *     dilSize = (hOut−1)·s + 1
 *     low     = p_low
 *     high    = (k−1)·d + dilSize − H − p_low
 *
 * with `k` the target kernel's spatial extent and `H` the primal input's. The
 * high side is the crop that absorbs the primal's floor-division remainder (a
 * stride-2 conv over height 5 leaves an unused row), which is why it can be
 * negative. [wTemplate] contributes SHAPE ONLY.
 */
@Suppress("LongParameterList")
fun <S : Shape> conv2dKernelAdjoint(
    x: DTensor<*, F32>,
    upstream: DTensor<*, F32>,
    wTemplate: DTensor<S, F32>,
    strideH: Int,
    strideW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F32> {
    val xD = x.dims
    val up = upstream.dims
    val target = wTemplate.dims
    val strides = intArrayOf(strideH, strideW)
    val dil = intArrayOf(rhsDilH, rhsDilW)
    val pLow = intArrayOf(padTop, padLeft)
    fun solve(axis: Int): IntArray {
        val dilSize = (up[2 + axis] - 1) * strides[axis] + 1
        val low = pLow[axis]
        return intArrayOf(low, (target[2 + axis] - 1) * dil[axis] + dilSize - xD[2 + axis] - low)
    }
    val h = solve(0)
    val w = solve(1)
    @Suppress("UNCHECKED_CAST")
    val xT = (x as DTensor<Shape, F32>).transpose(1, 0, 2, 3)
    @Suppress("UNCHECKED_CAST")
    val upT = (upstream as DTensor<Shape, F32>).transpose(1, 0, 2, 3)
    // [Ci, Co, kh, kw]: the target with its channel axes swapped.
    val dwt = conv2dEngine(
        xT, upT, false,
        rhsDilH, rhsDilW, 1, 1, strideH, strideW,
        h[0], h[1], w[0], w[1], false, false,
    )
    val dw = dwt.transpose(1, 0, 2, 3)
    require(dw.dims.contentEquals(target)) {
        "conv2dKernelAdjoint: solved padding landed on ${dw.dims.toList()} but the template " +
            "is ${target.toList()} (x ${xD.toList()}, upstream ${up.toList()}, " +
            "strides [$strideH, $strideW], rhs_dilation [$rhsDilH, $rhsDilW], " +
            "primal padding low [$padTop, $padLeft])"
    }
    @Suppress("UNCHECKED_CAST")
    return dw as DTensor<S, F32>
}

/**
 * §0.4.391 — the shared engine for the two transposed-conv adjoint twins. A port of
 * the interpreter's `evalConvTransposeAdjoint`, kept literal so the host result is
 * bit-exact against the interpreted dxir (same Double accumulator, same loop order,
 * same single Float conversion).
 *
 * No padding solve is involved: the primal's tap maps input↔output through
 * `yDil = yo·s + ky·d − p_low` with `yDil` a multiple of the lhs dilation, so
 * inverting that one equation per tap absorbs the padding, both dilations, the
 * strides and the kernel reversal together. Only the LOW padding is therefore
 * needed, as in [conv2dDataAdjoint].
 *
 * [up] is the upstream and [other] the kernel (IOHW `[Ci, Co, kh, kw]`) when
 * [dataAdj], or `x` when not.
 */
@Suppress("LongParameterList")
private fun convTranspose2dAdjointEngine(
    up: FloatArray,
    other: FloatArray,
    dataAdj: Boolean,
    nB: Int,
    cIn: Int,
    h: Int,
    w: Int,
    cOut: Int,
    hOut: Int,
    wOut: Int,
    kh: Int,
    kw: Int,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): FloatArray = if (dataAdj) {
    val out = FloatArray(nB * cIn * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (i in 0 until cIn) {
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    var acc = 0.0
                    for (ky in 0 until kh) {
                        val num = iy * lhsDilH + padTop - ky * rhsDilH
                        if (num < 0 || num % strideH != 0) continue
                        val yo = num / strideH
                        if (yo >= hOut) continue
                        val wKy = if (revH) kh - 1 - ky else ky
                        for (kx in 0 until kw) {
                            val num2 = ix * lhsDilW + padLeft - kx * rhsDilW
                            if (num2 < 0 || num2 % strideW != 0) continue
                            val xo = num2 / strideW
                            if (xo >= wOut) continue
                            val wKx = if (revW) kw - 1 - kx else kx
                            val upBase = ((n * cOut) * hOut + yo) * wOut + xo
                            for (o in 0 until cOut) {
                                acc += up[upBase + o * hOut * wOut].toDouble() *
                                    other[((i * cOut + o) * kh + wKy) * kw + wKx]
                            }
                        }
                    }
                    out[outIdx++] = acc.toFloat()
                }
            }
        }
    }
    out
} else {
    val acc = DoubleArray(cIn * cOut * kh * kw)
    for (n in 0 until nB) {
        for (o in 0 until cOut) {
            for (yo in 0 until hOut) {
                for (xo in 0 until wOut) {
                    val upVal = up[((n * cOut + o) * hOut + yo) * wOut + xo].toDouble()
                    for (i in 0 until cIn) {
                        for (ky in 0 until kh) {
                            val yD = yo * strideH + ky * rhsDilH - padTop
                            if (yD < 0 || yD % lhsDilH != 0) continue
                            val inY = yD / lhsDilH
                            if (inY >= h) continue
                            val wKy = if (revH) kh - 1 - ky else ky
                            for (kx in 0 until kw) {
                                val xD = xo * strideW + kx * rhsDilW - padLeft
                                if (xD < 0 || xD % lhsDilW != 0) continue
                                val inX = xD / lhsDilW
                                if (inX >= w) continue
                                val wKx = if (revW) kw - 1 - kx else kx
                                acc[((i * cOut + o) * kh + wKy) * kw + wKx] +=
                                    upVal * other[((n * cIn + i) * h + inY) * w + inX].toDouble()
                            }
                        }
                    }
                }
            }
        }
    }
    FloatArray(acc.size) { acc[it].toFloat() }
}

/**
 * §0.4.391 — the host twin of `OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT`: the gradient
 * of a transposed convolution w.r.t. its INPUT. [xTemplate] contributes SHAPE ONLY.
 *
 * Attrs are the primal transposed conv's, all literals. The StableHLO emitter also
 * has an arm (§0.4.393), reached through the conv-of-the-dilated-input identity
 * rather than this index inversion, and it rejects `window_reversal` — which this
 * twin handles, as does the interpreter.
 */
@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dDataAdjoint(
    upstream: DTensor<*, F32>,
    kernel: DTensor<*, F32>,
    xTemplate: DTensor<S, F32>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F32> {
    val up = upstream.dims
    val k = kernel.dims
    val target = xTemplate.dims
    require(up.size == 4 && k.size == 4 && target.size == 4) {
        "convTranspose2dDataAdjoint: rank-4 NCHW upstream, IOHW kernel and template required; " +
            "got ${up.toList()} / ${k.toList()} / ${target.toList()}"
    }
    require(up[0] == target[0] && k[0] == target[1] && k[1] == up[1]) {
        "convTranspose2dDataAdjoint: channel/batch mismatch — upstream ${up.toList()}, " +
            "kernel ${k.toList()}, x ${target.toList()}"
    }
    val dx = convTranspose2dAdjointEngine(
        upstream.hostF32(), kernel.hostF32(), true,
        target[0], target[1], target[2], target[3],
        up[1], up[2], up[3], k[2], k[3],
        strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padLeft, revH, revW,
    )
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F32>(HostF32Storage(dx), target.copyOf(), F32) as DTensor<S, F32>
}

/**
 * §0.4.391 — the host twin of `OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT`: the gradient
 * of a transposed convolution w.r.t. its IOHW KERNEL. [x] is a VALUE operand here (the
 * gather reads it); [wTemplate] contributes shape only.
 */
@Suppress("LongParameterList")
fun <S : Shape> convTranspose2dKernelAdjoint(
    x: DTensor<*, F32>,
    upstream: DTensor<*, F32>,
    wTemplate: DTensor<S, F32>,
    strideH: Int,
    strideW: Int,
    lhsDilH: Int,
    lhsDilW: Int,
    rhsDilH: Int,
    rhsDilW: Int,
    padTop: Int,
    padLeft: Int,
    revH: Boolean,
    revW: Boolean,
): DTensor<S, F32> {
    val xd = x.dims
    val up = upstream.dims
    val target = wTemplate.dims
    require(xd.size == 4 && up.size == 4 && target.size == 4) {
        "convTranspose2dKernelAdjoint: rank-4 NCHW x, NCHW upstream and IOHW template " +
            "required; got ${xd.toList()} / ${up.toList()} / ${target.toList()}"
    }
    require(up[0] == xd[0] && target[0] == xd[1] && target[1] == up[1]) {
        "convTranspose2dKernelAdjoint: channel/batch mismatch — x ${xd.toList()}, " +
            "upstream ${up.toList()}, kernel ${target.toList()}"
    }
    val dw = convTranspose2dAdjointEngine(
        upstream.hostF32(), x.hostF32(), false,
        xd[0], xd[1], xd[2], xd[3],
        up[1], up[2], up[3], target[2], target[3],
        strideH, strideW, lhsDilH, lhsDilW, rhsDilH, rhsDilW,
        padTop, padLeft, revH, revW,
    )
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F32>(HostF32Storage(dw), target.copyOf(), F32) as DTensor<S, F32>
}

/**
 * §0.4.386 — the host pooling engine: a port of the dxir interpreter's
 * `evalPool2d`, kept deliberately literal so the host result is bit-exact against
 * the interpreted dxir (same `Double` accumulator, same `n → c → y → x → ky → kx`
 * order, same single Float conversion at the end). §0.4.389 generalised it from
 * avg-only to both pooling kinds with an [isMax] flag, mirroring the interpreter's
 * own single-`evalPool2d`-two-kinds structure.
 *
 * count_include_pad for the average branch, the interpreter's convention: the sum
 * divides by the FULL window `kh·kw`, padding included, so a window hanging off
 * the edge is averaged over taps that were not there. That choice is what makes
 * [avgPool2dGrad]'s uniform `1/(kh·kw)` spread the exact adjoint.
 */
private fun pool2dEngine(
    x: DTensor<*, F32>,
    isMax: Boolean,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F32> {
    val opName = if (isMax) "maxPool2d" else "avgPool2d"
    val d = x.dims
    require(d.size == 4) { "$opName: rank-4 NCHW input required; got ${d.toList()}" }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "$opName: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    val nB = d[0]
    val c = d[1]
    val h = d[2]
    val w = d[3]
    val hOut = (h + padTop + padBottom - windowH) / strideH + 1
    val wOut = (w + padLeft + padRight - windowW) / strideW + 1
    require(hOut > 0 && wOut > 0) {
        "$opName: derived output extents [$hOut, $wOut] are empty — window " +
            "[$windowH, $windowW] does not fit the padded input [$h, $w]"
    }
    val v = x.hostF32()
    val windowSize = windowH * windowW
    val out = FloatArray(nB * c * hOut * wOut)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val planeBase = (n * c + ch) * h * w
            for (y in 0 until hOut) {
                for (xo in 0 until wOut) {
                    var acc = if (isMax) Double.NEGATIVE_INFINITY else 0.0
                    for (ky in 0 until windowH) {
                        val inY = y * strideH + ky - padTop
                        if (inY < 0 || inY >= h) continue
                        for (kx in 0 until windowW) {
                            val inX = xo * strideW + kx - padLeft
                            if (inX < 0 || inX >= w) continue
                            val e = v[planeBase + inY * w + inX].toDouble()
                            acc = if (isMax) maxOf(acc, e) else acc + e
                        }
                    }
                    out[outIdx++] = if (isMax) acc.toFloat() else (acc / windowSize).toFloat()
                }
            }
        }
    }
    return DTensor(HostF32Storage(out), intArrayOf(nB, c, hOut, wOut), F32)
}

/**
 * §0.4.386 — 2-D average pooling, NCHW. Two arities and no default parameter
 * values, for the reason documented on [conv2d]: K2 unwraps a named argument
 * before the plugin's FIR lowering sees it and does not reorder it, so attrs must
 * be positional to be unambiguous. `avgPool2d(windowH, windowW)` is the
 * non-overlapping pool (strides default to the window — the interpreter's own
 * default, and PyTorch's `AvgPool2d(k)` shape); the 8-argument form adds strides
 * and all four padding sides. The result erases to `DTensor<Shape, F32>` like
 * every other extent-changing host op.
 */
fun <S : Shape> DTensor<S, F32>.avgPool2d(windowH: Int, windowW: Int): DTensor<Shape, F32> =
    avgPool2dGeneral(this, windowH, windowW, windowH, windowW, 0, 0, 0, 0)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F32>.avgPool2d(
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F32> = avgPool2dGeneral(
    this, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
)

/**
 * §0.4.386 — fixed-arity synthesis delegate for `OpKind.AVGPOOL2D`, every attr
 * explicit and positional (the usual IrVararg reason, and the result's shape
 * witness `S` comes from the caller's derived IrType).
 */
@Suppress("LongParameterList")
fun <S : Shape> avgPool2dGeneral(
    x: DTensor<*, F32>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return pool2dEngine(
        x, false, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
    ) as DTensor<S, F32>
}

/**
 * §0.4.386 — the host twin of `OpKind.AVGPOOL2D_GRAD`: each input element collects
 * the upstream of every output window covering it, divided by the FULL window
 * `kh·kw` (count_include_pad, mirroring [avgPool2dEngine]).
 *
 * The covering output is found by INVERTING the window rather than by solving a
 * padding: for target row `iy` and tap `ky` it is `(iy + padTop − ky) / strideH`
 * when that divides evenly and lands in range. So the only attrs are literal facts
 * off the primal, [xTemplate] contributes SHAPE ONLY (its values are never read),
 * and nothing needs an extent at compile time — which is the whole point, since
 * under `grad {}` every extent is a -1 sentinel. Only the LOW padding participates:
 * the high side is implied by the upstream's runtime shape, exactly as for
 * [conv2dDataAdjoint].
 */
@Suppress("LongParameterList")
fun <S : Shape> avgPool2dGrad(
    upstream: DTensor<*, F32>,
    xTemplate: DTensor<S, F32>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F32> {
    val up = upstream.dims
    val target = xTemplate.dims
    require(up.size == 4 && target.size == 4) {
        "avgPool2dGrad: rank-4 NCHW upstream and template required; got " +
            "${up.toList()} / ${target.toList()}"
    }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "avgPool2dGrad: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    require(up[0] == target[0] && up[1] == target[1]) {
        "avgPool2dGrad: upstream batch/channels [${up[0]}, ${up[1]}] ≠ target " +
            "[${target[0]}, ${target[1]}]"
    }
    val nB = target[0]
    val c = target[1]
    val h = target[2]
    val w = target[3]
    val hOut = up[2]
    val wOut = up[3]
    val uv = upstream.hostF32()
    val windowSize = windowH * windowW
    val out = FloatArray(nB * c * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val upBase = (n * c + ch) * hOut * wOut
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    var acc = 0.0
                    for (ky in 0 until windowH) {
                        val dy = iy + padTop - ky
                        if (dy < 0 || dy % strideH != 0) continue
                        val y = dy / strideH
                        if (y >= hOut) continue
                        for (kx in 0 until windowW) {
                            val dx = ix + padLeft - kx
                            if (dx < 0 || dx % strideW != 0) continue
                            val x = dx / strideW
                            if (x >= wOut) continue
                            acc += uv[upBase + y * wOut + x].toDouble()
                        }
                    }
                    out[outIdx++] = (acc / windowSize).toFloat()
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F32>(HostF32Storage(out), target.copyOf(), F32) as DTensor<S, F32>
}

/**
 * §0.4.389 — 2-D max pooling, NCHW. Same two-arity, no-defaults contract as
 * [avgPool2d] and for the same reason (K2 unwraps named arguments before the
 * plugin's FIR lowering sees them, so attrs must be positional to be
 * unambiguous): `maxPool2d(windowH, windowW)` is the classic non-overlapping pool
 * (strides default to the window, PyTorch's `MaxPool2d(k)` shape), and the
 * 8-argument form adds strides and all four padding sides.
 */
fun <S : Shape> DTensor<S, F32>.maxPool2d(windowH: Int, windowW: Int): DTensor<Shape, F32> =
    maxPool2dGeneral(this, windowH, windowW, windowH, windowW, 0, 0, 0, 0)

@Suppress("LongParameterList")
fun <S : Shape> DTensor<S, F32>.maxPool2d(
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<Shape, F32> = maxPool2dGeneral(
    this, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
)

/** §0.4.389 — fixed-arity synthesis delegate for `OpKind.MAXPOOL2D`. */
@Suppress("LongParameterList")
fun <S : Shape> maxPool2dGeneral(
    x: DTensor<*, F32>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padBottom: Int,
    padLeft: Int,
    padRight: Int,
): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return pool2dEngine(
        x, true, windowH, windowW, strideH, strideW, padTop, padBottom, padLeft, padRight,
    ) as DTensor<S, F32>
}

/**
 * §0.4.389 — the host twin of `OpKind.MAXPOOL2D_GRAD`: each input element receives
 * the upstream of every output window it WINS, i.e. every window whose max it
 * equals. Ties route the full upstream to every winner (the MaxRule/JAX-select
 * convention) — deliberately not XLA's `select_and_scatter`, which picks a single
 * winner and would make the GPU gradient disagree with the host one on exact ties,
 * which are common after a relu.
 *
 * Like [avgPool2dGrad] this INVERTS the window per input element rather than
 * nearest-upsampling the pooled value and the upstream back to x's shape and
 * masking — the rank-6 formulation that kept maxpool out of `grad {}`, since those
 * intermediates bake extents that are -1 sentinels there. Nothing here reads a
 * compile-time extent.
 *
 * [y] is the pooled value `maxPool2d(x)`, supplied rather than recomputed (the rule
 * materialises it in the gradient body). Unlike the conv/avgpool templates, [x] is
 * a VALUE operand — its elements are compared against [y] — though its extents
 * still come from runtime `dims`.
 */
@Suppress("LongParameterList")
fun <S : Shape> maxPool2dGrad(
    upstream: DTensor<*, F32>,
    x: DTensor<*, F32>,
    y: DTensor<*, F32>,
    windowH: Int,
    windowW: Int,
    strideH: Int,
    strideW: Int,
    padTop: Int,
    padLeft: Int,
): DTensor<S, F32> {
    val up = upstream.dims
    val target = x.dims
    require(up.size == 4 && target.size == 4 && y.dims.size == 4) {
        "maxPool2dGrad: rank-4 NCHW upstream/x/y required; got " +
            "${up.toList()} / ${target.toList()} / ${y.dims.toList()}"
    }
    require(windowH > 0 && windowW > 0 && strideH > 0 && strideW > 0) {
        "maxPool2dGrad: window and strides must be positive; got window [$windowH, $windowW], " +
            "strides [$strideH, $strideW]"
    }
    require(up[0] == target[0] && up[1] == target[1]) {
        "maxPool2dGrad: upstream batch/channels [${up[0]}, ${up[1]}] ≠ x's " +
            "[${target[0]}, ${target[1]}]"
    }
    val nB = target[0]
    val c = target[1]
    val h = target[2]
    val w = target[3]
    val hOut = up[2]
    val wOut = up[3]
    val uv = upstream.hostF32()
    val xv = x.hostF32()
    val yv = y.hostF32()
    val out = FloatArray(nB * c * h * w)
    var outIdx = 0
    for (n in 0 until nB) {
        for (ch in 0 until c) {
            val plane = (n * c + ch) * h * w
            val upPlane = (n * c + ch) * hOut * wOut
            for (iy in 0 until h) {
                for (ix in 0 until w) {
                    val xe = xv[plane + iy * w + ix]
                    var acc = 0.0
                    for (ky in 0 until windowH) {
                        val dy = iy + padTop - ky
                        if (dy < 0 || dy % strideH != 0) continue
                        val oy = dy / strideH
                        if (oy >= hOut) continue
                        for (kx in 0 until windowW) {
                            val dx = ix + padLeft - kx
                            if (dx < 0 || dx % strideW != 0) continue
                            val ox = dx / strideW
                            if (ox >= wOut) continue
                            val wIdx = upPlane + oy * wOut + ox
                            if (xe == yv[wIdx]) acc += uv[wIdx].toDouble()
                        }
                    }
                    out[outIdx++] = acc.toFloat()
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F32>(HostF32Storage(out), target.copyOf(), F32) as DTensor<S, F32>
}

/**
 * §0.4.390 — TRAINING-mode batch normalisation, NCHW with the feature axis at 1:
 * per channel, subtract the mean and divide by the standard deviation computed
 * over the batch AND spatial extents of this call, then apply the per-channel
 * [scale] (γ) and [offset] (β).
 *
 * Two conventions worth stating because the AD path inherits both: the variance is
 * BIASED (divide by `N·H·W`, i.e. `mean((x−μ)²)`, matching PyTorch's training-mode
 * normalisation — the unbiased correction belongs only in the running-variance
 * update, which is not this function's job), and the statistics come from the
 * argument rather than from running estimates. That distinguishes it from
 * `OpKind.BATCHNORM`, which is the INFERENCE form (five operands: input, scale,
 * offset, mean, variance) that the Layer-3 recognizer emits for fused kernels.
 *
 * `grad {}` gets this without a VjpRule: the plugin's FIR lowering desugars the
 * call into MEAN/SUB/MUL/ADD/SQRT/DIV nodes whose adjoints already exist and are
 * sentinel-safe (the `maximum`/`clip` pattern). This host function is the
 * plain-runtime twin of that desugaring and does the same arithmetic.
 */
fun <S : Shape> DTensor<S, F32>.batchNorm(
    scale: DTensor<*, F32>,
    offset: DTensor<*, F32>,
): DTensor<Shape, F32> = batchNormGeneral(this, scale, offset, 1e-5f)

fun <S : Shape> DTensor<S, F32>.batchNorm(
    scale: DTensor<*, F32>,
    offset: DTensor<*, F32>,
    eps: Float,
): DTensor<Shape, F32> = batchNormGeneral(this, scale, offset, eps)

/** §0.4.390 — fixed-arity form with an explicit `eps`. Rank-4 NCHW only in v1. */
fun <S : Shape> batchNormGeneral(
    x: DTensor<*, F32>,
    scale: DTensor<*, F32>,
    offset: DTensor<*, F32>,
    eps: Float,
): DTensor<S, F32> {
    val d = x.dims
    require(d.size == 4) { "batchNorm: rank-4 NCHW input required; got ${d.toList()}" }
    val nB = d[0]
    val c = d[1]
    val h = d[2]
    val w = d[3]
    require(
        scale.dims.size == 1 && scale.dims[0] == c && offset.dims.size == 1 && offset.dims[0] == c
    ) {
        "batchNorm: scale/offset must be rank-1 [C=$c]; got " +
            "${scale.dims.toList()} / ${offset.dims.toList()}"
    }
    val spatial = h * w
    val count = (nB * spatial).toDouble()
    require(count > 0.0) { "batchNorm: empty input ${d.toList()}" }
    val v = x.hostF32()
    val g = scale.hostF32()
    val b = offset.hostF32()
    val out = FloatArray(v.size)
    for (ch in 0 until c) {
        // μ over this channel's batch and spatial extents.
        var sum = 0.0
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) sum += v[base + i].toDouble()
        }
        val mean = sum / count
        // Biased variance: mean((x − μ)²).
        var sq = 0.0
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) {
                val e = v[base + i].toDouble() - mean
                sq += e * e
            }
        }
        val invStd = 1.0 / sqrt(sq / count + eps.toDouble())
        val gamma = g[ch].toDouble()
        val beta = b[ch].toDouble()
        for (n in 0 until nB) {
            val base = (n * c + ch) * spatial
            for (i in 0 until spatial) {
                out[base + i] = ((v[base + i].toDouble() - mean) * invStd * gamma + beta).toFloat()
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return DTensor<Shape, F32>(HostF32Storage(out), d.copyOf(), F32) as DTensor<S, F32>
}

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
 * §0.4.397 — Phase A5c-3(iv): DiffKT's `stats()` — the `(mean, variance)`
 * pair over all elements, thin sugar over the A1 full reductions. Variance is
 * BIASED (divide by N, not N−1), matching both DiffKT's convention and the
 * per-channel statistic `batchNormGeneral` takes (§0.4.390). Host-level only
 * by design: DiffKT's `stats` is a convenience accessor, not a
 * differentiation surface — a Pair-returning body has no `grad {}` lowering
 * (the loss contract is scalar), and a loss that needs the pieces writes
 * `x.mean()` and the squared-deviation mean directly, both of which
 * differentiate today.
 */
fun <S : Shape> DTensor<S, F32>.stats(): Pair<DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>> {
    val v = hostF32()
    if (v.isEmpty()) {
        val zero = { DTensor<ScalarShape, F32>(HostF32Storage(floatArrayOf(0f)), intArrayOf(), F32) }
        return Pair(zero(), zero())
    }
    var acc = 0f
    for (x in v) acc += x
    val mu = acc / v.size
    var sq = 0f
    for (x in v) {
        val d = x - mu
        sq += d * d
    }
    return Pair(
        DTensor(HostF32Storage(floatArrayOf(mu)), intArrayOf(), F32),
        DTensor(HostF32Storage(floatArrayOf(sq / v.size)), intArrayOf(), F32),
    )
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
 * §0.4.390 — the three-axis shims. A rank-4 NCHW surface needs them: reducing over
 * the batch and both spatial axes (`mean(0, 2, 3)`, which is how training batchNorm
 * takes its per-channel statistics) is a three-axis reduction, and the fixed-arity
 * family stopped at two, so such a body fell out of synthesis scope.
 */
fun <S : Shape> sumOver3(
    x: DTensor<S, F32>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F32> = x.sum(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> meanOver3(
    x: DTensor<S, F32>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F32> = x.mean(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> maxOver3(
    x: DTensor<S, F32>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F32> = x.max(d0, d1, d2, keepDims = keepDims)

fun <S : Shape> minOver3(
    x: DTensor<S, F32>, d0: Int, d1: Int, d2: Int, keepDims: Boolean,
): DTensor<Shape, F32> = x.min(d0, d1, d2, keepDims = keepDims)

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
 * §0.4.373 — numpy unbroadcast: reduce [value] down to [template]'s RUNTIME
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
 * §0.4.415 — Phase B5 (customVjp): the host twin of the dxir
 * `CHECK_SHAPE_LIKE` op — a value-identity that ASSERTS the user vjpFn's
 * returned gradient has its operand's runtime shape before it is accumulated.
 * [template] is the customVjp operand the contribution belongs to and
 * contributes SHAPE ONLY (its values are never read). Under `grad {}`'s -1
 * sentinel dims the contract is undecidable at compile time, so this is where
 * a wrong-shaped user adjoint fails LOUDLY instead of silently corrupting the
 * gradient (design doc §4.1; the `conv2dDataAdjoint` template-assert
 * precedent). The value passes through as a re-wrapped view — no copy, its
 * storage is the user body's freshly computed tensor.
 */
fun <S : Shape> checkShapeLike(value: DTensor<*, F32>, template: DTensor<S, F32>): DTensor<S, F32> {
    require(value.dims.contentEquals(template.dims)) {
        "checkShapeLike: customVjp gradient body returned shape ${value.dims.toList()} for an " +
            "operand of shape ${template.dims.toList()} — the user vjpFn violates the VJP shape " +
            "contract (each d_operand must match its operand's shape)"
    }
    return DTensor(value.storage, value.dims, F32)
}

fun <S : Shape> sumToLike(value: DTensor<*, F32>, template: DTensor<S, F32>): DTensor<S, F32> {
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
        return DTensor(HostF32Storage(value.hostF32().copyOf()), t.copyOf(), F32)
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
    val v = value.hostF32()
    val out = FloatArray(outSize)
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
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/**
 * §0.4.399 — broadcast-to-template: stretch [value] up to [template]'s RUNTIME
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
fun <S : Shape> broadcastToLike(value: DTensor<*, F32>, template: DTensor<S, F32>): DTensor<S, F32> {
    val u = value.dims
    val t = template.dims
    // Identity fast path — copy rather than alias (a contribution may feed an
    // in-place accumulator downstream), the sumToLike convention.
    if (u.contentEquals(t)) {
        return DTensor(HostF32Storage(value.hostF32().copyOf()), t.copyOf(), F32)
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
    val v = value.hostF32()
    val out = FloatArray(outSize)
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
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/**
 * §0.4.374 — zero-pad-to-template: place [value] into a zero tensor of
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
fun <S : Shape> padToLike(value: DTensor<*, F32>, template: DTensor<S, F32>, low: IntArray): DTensor<S, F32> {
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
    val v = value.hostF32()
    val out = FloatArray(outSize)
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
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/** §0.4.374 — fixed-arity `padToLike` shims (synthesis bakes the `low`
 * offsets as Int consts, one per axis; mirror of the `stretchToRankN` family). */
fun <S : Shape> padToLikeRank1(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int): DTensor<S, F32> =
    padToLike(value, template, intArrayOf(l0))

fun <S : Shape> padToLikeRank2(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int, l1: Int): DTensor<S, F32> =
    padToLike(value, template, intArrayOf(l0, l1))

fun <S : Shape> padToLikeRank3(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int, l1: Int, l2: Int): DTensor<S, F32> =
    padToLike(value, template, intArrayOf(l0, l1, l2))

/**
 * §0.4.399 — window-at-literal-offset: cut out of [value] the window of
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
fun <S : Shape> sliceAtLike(value: DTensor<*, F32>, template: DTensor<S, F32>, low: IntArray): DTensor<S, F32> {
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
    val v = value.hostF32()
    val out = FloatArray(outSize)
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
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/** §0.4.399 — fixed-arity `sliceAtLike` shims (synthesis bakes the `low`
 * offsets as Int consts, one per axis; mirror of the `padToLikeRankN` family). */
fun <S : Shape> sliceAtLikeRank1(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int): DTensor<S, F32> =
    sliceAtLike(value, template, intArrayOf(l0))

fun <S : Shape> sliceAtLikeRank2(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int, l1: Int): DTensor<S, F32> =
    sliceAtLike(value, template, intArrayOf(l0, l1))

fun <S : Shape> sliceAtLikeRank3(value: DTensor<*, F32>, template: DTensor<S, F32>, l0: Int, l1: Int, l2: Int): DTensor<S, F32> =
    sliceAtLike(value, template, intArrayOf(l0, l1, l2))

/**
 * Phase A2b — the host twin of dxir `SLICE_LIKE`, i.e. CONCAT's adjoint: cut out
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
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    axis: Int,
    priors: List<DTensor<*, F32>>,
): DTensor<S, F32> {
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
    val src = value.hostF32()
    val out = FloatArray(outer * len * inner)
    var dst = 0
    for (o in 0 until outer) {
        val from = o * (v[axis] * inner) + start * inner
        src.copyInto(out, dst, from, from + len * inner)
        dst += len * inner
    }
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/** Fixed-arity `SLICE_LIKE` twins, one per PRIOR-template count — the usual
 * IrVararg reason (see [broadcastDimsRank1]): synthesis builds positional
 * `IrCall` arguments, so the operand count has to be in the callee's name.
 * §0.4.425 lifted the family from 3 priors to 7 (an 8-operand IR-level
 * CONCAT); the user-facing [concat]'s fold-to-binary never needs more than
 * ONE prior, so the wider twins serve only hand-built variadic CONCAT nodes
 * differentiated through the plugin. */
fun <S : Shape> sliceLikeStart(value: DTensor<*, F32>, thisTemplate: DTensor<S, F32>, axis: Int): DTensor<S, F32> =
    sliceWindow(value, thisTemplate, axis, emptyList())

fun <S : Shape> sliceLikeAfter1(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = sliceWindow(value, thisTemplate, axis, listOf(prior0))

fun <S : Shape> sliceLikeAfter2(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1))

fun <S : Shape> sliceLikeAfter3(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2))

fun <S : Shape> sliceLikeAfter4(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3))

fun <S : Shape> sliceLikeAfter5(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4))

fun <S : Shape> sliceLikeAfter6(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    prior5: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> =
    sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5))

fun <S : Shape> sliceLikeAfter7(
    value: DTensor<*, F32>,
    thisTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    prior5: DTensor<*, F32>,
    prior6: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> =
    sliceWindow(value, thisTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5, prior6))

/**
 * §0.4.404 — the host twin of dxir `PAD_LIKE`, i.e. SLICE_LIKE's transpose and
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
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    axis: Int,
    priors: List<DTensor<*, F32>>,
): DTensor<S, F32> {
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
    val src = value.hostF32()
    val out = FloatArray(outSize)
    var from = 0
    for (o in 0 until outer) {
        val dst = o * (t[axis] * inner) + start * inner
        src.copyInto(out, dst, from, from + len * inner)
        from += len * inner
    }
    return DTensor(HostF32Storage(out), t.copyOf(), F32)
}

/** Fixed-arity `PAD_LIKE` twins, one per PRIOR-template count — the usual
 * IrVararg reason (see [sliceLikeStart]): synthesis builds positional
 * `IrCall` arguments, so the operand count has to be in the callee's name.
 * §0.4.425 lifted the family from 3 priors to 7, mirroring the `SLICE_LIKE`
 * twins — the pair must stay closed under differentiation (SLICE_LIKE's VJP
 * is PAD_LIKE with the SAME priors), so the two bounds move together. */
fun <S : Shape> padLikeStart(value: DTensor<*, F32>, outTemplate: DTensor<S, F32>, axis: Int): DTensor<S, F32> =
    padWindow(value, outTemplate, axis, emptyList())

fun <S : Shape> padLikeAfter1(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = padWindow(value, outTemplate, axis, listOf(prior0))

fun <S : Shape> padLikeAfter2(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = padWindow(value, outTemplate, axis, listOf(prior0, prior1))

fun <S : Shape> padLikeAfter3(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2))

fun <S : Shape> padLikeAfter4(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3))

fun <S : Shape> padLikeAfter5(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> = padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4))

fun <S : Shape> padLikeAfter6(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    prior5: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> =
    padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5))

fun <S : Shape> padLikeAfter7(
    value: DTensor<*, F32>,
    outTemplate: DTensor<S, F32>,
    prior0: DTensor<*, F32>,
    prior1: DTensor<*, F32>,
    prior2: DTensor<*, F32>,
    prior3: DTensor<*, F32>,
    prior4: DTensor<*, F32>,
    prior5: DTensor<*, F32>,
    prior6: DTensor<*, F32>,
    axis: Int,
): DTensor<S, F32> =
    padWindow(value, outTemplate, axis, listOf(prior0, prior1, prior2, prior3, prior4, prior5, prior6))

/**
 * Phase A2b — the two-operand concat the K2 plugin synthesises with.
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
fun <R : Shape> concatPair(axis: Int, a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> {
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
    val av = a.hostF32()
    val bv = b.hostF32()
    val out = FloatArray(outer * (aRun + bRun))
    var dst = 0
    for (o in 0 until outer) {
        av.copyInto(out, dst, o * aRun, o * aRun + aRun)
        dst += aRun
        bv.copyInto(out, dst, o * bRun, o * bRun + bRun)
        dst += bRun
    }
    val outDims = ad.copyOf().also { it[axis] = ad[axis] + bd[axis] }
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * Phase A2b (DiffKT parity) — concatenate [tensors] along [axis]. The result erases
 * to `DTensor<Shape, F32>`: the concat axis's extent is a runtime SUM of the
 * operands' extents, which no static shape witness can carry — the same convention
 * `slice`, `reshape` and `broadcastTo` follow. Differentiable in `grad {}`: the
 * adjoint gives each operand its window of the upstream via `SLICE_LIKE`, whose
 * bounds are read off the operands' runtime shapes.
 */
fun concat(axis: Int, vararg tensors: DTensor<*, F32>): DTensor<Shape, F32> {
    require(tensors.size >= 2) { "concat: needs at least 2 tensors, got ${tensors.size}" }
    var acc: DTensor<Shape, F32> = concatPair(axis, tensors[0], tensors[1])
    for (i in 2 until tensors.size) acc = concatPair(axis, acc, tensors[i])
    return acc
}

/**
 * Phase A2b (DiffKT parity) — `stack(axis, tensors)`: give each operand a new
 * size-1 axis at [axis], then concatenate along it, so the result has rank
 * `operand.rank + 1`. Pure sugar over [unsqueeze] + [concat], which is exactly how
 * the K2 plugin lowers it.
 */
fun stack(axis: Int, vararg tensors: DTensor<*, F32>): DTensor<Shape, F32> {
    require(tensors.size >= 2) { "stack: needs at least 2 tensors, got ${tensors.size}" }
    val lifted = tensors.map { it.unsqueeze(axis) }
    var acc: DTensor<Shape, F32> = concatPair(axis, lifted[0], lifted[1])
    for (i in 2 until lifted.size) acc = concatPair(axis, acc, lifted[i])
    return acc
}

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
 * §0.4.390 — three inserted unit axes: what `[C] → [1,C,1,1]` needs, i.e. how a
 * per-channel parameter becomes broadcastable against an NCHW tensor. Without it a
 * rank-4 body containing that reshape falls out of synthesis scope.
 */
fun <S : Shape> unsqueezeAxes3(x: DTensor<*, F32>, a0: Int, a1: Int, a2: Int): DTensor<S, F32> =
    DTensor(HostF32Storage(x.hostF32().copyOf()), unsqueezeAxes(x, intArrayOf(a0, a1, a2)), F32)

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
 * §0.4.396 — axis flip (DiffKT `flip`, Phase C3): reverse the element order
 * along each listed axis, all other axes untouched. Shape-preserving, so the
 * receiver's precise shape type survives. Negative axes count from the end;
 * axes must be distinct. Row-major stride walk mirroring the dxir
 * interpreter's REVERSE arm: the output element at multi-index (i_0, …,
 * i_{N-1}) reads the input at `j_k = dims[k] − 1 − i_k` on flipped axes.
 * Its `grad {}` adjoint is the same flip of the upstream (REVERSE is
 * self-adjoint), so no runtime-extent template is involved anywhere.
 */
fun <S : Shape> DTensor<S, F32>.flip(vararg axes: Int): DTensor<S, F32> {
    val r = dims.size
    require(axes.isNotEmpty()) { "flip: needs at least one axis" }
    val norm = BooleanArray(r)
    for (ax in axes) {
        val a = if (ax < 0) ax + r else ax
        require(a in 0 until r) { "flip: axis $ax out of range for rank $r" }
        require(!norm[a]) { "flip: axes ${axes.toList()} must be distinct" }
        norm[a] = true
    }
    val v = hostF32()
    val strides = IntArray(r)
    var st = 1
    for (i in r - 1 downTo 0) { strides[i] = st; st *= dims[i] }
    val out = FloatArray(v.size) { flat ->
        var rem = flat
        var src = 0
        for (k in 0 until r) {
            val coord = rem / strides[k]
            rem -= coord * strides[k]
            src += (if (norm[k]) dims[k] - 1 - coord else coord) * strides[k]
        }
        v[src]
    }
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.396 — fixed-arity synthesis delegates for [flip] (the usual IrVararg
 * reason: synthesis cannot build an `IrVararg`, so each REVERSE node's literal
 * `dimensions` attr rides as positional Int constants — the `transposePerm{N}`
 * precedent exactly).
 */
fun <S : Shape> flipAxes1(x: DTensor<*, F32>, a0: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).flip(a0) as DTensor<S, F32>
}

fun <S : Shape> flipAxes2(x: DTensor<*, F32>, a0: Int, a1: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).flip(a0, a1) as DTensor<S, F32>
}

fun <S : Shape> flipAxes3(x: DTensor<*, F32>, a0: Int, a1: Int, a2: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).flip(a0, a1, a2) as DTensor<S, F32>
}

/**
 * §0.4.371 — rank-increasing broadcast (DiffKT `broadcastTo`/`expand`, Phase
 * A2b). NumPy right-alignment: the receiver's axes map to the TRAILING axes of
 * [newDims]; the new leading axes are replicated. §0.4.373 — in-place size-1
 * stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`) is now supported too: an operand axis
 * of extent 1 replicates across its (equal-position) target extent (full
 * `stablehlo.broadcast_in_dim` semantics, matching the dxir interpreter's
 * BROADCAST arm). A non-1 operand axis must match its target exactly.
 */
fun <S : Shape> DTensor<S, F32>.broadcastTo(vararg newDims: Int): DTensor<Shape, F32> {
    val r = dims.size
    val outRank = newDims.size
    require(outRank >= r) { "broadcastTo: target rank $outRank < operand rank $r (only new leading axes)" }
    for (d in newDims) require(d > 0) { "broadcastTo: dims must be positive, got ${newDims.toList()}" }
    val offset = outRank - r
    for (j in 0 until r) require(dims[j] == newDims[offset + j] || dims[j] == 1) {
        "broadcastTo: operand dim $j = ${dims[j]} must match target ${newDims[offset + j]} or be a size-1 stretch"
    }
    val v = hostF32()
    // General broadcast_in_dim eval: operand axis j maps to output axis
    // (offset + j); a size-1 operand axis contributes index 0 (replicates).
    val inStrides = IntArray(r)
    run { var s = 1; for (i in r - 1 downTo 0) { inStrides[i] = s; s *= dims[i] } }
    val outStrides = IntArray(outRank)
    run { var s = 1; for (i in outRank - 1 downTo 0) { outStrides[i] = s; s *= newDims[i] } }
    var outSize = 1
    for (d in newDims) outSize *= d
    val out = FloatArray(outSize) { flat ->
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
    return DTensor(HostF32Storage(out), newDims.copyOf(), F32)
}

/**
 * §0.4.374 — single-axis slice (DiffKT parity, Phase A2b): take elements
 * `[start, end)` along [axis], all other axes full. `start`/`end`/`axis` are
 * compile-time literals; the result shape equals the receiver's with `axis`'s
 * extent replaced by `end − start`. This is the differentiable `slice`: its
 * `grad {}` adjoint is [padToLike] (the upstream zero-padded back into the
 * sliced window, reading the receiver's extent at runtime). Unit stride only.
 */
fun <S : Shape> DTensor<S, F32>.slice(start: Int, end: Int, axis: Int): DTensor<Shape, F32> {
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
    val v = hostF32()
    val out = FloatArray(outSize) { flat ->
        var rem = flat
        var src = 0
        for (k in 0 until r) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            src += (coord + if (k == ax) start else 0) * inStrides[k]
        }
        v[src]
    }
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * §0.4.428 — `view(range, axis)` (DiffKT parity, the A2 indexing-sugar tail):
 * the contiguous-range view of one axis, all other axes full. Pure sugar over
 * [slice] — `view(a..b, axis) == slice(a, b + 1, axis)` — with DiffKT's
 * inclusive-range spelling. The axis survives (rank is preserved); the
 * single-index overload below drops it. Differentiable in `grad {}` through
 * the same SLICE lowering and PAD_TO adjoint as `slice`.
 */
fun <S : Shape> DTensor<S, F32>.view(range: IntRange, axis: Int): DTensor<Shape, F32> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "view: axis $axis out of range for rank $r" }
    require(!range.isEmpty()) { "view: empty range $range" }
    return slice(range.first, range.last + 1, ax)
}

/**
 * §0.4.428 — `view(index, axis)`: pick one index along [axis] and DROP the
 * axis (DiffKT's indexing view — the result has rank `r − 1`). Sugar over
 * [slice] + [squeeze]. Its `grad {}` adjoint routes the upstream into the
 * indexed window and zeros elsewhere (the slice adjoint after the unit-axis
 * reshape restores the receiver's rank).
 */
fun <S : Shape> DTensor<S, F32>.view(index: Int, axis: Int): DTensor<Shape, F32> {
    val r = dims.size
    val ax = if (axis < 0) axis + r else axis
    require(ax in 0 until r) { "view: axis $axis out of range for rank $r" }
    require(index in 0 until dims[ax]) {
        "view: index $index out of range for axis $ax extent ${dims[ax]}"
    }
    return slice(index, index + 1, ax).squeeze(ax)
}

/**
 * §0.4.428 — `withChange(range, axis, replacement)`: the FUNCTIONAL update
 * (DiffKT parity, A2) — a copy of the receiver with the contiguous window
 * `[range.first, range.last]` along [axis] replaced by [replacement] (same
 * rank; the window's shape). The receiver is untouched. Inside `grad {}` the
 * K2 plugin lowers this as `x + PAD_TO(replacement − slice(x), template = x)`
 * — every piece an existing fully-ruled op, so the adjoint routes the
 * upstream's window to `replacement` and zeros that window in `d_x` with no
 * new IR: the PAD_TO ⇄ SLICE_AT pair (§0.4.399) does the bookkeeping.
 */
fun <S : Shape> DTensor<S, F32>.withChange(
    range: IntRange,
    axis: Int,
    replacement: DTensor<*, F32>,
): DTensor<Shape, F32> {
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
    val out = hostF32().copyOf()
    val rv = replacement.hostF32()
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
    return DTensor(HostF32Storage(out), dims.copyOf(), F32)
}

/**
 * §0.4.428 — `withChange(index, axis, replacement)`: replace the single
 * `view(index, axis)` slice — [replacement] has rank `r − 1` (the view's
 * shape). Sugar over [unsqueeze] + the range overload above.
 */
fun <S : Shape> DTensor<S, F32>.withChange(
    index: Int,
    axis: Int,
    replacement: DTensor<*, F32>,
): DTensor<Shape, F32> {
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
 * §0.4.428 — `meld(tensors…)` (DiffKT parity, A2): flatten every operand
 * row-major and concatenate the flats into one rank-1 tensor of the total
 * element count. The inverse of [split]. Inside `grad {}` this is pure sugar
 * — RESHAPE-to-rank-1 per operand + the binary-CONCAT fold — so each
 * operand's gradient is its window of the upstream reshaped back to its own
 * shape (SLICE_LIKE + the reshape adjoint, both existing rules).
 */
fun meld(vararg tensors: DTensor<*, F32>): DTensor<Shape, F32> {
    require(tensors.isNotEmpty()) { "meld: needs at least 1 tensor" }
    var n = 0
    val flats = tensors.map { it.hostF32() }
    for (v in flats) n += v.size
    val out = FloatArray(n)
    var k = 0
    for (v in flats) {
        v.copyInto(out, k)
        k += v.size
    }
    return DTensor(HostF32Storage(out), intArrayOf(n), F32)
}

/**
 * §0.4.428 — `split(shapes)` (DiffKT parity, A2): cut the receiver's
 * row-major data into consecutive tensors of the given shapes, which must
 * consume every element exactly. The inverse of [meld]. HOST-LEVEL ONLY:
 * a `List<DTensor>`-valued expression has no value model in the `grad {}`
 * lambda lowering, so the differentiable spelling stays `view`/`slice`
 * per piece (recorded as a named deferral in the parity plan).
 */
fun DTensor<*, F32>.split(shapes: List<IntArray>): List<DTensor<Shape, F32>> {
    require(shapes.isNotEmpty()) { "split: needs at least 1 shape" }
    val v = hostF32()
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
        DTensor(HostF32Storage(piece), s.copyOf(), F32)
    }
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

/**
 * §0.4.390 — the adjoint of [unsqueezeAxes3]: `[1,C,1,1] → [C]`, which is what the
 * gradient of a per-channel parameter reshape needs.
 */
fun <S : Shape> squeezeAxes3(x: DTensor<*, F32>, a0: Int, a1: Int, a2: Int): DTensor<S, F32> {
    require(a0 < a1 && a1 < a2) { "squeezeAxes3: axes must be ascending" }
    require(x.dims[a0] == 1 && x.dims[a1] == 1 && x.dims[a2] == 1) {
        "squeezeAxes3: axes must have size 1"
    }
    val out = IntArray(x.dims.size - 3)
    var k = 0
    for (i in x.dims.indices) if (i != a0 && i != a1 && i != a2) out[k++] = x.dims[i]
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

/** §0.4.390 — rank-4 relayout, for the NCHW surfaces (conv/pool/batchNorm). */
fun <S : Shape> reshapeToRank4(
    x: DTensor<*, F32>, d0: Int, d1: Int, d2: Int, d3: Int,
): DTensor<S, F32> =
    DTensor(
        HostF32Storage(reshapeTo(x, intArrayOf(d0, d1, d2, d3))),
        intArrayOf(d0, d1, d2, d3),
        F32,
    )

fun <S : Shape> transposePerm2(x: DTensor<*, F32>, p0: Int, p1: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).transpose(p0, p1) as DTensor<S, F32>
}

fun <S : Shape> transposePerm3(x: DTensor<*, F32>, p0: Int, p1: Int, p2: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).transpose(p0, p1, p2) as DTensor<S, F32>
}

/**
 * §0.4.384 — the rank-4 permutation transpose, for the batch↔feature swap
 * `[0,1,2,3] → [1,0,2,3]` that [io.tlaloc.ir.passes.VjpRegistry.Conv2dRule]'s
 * `dW = conv(Xᵀ, dYᵀ)` trick emits on both sides. The vararg [transpose] above
 * already walks any rank; this delegate exists for the usual IrVararg reason and
 * to keep the permutation's compile-time constants positional.
 */
fun <S : Shape> transposePerm4(x: DTensor<*, F32>, p0: Int, p1: Int, p2: Int, p3: Int): DTensor<S, F32> {
    @Suppress("UNCHECKED_CAST")
    return (x as DTensor<Shape, F32>).transpose(p0, p1, p2, p3) as DTensor<S, F32>
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
