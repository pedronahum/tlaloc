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
