package io.tlaloc.autograd

import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.ir.OpKind

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
