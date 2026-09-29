/**
 * Tlaloc per-example gradients — `vmap { grad { } }` on a small classifier.
 *
 * The model: a two-layer network, `logits = tanh(x · W1) · W2`, with a cross-entropy
 * loss against a one-hot label. An example is a feature row `x` (1 x 4) and a one-hot
 * label row `y` (1 x 3); a batch stacks B of each along a leading `Batch` axis.
 *
 * [1] Per-example gradients. `vmap2(batchAxis(Batch), Batched, Batched) { x, y ->
 *     grad { w -> loss(w, W2, x, y) }(W1) }` returns one gradient of W1 per example,
 *     `[B, 4, 8]`, and the same for W2, `[B, 8, 3]`. The Tlaloc K2 plugin writes the
 *     batched gradient code at compile time: the gradient of one example is derived,
 *     then batched along the new leading axis. From these come the per-example gradient
 *     norms, what DP-SGD clips.
 * [2] The batch gradient. `grad { w -> vmap2(...) { x, y -> loss(w, W2, x, y) }(xs, ys).mean() }`
 *     differentiates the batched loss.
 *
 * NOTHING HERE IS TAKEN ON TRUST. [1] is checked, example by example, against
 * `referenceGradients`: the same network's backpropagation written out in plain Kotlin
 * `Double`, itself checked against a central finite difference of `referenceLoss`. [2] is
 * checked against the mean of the per-example gradients.
 */
@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

import io.tlaloc.autograd.Batched
import io.tlaloc.autograd.batchAxis
import io.tlaloc.autograd.grad
import io.tlaloc.autograd.vmap2
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.crossEntropyLoss
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.mean
import io.tlaloc.core.ops.tanh
import io.tlaloc.core.ops.toFloat
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

object Feature : IndexName { override val name = "feature" }
object Hidden : IndexName { override val name = "hidden" }
object Class : IndexName { override val name = "class" }

typealias X = DTensor<Rank2<Sym, Named<Feature, Sym>>, F32>
typealias Y = DTensor<Rank2<Sym, Named<Class, Sym>>, F32>
typealias W1 = DTensor<Rank2<Named<Feature, Sym>, Named<Hidden, Sym>>, F32>
typealias W2 = DTensor<Rank2<Named<Hidden, Sym>, Named<Class, Sym>>, F32>
typealias Xs = DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feature, Sym>>, F32>
typealias Ys = DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Class, Sym>>, F32>

const val FEATURES = 4
const val HIDDEN = 8
const val CLASSES = 3
const val B = 16

fun <S : Shape> tensor(data: FloatArray, vararg dims: Int) = DTensor<S, F32>(HostF32Storage(data), dims, F32)

// ---------------------------------------------------------------------------
// The batched gradients. Each lambda is one example's computation; `vmap2` adds the
// batch axis in front of the two batched arguments and of the result, so the result
// types below are checked by the Kotlin compiler.
// ---------------------------------------------------------------------------

fun perExampleGradients(
    xs: Xs,
    ys: Ys,
    w1: W1,
    w2: W2,
): Pair<DTensor<Rank3<Named<Batch, Sym>, Named<Feature, Sym>, Named<Hidden, Sym>>, F32>,
    DTensor<Rank3<Named<Batch, Sym>, Named<Hidden, Sym>, Named<Class, Sym>>, F32>> {
    val g1 = vmap2(batchAxis(Batch), Batched, Batched) { x: X, y: Y ->
        grad { w: W1 -> crossEntropyLoss((x matmul w).tanh() matmul w2, y).toFloat() }(w1)
    }
    val g2 = vmap2(batchAxis(Batch), Batched, Batched) { x: X, y: Y ->
        grad { w: W2 -> crossEntropyLoss((x matmul w1).tanh() matmul w, y).toFloat() }(w2)
    }
    return g1(xs, ys) to g2(xs, ys)
}

fun batchGradient(xs: Xs, ys: Ys, w1: W1, w2: W2): W1 {
    val g = grad { w: W1 ->
        vmap2(batchAxis(Batch), Batched, Batched) { x: X, y: Y ->
            crossEntropyLoss((x matmul w).tanh() matmul w2, y)
        }(xs, ys).mean().toFloat()
    }
    return g(w1)
}

// ---------------------------------------------------------------------------
// The same network in plain Kotlin Double: the loss, and its gradients written out by
// hand (backpropagation), for one example.
// ---------------------------------------------------------------------------

class Reference(val loss: Double, val dW1: DoubleArray, val dW2: DoubleArray)

fun referenceLoss(w1: DoubleArray, w2: DoubleArray, x: FloatArray, y: FloatArray): Double =
    referenceGradients(w1, w2, x, y).loss

fun referenceGradients(w1: DoubleArray, w2: DoubleArray, x: FloatArray, y: FloatArray): Reference {
    val h = DoubleArray(HIDDEN) { j ->
        var s = 0.0
        for (i in 0 until FEATURES) s += x[i] * w1[i * HIDDEN + j]
        kotlin.math.tanh(s)
    }
    val logits = DoubleArray(CLASSES) { c ->
        var s = 0.0
        for (j in 0 until HIDDEN) s += h[j] * w2[j * CLASSES + c]
        s
    }
    val m = logits.max()
    val z = logits.sumOf { exp(it - m) }
    val p = DoubleArray(CLASSES) { exp(logits[it] - m) / z }
    var loss = 0.0
    for (c in 0 until CLASSES) loss -= y[c] * (logits[c] - m - ln(z))
    // dL/dlogits = p·Σy − y (Σy = 1 for a one-hot label)
    val ySum = y.sum()
    val dLogits = DoubleArray(CLASSES) { p[it] * ySum - y[it] }
    val dW2 = DoubleArray(HIDDEN * CLASSES) { k -> h[k / CLASSES] * dLogits[k % CLASSES] }
    val dPre = DoubleArray(HIDDEN) { j ->
        var s = 0.0
        for (c in 0 until CLASSES) s += dLogits[c] * w2[j * CLASSES + c]
        s * (1 - h[j] * h[j])
    }
    val dW1 = DoubleArray(FEATURES * HIDDEN) { k -> x[k / HIDDEN] * dPre[k % HIDDEN] }
    return Reference(loss, dW1, dW2)
}

fun main() {
    val rnd = Random(3)
    val w1Data = FloatArray(FEATURES * HIDDEN) { rnd.nextFloat() - 0.5f }
    val w2Data = FloatArray(HIDDEN * CLASSES) { rnd.nextFloat() - 0.5f }
    val xData = FloatArray(B * FEATURES) { rnd.nextFloat() * 2f - 1f }
    val yData = FloatArray(B * CLASSES)
    for (b in 0 until B) yData[b * CLASSES + rnd.nextInt(CLASSES)] = 1f

    val w1: W1 = tensor(w1Data, FEATURES, HIDDEN)
    val w2: W2 = tensor(w2Data, HIDDEN, CLASSES)
    val xs: Xs = tensor(xData, B, 1, FEATURES)
    val ys: Ys = tensor(yData, B, 1, CLASSES)
    val w1d = DoubleArray(w1Data.size) { w1Data[it].toDouble() }
    val w2d = DoubleArray(w2Data.size) { w2Data[it].toDouble() }
    fun x(b: Int) = xData.copyOfRange(b * FEATURES, (b + 1) * FEATURES)
    fun y(b: Int) = yData.copyOfRange(b * CLASSES, (b + 1) * CLASSES)

    // The reference itself, against a central finite difference of its loss (example 5).
    val k = 2 * HIDDEN + 3
    val h = 1e-6
    val fd = (
        referenceLoss(w1d.copyOf().also { it[k] += h }, w2d, x(5), y(5)) -
            referenceLoss(w1d.copyOf().also { it[k] -= h }, w2d, x(5), y(5))
        ) / (2 * h)
    val ref5 = referenceGradients(w1d, w2d, x(5), y(5)).dW1[k]
    check(abs(ref5 - fd) <= 1e-7) { "the reference gradient disagrees with its finite difference" }

    // [1] per-example gradients, against the reference example by example
    val (g1, g2) = perExampleGradients(xs, ys, w1, w2)
    val v1 = g1.hostF32()
    val v2 = g2.hostF32()
    val n1 = FEATURES * HIDDEN
    val n2 = HIDDEN * CLASSES
    var worst = 0.0
    var scale = 0.0
    val norms = DoubleArray(B)
    for (b in 0 until B) {
        val ref = referenceGradients(w1d, w2d, x(b), y(b))
        var sq = 0.0
        for (i in 0 until n1) {
            worst = maxOf(worst, abs(v1[b * n1 + i] - ref.dW1[i]))
            scale = maxOf(scale, abs(ref.dW1[i]))
            sq += v1[b * n1 + i].toDouble() * v1[b * n1 + i]
        }
        for (i in 0 until n2) {
            worst = maxOf(worst, abs(v2[b * n2 + i] - ref.dW2[i]))
            scale = maxOf(scale, abs(ref.dW2[i]))
            sq += v2[b * n2 + i].toDouble() * v2[b * n2 + i]
        }
        norms[b] = sqrt(sq)
    }
    println("[1] per-example gradients: W1 ${g1.dims.toList()}, W2 ${g2.dims.toList()}")
    println("    per-example gradient norms:")
    for (b in 0 until B step 4) {
        println("    " + (b until b + 4).joinToString("   ") { "#%-2d %.5f".format(it, norms[it]) })
    }
    println(
        "    largest difference from the Double reference: %.2e of the largest entry".format(worst / scale),
    )
    println("    example 5, dL/dW1[2][3]: compiled %.7f, reference %.7f, finite difference %.7f".format(v1[5 * n1 + k], ref5, fd))
    check(worst / scale <= 1e-5) { "per-example gradients disagree with the reference" }

    // [2] the gradient of the mean loss, against the mean of the per-example gradients
    val gb = batchGradient(xs, ys, w1, w2).hostF32()
    var worstMean = 0.0
    for (i in 0 until n1) {
        var mean = 0.0
        for (b in 0 until B) mean += v1[b * n1 + i]
        worstMean = maxOf(worstMean, abs(gb[i] - mean / B))
    }
    println("[2] gradient of the mean loss over the batch: W1 ${FEATURES} x ${HIDDEN}")
    println("    largest difference from the mean of the per-example gradients: %.2e".format(worstMean))
    check(worstMean <= 1e-6) { "the batch gradient disagrees with the mean of the per-example gradients" }
}
