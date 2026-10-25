package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [MixtureOfTopKAttention] against a direct implementation of the paper's equations in
 * Double: the value of `Σ out ⊙ w`, and every parameter's gradient against central
 * differences of that implementation (the landmark and routing choices do not change
 * within the step).
 */
class MixtureOfTopKAttentionTest {

    private val dModel = 8
    private val heads = 2
    private val seq = 12
    private val batch = 2

    private fun layer(m: Int = 3, k: Int = 4, seed: Long = 5): MixtureOfTopKAttention =
        MixtureOfTopKAttention(dModel, heads, landmarks = m, topK = k, key = RandomKey.fromSeed(seed), initStd = 0.5f)

    private fun uniform(key: RandomKey, n: Int, lo: Float, hi: Float): FloatArray =
        uniformFloats(key, n).let { u -> FloatArray(n) { lo + (hi - lo) * u[it] } }

    private fun tensor(v: FloatArray, vararg dims: Int) = DTensor<Shape, F32>(HostF32Storage(v), dims, F32)

    /** `Σ out ⊙ w` for weights [p] (q, k, v, o, each `[in, out]`), straight from the equations. */
    private fun reference(
        layer: MixtureOfTopKAttention,
        x: DoubleArray,
        p: Map<String, DoubleArray>,
        w: DoubleArray,
    ): Double {
        val d = layer.headDim
        val m = layer.landmarks
        val kk = layer.topK
        val hd = heads * d
        val pool = layer.poolingMatrix(seq)
        val scale = 1.0 / kotlin.math.sqrt(d.toDouble())
        fun proj(key: String, b: Int, t: Int, col: Int): Double {
            val wt = p.getValue(key)
            var acc = 0.0
            for (i in 0 until dModel) acc += x[(b * seq + t) * dModel + i] * wt[i * hd + col]
            return acc
        }
        fun softmax(z: DoubleArray): DoubleArray {
            val mx = z.max()
            val e = z.map { exp(it - mx) }
            val s = e.sum()
            return DoubleArray(z.size) { e[it] / s }
        }
        var loss = 0.0
        for (b in 0 until batch) {
            val merged = Array(seq) { DoubleArray(hd) }
            for (h in 0 until heads) {
                val qm = Array(seq) { t -> DoubleArray(d) { proj("q.w", b, t, h * d + it) } }
                val km = Array(seq) { t -> DoubleArray(d) { proj("k.w", b, t, h * d + it) } }
                val vm = Array(seq) { t -> DoubleArray(d) { proj("v.w", b, t, h * d + it) } }
                val lq = Array(m) { i -> DoubleArray(d) { c -> (0 until seq).sumOf { pool[i * seq + it] * qm[it][c] } } }
                val lv = Array(m) { DoubleArray(d) }
                val routed = Array(m) { IntArray(kk) }
                for (i in 0 until m) {
                    val s = DoubleArray(seq) { t -> (0 until d).sumOf { lq[i][it] * km[t][it] } }
                    routed[i] = s.indices.sortedWith(compareByDescending<Int> { s[it] }.thenBy { it }).take(kk).toIntArray()
                    val pr = softmax(DoubleArray(seq) { s[it] * scale })
                    for (c in 0 until d) lv[i][c] = (0 until seq).sumOf { pr[it] * vm[it][c] }
                }
                for (t in 0 until seq) {
                    val r = DoubleArray(m) { i -> (0 until d).sumOf { qm[t][it] * lq[i][it] } }
                    var a = 0
                    for (i in 1 until m) if (r[i] > r[a]) a = i
                    val keys = routed[a]
                    val logits = DoubleArray(m + kk) { j ->
                        if (j < m) r[j] * scale else (0 until d).sumOf { qm[t][it] * km[keys[j - m]][it] } * scale
                    }
                    val pr = softmax(logits)
                    for (c in 0 until d) {
                        var acc = 0.0
                        for (i in 0 until m) acc += pr[i] * lv[i][c]
                        for (j in 0 until kk) acc += pr[m + j] * vm[keys[j]][c]
                        merged[t][h * d + c] = acc
                    }
                }
            }
            val wo = p.getValue("o.w")
            for (t in 0 until seq) for (c in 0 until dModel) {
                var acc = 0.0
                for (i in 0 until hd) acc += merged[t][i] * wo[i * dModel + c]
                loss += acc * w[(b * seq + t) * dModel + c]
            }
        }
        return loss
    }

    @Test
    fun valueAndGradientsMatchTheEquations() {
        val model = layer()
        val keys = RandomKey.fromSeed(11).split(2)
        val x = uniform(keys[0], batch * seq * dModel, -1f, 1f)
        val w = uniform(keys[1], batch * seq * dModel, -1f, 1f)
        val step = capture(model, listOf(tensor(x, batch, seq, dModel))) { out: Tracer<Shape> ->
            (out * out.constant<Shape>(w, out.dims)).sum()
        }
        val result = step.run(model, listOf(tensor(x, batch, seq, dModel)))

        val params = model.parameters.associate { it.key to it.tensor.hostF32().let { a -> DoubleArray(a.size) { a[it].toDouble() } } }
        val xd = DoubleArray(x.size) { x[it].toDouble() }
        val wd = DoubleArray(w.size) { w[it].toDouble() }
        val want = reference(model, xd, params, wd)
        assertTrue(abs(result.loss - want) <= 1e-5 * maxOf(1.0, abs(want)), "loss ${result.loss} against $want")

        val h = 1e-5
        for ((key, base) in params) {
            val got = result.gradients.getValue(key).hostF32()
            val fd = DoubleArray(base.size) { i ->
                val plus = base.copyOf().also { it[i] += h }
                val minus = base.copyOf().also { it[i] -= h }
                (reference(model, xd, params + (key to plus), wd) - reference(model, xd, params + (key to minus), wd)) / (2 * h)
            }
            val scale = maxOf(1.0, fd.maxOf { abs(it) })
            for (i in base.indices) {
                assertTrue(abs(got[i] - fd[i]) <= 2e-4 * scale, "d $key[$i]: ${got[i]} against ${fd[i]}")
            }
        }
    }

    @Test
    fun landmarkWindows() {
        // 12 tokens into 5 landmarks: PyTorch's adaptive windows overlap where 12/5 is not whole.
        val pool = layer(m = 5).poolingMatrix(12)
        val windows = (0 until 5).map { i -> (0 until 12).filter { pool[i * 12 + it] != 0f } }
        assertEquals(listOf(listOf(0, 1, 2), listOf(2, 3, 4), listOf(4, 5, 6, 7), listOf(7, 8, 9), listOf(9, 10, 11)), windows)
        for (i in 0 until 5) assertTrue(abs((0 until 12).sumOf { pool[i * 12 + it].toDouble() } - 1.0) < 1e-6)

        // A 4×4 patch grid into 2×2 landmarks: each averages one 2×2 block.
        val grid = MixtureOfTopKAttention(
            dModel, heads, landmarks = 4, topK = 2, key = RandomKey.fromSeed(1),
            tokenGrid = 4 to 4, landmarkGrid = 2 to 2,
        ).poolingMatrix(16)
        val block = (0 until 16).filter { grid[3 * 16 + it] != 0f }
        assertContentEquals(intArrayOf(10, 11, 14, 15), block.toIntArray())
        assertEquals(0.25f, grid[3 * 16 + 10])
    }

    @Test
    fun trainingStepsReduceTheLoss() {
        // Fit the layer's output to a fixed target with plain gradient steps.
        var model = layer(seed = 9)
        val keys = RandomKey.fromSeed(3).split(2)
        val x = tensor(uniform(keys[0], batch * seq * dModel, -1f, 1f), batch, seq, dModel)
        val target = uniform(keys[1], batch * seq * dModel, -0.5f, 0.5f)
        val step = capture(model, listOf(x)) { out: Tracer<Shape> ->
            val diff = out - out.constant<Shape>(target, out.dims)
            (diff * diff).sum()
        }
        val first = step.run(model, listOf(x)).loss
        var last = first
        repeat(20) {
            val r = step.run(model, listOf(x))
            last = r.loss
            model = model.withParameters(r.gradients.mapValues { (k, g) ->
                val p = model.parameters.first { it.key == k }.tensor.hostF32()
                val gv = g.hostF32()
                tensor(FloatArray(p.size) { p[it] - 0.01f * gv[it] }, *g.dims)
            })
        }
        assertTrue(last < 0.8f * first, "loss $first → $last")
    }
}
