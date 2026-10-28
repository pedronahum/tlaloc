package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MixtureOfExpertsTest {

    private val batch = 2
    private val seq = 5
    private val dModel = 8
    private val hidden = 6
    private val experts = 4
    private val topK = 2

    private fun layer(seed: Long = 5) = MixtureOfExperts(dModel, hidden, experts, topK, RandomKey.fromSeed(seed), initStd = 0.5f)

    private fun uniform(key: RandomKey, n: Int, lo: Float, hi: Float): FloatArray = uniformFloats(key, n).let { u -> FloatArray(n) { lo + (hi - lo) * u[it] } }

    private fun tensor(v: FloatArray, vararg dims: Int): DTensor<Shape, F32> = DTensor(HostF32Storage(v), dims, F32)

    /** `sum(moe(x) · w)` in double, from the equations. */
    private fun reference(x: DoubleArray, p: Map<String, DoubleArray>, w: DoubleArray): Double {
        val router = p.getValue("router.w")
        val gateUp = p.getValue("gateUp")
        val down = p.getValue("down")
        var loss = 0.0
        for (r in 0 until batch * seq) {
            val logits = DoubleArray(experts) { e -> (0 until dModel).sumOf { h -> x[r * dModel + h] * router[h * experts + e] } }
            val mx = logits.max()
            val ex = logits.map { exp(it - mx) }
            val probs = ex.map { it / ex.sum() }
            val top = probs.indices.sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it }).take(topK)
            val total = top.sumOf { probs[it] }
            for (e in top) {
                val gu = DoubleArray(2 * hidden) { c -> (0 until dModel).sumOf { h -> x[r * dModel + h] * gateUp[(e * dModel + h) * 2 * hidden + c] } }
                val a = DoubleArray(hidden) { i -> gu[i] / (1 + exp(-gu[i])) * gu[hidden + i] }
                for (o in 0 until dModel) {
                    val y = (0 until hidden).sumOf { i -> a[i] * down[(e * hidden + i) * dModel + o] }
                    loss += probs[e] / total * y * w[r * dModel + o]
                }
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
        // The experts run as two ragged dots.
        assertEquals(2, step.primal.body.count { it is io.tlaloc.ir.DxirOp && it.op == OpKind.RAGGED_DOT })
        val result = step.run(model, listOf(tensor(x, batch, seq, dModel)))

        val params = model.parameters.associate { it.key to it.tensor.hostF32().let { a -> DoubleArray(a.size) { a[it].toDouble() } } }
        val xd = DoubleArray(x.size) { x[it].toDouble() }
        val wd = DoubleArray(w.size) { w[it].toDouble() }
        val want = reference(xd, params, wd)
        assertTrue(abs(result.loss - want) <= 1e-5 * maxOf(1.0, abs(want)), "loss ${result.loss} against $want")

        val h = 1e-5
        for ((key, base) in params) {
            val got = result.gradients.getValue(key).hostF32()
            val fd = DoubleArray(base.size) { i ->
                val plus = base.copyOf().also { it[i] += h }
                val minus = base.copyOf().also { it[i] -= h }
                (reference(xd, params + (key to plus), wd) - reference(xd, params + (key to minus), wd)) / (2 * h)
            }
            val scale = maxOf(1.0, fd.maxOf { abs(it) })
            for (i in base.indices) {
                assertTrue(abs(got[i] - fd[i]) <= 2e-4 * scale, "d $key[$i]: ${got[i]} against ${fd[i]}")
            }
        }
    }

    @Test
    fun trainingStepsReduceTheLoss() {
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

    @Test
    fun aCausalLmWithExpertsHasTheirParameters() {
        val config = CausalLmConfig(vocabSize = 32, dModel = 16, numLayers = 2, numHeads = 2, ffHidden = 24, numExperts = 4, expertsPerToken = 2)
        val lm = CausalLM.llama(config, RandomKey.fromSeed(1))
        val keys = lm.parameters.map { it.key }
        assertTrue("blocks.0.mlp.router.w" in keys && "blocks.1.mlp.gateUp" in keys && "blocks.1.mlp.down" in keys, "$keys")
        assertEquals(listOf(4, 16, 48), lm.parameters.first { it.key == "blocks.0.mlp.gateUp" }.tensor.dims.toList())
    }
}
