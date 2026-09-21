package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.autograd.sum
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.443 — Phase F7: the GRU certification through the F1 compiler route.
 *
 * Oracles, per the slice:
 *
 * - FORWARD pins against a JUnit-side Float replication of the audited gate
 *   equations. The same-JVM-ops claim, stated precisely: the loss reported by
 *   [valueAndGradients] is the `DxirInterpreter`'s evaluation of the
 *   transform's `includeForward` re-emission of the traced ops, and
 *   [refLossF] mirrors those interpreter arms operation by operation —
 *   MATMUL's f32 accumulation in (k, n) order from a zero accumulator, the
 *   bias ADD after (not fused into) the matmul, SIGMOID as
 *   `(1.0 / (1.0 + exp(-x.toDouble()))).toFloat()` and TANH as
 *   `kotlin.math.tanh(x.toDouble()).toFloat()` (both arms compute in Double
 *   and narrow), the elementwise combine `(1f − u) · n + u · h` in f32, and
 *   the final SUM accumulated in flat row-major f32 order. Agreement is
 *   therefore expected to the bit; the pins assert the slice's 1e-5 contract.
 *
 * - GRADIENT pins against central finite differences (step 1e-5) over an
 *   INDEPENDENT double-precision replication of the same equations — no code
 *   shared with `:nn` or the Float replication's op-order mirroring. FD
 *   truncation error in Double is O(1e-10) here; the 1e-5 tolerance absorbs
 *   the f32 rounding of the transform's analytic gradients.
 *
 * - The BPTT sanity is definitional on this route: the captured graph IS the
 *   unrolled sequence, so reverse-mode through it is backpropagation through
 *   time — pinned by the last-step loss reaching the step-0 input slice.
 *
 * The trace-vs-hand-built-DXIR spot check is NOT taken here (the slice said
 * "if cheap"; the unrolled single step is ~20 hand-built ops and every
 * constituent spelling — CONCAT, SLICE, RESHAPE, MATMUL, BROADCAST, the
 * elementwise family — already carries its own F2/F5/F6 trace-vs-DXIR pin
 * through the same transform).
 */
class GruTest {

    // Quarter-integer weights, [numInputs + numHidden, numHidden] = [4, 2] row-major.
    private val wU = floatArrayOf(0.25f, -0.5f, 0.5f, 0.25f, -0.25f, 0.5f, 0.25f, -0.25f)
    private val bU = floatArrayOf(0.25f, -0.25f)
    private val wR = floatArrayOf(0.5f, 0.25f, -0.25f, -0.5f, 0.25f, 0.25f, 0.5f, -0.25f)
    private val bR = floatArrayOf(-0.25f, 0.5f)
    private val wN = floatArrayOf(0.25f, 0.5f, 0.5f, -0.25f, -0.5f, 0.25f, 0.25f, 0.5f)
    private val bN = floatArrayOf(0.25f, 0.25f)

    // The linearBeforeReset candidate pair, [2, 2] each.
    private val wXn = floatArrayOf(0.5f, -0.25f, 0.25f, 0.5f)
    private val bXn = floatArrayOf(0.25f, -0.5f)
    private val wHn = floatArrayOf(-0.25f, 0.5f, 0.5f, 0.25f)
    private val bHn = floatArrayOf(0.5f, 0.25f)

    // Input [1, seq, 2] prefixes: seq = 1, 2, 3 all read off the same stream.
    private val xSeq = floatArrayOf(0.5f, -0.25f, 1f, 0.25f, -0.5f, 0.75f)

    private fun x(seq: Int) =
        Tensors.f32Tensor3<Sym, Sym, Sym>(1, seq, 2, xSeq.copyOfRange(0, seq * 2))

    private fun denseOf(nIn: Int, w: FloatArray, b: FloatArray, act: Activation) =
        Dense(Tensors.f32Matrix<Sym, Sym>(nIn, 2, w), Tensors.f32Vector<Sym>(b), act)

    private fun afterResetCell(acc: GRU.AccType = GRU.AccType.Fold) = GRU(
        denseOf(4, wU, bU, Activation.Sigmoid),
        denseOf(4, wR, bR, Activation.Sigmoid),
        denseOf(4, wN, bN, Activation.Tanh),
        acc,
    )

    private fun beforeResetCell() = GRU(
        denseOf(4, wU, bU, Activation.Sigmoid),
        denseOf(4, wR, bR, Activation.Sigmoid),
        denseOf(2, wXn, bXn, Activation.Identity),
        denseOf(2, wHn, bHn, Activation.Identity),
    )

    // ------------------------------------------------------------------
    // The double-precision reference (the FD substrate).
    // ------------------------------------------------------------------

    private fun FloatArray.d() = DoubleArray(size) { this[it].toDouble() }

    private fun paramsD(seq: Int, beforeReset: Boolean): Map<String, DoubleArray> = buildMap {
        put("x", xSeq.copyOfRange(0, seq * 2).d())
        put("xh2u.w", wU.d()); put("xh2u.b", bU.d())
        put("xh2r.w", wR.d()); put("xh2r.b", bR.d())
        if (!beforeReset) {
            put("xh2n.w", wN.d()); put("xh2n.b", bN.d())
        } else {
            put("x2n.w", wXn.d()); put("x2n.b", bXn.d())
            put("h2n.w", wHn.d()); put("h2n.b", bHn.d())
        }
    }

    private fun affineD(v: DoubleArray, w: DoubleArray, b: DoubleArray, out: DoubleArray) {
        val nOut = out.size
        for (j in 0 until nOut) out[j] = b[j]
        for (i in v.indices) for (j in 0 until nOut) out[j] += v[i] * w[i * nOut + j]
    }

    private fun refLossD(
        p: Map<String, DoubleArray>,
        seq: Int,
        beforeReset: Boolean,
        accMap: Boolean,
    ): Double {
        val nIn = 2
        val nH = 2
        val x = p.getValue("x")
        var h = DoubleArray(nH)
        var acc = 0.0
        for (t in 0 until seq) {
            val xt = DoubleArray(nIn) { x[t * nIn + it] }
            val xh = xt + h
            val u = DoubleArray(nH)
            val r = DoubleArray(nH)
            affineD(xh, p.getValue("xh2u.w"), p.getValue("xh2u.b"), u)
            affineD(xh, p.getValue("xh2r.w"), p.getValue("xh2r.b"), r)
            for (j in 0 until nH) {
                u[j] = 1.0 / (1.0 + exp(-u[j]))
                r[j] = 1.0 / (1.0 + exp(-r[j]))
            }
            val n = DoubleArray(nH)
            if (!beforeReset) {
                val rh = DoubleArray(nH) { r[it] * h[it] }
                affineD(xt + rh, p.getValue("xh2n.w"), p.getValue("xh2n.b"), n)
                for (j in 0 until nH) n[j] = kotlin.math.tanh(n[j])
            } else {
                val a = DoubleArray(nH)
                val bp = DoubleArray(nH)
                affineD(xt, p.getValue("x2n.w"), p.getValue("x2n.b"), a)
                affineD(h, p.getValue("h2n.w"), p.getValue("h2n.b"), bp)
                for (j in 0 until nH) n[j] = kotlin.math.tanh(a[j] + r[j] * bp[j])
            }
            h = DoubleArray(nH) { (1.0 - u[it]) * n[it] + u[it] * h[it] }
            if (accMap) for (v in h) acc += v
        }
        return if (accMap) acc else h.sum()
    }

    // ------------------------------------------------------------------
    // The Float replication in the interpreter's own op order (see class KDoc).
    // ------------------------------------------------------------------

    private fun affineF(v: FloatArray, w: FloatArray, b: FloatArray): FloatArray {
        val nOut = b.size
        val out = FloatArray(nOut)
        for (i in v.indices) {
            val vi = v[i]
            if (vi == 0f) continue
            for (j in 0 until nOut) out[j] += vi * w[i * nOut + j]
        }
        for (j in 0 until nOut) out[j] = out[j] + b[j]
        return out
    }

    private fun sigF(z: Float): Float = (1.0 / (1.0 + exp(-z.toDouble()))).toFloat()

    private fun tanhF(z: Float): Float = kotlin.math.tanh(z.toDouble()).toFloat()

    private fun refLossF(seq: Int, beforeReset: Boolean, accMap: Boolean): Float {
        val nIn = 2
        val nH = 2
        var h = FloatArray(nH)
        val outs = ArrayList<FloatArray>(seq)
        for (t in 0 until seq) {
            val xt = FloatArray(nIn) { xSeq[t * nIn + it] }
            val xh = xt + h
            val u = affineF(xh, wU, bU).also { for (j in it.indices) it[j] = sigF(it[j]) }
            val r = affineF(xh, wR, bR).also { for (j in it.indices) it[j] = sigF(it[j]) }
            val n: FloatArray
            if (!beforeReset) {
                val rh = FloatArray(nH) { r[it] * h[it] }
                n = affineF(xt + rh, wN, bN).also { for (j in it.indices) it[j] = tanhF(it[j]) }
            } else {
                val a = affineF(xt, wXn, bXn)
                val bp = affineF(h, wHn, bHn)
                n = FloatArray(nH) { tanhF(a[it] + r[it] * bp[it]) }
            }
            h = FloatArray(nH) { (1f - u[it]) * n[it] + u[it] * h[it] }
            outs += h
        }
        var loss = 0f
        if (accMap) {
            for (o in outs) for (v in o) loss += v
        } else {
            for (v in h) loss += v
        }
        return loss
    }

    // ------------------------------------------------------------------
    // FD machinery.
    // ------------------------------------------------------------------

    private fun fdGrad(
        p: Map<String, DoubleArray>,
        key: String,
        idx: Int,
        eval: (Map<String, DoubleArray>) -> Double,
    ): Double {
        val eps = 1e-5
        fun at(delta: Double): Double {
            val q = p.mapValues { (k, v) -> if (k == key) v.copyOf().also { it[idx] += delta } else v }
            return eval(q)
        }
        return (at(eps) - at(-eps)) / (2 * eps)
    }

    private fun assertGradsMatchFd(
        result: StepResult,
        p: Map<String, DoubleArray>,
        eval: (Map<String, DoubleArray>) -> Double,
    ) {
        for ((key, g) in result.gradients) {
            val arr = g.hostF32()
            assertEquals(p.getValue(key).size, arr.size, "gradient size for $key")
            for (i in arr.indices) {
                assertEquals(fdGrad(p, key, i, eval).toFloat(), arr[i], 1e-5f, "grad $key[$i]")
            }
        }
        assertEquals(1, result.inputGradients.size)
        val gx = result.inputGradients[0].hostF32()
        assertEquals(p.getValue("x").size, gx.size, "input gradient size")
        for (i in gx.indices) {
            assertEquals(fdGrad(p, "x", i, eval).toFloat(), gx[i], 1e-5f, "input grad [$i]")
        }
    }

    // ------------------------------------------------------------------
    // The certs.
    // ------------------------------------------------------------------

    @Test
    fun singleStepForwardMatchesTheSameOrderFloatReplication() {
        val result = valueAndGradients(afterResetCell(), listOf(x(1))) { y -> y.sum() }
        assertEquals(refLossF(1, beforeReset = false, accMap = false), result.loss, 1e-5f)
        assertEquals(
            listOf("xh2u.w", "xh2u.b", "xh2r.w", "xh2r.b", "xh2n.w", "xh2n.b"),
            result.gradients.keys.toList(),
        )
    }

    @Test
    fun singleStepGradientsMatchCentralDifferencesOnTheDoubleReplication() {
        val result = valueAndGradients(afterResetCell(), listOf(x(1))) { y -> y.sum() }
        val p = paramsD(1, beforeReset = false)
        assertGradsMatchFd(result, p) { refLossD(it, 1, beforeReset = false, accMap = false) }

        // The recorded single-step structural fact: with h₀ = 0 the reset gate
        // only reaches the loss through r·h = r·0, so its gradients are EXACT
        // (±)0 — the transform multiplies every reset contribution by the zeros
        // constant. The 3-step cert below pins the gate live once h ≠ 0.
        for (key in listOf("xh2r.w", "xh2r.b")) {
            for (v in result.gradients[key]!!.hostF32()) {
                assertTrue(v == 0f, "reset-gate grad $key must be exactly zero at step 1, got $v")
            }
        }
    }

    @Test
    fun threeStepSequenceGradientsMatchCentralDifferences() {
        val result = valueAndGradients(afterResetCell(), listOf(x(3))) { y -> y.sum() }
        assertEquals(refLossF(3, beforeReset = false, accMap = false), result.loss, 1e-5f)
        val p = paramsD(3, beforeReset = false)
        assertGradsMatchFd(result, p) { refLossD(it, 3, beforeReset = false, accMap = false) }

        // The reset gate is live once the state is nonzero.
        assertTrue(
            result.gradients["xh2r.w"]!!.hostF32().any { abs(it) > 1e-4f },
            "reset-gate weights must carry gradient over a 3-step sequence",
        )
    }

    @Test
    fun bpttSanityLastStepLossReachesTheStepZeroInput() {
        // AccType.Fold: the loss reads ONLY h₃, yet the unrolled graph carries
        // the chain back through h₂ and h₁ to x₀ — reverse-mode on the captured
        // unroll IS backpropagation through time.
        val result = valueAndGradients(afterResetCell(), listOf(x(3))) { y -> y.sum() }
        assertContentEquals(intArrayOf(1, 3, 2), result.inputGradients[0].dims)
        val gx = result.inputGradients[0].hostF32()
        assertTrue(
            abs(gx[0]) > 1e-4f || abs(gx[1]) > 1e-4f,
            "last-step loss must yield a nonzero gradient w.r.t. the step-0 input " +
                "(got [${gx[0]}, ${gx[1]}])",
        )
    }

    @Test
    fun accMapForwardAndGradientsOverThreeSteps() {
        val result = valueAndGradients(afterResetCell(GRU.AccType.AccMap), listOf(x(3))) { y -> y.sum() }
        assertEquals(refLossF(3, beforeReset = false, accMap = true), result.loss, 1e-5f)
        val p = paramsD(3, beforeReset = false)
        assertGradsMatchFd(result, p) { refLossD(it, 3, beforeReset = false, accMap = true) }
    }

    @Test
    fun linearBeforeResetTwoStepForwardAndGradients() {
        val model = beforeResetCell()
        assertTrue(model.linearBeforeReset)
        assertEquals(
            listOf("xh2u.w", "xh2u.b", "xh2r.w", "xh2r.b", "x2n.w", "x2n.b", "h2n.w", "h2n.b"),
            model.parameters.map { it.key },
        )
        // Two steps so h ≠ 0 exercises both r and the h2n path.
        val result = valueAndGradients(model, listOf(x(2))) { y -> y.sum() }
        assertEquals(refLossF(2, beforeReset = true, accMap = false), result.loss, 1e-5f)
        val p = paramsD(2, beforeReset = true)
        assertGradsMatchFd(result, p) { refLossD(it, 2, beforeReset = true, accMap = false) }
    }

    @Test
    fun companionDrawsBitExactUnderTheGatewiseKeyDiscipline() {
        val key = RandomKey.fromSeed(11L)
        val g = GRU(2, 3, key)
        assertEquals(2, g.numInputs)
        assertEquals(3, g.numHidden)
        assertEquals(GRU.AccType.Fold, g.acc)
        val keys = key.split(3)
        assertContentEquals(
            Dense(5, 3, keys[0], activation = Activation.Sigmoid).w.hostF32(),
            g.xh2u.w.hostF32(),
        )
        assertContentEquals(
            Dense(5, 3, keys[1], activation = Activation.Sigmoid).b!!.hostF32(),
            g.xh2r.b!!.hostF32(),
        )
        assertContentEquals(
            Dense(5, 3, keys[2], activation = Activation.Tanh).w.hostF32(),
            g.xh2n!!.w.hostF32(),
        )

        val gb = GRU(2, 3, key, linearBeforeReset = true)
        assertTrue(gb.linearBeforeReset)
        val keys4 = key.split(4)
        assertContentEquals(
            Dense(5, 3, keys4[0], activation = Activation.Sigmoid).w.hostF32(),
            gb.xh2u.w.hostF32(),
        )
        assertContentEquals(Dense(2, 3, keys4[2]).w.hostF32(), gb.x2n!!.w.hostF32())
        assertContentEquals(Dense(3, 3, keys4[3]).w.hostF32(), gb.h2n!!.w.hostF32())
    }

    @Test
    fun withParametersIsFunctionalAndTheGuardsRefuse() {
        val model = afterResetCell()
        val newW = Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8) { 0.125f * (it + 1) })
        val updated = model.withParameters(mapOf("xh2u.w" to newW))
        assertContentEquals(newW.hostF32(), updated.xh2u.w.hostF32())
        assertContentEquals(wU, model.xh2u.w.hostF32()) // original untouched
        assertContentEquals(wR, updated.xh2r.w.hostF32()) // untargeted gates pass through
        assertContentEquals(bU, updated.xh2u.b!!.hostF32()) // untargeted param inside the gate too

        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("nope.w" to newW)) }
        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("x2n.w" to newW)) }
        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("xh2u.q" to newW)) }
        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("w" to newW)) }

        // The activation parity guard: DiffKT's gates are σ/σ/tanh.
        assertFailsWith<IllegalArgumentException> {
            GRU(
                denseOf(4, wU, bU, Activation.Identity),
                denseOf(4, wR, bR, Activation.Sigmoid),
                denseOf(4, wN, bN, Activation.Tanh),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            GRU(
                denseOf(4, wU, bU, Activation.Sigmoid),
                denseOf(4, wR, bR, Activation.Sigmoid),
                denseOf(4, wN, bN, Activation.Sigmoid),
            )
        }
    }
}
