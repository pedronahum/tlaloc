package io.tlaloc.nn

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.cast
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.458 (G1d) — the mixed-precision training certification, HOST lane.
 * The convention on trial is [Precision]'s: f32 master weights, bf16 compute
 * between injected boundary casts, f32 loss and gradients, no loss scaling.
 * The GPU half of this slice is PjrtMixedPrecisionSmokeTest (runtime-pjrt).
 *
 * Oracle story per test, summarized:
 *  - STRUCTURE: the mixed primal's params stay F32-typed (master weights),
 *    every f32 leaf gets exactly one injected CAST→bf16, exactly one
 *    CAST→f32 closes the bf16 region at the model output — counted on the
 *    captured graph itself, not inferred.
 *  - SNAP LANE (bit-exact): a batch-1 MLP with values chosen so every f32
 *    add/mul between snap points is EXACT (snapped bf16 operands have
 *    ≤8-bit mantissas; their products need ≤16 bits and the 2-term dot sums
 *    ≤ ~20 bits, all inside f32's 24) — so the interpreter's internal
 *    accumulation order/width cannot matter, and the loss must BIT-MATCH a
 *    reference computed in this test with nothing but the §0.4.455 snap
 *    helpers and plain f32 arithmetic at the stated snap points. That pins
 *    the whole convention (which values snap, and where) end to end.
 *  - EXACT LANE (bit-exact): with every input/weight a small power of two,
 *    every intermediate is bf16-exact, all snaps are no-ops, and the mixed
 *    capture must agree with the f32 capture BIT-FOR-BIT — loss AND every
 *    gradient. Straight-through means the casts contribute exactly nothing
 *    when nothing rounds.
 *  - ENVELOPE (derived, not guessed): each RNE snap perturbs a value by at
 *    most one relative half-ulp, 2⁻⁸ (§0.4.455: 8-bit significand). A
 *    forward path crosses R = 8 snap points (x, w1, dot1, b1, add1 — the
 *    relu re-snap is a no-op on an already-snapped value — w2, dot2, add2),
 *    so |Δy_i| ≤ ((1+2⁻⁸)⁸ − 1)·A_i with A_i the triangle-inequality
 *    magnitude bound Σ|w2|·(Σ|x||w1| + |b1|) + |b2| computed from the f32
 *    parameters. The loss difference then telescopes through the f32 MSE:
 *    |ΔL| ≤ (1/n)Σ Δy_i·(2(A_i + |t_i|) + Δy_i). Every quantity in the
 *    bound is computed in the test; nothing is a tuned constant.
 *  - TRAINING: 10 mixed steps decrease the loss strictly on the fixed
 *    threefry task, and the whole run is bit-deterministic (two runs,
 *    identical raw-bit loss curves).
 *  - REFUSALS: bf16/f32 operand mixing on the tape and out-of-scope casts
 *    refuse BY NAME; `gradSource()` on a mixed capture refuses by name too
 *    (the §0.4.456 ratified story: the readable reverse of a bf16 program
 *    is the f32 graph between the casts — no silent wrong-precision source).
 */
class MixedPrecisionTrainingTest {

    private fun snap(v: Float): Float = bf16BitsToFloat(floatToBf16Bits(v))

    // ------------------------------------------------------------------
    // The tiny hand-controlled MLP: Dense(2,2,relu) → Dense(2,1).
    // ------------------------------------------------------------------

    private fun tinyModel(
        w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray,
    ): Sequential = Sequential(
        Dense(
            Tensors.f32Matrix<Sym, Sym>(2, 2, w1),
            Tensors.f32Vector<Sym>(b1),
            Activation.Relu,
        ),
        Dense(
            Tensors.f32Matrix<Sym, Sym>(2, 1, w2),
            Tensors.f32Vector<Sym>(b2),
        ),
    )

    private fun mseLoss(target: FloatArray, rows: Int): (io.tlaloc.autograd.Tracer<io.tlaloc.core.Shape>) -> io.tlaloc.autograd.Tracer<*> = { y ->
        val t = y.constant<io.tlaloc.core.Shape>(target, intArrayOf(rows, 1))
        val diff = y - t
        (diff * diff).mean()
    }

    // ------------------------------------------------------------------
    // (1) Structure: master weights stay f32, casts are injected where the
    // convention says and nowhere else, gradients come back f32-shaped.
    // ------------------------------------------------------------------

    @Test
    fun mixedCaptureInjectsBoundaryCastsAndKeepsMasterWeightsF32() {
        val w1 = floatArrayOf(0.5f, 2f, 1f, 0.25f)
        val model = tinyModel(w1, floatArrayOf(0.5f, -1f), floatArrayOf(2f, 0.5f), floatArrayOf(0.25f))
        val x = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 0.5f, -4f))
        val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mseLoss(floatArrayOf(8f, 0f), 2))

        // Master weights: every primal param is F32-typed — the model's own
        // tensors, never bf16-narrowed at the parameter level.
        for (p in step.primal.params) {
            assertEquals(F32, p.type.dtype, "primal param ${p.name} must stay F32 (master weights)")
        }
        // Injected casts, counted on the graph: one →bf16 per f32 leaf
        // (1 input + 4 parameters), exactly one bf16→f32 closing the region
        // at the model output (the loss then runs f32).
        val ops = step.primal.body.filterIsInstance<DxirOp>()
        val toBf16 = ops.count { it.op == OpKind.CAST && it.type.dtype == BF16 }
        val toF32 = ops.count {
            it.op == OpKind.CAST && it.type.dtype == F32 && it.operands[0].type.dtype == BF16
        }
        assertEquals(5, toBf16, "expected one injected f32→bf16 cast per f32 leaf (1 input + 4 params)")
        assertEquals(1, toF32, "expected exactly one bf16→f32 cast at the model-output boundary")
        // The compute region really is bf16: every non-cast op between the
        // boundaries carries the bf16 type (loss ops after the boundary are
        // f32; here that is SUB/MUL/MEAN).
        assertTrue(
            ops.any { it.op == OpKind.MATMUL && it.type.dtype == BF16 },
            "the matmuls must record bf16-typed in the mixed trace",
        )
        assertTrue(
            ops.filter { it.op == OpKind.MEAN }.all { it.type.dtype == F32 },
            "the loss reduction must stay f32 (the cast-the-loss shape was rejected)",
        )

        // Gradients: f32 at the parameter level, param-shaped, and not the
        // silent zero. StepResult's gradients are DTensor<*, F32> by type;
        // what needs pinning is shape agreement and non-vacuousness.
        val res = step.run(model, listOf(x))
        for (p in model.parameters) {
            val g = res.gradients[p.key] ?: error("missing gradient for ${p.key}")
            assertTrue(
                g.dims.contentEquals(p.tensor.dims),
                "gradient for ${p.key}: dims ${g.dims.toList()} != param dims ${p.tensor.dims.toList()}",
            )
        }
        assertTrue(
            res.gradients.values.any { g -> g.hostF32().any { it != 0f } },
            "mixed-precision gradients came back all-zero — the silent-zero failure mode",
        )
    }

    // ------------------------------------------------------------------
    // (2) The snap lane: bit-exact against the convention, restated in this
    // test with nothing but the snap helpers.
    // ------------------------------------------------------------------

    @Test
    fun mixedForwardBitMatchesSnapReferenceOnBatchOne() {
        // Chosen so snaps BITE (1.005, 0.3, 1.2, 0.7, −0.35, 1.1, 0.9 are not
        // bf16-representable) while every f32 add/mul between snap points is
        // exact (see the class KDoc's exactness argument).
        val xv = floatArrayOf(1.005f, -0.4f)
        val w1 = floatArrayOf(0.3f, 1.2f, 0.7f, -0.35f)
        val b1 = floatArrayOf(0.1f, -0.2f)
        val w2 = floatArrayOf(1.1f, 0.9f)
        val b2 = floatArrayOf(0.05f)
        val target = floatArrayOf(0.5f)

        val model = tinyModel(w1, b1, w2, b2)
        val x = Tensors.f32Matrix<Sym, Sym>(1, 2, xv)
        val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mseLoss(target, 1))
        val got = step.run(model, listOf(x)).loss

        // The reference: the [Precision] convention, spelled out. Snap every
        // f32 leaf, snap at every bf16 op output, exact-copy back to f32 at
        // the model output, f32 loss.
        val xb = FloatArray(2) { snap(xv[it]) }
        val w1b = FloatArray(4) { snap(w1[it]) }
        val b1b = FloatArray(2) { snap(b1[it]) }
        val w2b = FloatArray(2) { snap(w2[it]) }
        val b2b = snap(b2[0])
        val h = FloatArray(2) { j ->
            val dot = snap(xb[0] * w1b[j] + xb[1] * w1b[2 + j])   // MATMUL snaps its output
            val z = snap(dot + b1b[j])                             // ADD snaps its output
            snap(max(0f, z))                                       // RELU snaps (no-op on snapped input)
        }
        val yRef = snap(snap(h[0] * w2b[0] + h[1] * w2b[1]) + b2b) // dot snap, then add snap
        val diff = yRef - target[0]                                // f32 from here on
        val lossRef = diff * diff                                  // mean over 1 element

        assertEquals(
            lossRef.toRawBits(), got.toRawBits(),
            "mixed loss $got != snap-reference $lossRef — the trace's snap points drifted " +
                "from the §0.4.456 compute-in-f32, round-once-at-op-boundary convention",
        )
        // The lane must be non-vacuous: the same model captured f32 must
        // give a DIFFERENT loss (the snaps really happened).
        val f32Loss = capture(model, listOf(x), lossFn = mseLoss(target, 1)).run(model, listOf(x)).loss
        assertTrue(
            f32Loss.toRawBits() != got.toRawBits(),
            "snap lane is vacuous — f32 and mixed losses are bit-identical ($got)",
        )
    }

    // ------------------------------------------------------------------
    // (3) The exact lane: when nothing rounds, mixed == f32 to the bit,
    // loss and every gradient — straight-through adds exactly nothing.
    // ------------------------------------------------------------------

    @Test
    fun bf16ExactLaneMixedMatchesF32BitExactIncludingGradients() {
        val w1 = floatArrayOf(0.5f, 2f, 1f, 0.25f)
        val b1 = floatArrayOf(0.5f, -1f)
        val w2 = floatArrayOf(2f, 0.5f)
        val b2 = floatArrayOf(0.25f)
        val xv = floatArrayOf(1f, 2f, 0.5f, -4f)
        val target = floatArrayOf(8f, 0f)

        val model = tinyModel(w1, b1, w2, b2)
        val x = Tensors.f32Matrix<Sym, Sym>(2, 2, xv)
        val mixed = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mseLoss(target, 2))
            .run(model, listOf(x))
        val plain = capture(model, listOf(x), lossFn = mseLoss(target, 2))
            .run(model, listOf(x))

        assertEquals(
            plain.loss.toRawBits(), mixed.loss.toRawBits(),
            "bf16-exact lane: mixed loss ${mixed.loss} != f32 loss ${plain.loss}",
        )
        for (key in plain.gradients.keys) {
            val gm = mixed.gradients[key]!!.hostF32()
            val gp = plain.gradients[key]!!.hostF32()
            for (i in gp.indices) {
                assertTrue(
                    gm[i].toRawBits() == gp[i].toRawBits(),
                    "bf16-exact lane grad '$key'[$i]: mixed ${gm[i]} != f32 ${gp[i]} — " +
                        "straight-through casts may not perturb exact values",
                )
            }
        }
        assertTrue(
            plain.gradients.values.any { g -> g.hostF32().any { it != 0f } },
            "exact lane is vacuous — all gradients zero",
        )
    }

    // ------------------------------------------------------------------
    // (4) The derived envelope on the threefry MLP.
    // ------------------------------------------------------------------

    @Test
    fun mixedForwardWithinDerivedBf16EnvelopeOfF32() {
        val n = 16
        val d = 4
        val hDim = 8
        val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        val xv = FloatArray(u.size) { 2f * u[it] - 1f }
        val tv = FloatArray(n) { i ->
            val r = i * d
            xv[r] * xv[r + 1] + 0.5f * xv[r + 2] - 0.25f * xv[r + 3]
        }
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        val model = Sequential(
            Dense(d, hDim, RandomKey.fromSeed(7).split(2)[0]),
            ReluLayer,
            Dense(hDim, 1, RandomKey.fromSeed(7).split(2)[1]),
        )
        val lossFn = mseLoss(tv, n)
        val l32 = capture(model, listOf(x), lossFn = lossFn).run(model, listOf(x)).loss
        val lmp = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = lossFn)
            .run(model, listOf(x)).loss

        // The derived bound (class KDoc): R = 8 snap points per forward path,
        // each ≤ 2⁻⁸ relative; A_i is the triangle-inequality magnitude bound.
        val params = model.parameters.associate { it.key to it.tensor.hostF32() }
        val w1 = params["0.w"]!!; val b1 = params["0.b"]!!
        val w2 = params["2.w"]!!; val b2 = params["2.b"]!!
        val rel = (1.0 + 2.0.pow(-8)).pow(8) - 1.0
        var bound = 0.0
        for (i in 0 until n) {
            var a = abs(b2[0]).toDouble()
            for (j in 0 until hDim) {
                var hAbs = abs(b1[j]).toDouble()
                for (k in 0 until d) hAbs += abs(xv[i * d + k]).toDouble() * abs(w1[k * hDim + j])
                a += hAbs * abs(w2[j])
            }
            val dy = rel * a
            bound += dy * (2.0 * (a + abs(tv[i])) + dy)
        }
        bound /= n
        val gap = abs(lmp - l32).toDouble()
        println("[nn-mp] envelope: |L_mixed - L_f32| = $gap, derived bound = $bound (L32=$l32, Lmp=$lmp)")
        assertTrue(gap > 0.0, "envelope test is vacuous — mixed and f32 losses are identical")
        assertTrue(
            gap <= bound,
            "mixed loss diverges beyond the derived bf16 envelope: |$lmp - $l32| = $gap > $bound",
        )
    }

    // ------------------------------------------------------------------
    // (5) 10 mixed training steps: loss strictly decreases, bit-deterministic.
    // ------------------------------------------------------------------

    @Test
    fun tenMixedStepsDecreaseLossDeterministically() {
        fun runOnce(): List<Float> {
            val n = 16
            val d = 4
            val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
            val xv = FloatArray(u.size) { 2f * u[it] - 1f }
            val tv = FloatArray(n) { i ->
                val r = i * d
                xv[r] * xv[r + 1] + 0.5f * xv[r + 2] - 0.25f * xv[r + 3]
            }
            val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
            var model = Sequential(
                Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
                ReluLayer,
                Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
            )
            val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mseLoss(tv, n))
            val opt = Adam(learningRate = 0.05f)
            var state = opt.initialState()
            val losses = ArrayList<Float>(11)
            repeat(10) {
                val res = step.run(model, listOf(x))
                losses += res.loss
                val (next, nextState) = opt.step(model, res.gradients, state)
                model = next
                state = nextState
            }
            losses += step.run(model, listOf(x)).loss
            return losses
        }

        val a = runOnce()
        println("[nn-mp] 10 mixed steps: ${a.first()} -> ${a.last()} ($a)")
        for (i in 1 until a.size) {
            assertTrue(
                a[i] < a[i - 1],
                "mixed training loss must strictly decrease over the first 10 steps " +
                    "(early training, gradients >> bf16 noise); step $i: ${a[i - 1]} -> ${a[i]} ($a)",
            )
        }
        // Bit-determinism: the master weights are f32, the graph is fixed,
        // threefry is bit-pinned — the run must reproduce exactly.
        val b = runOnce()
        assertEquals(a.size, b.size)
        for (i in a.indices) {
            assertEquals(
                a[i].toRawBits(), b[i].toRawBits(),
                "mixed training is not bit-deterministic at step $i: ${a[i]} vs ${b[i]}",
            )
        }
    }

    // ------------------------------------------------------------------
    // (6) Refusals, by name.
    // ------------------------------------------------------------------

    @Test
    fun mixedDtypeOperandsAndOutOfScopeCastsRefuseByName() {
        // bf16 ⊕ f32 on the tape: refused at trace time, named.
        val exMix = assertFailsWith<IllegalArgumentException> {
            captureN(
                listOf(
                    Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f)),
                    Tensors.f32Vector<Sym>(floatArrayOf(3f, 4f)),
                ),
            ) { leaves -> leaves[0].cast(BF16) + leaves[1] }
        }
        assertTrue(
            "dtype-homogeneous" in (exMix.message ?: ""),
            "mixing refusal must be named (got: ${exMix.message})",
        )
        // Casts outside F32↔BF16: refused, named.
        val exCast = assertFailsWith<IllegalArgumentException> {
            captureN(listOf(Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f)))) { leaves ->
                leaves[0].cast(io.tlaloc.core.I32)
            }
        }
        assertTrue(
            "Tracer.cast v1" in (exCast.message ?: ""),
            "cast-scope refusal must be named (got: ${exCast.message})",
        )
    }

    @Test
    fun gradSourceOnMixedCaptureRefusesByName() {
        val model = tinyModel(
            floatArrayOf(0.5f, 2f, 1f, 0.25f), floatArrayOf(0.5f, -1f),
            floatArrayOf(2f, 0.5f), floatArrayOf(0.25f),
        )
        val x = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 0.5f, -4f))
        val step = capture(model, listOf(x), precision = Precision.MIXED_BF16, lossFn = mseLoss(floatArrayOf(8f, 0f), 2))
        val ex = assertFailsWith<IllegalStateException> { step.gradSource() }
        assertTrue(
            "bf16" in (ex.message ?: ""),
            "the readable-reverse refusal must name bf16 (got: ${ex.message}) — " +
                "the §0.4.456 ratified story, not a silent wrong-precision print",
        )
    }
}

/** Local spelling so the snap reference reads like the convention it restates. */
private fun max(a: Float, b: Float): Float = if (a > b) a else b
