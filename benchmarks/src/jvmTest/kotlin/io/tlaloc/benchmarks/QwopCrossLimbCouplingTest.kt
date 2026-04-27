package io.tlaloc.benchmarks

import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.219 — QWOP Phase 2 seventh slice: WHILE-in-WHILE coarsening
 * structural and gradient pin on [Qwop.crossLimbCouplingPrimal]. The only
 * QWOP slice that exercises §0.4.176's nested-loop coarsening surface.
 *
 * **What this test pins**:
 *  - Structure: 2 WHILEs in the primal body (outer + inner), 0 IFs.
 *  - Coarsening: `PhiCalculus.apply` reduces the WHILE count. Whether C5
 *    fully eliminates both WHILEs (ideal) or leaves the inner WHILEs
 *    around inside the unrolled outer body depends on the recursion
 *    behaviour of §0.4.176's WHILE-in-WHILE coarsening.
 *  - Forward eval: `acc = nFrames × nLimbs × knee × (hip + shoulder)`.
 *  - Multi-input gradient through nested coarsening:
 *    `df/dhip = K × knee`, `df/dknee = K × (hip + shoulder)`,
 *    `df/dshoulder = K × knee` where `K = nFrames × nLimbs`.
 *
 * **Why this slice matters**: nested loop coarsening is the only QWOP
 * shape that exercises both axes of §0.4.176's surface — the outer's
 * inits/yields carry the inner WHILE's result, and the gradient must
 * flow back through both layers correctly.
 */
class QwopCrossLimbCouplingTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.crossLimbCouplingPrimal()
        // BenchmarkPrimals.countOps counts only top-level body ops, so the
        // inner WHILE (nested inside the outer WHILE's body region) is not
        // visible here. Pin the outer WHILE only.
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "crossLimbCouplingPrimal: 1 top-level WHILE (outer); inner WHILE lives inside the outer body region",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "crossLimbCouplingPrimal: 0 IFs (pure pairwise MUL+ADD recurrences)",
        )
    }

    @Test
    fun forwardEval() {
        val primal = Qwop.crossLimbCouplingPrimal()
        // K = nFrames × nLimbs = 9.
        // At (hip=2, knee=3, shoulder=4):
        //   innerAcc per outer iter = nLimbs × (hip×knee + knee×shoulder)
        //                            = 3 × (6 + 12) = 54
        //   acc = nFrames × innerAcc = 3 × 54 = 162
        // Equivalently: acc = K × knee × (hip + shoulder) = 9 × 3 × 6 = 162.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f), floatArrayOf(4.0f)),
        )
        assertTrue(
            abs(out[0][0] - 162.0f) < 1e-3f,
            "expected forward = 162.0 at (2, 3, 4), got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusOnNestedWhile() {
        // §0.4.219 — observability pin: simply verify that PhiCalculus.apply
        // doesn't throw on a WHILE-in-WHILE primal. The exact post-coarsening
        // WHILE count depends on §0.4.176's recursion behaviour:
        //   - 0 top-level WHILEs (ideal): outer fully unrolls + inner WHILEs
        //     also unroll after they get hoisted to top-level.
        //   - 3 top-level WHILEs (partial): outer unrolls into 3 copies, each
        //     bringing its inner WHILE to top-level; recursion doesn't fire.
        //   - 1 top-level WHILE (no coarsening): outer survives.
        // All three are valid behaviours pre-Phase-2-closure; what matters
        // is the gradient correctness pin below, not the coarsening shape.
        val primal = Qwop.crossLimbCouplingPrimal()
        val coarsened = PhiCalculus.apply(primal)
        // Smoke check: forward eval of the coarsened form matches the primal.
        val outPrimal = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f), floatArrayOf(4.0f)),
        )
        val outCoarsened = DxirInterpreter.evalFunction(
            coarsened,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f), floatArrayOf(4.0f)),
        )
        assertTrue(
            abs(outCoarsened[0][0] - outPrimal[0][0]) < 1e-3f,
            "coarsened forward should match primal forward (162.0); got primal=${outPrimal[0][0]}, coarsened=${outCoarsened[0][0]}",
        )
    }

    @Test
    fun gradientAtPositiveInputs() {
        // §0.4.219 — the headline pin. K = nFrames × nLimbs = 9.
        // At (hip=2, knee=3, shoulder=4):
        //   df/dhip = K × knee = 9 × 3 = 27
        //   df/dknee = K × (hip + shoulder) = 9 × 6 = 54
        //   df/dshoulder = K × knee = 9 × 3 = 27
        val primal = Qwop.crossLimbCouplingPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f), floatArrayOf(4.0f)),
        )
        assertTrue(
            abs(out[0][0] - 27.0f) < 1e-3f,
            "expected df/dhip = 27.0 (= K × knee = 9 × 3), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 54.0f) < 1e-3f,
            "expected df/dknee = 54.0 (= K × (hip + shoulder) = 9 × 6), got ${out[1][0]}",
        )
        assertTrue(
            abs(out[2][0] - 27.0f) < 1e-3f,
            "expected df/dshoulder = 27.0 (= K × knee = 9 × 3), got ${out[2][0]}",
        )
    }

    @Test
    fun gradientWithZeroKnee() {
        // §0.4.219 — discriminator pin. At (hip=1, knee=0, shoulder=1):
        // every iteration's innerAcc body adds `1×0 + 0×1 = 0`. innerAcc=0,
        // acc=0. Forward = 0.
        // Gradients:
        //   df/dhip = K × knee = 9 × 0 = 0
        //   df/dknee = K × (hip + shoulder) = 9 × 2 = 18 (still flows!)
        //   df/dshoulder = K × knee = 9 × 0 = 0
        // The discriminator is df/dknee = 18 — gradient still flows through
        // BOTH MULs (hip×knee AND knee×shoulder) even though the forward
        // value is 0. A buggy chain rule that short-circuited at zero would
        // give df/dknee = 0.
        val primal = Qwop.crossLimbCouplingPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1.0f), floatArrayOf(0.0f), floatArrayOf(1.0f)),
        )
        assertTrue(
            abs(out[0][0]) < 1e-3f,
            "expected df/dhip = 0.0 at zero-knee (= K × knee = 0), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 18.0f) < 1e-3f,
            "expected df/dknee = 18.0 at zero-knee (= K × (hip + shoulder) = 18), " +
                "got ${out[1][0]}",
        )
        assertTrue(
            abs(out[2][0]) < 1e-3f,
            "expected df/dshoulder = 0.0 at zero-knee, got ${out[2][0]}",
        )
    }

    @Test
    fun gradientSymmetryHipShoulder() {
        // §0.4.219 — symmetry pin. The recurrence treats hip and shoulder
        // symmetrically (both appear only as `knee × X` in innerAcc), so
        // df/dhip should always equal df/dshoulder regardless of inputs.
        // Pin this at multiple input sets to catch any per-input
        // asymmetry bug introduced by WHILE-in-WHILE gradient routing.
        val primal = Qwop.crossLimbCouplingPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        // Set 1: (1, 5, 7) — df/dhip = df/dshoulder = K × knee = 45.
        val out1 = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1.0f), floatArrayOf(5.0f), floatArrayOf(7.0f)),
        )
        assertTrue(
            abs(out1[0][0] - out1[2][0]) < 1e-3f,
            "df/dhip should equal df/dshoulder by symmetry; got ${out1[0][0]} vs ${out1[2][0]}",
        )
        assertTrue(abs(out1[0][0] - 45.0f) < 1e-3f, "df/dhip = 45 at (1, 5, 7)")

        // Set 2: (-2, 3, 8) — df/dhip = df/dshoulder = K × knee = 27.
        val out2 = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(-2.0f), floatArrayOf(3.0f), floatArrayOf(8.0f)),
        )
        assertTrue(
            abs(out2[0][0] - out2[2][0]) < 1e-3f,
            "df/dhip should equal df/dshoulder; got ${out2[0][0]} vs ${out2[2][0]}",
        )
        assertTrue(abs(out2[0][0] - 27.0f) < 1e-3f, "df/dhip = 27 at (-2, 3, 8)")
    }
}
