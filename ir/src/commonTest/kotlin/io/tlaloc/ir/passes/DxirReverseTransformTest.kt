package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Pure-IR tests for [DxirReverseTransform]. These exercise the structural shape of the
 * generated gradient function (returns, body composition, op kinds) without relying on
 * the K2 plugin's compile-and-run harness — those equivalence tests live next to the
 * plugin and pin the *numerical* answer; these pin the *graph shape*.
 */
class DxirReverseTransformTest {

    private val f32 = DxirType(F32, emptyList())

    // --- Hard-gate violations ---

    @Test
    fun rejectsMultipleReturns() {
        val primal = DxirBuilder.function("two_returns") {
            val x = param("x", f32)
            val a = op(OpKind.MUL, listOf(x, x), f32)
            val b = op(OpKind.ADD, listOf(x, x), f32)
            listOf(a, b)
        }
        assertFailsWith<IllegalArgumentException> { DxirReverseTransform.apply(primal) }
    }

    @Test
    fun rejectsNonScalarReturn() {
        val tensorType = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("vec_return") {
            val x = param("x", tensorType)
            val y = op(OpKind.NEG, listOf(x), tensorType)
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> { DxirReverseTransform.apply(primal) }
    }

    @Test
    fun rejectsUnsupportedOp() {
        // ABS has no registered VjpRule (no rule emits it and it's never exercised on
        // the tape path today); expect IllegalStateException. The outer SUM keeps the
        // scalar-return hard gate satisfied. When the reverse walk hits ABS after
        // processing SUM (registered → BROADCAST), it throws. MATMUL was the canonical
        // unregistered op in §0.4.8; it's registered now (§0.4.9), so the test needs
        // a new placeholder — ABS is the natural pick among unary-elementwise ops with
        // no rule.
        val vec = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("uses_abs") {
            val x = param("x", vec)
            val y = op(OpKind.ABS, listOf(x), vec)
            val s = op(OpKind.SUM, listOf(y), f32)
            listOf(s)
        }
        assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(primal) }
    }

    // --- Edge cases ---

    @Test
    fun gradientOfIdentityIsConstantOne() {
        val primal = DxirBuilder.function("id") {
            val x = param("x", f32)
            listOf(x)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals("id_grad", grad.name)
        assertEquals(1, grad.params.size)
        assertEquals(1, grad.returns.size)
        // Returned node is the seed const (1.0f).
        val ret = grad.returns.single()
        assertTrue(ret is DxirConst, "expected return to be a DxirConst (the seed)")
        assertEquals(1.0f, ret.value)
    }

    @Test
    fun gradientOfConstantIsZero() {
        val primal = DxirBuilder.function("k") {
            val x = param("x", f32)
            val k = const(7.0f, f32)
            listOf(k)
        }
        val grad = DxirReverseTransform.apply(primal)
        val ret = grad.returns.single()
        assertTrue(ret is DxirConst)
        assertEquals(0.0f, ret.value, "gradient of a constant w.r.t. its (unused) param is 0")
    }

    // --- Per-rule structural checks ---

    @Test
    fun gradientOfAddDoesNotEmitNewArithmeticForUpstreamPropagation() {
        // d(x + x)/dx accumulates upstream twice via a single ADD; no NEG / MUL needed.
        val primal = DxirBuilder.function("xpx") {
            val x = param("x", f32)
            val y = op(OpKind.ADD, listOf(x, x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        // Body is allowed to contain: cloned x is a param (no body entry), cloned ADD,
        // seed const, and the accumulator ADD. No MUL / NEG should appear.
        for (n in grad.body) {
            if (n is DxirOp) {
                assertTrue(
                    n.op == OpKind.ADD,
                    "unexpected op in gradient body for `x + x`: ${n.op}",
                )
            }
        }
    }

    @Test
    fun gradientOfNegEmitsSingleNeg() {
        val primal = DxirBuilder.function("neg") {
            val x = param("x", f32)
            val y = op(OpKind.NEG, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        // §0.4.48 — const-fold folds `NEG(seed=const 1.0)` to `const -1.0`, so the
        // post-optimization grad body has zero NEG ops. The primal NEG clone is
        // also absent (NegRule reads no operands, so it's dropped by usedByAdjoint).
        val negOps = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.NEG }
        assertEquals(0, negOps.size, "post-const-fold: NEG(const 1.0) folded to const -1.0; no NEG ops remain")
        // Gradient correctness: d/dx(-x) = -1.
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(-1f, out[0][0])
    }

    @Test
    fun dropsCloneOfDeadMulSubgraph() {
        // Primal: ADD(MUL(x, x), CONST).  ADD's rule reads no operands, so the primal
        // MUL(x, x) subgraph is dead on the gradient side — the rule never dereferences
        // the MUL's cloned value. MUL's own rule still dereferences its operands (so
        // cloned x IS needed), but the primal MUL node itself lowers to a detached
        // phantom and does not appear in the gradient body. Nor does the primal ADD,
        // nor the primal CONST.
        val primal = DxirBuilder.function("add_mul_const") {
            val x = param("x", f32)
            val xx = op(OpKind.MUL, listOf(x, x), f32)
            val k = const(5.0f, f32)
            val y = op(OpKind.ADD, listOf(xx, k), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        // §0.4.48 — post-CSE+const-fold: the two adjoint MULs `MUL(seed=1.0, x)` both
        // fold to `x`, then CSE merges them. Grad body is just `ADD(x, x)` (= 2x).
        // Primal MUL + primal ADD + primal CONST are all absent (no adjoint rule
        // dereferences them, so usedByAdjoint drops their clones).
        val muls = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.MUL }
        assertEquals(0, muls.size, "post-const-fold: MUL(1.0, x) folded to x; no MULs remain in grad body")
        val adds = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.ADD }
        assertEquals(1, adds.size, "expected the accumulator ADD(x, x)")
        // Correctness: d(x² + 5)/dx = 2x. At x=3: 6.
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(6f, out[0][0])
    }

    @Test
    fun gradientOfMulSquareAccumulatesViaAdd() {
        // d(x * x)/dx = 2x.  In our SCT: gradAccum[x] = MUL(seed, x) + MUL(seed, x) =
        // ADD(MUL, MUL).  The returned node should be an ADD whose operands are MULs.
        val primal = DxirBuilder.function("sq") {
            val x = param("x", f32)
            val y = op(OpKind.MUL, listOf(x, x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        // §0.4.48 — with const-fold, MUL(seed=1.0, x) folds to x, so the accumulator
        // ADD's operands are both param refs `x` (not MULs). Return is still an ADD.
        val ret = grad.returns.single()
        assertTrue(ret is DxirOp && ret.op == OpKind.ADD)
        // Correctness: d(x²)/dx = 2x. At x=4: 8.
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertEquals(8f, out[0][0])
    }

    @Test
    fun gradientOfSumEmitsBroadcast() {
        // d(sum(x))/dx = 1 broadcast to x's shape.  SumRule reads no operands, so the
        // primal SUM's clone drops out of the gradient body (usedByAdjoint analysis).
        // The gradient body should contain the seed const plus exactly one BROADCAST.
        val vec = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("sum_vec") {
            val x = param("x", vec)
            val y = op(OpKind.SUM, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val bcasts = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.BROADCAST }
        assertEquals(1, bcasts.size, "expected exactly one BROADCAST in SUM's gradient body")
        val bcast = bcasts.single()
        assertEquals(vec, bcast.type, "BROADCAST output shape should match the operand's shape")
        assertEquals(emptyList<Int>(), bcast.attrs["broadcast_dimensions"])
        // Returned gradient is the BROADCAST.
        assertEquals(bcast, grad.returns.single())
        // Primal SUM must not appear in the gradient body.
        val sums = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.SUM }
        assertEquals(0, sums.size, "primal SUM must not be cloned into gradient body")
    }

    @Test
    fun gradientOfMeanEmitsMulAndBroadcast() {
        // d(mean(x))/dx = (1/N) broadcast to x's shape.  Body should contain: the seed
        // const, a 1/N const, a MUL (upstream * 1/N), and a BROADCAST.
        val vec = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("mean_vec") {
            val x = param("x", vec)
            val y = op(OpKind.MEAN, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        // §0.4.48 — const-fold collapses MUL(seed=1.0, invN=0.25) into a single const
        // 0.25. Post-fold grad body has just the BROADCAST over that const plus DCE'd
        // leftovers. The 0.25 const is the BROADCAST's scalar operand.
        val bcasts = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.BROADCAST }
        assertEquals(1, bcasts.size, "expected exactly one BROADCAST in MEAN's gradient body")
        val muls = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.MUL }
        assertEquals(0, muls.size, "post-const-fold: MUL(1.0, 0.25) folded to const 0.25; no MUL remains")
        val invN = grad.body.filterIsInstance<DxirConst>().firstOrNull { it.value == 0.25f }
        assertNotNull(invN, "expected a 0.25f const for 1/N where N=4 (const-folded from MUL(1.0, 0.25))")
        assertEquals(vec, bcasts.single().type)
        assertEquals(bcasts.single(), grad.returns.single())
    }

    @Test
    fun gradientOfMatmulEmitsTwoTransposesAndTwoMatmuls() {
        // d(sum(A @ B))/dA = ones @ Bᵀ, d/dB = Aᵀ @ ones.  MatmulRule emits two
        // TRANSPOSEs + two MATMULs into the gradient body. The primal MATMUL itself
        // is NOT cloned: no downstream adjoint rule dereferences its value
        // (SumRule.readsPrimalOperandIndices is empty), so the usedByAdjoint analysis
        // drops it. The primal SUM is likewise dropped. MatmulRule reads both primal
        // operands (A and B), so cloned A + cloned B live as params in the gradient.
        val mat = DxirType(F32, listOf(2, 2))
        val primal = DxirBuilder.function("sum_matmul") {
            val a = param("a", mat)
            val b = param("b", mat)
            val ab = op(OpKind.MATMUL, listOf(a, b), mat)
            val s = op(OpKind.SUM, listOf(ab), f32)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        val transposes = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRANSPOSE }
        assertEquals(2, transposes.size, "expected exactly 2 TRANSPOSEs in MATMUL's gradient body")
        val matmuls = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.MATMUL }
        assertEquals(
            2, matmuls.size,
            "expected exactly 2 MATMULs (dA, dB); primal MATMUL must not be cloned into gradient body",
        )
        val sums = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.SUM }
        assertEquals(0, sums.size, "primal SUM must not be cloned into gradient body")
        val bcasts = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.BROADCAST }
        assertEquals(1, bcasts.size, "expected exactly one BROADCAST (SumRule's adjoint of the outer SUM)")
        for (t in transposes) {
            assertEquals(listOf(1, 0), t.attrs["permutation"])
        }
        // Two gradient returns, both rank-2 f32[2,2].
        assertEquals(2, grad.returns.size)
        for (r in grad.returns) assertEquals(mat, r.type)
    }

    @Test
    fun returnedFunctionPassesValidation() {
        // SSA ref-integrity is checked in DxirFunction's init {} block; if the transform
        // emits dangling references the construction throws. This test pins that the
        // composition `clone + reverse-walk + accumulator emit` produces a well-formed
        // function for a non-trivial primal (`x³`).
        val primal = DxirBuilder.function("cube") {
            val x = param("x", f32)
            val xx = op(OpKind.MUL, listOf(x, x), f32)
            val xxx = op(OpKind.MUL, listOf(xx, x), f32)
            listOf(xxx)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertNotNull(grad)
        assertEquals(1, grad.returns.size)
        assertEquals(f32, grad.returns.single().type)
    }

    // --- Multi-param + includeForward ---

    @Test
    fun multiParamGradientReturnsOnePerParam() {
        // f(a, b) = a * b + a.  ∂f/∂a = b + 1,  ∂f/∂b = a.
        // We check structural shape: 2 returns (one per param), both typed f32.
        val primal = DxirBuilder.function("ab_plus_a") {
            val a = param("a", f32)
            val b = param("b", f32)
            val ab = op(OpKind.MUL, listOf(a, b), f32)
            val out = op(OpKind.ADD, listOf(ab, a), f32)
            listOf(out)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size)
        assertEquals(f32, grad.returns[0].type)
        assertEquals(f32, grad.returns[1].type)
        assertEquals(2, grad.params.size)
    }

    @Test
    fun gradientOfUnusedParamIsZero() {
        // f(a, b) = a * a (b unused).  ∂f/∂a = 2a,  ∂f/∂b = 0.
        val primal = DxirBuilder.function("unused_b") {
            val a = param("a", f32)
            @Suppress("UNUSED_VARIABLE") val b = param("b", f32)
            val aa = op(OpKind.MUL, listOf(a, a), f32)
            listOf(aa)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size)
        // b's gradient is a constant 0.
        val bGrad = grad.returns[1]
        assertTrue(bGrad is DxirConst)
        assertEquals(0.0f, bGrad.value)
    }

    @Test
    fun includeForwardPrependsPrimalReturn() {
        // valueAndGrad shape: returns = [forward, ∂f/∂x].  `f(x) = x * x` → forward = x*x,
        // gradient = ADD(MUL(seed, x), MUL(seed, x)).
        val primal = DxirBuilder.function("sq_vg") {
            val x = param("x", f32)
            val y = op(OpKind.MUL, listOf(x, x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal, includeForward = true)
        assertEquals(2, grad.returns.size)
        // First return is the cloned forward MUL; second return is the gradient ADD.
        val forwardReturn = grad.returns[0]
        assertTrue(forwardReturn is DxirOp && forwardReturn.op == OpKind.MUL)
        val gradReturn = grad.returns[1]
        assertTrue(gradReturn is DxirOp && gradReturn.op == OpKind.ADD)
    }

    @Test
    fun includeForwardWithMultiParamProducesForwardPlusTwoGradients() {
        // valueAndGrad2 shape: f(a, b) = a * b + a → returns = [forward, ∂a, ∂b].
        val primal = DxirBuilder.function("ab_plus_a_vg2") {
            val a = param("a", f32)
            val b = param("b", f32)
            val ab = op(OpKind.MUL, listOf(a, b), f32)
            val out = op(OpKind.ADD, listOf(ab, a), f32)
            listOf(out)
        }
        val grad = DxirReverseTransform.apply(primal, includeForward = true)
        assertEquals(3, grad.returns.size)
        for (r in grad.returns) assertEquals(f32, r.type)
    }

    @Test
    fun gradientPreservesF64Dtype() {
        val f64 = DxirType(F64, emptyList())
        val primal = DxirBuilder.function("d_sq") {
            val x = param("x", f64)
            val y = op(OpKind.MUL, listOf(x, x), f64)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(f64, grad.returns.single().type)
        // §0.4.48 — seed const 1.0 gets folded into MULs in the adjoint chain, which
        // then fold away (MUL(1.0, x) → x). The dtype preservation lives on in the
        // gradient's RETURN type (checked above) and in the intermediate op types.
        // Correctness check: d(x²)/dx = 2x. At x=3.0: 6.0 (as Double).
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(6f, out[0][0])
    }

    // --- PowRule (Stage B.3 §0.4.21) ---

    @Test
    fun gradOfXToConstExpGivesConstTimesXPowerMinusOne() {
        // grad(x → x^3) at x=2 should be 3·x^2 = 3·4 = 12.
        val primal = DxirBuilder.function("cube") {
            val x = param("x", f32)
            val three = const(3f, f32)
            val y = op(OpKind.POW, listOf(x, three), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertEquals(1, out.size)
        assertTrue(
            kotlin.math.abs(out[0][0] - 12f) < 1e-3f,
            "expected ~12.0 for d/dx(x^3) at x=2, got ${out[0][0]}",
        )
    }

    @Test
    fun gradOfConstBaseToXGivesBasePowXTimesLnBase() {
        // grad(x → 2^x) at x=3 should be 2^3 · ln(2) = 8 · 0.6931472 ≈ 5.545.
        val primal = DxirBuilder.function("expCurve") {
            val x = param("x", f32)
            val two = const(2f, f32)
            val y = op(OpKind.POW, listOf(two, x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        val expected = 8f * kotlin.math.ln(2f)
        assertTrue(
            kotlin.math.abs(out[0][0] - expected) < 1e-3f,
            "expected $expected for d/dx(2^x) at x=3, got ${out[0][0]}",
        )
    }

    @Test
    fun gradOfXToYGivesBothAdjointsCorrectly() {
        // grad_{x,y}(x^y) at (2, 3):
        //   d/dx = y·x^(y-1) = 3·4 = 12
        //   d/dy = x^y · ln(x) = 8·ln(2) ≈ 5.545
        val primal = DxirBuilder.function("xy") {
            val x = param("x", f32)
            val y = param("y", f32)
            val z = op(OpKind.POW, listOf(x, y), f32)
            listOf(z)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size, "2-param primal should yield 2 gradients")
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f), floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(out[0][0] - 12f) < 1e-3f,
            "expected ~12.0 for d/dx, got ${out[0][0]}",
        )
        val expectedDy = 8f * kotlin.math.ln(2f)
        assertTrue(
            kotlin.math.abs(out[1][0] - expectedDy) < 1e-3f,
            "expected $expectedDy for d/dy, got ${out[1][0]}",
        )
    }

    @Test
    fun gradOfXTimesXSquaredGivesThreeXSquared() {
        // grad(x → x · x^2) = grad(x → x^3) via an ADD/MUL path rather than direct POW.
        // Actually test: grad(x → x^2 + x^3). d/dx = 2x + 3x^2 = 2·3 + 3·9 = 33 at x=3.
        val primal = DxirBuilder.function("polyPow") {
            val x = param("x", f32)
            val two = const(2f, f32)
            val three = const(3f, f32)
            val xSq = op(OpKind.POW, listOf(x, two), f32)
            val xCube = op(OpKind.POW, listOf(x, three), f32)
            val y = op(OpKind.ADD, listOf(xSq, xCube), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(out[0][0] - 33f) < 1e-3f,
            "expected 33.0 for d/dx(x^2 + x^3) at x=3, got ${out[0][0]}",
        )
    }

    @Test
    fun powRuleIsRegistered() {
        // Structural pin: POW is in the registry's supportedKinds set.
        assertTrue(OpKind.POW in VjpRegistry.supportedKinds, "POW rule should be registered")
    }

    // --- Transcendental rules (Stage A §0.4.22) ------------------------------
    //
    // ExpRule, LogRule, SqrtRule, TanhRule, SigmoidRule. Each rule widens Stage A's
    // differentiable-primitive coverage by one elementwise transcendental. Numerical
    // tolerance: 1e-3 f32 absolute (tanh/sigmoid) or 1e-3 relative (exp, sqrt where
    // the gradient's magnitude varies widely).

    @Test
    fun gradOfExpGivesExp() {
        // d/dx(exp(x)) = exp(x). At x = 1: e ≈ 2.71828.
        val primal = DxirBuilder.function("expPrimal") {
            val x = param("x", f32)
            val y = op(OpKind.EXP, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(1f)))
        assertTrue(
            kotlin.math.abs(out[0][0] - kotlin.math.E.toFloat()) < 1e-3f,
            "expected ~e for d/dx(exp(x)) at x=1, got ${out[0][0]}",
        )
        // At x = 0: exp(0) = 1.
        val outZero = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f)))
        assertTrue(
            kotlin.math.abs(outZero[0][0] - 1f) < 1e-3f,
            "expected 1.0 for d/dx(exp(x)) at x=0, got ${outZero[0][0]}",
        )
    }

    @Test
    fun gradOfLogGivesReciprocal() {
        // d/dx(log(x)) = 1/x. At x = 2: 0.5. At x = 4: 0.25.
        val primal = DxirBuilder.function("logPrimal") {
            val x = param("x", f32)
            val y = op(OpKind.LOG, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out2 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertTrue(
            kotlin.math.abs(out2[0][0] - 0.5f) < 1e-3f,
            "expected 0.5 at x=2, got ${out2[0][0]}",
        )
        val out4 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertTrue(
            kotlin.math.abs(out4[0][0] - 0.25f) < 1e-3f,
            "expected 0.25 at x=4, got ${out4[0][0]}",
        )
    }

    @Test
    fun gradOfSqrtGivesOneOverTwoSqrt() {
        // d/dx(sqrt(x)) = 1 / (2·sqrt(x)). At x = 4: 1/(2·2) = 0.25.
        val primal = DxirBuilder.function("sqrtPrimal") {
            val x = param("x", f32)
            val y = op(OpKind.SQRT, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertTrue(
            kotlin.math.abs(out[0][0] - 0.25f) < 1e-3f,
            "expected 0.25 for d/dx(sqrt(x)) at x=4, got ${out[0][0]}",
        )
        // At x = 9: 1/(2·3) ≈ 0.1667.
        val out9 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(9f)))
        assertTrue(
            kotlin.math.abs(out9[0][0] - (1f / 6f)) < 1e-3f,
            "expected ~0.1667 at x=9, got ${out9[0][0]}",
        )
    }

    @Test
    fun gradOfTanhGivesOneMinusTanhSquared() {
        // d/dx(tanh(x)) = 1 - tanh(x)². At x = 0: 1. At x = 1: 1 - tanh(1)² ≈ 0.4200.
        val primal = DxirBuilder.function("tanhPrimal") {
            val x = param("x", f32)
            val y = op(OpKind.TANH, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out0 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f)))
        assertTrue(
            kotlin.math.abs(out0[0][0] - 1f) < 1e-3f,
            "expected 1.0 for d/dx(tanh(x)) at x=0, got ${out0[0][0]}",
        )
        val out1 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(1f)))
        val expected1 = 1f - kotlin.math.tanh(1.0).let { it.toFloat() * it.toFloat() }
        assertTrue(
            kotlin.math.abs(out1[0][0] - expected1) < 1e-3f,
            "expected $expected1 at x=1, got ${out1[0][0]}",
        )
    }

    @Test
    fun gradOfSigmoidGivesSigmoidTimesOneMinusSigmoid() {
        // d/dx(σ(x)) = σ(x)·(1-σ(x)). At x = 0: σ(0)=0.5, so grad = 0.25.
        val primal = DxirBuilder.function("sigmoidPrimal") {
            val x = param("x", f32)
            val y = op(OpKind.SIGMOID, listOf(x), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out0 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f)))
        assertTrue(
            kotlin.math.abs(out0[0][0] - 0.25f) < 1e-3f,
            "expected 0.25 for d/dx(σ(x)) at x=0, got ${out0[0][0]}",
        )
        // At x = 1: σ(1) ≈ 0.7311; grad = 0.7311·0.2689 ≈ 0.1966.
        val out1 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(1f)))
        val sig1 = 1.0 / (1.0 + kotlin.math.exp(-1.0))
        val expected1 = (sig1 * (1.0 - sig1)).toFloat()
        assertTrue(
            kotlin.math.abs(out1[0][0] - expected1) < 1e-3f,
            "expected $expected1 at x=1, got ${out1[0][0]}",
        )
    }

    @Test
    fun transcendentalRulesAreAllRegistered() {
        for (kind in listOf(OpKind.EXP, OpKind.LOG, OpKind.SQRT, OpKind.TANH, OpKind.SIGMOID)) {
            assertTrue(kind in VjpRegistry.supportedKinds, "$kind rule should be registered")
        }
    }

    @Test
    fun composedTranscendentalChainDifferentiatesCorrectly() {
        // d/dx(exp(log(x))) = d/dx(x) = 1 (since exp(log(x)) = x).
        // Tests the composition of ExpRule + LogRule.
        val primal = DxirBuilder.function("expLog") {
            val x = param("x", f32)
            val logX = op(OpKind.LOG, listOf(x), f32)
            val y = op(OpKind.EXP, listOf(logX), f32)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        for (xVal in listOf(1f, 2f, 5f, 0.5f)) {
            val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(xVal)))
            assertTrue(
                kotlin.math.abs(out[0][0] - 1f) < 1e-2f,
                "expected 1.0 for d/dx(exp(log(x))) at x=$xVal, got ${out[0][0]}",
            )
        }
    }

    // --- IfRule (Stage A §0.4.23) --------------------------------------------
    //
    // SCT differentiation through OpKind.IF via the `handleIfAdjoint` special-case.
    // Per paper C2 — `d/dx(φ(a, b)) = φ(da/dx, db/dx)` — the gradient of an IF result
    // distributes through both branches, with each contribution wrapped in an IF that
    // picks the appropriate per-branch gradient based on the runtime predicate.

    private val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    @Test
    fun gradOfIfWithComputationInBranchesPicksCorrectBranchAdjoint() {
        // f(x) = if (x > 0) x*x else -x
        // d/dx = if (x > 0) 2x else -1
        // At x = 2: 2x = 4 (then branch). At x = -3: -1 (else branch). At x = 0: -1 (STEP(0)=0 → else).
        val primal = DxirBuilder.function("ifSquareElseNeg") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32)
                    yields(sq)
                },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32)
                    yields(neg)
                },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 2 (then-branch): expected d/dx = 2x = 4
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - 4f) < 1e-3f,
            "expected 4.0 at x=2 (then-branch active), got ${outPos[0][0]}",
        )
        // x = -3 (else-branch): expected d/dx = -1
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-3f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - (-1f)) < 1e-3f,
            "expected -1.0 at x=-3 (else-branch active), got ${outNeg[0][0]}",
        )
        // x = 0 (STEP(0)=0 → else-branch): expected d/dx = -1
        val outZero = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f)))
        assertTrue(
            kotlin.math.abs(outZero[0][0] - (-1f)) < 1e-3f,
            "expected -1.0 at x=0 (else-branch via STEP(0)=0), got ${outZero[0][0]}",
        )
    }

    @Test
    fun gradOfIfYieldingOuterScopeIsAbsoluteValue() {
        // f(x) = if (x > 0) x else -x  (i.e., abs(x))
        // d/dx = if (x > 0) 1 else -1  (i.e., sign(x))
        val primal = DxirBuilder.function("absViaIf") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(negX) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        for ((xVal, expected) in listOf(2f to 1f, 5f to 1f, -3f to -1f, -0.5f to -1f)) {
            val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(xVal)))
            assertTrue(
                kotlin.math.abs(out[0][0] - expected) < 1e-3f,
                "at x=$xVal: expected $expected, got ${out[0][0]}",
            )
        }
    }

    @Test
    fun gradOfMaxFunctionPicksActiveOperand() {
        // f(x, y) = if (x > y) x else y  (i.e., max(x, y))
        // d/dx = if (x > y) 1 else 0
        // d/dy = if (x > y) 0 else 1
        val primal = DxirBuilder.function("max2") {
            val x = param("x", f32)
            val y = param("y", f32)
            val diff = op(OpKind.SUB, listOf(x, y), f32)
            val pred = op(OpKind.STEP, listOf(diff), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(y) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size, "max(x, y) should yield 2 gradients")
        // x > y: d/dx = 1, d/dy = 0
        val outXLarger = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(5f), floatArrayOf(2f)))
        assertTrue(kotlin.math.abs(outXLarger[0][0] - 1f) < 1e-3f, "expected d/dx=1, got ${outXLarger[0][0]}")
        assertTrue(kotlin.math.abs(outXLarger[1][0] - 0f) < 1e-3f, "expected d/dy=0, got ${outXLarger[1][0]}")
        // x < y: d/dx = 0, d/dy = 1
        val outYLarger = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(1f), floatArrayOf(7f)))
        assertTrue(kotlin.math.abs(outYLarger[0][0] - 0f) < 1e-3f, "expected d/dx=0, got ${outYLarger[0][0]}")
        assertTrue(kotlin.math.abs(outYLarger[1][0] - 1f) < 1e-3f, "expected d/dy=1, got ${outYLarger[1][0]}")
    }

    @Test
    fun gradOfIfWithMultipleOpsInBranchAccumulatesContributions() {
        // f(x) = if (x > 0) x*x + x else -x*x*x
        // d/dx = if (x > 0) 2x + 1 else -3x²
        // At x = 2 (then): 2·2 + 1 = 5
        // At x = -1 (else): -3·1 = -3
        val primal = DxirBuilder.function("multiOpBranches") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32)
                    val sumXX = op(OpKind.ADD, listOf(sq, x), f32)
                    yields(sumXX)
                },
                elseRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32)
                    val cube = op(OpKind.MUL, listOf(sq, x), f32)
                    val negCube = op(OpKind.NEG, listOf(cube), f32)
                    yields(negCube)
                },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - 5f) < 1e-3f,
            "expected 5.0 at x=2 (then), got ${outPos[0][0]}",
        )
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-1f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - (-3f)) < 1e-3f,
            "expected -3.0 at x=-1 (else), got ${outNeg[0][0]}",
        )
    }

    @Test
    fun ifOpAcceptedByReverseTransformGate() {
        // The pre-§0.4.23 transform hard-gated on `hasRegions = true`. After §0.4.23,
        // IF is allowed (other region-bearing ops still rejected). This test pins the
        // gate's relaxation by constructing a minimal IF primal and verifying the
        // transform doesn't throw at gate-check time.
        val primal = DxirBuilder.function("ifGate") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region { yields(x) },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32)
                    yields(neg)
                },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal) // should not throw
        assertEquals(1, grad.returns.size)
    }

    @Test
    fun gradReverseTransformRejectsWhileWithNoRule() {
        // WHILE has regions but no rule supports it; the post-§0.4.23 gate still rejects.
        val primal = DxirBuilder.function("whileGate") {
            val x = param("x", f32)
            val nBound = const(2f, f32)
            val zero = const(0f, f32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val newX = op(OpKind.NEG, listOf(args[0]), f32)
                    val one = const(1f, f32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        assertFailsWith<IllegalArgumentException> { DxirReverseTransform.apply(primal) }
    }
}
