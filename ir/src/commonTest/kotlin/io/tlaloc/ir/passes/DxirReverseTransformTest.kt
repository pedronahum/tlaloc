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
        // RSQRT has no registered VjpRule; expect IllegalStateException. The outer SUM
        // keeps the scalar-return hard gate satisfied. When the reverse walk hits RSQRT
        // after processing SUM (registered → BROADCAST), it throws.
        //
        // History: MATMUL was the canonical unregistered op in §0.4.8; registered §0.4.9.
        // ABS replaced it but was registered §0.4.167 alongside CartPole's port. RSQRT is
        // the next natural pick — unary-elementwise reciprocal square root with no
        // present-day use case in our benchmarks.
        val vec = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("uses_rsqrt") {
            val x = param("x", vec)
            val y = op(OpKind.RSQRT, listOf(x), vec)
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
    fun gradOfNestedIfInsideThenArmFlowsCorrectly() {
        // §0.4.140 — outer IF whose then-arm body contains a nested IF.
        // f(x) = if (x > 0) { if (-x > 0) -x else x*x } else { x }
        // For x > 0: outer-then fires. -x is negative, so STEP(-x) = 0 → inner-else
        //   fires, yielding x*x. Therefore f(x) = x² for x > 0; d/dx = 2x.
        // For x <= 0: outer-else fires, yielding x. Therefore f(x) = x for x <= 0;
        //   d/dx = 1.
        val primal = DxirBuilder.function("nestedIfInThen") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val pInner = op(OpKind.STEP, listOf(negX), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region {
                    val xx = op(OpKind.MUL, listOf(x, x), f32)
                    val innerIf = ifOp(
                        cond = pInner,
                        types = listOf(f32),
                        thenRegion = region { yields(negX) },
                        elseRegion = region { yields(xx) },
                    )
                    yields(innerIf)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 3 (then-arm fires; STEP(-3) = 0 → else-of-inner fires; f = 9; d/dx = 2x = 6).
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - 6f) < 1e-3f,
            "expected 6 at x=3 (inner else fires; d/dx of x² = 2x), got ${outPos[0][0]}",
        )
        // x = -2 (outer else; f = x; d/dx = 1).
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - 1f) < 1e-3f,
            "expected 1 at x=-2 (outer else; d/dx of x = 1), got ${outNeg[0][0]}",
        )
    }

    @Test
    fun gradOfWhileInIfBranchAfterPhiCalculusUnrollFlowsCorrectly() {
        // §0.4.153 — Multi-result IF AD Phase 4 AD-side: end-to-end check that
        // `PhiCalculus.apply` (region-recursive C5, §0.4.152) followed by
        // `DxirReverseTransform.apply` produces a correct gradient.
        //
        // f(x) = if (x > 0) iterate3(x) else x
        // After PhiCalculus.apply: F3 swaps branches (then-yield's id > else-yield's
        // id under the canonical-order rule), then region-recursive C5 unrolls the
        // WHILE inside what is now the else-region. The resulting IF has shape
        // `if (NOT(x>0)) x else mulChain`.
        // For x > 0: f = 8x; df/dx = 8.
        // For x ≤ 0: f = x; df/dx = 1.
        val i32 = DxirType(io.tlaloc.core.I32, emptyList())
        val primal = DxirBuilder.function("ifWithInnerWhileAd") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region {
                    val nConst = const(3, i32)
                    val zero = const(0, i32)
                    val w = whileOp(
                        inits = listOf(x, zero),
                        cond = { args ->
                            val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                            val pred = op(OpKind.STEP, listOf(diff), boolS)
                            yields(pred)
                        },
                        body = { args ->
                            val two = const(2f, f32)
                            val newX = op(OpKind.MUL, listOf(args[0], two), f32)
                            val one = const(1, i32)
                            val newI = op(OpKind.ADD, listOf(args[1], one), i32)
                            yields(newX, newI)
                        },
                    )
                    yields(w.result(0))
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        // x = 3 → then-branch (post-F3 swap → else-of-rewritten = mulChain) fires;
        // f(3) = 8·3 = 24; df/dx = 8.
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - 8f) < 1e-3f,
            "expected df/dx = 8 at x=3 (8x branch fires), got ${outPos[0][0]}",
        )
        // x = -2 → then-branch (post-F3 swap → then-of-rewritten = identity) fires;
        // f(-2) = -2; df/dx = 1.
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - 1f) < 1e-3f,
            "expected df/dx = 1 at x=-2 (identity branch fires), got ${outNeg[0][0]}",
        )
        // Boundary check at x=0: STEP(0) = 0 → identity branch; f(0) = 0; df/dx = 1.
        val outZero = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f)))
        assertTrue(
            kotlin.math.abs(outZero[0][0] - 1f) < 1e-3f,
            "expected df/dx = 1 at x=0 (identity branch fires), got ${outZero[0][0]}",
        )
    }

    @Test
    fun gradOfNestedIfInsideElseArmFlowsCorrectly() {
        // §0.4.140 — outer IF with nested IF in else-arm only.
        // f(x) = if (x > 0) x else (if (-x > 5) x*x else -x)
        // For x > 0: outer-then; f = x; d/dx = 1.
        // For x <= 0: outer-else fires.
        //   -x > 5 ↔ x < -5: inner-then fires; f = x*x; d/dx = 2x.
        //   -x <= 5 ↔ x >= -5 (and x <= 0): inner-else; f = -x; d/dx = -1.
        val primal = DxirBuilder.function("nestedIfInElse") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val five = const(5f, f32)
            val negXMinusFive = op(OpKind.SUB, listOf(negX, five), f32)
            val pInner = op(OpKind.STEP, listOf(negXMinusFive), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region { yields(x) },
                elseRegion = region {
                    val xx = op(OpKind.MUL, listOf(x, x), f32)
                    val innerIf = ifOp(
                        cond = pInner,
                        types = listOf(f32),
                        thenRegion = region { yields(xx) },
                        elseRegion = region { yields(negX) },
                    )
                    yields(innerIf)
                },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 4 (outer-then; d/dx of x = 1).
        val outA = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertTrue(
            kotlin.math.abs(outA[0][0] - 1f) < 1e-3f,
            "expected 1 at x=4 (outer-then; d/dx of x), got ${outA[0][0]}",
        )
        // x = -10 (outer-else; -x > 5; inner-then; d/dx of x² = 2x = -20).
        val outB = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-10f)))
        assertTrue(
            kotlin.math.abs(outB[0][0] - (-20f)) < 1e-3f,
            "expected -20 at x=-10 (inner-then; d/dx of x² = 2x), got ${outB[0][0]}",
        )
        // x = -2 (outer-else; -x = 2 < 5; inner-else; d/dx of -x = -1).
        val outC = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertTrue(
            kotlin.math.abs(outC[0][0] - (-1f)) < 1e-3f,
            "expected -1 at x=-2 (inner-else; d/dx of -x = -1), got ${outC[0][0]}",
        )
    }

    @Test
    fun gradOfNestedMultiResultIfWithLiveIndexZeroFlowsCorrectly() {
        // §0.4.144 — outer single-result IF whose then-arm contains a nested
        // multi-result IF; only result(0) is live downstream of the inner IF.
        // f(x) = if (x > 0) innerMrIf.result(0) else x
        // innerMrIf cond = STEP(x - 5):
        //   inner-then  yields (x², x³)
        //   inner-else  yields (-x, x³)
        // For x > 5: outer-then; inner-then; result(0) = x²; d/dx = 2x.
        // For 0 < x ≤ 5: outer-then; inner-else; result(0) = -x; d/dx = -1.
        // For x ≤ 0: outer-else; result = x; d/dx = 1.
        val primal = DxirBuilder.function("nestedMrIfLive0") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val five = const(5f, f32)
            val xMinus5 = op(OpKind.SUB, listOf(x, five), f32)
            val pInner = op(OpKind.STEP, listOf(xMinus5), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region {
                    val negX = op(OpKind.NEG, listOf(x), f32)
                    val xx = op(OpKind.MUL, listOf(x, x), f32)
                    val xxx = op(OpKind.MUL, listOf(xx, x), f32)
                    val innerIf = ifOp(
                        cond = pInner,
                        types = listOf(f32, f32),
                        thenRegion = region { yields(xx, xxx) },
                        elseRegion = region { yields(negX, xxx) },
                    )
                    yields(innerIf.result(0))
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 8 (outer-then; inner-then; d/dx of x² = 2x = 16).
        val outA = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(8f)))
        assertTrue(
            kotlin.math.abs(outA[0][0] - 16f) < 1e-3f,
            "expected 16 at x=8 (inner-then; d/dx of x² = 2x), got ${outA[0][0]}",
        )
        // x = 3 (outer-then; inner-else; d/dx of -x = -1).
        val outB = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(outB[0][0] - (-1f)) < 1e-3f,
            "expected -1 at x=3 (inner-else; d/dx of -x), got ${outB[0][0]}",
        )
        // x = -2 (outer-else; d/dx of x = 1).
        val outC = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertTrue(
            kotlin.math.abs(outC[0][0] - 1f) < 1e-3f,
            "expected 1 at x=-2 (outer-else; d/dx of x), got ${outC[0][0]}",
        )
    }

    @Test
    fun gradOfNestedMultiResultIfWithLiveIndexOneFlowsCorrectly() {
        // §0.4.144 — same shape as the live-index-0 test but the outer-then yields
        // innerMrIf.result(1) instead. Both inner branches yield x³ at result(1),
        // so within outer-then f = x³ regardless of inner predicate; d/dx = 3x².
        val primal = DxirBuilder.function("nestedMrIfLive1") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val five = const(5f, f32)
            val xMinus5 = op(OpKind.SUB, listOf(x, five), f32)
            val pInner = op(OpKind.STEP, listOf(xMinus5), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region {
                    val negX = op(OpKind.NEG, listOf(x), f32)
                    val xx = op(OpKind.MUL, listOf(x, x), f32)
                    val xxx = op(OpKind.MUL, listOf(xx, x), f32)
                    val innerIf = ifOp(
                        cond = pInner,
                        types = listOf(f32, f32),
                        thenRegion = region { yields(xx, xxx) },
                        elseRegion = region { yields(negX, xxx) },
                    )
                    yields(innerIf.result(1))
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 4 (outer-then; result(1) = x³; d/dx = 3x² = 48).
        val outA = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertTrue(
            kotlin.math.abs(outA[0][0] - 48f) < 1e-2f,
            "expected 48 at x=4 (d/dx of x³ = 3x²), got ${outA[0][0]}",
        )
        // x = 2 (outer-then; d/dx of x³ = 3x² = 12).
        val outB = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertTrue(
            kotlin.math.abs(outB[0][0] - 12f) < 1e-3f,
            "expected 12 at x=2, got ${outB[0][0]}",
        )
        // x = -1 (outer-else; d/dx of x = 1).
        val outC = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-1f)))
        assertTrue(
            kotlin.math.abs(outC[0][0] - 1f) < 1e-3f,
            "expected 1 at x=-1 (outer-else), got ${outC[0][0]}",
        )
    }

    @Test
    fun gradOfNestedMultiResultIfWithMultipleLiveIndicesFlowsCorrectly() {
        // §0.4.155 — Phase 5b: nested multi-live-index MR IF AD also lands. Phase 3
        // (§0.4.144) rejected this shape inside an outer branch; Phase 5b accepts
        // it via the same per-(id, idx) substrate, with `nestedIfLiveIndices` now
        // a `Set<Int>` mirroring the top-level path.
        //
        // f(x) = if (x>0) {
        //   let innerIf = if (x>0) yields(-x, x²) else yields(x, x²)
        //   yields(innerIf.result(0) + innerIf.result(1))
        // } else { yields(x) }
        //
        // For x > 0: outer-then; inner-then (also x > 0). innerIf yields (-x, x²).
        //   sum = -x + x²; df/dx = -1 + 2x.
        // For x ≤ 0: outer-else; f = x; df/dx = 1.
        val primal = DxirBuilder.function("nestedMrIfMultiLive") {
            val x = param("x", f32)
            val pOuter = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pOuter,
                types = listOf(f32),
                thenRegion = region {
                    val negX = op(OpKind.NEG, listOf(x), f32)
                    val xx = op(OpKind.MUL, listOf(x, x), f32)
                    val pInner = op(OpKind.STEP, listOf(x), boolS)
                    val innerIf = ifOp(
                        cond = pInner,
                        types = listOf(f32, f32),
                        thenRegion = region { yields(negX, xx) },
                        elseRegion = region { yields(x, xx) },
                    )
                    val sum = op(OpKind.ADD, listOf(innerIf.result(0), innerIf.result(1)), f32)
                    yields(sum)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 3 (outer-then; inner-then): df/dx = -1 + 6 = 5.
        val outA = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(outA[0][0] - 5f) < 1e-3f,
            "expected df/dx = 5 at x=3 (outer-then; inner-then: -1 + 2x), got ${outA[0][0]}",
        )
        // x = -2 (outer-else): df/dx = 1.
        val outB = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2f)))
        assertTrue(
            kotlin.math.abs(outB[0][0] - 1f) < 1e-3f,
            "expected df/dx = 1 at x=-2 (outer-else: identity), got ${outB[0][0]}",
        )
    }

    @Test
    fun gradOfMultiResultIfWithLiveIndexZeroFlowsCorrectly() {
        // §0.4.139 — multi-result IF where the function returns DxirOpResult(if, 0).
        // f(x) = (if (x > 0) -x else x).result(0). At x=2: then-arm picks -x = -2;
        // d/dx = -1. At x=-3: else-arm picks x = -3; d/dx = 1.
        val primal = DxirBuilder.function("ifMultiResultIdx0") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val twoX = op(OpKind.MUL, listOf(x, x), f32)
            // Multi-result IF with two outputs: [f32, f32]. Only result(0) is referenced.
            val ifOp = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(negX, twoX) },  // then: result0 = -x, result1 = x²
                    region { yields(x, twoX) },      // else: result0 = x, result1 = x²
                ),
            )
            listOf(ifOp.result(0))
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 2 (then-branch): expected d/dx of -x = -1.
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(2f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - (-1f)) < 1e-3f,
            "expected -1.0 at x=2 (then-branch returns -x), got ${outPos[0][0]}",
        )
        // x = -3 (else-branch): expected d/dx of x = 1.
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-3f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - 1f) < 1e-3f,
            "expected 1.0 at x=-3 (else-branch returns x), got ${outNeg[0][0]}",
        )
    }

    @Test
    fun gradOfMultiResultIfWithLiveIndexOneFlowsCorrectly() {
        // §0.4.139 — same MR IF but the function returns result(1) instead.
        // result(1) is x² in both branches. d/dx of x² is 2x at every x.
        val primal = DxirBuilder.function("ifMultiResultIdx1") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val twoX = op(OpKind.MUL, listOf(x, x), f32)
            val ifOp = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(negX, twoX) },
                    region { yields(x, twoX) },
                ),
            )
            listOf(ifOp.result(1))
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 5 → 2x = 10. (Both branches yield twoX at result(1), so the IF reduces to x².)
        val outPos = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(5f)))
        assertTrue(
            kotlin.math.abs(outPos[0][0] - 10f) < 1e-3f,
            "expected 10 at x=5 (d/dx of x² = 2x), got ${outPos[0][0]}",
        )
        // x = -3 → 2x = -6.
        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-3f)))
        assertTrue(
            kotlin.math.abs(outNeg[0][0] - (-6f)) < 1e-3f,
            "expected -6 at x=-3, got ${outNeg[0][0]}",
        )
    }

    @Test
    fun gradOfMultiResultIfWithMultipleLiveIndicesFlowsCorrectly() {
        // §0.4.155 — Phase 5b: multi-live-index MR IF AD lands. Phase 1 (§0.4.139)
        // rejected this shape; Phase 5b accepts it via the per-(id, idx) gradAccum
        // substrate (§0.4.154). Both result(0) and result(1) are referenced
        // downstream; their per-index upstream contributions seed the branch walk
        // at distinct terminator slots.
        //
        // f(x) = ifop.result(0) + ifop.result(1)
        //   where ifop = if (x>0) yields(-x, x²) else yields(x, x²)
        // For x > 0:  f = -x + x²;  df/dx = -1 + 2x.
        // For x ≤ 0:  f =  x + x²;  df/dx =  1 + 2x.
        val primal = DxirBuilder.function("ifMultiLive") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val xSquared = op(OpKind.MUL, listOf(x, x), f32)
            val ifOp = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(negX, xSquared) },
                    region { yields(x, xSquared) },
                ),
            )
            // Reference BOTH result(0) and result(1) — multiple live indices.
            val sum = op(OpKind.ADD, listOf(ifOp.result(0), ifOp.result(1)), f32)
            listOf(sum)
        }
        val grad = DxirReverseTransform.apply(primal)
        // x = 4 (then-branch): df/dx = -1 + 8 = 7.
        val outA = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(4f)))
        assertTrue(
            kotlin.math.abs(outA[0][0] - 7f) < 1e-3f,
            "expected df/dx = 7 at x=4 (then-arm: -1 + 2x), got ${outA[0][0]}",
        )
        // x = 3 (then-branch): df/dx = -1 + 6 = 5.
        val outB = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertTrue(
            kotlin.math.abs(outB[0][0] - 5f) < 1e-3f,
            "expected df/dx = 5 at x=3 (then-arm: -1 + 2x), got ${outB[0][0]}",
        )
        // x = -3 (else-branch): df/dx = 1 + (-6) = -5.
        val outC = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-3f)))
        assertTrue(
            kotlin.math.abs(outC[0][0] - (-5f)) < 1e-3f,
            "expected df/dx = -5 at x=-3 (else-arm: 1 + 2x), got ${outC[0][0]}",
        )
    }

    @Test
    fun gradientOfRank3BatchedMatmulEmitsBatchedTransposes() {
        // §0.4.137 — rank-3 batched matmul flows through the same MatmulRule.
        // Permutation should be [0, 2, 1] (preserve batch axis, swap last two).
        val rank3 = DxirType(F32, listOf(2, 2, 2))
        val primal = DxirBuilder.function("sum_bmm") {
            val a = param("a", rank3)
            val b = param("b", rank3)
            val ab = op(OpKind.MATMUL, listOf(a, b), rank3)
            val s = op(OpKind.SUM, listOf(ab), f32)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        val transposes = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRANSPOSE }
        assertEquals(2, transposes.size, "expected exactly 2 TRANSPOSEs for rank-3 batched matmul")
        for (t in transposes) {
            assertEquals(listOf(0, 2, 1), t.attrs["permutation"], "rank-3 permutation should be [0, 2, 1]")
        }
        // Numerical pin: at A = ones, B = ones, both rank-3 [2, 2, 2], the gradient
        // of sum(A @@ B) is a rank-3 tensor of size 2 (the K dim) at every cell.
        // dA[b, m, k] = sum_n B[b, k, n]; for B = ones, this is N = 2.
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(8) { 1f }, FloatArray(8) { 1f }),
        )
        assertEquals(2, out.size, "two gradient returns")
        for (g in out) {
            assertEquals(8, g.size, "gradient shape matches rank-3 input")
            for (v in g) assertEquals(2f, v, "every cell should be 2 (N=2)")
        }
    }

    @Test
    fun gradientOfRank4BatchedMatmulEmitsTransposesWithMultiBatchPermutation() {
        // §0.4.138 — rank-4 batched matmul: (B0, B1, M, K) × (B0, B1, K, N).
        // The MatmulRule generalises to any rank ≥ 2; for rank 4 the permutation
        // becomes [0, 1, 3, 2] (preserve B0, B1; swap last two).
        val rank4 = DxirType(F32, listOf(2, 3, 2, 2))   // (B0=2, B1=3, M=2, K=2)
        val rank4B = DxirType(F32, listOf(2, 3, 2, 2)) // (B0=2, B1=3, K=2, N=2)
        val primal = DxirBuilder.function("sum_rank4_matmul") {
            val a = param("a", rank4)
            val b = param("b", rank4B)
            val ab = op(OpKind.MATMUL, listOf(a, b), rank4)
            val s = op(OpKind.SUM, listOf(ab), f32)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        val transposes = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRANSPOSE }
        assertEquals(2, transposes.size, "expected exactly 2 TRANSPOSEs for rank-4 batched matmul")
        for (t in transposes) {
            assertEquals(
                listOf(0, 1, 3, 2),
                t.attrs["permutation"],
                "rank-4 permutation should be [0, 1, 3, 2] — preserve both batch axes",
            )
        }
        // Numerical pin: at A = ones, B = ones, both rank-4 (2, 3, 2, 2). For each
        // (b0, b1) batch slice (2×2 matmul of ones × ones), dA[b0, b1, m, k] =
        // sum_n B[b0, b1, k, n] = N = 2.
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(2 * 3 * 2 * 2) { 1f }, FloatArray(2 * 3 * 2 * 2) { 1f }),
        )
        assertEquals(2, out.size)
        for (g in out) {
            assertEquals(2 * 3 * 2 * 2, g.size)
            for (v in g) assertEquals(2f, v, "every cell of the gradient should be N = 2")
        }
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

    // --- Region-internal CSE (§0.4.118) -------------------------------------

    @Test
    fun cseDeduplicatesOpsInsideIfBranch() {
        // Hand-build a function whose IF then-branch has two structurally-identical
        // ADD ops. Pre-§0.4.118 the early-out skipped CSE entirely when ANY body op
        // had regions; now CSE recurses into IF region bodies. The duplicate inside
        // the branch should be deduplicated.
        val pred = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("ifInternalDup") {
            val x = param("x", f32)
            val p = op(OpKind.STEP, listOf(x), pred)
            val ifResult = ifOp(
                cond = p,
                types = listOf(f32),
                thenRegion = region {
                    val a = op(OpKind.ADD, listOf(x, x), f32)
                    val b = op(OpKind.ADD, listOf(x, x), f32)
                    val sum = op(OpKind.ADD, listOf(a, b), f32)
                    yields(sum)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        // Locate the IF op in the post-CSE body.
        val ifOpPost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        val thenBlock = ifOpPost.regions[0].blocks.single()
        // Pre-CSE the then-block had 3 ADD ops (a, b, sum); post-CSE the duplicate
        // ADD(x, x) should be merged → 2 ADD ops total in the branch.
        val thenAdds = thenBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        assertEquals(2, thenAdds, "duplicate ADD inside then-branch should be CSE'd")
    }

    @Test
    fun cseDeduplicatesAcrossOuterToInnerScope() {
        // Pin: an outer-scope op's canonical entry should be visible inside an IF
        // region. If the outer body computes ADD(x, x) and the then-branch ALSO
        // computes ADD(x, x), the inner ADD should dedup to the outer one.
        val pred = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("crossScopeDup") {
            val x = param("x", f32)
            val outerAdd = op(OpKind.ADD, listOf(x, x), f32)
            val p = op(OpKind.STEP, listOf(x), pred)
            val ifResult = ifOp(
                cond = p,
                types = listOf(f32),
                thenRegion = region {
                    val innerAdd = op(OpKind.ADD, listOf(x, x), f32) // dup of outerAdd
                    val scaled = op(OpKind.MUL, listOf(innerAdd, x), f32)
                    yields(scaled)
                },
                elseRegion = region { yields(x) },
            )
            // Use both the outer ADD and the IF to keep them live.
            val combined = op(OpKind.ADD, listOf(outerAdd, ifResult), f32)
            listOf(combined)
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val ifOpPost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        val thenBlock = ifOpPost.regions[0].blocks.single()
        // The then-branch's ADD should now be dropped (deduped to outer canonical),
        // leaving only the MUL.
        val thenAdds = thenBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        assertEquals(0, thenAdds, "inner ADD should dedup to outer canonical; got $thenAdds")
        val thenMuls = thenBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.MUL }
        assertEquals(1, thenMuls)
    }

    @Test
    fun cseDoesNotShareRegistrationsBetweenSiblingBranches() {
        // The then-branch and else-branch are scope-isolated. An op registered in the
        // then-branch must NOT be visible to the else-branch (they're different
        // control-flow scopes; an else-branch op cannot reference a then-branch op).
        // Both branches independently build ADD(x, x). The else-branch's ADD must
        // survive — it can't be deduplicated to the then-branch's.
        val pred = DxirType(io.tlaloc.core.Bool, emptyList())
        val fn = DxirBuilder.function("siblingScope") {
            val x = param("x", f32)
            val p = op(OpKind.STEP, listOf(x), pred)
            val ifResult = ifOp(
                cond = p,
                types = listOf(f32),
                thenRegion = region {
                    val tAdd = op(OpKind.ADD, listOf(x, x), f32)
                    yields(tAdd)
                },
                elseRegion = region {
                    val eAdd = op(OpKind.ADD, listOf(x, x), f32)
                    yields(eAdd)
                },
            )
            listOf(ifResult)
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val ifOpPost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        val thenBlock = ifOpPost.regions[0].blocks.single()
        val elseBlock = ifOpPost.regions[1].blocks.single()
        // Each branch must have its own ADD — they can't dedup across siblings.
        assertEquals(1, thenBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD })
        assertEquals(1, elseBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD })
    }

    @Test
    fun cseAppliesToCoarsenedNestedFunctions() {
        // §0.4.119 — COARSENED stores `primal_body` and `gradient_body` as
        // DxirFunctions in attrs (NOT as regions). When applyCSE encounters a
        // COARSENED op, it should also CSE those nested functions. Build a
        // minimal COARSENED whose primal_body contains a duplicate ADD; verify
        // the post-CSE COARSENED's primal_body has the duplicate removed.
        val pred = DxirType(io.tlaloc.core.Bool, emptyList())
        @Suppress("UNUSED_VARIABLE") val _pred = pred  // silence linter on unused boolS
        // Build the inner primal_body with a redundant ADD pair.
        val innerPrimal = DxirBuilder.function("inner_primal") {
            val x = param("x", f32)
            val a = op(OpKind.ADD, listOf(x, x), f32)
            val b = op(OpKind.ADD, listOf(x, x), f32)
            val sum = op(OpKind.ADD, listOf(a, b), f32)
            listOf(sum)
        }
        // Hand-build a gradient_body shape that satisfies handleCoarsenedAdjoint's
        // require: 1 + N params (upstream + N primal operands), N returns. Here
        // primal has 1 operand (x), so 2 params (upstream, x), 1 return (the
        // adjoint w.r.t. x).
        val innerGrad = DxirBuilder.function("inner_grad") {
            val upstream = param("upstream", f32)
            val xPrim = param("x", f32)
            val two = const(2f, f32)
            val grad = op(OpKind.MUL, listOf(upstream, two), f32)
            // dummy use of xPrim to keep ref-integrity
            val _u = op(OpKind.ADD, listOf(grad, xPrim), f32)
            listOf(_u)
        }
        // Outer fn that contains the COARSENED op.
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32)
            val c = coarsened(
                operands = listOf(x),
                primalBody = innerPrimal,
                gradientBody = innerGrad,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        val outerPrimalBefore = (outer.body.first { it is DxirOp && (it as DxirOp).op == OpKind.COARSENED } as DxirOp)
            .attrs["primal_body"] as io.tlaloc.ir.DxirFunction
        val addsBefore = outerPrimalBefore.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        assertEquals(3, addsBefore, "pre-CSE primal_body should have 3 ADDs (a, b, sum)")

        val cseOuter = DxirReverseTransform.applyCSE(outer)
        val coarsenedPost = cseOuter.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
        val csedPrimal = coarsenedPost.attrs["primal_body"] as io.tlaloc.ir.DxirFunction
        val addsAfter = csedPrimal.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        assertEquals(2, addsAfter, "post-CSE primal_body must have 2 ADDs (duplicate merged)")
    }

    @Test
    fun cseStructurallyConvergesOnAlreadyCsedCoarsenedNestedFunctions() {
        // Running applyCSE twice on the same outer function should converge: the
        // second pass produces a function structurally equivalent to the first
        // (same op counts in the COARSENED's nested primal_body). Pre-§0.4.119
        // the pass would never recurse into COARSENED's attrs, so the inner
        // duplicate ADD persisted; post-§0.4.119 the duplicate is gone after pass 1
        // and stays gone after pass 2.
        val innerPrimal = DxirBuilder.function("inner_primal") {
            val x = param("x", f32)
            val a = op(OpKind.ADD, listOf(x, x), f32)
            val b = op(OpKind.ADD, listOf(x, x), f32)
            val sum = op(OpKind.ADD, listOf(a, b), f32)
            listOf(sum)
        }
        val innerGrad = DxirBuilder.function("inner_grad") {
            val upstream = param("upstream", f32)
            val xPrim = param("x", f32)
            val two = const(2f, f32)
            val grad = op(OpKind.MUL, listOf(upstream, two), f32)
            val out = op(OpKind.ADD, listOf(grad, xPrim), f32)
            listOf(out)
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32)
            val c = coarsened(
                operands = listOf(x),
                primalBody = innerPrimal,
                gradientBody = innerGrad,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        val once = DxirReverseTransform.applyCSE(outer)
        val twice = DxirReverseTransform.applyCSE(once)
        // Both passes should yield the same op-count shape inside primal_body.
        fun primalAddCount(fn: io.tlaloc.ir.DxirFunction): Int {
            val coars = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
            val pb = coars.attrs["primal_body"] as io.tlaloc.ir.DxirFunction
            return pb.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        }
        assertEquals(2, primalAddCount(once), "after pass 1, primal_body has 2 ADDs (duplicate merged)")
        assertEquals(2, primalAddCount(twice), "after pass 2, primal_body still has 2 ADDs (idempotent)")
    }

    // --- Region-internal CSE for WHILE (§0.4.129) ---------------------------

    @Test
    fun cseDeduplicatesOpsInsideWhileBodyRegion() {
        // §0.4.129 — pre-§0.4.129 the dispatch in cseNode short-circuited multi-result
        // ops (incl. WHILE) before reaching cseRegionBearingOp, so a WHILE's body
        // never benefited from internal CSE. Now WHILE regions get the same recursion
        // as IF regions. Build a WHILE whose body computes `MUL(args[0], 2)` twice;
        // verify the duplicate is dedup'd inside the body region.
        val fn = DxirBuilder.function("whileBodyDup") {
            val x = param("x", f32)
            val zero = const(0f, f32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val n = const(3f, f32)
                    val diff = op(OpKind.SUB, listOf(n, args[1]), f32)
                    yields(op(OpKind.STEP, listOf(diff), boolS))
                },
                body = { args ->
                    val two = const(2f, f32)
                    val a = op(OpKind.MUL, listOf(args[0], two), f32)
                    // Duplicate: same operands, same kind, same attrs.
                    val b = op(OpKind.MUL, listOf(args[0], two), f32)
                    val newX = op(OpKind.ADD, listOf(a, b), f32)
                    val one = const(1f, f32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val whilePost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.WHILE }
        val bodyBlock = whilePost.regions[1].blocks.single()
        // Pre-CSE the body had 2 MUL ops; post-CSE only one survives.
        val muls = bodyBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.MUL }
        assertEquals(1, muls, "duplicate MUL inside WHILE body should be CSE'd")
    }

    @Test
    fun cseDeduplicatesConstsInsideWhileCondRegion() {
        // §0.4.129 — the cond region also gets internal CSE. Build a WHILE whose
        // cond region declares the same const twice; verify the dup is collapsed.
        val fn = DxirBuilder.function("whileCondConstDup") {
            val x = param("x", f32)
            val zero = const(0f, f32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val n1 = const(5f, f32)
                    val n2 = const(5f, f32)  // duplicate const
                    // Sig-different ops (one uses n1, one uses n2) but the CONSTs
                    // themselves dedup → operand canonicalisation merges the SUBs too.
                    val diffA = op(OpKind.SUB, listOf(n1, args[1]), f32)
                    val diffB = op(OpKind.SUB, listOf(n2, args[1]), f32)
                    val sum = op(OpKind.ADD, listOf(diffA, diffB), f32)
                    yields(op(OpKind.STEP, listOf(sum), boolS))
                },
                body = { args ->
                    val two = const(2f, f32)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32)
                    val one = const(1f, f32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val whilePost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.WHILE }
        val condBlock = whilePost.regions[0].blocks.single()
        // Pre-CSE: 2 const(5f). Post-CSE: 1 const(5f) (the other folded into the
        // canonical entry; the SUB ops then sig-merge to a single SUB).
        val consts = condBlock.body.filterIsInstance<io.tlaloc.ir.DxirConst>().count {
            (it.value as? Number)?.toFloat() == 5f
        }
        assertEquals(1, consts, "duplicate const(5f) in cond region should be CSE'd")
        val subs = condBlock.body.filterIsInstance<DxirOp>().count { it.op == OpKind.SUB }
        assertEquals(1, subs, "post-CSE the two SUB(n, args[1]) collapse to one canonical SUB")
    }

    @Test
    fun cseDoesNotShareCondAndBodyRegistrationsAcrossWhile() {
        // §0.4.129 — the cond region and body region are scope-isolated — they have
        // independent block args (different ids), so an op rooted at cond's args[0]
        // can't dedup against an op rooted at body's args[0]. Build a WHILE where
        // both regions independently compute MUL(args[0], 2); verify both survive.
        val fn = DxirBuilder.function("whileScopeIsolation") {
            val x = param("x", f32)
            val zero = const(0f, f32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    // cond region's MUL — uses cond args.
                    val two = const(2f, f32)
                    val scaled = op(OpKind.MUL, listOf(args[0], two), f32)
                    val n = const(10f, f32)
                    val diff = op(OpKind.SUB, listOf(n, scaled), f32)
                    yields(op(OpKind.STEP, listOf(diff), boolS))
                },
                body = { args ->
                    // body region's MUL — uses body args (different ids than cond's).
                    val two = const(2f, f32)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32)
                    val one = const(1f, f32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val whilePost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.WHILE }
        val condMuls = whilePost.regions[0].blocks.single().body.filterIsInstance<DxirOp>().count { it.op == OpKind.MUL }
        val bodyMuls = whilePost.regions[1].blocks.single().body.filterIsInstance<DxirOp>().count { it.op == OpKind.MUL }
        // Each region's MUL must survive — they reference different block args.
        assertEquals(1, condMuls, "cond region's MUL must survive (args from cond block)")
        assertEquals(1, bodyMuls, "body region's MUL must survive (args from body block)")
    }

    // --- §0.4.130: cseRegion DxirOpResult terminator latent bug fix ----------

    @Test
    fun cseRegionPreservesDxirOpResultIndexInTerminator() {
        // §0.4.130 — the cseRegion terminator handler used to do a flat
        // `innerById[term.id] ?: term` lookup, which for a DxirOpResult resolved to
        // the source op (not .result(k)). With multi-result ops inside IF arms,
        // that produced terminators whose type didn't match op.types[k] — the
        // DxirFunction validator caught it as "terminator type ≠ op.types".
        // Construct an IF whose then-arm yields `whileOp.result(1)` (i32 counter,
        // not the f32 carried at index 0); §0.4.130's fix routes through .result(k).
        val i32s = DxirType(io.tlaloc.core.I32, emptyList())
        val fn = DxirBuilder.function("cseMrTerminator") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(i32s),
                thenRegion = region {
                    // Multi-result WHILE: types = [f32, i32], yields(f32, i32) per iter.
                    val nBound = const(2, i32s)
                    val zero = const(0, i32s)
                    val w = whileOp(
                        inits = listOf(x, zero),
                        cond = { args ->
                            val diff = op(OpKind.SUB, listOf(nBound, args[1]), i32s)
                            yields(op(OpKind.STEP, listOf(diff), boolS))
                        },
                        body = { args ->
                            val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32)), f32)
                            val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                            yields(newX, newI)
                        },
                    )
                    // Yield the i32 counter (index 1), NOT the f32 carried at index 0.
                    // Pre-§0.4.130 the CSE rebuild collapsed this to `w` (DxirOp.type
                    // = f32), failing IF validation: terminator type f32 ≠ op.types[0] i32.
                    yields(w.result(1))
                },
                elseRegion = region {
                    val zero = const(0, i32s)
                    yields(zero)
                },
            )
            listOf(ifResult)
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        // The post-CSE function must validate (would throw IllegalArgumentException
        // pre-fix). Sanity: the IF op survives with the right shape.
        val ifOpPost = cseFn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        assertEquals(listOf(i32s), ifOpPost.types, "IF result type preserved")
        val thenTerminator = ifOpPost.regions[0].blocks.single().terminator.single()
        assertEquals(i32s, thenTerminator.type, "then-arm terminator type matches op.types[0] = i32")
    }

    // --- gradient_body with nested IF (§0.4.120) -----------------------------

    @Test
    fun coarsenedWithIfInGradientBodyClonesAndEvaluates() {
        // §0.4.120 — handleCoarsenedAdjoint accepts IF inside gradient_body. Build a
        // hand-crafted COARSENED whose gradient_body computes `if (true) upstream
        // else upstream`, an identity-IF. The reverse pass must clone the IF into
        // the outer gradient builder, and the resulting gradient must evaluate to
        // upstream (= 1.0 for a unit primal seed).
        // Primal body: just `x` (identity). Coarsened so it gets routed through
        // handleCoarsenedAdjoint.
        val primalBody = DxirBuilder.function("inner_primal") {
            val x = param("x", f32)
            listOf(x)
        }
        // Gradient body: (upstream, x) → if (true) upstream else upstream = upstream.
        // Need a Bool predicate that's a function param or const. We'll use const(true).
        val boolType = DxirType(io.tlaloc.core.Bool, emptyList())
        val gradientBody = DxirBuilder.function("inner_grad") {
            val upstream = param("upstream", f32)
            val xPrim = param("x", f32)  // required by handleCoarsenedAdjoint's contract
            // Bool predicate via STEP(xPrim) — we don't actually care which branch fires;
            // both branches yield `upstream`, so the IF is identity on upstream.
            val pred = op(OpKind.STEP, listOf(xPrim), boolType)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region { yields(upstream) },
                elseRegion = region { yields(upstream) },
            )
            listOf(ifResult)
        }
        val outer = DxirBuilder.function("outer") {
            val x = param("x", f32)
            val c = coarsened(
                operands = listOf(x),
                primalBody = primalBody,
                gradientBody = gradientBody,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        // Pre-§0.4.120 this would throw: "gradient_body op IF has regions (...not supported)".
        // Post-§0.4.120 it cleanly clones the IF into the gradient.
        val grad = DxirReverseTransform.apply(outer)
        // Numerical check: f(x) = x → df/dx = 1.
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(1f, out[0][0], "df/dx for identity primal should be 1")
    }

    @Test
    fun coarsenedWithNestedIfInsideIfArmClonesAndEvaluates() {
        // §0.4.121 — handleCoarsenedAdjoint allows IF nested inside another IF's arm
        // in gradient_body. Build a hand-crafted COARSENED whose gradient_body computes:
        //   if (STEP(x)) { if (STEP(x)) upstream else upstream } else upstream
        // All three branches yield `upstream`, so the nested IF is identity. The reverse
        // pass must clone both IFs (outer + nested) into the gradient builder context.
        val primalBody = DxirBuilder.function("inner_primal") {
            val x = param("x", f32)
            listOf(x)
        }
        val boolType = DxirType(io.tlaloc.core.Bool, emptyList())
        val gradientBody = DxirBuilder.function("inner_grad") {
            val upstream = param("upstream", f32)
            val xPrim = param("x", f32)
            val pred = op(OpKind.STEP, listOf(xPrim), boolType)
            val outer = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region {
                    // Nested IF inside the then-arm.
                    val innerPred = op(OpKind.STEP, listOf(xPrim), boolType)
                    val nested = ifOp(
                        cond = innerPred,
                        types = listOf(f32),
                        thenRegion = region { yields(upstream) },
                        elseRegion = region { yields(upstream) },
                    )
                    yields(nested)
                },
                elseRegion = region { yields(upstream) },
            )
            listOf(outer)
        }
        val outerFn = DxirBuilder.function("outer") {
            val x = param("x", f32)
            val c = coarsened(
                operands = listOf(x),
                primalBody = primalBody,
                gradientBody = gradientBody,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        // Pre-§0.4.121 this would throw at the inner-arm IF: "block op IF has regions
        // (nested IF inside an IF arm not supported)". Post-§0.4.121 the recursive
        // clone handles it.
        val grad = DxirReverseTransform.apply(outerFn)
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
        assertEquals(1f, out[0][0], "df/dx for identity primal should be 1, regardless of nested IF structure")
    }

    @Test
    fun csePreservesExistingTopLevelBehaviorWhenNoRegions() {
        // Regression: a region-free function should still get top-level CSE applied.
        // Pin §0.4.48's existing dedup behavior to ensure §0.4.118's restructure
        // didn't break it.
        val fn = DxirBuilder.function("noRegions") {
            val x = param("x", f32)
            val a = op(OpKind.ADD, listOf(x, x), f32)
            val b = op(OpKind.ADD, listOf(x, x), f32)
            val sum = op(OpKind.ADD, listOf(a, b), f32)
            listOf(sum)
        }
        val cseFn = DxirReverseTransform.applyCSE(fn)
        val adds = cseFn.body.filterIsInstance<DxirOp>().count { it.op == OpKind.ADD }
        assertEquals(2, adds, "duplicate ADD(x,x) should be merged at top level")
    }

    // --- TransposeRule (rank-2 self-inverse + rank-3 non-trivial inverse) ---

    @Test
    fun gradOfTransposeRank2EmitsTransposeWithSamePermutation() {
        // y = transpose(x, [1, 0]); loss = sum(y). Expected gradient body:
        // upstream BROADCAST (rank-2 ones from SumRule) → TRANSPOSE([1,0]) of upstream → returned as dx.
        // For rank-2 the inverse permutation [1,0] equals the forward permutation, so
        // we additionally pin that the emitted permutation attr equals [1, 0].
        val mat = DxirType(F32, listOf(2, 3))
        val matT = DxirType(F32, listOf(3, 2))
        val primal = DxirBuilder.function("sum_transpose") {
            val x = param("x", mat)
            val y = op(
                OpKind.TRANSPOSE, listOf(x), matT,
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            val s = op(OpKind.SUM, listOf(y), f32)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        val transposes = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRANSPOSE }
        assertEquals(1, transposes.size, "expected exactly one TRANSPOSE in TRANSPOSE's gradient body")
        assertEquals(listOf(1, 0), transposes.single().attrs["permutation"])
        // The single returned gradient is rank-2 f32[2, 3], matching x's shape.
        assertEquals(1, grad.returns.size)
        assertEquals(mat, grad.returns.single().type)
    }

    @Test
    fun gradOfTransposeRank3UsesInversePermutation() {
        // perm = [2, 0, 1] sends (i,j,k) → output position (k,i,j) — i.e. y[a,b,c] = x[b,c,a].
        // The inverse permutation is [1, 2, 0] (sanity-checked: composition with perm yields identity).
        val rank3 = DxirType(F32, listOf(2, 3, 4))
        val rank3T = DxirType(F32, listOf(4, 2, 3))
        val primal = DxirBuilder.function("sum_transpose3") {
            val x = param("x", rank3)
            val y = op(
                OpKind.TRANSPOSE, listOf(x), rank3T,
                attrs = mapOf("permutation" to listOf(2, 0, 1)),
            )
            val s = op(OpKind.SUM, listOf(y), f32)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        val transposes = grad.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRANSPOSE }
        assertEquals(1, transposes.size)
        assertEquals(listOf(1, 2, 0), transposes.single().attrs["permutation"])
        assertEquals(1, grad.returns.size)
        assertEquals(rank3, grad.returns.single().type, "dx must match the primal x's shape")
    }

    @Test
    fun gradOfTransposeRejectsMissingPermutationAttr() {
        // Defensive: TRANSPOSE without permutation cannot be differentiated. The
        // :stablehlo emitter already rejects this on the forward side, but the rule
        // must surface a load-bearing error rather than crashing with NPE inside
        // DxirReverseTransform.
        val mat = DxirType(F32, listOf(2, 2))
        val primal = DxirBuilder.function("transpose_no_perm") {
            val x = param("x", mat)
            val y = op(OpKind.TRANSPOSE, listOf(x), mat)
            val s = op(OpKind.SUM, listOf(y), f32)
            listOf(s)
        }
        val ex = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(primal) }
        assertTrue(
            "permutation" in (ex.message ?: ""),
            "expected error message to mention 'permutation'; got: ${ex.message}",
        )
    }
}
