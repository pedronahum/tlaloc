package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * §0.4.31 — Stage C.3b.1 tests. Hand-build [OpKind.COARSENED] primals and run them
 * through [DxirInterpreter]'s new `evalCoarsened` arm; verify validation catches
 * malformed COARSENED shapes; verify [DxirReverseTransform] rejects COARSENED-bearing
 * primals pending C.3b.2.
 *
 * C.3b.2 will add `VjpRule` + `DxirReverseTransform` integration; those tests will
 * live in their own file alongside the new `handleCoarsenedAdjoint` helper.
 */
class CoarsenedOpTest {

    private val f32s = DxirType(F32, emptyList())

    /** Build a simple primal: f(x) = x * x.  Single-result, one operand. */
    private fun squarePrimal(): DxirFunction =
        DxirBuilder.function("square_primal") {
            val x = param("x", f32s)
            val sq = op(OpKind.MUL, listOf(x, x), f32s)
            listOf(sq)
        }

    /** Build a matching gradient body: (upstream, x) → (2 * upstream * x). */
    private fun squareGradient(): DxirFunction =
        DxirBuilder.function("square_gradient") {
            val upstream = param("upstream", f32s)
            val x = param("x", f32s)
            val two = const(2f, f32s)
            val twoX = op(OpKind.MUL, listOf(two, x), f32s)
            val dx = op(OpKind.MUL, listOf(upstream, twoX), f32s)
            listOf(dx)
        }

    @Test
    fun coarsenedOpRoundTripsThroughInterpreter() {
        // Outer function: g(a) = coarsened(a) where primal_body is a → a * a.
        // Expected: g(3) = 9.
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val coarsened = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            listOf(coarsened)
        }
        val out = DxirInterpreter.evalFunction(outer, listOf(floatArrayOf(3f)))
        assertEquals(1, out.size)
        assertEquals(1, out[0].size)
        assertEquals(9f, out[0][0])
    }

    @Test
    fun coarsenedOpInsideLargerFunctionComposesCorrectly() {
        // Outer: g(a) = coarsened(a) + a.  primal = a * a → 9 + 3 = 12 at a=3.
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("compose") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            val r = op(OpKind.ADD, listOf(c, a), f32s)
            listOf(r)
        }
        val out = DxirInterpreter.evalFunction(outer, listOf(floatArrayOf(3f)))
        assertEquals(12f, out[0][0])
    }

    @Test
    fun coarsenedMultiResultDispatchesThroughMultiResultKey() {
        // Primal with 2 returns: (a, b) → (a + b, a - b).
        val primal = DxirBuilder.function("addsub") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val sum = op(OpKind.ADD, listOf(a, b), f32s)
            val diff = op(OpKind.SUB, listOf(a, b), f32s)
            listOf(sum, diff)
        }
        // §0.4.179 — Phase 5c gradient_body shape for K=2, N=2: (u_sum, u_diff, a, b) → (da, db).
        // Semantic correctness isn't exercised by the interpreter test (which only
        // evaluates primal_body); only shape validity matters here.
        val grad = DxirBuilder.function("addsub_grad") {
            val uSum = param("u_sum", f32s)
            val uDiff = param("u_diff", f32s)
            val a = param("a", f32s)
            val b = param("b", f32s)
            val _ign = op(OpKind.ADD, listOf(a, b), f32s)
            val _ign2 = op(OpKind.ADD, listOf(_ign, uSum), f32s)
            val _ign3 = op(OpKind.ADD, listOf(_ign2, uDiff), f32s)
            listOf(uSum, uDiff)
        }
        val outer = DxirBuilder.function("outer") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0, 1),
            )
            // c is multi-result; route both results via .result(i).
            listOf(c.result(0), c.result(1))
        }
        val out = DxirInterpreter.evalFunction(outer, listOf(floatArrayOf(5f), floatArrayOf(2f)))
        assertEquals(2, out.size)
        assertEquals(7f, out[0][0], "sum")
        assertEquals(3f, out[1][0], "diff")
    }

    @Test
    fun coarsenedValidationRejectsMismatchedOperandCount() {
        val primal = squarePrimal()
        val grad = squareGradient()
        try {
            DxirBuilder.function("bad") {
                val a = param("a", f32s)
                val b = param("b", f32s)
                // primal takes 1 param but we pass 2 operands.
                coarsened(
                    operands = listOf(a, b),
                    primalBody = primal,
                    gradientBody = grad,
                    readsPrimalIndices = setOf(0),
                )
                listOf(a)
            }
            fail("expected IllegalArgumentException for operand-count mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(
                e.message!!.contains("operand count") ||
                    e.message!!.contains("primal_body.params count"),
                "unexpected error: ${e.message}",
            )
        }
    }

    @Test
    fun coarsenedValidationRejectsMismatchedOperandType() {
        val primal = squarePrimal()
        val grad = squareGradient()
        try {
            DxirBuilder.function("bad_type") {
                val a = param("a", DxirType(io.tlaloc.core.I32, emptyList()))
                // primal expects f32 operand, we pass i32.
                coarsened(
                    operands = listOf(a),
                    primalBody = primal,
                    gradientBody = grad,
                    readsPrimalIndices = setOf(0),
                )
                listOf(a)
            }
            fail("expected IllegalArgumentException for operand-type mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(
                e.message!!.contains("type") || e.message!!.contains("≠"),
                "unexpected error: ${e.message}",
            )
        }
    }

    @Test
    fun coarsenedValidationRejectsOutOfRangeReadsIndex() {
        val primal = squarePrimal()
        val grad = squareGradient()
        try {
            DxirBuilder.function("bad_reads") {
                val a = param("a", f32s)
                coarsened(
                    operands = listOf(a),
                    primalBody = primal,
                    gradientBody = grad,
                    readsPrimalIndices = setOf(0, 5),  // 5 is out of range
                )
                listOf(a)
            }
            fail("expected IllegalArgumentException for out-of-range reads index")
        } catch (e: IllegalArgumentException) {
            assertTrue(
                e.message!!.contains("out-of-range") || e.message!!.contains("reads_primal_indices"),
                "unexpected error: ${e.message}",
            )
        }
    }

    @Test
    fun coarsenedValidationRejectsBadGradientSignature() {
        val primal = squarePrimal()
        // Gradient body has wrong signature: missing the primal operand after upstream.
        val badGrad = DxirBuilder.function("bad_grad") {
            val upstream = param("upstream", f32s)
            listOf(upstream)
        }
        try {
            DxirBuilder.function("bad_grad_sig") {
                val a = param("a", f32s)
                coarsened(
                    operands = listOf(a),
                    primalBody = primal,
                    gradientBody = badGrad,
                    readsPrimalIndices = setOf(0),
                )
                listOf(a)
            }
            fail("expected IllegalArgumentException for gradient-body signature mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(
                e.message!!.contains("gradient_body.params"),
                "unexpected error: ${e.message}",
            )
        }
    }

    // ------------------------------------------------------------------------
    // §0.4.32 — C.3b.2: gradient splice via handleCoarsenedAdjoint
    // ------------------------------------------------------------------------

    @Test
    fun gradThroughCoarsenedSinglePrimalReturnsCorrectDerivative() {
        // outer: f(a) = coarsened(a) where primal = a * a.
        // Expected: df/da = 2a.  At a = 3 → 6.
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(3f)))
        assertEquals(1, out.size)
        assertEquals(1, out[0].size)
        assertEquals(6f, out[0][0], "df/da at a=3 must be 2*3=6")
    }

    @Test
    fun gradThroughCoarsenedComposedWithAddition() {
        // outer: f(a) = coarsened(a) + 1. primal = a * a.
        // df/da = 2a (the +1 contributes 0). At a=4 → 8.
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            val one = const(1f, f32s)
            val r = op(OpKind.ADD, listOf(c, one), f32s)
            listOf(r)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(4f)))
        assertEquals(8f, out[0][0])
    }

    @Test
    fun gradThroughCoarsenedWithMultipleOperands() {
        // outer: f(a, b) = coarsened(a, b) where primal = a * b.
        // df/da = b, df/db = a.  At (a=5, b=3) → (3, 5).
        val primal = DxirBuilder.function("mul_primal") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val r = op(OpKind.MUL, listOf(a, b), f32s)
            listOf(r)
        }
        // gradient_body: (upstream, a, b) → (upstream * b, upstream * a).
        val grad = DxirBuilder.function("mul_gradient") {
            val upstream = param("upstream", f32s)
            val a = param("a", f32s)
            val b = param("b", f32s)
            val da = op(OpKind.MUL, listOf(upstream, b), f32s)
            val db = op(OpKind.MUL, listOf(upstream, a), f32s)
            listOf(da, db)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0, 1),
            )
            listOf(c)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(
            gradFn,
            listOf(floatArrayOf(5f), floatArrayOf(3f)),
        )
        assertEquals(2, out.size, "grad fn has 2 returns (one per outer param)")
        assertEquals(3f, out[0][0], "df/da = b = 3")
        assertEquals(5f, out[1][0], "df/db = a = 5")
    }

    @Test
    fun gradThroughCoarsenedNumericallyMatchesUncoarsenedEquivalent() {
        // Equivalence test: the COARSENED form must produce the same gradient as a
        // plain primal with the same math (a → a*a). Any drift surfaces here.
        val coarsenedPrimal = squarePrimal()
        val coarsenedGrad = squareGradient()
        val outerCoarsened = DxirBuilder.function("fc") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = coarsenedPrimal,
                gradientBody = coarsenedGrad,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        val outerPlain = DxirBuilder.function("fp") {
            val a = param("a", f32s)
            val sq = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(sq)
        }
        val gradCoarsened = DxirReverseTransform.apply(outerCoarsened)
        val gradPlain = DxirReverseTransform.apply(outerPlain)
        for (aVal in listOf(-3f, -1f, 0f, 1f, 2f, 5f, 10f)) {
            val outC = DxirInterpreter.evalFunction(gradCoarsened, listOf(floatArrayOf(aVal)))
            val outP = DxirInterpreter.evalFunction(gradPlain, listOf(floatArrayOf(aVal)))
            assertEquals(
                outP[0][0], outC[0][0],
                "COARSENED grad must match plain grad at a=$aVal (plain=${outP[0][0]}, coarsened=${outC[0][0]})",
            )
        }
    }

    @Test
    fun gradThroughCoarsenedDeadOperandProducesZeroContribution() {
        // outer: f(a, b) = coarsened(a, b) where primal = a * a (ignores b).
        // gradient_body returns d/da = 2*upstream*a, d/db = 0 (hard-coded zero).
        // reads_primal_indices = {0} — b's subgraph isn't cloned into grad body.
        val primal = DxirBuilder.function("sq_only_a") {
            val a = param("a", f32s)
            val _b = param("b", f32s)
            val sq = op(OpKind.MUL, listOf(a, a), f32s)
            listOf(sq)
        }
        val grad = DxirBuilder.function("sq_only_a_grad") {
            val upstream = param("upstream", f32s)
            val a = param("a", f32s)
            val _b = param("b", f32s)
            val two = const(2f, f32s)
            val twoA = op(OpKind.MUL, listOf(two, a), f32s)
            val da = op(OpKind.MUL, listOf(upstream, twoA), f32s)
            val zero = const(0f, f32s)
            listOf(da, zero)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),  // b not read → its operand subgraph need not be cloned
            )
            listOf(c)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(
            gradFn,
            listOf(floatArrayOf(4f), floatArrayOf(99f)),
        )
        assertEquals(2, out.size)
        assertEquals(8f, out[0][0], "d/da = 2*4 = 8")
        assertEquals(0f, out[1][0], "d/db = 0 (b unused in primal)")
    }

    // ------------------------------------------------------------------------
    // §0.4.34 — C.3b.3b1: COARSENED inside an IF branch
    // ------------------------------------------------------------------------

    @Test
    fun gradThroughCoarsenedInsideIfThenBranch() {
        // outer: f(x) = if (x > 0) coarsened(x) else x.
        // primal inside then-branch: x*x. Expected d/dx = if (x > 0) 2x else 1.
        // At x=3 (then-branch): grad = 6. At x=-2 (else-branch): grad = 1.
        val boolS = io.tlaloc.ir.DxirType(io.tlaloc.core.Bool, emptyList())
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("if_coarsened") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val c = coarsened(
                        operands = listOf(x),
                        primalBody = primal,
                        gradientBody = grad,
                        readsPrimalIndices = setOf(0),
                    )
                    yields(c)
                },
                elseRegion = region { yields(x) },
            )
            listOf(result)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        // At x=3 (then-branch active): gradient should be d/dx (x*x) = 2x = 6.
        val outPos = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(3f)))
        assertEquals(6f, outPos[0][0], "then-branch grad at x=3 must be 6")
        // At x=-2 (else-branch active): gradient should be d/dx (x) = 1.
        val outNeg = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(-2f)))
        assertEquals(1f, outNeg[0][0], "else-branch grad at x=-2 must be 1")
    }

    @Test
    fun gradThroughCoarsenedInsideIfElseBranch() {
        // Symmetric: f(x) = if (x > 0) x else coarsened(x). primal inside else = x*x.
        // d/dx = if (x > 0) 1 else 2x. At x=3 (then) → 1; at x=-2 (else) → -4.
        val boolS = io.tlaloc.ir.DxirType(io.tlaloc.core.Bool, emptyList())
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("if_coarsened_else") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region {
                    val c = coarsened(
                        operands = listOf(x),
                        primalBody = primal,
                        gradientBody = grad,
                        readsPrimalIndices = setOf(0),
                    )
                    yields(c)
                },
            )
            listOf(result)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val outPos = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(3f)))
        assertEquals(1f, outPos[0][0])
        val outNeg = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(-2f)))
        assertEquals(-4f, outNeg[0][0], "d/dx (x*x) at x=-2 = 2*(-2) = -4")
    }

    @Test
    fun gradThroughCoarsenedInBothBranchesNumericallyMatchesUncoarsened() {
        // f(x) = if (x > 0) coarsened_sq(x) else coarsened_sq(x)   (both branches identical).
        // Equivalent to just f(x) = x*x; d/dx = 2x regardless of branch.
        // This verifies COARSENED in both branches compose correctly with handleIfAdjoint's
        // synthesized IF(pred, thenAdj, elseAdj) pattern.
        val boolS = io.tlaloc.ir.DxirType(io.tlaloc.core.Bool, emptyList())
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("both_coarsened") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val c = coarsened(
                        operands = listOf(x),
                        primalBody = primal,
                        gradientBody = grad,
                        readsPrimalIndices = setOf(0),
                    )
                    yields(c)
                },
                elseRegion = region {
                    val c = coarsened(
                        operands = listOf(x),
                        primalBody = primal,
                        gradientBody = grad,
                        readsPrimalIndices = setOf(0),
                    )
                    yields(c)
                },
            )
            listOf(result)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        for (xVal in listOf(-3f, -1f, 0.5f, 2f, 10f)) {
            val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(xVal)))
            val expected = 2f * xVal
            // At x=0 the STEP predicate is 0 → else-branch active → grad = 2*0 = 0.
            // Tolerance: exact.
            assertEquals(expected, out[0][0], "grad at x=$xVal must be 2*x=$expected, got ${out[0][0]}")
        }
    }

    @Test
    fun gradThroughCoarsenedInsideValueAndGradPath() {
        // `includeForward = true` (valueAndGrad mode): returns [forward_result, grad_0].
        // Primal = a*a; at a=3 → forward = 9, grad = 6.
        val primal = squarePrimal()
        val grad = squareGradient()
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            listOf(c)
        }
        val gradFn = DxirReverseTransform.apply(outer, includeForward = true)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(3f)))
        assertEquals(2, out.size, "valueAndGrad returns [forward, grad]")
        assertEquals(9f, out[0][0], "forward: 3*3 = 9")
        assertEquals(6f, out[1][0], "grad: 2*3 = 6")
    }

    // ------------------------------------------------------------------------
    // §0.4.179 — Phase 5c: multi-result COARSENED gradient via handleCoarsenedAdjoint
    // ------------------------------------------------------------------------

    @Test
    fun gradThroughMultiResultCoarsenedRoutesPerIndexUpstreams() {
        // primal_body: (a, b) → (a + b, a - b).
        //   ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        //   ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
        // gradient_body: (u_sum, u_diff, a, b) → (da, db) where
        //   da = u_sum + u_diff   (chain through both results' contribution to a)
        //   db = u_sum - u_diff   (chain through both results' contribution to b)
        // outer: f(a, b) = coarsened.result(0) + 2*coarsened.result(1)
        //                = (a + b) + 2*(a - b) = 3a - b
        // df/da = 3, df/db = -1.
        val primal = DxirBuilder.function("addsub_primal") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val sum = op(OpKind.ADD, listOf(a, b), f32s)
            val diff = op(OpKind.SUB, listOf(a, b), f32s)
            listOf(sum, diff)
        }
        val grad = DxirBuilder.function("addsub_grad") {
            val uSum = param("u_sum", f32s)
            val uDiff = param("u_diff", f32s)
            val a = param("a", f32s)
            val b = param("b", f32s)
            // dummy reads of a/b to keep ref-integrity.
            val _refA = op(OpKind.MUL, listOf(a, uSum), f32s)
            val _refB = op(OpKind.MUL, listOf(b, uSum), f32s)
            val da = op(OpKind.ADD, listOf(uSum, uDiff), f32s)
            val db = op(OpKind.SUB, listOf(uSum, uDiff), f32s)
            listOf(da, db)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0, 1),
            )
            // f = c.result(0) + 2 * c.result(1)
            val two = const(2f, f32s)
            val twoDiff = op(OpKind.MUL, listOf(two, c.result(1)), f32s)
            val r = op(OpKind.ADD, listOf(c.result(0), twoDiff), f32s)
            listOf(r)
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(5f), floatArrayOf(3f)))
        assertEquals(2, out.size, "grad fn has 2 returns (one per outer param)")
        assertEquals(3f, out[0][0], "df/da = 3 (1 from sum + 2*1 from diff)")
        assertEquals(-1f, out[1][0], "df/db = -1 (1 from sum + 2*(-1) from diff)")
    }

    @Test
    fun gradThroughMultiResultCoarsenedProductConsumerKeepsResultIndicesDistinct() {
        // §0.4.430 — regression for the result-index collapse the multi-result
        // JVP⇄VJP cross-identity exposed: z = c0·c1 makes the MUL adjoint emit
        // MUL(seed, %c#1) and MUL(seed, %c#0) — IDENTICAL (kind, operand-id,
        // attrs, types) signatures under applyCSE's old bare-id operand key,
        // since %c#0 and %c#1 share the source id. Pre-§0.4.430 the CSE merged
        // them (and applyConstFold's MUL-by-1 arm then collapsed the survivor
        // to the source op — result 0), so BOTH upstreams fed the gradient_body
        // the same value. The §0.4.179 tests never built two same-signature
        // ops over distinct result indices, which is how this survived.
        //   primal (a, b) → (a + b², a·b);  z = c0·c1.
        //   ∂z/∂a = c1 + c0·b;  ∂z/∂b = c1·2b + c0·a.
        val primal = DxirBuilder.function("mr_primal") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val bb = op(OpKind.MUL, listOf(b, b), f32s)
            listOf(op(OpKind.ADD, listOf(a, bb), f32s), op(OpKind.MUL, listOf(a, b), f32s))
        }
        val gradBody = DxirBuilder.function("mr_gradient") {
            val up0 = param("up0", f32s)
            val up1 = param("up1", f32s)
            val a = param("a", f32s)
            val b = param("b", f32s)
            val two = const(2f, f32s)
            val da = op(OpKind.ADD, listOf(up0, op(OpKind.MUL, listOf(up1, b), f32s)), f32s)
            val db = op(
                OpKind.ADD,
                listOf(
                    op(OpKind.MUL, listOf(up0, op(OpKind.MUL, listOf(two, b), f32s)), f32s),
                    op(OpKind.MUL, listOf(up1, a), f32s),
                ),
                f32s,
            )
            listOf(da, db)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = coarsened(
                operands = listOf(a, b),
                primalBody = primal,
                gradientBody = gradBody,
                readsPrimalIndices = setOf(0, 1),
            )
            listOf(op(OpKind.MUL, listOf(c.result(0), c.result(1)), f32s))
        }
        val gradFn = DxirReverseTransform.apply(outer)
        // Quarter-integer grid — exact in f32.
        for ((a, b) in listOf(1.5f to 0.5f, -0.75f to 2f, 0.25f to -1.5f)) {
            val c0 = a + b * b
            val c1 = a * b
            val out = DxirInterpreter.evalFunction(
                gradFn, listOf(floatArrayOf(a), floatArrayOf(b)),
            )
            assertEquals(c1 + c0 * b, out[0][0], "∂z/∂a at a=$a b=$b (upstream 0 vs 1 mixed?)")
            assertEquals(c1 * 2f * b + c0 * a, out[1][0], "∂z/∂b at a=$a b=$b")
        }
    }

    @Test
    fun gradThroughMultiResultCoarsenedDeadIndexSeedsZero() {
        // Verify Phase 5c's dead-index handling: when only ONE of K results is
        // consumed downstream, the other index gets seeded with const(0) inside
        // handleCoarsenedAdjoint. The gradient must still match the analytic
        // single-result-consumed case.
        //
        // primal_body: (a) → (a + 1, a * 2)
        // gradient_body: (u0, u1, a) → (u0 + 2 * u1)  (single operand: a)
        // outer: f(a) = c.result(0)  (uses sum only; result(1) is dead)
        //   f = a + 1, so df/da = 1.
        // The dead u1 should be seeded with const(0), giving da = u0 + 2*0 = u0 = 1.
        val primal = DxirBuilder.function("dead_idx_primal") {
            val a = param("a", f32s)
            val one = const(1f, f32s)
            val sum = op(OpKind.ADD, listOf(a, one), f32s)
            val two = const(2f, f32s)
            val doubled = op(OpKind.MUL, listOf(a, two), f32s)
            listOf(sum, doubled)
        }
        val grad = DxirBuilder.function("dead_idx_grad") {
            val u0 = param("u0", f32s)
            val u1 = param("u1", f32s)
            val a = param("a", f32s)
            val _refA = op(OpKind.MUL, listOf(a, u0), f32s)  // ref-integrity for a
            val two = const(2f, f32s)
            val twoU1 = op(OpKind.MUL, listOf(two, u1), f32s)
            val da = op(OpKind.ADD, listOf(u0, twoU1), f32s)
            listOf(da)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            // Only consume result(0).
            listOf(c.result(0))
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(7f)))
        assertEquals(1, out.size)
        assertEquals(1f, out[0][0], "df/da = 1 (u1 seeded with 0 since result(1) is dead)")
    }

    @Test
    fun gradThroughMultiResultCoarsenedDeadIndex0SeedsZero() {
        // §0.4.231 — mirror of [gradThroughMultiResultCoarsenedDeadIndexSeedsZero]
        // but with index 0 dead instead of index 1. Validates that Phase 5c's
        // dead-index detection is symmetric — neither dead-at-low-index nor
        // dead-at-high-index is special-cased.
        //
        // primal_body: (a) → (a + 1, a * 2)        (same as the index-1-dead test)
        // gradient_body: (u0, u1, a) → (u0 + 2 * u1)
        // outer: f(a) = c.result(1)                 // result(0) is dead
        //   f = a * 2, so df/da = 2.
        // The dead u0 should be seeded with const(0), giving da = 0 + 2 * u1
        //   = 0 + 2 * 1 = 2.
        val primal = DxirBuilder.function("dead_idx0_primal") {
            val a = param("a", f32s)
            val one = const(1f, f32s)
            val sum = op(OpKind.ADD, listOf(a, one), f32s)
            val two = const(2f, f32s)
            val doubled = op(OpKind.MUL, listOf(a, two), f32s)
            listOf(sum, doubled)
        }
        val grad = DxirBuilder.function("dead_idx0_grad") {
            val u0 = param("u0", f32s)
            val u1 = param("u1", f32s)
            val a = param("a", f32s)
            val _refA = op(OpKind.MUL, listOf(a, u0), f32s)  // ref-integrity for a
            val two = const(2f, f32s)
            val twoU1 = op(OpKind.MUL, listOf(two, u1), f32s)
            val da = op(OpKind.ADD, listOf(u0, twoU1), f32s)
            listOf(da)
        }
        val outer = DxirBuilder.function("f") {
            val a = param("a", f32s)
            val c = coarsened(
                operands = listOf(a),
                primalBody = primal,
                gradientBody = grad,
                readsPrimalIndices = setOf(0),
            )
            // Only consume result(1) — result(0) is dead.
            listOf(c.result(1))
        }
        val gradFn = DxirReverseTransform.apply(outer)
        val out = DxirInterpreter.evalFunction(gradFn, listOf(floatArrayOf(7f)))
        assertEquals(1, out.size)
        assertEquals(2f, out[0][0], "df/da = 2 (u0 seeded with 0 since result(0) is dead)")
    }
}
