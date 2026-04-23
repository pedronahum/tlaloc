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
        // Gradient body: (upstream, a, b) → (da). Just returns upstream for C.3b.1 shape
        // validity; semantic correctness isn't exercised in interpreter tests.
        val grad = DxirBuilder.function("addsub_grad") {
            val upstream = param("upstream", f32s)
            val a = param("a", f32s)
            val b = param("b", f32s)
            val _ign = op(OpKind.ADD, listOf(a, b), f32s)
            val _ign2 = op(OpKind.ADD, listOf(_ign, upstream), f32s)
            listOf(upstream, upstream)
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
}
