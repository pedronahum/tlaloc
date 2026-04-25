package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * **D.1i Phase 3b (§0.4.106)** — IR-size delta harness for `PhiCalculus.simplifyReturns`.
 * Each test builds a representative scalar primal, runs `DxirReverseTransform.apply` to
 * get a realistic gradient `DxirFunction`, then pins `body.size` both before and after
 * `simplifyReturns`. The before-numbers are the baseline gradient body the IR pipeline
 * would otherwise hand to synthesis; the after-numbers are what `tlaloc.simplify.enabled
 * =true` produces.
 *
 * **Empirical finding (§0.4.106).** On the small straight-line scalar primals exercised
 * here, the reverse rules in `Vjp.kt` already produce tightly-folded gradient bodies
 * (e.g., `d/dx[x*x]` emits `add(x, x)` directly, not `1*x + x*1`). Symja Simplify on
 * top of that fires meaningful semantic rewrites — `add(x, x) → 2*x`, `add(add(a,b),b)
 * → a + 2*b`, `pow` introduction for repeated multiplications — but the resulting
 * `body.size` is similar or sometimes slightly larger because Symja prefers explicit
 * `2 * x` (2 nodes: const + mul) over `add(x, x)` (1 node, no const). For the simple
 * cases below the delta is zero ± 1 node. The harness's value is therefore (a) a
 * regression detector against future widening of either the reverse rules or Symja's
 * surface, and (b) a numerical-correctness pin showing simplify never CHANGES the
 * mathematical result. Larger gains will surface when Phase 4's opaque-leaf widening
 * exposes tensor reductions and gather chains to Simplify.
 *
 * The tests deliberately use straight-line scalar primals (no IF / WHILE / coarsening)
 * because that is the surface `simplifyReturns` covers in Phases 1/2.
 */
class PhiCalculusSimplifyIrSizeTest {

    private val engine = SymjaEngine()
    private val f32 = DxirType(F32, emptyList())

    private data class SizeDelta(val pre: Int, val post: Int)

    private fun measure(primal: io.tlaloc.ir.DxirFunction): Pair<SizeDelta, io.tlaloc.ir.DxirFunction> {
        val grad = DxirReverseTransform.apply(primal)
        val simplified = PhiCalculus.simplifyReturns(grad, engine)
        return SizeDelta(grad.body.size, simplified.body.size) to simplified
    }

    @Test
    fun simplifyOfSquareGradient() {
        // f(x) = x*x. Reverse rule emits `dx = add(x, x)` (1 node body — the ADD;
        // the params don't count). Simplify rewrites to `2 * x`, which lowers as
        // `const 2.0` + `mul(2, x)` = 2 nodes. So the body grows by 1 in raw node
        // count, but the operation count (ops only, ignoring consts) is 1 in both
        // cases. We pin the observed numbers.
        val primal = DxirBuilder.function("square") {
            val x = param("x", f32)
            val out = op(OpKind.MUL, listOf(x, x), f32)
            listOf(out)
        }
        val (d, simplified) = measure(primal)
        assertEquals(1, d.pre, "pre-simplify body size for d/dx(x*x)")
        assertEquals(2, d.post, "post-simplify body size for d/dx(x*x)")
        // Numerical correctness: at x=3, derivative is 6.0.
        assertEquals(6f, DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(3f))).single()[0])
    }

    @Test
    fun simplifyOfCubeGradient() {
        // f(x) = x*x*x. Reverse expands into 4 nodes (two `x*x` mults plus two
        // adds collecting `2*x*x + x*x = 3*x*x`). Simplify recognises the cube and
        // emits `3 * x^2` (const 3, const 2, pow(x,2), mul) = 4 nodes. Same size,
        // but the post form uses POW instead of repeated MUL — semantically cleaner
        // for downstream symbolic passes (less duplicated subexpression).
        val primal = DxirBuilder.function("cube") {
            val x = param("x", f32)
            val sq = op(OpKind.MUL, listOf(x, x), f32)
            val out = op(OpKind.MUL, listOf(sq, x), f32)
            listOf(out)
        }
        val (d, simplified) = measure(primal)
        assertEquals(4, d.pre, "pre-simplify body size for d/dx(x^3)")
        assertEquals(4, d.post, "post-simplify body size for d/dx(x^3)")
        // Numerical correctness: at x=2, derivative of x^3 is 12.
        assertEquals(12f, DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(2f))).single()[0])
    }

    @Test
    fun simplifyOfPolynomialGradient() {
        // f(x) = (x+1)^2 + 2x. d/dx = 2(x+1) + 2 = 2x + 4. The reverse rules
        // emit a 5-node body that includes the `(x+1)` re-use. Simplify folds
        // it to `2 * (2 + x)` — 4 nodes (two const-2s + ADD + MUL). One-node
        // shrink. The two const-2s are not de-duped because PhiCalculus.simplifyReturns
        // doesn't run CSE over the lowered body — that's intentional, the existing
        // post-pipeline CSE step (§0.4.48) handles cross-function CSE.
        val primal = DxirBuilder.function("poly") {
            val x = param("x", f32)
            val one = const(1L, f32)
            val two = const(2L, f32)
            val xp1 = op(OpKind.ADD, listOf(x, one), f32)
            val sq = op(OpKind.MUL, listOf(xp1, xp1), f32)
            val twox = op(OpKind.MUL, listOf(two, x), f32)
            val out = op(OpKind.ADD, listOf(sq, twox), f32)
            listOf(out)
        }
        val (d, simplified) = measure(primal)
        assertEquals(5, d.pre, "pre-simplify body size for d/dx((x+1)^2 + 2x)")
        assertEquals(4, d.post, "post-simplify body size for d/dx((x+1)^2 + 2x)")
        assertTrue(d.post < d.pre, "simplify should shrink the polynomial gradient body")
        // Numerical correctness: at x=3, gradient = 2*4 + 2 = 10.
        assertEquals(10f, DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(3f))).single()[0])
    }

    @Test
    fun simplifyOfFractionalConstantGradient() {
        // f(x) = 0.5 * x. Reverse emits `dx = const 0.5` (1 node — the constant
        // is the gradient). Simplify is a no-op on a bare constant. Phase 2 fix
        // (§0.4.104) ensures the 0.5f doesn't get truncated to 0L during lift —
        // pre-fix this would have been `const 0.0` returning the wrong gradient.
        val primal = DxirBuilder.function("halfx") {
            val x = param("x", f32)
            val half = const(0.5f, f32)
            val out = op(OpKind.MUL, listOf(half, x), f32)
            listOf(out)
        }
        val (d, simplified) = measure(primal)
        assertEquals(1, d.pre, "pre-simplify body size for d/dx(0.5 * x)")
        assertEquals(1, d.post, "post-simplify body size for d/dx(0.5 * x)")
        // Numerical correctness: derivative is 0.5 at any x.
        assertEquals(0.5f, DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(7f))).single()[0])
    }

    @Test
    fun simplifyPreservesIdentityGradient() {
        // f(x) = x. Reverse seed is just `const 1`. Simplify must not change
        // the body. Pins the trivial case as a "do no harm" baseline.
        val primal = DxirBuilder.function("identity") {
            val x = param("x", f32)
            listOf(x)
        }
        val (d, simplified) = measure(primal)
        assertEquals(d.pre, d.post, "trivial identity gradient must be size-stable through simplify")
        // Numerical correctness: derivative of x w.r.t. x is 1.
        assertEquals(1f, DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(42f))).single()[0])
    }

    @Test
    fun simplifyAgreesWithUnsimplifiedNumerically() {
        // Bundle correctness: for each primal in this harness, the simplified
        // gradient must produce the same numerical answer as the un-simplified
        // gradient at a sample input. Catches any silent mis-simplification —
        // the most dangerous failure mode for this pass.
        val cases = listOf(
            Triple(
                DxirBuilder.function("c1") {
                    val x = param("x", f32)
                    val out = op(OpKind.MUL, listOf(x, x), f32)
                    listOf(out)
                },
                3f, 6f,
            ),
            Triple(
                DxirBuilder.function("c2") {
                    val x = param("x", f32)
                    val sq = op(OpKind.MUL, listOf(x, x), f32)
                    val out = op(OpKind.MUL, listOf(sq, x), f32)
                    listOf(out)
                },
                2f, 12f,
            ),
            Triple(
                DxirBuilder.function("c3") {
                    val x = param("x", f32)
                    val one = const(1L, f32)
                    val two = const(2L, f32)
                    val xp1 = op(OpKind.ADD, listOf(x, one), f32)
                    val sq = op(OpKind.MUL, listOf(xp1, xp1), f32)
                    val twox = op(OpKind.MUL, listOf(two, x), f32)
                    val out = op(OpKind.ADD, listOf(sq, twox), f32)
                    listOf(out)
                },
                3f, 10f,
            ),
        )
        for ((primal, xVal, expected) in cases) {
            val grad = DxirReverseTransform.apply(primal)
            val simplified = PhiCalculus.simplifyReturns(grad, engine)
            val gradVal = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(xVal))).single()[0]
            val simplifiedVal = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(xVal))).single()[0]
            assertEquals(expected, gradVal, "${primal.name}: un-simplified gradient mismatch at x=$xVal")
            assertEquals(expected, simplifiedVal, "${primal.name}: simplified gradient mismatch at x=$xVal")
        }
    }
}
