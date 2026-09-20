package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.pretty
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * §0.4.407 — the IF direct forward arm (B3's recorded tail). The tangent of an
 * IF is a SECOND IF over the SAME condition (piecewise-constant — zero
 * tangent, value cloned) whose branches yield the tangents of the primal
 * branches' yields; branch bodies FLATTEN into the outer forward stream, the
 * unconditional-hoist trade `walkBranchReverse` has made since §0.4.23.
 *
 * Oracles, in the house pattern:
 *  1. **Central differences on the primal** (interpreter, no AD) on BOTH sides
 *     of the branch point.
 *  2. **The JVP⇄VJP cross-identity** against [DxirReverseTransform]'s existing
 *     IF support (`handleIfAdjoint`, §0.4.23) — the strongest oracle here:
 *     both transforms walk the SAME IF body through entirely different
 *     mechanisms, and `⟨∇f(x), v⟩ == jvp_f(x, v).tangent` ties them to one
 *     number.
 *  3. **Analytic hand pins** per branch.
 *
 * Multi-result IFs are SUPPORTED (the tangent IF mirrors the primal's index
 * layout one-to-one), certified against the reverse side's multi-live-index
 * walk (§0.4.155). The forward-over-reverse composition through an IF-bearing
 * gradient body — the exact shape `hessian {}` composes — is pinned too:
 * before this slice the forward transform refused every IF, including the
 * empty-region adjoint IFs `handleIfAdjoint` itself emits.
 */
class DxirForwardIfTest {

    private val f32 = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    /** f(x) = if (x > 0) x*x else -x — branch bodies with real ops (the raw
     * FIR-lowered shape, pre-lift). */
    private fun ifSquareElseNeg(): DxirFunction = DxirBuilder.function("ifSquareElseNeg") {
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

    /** f(x) = if (x > 0) x*x + x else -(x*x*x) — multiple ops per branch. */
    private fun ifPolyElseCubic(): DxirFunction = DxirBuilder.function("ifPolyElseCubic") {
        val x = param("x", f32)
        val pred = op(OpKind.STEP, listOf(x), boolS)
        val ifResult = ifOp(
            cond = pred,
            types = listOf(f32),
            thenRegion = region {
                val sq = op(OpKind.MUL, listOf(x, x), f32)
                val sum = op(OpKind.ADD, listOf(sq, x), f32)
                yields(sum)
            },
            elseRegion = region {
                val sq = op(OpKind.MUL, listOf(x, x), f32)
                val cube = op(OpKind.MUL, listOf(sq, x), f32)
                val neg = op(OpKind.NEG, listOf(cube), f32)
                yields(neg)
            },
        )
        listOf(ifResult)
    }

    @Test
    fun jvpThroughIfMatchesAnalyticOnBothSidesOfTheBranch() {
        // f(x) = if (x > 0) x*x else -x;  df = if (x > 0) 2x·dx else -dx.
        val jvp = DxirForwardTransform.apply(ifSquareElseNeg())
        assertEquals(2, jvp.params.size, "params + tangent params")
        assertEquals(2, jvp.returns.size, "primal value + tangent")

        // x = 2 (then branch): y = 4, dy = 4·dx.
        val outPos = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(2f), floatArrayOf(1f)))
        assertTrue(abs(outPos[0][0] - 4f) < 1e-4f, "value stream at x=2: got ${outPos[0][0]}")
        assertTrue(abs(outPos[1][0] - 4f) < 1e-4f, "tangent at x=2, dx=1: got ${outPos[1][0]}")

        // Seed scaling is linear: dx = -2 → dy = -8.
        val outScaled = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(2f), floatArrayOf(-2f)))
        assertTrue(abs(outScaled[1][0] + 8f) < 1e-4f, "tangent at x=2, dx=-2: got ${outScaled[1][0]}")

        // x = -3 (else branch): y = 3, dy = -dx.
        val outNeg = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(-3f), floatArrayOf(1f)))
        assertTrue(abs(outNeg[0][0] - 3f) < 1e-4f, "value stream at x=-3: got ${outNeg[0][0]}")
        assertTrue(abs(outNeg[1][0] + 1f) < 1e-4f, "tangent at x=-3, dx=1: got ${outNeg[1][0]}")
    }

    @Test
    fun jvpThroughIfMatchesCentralDifferencesOnBothSides() {
        val fn = ifPolyElseCubic()
        val jvp = DxirForwardTransform.apply(fn)
        val eps = 1e-3f
        for (x in listOf(1.7f, 0.9f, -0.7f, -2.1f)) {
            val tangent = DxirInterpreter.evalFunction(
                jvp, listOf(floatArrayOf(x), floatArrayOf(1f)),
            )[1][0]
            fun evalAt(v: Float): Float =
                DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(v))).single().single()
            val fd = (evalAt(x + eps) - evalAt(x - eps)) / (2 * eps)
            assertTrue(
                abs(tangent - fd) <= 2e-2f * maxOf(1f, abs(fd)),
                "at x=$x: jvp tangent $tangent vs central differences $fd",
            )
        }
    }

    @Test
    fun jvpVjpCrossIdentityThroughIf() {
        // ⟨∇f(x), v⟩ == jvp_f(x, v).tangent — the reverse transform's §0.4.23
        // branch reverse walk (`handleIfAdjoint`) against the forward arm, on
        // the SAME IF body: two entirely different derivative encodings, one
        // number. Both sides analytic — tight tolerance.
        for (fn in listOf(ifSquareElseNeg(), ifPolyElseCubic())) {
            val jvp = DxirForwardTransform.apply(fn)
            val grad = DxirReverseTransform.apply(fn)
            for (x in listOf(2.3f, 0.6f, -0.4f, -1.9f)) {
                for (v in listOf(1f, -0.5f)) {
                    val tangent = DxirInterpreter.evalFunction(
                        jvp, listOf(floatArrayOf(x), floatArrayOf(v)),
                    )[1][0]
                    val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))[0][0]
                    assertTrue(
                        abs(tangent - g * v) <= 1e-5f * maxOf(1f, abs(g * v)),
                        "${fn.name} at x=$x, v=$v: jvp=$tangent vs ⟨∇f,v⟩=${g * v}",
                    )
                }
            }
        }
    }

    @Test
    fun jvpThroughLiftedIfEmitsAValueIfAndATangentIf() {
        // The post-lift shape (§0.4.174): empty-region IF yielding outer-scope
        // values only — f(x) = if (x > 0) x else -x, i.e. abs. The tangent
        // stream is a SECOND IF over the same condition: exactly two IF ops in
        // the jvp body, both with empty-body regions (the only IF shape
        // DxirToIrSynthesis.irIfOp and the emitter accept).
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
        val jvp = DxirForwardTransform.apply(primal)

        val ifOps = jvp.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.IF }
        assertEquals(2, ifOps.size, "value IF + tangent IF, no more:\n${jvp.pretty()}")
        for (node in ifOps) {
            assertTrue(
                node.regions.all { r -> r.blocks.single().body.isEmpty() },
                "forward-arm IFs must have empty-body regions (yield-only)",
            )
        }

        // dy = sign(x)·dx.
        for ((x, want) in listOf(2f to 1f, -3f to -1f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertTrue(abs(out[1][0] - want) < 1e-6f, "at x=$x: dy=${out[1][0]} want $want")
        }
    }

    @Test
    fun jvpThroughIfWithUnsafeBranchOpStaysFiniteOnTheOtherSide() {
        // f(x) = if (x > 0) sqrt(x) else -x. The SQRT is outside
        // PhiCalculus.SAFE_LIFT_OPS, so the lift pass leaves it in the region —
        // this is the shape that reaches the arm with a body op the hoist must
        // flatten. Flattening evaluates sqrt on BOTH sides (the walkBranchReverse
        // trade); at x < 0 the hoisted sqrt is NaN but the IFs select the else
        // yields, so value and tangent stay finite.
        val primal = DxirBuilder.function("ifSqrtElseNeg") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32),
                thenRegion = region {
                    val root = op(OpKind.SQRT, listOf(x), f32)
                    yields(root)
                },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32)
                    yields(neg)
                },
            )
            listOf(ifResult)
        }
        val jvp = DxirForwardTransform.apply(primal)

        // x = 4: y = 2, dy = dx / (2·√x) = 0.25.
        val outPos = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(4f), floatArrayOf(1f)))
        assertTrue(abs(outPos[0][0] - 2f) < 1e-5f, "value at x=4: got ${outPos[0][0]}")
        assertTrue(abs(outPos[1][0] - 0.25f) < 1e-5f, "tangent at x=4: got ${outPos[1][0]}")

        // x = -1: y = 1, dy = -1 — and NOT NaN despite the hoisted sqrt(-1).
        val outNeg = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(-1f), floatArrayOf(1f)))
        assertFalse(outNeg[0][0].isNaN(), "value at x=-1 leaked the discarded branch's NaN")
        assertFalse(outNeg[1][0].isNaN(), "tangent at x=-1 leaked the discarded branch's NaN")
        assertTrue(abs(outNeg[0][0] - 1f) < 1e-6f, "value at x=-1: got ${outNeg[0][0]}")
        assertTrue(abs(outNeg[1][0] + 1f) < 1e-6f, "tangent at x=-1: got ${outNeg[1][0]}")
    }

    @Test
    fun jvpThroughNestedIfRecurses() {
        // f(x) = if (x > 0) { if (x > 1) x*x else x } else -x.
        // df   = if (x > 0) { if (x > 1) 2x  else 1 } else -1.
        val primal = DxirBuilder.function("nestedIf") {
            val x = param("x", f32)
            val predOuter = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = predOuter,
                types = listOf(f32),
                thenRegion = region {
                    val one = const(1.0f, f32)
                    val shifted = op(OpKind.SUB, listOf(x, one), f32)
                    val predInner = op(OpKind.STEP, listOf(shifted), boolS)
                    val inner = ifOp(
                        cond = predInner,
                        types = listOf(f32),
                        thenRegion = region {
                            val sq = op(OpKind.MUL, listOf(x, x), f32)
                            yields(sq)
                        },
                        elseRegion = region { yields(x) },
                    )
                    yields(inner)
                },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32)
                    yields(neg)
                },
            )
            listOf(ifResult)
        }
        val jvp = DxirForwardTransform.apply(primal)
        for ((x, want) in listOf(2f to 4f, 0.5f to 1f, -3f to -1f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertTrue(abs(out[1][0] - want) < 1e-4f, "at x=$x: dy=${out[1][0]} want $want")
        }
    }

    @Test
    fun jvpThroughMultiResultIfMatchesReverseMultiLiveWalk() {
        // §0.4.155's multi-live-index shape, forward:
        // f(x) = ifop.result(0) + ifop.result(1)
        //   where ifop = if (x > 0) yields(-x, x²) else yields(x, x²).
        // x > 0: f = -x + x², df = -1 + 2x.  x ≤ 0: f = x + x², df = 1 + 2x.
        // The tangent IF mirrors the primal's index layout one-to-one; the
        // reverse transform's multi-live walk is the cross-oracle.
        fun buildPrimal(): DxirFunction = DxirBuilder.function("ifMultiLiveFwd") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val xSquared = op(OpKind.MUL, listOf(x, x), f32)
            val mrIf = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(negX, xSquared) },
                    region { yields(x, xSquared) },
                ),
            )
            listOf(op(OpKind.ADD, listOf(mrIf.result(0), mrIf.result(1)), f32))
        }
        val jvp = DxirForwardTransform.apply(buildPrimal())
        val grad = DxirReverseTransform.apply(buildPrimal())
        for ((x, want) in listOf(2f to 3f, 0.5f to 0f, -3f to -5f)) {
            val out = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertTrue(abs(out[1][0] - want) < 1e-4f, "at x=$x: dy=${out[1][0]} want $want")
            val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(x)))[0][0]
            assertTrue(
                abs(out[1][0] - g) < 1e-5f,
                "at x=$x: forward ${out[1][0]} vs reverse multi-live $g",
            )
        }
        // Value stream sanity at x=2: -2 + 4 = 2.
        val v = DxirInterpreter.evalFunction(jvp, listOf(floatArrayOf(2f), floatArrayOf(1f)))[0][0]
        assertTrue(abs(v - 2f) < 1e-5f, "value stream at x=2: got $v")
    }

    @Test
    fun forwardOverReverseComposesThroughAnIfAdjointBody() {
        // The hessian composition (forward-over-reverse) through an IF body:
        // R(ifSquareElseNeg) is a gradient function whose body carries the
        // empty-region adjoint IF `handleIfAdjoint` emits — before §0.4.407 the
        // forward transform refused it wholesale. Now:
        //   f''(x) = if (x > 0) 2 else 0.
        val grad = DxirReverseTransform.apply(ifSquareElseNeg())
        val hvp = DxirForwardTransform.apply(grad)
        for ((x, want) in listOf(2f to 2f, 7f to 2f, -3f to 0f)) {
            val out = DxirInterpreter.evalFunction(hvp, listOf(floatArrayOf(x), floatArrayOf(1f)))
            assertTrue(abs(out[1][0] - want) < 1e-5f, "H at x=$x: got ${out[1][0]} want $want")
        }
    }
}
