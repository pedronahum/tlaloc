package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.36 — Stage C.4 empirical `L` tuning sweeps. Each test runs a benchmark primal
 * through [LProfiler.sweep] at `L ∈ {25, 50, 100, 200}` + a tiny `L = 5` data point,
 * checks the coarsening behaviour at each size, and logs the metrics (visible in
 * test output) so the default can be informed by data rather than paper-convention.
 *
 * The benchmark suite is intentionally small + scalar (no tensors yet, no loops via
 * raw `DxirBuilder.whileOp` in this test file since those exercise C5-C9 behaviour
 * that's separately covered). Each primal targets a different part of the `L`
 * spectrum:
 *  - `tinyScalar` — 3 ops, any L succeeds.
 *  - `mediumChain` — 16 ops, differentiates L=5 (too small) from L ≥ 50.
 *  - `ifBranches` — IF with non-empty branches, exercises multi-SOI splicing.
 */
class LSweepTest {

    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()
    private val sweepLimits = listOf(5, 25, 50, 100, 200)

    @Test
    fun sweepTinyScalarConfirmsAllLValuesCoarsen() {
        // f(x) = (x + 1) * (x + 2).  3 ops. Any L ≥ 3 should coarsen.
        val fn = DxirBuilder.function("tiny") {
            val x = param("x", f32s)
            val one = const(1f, f32s)
            val two = const(2f, f32s)
            val xp1 = op(OpKind.ADD, listOf(x, one), f32s)
            val xp2 = op(OpKind.ADD, listOf(x, two), f32s)
            val r = op(OpKind.MUL, listOf(xp1, xp2), f32s)
            listOf(r)
        }
        val metrics = LProfiler.sweep(fn, engine, sweepLimits)
        logSweep("tinyScalar", metrics)

        // At L=5 the root subtree (5 ops) exceeds L — coarsenFunction falls through
        // because the root is a leaf (no regions) → coarsenRootLeaf runs `apply`
        // which doesn't care about L, so coarsening still fires. The `sizeLimit`
        // parameter is a no-op for root-leaf coarsening today.
        assertTrue(metrics.all { it.coarseningFired }, "root-leaf coarsening is L-independent today")
        // All L values should produce 1 top-level op (the single COARSENED).
        assertTrue(metrics.all { it.coarsenedTopLevelOpCount == 1 })
    }

    @Test
    fun sweepMediumChainHasStableMetrics() {
        // f(x) = chained ADDs + MULs. 16 body ops (8 chained op pairs). Tests that
        // coarsening + grad gen both produce stable results across L values.
        val fn = DxirBuilder.function("medium") {
            val x = param("x", f32s)
            val c1 = const(1f, f32s)
            val c2 = const(2f, f32s)
            var cur: DxirNode = x
            repeat(7) {
                cur = op(OpKind.ADD, listOf(cur, c1), f32s)
                cur = op(OpKind.MUL, listOf(cur, c2), f32s)
            }
            listOf(cur)
        }
        val metrics = LProfiler.sweep(fn, engine, sweepLimits)
        logSweep("mediumChain", metrics)

        // Root-leaf coarsening always fires (L-independent for root-leaf).
        assertTrue(metrics.all { it.coarseningFired })
        // Grad op count should be stable across L values (same computation, same L
        // doesn't affect root-leaf path's output).
        val gradCounts = metrics.map { it.gradOpCount }.toSet()
        assertEquals(1, gradCounts.size, "root-leaf grad count is L-independent; got variation: $gradCounts")
    }

    @Test
    fun sweepIfBranchesShowsMultiSoiPathKicksInAtSmallL() {
        // f(x) = if (x > 0) x*x + x else -x - 1.
        // Root has STEP + IF. Then-branch subtreeSize=3, else-branch subtreeSize=3.
        // Root total = 2 (STEP + IF) + 3 (then) + 3 (else) = 8.
        //
        // At L = 5: root (8 > 5) marked large → children promoted as SOIs →
        //           coarsenMultiSoi fires → IF branches each replaced by a COARSENED.
        // At L = 10: root (8 ≤ 10) not marked large → root is the only SOI → but since
        //            root is NOT a leaf (has IF children), coarsenFunction's dispatch
        //            routes to coarsenMultiSoi. No branch SOIs there either (children
        //            aren't marked large). Result: fn returned unchanged (no replacements).
        val fn = DxirBuilder.function("ifSplit") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val result = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val sq = op(OpKind.MUL, listOf(x, x), f32s)
                    val r = op(OpKind.ADD, listOf(sq, x), f32s)
                    yields(r)
                },
                elseRegion = region {
                    val neg = op(OpKind.NEG, listOf(x), f32s)
                    val one = const(1f, f32s)
                    val r = op(OpKind.SUB, listOf(neg, one), f32s)
                    yields(r)
                },
            )
            listOf(result)
        }
        val metrics = LProfiler.sweep(fn, engine, sweepLimits)
        logSweep("ifBranches", metrics)

        // At small L, branches get promoted as SOIs → COARSENED in each branch.
        val smallL = metrics.first { it.sizeLimit == 5 }
        assertTrue(smallL.coarseningFired, "L=5 should trigger multi-SOI branch coarsening")

        // At larger L, branches aren't marked large + root isn't either → no SOIs fire.
        // With root NOT a leaf + no branch SOIs, coarsenFunction returns fn unchanged.
        val largeL = metrics.first { it.sizeLimit == 200 }
        assertTrue(!largeL.coarseningFired, "L=200: root not large, branches not large → no coarsening")
    }

    @Test
    fun sweepProducesCorrectGradientAtAllL() {
        // f(x) = x*x. Simple case. Regardless of L, the gradient should be 2x.
        val fn = DxirBuilder.function("sq") {
            val x = param("x", f32s)
            val r = op(OpKind.MUL, listOf(x, x), f32s)
            listOf(r)
        }
        val metrics = LProfiler.sweep(fn, engine, sweepLimits)
        logSweep("sq", metrics)

        // At every L, eval of the grad should produce 2x. Use DxirInterpreter.
        for (L in sweepLimits) {
            val coarsened = PhiCalculus.coarsenFunction(fn, engine, sizeLimit = L)
            val grad = DxirReverseTransform.apply(coarsened)
            val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))
            assertEquals(6f, out[0][0], "d/dx (x*x) at x=3 must be 6 at L=$L")
        }
    }

    /**
     * Emit a compact human-readable table of the sweep metrics. Not assertion-
     * bearing — shows up in Gradle test output via `println`, captured for human
     * review when tuning `L`.
     */
    private fun logSweep(benchmark: String, metrics: List<LProfiler.SweepMetric>) {
        val header = "L     | fired | primal_ops(top/total) | grad_ops | coarsen_ms | grad_ms"
        println("[LSweep/$benchmark] $header")
        for (m in metrics) {
            val primal = "${m.coarsenedTopLevelOpCount}/${m.coarsenedTotalOpCount}"
            println(
                "[LSweep/$benchmark] " +
                    m.sizeLimit.toString().padEnd(6) + "| " +
                    (if (m.coarseningFired) "yes" else "no ").padEnd(6) + "| " +
                    primal.padEnd(22) + "| " +
                    m.gradOpCount.toString().padEnd(9) + "| " +
                    m.coarsenTimeMs.toString().padEnd(11) + "| " +
                    m.gradTimeMs,
            )
        }
    }
}
