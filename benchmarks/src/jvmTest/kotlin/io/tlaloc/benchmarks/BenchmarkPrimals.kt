package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.148 — shared primal builders for the `:benchmarks` module's probes.
 * Extracted from §0.4.145–§0.4.147's three benchmarks (EndToEndAdBenchmark,
 * CoarseningThroughputBenchmark, SctThroughputBenchmark) which had the same
 * `iterateNTimes` helper duplicated three times. With consolidation, each
 * probe just calls [iterateNTimes] from this object.
 *
 * Helpers stay narrowly scoped: only primal builders and structural counters
 * common to MULTIPLE benchmarks belong here. Benchmark-specific construction
 * (e.g., a tracer-surface fixture, a shape that exercises a niche code path)
 * stays in the benchmark's own file. The "stay structurally close" principle
 * documented in §0.4.145 / §0.4.146 / §0.4.147 still applies — sharing widens
 * only when the duplication is observable across three or more callers.
 */
object BenchmarkPrimals {

    val f32 = DxirType(F32, emptyList())
    val i32 = DxirType(I32, emptyList())
    val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    /**
     * Build the canonical `iterateNTimes` primal: `f(x) = x · 2^n` via an
     * n-iteration WHILE loop whose body doubles the carried `x` each iter
     * and increments a counter. After [io.tlaloc.ir.passes.PhiCalculus.apply]
     * unrolls C5, the body has exactly n MUL ops and zero WHILEs (pinned by
     * §0.4.146's [CoarseningThroughputBenchmark]); after
     * [io.tlaloc.ir.passes.DxirReverseTransform.apply], the gradient evaluates
     * to `2^n` at any x (pinned by §0.4.147's [SctThroughputBenchmark]).
     */
    fun iterateNTimes(n: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("iterate$n") {
            val x = param("x", f32)
            val nConst = const(n, i32)
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
            listOf(w.result(0))
        }

    /** Count top-level body ops of a given [kind] in [fn]. */
    fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /**
     * §0.4.156 — multi-branch IF primal that exercises the §0.4.155 multi-live-index
     * MR IF AD path. Shape:
     * ```
     * f(x) = let r = if (x > 0) {x*x, 2*x} else {-x, x}
     *        in  r.result(0) + r.result(1)
     * ```
     * For `x > 0`: `r = (x², 2x)`, `sum = x² + 2x`, `df/dx = 2x + 2`.
     * For `x ≤ 0`: `r = (-x, x)`, `sum = -x + x = 0`, `df/dx = -1 + 1 = 0`.
     *
     * Both `r.result(0)` and `r.result(1)` are referenced downstream — pre-§0.4.155
     * this shape was hard-rejected by `findIfLiveResultIndex.singleOrNull()`. With
     * Phase 5b's per-index gradAccum dispatch, the primal flows correctly through
     * SCT and the gradient closes in expected per-branch closed forms.
     */
    fun multiBranchIfPrimal(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("multiBranchIf") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val xSquared = op(OpKind.MUL, listOf(x, x), f32)
            val two = const(2f, f32)
            val twoX = op(OpKind.MUL, listOf(two, x), f32)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val ifOp = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(xSquared, twoX) },
                    region { yields(negX, x) },
                ),
            )
            val sum = op(OpKind.ADD, listOf(ifOp.result(0), ifOp.result(1)), f32)
            listOf(sum)
        }

    /**
     * §0.4.223 — BGDHyperOpt outer-loop primal, mirrors `PhiCalculus.bgdHyperOptPrimal`
     * in [PhiCalculusBgdHyperOptTest][io.tlaloc.ir.passes]. Duplicated here because
     * `:benchmarks` cannot reach back into `:ir`'s test source set; the duplication is
     * intentional and ~30 lines. Future consolidation: lift to a `:ir`/`:core`-side
     * commonTest helper or expose the primal builder publicly from `:ir`.
     *
     * The signature is `(r, Sxy, Sx2, M) → w_final` with concrete `K` baked in.
     * Mirrors the OOPSLA 2021 paper's Fig. 6a structure (lines 3-11) with the
     * inner-for-loop d-computation pre-collapsed to scalar sums.
     *
     * The recurrence is `w_{k+1} = a · w_k + b` where:
     *   - `a = 1 + 2r·Sx2/M`
     *   - `b = -2r·Sxy/M`
     *
     * **Coarsening behaviour** (without a [io.tlaloc.ir.passes.SymbolicEngine]):
     *   - C5 unrolls the constant-trip-count outer WHILE → flat ADD/MUL chain.
     *   - With a SymjaEngine attached (`:ir`-test-internal), C6 produces the
     *     closed-form `w_K = b · (a^K − 1)/(a − 1)` (O(1) ops vs C5's O(K) chain).
     *   - The harness path uses the no-engine variant — closer to what the K2
     *     plugin emits today.
     *
     * **Closed-form Kotlin reference** is provided as [bgdHyperOptOuterLoopReference]
     * for FD validation in tests.
     */
    fun bgdHyperOptOuterLoopPrimal(K: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("bgdOuterLoop") {
            val r = param("r", f32)
            val Sxy = param("Sxy", f32)
            val Sx2 = param("Sx2", f32)
            val M = param("M", f32)
            val wInit = const(0f, f32)
            val kBound = const(K.toFloat(), f32)
            val kZero = const(0f, f32)
            val w = whileOp(
                inits = listOf(wInit, kZero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(kBound, args[1]), f32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val carriedW = args[0]
                    val counterArg = args[1]
                    val one = const(1f, f32)
                    val two = const(2f, f32)
                    val twoR = op(OpKind.MUL, listOf(two, r), f32)
                    val twoRSx2 = op(OpKind.MUL, listOf(twoR, Sx2), f32)
                    val twoRSx2OverM = op(OpKind.DIV, listOf(twoRSx2, M), f32)
                    val aExpr = op(OpKind.ADD, listOf(one, twoRSx2OverM), f32)
                    val twoRSxy = op(OpKind.MUL, listOf(twoR, Sxy), f32)
                    val twoRSxyOverM = op(OpKind.DIV, listOf(twoRSxy, M), f32)
                    val bExpr = op(OpKind.NEG, listOf(twoRSxyOverM), f32)
                    val aw = op(OpKind.MUL, listOf(aExpr, carriedW), f32)
                    val newW = op(OpKind.ADD, listOf(aw, bExpr), f32)
                    val counterIncr = const(1f, f32)
                    val newK = op(OpKind.ADD, listOf(counterArg, counterIncr), f32)
                    yields(newW, newK)
                },
            )
            listOf(w.result(0))
        }

    /**
     * §0.4.223 — Closed-form Kotlin reference for the BGDHyperOpt outer loop.
     * Used by the harness baseline test for FD-validated gradient pins.
     */
    fun bgdHyperOptOuterLoopReference(r: Float, Sxy: Float, Sx2: Float, M: Float, K: Int): Float {
        val a = 1f + 2f * r * Sx2 / M
        val b = -2f * r * Sxy / M
        var w = 0f
        for (k in 0 until K) w = a * w + b
        return w
    }
}
