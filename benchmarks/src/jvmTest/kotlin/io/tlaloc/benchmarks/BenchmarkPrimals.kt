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
}
