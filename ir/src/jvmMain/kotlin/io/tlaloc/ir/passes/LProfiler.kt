package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * §0.4.36 — Stage C.4 empirical `L` tuning harness. Runs a [DxirFunction] through
 * [PhiCalculus.coarsenFunction] + [DxirReverseTransform.apply] at a range of size
 * limits, capturing:
 *  - Whether coarsening fired (the resulting body contains a `COARSENED` op).
 *  - Op count of the coarsened top-level body.
 *  - Op count of the gradient function's body (via `apply` on the coarsened primal).
 *  - Wall-clock time for each coarsening step.
 *
 * Used by `LSweepTest` to sweep `L ∈ {25, 50, 100, 200}` over a benchmark suite +
 * inform the plugin's default. JVM-only because `PhiCalculus.coarsenFunction` needs
 * a [SymbolicEngine] impl ([SymjaEngine]).
 */
object LProfiler {

    data class SweepMetric(
        val sizeLimit: Int,
        /** `true` when `coarsenFunction` produced a function containing at least one
         *  [OpKind.COARSENED] op (either at the top level or inside an IF branch). */
        val coarseningFired: Boolean,
        /** Op count of the coarsened primal's top-level body. For root-leaf coarsening
         *  this is typically 1 (just the COARSENED op). For multi-SOI splicing it
         *  matches the outer structure (e.g., STEP + IF = 2 with per-branch COARSENEDs
         *  nested inside the IF regions). */
        val coarsenedTopLevelOpCount: Int,
        /** Total op count across the coarsened primal's top-level body + all nested
         *  region bodies. This is the "what the runtime actually evaluates" number. */
        val coarsenedTotalOpCount: Int,
        /** Op count of the gradient function's top-level body after
         *  [DxirReverseTransform.apply] processes the coarsened primal. */
        val gradOpCount: Int,
        /** Coarsening wall-clock (ms). May be noisy on first run due to Symja classload. */
        val coarsenTimeMs: Long,
        /** Grad generation wall-clock (ms). */
        val gradTimeMs: Long,
    )

    fun sweep(
        fn: DxirFunction,
        engine: SymbolicEngine?,
        sizeLimits: List<Int>,
    ): List<SweepMetric> {
        val metrics = mutableListOf<SweepMetric>()
        for (L in sizeLimits) {
            val coarsenStart = System.currentTimeMillis()
            val coarsened = PhiCalculus.coarsenFunction(fn, engine, sizeLimit = L)
            val coarsenTimeMs = System.currentTimeMillis() - coarsenStart

            val gradStart = System.currentTimeMillis()
            val gradFn = try {
                DxirReverseTransform.apply(coarsened)
            } catch (_: Exception) {
                null
            }
            val gradTimeMs = System.currentTimeMillis() - gradStart

            val fired = containsCoarsened(coarsened)
            metrics += SweepMetric(
                sizeLimit = L,
                coarseningFired = fired,
                coarsenedTopLevelOpCount = coarsened.body.size,
                coarsenedTotalOpCount = totalOpCount(coarsened),
                gradOpCount = gradFn?.body?.size ?: -1,
                coarsenTimeMs = coarsenTimeMs,
                gradTimeMs = gradTimeMs,
            )
        }
        return metrics
    }

    /** Recursively scan [fn]'s body + all nested regions for any [OpKind.COARSENED] op. */
    private fun containsCoarsened(fn: DxirFunction): Boolean {
        fun scan(nodes: List<io.tlaloc.ir.DxirNode>): Boolean {
            for (n in nodes) {
                if (n !is DxirOp) continue
                if (n.op == OpKind.COARSENED) return true
                for (r in n.regions) for (b in r.blocks) {
                    if (scan(b.body)) return true
                }
            }
            return false
        }
        return scan(fn.body)
    }

    /** Count ops across top-level body + all nested region bodies. */
    private fun totalOpCount(fn: DxirFunction): Int {
        fun count(nodes: List<io.tlaloc.ir.DxirNode>): Int {
            var total = nodes.size
            for (n in nodes) {
                if (n !is DxirOp) continue
                for (r in n.regions) for (b in r.blocks) {
                    total += count(b.body)
                }
            }
            return total
        }
        return count(fn.body)
    }
}
