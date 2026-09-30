@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.ir.passes

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.pretty
import kotlin.test.assertEquals
import kotlin.test.fail

/**
 * The property every batching rule is checked against: `vmap(f)(xs)` equals
 * `f` applied to each example and the results stacked along a new leading axis.
 *
 * [build] makes the per-example function at a dtype, with concrete extents.
 * [batched] says which parameters are batched. Each parameter gets deterministic
 * pseudo-random inputs per example ([input] can override them, e.g. to keep a value
 * positive); an unbatched parameter takes example 0's input. The batched function is
 * evaluated once, the per-example function once per example, both on the reference
 * interpreter of the dtype, and the stacked outputs must agree to [tolerance] relative
 * to the largest magnitude (0 = equal, `-0.0` equal to `0.0`).
 */
object VmapOracle {

    val batchSizes = listOf(1, 7, 64)

    fun check(
        build: (DType) -> DxirFunction,
        batched: List<Boolean>,
        batchSizes: List<Int> = VmapOracle.batchSizes,
        tolerance: Double = 0.0,
        dtypes: List<DType> = listOf(F32, F64),
        input: (param: Int, example: Int, size: Int) -> DoubleArray = ::defaultInput,
    ) {
        for (dt in dtypes) for (b in batchSizes) checkOnce(build(dt), dt, batched, b, tolerance, input)
    }

    fun defaultInput(param: Int, example: Int, size: Int): DoubleArray {
        var s = (param * 7919L + example * 104729L + 12345L) and 0x7fffffffL
        return DoubleArray(size) {
            s = (s * 1103515245L + 12345L) and 0x7fffffffL
            ((s % 20000L) - 10000L) / 7000.0
        }
    }

    private fun size(dims: List<Int>): Int = dims.fold(1) { a, d -> a * d }

    private fun checkOnce(
        fn: DxirFunction,
        dt: DType,
        batched: List<Boolean>,
        batch: Int,
        tolerance: Double,
        input: (Int, Int, Int) -> DoubleArray,
    ) {
        val vfn = DxirVmapTransform.apply(fn, batched, batch)
        val perExample: List<List<DoubleArray>> = (0 until batch).map { e ->
            fn.params.mapIndexed { p, prm ->
                input(p, if (batched[p]) e else 0, size(prm.type.dims))
            }
        }
        val batchedInputs = fn.params.indices.map { p ->
            if (batched[p]) perExample.flatMap { it[p].asList() }.toDoubleArray() else perExample[0][p]
        }
        val got = eval(vfn, dt, batchedInputs)
        val want = (0 until batch).map { e -> eval(fn, dt, perExample[e]) }
        assertEquals(fn.returns.size, got.size, "number of outputs")
        for (r in fn.returns.indices) {
            val stacked = want.flatMap { it[r].asList() }
            val g = got[r]
            assertEquals(
                stacked.size, g.size,
                "${fn.name} at $dt, batch $batch, output $r: size ${g.size}, stacked size ${stacked.size}",
            )
            val scale = maxOf(1.0, stacked.maxOfOrNull { kotlin.math.abs(it) } ?: 0.0)
            for (i in g.indices) {
                val a = g[i]
                val w = stacked[i]
                val ok = (a.isNaN() && w.isNaN()) || a == w || kotlin.math.abs(a - w) <= tolerance * scale
                if (!ok) {
                    fail(
                        "${fn.name} at $dt, batch $batch, output $r, element $i: vmap gives $a, " +
                            "stacking the examples gives $w\nbatched function:\n${vfn.pretty()}",
                    )
                }
            }
        }
    }

    private fun eval(fn: DxirFunction, dt: DType, inputs: List<DoubleArray>): List<DoubleArray> = when (dt) {
        F64 -> DxirInterpreterF64.evalFunction(fn, inputs)
        else -> DxirInterpreter.evalFunction(
            fn,
            inputs.map { d -> FloatArray(d.size) { d[it].toFloat() } },
        ).map { f -> DoubleArray(f.size) { f[it].toDouble() } }
    }
}
