package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.testing.OpCatalogue
import io.tlaloc.testing.OpSample
import io.tlaloc.testing.OpSweep
import io.tlaloc.testing.SampleFunctions
import io.tlaloc.testing.Surface
import io.tlaloc.testing.Tolerances
import io.tlaloc.testing.interpret
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test

/**
 * The op sweep's [Surface.DEVICE]: every catalogue sample, and its gradient where
 * reverse mode is claimed, on the PJRT device against the F32 interpreter, within
 * [Tolerances].
 */
class OpSweepDeviceTest {

    private fun deviceInputs(s: OpSample, f: DxirFunction, values: List<DoubleArray>): List<Any> =
        f.params.zip(values).map { (p, v) ->
            if (p.type.dtype == io.tlaloc.core.I32) IntArray(v.size) { v[it].toInt() } else FloatArray(v.size) { v[it].toFloat() }
        }.also { check(it.size == f.params.size) { s.name } }

    @Test
    fun device() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { session ->
            OpSweep.assertExpected(
                OpSweep.run(Surface.DEVICE) { s ->
                    val f = SampleFunctions(s, F32)
                    val x = f.inputs()
                    val tol = Tolerances.relative(s, F32, Surface.DEVICE)
                    val fns = listOf(f.primal to x.dropLast(1)) +
                        if (OpCatalogue.claims(s, Surface.VJP)) listOf(f.gradient to x) else emptyList()
                    for ((fn, values) in fns) {
                        val want = interpret(fn, F32, values)
                        val got = session.runOnHost(fn, deviceInputs(s, fn, values))
                        for ((i, w) in want.withIndex()) {
                            val g = when (val a = got[i]) {
                                is FloatArray -> DoubleArray(a.size) { a[it].toDouble() }
                                is IntArray -> DoubleArray(a.size) { a[it].toDouble() }
                                else -> error("result $i of ${fn.name} is ${a::class.simpleName}")
                            }
                            Tolerances.assertClose(g, w, tol, "${fn.name} result $i")
                        }
                    }
                },
            )
        }
    }
}
