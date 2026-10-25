package io.tlaloc.ir.sweep

import io.tlaloc.core.F32
import io.tlaloc.ir.render.KotlinSourceRenderer
import io.tlaloc.testing.OpSweep
import io.tlaloc.testing.SampleFunctions
import io.tlaloc.testing.Surface
import kotlin.test.Test

/**
 * The op sweep on the surfaces `:ir` owns: every sample of every
 * [io.tlaloc.testing.OpCatalogue] entry, each surface a test. A claimed surface that
 * fails and an excluded surface that works both fail the test.
 */
class OpSweepIrTest {

    private fun sweep(surface: Surface) = OpSweep.assertExpected(OpSweep.run(surface, check = OpSweep.irCheck(surface)))

    @Test fun interpreter() = sweep(Surface.INTERPRETER)

    @Test fun interpreterF64() = sweep(Surface.INTERPRETER_F64)

    @Test fun vjpAgainstCentralDifferences() = sweep(Surface.VJP)

    @Test fun jvpAgainstVjp() = sweep(Surface.JVP)

    @Test fun vmapAgainstPerExample() = sweep(Surface.VMAP)

    @Test
    fun render() = OpSweep.assertExpected(
        OpSweep.run(Surface.RENDER) { s ->
            val f = SampleFunctions(s, F32)
            KotlinSourceRenderer.render(f.primal)
            KotlinSourceRenderer.render(f.gradient)
        },
    )
}
