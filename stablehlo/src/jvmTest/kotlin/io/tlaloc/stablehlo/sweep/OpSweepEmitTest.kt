package io.tlaloc.stablehlo.sweep

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.stablehlo.toStablehlo
import io.tlaloc.testing.OpCatalogue
import io.tlaloc.testing.OpSweep
import io.tlaloc.testing.SampleFunctions
import io.tlaloc.testing.Surface
import kotlin.test.Test

/**
 * The op sweep's [Surface.EMIT] and [Surface.EMIT_F64]: every catalogue sample, and its
 * gradient where reverse mode is claimed, emits as StableHLO.
 */
class OpSweepEmitTest {

    private fun sweep(surface: Surface, dt: DType) = OpSweep.assertExpected(
        OpSweep.run(surface) { s ->
            val f = SampleFunctions(s, dt)
            f.primal.toStablehlo()
            if (OpCatalogue.claims(s, Surface.VJP)) f.gradient.toStablehlo()
        },
    )

    @Test fun emit() = sweep(Surface.EMIT, F32)

    @Test fun emitF64() = sweep(Surface.EMIT_F64, F64)
}
