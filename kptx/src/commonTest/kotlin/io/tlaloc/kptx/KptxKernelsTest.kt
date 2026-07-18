package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v2.7 (§0.4.344) — production kernel library pins. Numerical
 * certification is the GPU tests' job (the §0.4.335/336/337 oracles and
 * E2E now consume these templates); here we pin the v2 DoD's structural
 * half — byte-stability under parse/emit, ISA cleanliness — plus the
 * launch-ABI signatures the registry marshals positionally.
 */
class KptxKernelsTest {

    private fun specializations(): Map<String, PtxModule> = mapOf(
        "rmsNormEps" to KptxKernels.rmsNormEps.specialize(shapes = mapOf("block" to 256)),
        "rmsNormBwdDx" to KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 256)),
        "rmsNormBwdDw" to KptxKernels.rmsNormBwdDw.specialize(),
        "rope" to KptxKernels.rope.specialize(),
        "crossEntropy" to KptxKernels.crossEntropyModule(block = 256),
    )

    @Test
    fun specializationsAreByteStableUnderParseEmitAndIsaClean() {
        for ((name, module) in specializations()) {
            val text = module.emitPtx()
            assertEquals(text, parsePtx(text).emitPtx(), "$name must be byte-stable under parse/emit")
            assertEquals(emptyList(), module.validateIsaErrors(), "$name must be ISA-clean")
        }
    }

    @Test
    fun launchSignaturesMatchTheRegistryMarshalling() {
        // Inputs-then-outputs-then-trailing-scalars, positionally (§0.4.330).
        val fwd = KptxKernels.rmsNormEps.specialize(shapes = mapOf("block" to 256)).kernels.single()
        assertEquals("kptx_rms_norm", fwd.name)
        assertEquals(listOf("x_ptr", "eps_ptr", "out_ptr", "n_cols"), fwd.params.map { it.name })

        val dx = KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 256)).kernels.single()
        assertEquals("kptx_rms_norm_bwd_dx", dx.name)
        assertEquals(
            listOf("x_ptr", "w_ptr", "dy_ptr", "dx_ptr", "invr_ptr", "n_cols"),
            dx.params.map { it.name },
        )

        val dw = KptxKernels.rmsNormBwdDw.specialize().kernels.single()
        assertEquals("kptx_rms_norm_bwd_dw", dw.name)
        assertEquals(
            listOf("x_ptr", "dy_ptr", "invr_ptr", "dw_ptr", "n_rows", "n_cols"),
            dw.params.map { it.name },
        )
    }

    @Test
    fun blockShapeSizesTheReductionScratch() {
        val small = KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 128)).emitPtx()
        val large = KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 256)).emitPtx()
        assertTrue(small.contains(".shared .align 4 .b8 sdata[512];"))
        assertTrue(small.contains(".shared .align 4 .b8 sdata2[512];"))
        assertTrue(large.contains(".shared .align 4 .b8 sdata[1024];"))
        assertTrue(large.contains(".shared .align 4 .b8 sdata2[1024];"))
    }
}
