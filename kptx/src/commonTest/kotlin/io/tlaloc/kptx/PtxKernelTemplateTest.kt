package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * KPTX v2.5 (§0.4.342) — specialization-cache tests. The fixture is a
 * reduction-scratch template that uses its symbolic shapes the way the
 * rms_norm rewrite (task 15) will: `block` sizes the shared-memory
 * array, `n_cols` is baked as a loop-bound immediate instead of a
 * runtime scalar param.
 */
class PtxKernelTemplateTest {

    private fun reductionScratchTemplate() = PtxKernelTemplate("reduce_scratch") { env ->
        val block = env.shape("block")
        val nCols = env.shape("n_cols")
        val xPtr = param(".u64", "x_ptr")
        val p1 = pred()
        val r1 = r32()
        val rd1 = r64()
        val sdata = shared("sdata", sizeBytes = 4 * block)

        inst("ld.param.u64", rd1, mem(xPtr))
        inst("mov.u32", r1, tidX)
        // n_cols baked as an immediate — no runtime scalar param needed.
        inst("setp.ge.u32", p1, r1, imm(nCols))
        val done = label("DONE")
        inst("bra", done, guard = p1)
        inst("mov.u64", rd1, sdata)
        place(done)
        inst("ret")
    }

    @Test
    fun equalKeysReturnTheSameInstance() {
        val template = reductionScratchTemplate()
        val a = template.specialize(shapes = mapOf("block" to 256, "n_cols" to 512))
        val b = template.specialize(shapes = mapOf("n_cols" to 512, "block" to 256))
        assertSame(a, b, "structurally equal keys must hit the cache")
        assertEquals(1, template.specializationCount)
    }

    @Test
    fun differentShapesEmitDifferentCanonicalPtx() {
        val template = reductionScratchTemplate()
        val small = template.specialize(shapes = mapOf("block" to 128, "n_cols" to 512))
        val large = template.specialize(shapes = mapOf("block" to 256, "n_cols" to 2048))
        assertEquals(2, template.specializationCount)

        val smallText = small.emitPtx()
        val largeText = large.emitPtx()
        assertNotEquals(smallText, largeText)
        assertTrue(smallText.contains(".shared .align 4 .b8 sdata[512];"))
        assertTrue(largeText.contains(".shared .align 4 .b8 sdata[1024];"))
        assertTrue(smallText.contains("setp.ge.u32 %p1, %r1, 512;"))
        assertTrue(largeText.contains("setp.ge.u32 %p1, %r1, 2048;"))

        // Specializations are canonical from birth.
        for (text in listOf(smallText, largeText)) {
            assertEquals(text, parsePtx(text).emitPtx())
        }
        assertEquals(emptyList(), small.validateIsaErrors())
    }

    @Test
    fun archKeysTheCacheAndBecomesTheTarget() {
        val template = reductionScratchTemplate()
        val shapes = mapOf("block" to 256, "n_cols" to 512)
        val sm75 = template.specialize(arch = "sm_75", shapes = shapes)
        val sm121 = template.specialize(arch = "sm_121a", shapes = shapes)
        assertEquals(2, template.specializationCount)
        assertEquals("sm_75", sm75.target)
        assertEquals("sm_121a", sm121.target)
        assertTrue(sm121.emitPtx().startsWith(".version 7.0\n.target sm_121a\n"))
    }

    @Test
    fun unboundShapeAndArgFailWithContext() {
        val template = reductionScratchTemplate()
        val e = assertFailsWith<IllegalArgumentException> {
            template.specialize(shapes = mapOf("block" to 256))
        }
        assertTrue(e.message!!.contains("`n_cols` is unbound"), e.message)
        assertTrue(e.message!!.contains("reduce_scratch"), e.message)

        val argTemplate = PtxKernelTemplate("t") { env ->
            param(".u64", "p")
            comment("dtype = ${env.arg("dtype")}")
            inst("ret")
        }
        val e2 = assertFailsWith<IllegalArgumentException> { argTemplate.specialize() }
        assertTrue(e2.message!!.contains("template arg `dtype` is unbound"), e2.message)
        // Bound arg specializes fine and lands in the emitted text.
        val ok = argTemplate.specialize(args = mapOf("dtype" to "f32"))
        assertTrue(ok.emitPtx().contains("// dtype = f32"))
    }
}
