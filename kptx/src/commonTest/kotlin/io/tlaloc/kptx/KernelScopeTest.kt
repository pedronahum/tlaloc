package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * KPTX v2.4 (§0.4.341) — KernelScope DSL tests. The centerpiece is
 * [dslAddOneEmitsByteIdenticalPtx]: `add_one` written in pure Kotlin
 * emits **byte-identically** the §0.4.328 hand-written corpus text —
 * DSL-written and hand-written kernels are the same artifact, so the
 * driver-JIT validity of one is validity of the other
 * ([io.tlaloc.runtime.cuda.PtxIrDriverJitTest] already runs this exact
 * text on the GB10).
 */
class KernelScopeTest {

    private fun dslAddOne(): PtxModule = ptxKernel("add_one") {
        val inPtr = param(".u64", "in_ptr")
        val outPtr = param(".u64", "out_ptr")
        val n = param(".u32", "n")

        val p1 = pred()
        val r1 = r32(); val r2 = r32(); val r3 = r32(); val r4 = r32(); val r5 = r32()
        val f1 = f32(); val f2 = f32()
        val rd1 = r64(); val rd2 = r64(); val rd3 = r64(); val rd4 = r64()
        val rd5 = r64(); val rd6 = r64(); val rd7 = r64()

        inst("ld.param.u64", rd1, mem(inPtr))
        inst("ld.param.u64", rd2, mem(outPtr))
        inst("ld.param.u32", r1, mem(n))
        inst("cvta.to.global.u64", rd3, rd1)
        inst("cvta.to.global.u64", rd4, rd2)
        inst("mov.u32", r2, ctaidX)
        inst("mov.u32", r3, ntidX)
        inst("mov.u32", r4, tidX)
        inst("mad.lo.s32", r5, r2, r3, r4)
        inst("setp.ge.s32", p1, r5, r1)
        val done = label("DONE")
        inst("bra", done, guard = p1)
        inst("mul.wide.s32", rd5, r5, imm(4))
        inst("add.s64", rd6, rd3, rd5)
        inst("ld.global.f32", f1, mem(rd6))
        inst("add.f32", f2, f1, imm("0f3F800000"))
        inst("add.s64", rd7, rd4, rd5)
        inst("st.global.f32", mem(rd7), f2)
        place(done)
        inst("ret")
    }

    @Test
    fun dslAddOneEmitsByteIdenticalPtx() {
        assertEquals(PtxRoundTripCorpus.addOne + "\n", dslAddOne().emitPtx())
    }

    @Test
    fun escapeHatchValidatesAtTheCallSite() {
        val unknown = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val f1 = f32()
                inst("frobnicate.f32", f1)
            }
        }
        assertTrue(unknown.message!!.contains("unknown opcode base"), unknown.message)

        val classMismatch = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val rd1 = r64()
                inst("ld.global.f32", rd1, mem(rd1))
            }
        }
        assertTrue(classMismatch.message!!.contains("expected %f"), classMismatch.message)

        val badGuard = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val r1 = r32()
                val l = label("L")
                inst("bra", l, guard = r1)
                place(l)
            }
        }
        assertTrue(badGuard.message!!.contains("guard register"), badGuard.message)
    }

    @Test
    fun dslKernelWithSharedGuardsAndCommentsRoundTrips() {
        val module = ptxKernel("scratch") {
            param(".u64", "out_ptr")
            val p1 = pred()
            val f1 = f32()
            val rd1 = r64()
            val sdata = shared("sdata", sizeBytes = 1024)

            comment("address-of-shared then offset store")
            inst("mov.u64", rd1, sdata)
            inst("st.shared.f32", mem(rd1, 4), f1, comment = "scratch slot 1")
            blank()
            val end = label("END")
            inst("bra", end, guard = !p1)
            place(end)
            inst("ret")
        }
        val text = module.emitPtx()
        // Canonical from birth: parse → emit is byte-stable, ISA-clean.
        assertEquals(text, parsePtx(text).emitPtx())
        assertEquals(emptyList(), module.validateIsaErrors())
        assertTrue(text.contains("@!%p1 bra END;"))
        assertTrue(text.contains("st.shared.f32 [%rd1+4], %f1;    // scratch slot 1"))
        assertTrue(text.contains(".shared .align 4 .b8 sdata[1024];"))
    }

    @Test
    fun registerBanksAreSizedFromAllocationAndUnusedClassesOmitted() {
        val kernel = ptxKernel("banks") {
            param(".u64", "p")
            f32(); f32(); f32()
            r64()
            inst("ret")
        }.kernels.single()
        assertEquals(
            listOf(PtxRegDecl(".f32", "%f", 4), PtxRegDecl(".b64", "%rd", 2)),
            kernel.body.filterIsInstance<PtxRegDecl>(),
        )
    }

    @Test
    fun labelDisciplineIsEnforced() {
        assertFailsWith<IllegalArgumentException> {
            ptxKernel("k") {
                label("L") // declared, never placed
                inst("ret")
            }
        }
        assertFailsWith<IllegalArgumentException> {
            ptxKernel("k") {
                val l = label("L")
                place(l); place(l)
            }
        }
        assertFailsWith<IllegalArgumentException> {
            ptxKernel("k") {
                label("L"); label("L")
            }
        }
    }
}
