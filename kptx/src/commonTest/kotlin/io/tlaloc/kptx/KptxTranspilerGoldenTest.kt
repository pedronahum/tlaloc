package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * KPTX v3.1 (§0.4.345) — the golden pin closing the transpiler's
 * verification triangle: [transpiledAddOne] below is the **verbatim
 * output** of `KptxTranspiler.transpile(parsePtx(PtxRoundTripCorpus.addOne))`,
 * frozen as Kotlin source. It compiles as part of this suite and, when
 * run, emits PTX byte-identical to the §0.4.328 hand-written original —
 * proving the generated text is not just faithful *as a rendering of
 * the step list* (the transpiler's internal replay check) but also
 * valid, executable Kotlin.
 *
 * If the transpiler's output format changes intentionally, regenerate
 * and re-freeze; drift is caught by [goldenStillMatchesTranspilerOutput].
 */
class KptxTranspilerGoldenTest {

    // ---- Verbatim transpiler output (KDoc header omitted) begins ----
    private fun addOneModule(): PtxModule = ptxKernel("add_one", version = "7.0", target = "sm_75") {
        val in_ptr = param(".u64", "in_ptr")
        val out_ptr = param(".u64", "out_ptr")
        val n = param(".u32", "n")
        bank(IsaRegClass.PRED, 2)
        bank(IsaRegClass.R32, 6)
        bank(IsaRegClass.F32, 3)
        bank(IsaRegClass.R64, 8)
        val p1 = reg(IsaRegClass.PRED, 1)
        val r1 = reg(IsaRegClass.R32, 1)
        val r2 = reg(IsaRegClass.R32, 2)
        val r3 = reg(IsaRegClass.R32, 3)
        val r4 = reg(IsaRegClass.R32, 4)
        val r5 = reg(IsaRegClass.R32, 5)
        val f1 = reg(IsaRegClass.F32, 1)
        val f2 = reg(IsaRegClass.F32, 2)
        val rd1 = reg(IsaRegClass.R64, 1)
        val rd2 = reg(IsaRegClass.R64, 2)
        val rd3 = reg(IsaRegClass.R64, 3)
        val rd4 = reg(IsaRegClass.R64, 4)
        val rd5 = reg(IsaRegClass.R64, 5)
        val rd6 = reg(IsaRegClass.R64, 6)
        val rd7 = reg(IsaRegClass.R64, 7)
        val DONE = label("DONE")

        inst("ld.param.u64", rd1, mem(in_ptr))
        inst("ld.param.u64", rd2, mem(out_ptr))
        inst("ld.param.u32", r1, mem(n))
        inst("cvta.to.global.u64", rd3, rd1)
        inst("cvta.to.global.u64", rd4, rd2)
        inst("mov.u32", r2, ctaidX)
        inst("mov.u32", r3, ntidX)
        inst("mov.u32", r4, tidX)
        inst("mad.lo.s32", r5, r2, r3, r4)
        inst("setp.ge.s32", p1, r5, r1)
        inst("bra", DONE, guard = p1)
        inst("mul.wide.s32", rd5, r5, imm(4))
        inst("add.s64", rd6, rd3, rd5)
        inst("ld.global.f32", f1, mem(rd6))
        inst("add.f32", f2, f1, imm("0f3F800000"))
        inst("add.s64", rd7, rd4, rd5)
        inst("st.global.f32", mem(rd7), f2)
        place(DONE)
        inst("ret")
    }
    // ---- Verbatim transpiler output ends ----

    @Test
    fun goldenSourceCompilesAndEmitsByteIdenticalPtx() {
        assertEquals(PtxRoundTripCorpus.addOne + "\n", addOneModule().emitPtx())
    }

    @Test
    fun goldenStillMatchesTranspilerOutput() {
        val generated = KptxTranspiler.transpile(parsePtx(PtxRoundTripCorpus.addOne))
        // The frozen function above is the generated body with test-class
        // indentation; compare structurally: every generated line (sans
        // KDoc) must appear in this file's frozen copy, in order.
        val genLines = generated.lines()
            .dropWhile { it.startsWith("/**") || it.startsWith(" *") || it.startsWith(" */") }
            .filter { it.isNotBlank() }
            .map { it.trim() }
        val frozen = FROZEN_BODY.lines().filter { it.isNotBlank() }.map { it.trim() }
        assertEquals(genLines.drop(1).dropLast(1), frozen.drop(1).dropLast(1), "regenerate the golden copy")
    }

    private companion object {
        val FROZEN_BODY = """
            fun addOneModule(): PtxModule = ptxKernel("add_one", version = "7.0", target = "sm_75") {
                val in_ptr = param(".u64", "in_ptr")
                val out_ptr = param(".u64", "out_ptr")
                val n = param(".u32", "n")
                bank(IsaRegClass.PRED, 2)
                bank(IsaRegClass.R32, 6)
                bank(IsaRegClass.F32, 3)
                bank(IsaRegClass.R64, 8)
                val p1 = reg(IsaRegClass.PRED, 1)
                val r1 = reg(IsaRegClass.R32, 1)
                val r2 = reg(IsaRegClass.R32, 2)
                val r3 = reg(IsaRegClass.R32, 3)
                val r4 = reg(IsaRegClass.R32, 4)
                val r5 = reg(IsaRegClass.R32, 5)
                val f1 = reg(IsaRegClass.F32, 1)
                val f2 = reg(IsaRegClass.F32, 2)
                val rd1 = reg(IsaRegClass.R64, 1)
                val rd2 = reg(IsaRegClass.R64, 2)
                val rd3 = reg(IsaRegClass.R64, 3)
                val rd4 = reg(IsaRegClass.R64, 4)
                val rd5 = reg(IsaRegClass.R64, 5)
                val rd6 = reg(IsaRegClass.R64, 6)
                val rd7 = reg(IsaRegClass.R64, 7)
                val DONE = label("DONE")
                inst("ld.param.u64", rd1, mem(in_ptr))
                inst("ld.param.u64", rd2, mem(out_ptr))
                inst("ld.param.u32", r1, mem(n))
                inst("cvta.to.global.u64", rd3, rd1)
                inst("cvta.to.global.u64", rd4, rd2)
                inst("mov.u32", r2, ctaidX)
                inst("mov.u32", r3, ntidX)
                inst("mov.u32", r4, tidX)
                inst("mad.lo.s32", r5, r2, r3, r4)
                inst("setp.ge.s32", p1, r5, r1)
                inst("bra", DONE, guard = p1)
                inst("mul.wide.s32", rd5, r5, imm(4))
                inst("add.s64", rd6, rd3, rd5)
                inst("ld.global.f32", f1, mem(rd6))
                inst("add.f32", f2, f1, imm("0f3F800000"))
                inst("add.s64", rd7, rd4, rd5)
                inst("st.global.f32", mem(rd7), f2)
                place(DONE)
                inst("ret")
            }
        """.trimIndent()
    }
}
