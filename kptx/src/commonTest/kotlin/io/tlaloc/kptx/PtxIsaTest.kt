package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * KPTX v2.3 (§0.4.340) — ISA spec table tests. The v1 corpus is the
 * positive fixture (every real kernel validates clean); the negative
 * cases pin each rejection dimension: unknown base, operand
 * count/kind, register class, `.wide` widening, modifier slots, guard
 * class.
 */
class PtxIsaTest {

    @Test
    fun v1CorpusValidatesClean() {
        for ((idx, text) in PtxRoundTripCorpus.all.withIndex()) {
            val errors = parsePtx(text).validateIsaErrors()
            assertEquals(emptyList(), errors, "corpus[$idx] should validate clean")
        }
    }

    @Test
    fun rejectsUnknownOpcodeBase() {
        val errors = validateInst(PtxInst("frobnicate.f32", listOf(PtxReg("%f1"))))
        assertTrue(errors.single().contains("unknown opcode base"), errors.toString())
    }

    @Test
    fun rejectsOperandCountAndKindMismatches() {
        // add.f32 wants 3 operands.
        assertTrue(
            validateInst(PtxInst("add.f32", listOf(PtxReg("%f1"), PtxReg("%f2"))))
                .single().contains("expected 3 operand(s)"),
        )
        // bra wants a symbol, not a register.
        assertTrue(
            validateInst(PtxInst("bra", listOf(PtxReg("%r1"))))
                .single().contains("kind REG"),
        )
        // st.global.f32 wants [mem], reg — reversed is two kind errors.
        assertEquals(
            2,
            validateInst(PtxInst("st.global.f32", listOf(PtxReg("%f1"), PtxMem("%rd1")))).size,
        )
    }

    @Test
    fun rejectsRegisterClassMismatches() {
        // f32 load into a %rd register.
        assertTrue(
            validateInst(PtxInst("ld.global.f32", listOf(PtxReg("%rd1"), PtxMem("%rd2"))))
                .single().contains("expected %f"),
        )
        // setp destination must be a predicate.
        assertTrue(
            validateInst(PtxInst("setp.ge.u32", listOf(PtxReg("%r1"), PtxReg("%r2"), PtxReg("%r3"))))
                .single().contains("expected %p"),
        )
        // Special registers are 32-bit and mov's dst class comes from the
        // type: mov.u32 into a %rd register is a class error (the %ctaid.x
        // src passes — mov's src rule is Any, matching `mov.u64 %rd, sdata`).
        assertTrue(
            validateInst(PtxInst("mov.u32", listOf(PtxReg("%rd1"), PtxReg("%ctaid.x"))))
                .single().contains("expected %r"),
        )
    }

    @Test
    fun wideWidensTheDestinationClass() {
        // mul.wide.u32 writes a 64-bit result: %rd accepted, %r rejected.
        assertEquals(
            emptyList(),
            validateInst(PtxInst("mul.wide.u32", listOf(PtxReg("%rd5"), PtxReg("%r5"), PtxImm("4")))),
        )
        assertTrue(
            validateInst(PtxInst("mul.wide.u32", listOf(PtxReg("%r6"), PtxReg("%r5"), PtxImm("4"))))
                .single().contains("expected %rd"),
        )
        // Without .wide the destination stays 32-bit.
        assertEquals(
            emptyList(),
            validateInst(PtxInst("mul.lo.u32", listOf(PtxReg("%r6"), PtxReg("%r5"), PtxImm("4")))),
        )
    }

    @Test
    fun rejectsModifierSlotViolations() {
        // ld's state-space slot is required.
        assertTrue(
            validateInst(PtxInst("ld.f32", listOf(PtxReg("%f1"), PtxMem("%rd1"))))
                .single().contains("missing required modifier"),
        )
        // Unknown modifier token.
        assertTrue(
            validateInst(PtxInst("ld.warp.f32", listOf(PtxReg("%f1"), PtxMem("%rd1"))))
                .any { it.contains("unknown modifier") || it.contains("missing required") },
        )
        // fma requires an explicit rounding mode.
        assertTrue(
            validateInst(
                PtxInst("fma.f32", listOf(PtxReg("%f1"), PtxReg("%f2"), PtxReg("%f3"), PtxReg("%f4"))),
            ).single().contains("missing required modifier"),
        )
    }

    @Test
    fun rejectsNonPredicateGuardAndValidatesModuleWithLocation() {
        val guarded = PtxInst("bra", listOf(PtxSym("L")), guard = PtxGuard("%r1"))
        assertTrue(validateInst(guarded).single().contains("guard register"), "guard class")

        val module = PtxModule(
            kernels = listOf(
                PtxKernel(
                    "bad", listOf(PtxParam(".u64", "p")),
                    body = listOf(
                        PtxRegDecl(".b32", "%r", 2),
                        PtxInst("frobnicate", emptyList()),
                        PtxInst("ret"),
                    ),
                ),
            ),
        )
        val e = assertFailsWith<IsaValidationException> { module.validateIsa() }
        assertTrue(e.errors.single().startsWith("kernel bad, stmt 1:"), e.errors.toString())
    }
}
