package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * KPTX v2.2 (§0.4.339) — parser tests: byte-identical round-trip over
 * the real v1 kernel corpus, structural decode assertions, and strict
 * rejection of non-canonical input (leniency would silently break the
 * round-trip contract; see [parsePtx]'s KDoc).
 */
class PtxParserTest {

    @Test
    fun roundTripsEveryV1KernelByteIdentically() {
        for ((idx, text) in PtxRoundTripCorpus.all.withIndex()) {
            val canonical = text + "\n"
            val emitted = parsePtx(canonical).emitPtx()
            assertEquals(canonical, emitted, "corpus[$idx] failed byte-identical round-trip")
        }
    }

    @Test
    fun roundTripIsIdempotent() {
        for (text in PtxRoundTripCorpus.all) {
            val once = parsePtx(text).emitPtx()
            assertEquals(once, parsePtx(once).emitPtx())
        }
    }

    @Test
    fun decodesAddOneStructure() {
        val module = parsePtx(PtxRoundTripCorpus.addOne)
        assertEquals("7.0", module.version)
        assertEquals("sm_75", module.target)
        assertEquals(64, module.addressSize)

        val kernel = module.kernels.single()
        assertEquals("add_one", kernel.name)
        assertEquals(listOf(".u64", ".u64", ".u32"), kernel.params.map { it.type })
        assertEquals(listOf("in_ptr", "out_ptr", "n"), kernel.params.map { it.name })

        val insts = kernel.body.filterIsInstance<PtxInst>()
        assertEquals(18, insts.size)
        assertEquals(4, kernel.body.filterIsInstance<PtxRegDecl>().size)
        assertEquals(listOf("DONE"), kernel.body.filterIsInstance<PtxLabel>().map { it.name })

        val guarded = insts.single { it.guard != null }
        assertEquals("bra", guarded.opcode)
        assertEquals(PtxGuard("%p1"), guarded.guard)
        assertEquals(listOf<PtxOperand>(PtxSym("DONE")), guarded.operands)

        val ldParam = insts.first()
        assertEquals("ld.param.u64", ldParam.opcode)
        assertIs<PtxReg>(ldParam.operands[0])
        assertEquals(PtxMem("in_ptr"), ldParam.operands[1])

        val hexImm = insts.single { it.opcode == "add.f32" }.operands[2]
        assertEquals(PtxImm("0f3F800000"), hexImm)
    }

    @Test
    fun decodesTrailingCommentsSharedDeclsAndNegatedGuards() {
        val bwdDx = parsePtx(PtxRoundTripCorpus.rmsNormBwdDx).kernels.single()
        val withComment = bwdDx.body.filterIsInstance<PtxInst>().filter { it.comment != null }
        assertEquals(listOf("x row", "dy row", "dx row"), withComment.map { it.comment })
        assertEquals(
            listOf(PtxSharedDecl(4, "sdata", 1024), PtxSharedDecl(4, "sdata2", 1024)),
            bwdDx.body.filterIsInstance<PtxSharedDecl>(),
        )
        // §0.4.335's SKIP_INVR gate: `setp.ne` + positive guard.
        assertTrue(bwdDx.body.filterIsInstance<PtxInst>().any {
            it.guard == PtxGuard("%p4") && it.operands == listOf<PtxOperand>(PtxSym("SKIP_INVR"))
        })

        val synthetic = """
            .version 7.0
            .target sm_75
            .address_size 64

            .visible .entry g(
                .param .u64 p
            )
            {
                @!%p1 bra END;
                ld.shared.f32 %f1, [%rd1+4];
            END:
                ret;
            }
        """.trimIndent()
        val kernel = parsePtx(synthetic).kernels.single()
        val insts = kernel.body.filterIsInstance<PtxInst>()
        assertEquals(PtxGuard("%p1", negated = true), insts[0].guard)
        assertEquals(PtxMem("%rd1", 4), insts[1].operands[1])
        assertEquals(synthetic + "\n", parsePtx(synthetic).emitPtx())
    }

    @Test
    fun rejectsNonCanonicalInputWithLineNumbers() {
        val base = PtxRoundTripCorpus.addOne

        // Missing semicolon.
        val noSemi = base.replace("ret;", "ret")
        val e1 = assertFailsWith<PtxParseException> { parsePtx(noSemi) }
        assertTrue(e1.message!!.contains("line"), e1.message)

        // Bad header.
        val e2 = assertFailsWith<PtxParseException> { parsePtx(".versio 7.0\n") }
        assertEquals(1, e2.line)

        // Tab-indented body line (non-canonical whitespace).
        val tabbed = base.replace("    ret;", "\tret;")
        assertFailsWith<PtxParseException> { parsePtx(tabbed) }
    }
}
