package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * KPTX v3.1 (§0.4.345) — transpiler tests. [transpile] self-verifies
 * (replayed module must emit byte-identical PTX before source is
 * returned), so every green transpile IS a fidelity proof; the golden
 * test ([KptxTranspilerGoldenTest]) additionally pins that generated
 * source compiles and runs as Kotlin.
 */
class KptxTranspilerTest {

    @Test
    fun transpilesEveryCorpusKernelSelfVerified() {
        for ((idx, text) in PtxRoundTripCorpus.all.withIndex()) {
            val source = KptxTranspiler.transpile(parsePtx(text))
            assertTrue(source.isNotBlank(), "corpus[$idx] produced empty source")
            assertTrue(source.contains("fun "), "corpus[$idx] missing function decl")
        }
    }

    @Test
    fun transpilesTheDslKernelLibraryBackToSource() {
        // Full circle: DSL → PTX → transpiler → DSL source, self-verified.
        for (module in listOf(
            KptxKernels.rmsNormEps.specialize(shapes = mapOf("block" to 256)),
            KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 256)),
            KptxKernels.rmsNormBwdDw.specialize(),
        )) {
            KptxTranspiler.transpile(module)
        }
    }

    @Test
    fun generatedSourcePreservesBanksSharedGuardsAndTrailingComments() {
        val source = KptxTranspiler.transpile(parsePtx(PtxRoundTripCorpus.rmsNormBwdDx))
        assertTrue(source.contains("bank(IsaRegClass.PRED, 6)"), source.lines().take(30).joinToString("\n"))
        assertTrue(source.contains("bank(IsaRegClass.F32, 32)"))
        assertTrue(source.contains("val sdata2 = shared(\"sdata2\", sizeBytes = 1024)"))
        assertTrue(source.contains("val SKIP_INVR = label(\"SKIP_INVR\")"))
        assertTrue(source.contains(", guard = p1)"))
        assertTrue(source.contains("comment = \"x row\""))
        assertTrue(source.contains("imm(\"0f3727C5AC\")"))
        assertTrue(source.contains("mem(x_ptr)"))
    }

    @Test
    fun refusesModulesTheStepMappingCannotExpress() {
        // Memory base that is neither a register nor a param.
        val badMem = PtxModule(
            kernels = listOf(
                PtxKernel(
                    "bad", listOf(PtxParam(".u64", "p")),
                    body = listOf(
                        PtxRegDecl(".f32", "%f", 2),
                        PtxBlank,
                        PtxInst("ld.shared.f32", listOf(PtxReg("%f1"), PtxMem("sdata"))),
                        PtxInst("ret"),
                    ),
                ),
            ),
        )
        val e = assertFailsWith<IllegalArgumentException> { KptxTranspiler.transpile(badMem) }
        assertTrue(e.message!!.contains("neither a register nor a param"), e.message)

        // Unknown special register in an operand.
        val badSpecial = PtxModule(
            kernels = listOf(
                PtxKernel(
                    "bad", listOf(PtxParam(".u64", "p")),
                    body = listOf(
                        PtxRegDecl(".b32", "%r", 2),
                        PtxBlank,
                        PtxInst("mov.u32", listOf(PtxReg("%r1"), PtxReg("%clock"))),
                        PtxInst("ret"),
                    ),
                ),
            ),
        )
        assertFailsWith<IllegalArgumentException> { KptxTranspiler.transpile(badSpecial) }
    }
}
