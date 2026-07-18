package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.352 — lenient front-end tests. The fixture is the corpus
 * `add_one` mangled with every tolerated variance at once; the contract
 * is *structural* equality with the strict parse (comments/blanks are
 * dropped — byte fidelity starts at the canonical form).
 */
class PtxNormalizerTest {

    /** add_one with tabs, wrapped statements, block + line comments,
     * missing `.visible`, zero blank lines, spaced-out memory operands. */
    private val mangledAddOne = """
        .version 7.0
        .target sm_75
        .address_size 64
        /* expert kernel,
           pasted from somewhere */
        .entry add_one( .param .u64 in_ptr,
              .param .u64 out_ptr , .param .u32 n )
        {
        	.reg .pred %p<2>; .reg .b32 %r<6>;
        	.reg .f32 %f<3>;
            .reg .b64 %rd<8>;
            ld.param.u64
                %rd1, [in_ptr];   // wrapped statement
            ld.param.u64 %rd2, [out_ptr];
            ld.param.u32 %r1, [n];
            cvta.to.global.u64 %rd3, %rd1;
            cvta.to.global.u64 %rd4, %rd2;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra DONE;
            mul.wide.s32 %rd5, %r5, 4;
            add.s64 %rd6, %rd3, %rd5;
            ld.global.f32 %f1, [ %rd6 ];
            add.f32 %f2, %f1, 0f3F800000;
            add.s64 %rd7, %rd4, %rd5;
            st.global.f32 [%rd7], %f2;
            DONE:  ret;
        }
    """.trimIndent()

    private fun stripFormatting(module: PtxModule): PtxModule = module.copy(
        kernels = module.kernels.map { k ->
            k.copy(body = k.body.mapNotNull { stmt ->
                when (stmt) {
                    is PtxComment, PtxBlank -> null
                    is PtxInst -> stmt.copy(comment = null)
                    else -> stmt
                }
            })
        },
    )

    @Test
    fun normalizesManglingToTheStrictParseStructure() {
        val normalized = normalizePtx(mangledAddOne)
        val strict = stripFormatting(parsePtx(PtxRoundTripCorpus.addOne))
        // The normalizer inserts the canonical decls-blank-body separator;
        // compare with it stripped on both sides.
        assertEquals(stripFormatting(strict), stripFormatting(normalized))
        assertEquals(emptyList(), normalized.validateIsaErrors())
    }

    @Test
    fun normalizedOutputEntersTheStrictPipelineByteStably() {
        val canonical = normalizePtx(mangledAddOne).emitPtx()
        assertEquals(canonical, parsePtx(canonical).emitPtx())
    }

    @Test
    fun feedsTheTranspilerEndToEnd() {
        // The full bootstrap mouth: mangled foreign PTX → normalize →
        // transpile (self-verified against the canonical emit).
        val source = KptxTranspiler.transpile(normalizePtx(mangledAddOne))
        assertTrue(source.contains("fun addOneModule(): PtxModule"))
        assertTrue(source.contains("inst(\"mad.lo.s32\""))
    }

    @Test
    fun refusesUnsupportedConstructsLoudly() {
        val func = assertFailsWith<PtxNormalizeException> {
            normalizePtx(".version 7.0\n.func helper() { ret; }")
        }
        assertTrue(func.message!!.contains(".func"), func.message)

        val globalVar = assertFailsWith<PtxNormalizeException> {
            normalizePtx(".version 7.0\n.global .f32 lut[4];")
        }
        assertTrue(globalVar.message!!.contains("module-scope"), globalVar.message)

        val perfDirective = assertFailsWith<PtxNormalizeException> {
            normalizePtx(".visible .entry k(.param .u64 p) .maxntid 128, 1, 1 { ret; }")
        }
        assertTrue(perfDirective.message!!.contains("performance directives"), perfDirective.message)

        val typedShared = assertFailsWith<PtxNormalizeException> {
            normalizePtx(".visible .entry k(.param .u64 p) { .shared .f32 s[64]; ret; }")
        }
        assertTrue(typedShared.message!!.contains(".b8"), typedShared.message)
    }
}
