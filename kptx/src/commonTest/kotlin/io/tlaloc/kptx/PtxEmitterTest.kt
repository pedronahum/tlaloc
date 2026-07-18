package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * KPTX v2.1 (§0.4.338) — canonical-emit pins. Every expectation is a
 * byte-exact string compare: the emitter's output format is the
 * round-trip corpus format (task 10), so formatting *is* contract.
 *
 * [emitsAddOneKernelByteExact] transcribes the §0.4.328 `add_one`
 * smoke-test kernel into IR and pins that the emit reproduces the
 * hand-written text exactly — the evidence that the canonical style
 * and the v1 corpus style are the same style.
 */
class PtxEmitterTest {

    private fun i(op: String, vararg ops: PtxOperand) = PtxInst(op, ops.toList())
    private fun r(name: String) = PtxReg(name)
    private fun m(base: String, offset: Int = 0) = PtxMem(base, offset)
    private fun imm(text: String) = PtxImm(text)

    private fun addOneIr(): PtxModule = PtxModule(
        kernels = listOf(
            PtxKernel(
                name = "add_one",
                params = listOf(
                    PtxParam(".u64", "in_ptr"),
                    PtxParam(".u64", "out_ptr"),
                    PtxParam(".u32", "n"),
                ),
                body = listOf(
                    PtxRegDecl(".pred", "%p", 2),
                    PtxRegDecl(".b32", "%r", 6),
                    PtxRegDecl(".f32", "%f", 3),
                    PtxRegDecl(".b64", "%rd", 8),
                    PtxBlank,
                    i("ld.param.u64", r("%rd1"), m("in_ptr")),
                    i("ld.param.u64", r("%rd2"), m("out_ptr")),
                    i("ld.param.u32", r("%r1"), m("n")),
                    i("cvta.to.global.u64", r("%rd3"), r("%rd1")),
                    i("cvta.to.global.u64", r("%rd4"), r("%rd2")),
                    i("mov.u32", r("%r2"), r("%ctaid.x")),
                    i("mov.u32", r("%r3"), r("%ntid.x")),
                    i("mov.u32", r("%r4"), r("%tid.x")),
                    i("mad.lo.s32", r("%r5"), r("%r2"), r("%r3"), r("%r4")),
                    i("setp.ge.s32", r("%p1"), r("%r5"), r("%r1")),
                    PtxInst("bra", listOf(PtxSym("DONE")), guard = PtxGuard("%p1")),
                    i("mul.wide.s32", r("%rd5"), r("%r5"), imm("4")),
                    i("add.s64", r("%rd6"), r("%rd3"), r("%rd5")),
                    i("ld.global.f32", r("%f1"), m("%rd6")),
                    i("add.f32", r("%f2"), r("%f1"), imm("0f3F800000")),
                    i("add.s64", r("%rd7"), r("%rd4"), r("%rd5")),
                    i("st.global.f32", m("%rd7"), r("%f2")),
                    PtxLabel("DONE"),
                    i("ret"),
                ),
            ),
        ),
    )

    @Test
    fun emitsAddOneKernelByteExact() {
        val expected = """
            .version 7.0
            .target sm_75
            .address_size 64

            .visible .entry add_one(
                .param .u64 in_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            )
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .f32 %f<3>;
                .reg .b64 %rd<8>;

                ld.param.u64 %rd1, [in_ptr];
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
                ld.global.f32 %f1, [%rd6];
                add.f32 %f2, %f1, 0f3F800000;
                add.s64 %rd7, %rd4, %rd5;
                st.global.f32 [%rd7], %f2;
            DONE:
                ret;
            }
        """.trimIndent() + "\n"
        assertEquals(expected, addOneIr().emitPtx())
    }

    @Test
    fun emitsSharedDeclNegatedGuardMemOffsetSymAndComment() {
        val module = PtxModule(
            kernels = listOf(
                PtxKernel(
                    name = "scratch",
                    params = listOf(PtxParam(".u64", "out_ptr")),
                    body = listOf(
                        PtxRegDecl(".pred", "%p", 2),
                        PtxRegDecl(".b64", "%rd", 4),
                        PtxSharedDecl(align = 4, name = "sdata", sizeBytes = 1024),
                        PtxBlank,
                        PtxComment("address-of-shared then offset store"),
                        i("mov.u64", r("%rd1"), PtxSym("sdata")),
                        i("st.shared.f32", m("%rd1", 4), r("%f0")),
                        PtxInst("bra", listOf(PtxSym("END")), guard = PtxGuard("%p1", negated = true)),
                        PtxLabel("END"),
                        i("ret"),
                    ),
                ),
            ),
        )
        val expected = """
            .version 7.0
            .target sm_75
            .address_size 64

            .visible .entry scratch(
                .param .u64 out_ptr
            )
            {
                .reg .pred %p<2>;
                .reg .b64 %rd<4>;
                .shared .align 4 .b8 sdata[1024];

                // address-of-shared then offset store
                mov.u64 %rd1, sdata;
                st.shared.f32 [%rd1+4], %f0;
                @!%p1 bra END;
            END:
                ret;
            }
        """.trimIndent() + "\n"
        assertEquals(expected, module.emitPtx())
    }

    @Test
    fun emitsMultiKernelModuleWithBlankSeparators() {
        val k = { name: String ->
            PtxKernel(name, listOf(PtxParam(".u64", "p")), listOf(i("ret")))
        }
        val expected = """
            .version 7.0
            .target sm_75
            .address_size 64

            .visible .entry a(
                .param .u64 p
            )
            {
                ret;
            }

            .visible .entry b(
                .param .u64 p
            )
            {
                ret;
            }
        """.trimIndent() + "\n"
        assertEquals(expected, PtxModule(kernels = listOf(k("a"), k("b"))).emitPtx())
    }
}
