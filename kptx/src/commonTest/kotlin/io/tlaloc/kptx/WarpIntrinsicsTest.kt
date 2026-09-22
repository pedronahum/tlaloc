package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * KPTX v2.6 (§0.4.343) — vector-operand surface + typed-wrapper tests.
 * GPU-side proof lives in `:runtime-cuda`'s WarpIntrinsicsGpuTest
 * (driver-JIT acceptance of the mma kernel; numerically-verified
 * shfl.sync warp reduction).
 */
class WarpIntrinsicsTest {

    @Test
    fun mmaWrapperEmitsCanonicalFragmentSyntaxAndRoundTrips() {
        val module = ptxKernel("mma_tile") {
            param(".u64", "p")
            val f = List(4) { f32() }        // %f1..%f4
            val r = List(6) { r32() }        // %r1..%r6
            mmaSyncM16N8K16F32F16(d = f, a = r.take(4), b = r.drop(4), c = f)
            inst("ret")
        }
        val text = module.emitPtx()
        assertTrue(
            text.contains(
                "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                    "{%f1, %f2, %f3, %f4}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%f1, %f2, %f3, %f4};\n",
            ),
            text,
        )
        // Vector operands survive parse → emit byte-identically and are ISA-clean.
        assertEquals(text, parsePtx(text).emitPtx())
        assertEquals(emptyList(), module.validateIsaErrors())

        val decoded = parsePtx(text).kernels.single().body.filterIsInstance<PtxInst>().first()
        assertEquals(
            PtxVec(listOf("%r1", "%r2", "%r3", "%r4")),
            decoded.operands[1],
        )
    }

    @Test
    fun mmaWrapperEnforcesFragmentArityAndClass() {
        val arity = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val f = List(4) { f32() }
                val r = List(4) { r32() }
                mmaSyncM16N8K16F32F16(d = f, a = r, b = r, c = f) // b must be 2 regs
            }
        }
        assertTrue(arity.message!!.contains("b fragment must have 2"), arity.message)

        val cls = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val f = List(4) { f32() }
                val r = List(2) { r32() }
                mmaSyncM16N8K16F32F16(d = f, a = f, b = r, c = f) // a must be %r-class
            }
        }
        assertTrue(cls.message!!.contains("a fragment must be %r-class"), cls.message)

        // The raw escape hatch is guarded by the ISA table's per-element
        // class rule too: an f32-typed d fragment holding %r registers.
        val table = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val r = List(4) { r32() }
                inst(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
                    vec(*r.toTypedArray()), vec(*r.toTypedArray()),
                    vec(r[0], r[1]), vec(*r.toTypedArray()),
                )
            }
        }
        assertTrue(table.message!!.contains("fragment element"), table.message)
    }

    @Test
    fun shflAndVoteWrappersEmitValidatedInstructions() {
        val module = ptxKernel("warp_ops") {
            param(".u64", "p")
            val p1 = pred()
            val r1 = r32(); val r2 = r32(); val r3 = r32()
            shflSync(KShflMode.DOWN, d = r2, a = r1, b = imm(16), c = imm("0x1f"))
            shflSync(KShflMode.BFLY, d = r3, a = r2, b = imm(1), c = imm("0x1f"), mask = imm("0xffffffff"))
            voteBallotSync(d = r3, p = p1)
            inst("ret")
        }
        val text = module.emitPtx()
        assertTrue(text.contains("shfl.sync.down.b32 %r2, %r1, 16, 0x1f, 0xffffffff;"), text)
        assertTrue(text.contains("shfl.sync.bfly.b32 %r3, %r2, 1, 0x1f, 0xffffffff;"), text)
        assertTrue(text.contains("vote.sync.ballot.b32 %r3, %p1, 0xffffffff;"), text)
        assertEquals(text, parsePtx(text).emitPtx())
        assertEquals(emptyList(), module.validateIsaErrors())

        // §0.4.493 — the wrapper now rejects on WIDTH, not class: a
        // 64-bit operand is what `shfl.sync.…b32` cannot carry. The
        // §0.4.343 spelling of this case (d=%f, a=%r) is legal PTX and
        // is asserted positive in [floatWarpReductionIsOneCall].
        val wrongWidth = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") {
                val rd1 = r64(); val r1 = r32()
                shflSync(KShflMode.DOWN, d = rd1, a = r1, b = imm(1), c = imm("0x1f"))
            }
        }
        assertTrue(wrongWidth.message!!.contains("must be a 32-bit class"), wrongWidth.message)
    }

    /**
     * §0.4.493 — the byte-level emission pin for the float warp
     * reduction. The point of the slice is that this is **one call** at
     * the kernel site and contains **no `mov.b32` round trip**: the
     * shuffle carries `%f` registers itself.
     */
    @Test
    fun floatWarpReductionIsOneCall() {
        val module = ptxKernel("warp_sum_f32") {
            param(".u64", "p")
            val acc = f32()          // %f1
            warpReduceSumF32(acc)    // allocates %f2 as scratch
            inst("ret")
        }
        val text = module.emitPtx()
        val body = text.lines().map { it.trim() }.filter { it.startsWith("shfl") || it.startsWith("add") }
        assertEquals(
            listOf(
                "shfl.sync.down.b32 %f2, %f1, 16, 0x1f, 0xffffffff;",
                "add.rn.f32 %f1, %f1, %f2;",
                "shfl.sync.down.b32 %f2, %f1, 8, 0x1f, 0xffffffff;",
                "add.rn.f32 %f1, %f1, %f2;",
                "shfl.sync.down.b32 %f2, %f1, 4, 0x1f, 0xffffffff;",
                "add.rn.f32 %f1, %f1, %f2;",
                "shfl.sync.down.b32 %f2, %f1, 2, 0x1f, 0xffffffff;",
                "add.rn.f32 %f1, %f1, %f2;",
                "shfl.sync.down.b32 %f2, %f1, 1, 0x1f, 0xffffffff;",
                "add.rn.f32 %f1, %f1, %f2;",
            ),
            body,
        )
        assertTrue("mov.b32" !in text, "no bit-reinterpretation round trip is needed:\n$text")
        assertEquals(text, parsePtx(text).emitPtx())
        assertEquals(emptyList(), module.validateIsaErrors())

        val wrongClass = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") { warpReduceSumF32(r32()) }
        }
        assertTrue(wrongClass.message!!.contains("must be %f-class"), wrongClass.message)
    }

    /** §0.4.493 — `mov.b32` as bit reinterpretation, both directions,
     * and the width rejection the ISA table now carries. */
    @Test
    fun movB32ReinterpretsAcrossThirtyTwoBitClasses() {
        val module = ptxKernel("bitcast") {
            param(".u64", "p")
            val f1 = f32(); val r1 = r32()
            movB32(r1, f1)
            movB32(f1, r1)
            inst("ret")
        }
        val text = module.emitPtx()
        assertTrue(text.contains("    mov.b32 %r1, %f1;\n"), text)
        assertTrue(text.contains("    mov.b32 %f1, %r1;\n"), text)
        assertEquals(text, parsePtx(text).emitPtx())
        assertEquals(emptyList(), module.validateIsaErrors())

        val tooWide = assertFailsWith<IllegalArgumentException> {
            ptxKernel("bad") { movB32(r64(), f32()) }
        }
        assertTrue(tooWide.message!!.contains("must be a 32-bit class"), tooWide.message)
    }
}
