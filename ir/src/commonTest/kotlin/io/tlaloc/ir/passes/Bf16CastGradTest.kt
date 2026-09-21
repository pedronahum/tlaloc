package io.tlaloc.ir.passes

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.456 (Phase G1b) — bf16 through the IR: CAST arms, the interpreter's
 * widened-value convention, and the straight-through cast adjoint.
 *
 * THE CONVENTION UNDER TEST (see DxirInterpreter.snapToBf16's KDoc):
 * interpreter value arrays stay FloatArray; a bf16-typed node's values are
 * the f32-WIDENED forms of bf16-ROUNDED numbers. Ops with bf16 result types
 * compute in f32 and round once at their own output (XLA's "f32 accumulate,
 * bf16 result" dot/reduce convention).
 *
 * Oracles are hand arithmetic on the bf16 format (7 mantissa bits, f32's
 * exponent), never a re-implementation: every expected value below is a
 * hand-derived decimal that is exactly representable in bf16 (and hence in
 * f32), so assertEquals on floats is a BIT pin.
 */
class Bf16CastGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** CAST f32→bf16 through the interpreter is bit-exact vs the §0.4.455 helpers. */
    @Test
    fun interpreterCastMatchesG1aHelpersBitExactly() {
        // The G1a pin family: an RNE tie (down to even), one-ulp neighbors,
        // an exactly-representable value, signed zeros, overflow-to-inf,
        // a subnormal that flushes, and NaN.
        val inputs = floatArrayOf(
            1.00390625f, // 1 + 2^-8: tie on the discarded half, kept lsb even → 1.0
            1.0078125f, // 1 + 2^-7: exactly representable, passes through
            Float.fromBits(0x3F80_8001.toInt()), // just above the tie → 1.0078125
            0.1f, // classic non-representable → 0.10009765625
            0.0f, -0.0f,
            Float.MAX_VALUE, // RNE carries past max-finite bf16 → +inf
            Float.fromBits(0x0000_0001), // f32 subnormal → flushes to +0 by rounding
            Float.NaN,
        )
        val fn = DxirBuilder.function("cast_bf16") {
            val x = param("x", DxirType(F32, listOf(inputs.size)))
            listOf(op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(inputs.size))))
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(inputs)).single()
        for (i in inputs.indices) {
            val expected = bf16BitsToFloat(floatToBf16Bits(inputs[i]))
            assertEquals(
                expected.toRawBits(), out[i].toRawBits(),
                "CAST f32→bf16 of ${inputs[i]} must be bit-identical to the G1a helper " +
                    "(expected $expected, got ${out[i]})",
            )
        }
    }

    /** CAST bf16→f32 is exact widening: round-tripping through bf16 twice is stable. */
    @Test
    fun interpreterWideningCastIsExact() {
        val fn = DxirBuilder.function("roundtrip") {
            val x = param("x", DxirType(F32, listOf(3)))
            val b = op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(3)))
            val w = op(OpKind.CAST, listOf(b), DxirType(F32, listOf(3)))
            val b2 = op(OpKind.CAST, listOf(w), DxirType(BF16, listOf(3)))
            listOf(op(OpKind.CAST, listOf(b2), DxirType(F32, listOf(3))))
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(0.1f, 3.14159f, -1.7f))).single()
        // Hand-derived: 0.1 → 0x3DCD (0.10009765625); 3.14159 → 0x4049 (3.140625);
        // -1.7 → 0xBFDA (-1.703125). Round-trip idempotence: cast-in twice = once.
        assertEquals(listOf(0.10009765625f, 3.140625f, -1.703125f), out.toList())
    }

    /**
     * The G1b program shape: cast-in, add, matmul, cast-out. All post-cast
     * values are chosen bf16-exact except the deliberate cast-in rounding of
     * x[0], so every expected number is short hand arithmetic.
     */
    @Test
    fun bf16ProgramCastInAddMatmulCastOut() {
        val t = DxirType(F32, listOf(2, 2))
        val bt = DxirType(BF16, listOf(2, 2))
        val fn = DxirBuilder.function("bf16_prog") {
            val x = param("x", t)
            val w = param("w", t)
            val bx = op(OpKind.CAST, listOf(x), bt)
            val bw = op(OpKind.CAST, listOf(w), bt)
            val s = op(OpKind.ADD, listOf(bx, bw), bt)
            val m = op(OpKind.MATMUL, listOf(s, bx), bt)
            listOf(op(OpKind.CAST, listOf(m), t))
        }
        // x = [[1.00390625, 2], [3, 4]] → bx = [[1, 2], [3, 4]] (tie rounds down to even).
        // w = [[0.5, 0.25], [-1.5, 1]] (all bf16-exact) → bw = w.
        // s = [[1.5, 2.25], [1.5, 5]] (all bf16-exact, no output rounding).
        // m = s @ bx = [[1.5·1+2.25·3, 1.5·2+2.25·4], [1.5·1+5·3, 1.5·2+5·4]]
        //   = [[8.25, 12], [16.5, 23]] — every entry bf16-exact (≤ 5 significant
        //   mantissa bits), so the output rounding is the identity here.
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(
                floatArrayOf(1.00390625f, 2f, 3f, 4f),
                floatArrayOf(0.5f, 0.25f, -1.5f, 1f),
            ),
        ).single()
        assertEquals(listOf(8.25f, 12f, 16.5f, 23f), out.toList())
    }

    /**
     * A bf16 ADD whose f32 sum is NOT bf16-representable pins the
     * output-rounding half of the convention: 256 + 1.0078125 = 257.0078125
     * in f32, which sits between bf16 neighbors 256 and 258 (spacing 2 at
     * exponent 8), above the midpoint 257 → RNE rounds UP to 258.
     */
    @Test
    fun bf16AddRoundsItsOutput() {
        val bt = DxirType(BF16, listOf(1))
        val fn = DxirBuilder.function("bf16_add_rounds") {
            val a = param("a", DxirType(F32, listOf(1)))
            val b = param("b", DxirType(F32, listOf(1)))
            val ba = op(OpKind.CAST, listOf(a), bt)
            val bb = op(OpKind.CAST, listOf(b), bt)
            val s = op(OpKind.ADD, listOf(ba, bb), bt)
            listOf(op(OpKind.CAST, listOf(s), DxirType(F32, listOf(1))))
        }
        val out = DxirInterpreter.evalFunction(
            fn, listOf(floatArrayOf(256f), floatArrayOf(1.0078125f)),
        ).single()
        assertEquals(258f, out[0], "f32 sum 257.0078125 must RNE-round to bf16 258")
    }

    /**
     * Reverse transform through precision casts: the adjoint of the narrowing
     * cast f32→bf16 is the WIDENING cast of the upstream (straight-through),
     * and vice versa — the §0.4.427 CastRule arm extended to bf16 in G1b.
     * Without the extension the rule returned an empty contribution and the
     * gradient was silently zero.
     *
     * loss = sum(cast_f32(cast_bf16(x) · cast_bf16(x))). Hand derivation with
     * straight-through casts: d loss/dx = 2 · bf16(x), each product and the
     * accumulation rounding to bf16 exactly (doubling a bf16 value only
     * increments its exponent). x = [1.00390625, 1.0078125, 2.5] →
     * bf16(x) = [1, 1.0078125, 2.5] → grad = [2, 2.015625, 5].
     */
    @Test
    fun reverseThroughCastsIsStraightThrough() {
        val vt = DxirType(F32, listOf(3))
        val bt = DxirType(BF16, listOf(3))
        val fn = DxirBuilder.function("bf16_loss") {
            val x = param("x", vt)
            val b = op(OpKind.CAST, listOf(x), bt)
            val m = op(OpKind.MUL, listOf(b, b), bt)
            val w = op(OpKind.CAST, listOf(m), vt)
            listOf(op(OpKind.SUM, listOf(w), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        assertEquals(vt.dims, grad.returns.single().type.dims, "gradient carries x's f32 shape")
        assertEquals(
            F32, grad.returns.single().type.dtype,
            "the last cast back to the param dtype makes the gradient f32",
        )
        val out = DxirInterpreter.evalFunction(
            grad, listOf(floatArrayOf(1.00390625f, 1.0078125f, 2.5f)),
        ).single()
        assertEquals(listOf(2f, 2.015625f, 5f), out.toList())
    }
}
