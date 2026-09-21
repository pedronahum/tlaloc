package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.bf16BitsToFloatArray
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.457 (G1c) — bf16 certified against real XLA on the GB10. This is a
 * LOCAL certification (CUDA plugin, Blackwell); G2b re-runs this suite on a
 * TPU and no TPU claim is made here.
 *
 * WHAT IS CERTIFIED, loudly: BOTH forms from the G1c menu.
 *
 *   1. **Native bf16 buffers** — `PJRT_Buffer_Type_BF16` (= 13) host
 *      marshalling both ways (ShortArray of raw upper-16-bit f32 patterns,
 *      the §0.4.455 convention), via [PjrtSession.runOnBf16]. The staged
 *      buffer's on-device size proves 2-byte storage — the device holds
 *      TRUE bf16, not silently widened f32.
 *   2. **Cast-at-boundary** — f32 params, in-graph CAST to bf16, bf16
 *      compute, CAST back to f32 outputs; XLA runs real bf16 in the middle.
 *      This is the form mixed-precision training will use.
 *
 * The oracle story per test is in each test's KDoc; the headline claims,
 * ALL MEASURED on the GB10 (2026-09-21, jax[cuda12] plugin):
 *
 *   - the device f32→bf16 convert BIT-MATCHES the §0.4.455 RNE helper on
 *     the full tie sweep;
 *   - XLA-CUDA rounds a bf16 dot's OUTPUT to bf16 before the next op
 *     (per-op rounding — the interpreter's convention, pinned at the 257
 *     tie in [matmulAddIntermediateRoundingConvention]);
 *   - BUT XLA's algebraic simplifier ELIDES an f32→bf16→f32 convert
 *     round-trip pair (and composes that with mul-by-one folding in
 *     reverse graphs), so a narrowing whose value flows straight back to
 *     f32 may never happen on device — the measured divergence bound is
 *     the snap error itself, < 2⁻⁸ relative (pinned in the gradient test).
 */
class PjrtBf16SmokeTest {

    /**
     * The device narrowing bit-matches G1a's RNE helper. Program: f32 param
     * → ONE CAST → bf16 return, dispatched through the dtype-agnostic
     * [PjrtSession.executeOn] lane (f32 buffer in, bf16 patterns out), so
     * the convert is the program's contract and cannot be simplified away.
     * That routing is not cosmetic: the obvious f32→bf16→f32 round-trip
     * program came back IDENTITY on the GB10 — XLA's algebraic simplifier
     * folds the convert pair (measured first, lane 0 returned its input
     * raw) — so a single-convert program is the only honest probe. Every
     * output pattern is then a statement about XLA's narrowing — asserted
     * equal to `floatToBf16Bits` (RNE, ties-to-even, overflow-to-inf,
     * signed-zero flush) on raw bits.
     *
     * NaN is deliberately NOT bit-pinned: our helper keeps payload top bits
     * and forces the quiet bit, but convert payload propagation is
     * target-defined — the honest pin is `isNaN`, nothing tighter.
     */
    @Test
    fun castRoundTripBitMatchesHostRneOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // The G1a pin sweep: ties (even-down / odd-up), one-ulp neighbors,
        // the mirrored negative ties, signed zeros, infinities, RNE overflow
        // (MAX_VALUE→inf, just-below-tie stays finite), subnormal flushes,
        // and ordinary representable/non-representable values.
        val probes = floatArrayOf(
            Float.fromBits(0x3F808000),          // tie, even keeps → 1.0
            Float.fromBits(0x3F818000),          // tie, odd keeps → up to even
            Float.fromBits(0x3F808001),          // just above tie → up
            Float.fromBits(0x3F817FFF),          // just below tie → down
            Float.fromBits(0xBF808000.toInt()),  // negative tie → down
            Float.fromBits(0xBF818000.toInt()),  // negative tie → up to even
            0f, -0f,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.MAX_VALUE,                     // RNE overflow → +inf
            Float.fromBits(0x7F7F7FFF),          // below overflow tie → max finite
            Float.fromBits(0x00000001),          // min f32 subnormal → +0
            Float.fromBits(0x80000001.toInt()),  // sign survives the flush
            1.5f, 256f, 1.0078125f, 3.14159265f, -257f,
        )
        val n = probes.size
        val vec = DxirType(F32, listOf(n))
        val bvec = DxirType(BF16, listOf(n))
        val fn = DxirBuilder.function("bf16_narrow_probe") {
            val x = param("x", vec)
            listOf(op(OpKind.CAST, listOf(x), bvec))
        }

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val staged = session.bufferFromHostF32(probes, listOf(n))
            val got: ShortArray
            try {
                val outs = session.executeOn(fn, listOf(staged))
                try {
                    got = outs.single().toBf16Array(n)
                } finally {
                    outs.forEach { it.close() }
                }
            } finally {
                staged.close()
            }
            for (i in probes.indices) {
                val want = floatToBf16Bits(probes[i])
                assertTrue(
                    got[i] == want,
                    "narrow lane $i (probe bits 0x${probes[i].toRawBits().toUInt().toString(16)}): " +
                        "GPU pattern 0x${got[i].toUShort().toString(16)} != host RNE " +
                        "0x${want.toUShort().toString(16)} — device narrowing disagrees with §0.4.455",
                )
            }
            // The NaN lanes, pinned only as far as semantics guarantee.
            val nanProbes = floatArrayOf(Float.NaN, Float.fromBits(0x7FAA0000))
            val nanFn = DxirBuilder.function("bf16_nan_narrow") {
                val x = param("x", DxirType(F32, listOf(2)))
                listOf(op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(2))))
            }
            val nanStaged = session.bufferFromHostF32(nanProbes, listOf(2))
            try {
                val outs = session.executeOn(nanFn, listOf(nanStaged))
                try {
                    val nans = bf16BitsToFloatArray(outs.single().toBf16Array(2))
                    assertTrue(
                        nans[0].isNaN() && nans[1].isNaN(),
                        "NaN decayed through the bf16 convert: ${nans.toList()}",
                    )
                } finally {
                    outs.forEach { it.close() }
                }
            } finally {
                nanStaged.close()
            }
            println("[pjrt-bf16] device f32→bf16 narrowing BIT-EXACT vs G1a RNE on GB10 ($n lanes incl. ties)")
        }
    }

    /**
     * Native BF16 buffers: raw patterns up, TRUE bf16 storage on device, raw
     * patterns back. ADD(x, 0) is the identity for every non-zero lane, so
     * the returned patterns must be the uploaded ones bit-for-bit (no -0
     * lanes: -0 + +0 = +0 would legitimately flip the sign bit). The staged
     * buffer's on-device size must be 2 bytes/element — the proof the device
     * did not silently widen to f32.
     */
    @Test
    fun nativeBf16BuffersRoundTripExactly() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val values = floatArrayOf(1f, -2.5f, 0.0078125f, 340f, -1.0078125f, 65280f)
        val patterns = floatArrayToBf16Bits(values)
        val n = values.size
        val bvec = DxirType(BF16, listOf(n))
        val fn = DxirBuilder.function("bf16_identity") {
            val x = param("x", bvec)
            val z = param("z", bvec)
            listOf(op(OpKind.ADD, listOf(x, z), bvec))
        }

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            session.bufferFromHostBf16(patterns, listOf(n)).use { staged ->
                val deviceBytes = staged.deviceSizeInBytes()
                assertTrue(
                    deviceBytes in (2L * n) until (4L * n),
                    "staged bf16 buffer is $deviceBytes bytes for $n elements — not 2-byte storage",
                )
                println("[pjrt-bf16] staged bf16 buffer: $deviceBytes bytes for $n elements (true 2-byte device storage)")
            }
            val got = session.runOnBf16(fn, listOf(patterns, ShortArray(n)))
            assertEquals(1, got.size)
            for (i in 0 until n) {
                assertEquals(
                    patterns[i], got[0][i],
                    "native bf16 lane $i: uploaded 0x${patterns[i].toUShort().toString(16)} came back " +
                        "0x${got[0][i].toUShort().toString(16)} — the buffer path corrupts patterns",
                )
            }
            println("[pjrt-bf16] native BF16 buffer round-trip (type enum 13) BIT-EXACT on GB10")
        }
    }

    /**
     * A bf16 matmul+add graph vs the interpreter's compute-in-f32-store-bf16
     * convention, on NATIVE bf16 buffers, in two parts:
     *
     * **Part 1 — the bit-pinnable claim.** All operands bf16-exact, every
     * dot product and sum exactly representable in bf16, so per-op rounding
     * and fused-f32 evaluation give THE SAME bits (products of two 8-bit
     * mantissas are exact in f32's 24; the sums are exact). Whatever XLA
     * fuses, the output must bit-match the interpreter — pinned bit-exact.
     *
     * **Part 2 — the convention discriminator G1b left open.** matmul
     * [[256, 1]]·[[1],[1]] = 257, a bf16 TIE (halfway between 256 and 258),
     * then +1. Per-op rounding (the interpreter): 257 snaps to 256 (even),
     * +1 = 257 snaps again → 256. Fused f32 (no intermediate snap):
     * 257 + 1 = 258, exact → 258. The MEASURED GB10 answer, asserted here:
     * **256** — XLA-CUDA rounds the dot's output to bf16 before the add
     * (f32 accumulation INSIDE the dot, bf16 at the op boundary), i.e. the
     * interpreter's per-op snap IS what this device does on this graph.
     * The §0.4.456 caveat stands in general (op-boundary rounding is XLA's
     * choice, not a spec — see the convert-fold finding in the cast test),
     * so graphs where fusion COULD keep f32 across op boundaries still get
     * a one-bf16-ulp tolerance, but the dot+add boundary is pinned to the
     * per-op convention as measured.
     */
    @Test
    fun matmulAddIntermediateRoundingConvention() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            // Part 1: exact lanes — bit-pinned against the interpreter.
            val aT = DxirType(BF16, listOf(2, 3))
            val bT = DxirType(BF16, listOf(3, 2))
            val cT = DxirType(BF16, listOf(2, 2))
            val fn = DxirBuilder.function("bf16_matmul_add") {
                val a = param("a", aT)
                val b = param("b", bT)
                val c = param("c", cT)
                val mm = op(OpKind.MATMUL, listOf(a, b), cT)
                listOf(op(OpKind.ADD, listOf(mm, c), cT))
            }
            val a = floatArrayOf(1f, 2f, 4f, 0.5f, -3f, 8f)
            val b = floatArrayOf(2f, 16f, 4f, 0.25f, 1f, -2f)
            val c = floatArrayOf(0.5f, 1f, -1f, 2f)
            val want = DxirInterpreter.evalFunction(fn, listOf(a, b, c)).single()
            val got = session.runOnBf16(
                fn,
                listOf(floatArrayToBf16Bits(a), floatArrayToBf16Bits(b), floatArrayToBf16Bits(c)),
            ).single()
            val gotF = bf16BitsToFloatArray(got)
            for (i in want.indices) {
                assertTrue(
                    gotF[i].toRawBits() == want[i].toRawBits(),
                    "exact-lane $i: GPU ${gotF[i]} != interpreter ${want[i]} — " +
                        "bf16-exact matmul+add must be engine-independent",
                )
            }
            println("[pjrt-bf16] bf16 matmul+add (exact lanes) BIT-EXACT vs interpreter on GB10")

            // Part 2: the tie discriminator.
            val a1 = DxirType(BF16, listOf(1, 2))
            val b1 = DxirType(BF16, listOf(2, 1))
            val c1 = DxirType(BF16, listOf(1, 1))
            val disc = DxirBuilder.function("bf16_tie_disc") {
                val aa = param("a", a1)
                val bb = param("b", b1)
                val cc = param("c", c1)
                val mm = op(OpKind.MATMUL, listOf(aa, bb), c1)
                listOf(op(OpKind.ADD, listOf(mm, cc), c1))
            }
            val ins = listOf(
                floatArrayToBf16Bits(floatArrayOf(256f, 1f)),
                floatArrayToBf16Bits(floatArrayOf(1f, 1f)),
                floatArrayToBf16Bits(floatArrayOf(1f)),
            )
            val hostConvention = DxirInterpreter.evalFunction(
                disc,
                listOf(floatArrayOf(256f, 1f), floatArrayOf(1f, 1f), floatArrayOf(1f)),
            ).single()[0] // per-op snap: 257→256 (tie, even), +1=257→256
            val device = bf16BitsToFloat(session.runOnBf16(disc, ins).single()[0])
            println(
                "[pjrt-bf16] intermediate-rounding discriminator: interpreter=$hostConvention " +
                    "device=$device (per-op snap would give 256, fused f32 gives 258)",
            )
            assertEquals(
                256f, hostConvention,
                "interpreter convention drifted — the per-op snap should tie 257 down to 256",
            )
            assertEquals(
                256f, device,
                "GB10 measured answer changed: XLA-CUDA rounded the dot output to bf16 before the " +
                    "add (256, the per-op convention) when certified; a fused-f32 result (258) means " +
                    "the fusion behavior moved — re-certify the convention",
            )
        }
    }

    /**
     * A GRADIENT graph with bf16 compute — the CastRule straight-through
     * adjoint on real XLA, cast-at-boundary form: f32 params → CAST bf16 →
     * MUL → SUM (bf16 scalar) → CAST f32 loss. DxirReverseTransform emits
     * the reverse (no gradient math here — the one-engine rule); the GPU
     * grads are compared to the interpreter.
     *
     * THE MEASURED DIVERGENCE, pinned as the honest tolerance: in the
     * reverse graph d w = widen(bf16_mul(1, narrow(x))); XLA folds the
     * mul-by-one and then ELIDES the narrow→widen convert pair, so the
     * device adjoint sees RAW x where the interpreter (which snaps at
     * every bf16 node) sees snap(x). On the non-representable lane
     * x[0] = 1.005 that is |1.0078125 − 1.005| = 0.0028125 — exactly the
     * snap error, bounded by one bf16 ulp (< 2⁻⁸ relative). The suite pins
     * max|diff| at 2⁻⁸·max|operand| = 1/256 here (measured 0.0028125) and
     * the raw-x lane explicitly; the bf16-exact lanes (d x = w, and the
     * exact d w lanes) must agree BIT-EXACTLY — divergence only ever comes
     * from elided narrowings, never from the adjoint math. This is the
     * device-vs-convention gap G1b flagged, now measured and bounded, and
     * it is a GPU-plugin observation — G2b re-measures on TPU.
     */
    @Test
    fun bf16GradientGraphMatchesInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val vec = DxirType(F32, listOf(4))
        val bvec = DxirType(BF16, listOf(4))
        val bScalar = DxirType(BF16, emptyList())
        val fScalar = DxirType(F32, emptyList())
        val fn = DxirBuilder.function("bf16_grad_probe") {
            val x = param("x", vec)
            val w = param("w", vec)
            val xb = op(OpKind.CAST, listOf(x), bvec)
            val wb = op(OpKind.CAST, listOf(w), bvec)
            val p = op(OpKind.MUL, listOf(xb, wb), bvec)
            val s = op(OpKind.SUM, listOf(p), bScalar)
            listOf(op(OpKind.CAST, listOf(s), fScalar))
        }
        // x[0] = 1.005 is NOT bf16-representable — it must reach d w as the
        // snapped 1.0078125, on both engines. Everything else is exact.
        val x = floatArrayOf(1.005f, -2f, 0.25f, 3f)
        val w = floatArrayOf(2f, 0.5f, -8f, 1.25f)

        val grad = DxirReverseTransform.apply(fn)
        val want = DxirInterpreter.evalFunction(grad, listOf(x, w))
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(grad, listOf(x, w))
            var maxDiff = 0f
            var anyNonZero = false
            for (k in want.indices) {
                for (i in want[k].indices) {
                    maxDiff = maxOf(maxDiff, abs(got[k][i] - want[k][i]))
                    if (got[k][i] != 0f) anyNonZero = true
                }
            }
            println("[pjrt-bf16] bf16 gradient graph max|diff|=$maxDiff vs interpreter on GB10")
            assertTrue(anyNonZero, "bf16 gradient came back all-zero — the silent-zero failure mode")
            // One bf16 ulp at the operands' magnitude (~1): the elided-narrowing
            // bound. Measured 0.0028125 (exactly the x[0] snap error).
            assertTrue(
                maxDiff <= 1f / 256f,
                "bf16 gradient diverges beyond one bf16 ulp — this is no longer the convert-fold: $maxDiff",
            )
            // The bf16-exact lanes must agree bit-for-bit: d x = w (all exact),
            // and every d w lane whose x is bf16-representable.
            for (i in 0 until 4) {
                assertTrue(
                    got[0][i].toRawBits() == want[0][i].toRawBits(),
                    "d x[$i]: GPU ${got[0][i]} != interpreter ${want[0][i]} — exact lanes may not diverge",
                )
            }
            for (i in 1 until 4) {
                assertTrue(
                    got[1][i].toRawBits() == want[1][i].toRawBits(),
                    "d w[$i]: GPU ${got[1][i]} != interpreter ${want[1][i]} — exact lanes may not diverge",
                )
            }
            // The measured convert-fold pin: the device adjoint reads RAW x[0]
            // (XLA elided the narrow→widen pair); the interpreter reads the
            // snapped 1.0078125. Both are recorded; the gap IS the tolerance.
            assertEquals(
                1.005f, got[1][0],
                "d w[0] on device should be RAW x[0] (the measured XLA convert-fold); " +
                    "got ${got[1][0]} — the simplifier behavior moved, re-certify the bound",
            )
            assertEquals(
                1.0078125f, want[1][0],
                "interpreter d w[0] should be the snapped x[0] — the §0.4.456 convention drifted",
            )
        }
    }
}
