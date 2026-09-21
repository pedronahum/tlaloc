package io.tlaloc.core

// §0.4.455 (Phase G1a) — the bf16 foundation: bit conversions and cast
// surfaces. This file is the design record for the host side of bf16 (the
// gap open since §0.4.354: "needs a host-side representation design — no
// Kotlin primitive").
//
// REPRESENTATION. bf16 is the top 16 bits of IEEE-754 binary32: 1 sign,
// 8 exponent (same bias 127 as f32 — same dynamic range, which is WHY the
// format wins for training), 7 mantissa. Host storage is a ShortArray of
// raw bit patterns ([HostBf16Storage]); a Short is a 16-bit bucket, never
// a number. REJECTED alternatives, for the record:
//   - a `value class Bf16(val bits: Short)` element type: pleasant API but
//     KMP boxing at array granularity and no arithmetic anyway (host math
//     is f32); the raw ShortArray is the honest shape.
//   - CharArray / IntArray-packed-pairs storage: same information, worse
//     spelling; ShortArray states the 2-byte width directly.
//   - storing widened FloatArray and narrowing lazily: lies about
//     sizeBytes, and round-trip semantics (narrowing rounds!) would be
//     invisible until emission. Storage holds what the dtype claims.
//
// NARROWING CONVENTION. f32 -> bf16 uses round-to-nearest-even on the
// discarded low 16 bits, matching XLA (Eigen::bfloat16, and what a TPU or
// a Blackwell tensor core does on convert). The implementation is the
// standard bias trick: add 0x7FFF + (lsb of the kept part) to the raw u32
// and shift; the carry propagates a tie upward exactly when the kept lsb
// is odd. Consequences, all pinned in Bf16Test:
//   - ties round to the even 16-bit pattern;
//   - overflow past the largest finite bf16 carries into the exponent and
//     lands on infinity (f32 Float.MAX_VALUE -> +inf), as RNE requires;
//   - infinities pass through (mantissa 0, no carry reaches the exponent);
//   - signed zeros keep their sign bit;
//   - f32 subnormals round like any other value (the tiny ones flush to
//     signed zero by ROUNDING, not by an explicit flush branch);
//   - NaN is handled BEFORE the bias add (the add could carry a signaling
//     NaN's mantissa into infinity): the payload's top bits are kept and
//     the quiet bit 0x0040 is forced, so every NaN stays a NaN.
// WIDENING is exact: bf16 -> f32 is `bits << 16` — every bf16 value is an
// f32 value. Round-trip bf16 -> f32 -> bf16 is the identity (pinned by the
// 256-exponent sweep below).
//
// HOST COMPUTE CONVENTION (the v1 decision): compute-in-f32, store-bf16.
// A DTensor<S, BF16> is a storage/interchange type; host math goes through
// the cast surfaces ([toF32], [toBf16]) and the existing f32 kernels. Per-op
// bf16 host twins (add/matmul/... reading ShortArray directly) are a NAMED
// DEFERRAL: they would duplicate every host op for zero host-precision
// benefit — bf16 arithmetic units exist on accelerators, not in the JVM,
// and PyTorch/CPU does the same widen-compute-narrow dance. IR emission of
// real `bf16` element types (and mixed-precision accumulation policy) is
// G1b, deliberately not touched here.

/**
 * Narrow one f32 to its bf16 bit pattern, round-to-nearest-even (XLA's
 * convention). NaN payloads are truncated-and-quieted, never silently
 * un-NaN'd. The returned Short is a raw bit pattern.
 */
fun floatToBf16Bits(v: Float): Short {
    val bits = v.toRawBits()
    if (v.isNaN()) {
        // Keep sign + top payload bits, force the quiet bit so a payload
        // whose surviving mantissa bits are all zero cannot become an inf.
        return ((bits ushr 16) or 0x0040).toShort()
    }
    // RNE via bias-and-carry: Int addition is modular u32 addition, and no
    // non-NaN input (max 0xFF80_0000) can wrap.
    val lsb = (bits ushr 16) and 1
    return ((bits + 0x7FFF + lsb) ushr 16).toShort()
}

/** Widen one bf16 bit pattern to f32 — exact, every bf16 value is an f32. */
fun bf16BitsToFloat(bits: Short): Float = Float.fromBits(bits.toInt() shl 16)

/** Narrow a whole FloatArray to bf16 bit patterns ([floatToBf16Bits] each). */
fun floatArrayToBf16Bits(data: FloatArray): ShortArray =
    ShortArray(data.size) { floatToBf16Bits(data[it]) }

/** Widen a whole bf16 pattern array to f32 ([bf16BitsToFloat] each). */
fun bf16BitsToFloatArray(bits: ShortArray): FloatArray =
    FloatArray(bits.size) { bf16BitsToFloat(bits[it]) }

/**
 * Narrow an f32 tensor to bf16 storage (RNE per element). Shape is
 * preserved; this is the entry point of the compute-in-f32-store-bf16
 * convention.
 */
fun <S : Shape> DTensor<S, F32>.toBf16(): DTensor<S, BF16> =
    DTensor(HostBf16Storage(floatArrayToBf16Bits(hostF32())), dims.copyOf(), BF16)

/**
 * Widen a bf16 tensor to f32 storage — exact. Host math on bf16 tensors is
 * spelled `t.toF32() ... .toBf16()`; there are no bf16 host kernels (named
 * deferral, see the file header).
 */
fun <S : Shape> DTensor<S, BF16>.toF32(): DTensor<S, F32> =
    DTensor(HostF32Storage(bf16BitsToFloatArray(hostBf16())), dims.copyOf(), F32)

/** [hostF32]'s bf16 twin: the raw bit-pattern array behind a bf16 tensor. */
fun DTensor<*, BF16>.hostBf16(): ShortArray {
    val s = storage
    require(s is HostBf16Storage) {
        "operation requires HostBf16Storage, got ${s::class.simpleName}"
    }
    return s.data
}
