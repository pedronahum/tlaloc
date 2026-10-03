package io.tlaloc.core

sealed interface DType {
    val sizeBytes: Int
    val name: String
}

data object F32 : DType {
    override val sizeBytes = 4
    override val name = "f32"
}

data object F64 : DType {
    override val sizeBytes = 8
    override val name = "f64"
}

/**
 * bfloat16: the truncated-f32 format TPUs are built
 * around and Blackwell tensor cores prefer. bf16 IS the top 16 bits of an
 * IEEE-754 binary32 (1 sign + 8 exponent + 7 mantissa), so the host
 * representation is the raw upper-16-bit pattern in a Short (see
 * [HostBf16Storage] and Bf16.kt for the conversions). No Kotlin primitive
 * exists at this width — the storage carries BIT PATTERNS, never numeric
 * Short values; all host arithmetic happens in f32 (compute-in-f32,
 * store-bf16 — the convention recorded in Bf16.kt).
 */
data object BF16 : DType {
    override val sizeBytes = 2
    override val name = "bf16"
}

/**
 * Signed 8-bit integer. Used for weight-only quantized projections: the
 * weights are stored as int8 codes and widened in the graph (exactly, every
 * code is representable in bf16 and f32) before the matmul.
 */
data object I8 : DType {
    override val sizeBytes = 1
    override val name = "i8"
}

/**
 * float8 e4m3fn (1 sign, 4 exponent bits with bias 7, 3 mantissa bits; no
 * infinities, NaN at 0x7F/0xFF; largest finite 448). Used for weights staged
 * as f8 codes with a scale per output channel, and read from fp8 checkpoints.
 * Host storage is the raw byte ([HostBytesStorage]); see Fp8.kt for the
 * conversions.
 */
data object F8E4M3FN : DType {
    override val sizeBytes = 1
    override val name = "f8e4m3fn"
}

/**
 * Unsigned bytes, read from a checkpoint whose producer packs codes into them
 * (NVFP4's two e2m1 values per byte). Never computed on; decoded by the
 * reader that knows the packing.
 */
data object U8 : DType {
    override val sizeBytes = 1
    override val name = "u8"
}

data object I32 : DType {
    override val sizeBytes = 4
    override val name = "i32"
}

data object I64 : DType {
    override val sizeBytes = 8
    override val name = "i64"
}

data object Bool : DType {
    override val sizeBytes = 1
    override val name = "bool"
}
