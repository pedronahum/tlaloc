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
