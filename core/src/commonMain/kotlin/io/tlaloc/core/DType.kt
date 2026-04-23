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
