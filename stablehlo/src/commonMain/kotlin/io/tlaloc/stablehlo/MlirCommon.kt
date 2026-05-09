package io.tlaloc.stablehlo

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirType

/**
 * MLIR formatting helpers shared across emitters in this module.
 *
 * Currently only [StablehloEmitter] consumes these. They live in their own
 * file so a parallel emitter for a different MLIR dialect (Nagual, Linalg)
 * can pull them in by import without touching StableHLO-specific code.
 *
 * Nothing here is StableHLO-specific: the tensor-type printer, dtype-to-
 * MLIR-element-type map, dense-literal serialiser, and infinity-bit-pattern
 * literals are all generic MLIR-text utilities.
 */

/**
 * Print a [DxirType] in MLIR tensor syntax: `tensor<f32>` (rank 0),
 * `tensor<RxCxf32>` (rank 2), etc. The element-type segment matches what
 * MLIR's parser expects for any dialect built on the standard tensor type.
 */
fun DxirType.toMlir(): String {
    val elem = mlirElementType(dtype)
    return if (dims.isEmpty()) "tensor<$elem>"
    else "tensor<${dims.joinToString("x")}x$elem>"
}

internal fun mlirElementType(dtype: DType): String = when (dtype) {
    is F32 -> "f32"
    is F64 -> "f64"
    is I32 -> "i32"
    is I64 -> "i64"
    is Bool -> "i1"
}

/**
 * Format a row-major [FloatArray] as an MLIR dense literal matching [dims].
 * Rank-1 produces `[1.0, 2.0, 3.0]`; rank-2 produces `[[1.0, 2.0], [3.0, 4.0]]`;
 * rank-N is recursive. The output is the body that sits inside `dense<...>`.
 */
internal fun denseFromArray(values: FloatArray, dims: List<Int>): String {
    require(dims.isNotEmpty()) { "denseFromArray: empty dims (use the scalar arm instead)" }
    if (dims.size == 1) return values.joinToString(prefix = "[", postfix = "]") { it.toString() }
    val outer = dims[0]
    val inner = dims.drop(1)
    val chunkSize = values.size / outer
    val chunks = (0 until outer).map { i ->
        val slice = FloatArray(chunkSize) { j -> values[i * chunkSize + j] }
        denseFromArray(slice, inner)
    }
    return chunks.joinToString(prefix = "[", postfix = "]")
}

/**
 * MLIR bit-pattern literal for `-Inf` / `Int.MIN` / `false` for the given
 * [dtype]. Used as the identity element for MAX-style reductions.
 */
internal fun negInfLiteral(dtype: DType): String = when (dtype) {
    is F32 -> "0xFF800000"
    is F64 -> "0xFFF0000000000000"
    is I32 -> Int.MIN_VALUE.toString()
    is I64 -> Long.MIN_VALUE.toString()
    is Bool -> "false"
}

/**
 * MLIR bit-pattern literal for `+Inf` / `Int.MAX` / `true` for the given
 * [dtype]. Used as the identity element for MIN-style reductions.
 */
internal fun posInfLiteral(dtype: DType): String = when (dtype) {
    is F32 -> "0x7F800000"
    is F64 -> "0x7FF0000000000000"
    is I32 -> Int.MAX_VALUE.toString()
    is I64 -> Long.MAX_VALUE.toString()
    is Bool -> "true"
}
