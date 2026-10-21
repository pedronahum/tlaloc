package io.tlaloc.stablehlo

/** The `precision` a portable f32 dot asks for. On a TPU, HIGHEST is six bf16 passes on the MXU,
 * HIGH three, DEFAULT one; on a GPU, HIGHEST is f32 arithmetic and DEFAULT allows TF32. */
enum class F32DotPrecision { HIGHEST, HIGH, DEFAULT }

/**
 * [mlir] (from [toStablehlo]) with every f32 dot algorithm replaced by
 * `precision = [P, P]` for [precision] (HIGHEST unless asked).
 *
 * The emitter spells some f32 dots as an explicit dot algorithm (f32
 * operands, products and sums), which XLA's GPU backend runs as its own GEMM.
 * Which algorithms a backend accepts is backend-specific; `HIGHEST` precision
 * is accepted everywhere and asks for the same f32 arithmetic (on a TPU, as
 * several bf16 passes). A TPU session lowers through this. Parity checks keep
 * HIGHEST; a TPU serving or training lane that accepts bf16-level error in
 * these dots chooses HIGH or DEFAULT for a third or a sixth of the MXU work.
 */
fun portableF32Dots(mlir: String, precision: F32DotPrecision = F32DotPrecision.HIGHEST): String =
    mlir.replace(F32_DOT_ALGORITHM, ", precision = [${precision.name}, ${precision.name}]")
