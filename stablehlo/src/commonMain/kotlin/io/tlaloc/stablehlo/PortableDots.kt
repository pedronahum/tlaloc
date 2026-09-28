package io.tlaloc.stablehlo

/**
 * [mlir] (from [toStablehlo]) with every f32 dot algorithm replaced by
 * `precision = [HIGHEST, HIGHEST]`.
 *
 * The emitter spells some f32 dots as an explicit dot algorithm (f32
 * operands, products and sums), which XLA's GPU backend runs as its own GEMM.
 * Which algorithms a backend accepts is backend-specific; `HIGHEST` precision
 * is accepted everywhere and asks for the same f32 arithmetic (on a TPU, as
 * several bf16 passes). A TPU session lowers through this.
 */
fun portableF32Dots(mlir: String): String = mlir.replace(F32_DOT_ALGORITHM, ", precision = [HIGHEST, HIGHEST]")
