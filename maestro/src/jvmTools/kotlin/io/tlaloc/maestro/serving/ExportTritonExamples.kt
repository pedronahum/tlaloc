package io.tlaloc.maestro.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path

/**
 * Entry point of `./gradlew :maestro:exportTritonExamples -PoutDir=<triton/examples>`.
 *
 * Writes the example model repository that the Triton `tlaloc` backend serves
 * (`triton/examples/model_repository/`) and the reference values its verify
 * script checks against (`triton/examples/reference/`).
 *
 * Three models:
 *
 * - `matmul_sumsq`: `f(A) = sum(A · A)` for a 2x2 `A`, reverse-transformed with
 *   the forward value kept, so the served function returns `(f(A), df/dA)`.
 *   `df/dA = 1 · Aᵀ + Aᵀ · 1`, which for `A = [[1,2],[3,4]]` is `[[7,11],[9,13]]`
 *   and `f(A) = 54`.
 * - `dtypes`: `x · x + x` over f64, bf16 and i32 vectors of length 4, one
 *   output per dtype. It exists to exercise the backend's non-f32 paths.
 * - `buckets`: `x · x + x` over f32 vectors, emitted twice, for length 4 and
 *   for length 8. The model serves both files, one per shape.
 *
 * The reference file for `matmul_sumsq` holds the DXIR interpreter's result
 * for the same graph that was emitted, so the served value is compared
 * against Tlaloc's own evaluation and not only against a hand-written number.
 */
fun main(args: Array<String>) {
    val out = Path.of(args.firstOrNull() ?: "triton/examples")
    val repo = out.resolve("model_repository")
    val reference = out.resolve("reference")
    Files.createDirectories(reference)

    // matmul_sumsq --------------------------------------------------------
    val m22 = DxirType(F32, listOf(2, 2))
    val scalar = DxirType(F32, emptyList())
    val primal = DxirBuilder.function("matmul_sumsq") {
        val a = param("A", m22)
        val aa = op(OpKind.MATMUL, listOf(a, a), m22)
        listOf(op(OpKind.SUM, listOf(aa), scalar))
    }
    val valueAndGrad: DxirFunction = DxirReverseTransform.apply(primal, includeForward = true)
    val input = floatArrayOf(1f, 2f, 3f, 4f)
    val interpreted = DxirInterpreter.evalFunction(valueAndGrad, listOf(input))
    writeModule(repo.resolve("matmul_sumsq/1/model.mlir"), valueAndGrad.toStablehlo())
    Files.writeString(
        reference.resolve("matmul_sumsq.json"),
        buildString {
            append("{\n")
            append("  \"source\": \"DXIR interpreter, exportTritonExamples\",\n")
            append("  \"inputs\": {\"A\": {\"shape\": [2, 2], \"data\": ${floats(input)}}},\n")
            append("  \"outputs\": {\n")
            append("    \"VALUE\": {\"shape\": [], \"data\": ${floats(interpreted[0])}},\n")
            append("    \"GRAD\": {\"shape\": [2, 2], \"data\": ${floats(interpreted[1])}}\n")
            append("  }\n")
            append("}\n")
        },
    )
    println("matmul_sumsq: interpreter value=${interpreted[0].toList()} grad=${interpreted[1].toList()}")

    // dtypes --------------------------------------------------------------
    val dtypes = DxirBuilder.function("dtypes") {
        listOf(F64, BF16, I32).mapIndexed { i, dt ->
            val t = DxirType(dt, listOf(4))
            val x = param("x$i", t)
            op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, x), t), x), t)
        }
    }
    writeModule(repo.resolve("dtypes/1/model.mlir"), dtypes.toStablehlo())

    // buckets -------------------------------------------------------------
    for (n in listOf(4, 8)) {
        val bucket = DxirBuilder.function("square_plus_$n") {
            val t = DxirType(F32, listOf(n))
            val x = param("x", t)
            listOf(op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, x), t), x), t))
        }
        writeModule(repo.resolve("buckets/1/len$n.mlir"), bucket.toStablehlo())
    }
    println("wrote ${repo.toAbsolutePath()}")
}

private fun writeModule(path: Path, mlir: String) {
    Files.createDirectories(path.parent)
    Files.writeString(path, if (mlir.endsWith("\n")) mlir else mlir + "\n")
    println("  $path")
}

private fun floats(a: FloatArray): String = a.joinToString(", ", "[", "]") { it.toString() }
