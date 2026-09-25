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
 * Four models:
 *
 * - `matmul_sumsq`: `f(A) = sum(A · A)` for a 2x2 `A`, reverse-transformed with
 *   the forward value kept, so the served function returns `(f(A), df/dA)`.
 *   `df/dA = 1 · Aᵀ + Aᵀ · 1`, which for `A = [[1,2],[3,4]]` is `[[7,11],[9,13]]`
 *   and `f(A) = 54`.
 * - `dtypes`: `x · x + x` over f64, bf16 and i32 vectors of length 4, one
 *   output per dtype. It exists to exercise the backend's non-f32 paths.
 * - `buckets`: `x · x + x` over f32 vectors, emitted twice, for length 4 and
 *   for length 8. The model serves both files, one per shape.
 * - `reference_decode`: the reference decode graph's serving artifact (one
 *   attention layer over a paged KV cache, six compiled batch/context
 *   entries), written by [TritonModelRepository.write] in client mode. Its
 *   `config.pbtxt` is generated from the manifest rather than written by hand.
 * - `reference_sequence`: the same artifact in sequence mode (the backend
 *   keeps each sequence's KV pages by correlation ID).
 *
 * The reference files hold the DXIR interpreter's results for the same graphs
 * that were emitted, so served values are compared against Tlaloc's own
 * evaluation and not only against hand-written numbers. For
 * `reference_decode` that is a sequence of decode steps with the KV pools
 * carried from one step to the next, as the backend carries them.
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

    // reference_decode ----------------------------------------------------
    val artifact = Files.createTempDirectory("tlaloc-reference-decode")
    try {
        ReferenceDecodeGraph.exportTo(artifact)
        val model = TritonModelRepository.write(
            artifact, repo, "reference_decode", TritonModelRepository.KvMode.CLIENT,
        )
        println("  $model")
        // The same artifact in sequence mode: the backend keeps the pages.
        val sequence = TritonModelRepository.write(
            artifact, repo, "reference_sequence", TritonModelRepository.KvMode.SEQUENCE,
        )
        println("  $sequence")
    } finally {
        artifact.toFile().deleteRecursively()
    }
    Files.writeString(reference.resolve("reference_decode.json"), referenceDecodeSteps())
    println("wrote ${repo.toAbsolutePath()}")
}

private fun writeModule(path: Path, mlir: String) {
    Files.createDirectories(path.parent)
    Files.writeString(path, if (mlir.endsWith("\n")) mlir else mlir + "\n")
    println("  $path")
}

private fun floats(a: FloatArray): String = a.joinToString(", ", "[", "]") { it.toString() }

private fun ints(a: FloatArray): String = a.joinToString(", ", "[", "]") { it.toInt().toString() }

/**
 * Decode steps of the reference graph, evaluated by the DXIR interpreter with
 * the KV pools starting at zero and carried from each step to the next.
 *
 * One sequence of four tokens on pages 1 and 2 through the (batch 1,
 * context 4) entry, then two new sequences, one token each, on pages 3 and
 * 4 through the (batch 2, context 2) entry. The second pair reads pools that
 * already hold the first sequence's keys and values.
 */
private fun referenceDecodeSteps(): String {
    val model = ReferenceDecodeGraph.MODEL
    val specs = ServingArtifactWriter.decodeSpecs(model, ReferenceDecodeGraph.POLICY)
    val poolSize = model.numBlocks * model.blockSize * model.numKvHeads * model.headDim
    var key = FloatArray(poolSize)
    var value = FloatArray(poolSize)
    val bs = model.blockSize

    class Step(val batch: Int, val context: Int, val tokens: List<Int>, val positions: List<Int>,
               val tables: List<List<Int>>)
    val steps = listOf(3, 7, 1, 9).mapIndexed { p, t ->
        Step(1, 4, listOf(t), listOf(p), listOf(listOf(1, 2)))
    } + Step(2, 2, listOf(4, 10), listOf(0, 0), listOf(listOf(3), listOf(4)))

    val out = StringBuilder()
    out.append("{\n  \"source\": \"DXIR interpreter, exportTritonExamples\",\n")
    out.append("  \"steps\": [\n")
    steps.forEachIndexed { k, s ->
        val spec = specs.single { it.bucket.batch == s.batch && it.bucket.maxContext == s.context }
        val fn = ReferenceDecodeGraph.build(spec)
        val tokenIds = FloatArray(s.batch) { s.tokens[it].toFloat() }
        val positions = FloatArray(s.batch) { s.positions[it].toFloat() }
        val tables = FloatArray(s.batch * spec.maxBlocksPerSeq) {
            s.tables[it / spec.maxBlocksPerSeq][it % spec.maxBlocksPerSeq].toFloat()
        }
        val seqLens = FloatArray(s.batch) { (s.positions[it] + 1).toFloat() }
        val slots = FloatArray(s.batch) {
            val p = s.positions[it]
            (s.tables[it][p / bs] * bs + p % bs).toFloat()
        }
        val result = DxirInterpreter.evalFunction(
            fn, listOf(tokenIds, positions, tables, seqLens, slots, key, value),
        )
        key = result[1]
        value = result[2]
        val v = model.vocabSize
        out.append("    {\"inputs\": {")
        out.append("\"tokenIds\": {\"shape\": [${s.batch}, 1], \"data\": ${ints(tokenIds)}}, ")
        out.append("\"positions\": {\"shape\": [${s.batch}, 1], \"data\": ${ints(positions)}}, ")
        out.append("\"blockTables\": {\"shape\": [${s.batch}, ${spec.maxBlocksPerSeq}], ")
        out.append("\"data\": ${ints(tables)}}, ")
        out.append("\"seqLens\": {\"shape\": [${s.batch}], \"data\": ${ints(seqLens)}}, ")
        out.append("\"slotMapping\": {\"shape\": [${s.batch}], \"data\": ${ints(slots)}}},\n")
        out.append("     \"outputs\": {\"logits\": {\"shape\": [${s.batch}, 1, $v], ")
        out.append("\"data\": ${floats(result[0])}}}}")
        out.append(if (k == steps.lastIndex) "\n" else ",\n")
    }
    out.append("  ]\n}\n")
    return out.toString()
}
