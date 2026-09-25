package io.tlaloc.maestro.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.inference.AttentionKind
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecoderLayerSpec
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfModelFamily
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random

/**
 * Entry point of `./gradlew :maestro:exportTritonExamples -PoutDir=<triton/examples>`.
 *
 * Writes the example model repository that the Triton `tlaloc` backend serves
 * (`triton/examples/model_repository/`) and the reference values its verify
 * script checks against (`triton/examples/reference/`).
 *
 * The models:
 *
 * - `matmul_sumsq`: `f(A) = sum(A · A)` for a 2x2 `A`, reverse-transformed with
 *   the forward value kept, so the served function returns `(f(A), df/dA)`.
 *   `df/dA = 1 · Aᵀ + Aᵀ · 1`, which for `A = [[1,2],[3,4]]` is `[[7,11],[9,13]]`
 *   and `f(A) = 54`.
 * - `dtypes`: `x · x + x` over f64, bf16 and i32 vectors of length 4, one
 *   output per dtype. It exists to exercise the backend's non-f32 paths.
 * - `buckets`: `x · x + x` over f32 vectors, emitted twice, for length 4 and
 *   for length 8. The model serves both files, one per shape.
 * - `int64_bool`: i64 and bool tensors: `x · x + x` over i64, `(x · x > x) and b`
 *   and `not b`. The i64 values go past the i32 range.
 * - `grad_batched` and `grad_unbatched`: the gradient of `sum(A · A)` for a
 *   batch of 2x2 matrices, emitted for batch 1, 2, 4 and 8. The two models
 *   serve the same files; only `grad_batched` turns on Triton's dynamic
 *   batching. Rows are independent, so each row is `df/dA` of its own `A`.
 * - `ragged_batched` and `ragged_consecutive`: `x · x + x` over rows of 4 or
 *   of 8 f32 values, emitted for batch 1, 2, 4 and 8 at each width. Both
 *   serve the same files with dynamic batching of ragged requests, so one
 *   batch can hold requests of both widths. `ragged_batched` runs each width
 *   of a batch as one execution; `ragged_consecutive` sets `group_by_shape`
 *   false and runs only consecutive requests of one width together.
 * - `large_io` and `large_io_host`: `x · x + x` over 4Mi f32 values (16 MiB in,
 *   16 MiB out), for measuring what reading and writing GPU memory in place
 *   saves; `large_io_host` sets `zero_copy` false.
 * - `reference_decode`: the reference decode graph's serving artifact (one
 *   attention layer over a paged KV cache, six compiled batch/context
 *   entries), written by [TritonModelRepository.write] in client mode. Its
 *   `config.pbtxt` is generated from the manifest rather than written by hand.
 * - `reference_sequence`: the same artifact in sequence mode (the backend
 *   keeps each sequence's KV pages by correlation ID).
 * - `window_sequence` and `window_sequence_full`: a three-layer decoder with
 *   seeded random weights whose first two layers attend over a sliding
 *   window of 8 positions, in pages of 4, up to 64 positions. In
 *   `window_sequence` the two sliding layers keep their KV in a windowed pool
 *   (a ring of 3 pages per sequence); in `window_sequence_full` every layer
 *   keeps full-history pages. The two must give the same logits.
 *   `window_sequence/1/tlaloc-serving-short-ring.json` is its manifest with a
 *   ring of one page, which the backend must refuse at load.
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

    // int64_bool ----------------------------------------------------------
    val int64Bool = DxirBuilder.function("int64_bool") {
        val ti = DxirType(I64, listOf(4))
        val tb = DxirType(Bool, listOf(4))
        val x = param("x", ti)
        val b = param("b", tb)
        val xx = op(OpKind.MUL, listOf(x, x), ti)
        val y = op(OpKind.ADD, listOf(xx, x), ti)
        val above = op(OpKind.COMPARE, listOf(xx, x), tb, mapOf("direction" to "GT"))
        listOf(y, op(OpKind.LAND, listOf(above, b), tb), op(OpKind.NOT, listOf(b), tb))
    }
    writeModule(repo.resolve("int64_bool/1/model.mlir"), int64Bool.toStablehlo())

    // grad_batched / grad_unbatched ------------------------------------------
    val batches = listOf(1, 2, 4, 8)
    var batchedReference = ""
    for (b in batches) {
        val t = DxirType(F32, listOf(b, 2, 2))
        val sumsq = DxirBuilder.function("batched_sumsq_b$b") {
            val a = param("A", t)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MATMUL, listOf(a, a), t)), scalar))
        }
        val grad = DxirReverseTransform.apply(sumsq)
        for (model in listOf("grad_batched", "grad_unbatched")) {
            writeModule(repo.resolve("$model/1/grad_b$b.mlir"), grad.toStablehlo())
        }
        if (b == batches.last()) {
            // Small integers, so every product and sum is exact in f32.
            val a = FloatArray(b * 4) { ((it * 5) % 17 - 8).toFloat() }
            val g = DxirInterpreter.evalFunction(grad, listOf(a)).single()
            batchedReference = buildString {
                append("{\n")
                append("  \"source\": \"DXIR interpreter, exportTritonExamples\",\n")
                append("  \"inputs\": {\"A\": {\"shape\": [$b, 2, 2], \"data\": ${floats(a)}}},\n")
                append("  \"outputs\": {\"GRAD\": {\"shape\": [$b, 2, 2], \"data\": ${floats(g)}}}\n")
                append("}\n")
            }
        }
    }
    Files.writeString(reference.resolve("grad_batched.json"), batchedReference)

    // ragged_batched / ragged_consecutive ------------------------------------
    // x . x + x over rows of 4 or of 8 f32 values, for batch 1, 2, 4 and 8.
    for (n in listOf(4, 8)) for (b in batches) {
        val t = DxirType(F32, listOf(b, n))
        val fn = DxirBuilder.function("square_plus_b${b}_n$n") {
            val x = param("x", t)
            listOf(op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, x), t), x), t))
        }
        for (model in listOf("ragged_batched", "ragged_consecutive")) {
            writeModule(repo.resolve("$model/1/square_plus_b${b}_n$n.mlir"), fn.toStablehlo())
        }
    }

    // large_io / large_io_host ----------------------------------------------
    val large = DxirBuilder.function("large_square_plus") {
        val t = DxirType(F32, listOf(LARGE_IO_ELEMENTS))
        val x = param("x", t)
        listOf(op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, x), t), x), t))
    }
    for (model in listOf("large_io", "large_io_host")) {
        writeModule(repo.resolve("$model/1/model.mlir"), large.toStablehlo())
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

    // window_sequence / window_sequence_full --------------------------------
    for ((name, windowed) in listOf("window_sequence" to true, "window_sequence_full" to false)) {
        val dir = Files.createTempDirectory("tlaloc-$name")
        try {
            exportWindowModel(dir, windowed)
            val model = TritonModelRepository.write(dir, repo, name, TritonModelRepository.KvMode.SEQUENCE)
            println("  $model")
            if (windowed) {
                // For a load-time refusal: the same manifest with a ring of one
                // page, which holds 4 positions of a window of 8.
                val m = ServingManifest.fromJson(Files.readString(dir.resolve(ServingManifest.FILE_NAME)))
                val short = m.copy(model = m.model.copy(windowedKv = m.model.windowedKv!!.copy(ringPages = 1)))
                Files.writeString(model.resolve("1/$SHORT_RING_MANIFEST"), short.toJson())
            }
        } finally {
            dir.toFile().deleteRecursively()
        }
    }
    println("wrote ${repo.toAbsolutePath()}")
}

/**
 * The decoder of `window_sequence`: layers 0 and 1 attend over a sliding
 * window of 8 positions, layer 2 over the full history.
 */
private val WINDOW_MODEL = HfDecoderConfig(
    architecture = "LlamaForCausalLM", modelType = "llama",
    hiddenSize = 32, intermediateSize = 48, numLayers = 3,
    numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 64,
    rmsNormEps = 1e-6, ropeTheta = 10000.0, maxPositionEmbeddings = 64,
    tieWordEmbeddings = false, attentionBias = false, torchDtype = "float32",
    ropeScalingType = null, family = HfModelFamily.Llama,
    layers = listOf(
        DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 8),
        DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 8),
        DecoderLayerSpec(attention = AttentionKind.FULL),
    ),
)

/**
 * Export [WINDOW_MODEL] into [dir]: decode and prefill entries for batches 1
 * and 2 at contexts 16, 32 and 64, pages of 4, room for four sequences of 64
 * positions. With [windowed] the sliding layers get a windowed KV pool.
 */
private fun exportWindowModel(dir: Path, windowed: Boolean) {
    val config = WINDOW_MODEL
    val policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 64, blockSize = 4, minContext = 16)
    val numBlocks = 1 + 4 * 16
    val window = if (!windowed) null else config.windowedKvPool(4, 64, numBlocks)
    val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = 4, windowedKv = window)
    val specs = HfServingExport.specs(config, model, policy)
    val rng = Random(20260925)
    val weights = HfDecoderGraph.weightSlots(config).associate { slot ->
        val n = slot.type.dims.fold(1) { a, b -> a * b }
        slot.name to if (slot.type.dims.size == 1) {
            FloatArray(n) { 1f + 0.5f * (rng.nextFloat() - 0.5f) }
        } else {
            FloatArray(n) { 0.6f * (rng.nextFloat() - 0.5f) }
        }
    }
    ServingArtifactWriter.export(
        dir = dir,
        modelName = if (windowed) "window-sequence" else "window-sequence-full",
        modelHash = "window-sequence-v1",
        model = model,
        ladder = ServingArtifactWriter.ladderOf(policy),
        specs = specs,
        stageWeight = { slot -> weights.getValue(slot.name) },
        build = { spec -> HfDecoderGraph.build(spec, config, ServingArtifactWriter.ENTRY_POINT) },
    )
}

/** A manifest of `window_sequence` whose ring is too short; the backend refuses it at load. */
private const val SHORT_RING_MANIFEST = "tlaloc-serving-short-ring.json"

/** 4Mi f32 values: 16 MiB each way. */
private const val LARGE_IO_ELEMENTS = 4 * 1024 * 1024

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
