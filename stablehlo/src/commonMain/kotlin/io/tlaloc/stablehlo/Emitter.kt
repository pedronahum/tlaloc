package io.tlaloc.stablehlo

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBlock
import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirCall
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirRegion
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

fun DxirType.toMlir(): String {
    val elem = mlirElementType(dtype)
    return if (dims.isEmpty()) "tensor<$elem>"
    else "tensor<${dims.joinToString("x")}x$elem>"
}

private fun mlirElementType(dtype: DType): String = when (dtype) {
    is F32 -> "f32"
    is F64 -> "f64"
    is I32 -> "i32"
    is I64 -> "i64"
    is Bool -> "i1"
}

fun DxirModule.toStablehlo(): String = buildString {
    appendLine("module {")
    // Aggregate meshes: module-level + any declared per-function. SDY requires them
    // at module scope; our DxirBuilder places them per-function by default.
    val allMeshes = (meshes + functions.flatMap { it.meshes }).distinctBy { it.name }
    allMeshes.forEach { mesh -> appendLine("  " + mesh.toSdyDecl()) }
    if (allMeshes.isNotEmpty() && functions.isNotEmpty()) appendLine()
    functions.forEachIndexed { i, fn ->
        if (i > 0) appendLine()
        append(fn.toStablehlo(indent = "  "))
    }
    append("}")
    appendLine()
}

fun DxirFunction.toStablehlo(indent: String = ""): String =
    StablehloEmitter(this, indent).emit()

internal class StablehloEmitter(private val fn: DxirFunction, private val indent: String) {

    private val out = StringBuilder()
    /**
     * SSA names per dxir id. Single-result ops store `listOf(name)`; multi-result ops
     * store one entry per result (indexed by [DxirOpResult.index]).
     */
    private val ssa = HashMap<Int, List<String>>()
    private var nextSynth: Int = run {
        val maxDxirId = (fn.params + fn.body).maxOfOrNull { it.id } ?: -1
        maxDxirId + 1
    }

    /** Resolve an operand reference to its emitted SSA name. Handles DxirOpResult. */
    private fun ref(node: DxirNode): String = when (node) {
        is DxirOpResult -> ssa[node.source.id]?.getOrNull(node.index)
            ?: error("no SSA for result ${node.source.id}#${node.index}")
        else -> ssa[node.id]?.singleOrNull()
            ?: error("no single-result SSA for dxir id ${node.id}")
    }

    fun emit(): String {
        val paramList = fn.params.joinToString(", ") { p ->
            ssa[p.id] = listOf("%${p.id}")
            "%${p.id}: ${p.type.toMlir()}"
        }
        val retTypeList = fn.returns.joinToString(", ") { it.type.toMlir() }
        // MLIR requires parens around multi-result func return types; single is bare.
        val funcRetTypes = if (fn.returns.size == 1) retTypeList else "($retTypeList)"
        out.appendLine("${indent}func.func @${fn.name}($paramList) -> $funcRetTypes {")
        val step = "$indent  "

        for (node in fn.body) {
            when (node) {
                is DxirParam -> error("DxirParam should not appear in function body")
                is DxirConst -> emitConst(step, node)
                is DxirOp -> emitOp(step, node)
                is DxirCall -> error("DxirCall lowering not yet supported")
                is DxirOpResult -> error("DxirOpResult should not appear in function body")
                is DxirBlockArg -> error("DxirBlockArg should not appear in function body")
            }
        }

        val returnIds = fn.returns.joinToString(", ") { ref(it) }
        // `return` statement uses bare type list regardless of count.
        out.appendLine("${step}return $returnIds : $retTypeList")
        out.appendLine("${indent}}")
        return out.toString()
    }

    private fun emitConst(step: String, node: DxirConst) {
        val name = "%${node.id}"
        ssa[node.id] = listOf(name)
        val literal = when (val v = node.value) {
            is Float -> v.toString()
            is Double -> v.toString()
            is Int -> v.toString()
            is Long -> v.toString()
            // §0.4.73 — rank-N const carries a FloatArray (§0.4.71 Capture fix,
            // §0.4.72 DxirInterpreter fix). Format as a nested dense literal
            // matching the declared type's dims. Rank-0 literal goes through
            // the Float/Double arms above because the scalar capture path uses
            // `e.value[0]` directly; FloatArray here is strictly rank ≥ 1.
            is FloatArray -> denseFromArray(v, node.type.dims)
            else -> error("non-numeric DxirConst value: $v (${v::class.simpleName})")
        }
        out.appendLine("$step$name = stablehlo.constant dense<$literal> : ${node.type.toMlir()}")
    }

    /**
     * §0.4.73 — format a row-major `FloatArray` as an MLIR dense literal matching
     * [dims]. Rank-1 produces `[1.0, 2.0, 3.0]`; rank-2 produces
     * `[[1.0, 2.0], [3.0, 4.0]]`; rank-N is recursive. Matches the syntax
     * `stablehlo-translate` consumes for `stablehlo.constant dense<...> : tensor<RxCxf32>`.
     */
    private fun denseFromArray(values: FloatArray, dims: List<Int>): String {
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

    private fun emitOp(step: String, node: DxirOp) {
        val name = "%${node.id}"
        val outType = node.type.toMlir()
        val ops = node.operands.map { ref(it) }

        when (node.op) {
            // Elementwise unary — output type matches input type
            OpKind.NEG -> unary(step, name, "stablehlo.negate", ops[0], outType)
            OpKind.ABS -> unary(step, name, "stablehlo.abs", ops[0], outType)
            OpKind.EXP -> unary(step, name, "stablehlo.exponential", ops[0], outType)
            OpKind.LOG -> unary(step, name, "stablehlo.log", ops[0], outType)
            OpKind.SQRT -> unary(step, name, "stablehlo.sqrt", ops[0], outType)
            OpKind.RSQRT -> unary(step, name, "stablehlo.rsqrt", ops[0], outType)
            OpKind.TANH -> unary(step, name, "stablehlo.tanh", ops[0], outType)
            OpKind.SIGMOID -> unary(step, name, "stablehlo.logistic", ops[0], outType)

            // Elementwise binary
            OpKind.ADD -> binary(step, name, "stablehlo.add", ops[0], ops[1], outType)
            OpKind.SUB -> binary(step, name, "stablehlo.subtract", ops[0], ops[1], outType)
            OpKind.MUL -> binary(step, name, "stablehlo.multiply", ops[0], ops[1], outType)
            OpKind.DIV -> binary(step, name, "stablehlo.divide", ops[0], ops[1], outType)
            OpKind.POW -> binary(step, name, "stablehlo.power", ops[0], ops[1], outType)

            // Type conversion
            OpKind.CAST -> emitCast(step, name, ops[0], node.operands[0].type, node.type)

            // Shape ops
            OpKind.RESHAPE -> emitReshape(step, name, ops[0], node.operands[0].type, node.type)
            OpKind.TRANSPOSE -> emitTranspose(step, name, ops[0], node, node.operands[0].type)
            OpKind.BROADCAST -> emitBroadcast(step, name, ops[0], node, node.operands[0].type)
            OpKind.CONCAT -> emitConcat(step, name, ops, node)
            OpKind.SLICE -> emitSlice(step, name, ops[0], node, node.operands[0].type)

            // Composite lowerings
            OpKind.RELU -> emitRelu(step, name, ops[0], node.type)
            OpKind.STEP -> emitStep(step, name, ops[0], node.type)
            OpKind.SILU -> emitSilu(step, name, ops[0], node.type)
            OpKind.GELU -> emitGelu(step, name, ops[0], node.type)
            OpKind.SOFTMAX -> emitSoftmax(step, name, ops[0], node, node.operands[0].type)
            OpKind.LOGSUMEXP -> emitLogsumexp(step, name, ops[0], node, node.operands[0].type, node.type)
            OpKind.LAYERNORM -> emitLayerNorm(step, name, ops[0], node, node.operands[0].type)
            OpKind.RMSNORM -> emitRmsNorm(step, name, ops[0], node, node.operands[0].type)
            OpKind.SUM -> emitReduce(
                step, name, ops[0], node.operands[0].type, node.type,
                reducer = "stablehlo.add", initLiteral = "0.0",
                dims = readReductionDims(node, node.operands[0].type),
            )
            OpKind.MEAN -> emitReduceMean(
                step, name, ops[0], node.operands[0].type, node.type,
                dims = readReductionDims(node, node.operands[0].type),
            )
            OpKind.MAX -> emitReduce(
                step, name, ops[0], node.operands[0].type, node.type,
                reducer = "stablehlo.maximum", initLiteral = negInfLiteral(node.operands[0].type.dtype),
                dims = readReductionDims(node, node.operands[0].type),
            )
            OpKind.MIN -> emitReduce(
                step, name, ops[0], node.operands[0].type, node.type,
                reducer = "stablehlo.minimum", initLiteral = posInfLiteral(node.operands[0].type.dtype),
                dims = readReductionDims(node, node.operands[0].type),
            )

            OpKind.MATMUL -> emitMatmul(
                step,
                name,
                a = ops[0],
                b = ops[1],
                aType = node.operands[0].type,
                bType = node.operands[1].type,
                outType = node.type,
                node = node,
            )
            OpKind.CONV2D -> emitConv2d(
                step, name,
                lhs = ops[0], rhs = ops[1],
                node = node,
                lhsType = node.operands[0].type,
                rhsType = node.operands[1].type,
                kernelLayout = "[o, i, 0, 1]",
            )
            OpKind.CONV_TRANSPOSE2D -> emitConv2d(
                step, name,
                lhs = ops[0], rhs = ops[1],
                node = node,
                lhsType = node.operands[0].type,
                rhsType = node.operands[1].type,
                kernelLayout = "[i, o, 0, 1]",
            )
            OpKind.ARGMAX -> {
                val axis = readAxis(node, node.operands[0].type.rank)
                val refStr = emitArgmax(
                    step,
                    x = ops[0],
                    axis = axis,
                    inputType = node.operands[0].type,
                    outputType = node.type,
                )
                ssa[node.id] = listOf(refStr)
            }
            OpKind.CROSS_ENTROPY -> emitCrossEntropy(
                step, name,
                logits = ops[0], labels = ops[1],
                logitsType = node.operands[0].type,
                labelsType = node.operands[1].type,
                outType = node.type,
            )
            OpKind.SHARD_CONSTRAINT -> emitShardConstraint(
                step, name, ops[0], node, node.operands[0].type,
            )
            OpKind.SCATTER -> emitScatter(
                step, name,
                operand = ops[0], scatterIndices = ops[1], updates = ops[2],
                node = node,
                operandType = node.operands[0].type,
                indicesType = node.operands[1].type,
                updatesType = node.operands[2].type,
            )
            OpKind.SCATTER_ADD -> emitScatterAdd(
                step, name,
                base = ops[0], idx = ops[1], value = ops[2],
                node = node,
                baseType = node.operands[0].type,
                idxType = node.operands[1].type,
                valueType = node.operands[2].type,
            )
            OpKind.BATCHNORM -> emitBatchNorm(step, name, ops, node)
            OpKind.SPLIT -> emitSplit(step, node, ops[0], node.operands[0].type)
            OpKind.MANUAL_COMPUTATION -> emitManualComputation(step, name, ops, node)
            // §0.4.60 — `stablehlo.not` / `stablehlo.and` on Bool (i1) inputs. Added
            // alongside the existing `stablehlo.power` arm at line 140; these three
            // were listed as "out of scope" in §0.4.53 but the underlying MLIR ops
            // are a one-liner each. Keeping them as errors forced Stage B's
            // coarsening output to avoid NOT/LAND even where they were the
            // structurally correct form (§0.4.55's break-hoist uses
            // `LAND(cond, NOT(break_cond))` and can now flow through the emitter
            // once D.3i PhiCalculus closure lands — no more "add StableHLO arm"
            // prerequisite.
            OpKind.NOT -> unary(step, name, "stablehlo.not", ops[0], outType)
            OpKind.LAND -> binary(step, name, "stablehlo.and", ops[0], ops[1], outType)
            // Structured-control-flow ops: Stage B's coarsening pass closes IF / WHILE
            // regions into straight-line dxir before this emitter sees them. Lowering
            // either op directly to `stablehlo.if` / `stablehlo.while` is deferred
            // post-Stage-B; until then, an unclosed IF/WHILE reaching the emitter is a
            // compiler bug (the coarsening pass should have closed it or rejected the
            // primal). Loud failure beats silent miscompile.
            OpKind.IF -> error(
                "StableHLO lowering for OpKind.IF deferred post-Stage-B; the φ-calculus " +
                    "coarsening pass should close this IF before emission (op id=${node.id})",
            )
            OpKind.WHILE -> error(
                "StableHLO lowering for OpKind.WHILE deferred post-Stage-B; the φ-calculus " +
                    "coarsening pass should close this WHILE via C5/C6/C7/C8/C9 before " +
                    "emission (op id=${node.id})",
            )
            OpKind.COARSENED -> error(
                "StableHLO lowering for OpKind.COARSENED deferred — Stage C.3b.1's splice op " +
                    "is a compile-time artefact consumed by the grad + synthesis passes before " +
                    "emission. Reaching here means the coarsen → synthesis chain didn't inline " +
                    "the op back to straight-line dxir (op id=${node.id})",
            )
            OpKind.GATHER -> emitGather(
                step, name,
                operand = ops[0], startIndices = ops[1],
                node = node,
                operandType = node.operands[0].type,
                indicesType = node.operands[1].type,
            )
            OpKind.EMBEDDING -> emitEmbedding(
                step, name,
                table = ops[0], indices = ops[1],
                tableType = node.operands[0].type,
                indicesType = node.operands[1].type,
                outType = node.type,
            )
            OpKind.SCALED_DOT_PRODUCT_ATTENTION -> emitSdpa(
                step, name,
                q = ops[0], k = ops[1], v = ops[2],
                qType = node.operands[0].type,
                kType = node.operands[1].type,
                vType = node.operands[2].type,
                outType = node.type,
            )
            OpKind.DOT -> emitDot(
                step,
                name,
                a = ops[0],
                b = ops[1],
                aType = node.operands[0].type,
                bType = node.operands[1].type,
                outType = node.type,
            )

            else -> error("StableHLO lowering not yet implemented for ${node.op}")
        }
        // Most ops emit a single line whose result is the literal `%N` we named above.
        // Multi-output ops (e.g. ARGMAX, which uses `%pair:2` and takes result #1; SPLIT,
        // which emits N slices) set ssa[node.id] themselves earlier; we preserve those.
        if (node.id !in ssa) ssa[node.id] = listOf(name)
    }

    @Suppress("UNCHECKED_CAST")
    private fun intListAttr(node: DxirOp, key: String): List<Int> =
        (node.attrs[key] as? List<Int>)
            ?: error("op ${node.op} missing int-list attr '$key'; got attrs=${node.attrs}")

    private fun intAttr(node: DxirOp, key: String): Int =
        (node.attrs[key] as? Int)
            ?: error("op ${node.op} missing int attr '$key'; got attrs=${node.attrs}")

    private fun unary(step: String, name: String, op: String, x: String, type: String) {
        out.appendLine("$step$name = $op $x : $type")
    }

    private fun binary(step: String, name: String, op: String, a: String, b: String, type: String) {
        out.appendLine("$step$name = $op $a, $b : $type")
    }

    private fun emitRelu(step: String, name: String, x: String, type: DxirType) {
        val zero = synth()
        out.appendLine("$step$zero = stablehlo.constant dense<0.0> : ${type.toMlir()}")
        out.appendLine("$step$name = stablehlo.maximum $x, $zero : ${type.toMlir()}")
    }

    private fun emitStep(step: String, name: String, x: String, type: DxirType) {
        // step(x) = (x > 0) ? 1 : 0.  At x == 0 the output is 0, matching XLA's
        // `compare GT` + `select` semantics (NOT the mathematical H(0) = 1/2).
        val tMlir = type.toMlir()
        val boolType = DxirType(Bool, type.dims)
        val boolMlir = boolType.toMlir()
        val (zeroLit, oneLit, cmpSuffix) = when (type.dtype) {
            is F32, is F64 -> Triple("0.0", "1.0", "FLOAT")
            is I32, is I64 -> Triple("0", "1", "SIGNED")
            is Bool -> error("STEP on bool input is not meaningful")
        }
        val zero = synth(); val one = synth(); val gt = synth()
        out.appendLine("$step$zero = stablehlo.constant dense<$zeroLit> : $tMlir")
        out.appendLine("$step$one = stablehlo.constant dense<$oneLit> : $tMlir")
        out.appendLine(
            "$step$gt = stablehlo.compare  GT, $x, $zero,  $cmpSuffix : ($tMlir, $tMlir) -> $boolMlir",
        )
        out.appendLine("$step$name = stablehlo.select $gt, $one, $zero : $boolMlir, $tMlir")
    }

    private fun emitSilu(step: String, name: String, x: String, type: DxirType) {
        // silu(x) = x * sigmoid(x)
        val tMlir = type.toMlir()
        val sig = synth()
        out.appendLine("$step$sig = stablehlo.logistic $x : $tMlir")
        out.appendLine("$step$name = stablehlo.multiply $x, $sig : $tMlir")
    }

    private fun emitGelu(step: String, name: String, x: String, type: DxirType) {
        // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        val tMlir = type.toMlir()
        val half = synth(); val one = synth(); val sqrtTwoOverPi = synth(); val coeff = synth()
        out.appendLine("$step$half = stablehlo.constant dense<0.5> : $tMlir")
        out.appendLine("$step$one = stablehlo.constant dense<1.0> : $tMlir")
        out.appendLine("$step$sqrtTwoOverPi = stablehlo.constant dense<0.7978845608> : $tMlir")
        out.appendLine("$step$coeff = stablehlo.constant dense<0.044715> : $tMlir")

        val xSq = synth(); val xCubed = synth(); val scaled = synth()
        out.appendLine("$step$xSq = stablehlo.multiply $x, $x : $tMlir")
        out.appendLine("$step$xCubed = stablehlo.multiply $x, $xSq : $tMlir")
        out.appendLine("$step$scaled = stablehlo.multiply $coeff, $xCubed : $tMlir")

        val inner = synth(); val t = synth(); val tanh = synth()
        out.appendLine("$step$inner = stablehlo.add $x, $scaled : $tMlir")
        out.appendLine("$step$t = stablehlo.multiply $sqrtTwoOverPi, $inner : $tMlir")
        out.appendLine("$step$tanh = stablehlo.tanh $t : $tMlir")

        val onePlus = synth(); val halfX = synth()
        out.appendLine("$step$onePlus = stablehlo.add $one, $tanh : $tMlir")
        out.appendLine("$step$halfX = stablehlo.multiply $half, $x : $tMlir")
        out.appendLine("$step$name = stablehlo.multiply $halfX, $onePlus : $tMlir")
    }

    /**
     * Reads `axis` attr with default = last dim (rank - 1). Normalizes negatives.
     */
    private fun readAxis(node: DxirOp, inputRank: Int, default: Int = inputRank - 1): Int {
        val raw = node.attrs["axis"] ?: return normalizeAxis(default, inputRank)
        val asInt = (raw as? Int) ?: error("op ${node.op} 'axis' must be Int; got ${raw::class.simpleName}")
        return normalizeAxis(asInt, inputRank)
    }

    private fun emitSoftmax(step: String, name: String, x: String, node: DxirOp, inputType: DxirType) {
        val axis = readAxis(node, inputType.rank)
        val inT = inputType.toMlir()
        val reducedT = reducedType(inputType, listOf(axis))
        val reducedMlir = reducedT.toMlir()
        val scalarT = "tensor<${mlirElementType(inputType.dtype)}>"
        val bcDims = broadcastDimsAfterReducing(inputType.rank, listOf(axis))
        val bcDimsStr = bcDims.joinToString(", ")

        // max = reduce_max(x, dims=[axis])
        val maxInit = synth(); val max = synth()
        out.appendLine("$step$maxInit = stablehlo.constant dense<${negInfLiteral(inputType.dtype)}> : $scalarT")
        out.appendLine(
            "$step$max = stablehlo.reduce($x init: $maxInit) applies stablehlo.maximum across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        val maxBc = synth()
        out.appendLine("$step$maxBc = stablehlo.broadcast_in_dim $max, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        val shifted = synth(); val e = synth()
        out.appendLine("$step$shifted = stablehlo.subtract $x, $maxBc : $inT")
        out.appendLine("$step$e = stablehlo.exponential $shifted : $inT")

        // s = reduce_sum(e, dims=[axis])
        val sumInit = synth(); val s = synth()
        out.appendLine("$step$sumInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$s = stablehlo.reduce($e init: $sumInit) applies stablehlo.add across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        val sBc = synth()
        out.appendLine("$step$sBc = stablehlo.broadcast_in_dim $s, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        out.appendLine("$step$name = stablehlo.divide $e, $sBc : $inT")
    }

    private fun emitLogsumexp(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
        outputType: DxirType,
    ) {
        val axis = readAxis(node, inputType.rank)
        val expected = reducedType(inputType, listOf(axis))
        require(expected.dims == outputType.dims) {
            "LOGSUMEXP output shape ${outputType.dims} does not match expected ${expected.dims} for axis=$axis"
        }
        val inT = inputType.toMlir()
        val reducedMlir = outputType.toMlir()
        val scalarT = "tensor<${mlirElementType(inputType.dtype)}>"
        val bcDims = broadcastDimsAfterReducing(inputType.rank, listOf(axis))
        val bcDimsStr = bcDims.joinToString(", ")

        val maxInit = synth(); val max = synth()
        out.appendLine("$step$maxInit = stablehlo.constant dense<${negInfLiteral(inputType.dtype)}> : $scalarT")
        out.appendLine(
            "$step$max = stablehlo.reduce($x init: $maxInit) applies stablehlo.maximum across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        val maxBc = synth(); val shifted = synth(); val e = synth()
        out.appendLine("$step$maxBc = stablehlo.broadcast_in_dim $max, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        out.appendLine("$step$shifted = stablehlo.subtract $x, $maxBc : $inT")
        out.appendLine("$step$e = stablehlo.exponential $shifted : $inT")

        val sumInit = synth(); val s = synth(); val logS = synth()
        out.appendLine("$step$sumInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$s = stablehlo.reduce($e init: $sumInit) applies stablehlo.add across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        out.appendLine("$step$logS = stablehlo.log $s : $reducedMlir")
        // Final: logS + max, both reduced shape
        out.appendLine("$step$name = stablehlo.add $logS, $max : $reducedMlir")
    }

    private fun readEps(node: DxirOp, default: Float = 1e-5f): Float {
        val v = node.attrs["epsilon"]
        return when (v) {
            null -> default
            is Float -> v
            is Double -> v.toFloat()
            is Number -> v.toFloat()
            else -> error("op ${node.op} 'epsilon' attr must be a Number; got ${v::class.simpleName}")
        }
    }

    private fun emitLayerNorm(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val axis = readAxis(node, inputType.rank)
        val eps = readEps(node)
        val inT = inputType.toMlir()
        val reducedT = reducedType(inputType, listOf(axis))
        val reducedMlir = reducedT.toMlir()
        val scalarT = "tensor<${mlirElementType(inputType.dtype)}>"
        val bcDims = broadcastDimsAfterReducing(inputType.rank, listOf(axis))
        val bcDimsStr = bcDims.joinToString(", ")
        val n = inputType.dims[axis].toLong().coerceAtLeast(1L)

        // mean = sum(x, axis) / N
        val sumInit = synth(); val sum = synth(); val nConst = synth(); val mean = synth()
        out.appendLine("$step$sumInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$sum = stablehlo.reduce($x init: $sumInit) applies stablehlo.add across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        out.appendLine("$step$nConst = stablehlo.constant dense<$n.0> : $reducedMlir")
        out.appendLine("$step$mean = stablehlo.divide $sum, $nConst : $reducedMlir")

        // centered = x - broadcast(mean)
        val meanBc = synth(); val centered = synth()
        out.appendLine("$step$meanBc = stablehlo.broadcast_in_dim $mean, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        out.appendLine("$step$centered = stablehlo.subtract $x, $meanBc : $inT")

        // var = sum(centered^2, axis) / N
        val sq = synth(); val sqInit = synth(); val sqSum = synth(); val variance = synth()
        out.appendLine("$step$sq = stablehlo.multiply $centered, $centered : $inT")
        out.appendLine("$step$sqInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$sqSum = stablehlo.reduce($sq init: $sqInit) applies stablehlo.add across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        out.appendLine("$step$variance = stablehlo.divide $sqSum, $nConst : $reducedMlir")

        // std = sqrt(var + eps)
        val epsC = synth(); val varPlus = synth(); val std = synth(); val stdBc = synth()
        out.appendLine("$step$epsC = stablehlo.constant dense<$eps> : $reducedMlir")
        out.appendLine("$step$varPlus = stablehlo.add $variance, $epsC : $reducedMlir")
        out.appendLine("$step$std = stablehlo.sqrt $varPlus : $reducedMlir")
        out.appendLine("$step$stdBc = stablehlo.broadcast_in_dim $std, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        out.appendLine("$step$name = stablehlo.divide $centered, $stdBc : $inT")
    }

    private fun emitRmsNorm(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val axis = readAxis(node, inputType.rank)
        val eps = readEps(node)
        val inT = inputType.toMlir()
        val reducedT = reducedType(inputType, listOf(axis))
        val reducedMlir = reducedT.toMlir()
        val scalarT = "tensor<${mlirElementType(inputType.dtype)}>"
        val bcDims = broadcastDimsAfterReducing(inputType.rank, listOf(axis))
        val bcDimsStr = bcDims.joinToString(", ")
        val n = inputType.dims[axis].toLong().coerceAtLeast(1L)

        // ms = sum(x*x, axis) / N
        val sq = synth(); val sqInit = synth(); val sqSum = synth(); val nConst = synth(); val ms = synth()
        out.appendLine("$step$sq = stablehlo.multiply $x, $x : $inT")
        out.appendLine("$step$sqInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$sqSum = stablehlo.reduce($sq init: $sqInit) applies stablehlo.add across dimensions = [$axis] " +
                ": ($inT, $scalarT) -> $reducedMlir",
        )
        out.appendLine("$step$nConst = stablehlo.constant dense<$n.0> : $reducedMlir")
        out.appendLine("$step$ms = stablehlo.divide $sqSum, $nConst : $reducedMlir")

        // rms = sqrt(ms + eps)
        val epsC = synth(); val msPlus = synth(); val rms = synth(); val rmsBc = synth()
        out.appendLine("$step$epsC = stablehlo.constant dense<$eps> : $reducedMlir")
        out.appendLine("$step$msPlus = stablehlo.add $ms, $epsC : $reducedMlir")
        out.appendLine("$step$rms = stablehlo.sqrt $msPlus : $reducedMlir")
        out.appendLine("$step$rmsBc = stablehlo.broadcast_in_dim $rms, dims = [$bcDimsStr] : ($reducedMlir) -> $inT")
        out.appendLine("$step$name = stablehlo.divide $x, $rmsBc : $inT")
    }

    /**
     * Shape after reducing the given dims from [inputType]. Dims are removed (not kept-as-1).
     * Element dtype is preserved. Empty dims list means no reduction (identity shape).
     */
    private fun reducedType(inputType: DxirType, dims: List<Int>): DxirType {
        val kept = (0 until inputType.rank).filter { it !in dims }
        return DxirType(inputType.dtype, kept.map { inputType.dims[it] })
    }

    private fun emitReduce(
        step: String,
        name: String,
        x: String,
        inputType: DxirType,
        outputType: DxirType,
        reducer: String,
        initLiteral: String,
        dims: List<Int>,
    ) {
        val expected = reducedType(inputType, dims)
        require(expected.dims == outputType.dims) {
            "reduce output shape ${outputType.dims} does not match expected ${expected.dims} for reducing dims=$dims from ${inputType.dims}"
        }
        val elem = mlirElementType(inputType.dtype)
        val scalarT = "tensor<$elem>"
        val init = synth()
        out.appendLine("$step$init = stablehlo.constant dense<$initLiteral> : $scalarT")
        out.appendLine(
            "$step$name = stablehlo.reduce($x init: $init) applies $reducer across dimensions = [${dims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}, $scalarT) -> ${outputType.toMlir()}",
        )
    }

    private fun emitReduceMean(
        step: String,
        name: String,
        x: String,
        inputType: DxirType,
        outputType: DxirType,
        dims: List<Int>,
    ) {
        val expected = reducedType(inputType, dims)
        require(expected.dims == outputType.dims) {
            "MEAN output shape ${outputType.dims} does not match expected ${expected.dims} for reducing dims=$dims from ${inputType.dims}"
        }
        val elem = mlirElementType(inputType.dtype)
        val scalarT = "tensor<$elem>"
        val init = synth()
        val sumName = synth()

        out.appendLine("$step$init = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$sumName = stablehlo.reduce($x init: $init) applies stablehlo.add across dimensions = [${dims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}, $scalarT) -> ${outputType.toMlir()}",
        )
        // divisor = product of reduced-dim sizes, emitted at output type (tensor or scalar)
        val n = dims.map { inputType.dims[it].toLong() }.fold(1L) { acc, d -> acc * d }.coerceAtLeast(1L)
        val divisor = synth()
        out.appendLine("$step$divisor = stablehlo.constant dense<$n.0> : ${outputType.toMlir()}")
        out.appendLine("$step$name = stablehlo.divide $sumName, $divisor : ${outputType.toMlir()}")
    }

    /**
     * Read `reduction_dims` attr as List<Int>. Defaults to "all dims" (full reduction, scalar
     * output) which preserves the pre-axis-aware emitter behavior.
     */
    private fun readReductionDims(node: DxirOp, inputType: DxirType): List<Int> {
        val raw = node.attrs["reduction_dims"] ?: return (0 until inputType.rank).toList()
        @Suppress("UNCHECKED_CAST")
        val list = (raw as? List<Int>)
            ?: error("op ${node.op} 'reduction_dims' must be List<Int>; got ${raw::class.simpleName}")
        val normalized = list.map { normalizeAxis(it, inputType.rank) }.sorted().distinct()
        require(normalized.size == list.size) {
            "op ${node.op} has duplicate entries in reduction_dims=$list (after normalization: $normalized)"
        }
        return normalized
    }

    private fun normalizeAxis(axis: Int, rank: Int): Int {
        val n = if (axis < 0) axis + rank else axis
        require(n in 0 until rank) { "axis $axis out of bounds for rank $rank" }
        return n
    }

    /** Broadcast-dimensions mapping for "re-inflating" a reduced tensor back to the original shape. */
    private fun broadcastDimsAfterReducing(originalRank: Int, reducedDims: List<Int>): List<Int> =
        (0 until originalRank).filter { it !in reducedDims }

    private fun emitMatmul(
        step: String,
        name: String,
        a: String,
        b: String,
        aType: DxirType,
        bType: DxirType,
        outType: DxirType,
        node: DxirOp,
    ) {
        val explicit = listOf(
            "lhs_contracting_dims", "rhs_contracting_dims",
            "lhs_batching_dims", "rhs_batching_dims",
        ).any { it in node.attrs }

        if (!explicit) {
            // Default rank-2 path (pre-axis-aware behavior).
            require(aType.rank == 2 && bType.rank == 2) {
                "MATMUL without batching/contracting attrs requires rank-2 inputs; got ${aType.dims} x ${bType.dims}. " +
                    "For higher ranks, supply lhs_contracting_dims / rhs_contracting_dims (and optional *_batching_dims)."
            }
            out.appendLine(
                "$step$name = stablehlo.dot_general $a, $b, contracting_dims = [1] x [0] " +
                    ": (${aType.toMlir()}, ${bType.toMlir()}) -> ${outType.toMlir()}",
            )
            return
        }

        val lhsContract = intListAttr(node, "lhs_contracting_dims")
        val rhsContract = intListAttr(node, "rhs_contracting_dims")
        @Suppress("UNCHECKED_CAST")
        val lhsBatch = (node.attrs["lhs_batching_dims"] as? List<Int>) ?: emptyList()
        @Suppress("UNCHECKED_CAST")
        val rhsBatch = (node.attrs["rhs_batching_dims"] as? List<Int>) ?: emptyList()

        require(lhsContract.size == rhsContract.size) {
            "MATMUL contracting_dims length mismatch: lhs=$lhsContract, rhs=$rhsContract"
        }
        require(lhsBatch.size == rhsBatch.size) {
            "MATMUL batching_dims length mismatch: lhs=$lhsBatch, rhs=$rhsBatch"
        }
        lhsContract.forEach { require(it in 0 until aType.rank) { "lhs_contracting_dim $it out of bounds for rank ${aType.rank}" } }
        rhsContract.forEach { require(it in 0 until bType.rank) { "rhs_contracting_dim $it out of bounds for rank ${bType.rank}" } }
        lhsBatch.forEach { require(it in 0 until aType.rank) { "lhs_batching_dim $it out of bounds for rank ${aType.rank}" } }
        rhsBatch.forEach { require(it in 0 until bType.rank) { "rhs_batching_dim $it out of bounds for rank ${bType.rank}" } }

        val batchingPart = if (lhsBatch.isNotEmpty() || rhsBatch.isNotEmpty()) {
            "batching_dims = [${lhsBatch.joinToString(", ")}] x [${rhsBatch.joinToString(", ")}], "
        } else {
            ""
        }
        out.appendLine(
            "$step$name = stablehlo.dot_general $a, $b, " +
                batchingPart +
                "contracting_dims = [${lhsContract.joinToString(", ")}] x [${rhsContract.joinToString(", ")}] " +
                ": (${aType.toMlir()}, ${bType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    private fun emitConv2d(
        step: String,
        name: String,
        lhs: String,
        rhs: String,
        node: DxirOp,
        lhsType: DxirType,
        rhsType: DxirType,
        kernelLayout: String,   // "[o, i, 0, 1]" for CONV2D, "[i, o, 0, 1]" for CONV_TRANSPOSE2D
    ) {
        require(lhsType.rank == 4) { "conv input must be rank-4 NCHW; got dims ${lhsType.dims}" }
        require(rhsType.rank == 4) { "conv kernel must be rank-4; got dims ${rhsType.dims}" }
        require(node.type.rank == 4) { "conv output must be rank-4; got dims ${node.type.dims}" }

        val strides = intListAttr(node, "window_strides")
        require(strides.size == 2) { "conv window_strides must be length 2; got $strides" }

        val padding = conv2dPadding(node)
        val lhsDilation = conv2dDilation(node, "lhs_dilation")
        val rhsDilation = conv2dDilation(node, "rhs_dilation")
        val featureGroupCount = (node.attrs["feature_group_count"] as? Int) ?: 1
        val batchGroupCount = (node.attrs["batch_group_count"] as? Int) ?: 1

        val strideStr = strides.joinToString(", ")
        val padStr = padding.joinToString(", ") { "[${it.joinToString(", ")}]" }
        val lhsDilStr = lhsDilation.joinToString(", ")
        val rhsDilStr = rhsDilation.joinToString(", ")

        out.appendLine(
            "$step$name = stablehlo.convolution($lhs, $rhs) " +
                "dim_numbers = [b, f, 0, 1]x${kernelLayout}->[b, f, 0, 1], " +
                "window = {stride = [$strideStr], pad = [$padStr], " +
                "lhs_dilate = [$lhsDilStr], rhs_dilate = [$rhsDilStr]} " +
                "{batch_group_count = $batchGroupCount : i64, " +
                "feature_group_count = $featureGroupCount : i64} " +
                ": (${lhsType.toMlir()}, ${rhsType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun emitScatter(
        step: String,
        name: String,
        operand: String,
        scatterIndices: String,
        updates: String,
        node: DxirOp,
        operandType: DxirType,
        indicesType: DxirType,
        updatesType: DxirType,
    ) {
        @Suppress("UNCHECKED_CAST")
        val updateWindowDims = (node.attrs["update_window_dims"] as? List<Int>) ?: emptyList()
        @Suppress("UNCHECKED_CAST")
        val insertedWindowDims = (node.attrs["inserted_window_dims"] as? List<Int>) ?: emptyList()
        val scatterDimsToOperandDims = intListAttr(node, "scatter_dims_to_operand_dims")
        val indexVectorDim = intAttr(node, "index_vector_dim")
        val reduction = (node.attrs["reduction"] as? String) ?: "replace"
        val indicesAreSorted = (node.attrs["indices_are_sorted"] as? Boolean) ?: false
        val uniqueIndices = (node.attrs["unique_indices"] as? Boolean) ?: false

        require(node.type.dims == operandType.dims) {
            "SCATTER output shape ${node.type.dims} must match operand shape ${operandType.dims}"
        }
        require(indexVectorDim in 0..indicesType.rank) {
            "SCATTER index_vector_dim $indexVectorDim out of bounds for indices rank ${indicesType.rank}"
        }

        val scalarT = "tensor<${mlirElementType(operandType.dtype)}>"

        val dimNumbers = buildString {
            append("#stablehlo.scatter<")
            val parts = mutableListOf<String>()
            if (updateWindowDims.isNotEmpty()) {
                parts += "update_window_dims = [${updateWindowDims.joinToString(", ")}]"
            }
            if (insertedWindowDims.isNotEmpty()) {
                parts += "inserted_window_dims = [${insertedWindowDims.joinToString(", ")}]"
            }
            parts += "scatter_dims_to_operand_dims = [${scatterDimsToOperandDims.joinToString(", ")}]"
            parts += "index_vector_dim = $indexVectorDim"
            append(parts.joinToString(", "))
            append(">")
        }

        val flagParts = mutableListOf("scatter_dimension_numbers = $dimNumbers")
        if (indicesAreSorted) flagParts += "indices_are_sorted = true"
        if (uniqueIndices) flagParts += "unique_indices = true"

        val cur = synth(); val upd = synth(); val body = synth()

        out.appendLine(
            """$step$name = "stablehlo.scatter"($operand, $scatterIndices, $updates) <{${flagParts.joinToString(", ")}}> ({""",
        )
        out.appendLine("$step ^bb0($cur: $scalarT, $upd: $scalarT):")
        val returnVal = when (reduction) {
            "replace" -> upd
            "add" -> {
                out.appendLine("$step   $body = stablehlo.add $cur, $upd : $scalarT")
                body
            }
            "multiply", "mul" -> {
                out.appendLine("$step   $body = stablehlo.multiply $cur, $upd : $scalarT")
                body
            }
            "max", "maximum" -> {
                out.appendLine("$step   $body = stablehlo.maximum $cur, $upd : $scalarT")
                body
            }
            "min", "minimum" -> {
                out.appendLine("$step   $body = stablehlo.minimum $cur, $upd : $scalarT")
                body
            }
            else -> error(
                "SCATTER unknown 'reduction'='$reduction'; expected one of: replace, add, mul, max, min",
            )
        }
        out.appendLine("$step   stablehlo.return $returnVal : $scalarT")
        out.appendLine(
            "$step }) : (${operandType.toMlir()}, ${indicesType.toMlir()}, ${updatesType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /**
     * §0.4.112 — lowering for the `:autograd`-emitted [OpKind.SCATTER_ADD] substrate
     * shape (`base[idx] += value`, no attrs). Two operand-rank slices are supported,
     * matching the [DxirInterpreter] arms shipped in §0.4.45 + §0.4.111:
     *
     *  - rank-1: `base: tensor<NxF>, idx: scalar I32, value: scalar F`.
     *  - rank-2: `base: tensor<MxNxF>, idx: scalar I32, value: tensor<NxF>`.
     *
     * Both operands flow into `stablehlo.scatter` directly with no reshape: with
     * `index_vector_dim = 0` and a rank-0 `scatter_indices`, the StableHLO spec
     * implicitly expands by a trailing 1-dim, and the rank arithmetic
     * `rank(updates) == rank(scatter_indices_expanded) - 1 + size(update_window_dims)`
     * collapses cleanly to `rank(value)`. The body computation is `stablehlo.add`
     * (the "_ADD" in `SCATTER_ADD`). The general [emitScatter] path handles arbitrary
     * stablehlo-style scatter; this path is dedicated to the substrate shape that
     * `GatherRule` emits.
     *
     * The `in_place` attr (§0.4.46's destructive-mutation marker) is intentionally
     * IGNORED here: stablehlo.scatter is functional, not in-place. The marker is a
     * synthesis-side hint for the host-runtime path; emitting through StableHLO
     * always produces a fresh tensor regardless.
     */
    private fun emitScatterAdd(
        step: String,
        name: String,
        base: String,
        idx: String,
        value: String,
        node: DxirOp,
        baseType: DxirType,
        idxType: DxirType,
        valueType: DxirType,
    ) {
        require(baseType.rank == 1 || baseType.rank == 2) {
            "SCATTER_ADD base must be rank-1 or rank-2 (substrate shape); got rank=${baseType.rank}"
        }
        require(idxType.isScalar && idxType.dtype == I32) {
            "SCATTER_ADD idx must be scalar I32 (substrate shape); got $idxType"
        }
        val expectedValueRank = baseType.rank - 1
        require(valueType.rank == expectedValueRank) {
            "SCATTER_ADD value must be rank-$expectedValueRank for rank-${baseType.rank} base; got rank=${valueType.rank}"
        }
        require(node.type.dims == baseType.dims) {
            "SCATTER_ADD result shape ${node.type.dims} must match base shape ${baseType.dims}"
        }

        val scalarT = "tensor<${mlirElementType(baseType.dtype)}>"

        // Dimension numbers: scalar idx (rank-0) with index_vector_dim=0 triggers
        // implicit trailing-1 expansion, so the effective scatter_indices rank is 1.
        // `update_window_dims` indexes into UPDATES' axes (not operand's). For rank-2
        // base, updates is rank-1 (the row), so the window dim is axis 0 of updates,
        // which maps to operand axis 1 (since operand axis 0 is inserted via the
        // index). For rank-1 base, updates is rank-0 (scalar) and there are no
        // window dims at all.
        val updateWindowDims = when (baseType.rank) {
            1 -> emptyList()
            else -> listOf(0)
        }
        val dimNumbers = buildString {
            append("#stablehlo.scatter<")
            val parts = mutableListOf<String>()
            if (updateWindowDims.isNotEmpty()) {
                parts += "update_window_dims = [${updateWindowDims.joinToString(", ")}]"
            }
            parts += "inserted_window_dims = [0]"
            parts += "scatter_dims_to_operand_dims = [0]"
            parts += "index_vector_dim = 0"
            append(parts.joinToString(", "))
            append(">")
        }

        val cur = synth(); val upd = synth(); val sum = synth()
        out.appendLine(
            """$step$name = "stablehlo.scatter"($base, $idx, $value) <{scatter_dimension_numbers = $dimNumbers, unique_indices = true}> ({""",
        )
        out.appendLine("$step ^bb0($cur: $scalarT, $upd: $scalarT):")
        out.appendLine("$step   $sum = stablehlo.add $cur, $upd : $scalarT")
        out.appendLine("$step   stablehlo.return $sum : $scalarT")
        out.appendLine(
            "$step }) : (${baseType.toMlir()}, ${idxType.toMlir()}, ${valueType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun emitManualComputation(
        step: String,
        name: String,
        operandNames: List<String>,
        node: DxirOp,
    ) {
        @Suppress("UNCHECKED_CAST")
        val inShardings = (node.attrs["in_shardings"] as? List<DxirSharding>)
            ?: error("MANUAL_COMPUTATION requires 'in_shardings' attr: List<DxirSharding>")
        @Suppress("UNCHECKED_CAST")
        val outShardings = (node.attrs["out_shardings"] as? List<DxirSharding>)
            ?: error("MANUAL_COMPUTATION requires 'out_shardings' attr: List<DxirSharding>")
        @Suppress("UNCHECKED_CAST")
        val manualAxes = (node.attrs["manual_axes"] as? List<String>)
            ?: emptyList()

        require(inShardings.size == operandNames.size) {
            "MANUAL_COMPUTATION in_shardings length ${inShardings.size} ≠ operand count ${operandNames.size}"
        }
        require(outShardings.size == node.numResults) {
            "MANUAL_COMPUTATION out_shardings length ${outShardings.size} ≠ result count ${node.numResults}"
        }
        require(node.regions.size == 1) {
            "MANUAL_COMPUTATION expects exactly one region (the SPMD body); got ${node.regions.size}"
        }
        val region = node.regions.single()
        require(region.blocks.size == 1) {
            "MANUAL_COMPUTATION v0 supports single-block regions; got ${region.blocks.size}"
        }
        val block = region.entryBlock
        require(block.args.size == operandNames.size) {
            "MANUAL_COMPUTATION body must have one arg per operand; got args=${block.args.size}, operands=${operandNames.size}"
        }

        val inStr = inShardings.joinToString(", ") { it.toSdyAttr() }
        val outStr = outShardings.joinToString(", ") { it.toSdyAttr() }
        val axesStr = manualAxes.joinToString(", ") { "\"$it\"" }
        val operandList = operandNames.joinToString(", ")

        val blockArgList = block.args.joinToString(", ") { arg ->
            ssa[arg.id] = listOf("%${arg.id}")
            "%${arg.id}: ${arg.type.toMlir()}"
        }

        // Single-result output type vs multi-result tuple.
        val resultTypeMlir = if (node.isMultiResult) {
            "(${node.types.joinToString(", ") { it.toMlir() }})"
        } else {
            node.type.toMlir()
        }
        val lhs = if (node.isMultiResult) "$name:${node.numResults}" else name

        val operandTypes = node.operands.joinToString(", ") { it.type.toMlir() }

        out.appendLine(
            "$step$lhs = sdy.manual_computation($operandList) " +
                "in_shardings=[$inStr] " +
                "out_shardings=[$outStr] " +
                "manual_axes={$axesStr} " +
                "($blockArgList) {",
        )
        val innerStep = "$step  "
        emitBlockBody(innerStep, block)
        // Terminator: `sdy.return` with the block's yielded values.
        if (block.terminator.isNotEmpty()) {
            val termRefs = block.terminator.joinToString(", ") { ref(it) }
            val termTypes = block.terminator.joinToString(", ") { it.type.toMlir() }
            out.appendLine("$innerStep" + "sdy.return $termRefs : $termTypes")
        } else {
            out.appendLine("${innerStep}sdy.return")
        }
        out.appendLine("$step} : ($operandTypes) -> $resultTypeMlir")
    }

    /**
     * Emit the body of a block (ops only, excluding the block args which are emitted
     * inline at the block header, and excluding the terminator which is op-specific).
     */
    private fun emitBlockBody(step: String, block: DxirBlock) {
        for (node in block.body) {
            when (node) {
                is DxirConst -> emitConst(step, node)
                is DxirOp -> emitOp(step, node)
                is DxirParam -> error("DxirParam should not appear in a block body")
                is DxirCall -> error("DxirCall lowering in block bodies not yet supported")
                is DxirOpResult -> error("DxirOpResult should not appear directly in a block body")
                is DxirBlockArg -> error("DxirBlockArg should not appear directly in a block body")
            }
        }
    }

    private fun emitSplit(
        step: String,
        node: DxirOp,
        x: String,
        inputType: DxirType,
    ) {
        val axis = normalizeAxis(intAttr(node, "axis"), inputType.rank)
        val sizes = intListAttr(node, "sizes")
        require(sizes.isNotEmpty()) { "SPLIT 'sizes' must be non-empty" }
        require(sizes.all { it > 0 }) { "SPLIT 'sizes' must be positive; got $sizes" }
        require(sizes.sum() == inputType.dims[axis]) {
            "SPLIT sizes sum=${sizes.sum()} must equal input dim $axis = ${inputType.dims[axis]}"
        }
        require(sizes.size == node.numResults) {
            "SPLIT declared ${node.numResults} result types but 'sizes' has ${sizes.size} entries"
        }
        for (k in sizes.indices) {
            val expected = inputType.dims.toMutableList().also { it[axis] = sizes[k] }
            require(node.types[k].dims == expected) {
                "SPLIT result $k shape ${node.types[k].dims} does not match expected $expected"
            }
        }

        val outNames = ArrayList<String>(sizes.size)
        var offset = 0
        for (k in sizes.indices) {
            val start = offset
            val end = offset + sizes[k]
            offset = end
            val sliceName = synth()
            outNames += sliceName
            val rangesStr = (0 until inputType.rank).joinToString(", ") { i ->
                val lo = if (i == axis) start else 0
                val hi = if (i == axis) end else inputType.dims[i]
                "$lo:$hi"
            }
            out.appendLine(
                "$step$sliceName = stablehlo.slice $x [$rangesStr] : (${inputType.toMlir()}) -> ${node.types[k].toMlir()}",
            )
        }
        ssa[node.id] = outNames
    }

    private fun emitBatchNorm(
        step: String,
        name: String,
        operandNames: List<String>,
        node: DxirOp,
    ) {
        require(operandNames.size == 5) {
            "BATCHNORM requires 5 operands (input, scale, offset, mean, variance); got ${operandNames.size}"
        }
        val inputType = node.operands[0].type
        val scaleType = node.operands[1].type
        val offsetType = node.operands[2].type
        val meanType = node.operands[3].type
        val varianceType = node.operands[4].type
        val outputType = node.type

        val eps = readEps(node)
        val featureIndex = (node.attrs["feature_index"] as? Int) ?: 1

        require(inputType.rank >= 2) {
            "BATCHNORM input must be rank ≥ 2; got rank ${inputType.rank}"
        }
        require(featureIndex in 0 until inputType.rank) {
            "BATCHNORM feature_index $featureIndex out of bounds for input rank ${inputType.rank}"
        }
        require(inputType.dims == outputType.dims) {
            "BATCHNORM output shape ${outputType.dims} must match input ${inputType.dims}"
        }
        val expectedPerChannel = inputType.dims[featureIndex]
        for ((label, t) in listOf(
            "scale" to scaleType, "offset" to offsetType,
            "mean" to meanType, "variance" to varianceType,
        )) {
            require(t.rank == 1 && t.dims.single() == expectedPerChannel) {
                "BATCHNORM $label must be rank-1 of size $expectedPerChannel (input dim $featureIndex); got ${t.dims}"
            }
        }

        val (x, scale, offset, mean, variance) = operandNames
        val inputTypes = listOf(inputType, scaleType, offsetType, meanType, varianceType)
            .joinToString(", ") { it.toMlir() }
        out.appendLine(
            """$step$name = "stablehlo.batch_norm_inference"($x, $scale, $offset, $mean, $variance) <{epsilon = $eps : f32, feature_index = $featureIndex : i64}> : ($inputTypes) -> ${outputType.toMlir()}""",
        )
    }

    private fun emitShardConstraint(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val sharding = node.sharding
            ?: error("SHARD_CONSTRAINT requires a sharding attribute on the op; got null")
        require(inputType.dims == node.type.dims) {
            "SHARD_CONSTRAINT preserves shape; got input=${inputType.dims} output=${node.type.dims}"
        }
        out.appendLine(
            "$step$name = sdy.sharding_constraint $x ${sharding.toSdyAttr()} : ${inputType.toMlir()}",
        )
    }

    /**
     * Returns the SSA ref to use for the op's result (typically `%pair#1` since ARGMAX uses
     * a two-output reduce whose second result is the indices tensor).
     */
    private fun emitArgmax(
        step: String,
        x: String,
        axis: Int,
        inputType: DxirType,
        outputType: DxirType,
    ): String {
        val expected = reducedType(inputType, listOf(axis))
        require(expected.dims == outputType.dims) {
            "ARGMAX output shape ${outputType.dims} does not match expected ${expected.dims} for axis=$axis"
        }
        require(outputType.dtype is I32 || outputType.dtype is I64) {
            "ARGMAX output dtype must be integer; got ${outputType.dtype.name}"
        }
        val valueT = inputType.toMlir()
        val idxFullT = DxirType(outputType.dtype, inputType.dims).toMlir()
        val reducedValT = DxirType(inputType.dtype, expected.dims).toMlir()
        val reducedIdxT = outputType.toMlir()
        val scalarValT = "tensor<${mlirElementType(inputType.dtype)}>"
        val scalarIdxT = "tensor<${mlirElementType(outputType.dtype)}>"
        val cmpSuffix = when (inputType.dtype) {
            is F32, is F64 -> "FLOAT"
            is I32, is I64 -> "SIGNED"
            is Bool -> "UNSIGNED"
        }

        // 1. Iota along reduction axis at the output int dtype.
        val iota = synth()
        out.appendLine("$step$iota = stablehlo.iota dim = $axis : $idxFullT")

        // 2. Reduction init values.
        val neginf = synth(); val zeroIdx = synth()
        out.appendLine("$step$neginf = stablehlo.constant dense<${negInfLiteral(inputType.dtype)}> : $scalarValT")
        out.appendLine("$step$zeroIdx = stablehlo.constant dense<0> : $scalarIdxT")

        // 3. Two-input, two-output reduce.
        val pair = synth()
        val curV = synth(); val newV = synth(); val curI = synth(); val newI = synth()
        val gt = synth(); val selV = synth(); val selI = synth()
        out.appendLine(
            "$step$pair:2 = stablehlo.reduce($x init: $neginf), ($iota init: $zeroIdx) across dimensions = [$axis] " +
                ": ($valueT, $idxFullT, $scalarValT, $scalarIdxT) -> ($reducedValT, $reducedIdxT)",
        )
        out.appendLine("$step reducer($curV: $scalarValT, $newV: $scalarValT) ($curI: $scalarIdxT, $newI: $scalarIdxT)  {")
        out.appendLine("$step   $gt = stablehlo.compare  GT, $curV, $newV,  $cmpSuffix : ($scalarValT, $scalarValT) -> tensor<i1>")
        out.appendLine("$step   $selV = stablehlo.select $gt, $curV, $newV : tensor<i1>, $scalarValT")
        out.appendLine("$step   $selI = stablehlo.select $gt, $curI, $newI : tensor<i1>, $scalarIdxT")
        out.appendLine("$step   stablehlo.return $selV, $selI : $scalarValT, $scalarIdxT")
        out.appendLine("$step }")

        // The op's "result" is the indices, which is %pair#1.
        return "$pair#1"
    }

    private fun emitCrossEntropy(
        step: String,
        name: String,
        logits: String,
        labels: String,
        logitsType: DxirType,
        labelsType: DxirType,
        outType: DxirType,
    ) {
        require(logitsType.rank == 2) {
            "CROSS_ENTROPY logits must be rank-2 (B, C); got ${logitsType.dims}"
        }
        require(labelsType.rank == 1) {
            "CROSS_ENTROPY labels must be rank-1 (B,); got ${labelsType.dims}"
        }
        require(labelsType.dims[0] == logitsType.dims[0]) {
            "CROSS_ENTROPY batch dim mismatch: logits B=${logitsType.dims[0]}, labels B=${labelsType.dims[0]}"
        }
        require(labelsType.dtype is I32 || labelsType.dtype is I64) {
            "CROSS_ENTROPY labels must be integer; got ${labelsType.dtype.name}"
        }
        require(outType.dims == listOf(logitsType.dims[0])) {
            "CROSS_ENTROPY output must be rank-1 (B,); got ${outType.dims}"
        }

        val b = logitsType.dims[0]
        val c = logitsType.dims[1]
        val logitsMlir = logitsType.toMlir()
        val labelsMlir = labelsType.toMlir()
        val outMlir = outType.toMlir()

        val elemF = mlirElementType(logitsType.dtype)
        val elemI = mlirElementType(labelsType.dtype)
        val scalarT = "tensor<$elemF>"
        val reducedT = outMlir                                    // (B,)
        val iotaRowT = "tensor<${c}x${elemI}>"                    // (C,)
        val idxMatT = "tensor<${b}x${c}x${elemI}>"                // (B, C)
        val boolMatT = "tensor<${b}x${c}xi1>"

        // --- log_softmax(logits) = logits - logsumexp(logits, axis=-1).broadcast ---
        val maxInit = synth(); val max = synth(); val maxBc = synth(); val shifted = synth()
        out.appendLine("$step$maxInit = stablehlo.constant dense<${negInfLiteral(logitsType.dtype)}> : $scalarT")
        out.appendLine(
            "$step$max = stablehlo.reduce($logits init: $maxInit) applies stablehlo.maximum across dimensions = [1] " +
                ": ($logitsMlir, $scalarT) -> $reducedT",
        )
        out.appendLine("$step$maxBc = stablehlo.broadcast_in_dim $max, dims = [0] : ($reducedT) -> $logitsMlir")
        out.appendLine("$step$shifted = stablehlo.subtract $logits, $maxBc : $logitsMlir")

        val e = synth(); val sumInit = synth(); val s = synth(); val logS = synth()
        out.appendLine("$step$e = stablehlo.exponential $shifted : $logitsMlir")
        out.appendLine("$step$sumInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$s = stablehlo.reduce($e init: $sumInit) applies stablehlo.add across dimensions = [1] " +
                ": ($logitsMlir, $scalarT) -> $reducedT",
        )
        out.appendLine("$step$logS = stablehlo.log $s : $reducedT")

        val lse = synth(); val lseBc = synth(); val logSoftmax = synth()
        out.appendLine("$step$lse = stablehlo.add $logS, $max : $reducedT")
        out.appendLine("$step$lseBc = stablehlo.broadcast_in_dim $lse, dims = [0] : ($reducedT) -> $logitsMlir")
        out.appendLine("$step$logSoftmax = stablehlo.subtract $logits, $lseBc : $logitsMlir")

        // --- one_hot(labels, C) via iota + compare + convert ---
        val iota = synth(); val iotaBc = synth(); val labelsBc = synth()
        out.appendLine("$step$iota = stablehlo.iota dim = 0 : $iotaRowT")
        out.appendLine("$step$iotaBc = stablehlo.broadcast_in_dim $iota, dims = [1] : ($iotaRowT) -> $idxMatT")
        out.appendLine("$step$labelsBc = stablehlo.broadcast_in_dim $labels, dims = [0] : ($labelsMlir) -> $idxMatT")

        val eq = synth(); val oneHotF = synth()
        val intCmp = if (labelsType.dtype is I32 || labelsType.dtype is I64) "SIGNED" else "UNSIGNED"
        out.appendLine("$step$eq = stablehlo.compare  EQ, $labelsBc, $iotaBc,  $intCmp : ($idxMatT, $idxMatT) -> $boolMatT")
        out.appendLine("$step$oneHotF = stablehlo.convert $eq : ($boolMatT) -> $logitsMlir")

        // --- nll = -sum(one_hot * log_softmax, axis=-1) ---
        val product = synth(); val sumInit2 = synth(); val summed = synth()
        out.appendLine("$step$product = stablehlo.multiply $oneHotF, $logSoftmax : $logitsMlir")
        out.appendLine("$step$sumInit2 = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$summed = stablehlo.reduce($product init: $sumInit2) applies stablehlo.add across dimensions = [1] " +
                ": ($logitsMlir, $scalarT) -> $outMlir",
        )
        out.appendLine("$step$name = stablehlo.negate $summed : $outMlir")
    }

    @Suppress("UNCHECKED_CAST")
    private fun conv2dPadding(node: DxirOp): List<List<Int>> {
        val raw = node.attrs["padding"]
            ?: return listOf(listOf(0, 0), listOf(0, 0))
        val list = (raw as? List<List<Int>>)
            ?: error("CONV2D 'padding' must be List<List<Int>> [[top, bot],[left, right]]; got ${raw::class.simpleName}")
        require(list.size == 2 && list.all { it.size == 2 }) {
            "CONV2D padding shape must be [2][2]; got $list"
        }
        return list
    }

    private fun conv2dDilation(node: DxirOp, key: String): List<Int> {
        val raw = node.attrs[key] ?: return listOf(1, 1)
        @Suppress("UNCHECKED_CAST")
        val list = (raw as? List<Int>)
            ?: error("CONV2D '$key' must be List<Int> length 2; got ${raw::class.simpleName}")
        require(list.size == 2) { "CONV2D '$key' must be length 2; got $list" }
        return list
    }

    private fun emitSdpa(
        step: String,
        name: String,
        q: String, k: String, v: String,
        qType: DxirType,
        kType: DxirType,
        vType: DxirType,
        outType: DxirType,
    ) {
        require(qType.rank >= 2 && kType.rank == qType.rank && vType.rank == qType.rank) {
            "SDPA requires Q/K/V of equal rank ≥ 2; got ${qType.rank}, ${kType.rank}, ${vType.rank}"
        }
        val r = qType.rank
        // Last two dims: (sequence, head_dim). Everything before is batch.
        val dK = qType.dims[r - 1]
        val dKK = kType.dims[r - 1]
        val sQ = qType.dims[r - 2]
        val sK = kType.dims[r - 2]
        val sV = vType.dims[r - 2]
        val dV = vType.dims[r - 1]
        require(dK == dKK) { "SDPA Q and K must share last dim (head_dim); got $dK vs $dKK" }
        require(sK == sV) { "SDPA K and V must share sequence length dim; got $sK vs $sV" }
        // Batch dims must match.
        for (i in 0 until r - 2) {
            require(qType.dims[i] == kType.dims[i] && qType.dims[i] == vType.dims[i]) {
                "SDPA batch dim $i mismatch across Q/K/V: ${qType.dims[i]}, ${kType.dims[i]}, ${vType.dims[i]}"
            }
        }
        // Expected output shape: (...batch, sQ, dV)
        val expectedOutDims = qType.dims.take(r - 2) + listOf(sQ, dV)
        require(outType.dims == expectedOutDims) {
            "SDPA output shape ${outType.dims} does not match expected $expectedOutDims"
        }

        val scalarT = "tensor<${mlirElementType(qType.dtype)}>"
        val batchDims = (0 until r - 2).toList()
        val batchDimsStr = batchDims.joinToString(", ")

        // scores shape = (..., sQ, sK)
        val scoresType = DxirType(qType.dtype, qType.dims.take(r - 2) + listOf(sQ, sK))
        val scoresMlir = scoresType.toMlir()

        // 1. scores = Q @ K^T — contract last dim (head_dim) of both.
        val scores = synth()
        out.appendLine(
            "$step$scores = stablehlo.dot_general $q, $k, " +
                (if (batchDims.isNotEmpty()) "batching_dims = [$batchDimsStr] x [$batchDimsStr], " else "") +
                "contracting_dims = [${r - 1}] x [${r - 1}] " +
                ": (${qType.toMlir()}, ${kType.toMlir()}) -> $scoresMlir",
        )

        // 2. Scale by 1/sqrt(d_k), broadcast scalar → scores shape.
        val scaleVal = 1.0f / kotlin.math.sqrt(dK.toFloat())
        val scaleConst = synth(); val scaleBc = synth(); val scaled = synth()
        out.appendLine("$step$scaleConst = stablehlo.constant dense<$scaleVal> : $scalarT")
        out.appendLine("$step$scaleBc = stablehlo.broadcast_in_dim $scaleConst, dims = [] : ($scalarT) -> $scoresMlir")
        out.appendLine("$step$scaled = stablehlo.multiply $scores, $scaleBc : $scoresMlir")

        // 3. Softmax along sK (last dim of scores).
        val softmaxAxis = r - 1
        val reducedT = reducedType(scoresType, listOf(softmaxAxis))
        val reducedMlir = reducedT.toMlir()
        val bcDimsStr = broadcastDimsAfterReducing(r, listOf(softmaxAxis)).joinToString(", ")

        val maxInit = synth(); val max = synth()
        out.appendLine("$step$maxInit = stablehlo.constant dense<${negInfLiteral(qType.dtype)}> : $scalarT")
        out.appendLine(
            "$step$max = stablehlo.reduce($scaled init: $maxInit) applies stablehlo.maximum across dimensions = [$softmaxAxis] " +
                ": ($scoresMlir, $scalarT) -> $reducedMlir",
        )
        val maxBc = synth(); val shifted = synth(); val e = synth()
        out.appendLine("$step$maxBc = stablehlo.broadcast_in_dim $max, dims = [$bcDimsStr] : ($reducedMlir) -> $scoresMlir")
        out.appendLine("$step$shifted = stablehlo.subtract $scaled, $maxBc : $scoresMlir")
        out.appendLine("$step$e = stablehlo.exponential $shifted : $scoresMlir")

        val sumInit = synth(); val s = synth(); val sBc = synth(); val attn = synth()
        out.appendLine("$step$sumInit = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$s = stablehlo.reduce($e init: $sumInit) applies stablehlo.add across dimensions = [$softmaxAxis] " +
                ": ($scoresMlir, $scalarT) -> $reducedMlir",
        )
        out.appendLine("$step$sBc = stablehlo.broadcast_in_dim $s, dims = [$bcDimsStr] : ($reducedMlir) -> $scoresMlir")
        out.appendLine("$step$attn = stablehlo.divide $e, $sBc : $scoresMlir")

        // 4. output = attn @ V — contract attn's last dim (sK) with V's second-last dim (sK).
        out.appendLine(
            "$step$name = stablehlo.dot_general $attn, $v, " +
                (if (batchDims.isNotEmpty()) "batching_dims = [$batchDimsStr] x [$batchDimsStr], " else "") +
                "contracting_dims = [${r - 1}] x [${r - 2}] " +
                ": ($scoresMlir, ${vType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    private fun emitGather(
        step: String,
        name: String,
        operand: String,
        startIndices: String,
        node: DxirOp,
        operandType: DxirType,
        indicesType: DxirType,
    ) {
        val offsetDims = intListAttr(node, "offset_dims")
        @Suppress("UNCHECKED_CAST")
        val collapsedSliceDims = (node.attrs["collapsed_slice_dims"] as? List<Int>) ?: emptyList()
        val startIndexMap = intListAttr(node, "start_index_map")
        val indexVectorDim = intAttr(node, "index_vector_dim")
        val sliceSizes = intListAttr(node, "slice_sizes")
        val indicesAreSorted = (node.attrs["indices_are_sorted"] as? Boolean) ?: false

        require(sliceSizes.size == operandType.rank) {
            "GATHER slice_sizes length ${sliceSizes.size} must equal operand rank ${operandType.rank}"
        }
        require(indexVectorDim in 0..indicesType.rank) {
            "GATHER index_vector_dim $indexVectorDim out of bounds for indices rank ${indicesType.rank}"
        }

        emitGatherOp(
            step = step,
            name = name,
            operand = operand,
            startIndices = startIndices,
            operandType = operandType,
            indicesType = indicesType,
            outType = node.type,
            offsetDims = offsetDims,
            collapsedSliceDims = collapsedSliceDims,
            startIndexMap = startIndexMap,
            indexVectorDim = indexVectorDim,
            sliceSizes = sliceSizes,
            indicesAreSorted = indicesAreSorted,
        )
    }

    private fun emitEmbedding(
        step: String,
        name: String,
        table: String,
        indices: String,
        tableType: DxirType,
        indicesType: DxirType,
        outType: DxirType,
    ) {
        require(tableType.rank == 2) {
            "EMBEDDING table must be rank-2 (vocab, dim); got ${tableType.dims}"
        }
        require(indicesType.dtype is I32 || indicesType.dtype is I64) {
            "EMBEDDING indices must be integer (I32 or I64); got ${indicesType.dtype}"
        }
        val embedDim = tableType.dims[1]
        val expectedOutDims = indicesType.dims + listOf(embedDim)
        require(outType.dims == expectedOutDims) {
            "EMBEDDING output shape ${outType.dims} does not match expected $expectedOutDims " +
                "(indices ${indicesType.dims} ++ [$embedDim])"
        }

        // Canonical embedding-as-gather attrs.
        emitGatherOp(
            step = step,
            name = name,
            operand = table,
            startIndices = indices,
            operandType = tableType,
            indicesType = indicesType,
            outType = outType,
            offsetDims = listOf(indicesType.rank),  // single offset after all index dims
            collapsedSliceDims = listOf(0),         // collapse vocab axis
            startIndexMap = listOf(0),              // indices select from dim 0
            indexVectorDim = indicesType.rank,      // scalar index per position
            sliceSizes = listOf(1, embedDim),
            indicesAreSorted = false,
        )
    }

    private fun emitGatherOp(
        step: String,
        name: String,
        operand: String,
        startIndices: String,
        operandType: DxirType,
        indicesType: DxirType,
        outType: DxirType,
        offsetDims: List<Int>,
        collapsedSliceDims: List<Int>,
        startIndexMap: List<Int>,
        indexVectorDim: Int,
        sliceSizes: List<Int>,
        indicesAreSorted: Boolean,
    ) {
        val dimNumbers = buildString {
            append("#stablehlo.gather<")
            append("offset_dims = [${offsetDims.joinToString(", ")}]")
            append(", collapsed_slice_dims = [${collapsedSliceDims.joinToString(", ")}]")
            append(", start_index_map = [${startIndexMap.joinToString(", ")}]")
            append(", index_vector_dim = $indexVectorDim")
            append(">")
        }
        val sortedPart = if (indicesAreSorted) ", indices_are_sorted = true" else ""
        out.appendLine(
            """$step$name = "stablehlo.gather"($operand, $startIndices) <{dimension_numbers = $dimNumbers, slice_sizes = array<i64: ${sliceSizes.joinToString(", ")}>$sortedPart}> : (${operandType.toMlir()}, ${indicesType.toMlir()}) -> ${outType.toMlir()}""",
        )
    }

    private fun emitDot(
        step: String,
        name: String,
        a: String,
        b: String,
        aType: DxirType,
        bType: DxirType,
        outType: DxirType,
    ) {
        require(aType.rank == 1 && bType.rank == 1) {
            "DOT lowering requires rank-1 inputs; got ${aType.dims} x ${bType.dims}"
        }
        out.appendLine(
            "$step$name = stablehlo.dot_general $a, $b, contracting_dims = [0] x [0] " +
                ": (${aType.toMlir()}, ${bType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    private fun emitCast(
        step: String,
        name: String,
        x: String,
        inputType: DxirType,
        outputType: DxirType,
    ) {
        require(inputType.dims == outputType.dims) {
            "CAST preserves shape; got input=${inputType.dims} output=${outputType.dims}"
        }
        out.appendLine(
            "$step$name = stablehlo.convert $x : (${inputType.toMlir()}) -> ${outputType.toMlir()}",
        )
    }

    private fun emitReshape(
        step: String,
        name: String,
        x: String,
        inputType: DxirType,
        outputType: DxirType,
    ) {
        require(inputType.elementCount == outputType.elementCount) {
            "RESHAPE requires matching element count; got input=${inputType.elementCount}, output=${outputType.elementCount}"
        }
        out.appendLine(
            "$step$name = stablehlo.reshape $x : (${inputType.toMlir()}) -> ${outputType.toMlir()}",
        )
    }

    private fun emitTranspose(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val perm = intListAttr(node, "permutation")
        require(perm.size == inputType.rank) {
            "TRANSPOSE permutation length ${perm.size} must equal input rank ${inputType.rank}"
        }
        require(perm.toSet() == (0 until inputType.rank).toSet()) {
            "TRANSPOSE permutation $perm is not a valid permutation of [0..${inputType.rank - 1}]"
        }
        out.appendLine(
            "$step$name = stablehlo.transpose $x, dims = [${perm.joinToString(", ")}] " +
                ": (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun emitBroadcast(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val bcastDims = intListAttr(node, "broadcast_dimensions")
        require(bcastDims.size == inputType.rank) {
            "BROADCAST broadcast_dimensions length ${bcastDims.size} must equal input rank ${inputType.rank}"
        }
        out.appendLine(
            "$step$name = stablehlo.broadcast_in_dim $x, dims = [${bcastDims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun emitConcat(
        step: String,
        name: String,
        operandNames: List<String>,
        node: DxirOp,
    ) {
        val dim = intAttr(node, "dimension")
        require(operandNames.size >= 2) {
            "CONCAT requires at least 2 operands; got ${operandNames.size}"
        }
        val operandList = operandNames.joinToString(", ")
        val inputTypes = node.operands.joinToString(", ") { it.type.toMlir() }
        out.appendLine(
            "$step$name = stablehlo.concatenate $operandList, dim = $dim " +
                ": ($inputTypes) -> ${node.type.toMlir()}",
        )
    }

    private fun emitSlice(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val starts = intListAttr(node, "start_indices")
        val limits = intListAttr(node, "limit_indices")
        val strides = intListAttr(node, "strides")
        require(starts.size == inputType.rank && limits.size == inputType.rank && strides.size == inputType.rank) {
            "SLICE attr lengths must match input rank ${inputType.rank}"
        }
        out.appendLine(
            "$step$name = stablehlo.slice $x [" +
                starts.indices.joinToString(", ") { i ->
                    val stride = strides[i]
                    if (stride == 1) "${starts[i]}:${limits[i]}" else "${starts[i]}:${limits[i]}:$stride"
                } +
                "] : (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun negInfLiteral(dtype: DType): String = when (dtype) {
        is F32 -> "0xFF800000"
        is F64 -> "0xFFF0000000000000"
        is I32 -> Int.MIN_VALUE.toString()
        is I64 -> Long.MIN_VALUE.toString()
        is Bool -> "false"
    }

    private fun posInfLiteral(dtype: DType): String = when (dtype) {
        is F32 -> "0x7F800000"
        is F64 -> "0x7FF0000000000000"
        is I32 -> Int.MAX_VALUE.toString()
        is I64 -> Long.MAX_VALUE.toString()
        is Bool -> "true"
    }

    private fun synth(): String = "%s${nextSynth++}"
}
