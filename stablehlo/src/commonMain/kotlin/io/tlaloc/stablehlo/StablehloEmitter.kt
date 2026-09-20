package io.tlaloc.stablehlo

import io.tlaloc.core.Bool
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
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor

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
            // §0.4.400 — integer-typed rank-N consts (embedding index vectors in
            // gradient graphs) format as integer literals; the FloatArray is just
            // the dxir const carrier.
            is FloatArray ->
                if (node.type.dtype is I32 || node.type.dtype is I64) {
                    denseIntFromArray(v, node.type.dims)
                } else {
                    denseFromArray(v, node.type.dims)
                }
            else -> error("non-numeric DxirConst value: $v (${v::class.simpleName})")
        }
        out.appendLine("$step$name = stablehlo.constant dense<$literal> : ${node.type.toMlir()}")
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
            OpKind.SIN -> unary(step, name, "stablehlo.sine", ops[0], outType)
            OpKind.COS -> unary(step, name, "stablehlo.cosine", ops[0], outType)
            // §0.4.395 — Phase C2 trig tails. `stablehlo.tan` is a first-class op
            // (StableHLO ≥ 1.7 moved it in from CHLO; certified against the
            // GB10's XLA in PjrtTanAtanSmokeTest). StableHLO has no unary atan,
            // so ATAN emits as `atan2(x, splat 1.0)` — exact on the whole real
            // line since atan2(y, 1) ≡ atan(y).
            OpKind.TAN -> unary(step, name, "stablehlo.tan", ops[0], outType)
            OpKind.ATAN -> emitAtan(step, name, ops[0], node.type)
            OpKind.SQRT -> unary(step, name, "stablehlo.sqrt", ops[0], outType)
            OpKind.RSQRT -> unary(step, name, "stablehlo.rsqrt", ops[0], outType)
            OpKind.TANH -> unary(step, name, "stablehlo.tanh", ops[0], outType)
            OpKind.SIGMOID -> unary(step, name, "stablehlo.logistic", ops[0], outType)
            // §0.4.393 — SIGN was the one user-reachable kind with no emitter arm:
            // `ops.sign` is in the FIR unary map, and the MAX/MIN reduction rule
            // builds its extremum indicator as `1 - sign(y − x)`, so `grad { x.max(0)
            // .sum() }` could not be emitted. StableHLO's `sign` matches the
            // interpreter's ±1/0 convention; the only divergence is the sign bit of
            // zero for a −0.0 input (spec: −0.0, interpreter: 0.0), which is
            // unobservable here since every consumer compares or subtracts it.
            OpKind.SIGN -> unary(step, name, "stablehlo.sign", ops[0], outType)

            // Elementwise binary. §0.4.277 — operand shapes may differ from
            // the result type (NumPy-style broadcast). StableHLO requires
            // both operands to match the result type, so we inject an
            // explicit `broadcast_in_dim` whenever an operand needs widening.
            OpKind.ADD -> binary(step, name, "stablehlo.add", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)
            OpKind.SUB -> binary(step, name, "stablehlo.subtract", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)
            OpKind.MUL -> binary(step, name, "stablehlo.multiply", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)
            OpKind.DIV -> binary(step, name, "stablehlo.divide", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)
            OpKind.POW -> binary(step, name, "stablehlo.power", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)

            // Type conversion
            OpKind.CAST -> emitCast(step, name, ops[0], node.operands[0].type, node.type)

            // Shape ops
            OpKind.RESHAPE -> emitReshape(step, name, ops[0], node.operands[0].type, node.type)
            OpKind.TRANSPOSE -> emitTranspose(step, name, ops[0], node, node.operands[0].type)
            // §0.4.396 — REVERSE (flip along literal axes, Phase C3).
            OpKind.REVERSE -> emitReverse(step, name, ops[0], node, node.operands[0].type)
            OpKind.BROADCAST -> emitBroadcast(step, name, ops[0], node, node.operands[0].type)
            OpKind.CONCAT -> emitConcat(step, name, ops, node)
            OpKind.SLICE -> emitSlice(step, name, ops[0], node, node.operands[0].type)
            // §0.4.360 — shape-plumbing activation.
            OpKind.WHERE -> emitWhere(step, name, ops, node)
            OpKind.COMPARE -> emitCompare(step, name, ops, node)
            OpKind.PAD -> emitPad(step, name, ops[0], node, node.operands[0].type)

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
            // §0.4.373 — SUM_TO (numpy unbroadcast): reduce operand[0] to the
            // template (operand[1]) shape. Reduce axes derived from the concrete
            // operand dims (at emit time dims are concrete, never sentinels).
            OpKind.SUM_TO -> emitSumTo(
                step, name, ops[0], node.operands[0].type, node.operands[1].type,
            )
            // §0.4.399 — BROADCAST_LIKE (broadcast-to-template): SUM_TO's forward
            // twin and VJP. Emit-time dims are concrete, so it folds to a static
            // broadcast_in_dim; the template's SSA value goes unreferenced.
            OpKind.BROADCAST_LIKE -> emitBroadcastLike(step, name, ops[0], node)
            // §0.4.374 — PAD_TO (zero-pad to template): the SLICE adjoint. `high`
            // derived from the concrete template (operand[1] == node.type) dims.
            OpKind.PAD_TO -> emitPadTo(
                step, name, ops[0], node, node.operands[0].type,
            )
            // §0.4.399 — SLICE_AT (window at a literal offset): PAD_TO's reverse
            // mirror and VJP. The bounds fold to literals at emit time; the
            // template's SSA value goes unreferenced.
            OpKind.SLICE_AT -> emitSliceAt(step, name, ops[0], node)
            // Phase A2b — SLICE_LIKE (CONCAT's adjoint): extract the window whose
            // start is the sum of the prior templates' axis extents and whose length
            // is `thisTemplate`'s. At emit time every dim is concrete, so the bounds
            // fold to literals and this is an ordinary static `stablehlo.slice`. The
            // template operands' SSA values go unreferenced here (they exist for the
            // host path's runtime extents) — MLIR-legal, and DCE'd by XLA when
            // nothing else uses them.
            OpKind.SLICE_LIKE -> emitSliceLike(step, name, ops[0], node)
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
            // §0.4.385 — the fused conv adjoints (padding solved at emit time).
            OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT ->
                emitConvAdjoint(step, name, ops, node)
            // §0.4.393 — the fused transposed-conv adjoints, via the
            // conv-of-the-dilated-input identity.
            OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT, OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT ->
                emitConvTransposeAdjoint(step, name, ops, node)
            // §0.4.363 — window pooling via stablehlo.reduce_window.
            OpKind.MAXPOOL2D, OpKind.AVGPOOL2D -> emitReduceWindow(
                step, name, ops[0], node, node.operands[0].type,
            )
            // §0.4.386 — the fused avgpool adjoint (padding solved at emit time).
            OpKind.AVGPOOL2D_GRAD -> emitAvgPoolGrad(step, name, ops[0], node)
            // §0.4.389 — the fused maxpool adjoint (upsample-and-mask, expanded here).
            OpKind.MAXPOOL2D_GRAD -> emitMaxPoolGrad(step, name, ops, node)
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
            OpKind.LAND -> binary(step, name, "stablehlo.and", ops[0], ops[1], node.operands[0].type, node.operands[1].type, node.type)
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
            OpKind.COARSENED -> {
                // Layer 4.1 §0.4.261 — when L3.3's kernel-lowering pass has
                // attached a [KernelDescriptor] under [KernelDescriptor.ATTR_KEY],
                // emit a `stablehlo.custom_call` consuming that descriptor.
                // When the attr is absent the COARSENED is still a compile-time
                // artefact (the splice op): reaching here means the coarsen →
                // kernel-lowering chain neither annotated nor decomposed it,
                // which is a compiler bug. Loud failure beats silent miscompile.
                val descriptor = node.attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
                if (descriptor != null) {
                    emitCustomCall(step, name, ops, node, descriptor)
                } else {
                    error(
                        "StableHLO lowering for OpKind.COARSENED requires a " +
                            "'${KernelDescriptor.ATTR_KEY}' attr (set by L3.3 lowerKernelChoice) " +
                            "or decomposition; reaching here means the coarsen → kernel-lowering " +
                            "chain neither annotated nor decomposed this op (op id=${node.id})",
                    )
                }
            }
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
            // §0.4.400 — the shape template (operand 2) is deliberately not
            // referenced: the concrete result type already carries [V, D] here.
            OpKind.EMBEDDING_GRAD -> emitEmbeddingGrad(
                step, name,
                indices = ops[0], upstream = ops[1],
                node = node,
                indicesType = node.operands[0].type,
                upstreamType = node.operands[1].type,
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

    private fun binary(
        step: String,
        name: String,
        op: String,
        a: String,
        b: String,
        aType: DxirType,
        bType: DxirType,
        resultType: DxirType,
    ) {
        val aFinal = broadcastIfNeeded(step, a, aType, resultType)
        val bFinal = broadcastIfNeeded(step, b, bType, resultType)
        out.appendLine("$step$name = $op $aFinal, $bFinal : ${resultType.toMlir()}")
    }

    /**
     * §0.4.277 — Inject `stablehlo.broadcast_in_dim` if [operandType] doesn't
     * match [resultType]. Returns the SSA name to use downstream (the original
     * if no broadcast was needed; the broadcast result otherwise).
     *
     * v1 was same-rank only (each input dim equal to the result dim or 1).
     * Phase A5c generalises it to NumPy right-alignment, which is what the
     * elementwise binaries' implicit broadcasting needs: the operand's axis `j`
     * maps to result axis `offset + j` where `offset = resultRank − operandRank`,
     * a size-1 operand axis stretches, and the unlisted LEADING result axes are
     * new replicated ones. An operand of HIGHER rank than the result is still
     * refused — broadcasting never drops axes.
     */
    private fun broadcastIfNeeded(
        step: String,
        operandName: String,
        operandType: DxirType,
        resultType: DxirType,
    ): String {
        if (operandType == resultType) return operandName
        require(operandType.dtype == resultType.dtype) {
            "broadcastIfNeeded: dtype mismatch (operand=${operandType.dtype} result=${resultType.dtype}); " +
                "stablehlo.broadcast_in_dim is shape-only — dtype must match. Insert an explicit CAST first."
        }
        require(operandType.rank <= resultType.rank) {
            "broadcastIfNeeded: operand rank ${operandType.dims} exceeds result rank ${resultType.dims}; " +
                "broadcasting extends leading axes only (NumPy right-alignment)"
        }
        val offset = resultType.rank - operandType.rank
        for (i in 0 until operandType.rank) {
            require(operandType.dims[i] == resultType.dims[offset + i] || operandType.dims[i] == 1) {
                "broadcastIfNeeded: dim $i operand=${operandType.dims[i]} vs result=${resultType.dims[offset + i]} " +
                    "is not broadcast-compatible (operand must match or be 1)"
            }
        }
        val bcast = synth()
        // Right-aligned axis map: operand axis i → result axis offset + i.
        val dims = (0 until operandType.rank).joinToString(", ") { "${it + offset}" }
        out.appendLine(
            "$step$bcast = stablehlo.broadcast_in_dim $operandName, dims = [$dims] : " +
                "(${operandType.toMlir()}) -> ${resultType.toMlir()}",
        )
        return bcast
    }

    private fun emitAtan(step: String, name: String, x: String, type: DxirType) {
        // §0.4.395 — atan(x) = atan2(x, 1.0). StableHLO ships atan2 but no unary
        // atan; the x2 = 1 splat pins quadrant I/IV, where atan2(y, 1) ≡ atan(y)
        // exactly (including ±0.0 and ±∞ per IEEE atan2 semantics).
        val one = synth()
        out.appendLine("$step$one = stablehlo.constant dense<1.0> : ${type.toMlir()}")
        out.appendLine("$step$name = stablehlo.atan2 $x, $one : ${type.toMlir()}")
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

    /**
     * Resolve the intermediate (post-`stablehlo.reduce`) shape, plus whether
     * the caller wants keep-dims (size-1 reduced axes preserved in the output).
     *
     * § 0.4.274 — keep-dims is detected by [outputType] having the same rank
     * as [inputType] (with the reduced axes as size 1). The recognized pattern
     * for RmsNorm (`MUL → MEAN → ... → MUL`) emits the keepdims form, which
     * is what real Llama / Mistral code does. Without this, the LlamaDecoder
     * primal — and any other recognized RMS-norm chain — fails to lower.
     */
    private fun resolveReduceShape(
        inputType: DxirType,
        outputType: DxirType,
        dims: List<Int>,
    ): Pair<DxirType, Boolean> {
        val dropped = reducedType(inputType, dims)
        return when (outputType.rank) {
            dropped.rank -> {
                require(outputType.dims == dropped.dims) {
                    "reduce output shape ${outputType.dims} does not match expected ${dropped.dims} for reducing dims=$dims from ${inputType.dims}"
                }
                dropped to false
            }
            inputType.rank -> {
                val keepDimsExpected = inputType.dims.toMutableList().also { for (d in dims) it[d] = 1 }
                require(outputType.dims == keepDimsExpected) {
                    "keep-dims reduce output shape ${outputType.dims} does not match expected $keepDimsExpected for reducing dims=$dims from ${inputType.dims}"
                }
                dropped to true
            }
            else -> error(
                "reduce output rank ${outputType.rank} must equal either drop-dims rank ${dropped.rank} or input rank ${inputType.rank}; output=${outputType.dims} input=${inputType.dims} dims=$dims",
            )
        }
    }

    /**
     * Re-inflate a reduced (rank N-K) tensor back to the keep-dims (rank N)
     * shape via `stablehlo.broadcast_in_dim`. Returns the SSA name of the
     * inflated value.
     */
    private fun inflateKeepDims(
        step: String,
        name: String,
        intermName: String,
        intermType: DxirType,
        outputType: DxirType,
        inputRank: Int,
        dims: List<Int>,
    ) {
        val keptAxes = (0 until inputRank).filter { it !in dims }
        out.appendLine(
            "$step$name = stablehlo.broadcast_in_dim $intermName, dims = [${keptAxes.joinToString(", ")}] " +
                ": (${intermType.toMlir()}) -> ${outputType.toMlir()}",
        )
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
        val (intermType, keepDims) = resolveReduceShape(inputType, outputType, dims)
        val elem = mlirElementType(inputType.dtype)
        val scalarT = "tensor<$elem>"
        val init = synth()
        val reduceTarget = if (keepDims) synth() else name
        out.appendLine("$step$init = stablehlo.constant dense<$initLiteral> : $scalarT")
        out.appendLine(
            "$step$reduceTarget = stablehlo.reduce($x init: $init) applies $reducer across dimensions = [${dims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}, $scalarT) -> ${intermType.toMlir()}",
        )
        if (keepDims) {
            inflateKeepDims(step, name, reduceTarget, intermType, outputType, inputType.rank, dims)
        }
    }

    private fun emitReduceMean(
        step: String,
        name: String,
        x: String,
        inputType: DxirType,
        outputType: DxirType,
        dims: List<Int>,
    ) {
        val (intermType, keepDims) = resolveReduceShape(inputType, outputType, dims)
        val elem = mlirElementType(inputType.dtype)
        val scalarT = "tensor<$elem>"
        val intermMlir = intermType.toMlir()
        val init = synth()
        val sumName = synth()

        out.appendLine("$step$init = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$sumName = stablehlo.reduce($x init: $init) applies stablehlo.add across dimensions = [${dims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}, $scalarT) -> $intermMlir",
        )
        // divisor = product of reduced-dim sizes, emitted at *intermediate* type
        // (so the divide happens before keep-dims inflation when applicable)
        val n = dims.map { inputType.dims[it].toLong() }.fold(1L) { acc, d -> acc * d }.coerceAtLeast(1L)
        val divisor = synth()
        out.appendLine("$step$divisor = stablehlo.constant dense<$n.0> : $intermMlir")
        val divTarget = if (keepDims) synth() else name
        out.appendLine("$step$divTarget = stablehlo.divide $sumName, $divisor : $intermMlir")
        if (keepDims) {
            inflateKeepDims(step, name, divTarget, intermType, outputType, inputType.rank, dims)
        }
    }

    /**
     * §0.4.373 — SUM_TO (numpy unbroadcast): reduce [value] (shape U) down to
     * [template]'s shape (T, T.rank ≤ U.rank, right-aligned). Reduce axes = the
     * leading (U.rank − T.rank) axes ∪ the aligned axes where T == 1 but U > 1.
     * Emitted as a `stablehlo.reduce(add)` over those axes (yielding the dropped
     * shape) followed by a `stablehlo.reshape` re-inserting the size-1 axes to
     * land exactly on T. Dims are concrete at emit time (never sentinels).
     */
    private fun emitSumTo(
        step: String,
        name: String,
        x: String,
        valueType: DxirType,
        templateType: DxirType,
    ) {
        val u = valueType.dims
        val t = templateType.dims
        val ru = u.size
        val rt = t.size
        require(rt <= ru) { "SUM_TO template rank $rt exceeds value rank $ru" }
        val offset = ru - rt
        val reduceAxes = mutableListOf<Int>()
        for (i in 0 until offset) reduceAxes.add(i)
        for (i in 0 until rt) {
            require(t[i] == u[offset + i] || t[i] == 1) {
                "SUM_TO template dim $i = ${t[i]} incompatible with value axis ${offset + i} = ${u[offset + i]}"
            }
            if (t[i] == 1 && u[offset + i] != 1) reduceAxes.add(offset + i)
        }
        if (reduceAxes.isEmpty()) {
            // True identity (T == U): pass the value through unchanged.
            emitReshape(step, name, x, valueType, templateType)
            return
        }
        val dropped = reducedType(valueType, reduceAxes)
        val needReshape = dropped.dims != templateType.dims
        val reduceTarget = if (needReshape) synth() else name
        val elem = mlirElementType(valueType.dtype)
        val scalarT = "tensor<$elem>"
        val init = synth()
        out.appendLine("$step$init = stablehlo.constant dense<0.0> : $scalarT")
        out.appendLine(
            "$step$reduceTarget = stablehlo.reduce($x init: $init) applies stablehlo.add across dimensions = [${reduceAxes.joinToString(", ")}] " +
                ": (${valueType.toMlir()}, $scalarT) -> ${dropped.toMlir()}",
        )
        if (needReshape) {
            emitReshape(step, name, reduceTarget, dropped, templateType)
        }
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

        // Layer 1 §0.4.241+ — defensive named-inference path. When the
        // operands' DxirType.axisNames are populated AND no explicit
        // contracting/batching attrs were supplied (e.g. hand-built DXIR
        // from a builder test), derive contracting dim positions from the
        // shared axis name. The K2 plugin's contract lowering always sets
        // the explicit attrs (see emitContract in FirLambdaToDxirLowering),
        // so the production path goes through the explicit branch below;
        // this branch exists for emitter-test ergonomics and as a forward
        // compatibility hook for future builder APIs.
        if (!explicit && aType.axisNames.isNotEmpty() && bType.axisNames.isNotEmpty()) {
            val lhsNames = aType.axisNames
            val rhsNames = bType.axisNames
            val shared = lhsNames.filterNotNull().toSet()
                .intersect(rhsNames.filterNotNull().toSet())
            require(shared.size == 1) {
                "named MATMUL inference requires exactly one shared axis name; " +
                    "got lhs=$lhsNames rhs=$rhsNames shared=$shared. " +
                    "For multi-axis or no-shared-axis contraction, supply " +
                    "lhs_contracting_dims / rhs_contracting_dims explicitly."
            }
            val sharedName = shared.single()
            val lhsContract = listOf(lhsNames.indexOf(sharedName))
            val rhsContract = listOf(rhsNames.indexOf(sharedName))
            out.appendLine(
                "$step$name = stablehlo.dot_general $a, $b, " +
                    "contracting_dims = [${lhsContract.joinToString(", ")}] x [${rhsContract.joinToString(", ")}] " +
                    ": (${aType.toMlir()}, ${bType.toMlir()}) -> ${outType.toMlir()}",
            )
            return
        }

        if (!explicit) {
            // §0.4.135 — canonical batched MATMUL convention: all leading axes are
            // batching dims, and the last two are the M/K (lhs) / K/N (rhs) slot.
            // Rank-2 falls through with no batching dims (the original path).
            // Rank-3+ infers `batching_dims = [0..r-3]`, `contracting_dims = [r-1] x [r-2]`.
            require(aType.rank >= 2 && bType.rank >= 2) {
                "MATMUL without batching/contracting attrs requires rank ≥ 2 inputs; got ${aType.dims} x ${bType.dims}."
            }
            require(aType.rank == bType.rank) {
                "MATMUL without batching/contracting attrs requires matching ranks for canonical " +
                    "batched matmul; got ${aType.dims} x ${bType.dims}. For mixed ranks, supply " +
                    "lhs_contracting_dims / rhs_contracting_dims explicitly."
            }
            val r = aType.rank
            val batchPart = if (r == 2) {
                ""
            } else {
                val batchDims = (0 until r - 2).joinToString(", ")
                "batching_dims = [$batchDims] x [$batchDims], "
            }
            out.appendLine(
                "$step$name = stablehlo.dot_general $a, $b, ${batchPart}" +
                    "contracting_dims = [${r - 1}] x [${r - 2}] " +
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

    /**
     * §0.4.363 — MAXPOOL2D / AVGPOOL2D as `stablehlo.reduce_window` in
     * generic form (there is no compact sugar for reduce_window): max
     * with −∞ init, add with 0 init; AVGPOOL2D follows with a splat
     * multiply by 1/(kh·kw) — the count_include_pad convention pinned at
     * the interpreter's [DxirInterpreter.evalPool2d].
     */
    private fun emitReduceWindow(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        require(inputType.rank == 4 && node.type.rank == 4) {
            "${node.op} requires rank-4 NCHW input/output; got ${inputType.dims} / ${node.type.dims}"
        }
        val window = intListAttr(node, "window")
        require(window.size == 2) { "${node.op} needs `window` [kh, kw]; got $window" }
        val strides = (node.attrs["window_strides"] as? List<*>)?.map { (it as Number).toInt() }
            ?: window
        val padding = (node.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
            ?: listOf(listOf(0, 0), listOf(0, 0))

        val isMax = node.op == OpKind.MAXPOOL2D
        val elem = mlirElementType(inputType.dtype)
        val scalarT = "tensor<$elem>"
        val initLit = if (isMax) negInfLiteral(inputType.dtype) else "0.0"
        val reducer = if (isMax) "stablehlo.maximum" else "stablehlo.add"

        val init = synth()
        val a = synth(); val b = synth(); val r = synth()
        val windowTarget = if (isMax) name else synth()
        val padRows = "[[0, 0], [0, 0], [${padding[0][0]}, ${padding[0][1]}], " +
            "[${padding[1][0]}, ${padding[1][1]}]]"
        out.appendLine("$step$init = stablehlo.constant dense<$initLit> : $scalarT")
        out.appendLine(
            """$step$windowTarget = "stablehlo.reduce_window"($x, $init) <{""" +
                "window_dimensions = array<i64: 1, 1, ${window[0]}, ${window[1]}>, " +
                "window_strides = array<i64: 1, 1, ${strides[0]}, ${strides[1]}>, " +
                "padding = dense<$padRows> : tensor<4x2xi64>}> ({",
        )
        out.appendLine("$step ^bb0($a: $scalarT, $b: $scalarT):")
        out.appendLine("$step   $r = $reducer $a, $b : $scalarT")
        out.appendLine("$step   stablehlo.return $r : $scalarT")
        out.appendLine("$step}) : (${inputType.toMlir()}, $scalarT) -> ${node.type.toMlir()}")
        if (!isMax) {
            val inv = synth()
            val invVal = 1.0 / (window[0] * window[1])
            out.appendLine(
                "$step$inv = stablehlo.constant dense<$invVal> : ${node.type.toMlir()}",
            )
            out.appendLine(
                "$step$name = stablehlo.multiply $windowTarget, $inv : ${node.type.toMlir()}",
            )
        }
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
        val strides = intListAttr(node, "window_strides")
        require(strides.size == 2) { "conv window_strides must be length 2; got $strides" }
        // §0.4.362 — spatial kernel flip (the conv adjoint's dX needs it).
        val reversal = (node.attrs["window_reversal"] as? List<*>)?.map { it as Boolean }
        emitConvolution(
            step, name, lhs, rhs,
            lhsType = lhsType, rhsType = rhsType, outType = node.type,
            kernelLayout = kernelLayout,
            strides = strides,
            padding = conv2dPadding(node),
            lhsDilation = conv2dDilation(node, "lhs_dilation"),
            rhsDilation = conv2dDilation(node, "rhs_dilation"),
            reversal = reversal,
            featureGroupCount = (node.attrs["feature_group_count"] as? Int) ?: 1,
            batchGroupCount = (node.attrs["batch_group_count"] as? Int) ?: 1,
        )
    }

    /**
     * §0.4.385 — the `stablehlo.convolution` line itself, with every window attr
     * passed in rather than read off a [DxirOp]. [emitConv2d] supplies them from
     * the node's attrs; [emitConvAdjoint] supplies the padding it has just SOLVED
     * at emit time (and the dilations/stride/reversal the adjoint spelling needs).
     */
    private fun emitConvolution(
        step: String,
        name: String,
        lhs: String,
        rhs: String,
        lhsType: DxirType,
        rhsType: DxirType,
        outType: DxirType,
        kernelLayout: String,
        strides: List<Int>,
        padding: List<List<Int>>,
        lhsDilation: List<Int>,
        rhsDilation: List<Int>,
        reversal: List<Boolean>?,
        featureGroupCount: Int = 1,
        batchGroupCount: Int = 1,
    ) {
        require(lhsType.rank == 4) { "conv input must be rank-4 NCHW; got dims ${lhsType.dims}" }
        require(rhsType.rank == 4) { "conv kernel must be rank-4; got dims ${rhsType.dims}" }
        require(outType.rank == 4) { "conv output must be rank-4; got dims ${outType.dims}" }

        val strideStr = strides.joinToString(", ")
        val padStr = padding.joinToString(", ") { "[${it.joinToString(", ")}]" }
        val lhsDilStr = lhsDilation.joinToString(", ")
        val rhsDilStr = rhsDilation.joinToString(", ")
        val reverseStr = if (reversal != null && reversal.any { it }) {
            ", reverse = [${reversal.joinToString(", ")}]"
        } else {
            ""
        }

        out.appendLine(
            "$step$name = stablehlo.convolution($lhs, $rhs) " +
                "dim_numbers = [b, f, 0, 1]x${kernelLayout}->[b, f, 0, 1], " +
                "window = {stride = [$strideStr], pad = [$padStr], " +
                "lhs_dilate = [$lhsDilStr], rhs_dilate = [$rhsDilStr]$reverseStr} " +
                "{batch_group_count = $batchGroupCount : i64, " +
                "feature_group_count = $featureGroupCount : i64} " +
                ": (${lhsType.toMlir()}, ${rhsType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    /**
     * §0.4.385 — the fused conv adjoints (see [io.tlaloc.ir.OpKind.CONV2D_DATA_ADJOINT]).
     *
     * Their padding is a function of the primal's extents, and at EMIT time those
     * extents are concrete (the GPU path requires static shapes), so the very solve
     * the interpreter and the host twins perform at runtime is performed here once
     * — and what lands in the module is an ordinary `stablehlo.convolution`. The
     * resulting MLIR is the same the pre-fusion rule produced, which is why
     * §0.4.362's real-XLA certification still covers this path.
     *
     * The kernel adjoint emits its batch↔feature transposes EXPLICITLY rather than
     * spelling the swap through `dim_numbers`: identical semantics to the other two
     * engines and no reliance on output-dim-number subtleties. XLA folds the
     * transposes into the convolution's layout.
     */
    private fun emitConvAdjoint(step: String, name: String, ops: List<String>, node: DxirOp) {
        val dataAdj = node.op == OpKind.CONV2D_DATA_ADJOINT
        require(node.operands.size == 3) {
            "${node.op} takes (upstream, kernel, xTemplate) or (x, upstream, wTemplate); " +
                "got ${node.operands.size} operands"
        }
        val upType = if (dataAdj) node.operands[0].type else node.operands[1].type
        val otherType = if (dataAdj) node.operands[1].type else node.operands[0].type
        val upRef = if (dataAdj) ops[0] else ops[1]
        val otherRef = if (dataAdj) ops[1] else ops[0]
        val outType = node.type
        require(upType.rank == 4 && otherType.rank == 4 && outType.rank == 4) {
            "${node.op} requires rank-4 operands and result; got " +
                "${upType.dims} / ${otherType.dims} / ${outType.dims}"
        }
        val s = intListAttr(node, "window_strides")
        require(s.size == 2) { "${node.op} window_strides must be length 2; got $s" }
        val d = conv2dDilation(node, "rhs_dilation")
        val primalPad = conv2dPadding(node)
        // operands[2] is the shape template: its extents are the result's, which
        // `outType` already carries, so it contributes nothing to the MLIR.
        if (dataAdj) {
            emitDataAdjointConvolution(
                step, name, upRef, otherRef, upType, otherType, outType, s, d, primalPad,
            )
            return
        }
        emitKernelAdjointExpansion(
            step, name, otherRef, upRef, otherType, upType, outType, s, d, primalPad,
        )
    }

    /**
     * §0.4.393 — the solved-padding, lhs-dilated, tap-reversed transposed
     * convolution that IS a forward conv's data adjoint. Lifted out of
     * [emitConvAdjoint] so the transposed-conv adjoints can reuse it: there the
     * primal is `conv(dilate(x, L), swap01(w))`, so this runs against the DILATED
     * shape ([outType] = the dilated input's) and the caller slices the dilation
     * back out afterwards.
     *
     * The solve, per spatial axis, with `hOut` the upstream's extent and `H` the
     * target's: `dilSize = (hOut−1)·s + 1`, `kEff = (k−1)·d + 1`,
     * `low = kEff − 1 − p_low`, `high = p_low + H − dilSize`.
     */
    private fun emitDataAdjointConvolution(
        step: String,
        name: String,
        upRef: String,
        kernelRef: String,
        upType: DxirType,
        kernelType: DxirType,
        outType: DxirType,
        strides: List<Int>,
        rhsDil: List<Int>,
        primalPad: List<List<Int>>,
    ) {
        val padding = (0..1).map { a ->
            val dilSize = (upType.dims[2 + a] - 1) * strides[a] + 1
            val kEff = (kernelType.dims[2 + a] - 1) * rhsDil[a] + 1
            val low = kEff - 1 - primalPad[a][0]
            listOf(low, primalPad[a][0] + outType.dims[2 + a] - dilSize)
        }
        emitConvolution(
            step, name, upRef, kernelRef,
            lhsType = upType, rhsType = kernelType, outType = outType,
            kernelLayout = "[i, o, 0, 1]",
            strides = listOf(1, 1),
            padding = padding,
            lhsDilation = strides,
            rhsDilation = rhsDil,
            reversal = listOf(true, true),
        )
    }

    /**
     * §0.4.393 — the transpose/conv/transpose expansion that IS a forward conv's
     * kernel adjoint, lifted out of [emitConvAdjoint] for the same reason.
     * [outType] is the final OIHW kernel shape; the inner convolution produces it
     * with the channel axes swapped. Solve per axis: `dilSize = (hOut−1)·s + 1`,
     * `low = p_low`, `high = (k−1)·d + dilSize − H − p_low`, where `k` is the
     * target kernel's spatial extent and `H` the primal input's.
     */
    private fun emitKernelAdjointExpansion(
        step: String,
        name: String,
        xRef: String,
        upRef: String,
        xType: DxirType,
        upType: DxirType,
        outType: DxirType,
        strides: List<Int>,
        rhsDil: List<Int>,
        primalPad: List<List<Int>>,
    ) {
        val padding = (0..1).map { a ->
            val dilSize = (upType.dims[2 + a] - 1) * strides[a] + 1
            val low = primalPad[a][0]
            listOf(low, (outType.dims[2 + a] - 1) * rhsDil[a] + dilSize - xType.dims[2 + a] - low)
        }
        val swap = listOf(1, 0, 2, 3)
        val xTType = DxirType(outType.dtype, swap.map { xType.dims[it] })
        val upTType = DxirType(outType.dtype, swap.map { upType.dims[it] })
        val dwtType = DxirType(outType.dtype, swap.map { outType.dims[it] })
        val xT = synth()
        val upT = synth()
        val dwt = synth()
        out.appendLine(
            "$step$xT = stablehlo.transpose $xRef, dims = [1, 0, 2, 3] " +
                ": (${xType.toMlir()}) -> ${xTType.toMlir()}",
        )
        out.appendLine(
            "$step$upT = stablehlo.transpose $upRef, dims = [1, 0, 2, 3] " +
                ": (${upType.toMlir()}) -> ${upTType.toMlir()}",
        )
        emitConvolution(
            step, dwt, xT, upT,
            lhsType = xTType, rhsType = upTType, outType = dwtType,
            kernelLayout = "[o, i, 0, 1]",
            strides = rhsDil,
            padding = padding,
            lhsDilation = listOf(1, 1),
            rhsDilation = strides,
            reversal = null,
        )
        out.appendLine(
            "$step$name = stablehlo.transpose $dwt, dims = [1, 0, 2, 3] " +
                ": (${dwtType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    /**
     * §0.4.393 — the fused TRANSPOSED-conv adjoints (`CONV_TRANSPOSE2D_DATA_ADJOINT`
     * / `_KERNEL_ADJOINT`).
     *
     * The interpreter and host twins invert the primal's tap equation per element,
     * which StableHLO has no primitive for, so emission goes through the identity
     * that a transposed conv IS a forward conv of the dilated input:
     *
     *     convT(x, w; s, L, d, p) ≡ conv(dilate(x, L), swap01(w); s, d, p)
     *
     * which reduces both adjoints to pieces this emitter already has:
     * - `dX` = strided-slice( [emitDataAdjointConvolution] of `dy` against
     *   `swap01(w)`, stride `L` ). The data adjoint lands on the DILATED shape, and
     *   every L-th element of an interior-dilated tensor is the original value, so a
     *   strided slice undoes the dilation — the intervening positions are the
     *   inserted zeros, which the data gradient only ever contributes to spuriously.
     * - `dW` = `swap01`( [emitKernelAdjointExpansion] over `dilate(x, L)` and `dy` ).
     *   The kernel adjoint produces the gradient w.r.t. the FORWARD conv's OIHW
     *   kernel, so one channel swap maps it back to the IOHW layout `convT` uses.
     *
     * `window_reversal` is REJECTED here. With a reversed primal the data side would
     * need `!r` and the kernel side a compensating flip, and that identity is
     * unverified; nothing user-reachable sets it (`convTranspose2d`'s FIR arm never
     * emits the attr, and the host sugar passes `false`), while the interpreter and
     * host twins handle any reversal. Failing loudly beats shipping a
     * plausible-looking wrong kernel — the §0.4.389 maxpool emitter note is the same
     * call.
     */
    private fun emitConvTransposeAdjoint(step: String, name: String, ops: List<String>, node: DxirOp) {
        val dataAdj = node.op == OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT
        require(node.operands.size == 3) {
            "${node.op} takes (upstream, kernel, xTemplate) or (x, upstream, wTemplate); " +
                "got ${node.operands.size} operands"
        }
        val upType = if (dataAdj) node.operands[0].type else node.operands[1].type
        val otherType = if (dataAdj) node.operands[1].type else node.operands[0].type
        val upRef = if (dataAdj) ops[0] else ops[1]
        val otherRef = if (dataAdj) ops[1] else ops[0]
        val outType = node.type
        require(upType.rank == 4 && otherType.rank == 4 && outType.rank == 4) {
            "${node.op} requires rank-4 operands and result; got " +
                "${upType.dims} / ${otherType.dims} / ${outType.dims}"
        }
        val s = intListAttr(node, "window_strides")
        require(s.size == 2) { "${node.op} window_strides must be length 2; got $s" }
        val lhsDil = conv2dDilation(node, "lhs_dilation")
        val rhsDil = conv2dDilation(node, "rhs_dilation")
        val primalPad = conv2dPadding(node)
        val rev = (node.attrs["window_reversal"] as? List<*>)?.map { it as Boolean }
        require(rev == null || rev.none { it }) {
            "${node.op}: StableHLO emission does not support window_reversal $rev — the " +
                "convolution identity it is built on would need !r on the data side and a " +
                "compensating flip on the kernel side, which is unverified. The interpreter " +
                "and host twins handle any reversal."
        }

        // x's extents: the data adjoint's RESULT is x; for the kernel adjoint x is
        // operand 0. The IOHW kernel is operand 1 for dX and the result for dW.
        val xType = if (dataAdj) outType else otherType
        val wType = if (dataAdj) otherType else outType
        val hDil = (xType.dims[2] - 1) * lhsDil[0] + 1
        val wDil = (xType.dims[3] - 1) * lhsDil[1] + 1
        val dilType = DxirType(outType.dtype, listOf(xType.dims[0], xType.dims[1], hDil, wDil))
        // swap01 of the IOHW kernel: [Ci, Co, kh, kw] → OIHW [Co, Ci, kh, kw].
        val swappedType = DxirType(
            outType.dtype,
            listOf(wType.dims[1], wType.dims[0], wType.dims[2], wType.dims[3]),
        )

        if (dataAdj) {
            val wSwapped = synth()
            out.appendLine(
                "$step$wSwapped = stablehlo.transpose $otherRef, dims = [1, 0, 2, 3] " +
                    ": (${wType.toMlir()}) -> ${swappedType.toMlir()}",
            )
            val conv = synth()
            emitDataAdjointConvolution(
                step, conv, upRef, wSwapped, upType, swappedType, dilType, s, rhsDil, primalPad,
            )
            // Undo the dilation: take every lhsDil-th element of each spatial axis.
            val dims = dilType.dims
            out.appendLine(
                "$step$name = stablehlo.slice $conv [" +
                    dims.indices.joinToString(", ") { i ->
                        val stride = if (i >= 2) lhsDil[i - 2] else 1
                        if (stride == 1) "0:${dims[i]}" else "0:${dims[i]}:$stride"
                    } +
                    "] : (${dilType.toMlir()}) -> ${outType.toMlir()}",
            )
            return
        }

        val xDil = synth()
        emitPadLine(
            step, xDil, otherRef, otherType, dilType,
            low = List(4) { 0 },
            high = List(4) { 0 },
            interior = listOf(0, 0, lhsDil[0] - 1, lhsDil[1] - 1),
        )
        val dk = synth()
        emitKernelAdjointExpansion(
            step, dk, xDil, upRef, dilType, upType, swappedType, s, rhsDil, primalPad,
        )
        out.appendLine(
            "$step$name = stablehlo.transpose $dk, dims = [1, 0, 2, 3] " +
                ": (${swappedType.toMlir()}) -> ${outType.toMlir()}",
        )
    }

    /**
     * §0.4.386 — the fused avgpool adjoint, expanded into exactly the MLIR the
     * pre-fusion rule produced: fold channels into the batch dim, one lhs-dilated
     * `stablehlo.convolution` against a uniform `1/(kh·kw)` splat kernel (IOHW
     * `[1,1,kh,kw]`, so a single-channel kernel applies depthwise without
     * grouped-conv support), then fold back.
     *
     * Emit time is where the padding gets solved — the dims are concrete here,
     * unlike at transform time under `grad {}` — with the same formula the
     * interpreter and the host twin apply at runtime:
     *
     *     dilSize = (hOut−1)·s + 1,   low = kh − 1 − p_low,   high = p_low + H − dilSize
     *
     * Keeping this expansion rather than a `feature_group_count = C` depthwise
     * convolution means the module uses only patterns §0.4.362/§0.4.363 already
     * certified against real XLA. The value convention matches the interpreter's
     * count_include_pad: the splat carries `1/(kh·kw)` for the FULL window.
     */
    private fun emitAvgPoolGrad(step: String, name: String, up: String, node: DxirOp) {
        val upType = node.operands[0].type
        val outType = node.type
        require(upType.rank == 4 && outType.rank == 4) {
            "${node.op} requires rank-4 NCHW upstream/result; got ${upType.dims} / ${outType.dims}"
        }
        val window = intListAttr(node, "window")
        require(window.size == 2) { "${node.op} needs `window` [kh, kw]; got $window" }
        val strides = intListAttr(node, "window_strides").let { if (it.size == 2) it else window }
        val primalPad = conv2dPadding(node)
        val (nB, c, h, w) = outType.dims
        val hOut = upType.dims[2]
        val wOut = upType.dims[3]
        val padding = (0..1).map { a ->
            val dilSize = (upType.dims[2 + a] - 1) * strides[a] + 1
            val low = window[a] - 1 - primalPad[a][0]
            listOf(low, primalPad[a][0] + outType.dims[2 + a] - dilSize)
        }

        val foldedT = DxirType(outType.dtype, listOf(nB * c, 1, hOut, wOut))
        val kernelT = DxirType(outType.dtype, listOf(1, 1, window[0], window[1]))
        val convT = DxirType(outType.dtype, listOf(nB * c, 1, h, w))
        val folded = synth()
        val kernel = synth()
        val conv = synth()
        emitReshape(step, folded, up, upType, foldedT)
        val scale = mlirFloatLiteral((1.0f / (window[0] * window[1])).toString())
        out.appendLine("$step$kernel = stablehlo.constant dense<$scale> : ${kernelT.toMlir()}")
        emitConvolution(
            step, conv, folded, kernel,
            lhsType = foldedT, rhsType = kernelT, outType = convT,
            kernelLayout = "[i, o, 0, 1]",
            strides = listOf(1, 1),
            padding = padding,
            lhsDilation = strides,
            rhsDilation = listOf(1, 1),
            reversal = null,
        )
        emitReshape(step, name, conv, convT, outType)
    }

    /**
     * §0.4.389 — the fused maxpool adjoint, expanded into the upsample-and-mask MLIR
     * the pre-fusion rule produced (and §0.4.363 certified on the GB10):
     * nearest-upsample both the pooled value and the upstream back to x's shape via
     * `[N,C,Ho,Wo] → [N,C,Ho,1,Wo,1] → broadcast → [N,C,Ho,kh,Wo,kw] → [N,C,H,W]`,
     * then `select(x == U(y), U(dY), 0)`. Row-major flattening makes that
     * reshape/broadcast/reshape exactly per-window replication.
     *
     * Which is why it needs exact tiling — `strides == window`, zero padding, and
     * `H == Ho·kh` / `W == Wo·kw` — and why those are checked HERE rather than in
     * MaxPool2dRule: they are extent facts, and emit time is the first moment the
     * extents exist. Under `grad {}` they are -1 sentinels, which is what made the
     * rule's old `h % k[0] == 0` guard reject every symbolic maxpool. The host and
     * interpreter invert the window instead and handle a remainder correctly (inputs
     * the truncated last window never covered get no gradient), so a general-stride
     * or padded maxpool gradient works there and fails loudly here.
     *
     * Deliberately NOT `stablehlo.select_and_scatter`, the canonical single-op
     * spelling: it picks ONE winner per window, while every other engine routes the
     * full upstream to every within-window tie. Exact ties are common after a relu,
     * and the backends must not disagree.
     */
    private fun emitMaxPoolGrad(step: String, name: String, ops: List<String>, node: DxirOp) {
        require(node.operands.size == 3) {
            "${node.op} takes (upstream, x, y); got ${node.operands.size} operands"
        }
        val upType = node.operands[0].type
        val xType = node.operands[1].type
        val yType = node.operands[2].type
        val outType = node.type
        require(upType.rank == 4 && xType.rank == 4 && outType.rank == 4) {
            "${node.op} requires rank-4 NCHW operands/result; got " +
                "${upType.dims} / ${xType.dims} / ${outType.dims}"
        }
        val window = intListAttr(node, "window")
        require(window.size == 2) { "${node.op} needs `window` [kh, kw]; got $window" }
        val strides = intListAttr(node, "window_strides").let { if (it.size == 2) it else window }
        val padding = conv2dPadding(node)
        require(strides == window && padding.all { it == listOf(0, 0) }) {
            "${node.op}: StableHLO emission needs the non-overlapping zero-padding pool — the " +
                "upsample expansion cannot tile otherwise; got window=$window strides=$strides " +
                "padding=$padding"
        }
        val (nB, c, h, w) = outType.dims
        val hOut = upType.dims[2]
        val wOut = upType.dims[3]
        require(h == hOut * window[0] && w == wOut * window[1]) {
            "${node.op}: StableHLO emission needs window-divisible spatial dims " +
                "([$h, $w] vs [$hOut, $wOut] × $window); the host and interpreter paths handle " +
                "a remainder, this expansion cannot"
        }

        val narrow = DxirType(outType.dtype, listOf(nB, c, hOut, 1, wOut, 1))
        val wide = DxirType(outType.dtype, listOf(nB, c, hOut, window[0], wOut, window[1]))
        val boolT = DxirType(Bool, outType.dims)
        val dims6 = (0 until 6).joinToString(", ")

        fun upsample(src: String, srcType: DxirType): String {
            val r6 = synth()
            val b6 = synth()
            val back = synth()
            emitReshape(step, r6, src, srcType, narrow)
            out.appendLine(
                "$step$b6 = stablehlo.broadcast_in_dim $r6, dims = [$dims6] " +
                    ": (${narrow.toMlir()}) -> ${wide.toMlir()}",
            )
            emitReshape(step, back, b6, wide, outType)
            return back
        }

        val uy = upsample(ops[2], yType)
        val uup = upsample(ops[0], upType)
        val mask = synth()
        val zero = synth()
        // Both compare operands are at x's shape: `uy` is the UPSAMPLED pooled
        // value, so its type is `outType`, not the pooled `yType` it came from.
        out.appendLine(
            "$step$mask = stablehlo.compare  EQ, ${ops[1]}, $uy,  FLOAT : " +
                "(${xType.toMlir()}, ${outType.toMlir()}) -> ${boolT.toMlir()}",
        )
        out.appendLine("$step$zero = stablehlo.constant dense<0.0> : ${outType.toMlir()}")
        out.appendLine(
            "$step$name = stablehlo.select $mask, $uup, $zero : " +
                "${boolT.toMlir()}, ${outType.toMlir()}",
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
        // §0.4.114 — substrate-shape detection. The autograd-emitted SCATTER (§0.4.41,
        // §0.4.114) carries scalar I32 idx + no attrs and means `arr[idx] = v` (rank-1)
        // or `arr[idx, :] = row` (rank-2). Synthesize the canonical stablehlo.scatter
        // attrs for those shapes; the general attr-driven path handles everything else.
        val isSubstrateShape = "scatter_dims_to_operand_dims" !in node.attrs &&
            indicesType.isScalar &&
            indicesType.dtype == I32
        if (isSubstrateShape) {
            emitSubstrateScatter(
                step, name, operand, scatterIndices, updates,
                operandType = operandType,
                indicesType = indicesType,
                updatesType = updatesType,
                outType = node.type,
            )
            return
        }

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
        require(baseType.rank >= 1) {
            "SCATTER_ADD base must be rank ≥ 1 (substrate shape); got rank=${baseType.rank}"
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
        // `update_window_dims` indexes into UPDATES' axes (not operand's). For rank-r
        // base, updates is rank-(r-1) (all dims except axis 0 of operand), so the
        // window dims are 0..r-2 of updates — those map to operand axes 1..r-1
        // (since operand axis 0 is inserted via the index). For rank-1 base, updates
        // is rank-0 (scalar) and there are no window dims at all.
        val updateWindowDims = (0 until expectedValueRank).toList()
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

        val cur = synth(); val upd = synth()
        out.appendLine(
            """$step$name = "stablehlo.scatter"($base, $idx, $value) <{scatter_dimension_numbers = $dimNumbers, unique_indices = true}> ({""",
        )
        out.appendLine("$step ^bb0($cur: $scalarT, $upd: $scalarT):")
        if (isZeroBroadcastBase(node.operands[0])) {
            // §0.4.133 — scatter-into-zeros peephole. SCATTER_ADD with a base of
            // BROADCAST(const(0), …) (the canonical shape produced by [GatherRule]'s
            // adjoint) is structurally equivalent to a replace-body scatter, since
            // 0 + x = x. Emit `return upd` directly so the lowered MLIR matches the
            // SCATTER (replace) substrate's body — gives XLA's optimiser a head start
            // and removes a dead `cur` SSA value from the inner block.
            out.appendLine("$step   stablehlo.return $upd : $scalarT")
        } else {
            val sum = synth()
            out.appendLine("$step   $sum = stablehlo.add $cur, $upd : $scalarT")
            out.appendLine("$step   stablehlo.return $sum : $scalarT")
        }
        out.appendLine(
            "$step }) : (${baseType.toMlir()}, ${idxType.toMlir()}, ${valueType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /**
     * §0.4.133 — recognise the scatter-into-zeros base pattern: `BROADCAST(const(0))`
     * (the canonical shape [GatherRule] emits for the gradient of GATHER) or a
     * literal rank-N const tensor where every element is zero. Returns true if the
     * operand is a compile-time-known zero tensor.
     *
     * This is intentionally narrow — only the BROADCAST-of-scalar-zero shape and
     * the FloatArray-of-zeros shape match. More elaborate "is zero" detection
     * (e.g., zero literal arithmetic, transitive zero propagation) is out of
     * scope for this emitter peephole.
     */
    private fun isZeroBroadcastBase(node: io.tlaloc.ir.DxirNode): Boolean {
        if (node !is DxirOp) return false
        if (node.op != OpKind.BROADCAST) return false
        if (node.operands.size != 1) return false
        val operand = node.operands[0]
        if (operand !is io.tlaloc.ir.DxirConst) return false
        return when (val v = operand.value) {
            is Number -> v.toDouble() == 0.0
            is FloatArray -> v.all { it == 0f }
            else -> false
        }
    }

    /**
     * §0.4.114 — lowering for the autograd-emitted [OpKind.SCATTER] substrate shape
     * (`base[idx] = value`, no attrs). Two operand-rank slices are supported,
     * matching the [DxirInterpreter] arms shipped in §0.4.41 + §0.4.114:
     *
     *  - rank-1 base + scalar idx + scalar value → rank-1 result.
     *  - rank-2 base + scalar idx + rank-1 [N] value → rank-2 [M, N] result (replaces
     *    row [idx]).
     *
     * Same dimension-numbers shape as [emitScatterAdd]; the only difference is the
     * body computation: SCATTER returns `upd` directly (replace semantics), whereas
     * SCATTER_ADD adds `cur + upd`.
     */
    private fun emitSubstrateScatter(
        step: String,
        name: String,
        base: String,
        idx: String,
        value: String,
        operandType: DxirType,
        indicesType: DxirType,
        updatesType: DxirType,
        outType: DxirType,
    ) {
        require(operandType.rank >= 1) {
            "substrate SCATTER base must be rank ≥ 1; got rank=${operandType.rank}"
        }
        require(indicesType.isScalar && indicesType.dtype == I32) {
            "substrate SCATTER idx must be scalar I32; got $indicesType"
        }
        val expectedValueRank = operandType.rank - 1
        require(updatesType.rank == expectedValueRank) {
            "substrate SCATTER value must be rank-$expectedValueRank for rank-${operandType.rank} base; got rank=${updatesType.rank}"
        }
        require(outType.dims == operandType.dims) {
            "substrate SCATTER result shape ${outType.dims} must match base shape ${operandType.dims}"
        }

        val scalarT = "tensor<${mlirElementType(operandType.dtype)}>"

        // §0.4.132 — generalised to any rank ≥ 1. `update_window_dims` covers all
        // axes of `updates` (rank r-1), since the indexed axis 0 of operand is
        // inserted via the scalar index and not present in updates.
        val updateWindowDims = (0 until expectedValueRank).toList()
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

        val cur = synth(); val upd = synth()
        out.appendLine(
            """$step$name = "stablehlo.scatter"($base, $idx, $value) <{scatter_dimension_numbers = $dimNumbers, unique_indices = true}> ({""",
        )
        out.appendLine("$step ^bb0($cur: $scalarT, $upd: $scalarT):")
        // Replace semantics: return the update value, ignoring the current value.
        out.appendLine("$step   stablehlo.return $upd : $scalarT")
        out.appendLine(
            "$step }) : (${operandType.toMlir()}, ${indicesType.toMlir()}, ${updatesType.toMlir()}) -> ${outType.toMlir()}",
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
     * Layer 4.1 §0.4.261 — emit `stablehlo.custom_call` for a COARSENED op
     * carrying a [KernelDescriptor]. The descriptor's `kernelName` becomes
     * the call_target_name (`@<kernelName>`); `customCallAttrs` is encoded
     * into the `backend_config` string (deterministic alphabetic key order
     * so tests can pin the exact emitted text).
     *
     * Layer 4.2 §0.4.262 — when [node].sharding is non-null, attach
     * `sdy.sharding = #sdy.sharding_per_value<[<...>]>` so the SDY
     * propagation pass can carry shardings *across* the kernel boundary.
     * Custom calls are opaque to propagation — without an explicit
     * op-level sharding, propagation stops at the kernel and the result
     * stays unsharded. The `per_value` form handles single-result and
     * multi-result uniformly (multi-result COARSENED is reserved for
     * future kernel shapes; today's L3 emits only single-result).
     */
    private fun emitCustomCall(
        step: String,
        name: String,
        operandNames: List<String>,
        node: DxirOp,
        descriptor: KernelDescriptor,
    ) {
        val operandList = operandNames.joinToString(", ")
        val operandTypes = node.operands.joinToString(", ") { it.type.toMlir() }
        // §0.4.351 — scratch results (multi-stage launch-chain intermediates,
        // see KernelDescriptor.scratchResults) are appended after the op's
        // own results. They are XLA-owned and referenced by nothing: the
        // node's value stays result #0, pre-registered in the SSA map so
        // downstream ops print `%name#0`.
        val scratchTypes = descriptor.scratchResults.map { dims ->
            if (dims.isEmpty()) "tensor<f32>" else "tensor<${dims.joinToString("x")}xf32>"
        }
        val allResultTypes = node.types.map { it.toMlir() } + scratchTypes
        val resultTypeMlir = if (allResultTypes.size > 1) {
            "(${allResultTypes.joinToString(", ")})"
        } else {
            allResultTypes.single()
        }
        val lhs = if (allResultTypes.size > 1) "$name:${allResultTypes.size}" else name
        if (allResultTypes.size > 1) {
            ssa[node.id] = List(node.numResults) { i -> "$name#$i" }
        }
        val attrs = mutableListOf<String>()
        if (descriptor.typedFfi) {
            // KPTX v1.6 §0.4.332 — typed-FFI convention: api_version 4 with
            // attrs as a `backend_config` *dictionary* attribute (StableHLO's
            // canonical form for API_VERSION_TYPED_FFI; omitted when empty).
            // XLA converts the dict into XLA_FFI_Attrs delivered to the
            // registered handler's call frame. Note: NOT the JAX-lowering
            // `mhlo.backend_config` discardable-attr spelling — the plugin's
            // StableHLO import reads the dict from the op's own attr, and
            // attrs sent via `mhlo.backend_config` arrive empty (verified
            // against jaxlib 0.10.0 in KptxTypedFfiEmitDispatchTest).
            attrs += "api_version = 4 : i32"
            if (descriptor.customCallAttrs.isNotEmpty()) {
                attrs += "backend_config = {${encodeTypedFfiConfig(descriptor.customCallAttrs)}}"
            }
        } else {
            attrs += "backend_config = \"${encodeBackendConfig(descriptor.customCallAttrs)}\""
        }
        attrs += "has_side_effect = false"
        node.sharding?.let { sharding ->
            attrs += "sdy.sharding = #sdy.sharding_per_value<[${sharding.toSdyAttr()}]>"
        }
        out.appendLine(
            "$step$lhs = stablehlo.custom_call @${descriptor.kernelName}($operandList) " +
                "{${attrs.joinToString(", ")}} : " +
                "($operandTypes) -> $resultTypeMlir",
        )
    }

    /**
     * Encode [customCallAttrs] into the `backend_config` string body.
     * Empty map produces an empty string. Keys are sorted alphabetically
     * for deterministic emit. Strings are emitted bare (the v1 attrs in
     * [KernelDescriptor] are dtype tags like `f32`/`bf16`, not arbitrary
     * text); lists use `[v1, v2, ...]`; numbers and booleans are bare.
     */
    private fun encodeBackendConfig(customCallAttrs: Map<String, Any>): String {
        if (customCallAttrs.isEmpty()) return ""
        return customCallAttrs.entries
            .sortedBy { it.key }
            .joinToString(", ", "{", "}") { (k, v) -> "$k = ${encodeAttrValue(v)}" }
    }

    private fun encodeAttrValue(v: Any): String = when (v) {
        is List<*> -> v.joinToString(", ", "[", "]") { encodeAttrValue(it!!) }
        is Boolean, is Number, is String -> v.toString()
        else -> error("unsupported customCallAttrs value type: ${v::class.simpleName}")
    }

    /**
     * KPTX v1.6 §0.4.332 — encode [customCallAttrs] as the *body* of an
     * `mhlo.backend_config` dictionary attribute for typed-FFI custom
     * calls. Unlike [encodeBackendConfig]'s free-form string, dictionary
     * values are typed MLIR attribute literals: ints are `: i64`, floats
     * `: f32` (doubles `: f64`), strings quoted, booleans bare. XLA turns
     * these into `XLA_FFI_Attrs` scalars/strings on the handler's call
     * frame (decoded Kotlin-side by XlaFfi in :runtime-pjrt). Keys sorted
     * alphabetically for deterministic emit, matching §0.4.261.
     */
    private fun encodeTypedFfiConfig(customCallAttrs: Map<String, Any>): String =
        customCallAttrs.entries
            .sortedBy { it.key }
            .joinToString(", ") { (k, v) -> "$k = ${encodeTypedFfiValue(v)}" }

    private fun encodeTypedFfiValue(v: Any): String = when (v) {
        is Boolean -> v.toString()
        is Int, is Long -> "$v : i64"
        is Float -> "${mlirFloatLiteral(v.toString())} : f32"
        is Double -> "${mlirFloatLiteral(v.toString())} : f64"
        is String -> "\"$v\""
        is List<*> -> v.joinToString(", ", "[", "]") { encodeTypedFfiValue(it!!) }
        else -> error("unsupported typed-FFI customCallAttrs value type: ${v::class.simpleName}")
    }

    /** Kotlin renders exponents as `1.0E-5`; MLIR float literals use a
     * lowercase `e`. Also guarantees a decimal point for whole values. */
    private fun mlirFloatLiteral(s: String): String {
        val lower = s.lowercase()
        return if (lower.contains('.') || lower.contains('e')) lower else "$lower.0"
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
        // §0.4.113 — substrate-shape detection. The autograd-emitted GATHER (§0.4.41,
        // §0.4.111) carries scalar I32 idx + no attrs and means `arr[idx]` (rank-1) or
        // `arr[idx, :]` (rank-2). Synthesize the canonical stablehlo.gather attrs for
        // those shapes; the general attr-driven path handles everything else.
        val isSubstrateShape = "offset_dims" !in node.attrs &&
            indicesType.isScalar &&
            indicesType.dtype == I32
        if (isSubstrateShape) {
            emitSubstrateGather(
                step, name, operand, startIndices,
                operandType = operandType,
                indicesType = indicesType,
                outType = node.type,
            )
            return
        }

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

    /**
     * §0.4.113 — lower the autograd-emitted [OpKind.GATHER] substrate shape (scalar
     * I32 idx, no attrs) to `stablehlo.gather`. Two operand-rank slices are supported,
     * matching the [DxirInterpreter] arms shipped in §0.4.41 + §0.4.111:
     *
     *  - rank-1 operand (`arr: tensor<NxF>`) → scalar result. Equivalent to `arr[idx]`.
     *    `offset_dims = []`, `collapsed_slice_dims = [0]`, `slice_sizes = [1]`.
     *  - rank-2 operand (`arr: tensor<MxNxF>`) → rank-1 result of length N.
     *    Equivalent to `arr[idx, :]`. `offset_dims = [0]` (the single offset axis is
     *    axis 0 of the rank-1 result), `collapsed_slice_dims = [0]`, `slice_sizes = [1, N]`.
     *
     * For both, `start_index_map = [0]` and `index_vector_dim = 0` (with rank-0
     * `scatter_indices`, StableHLO's spec implicitly expands by a trailing-1 dim,
     * so the effective indices rank becomes 1 — consistent with the rank arithmetic
     * used by [emitScatterAdd] in §0.4.112).
     *
     * The general [emitGatherOp] path is reused; this method only synthesizes the
     * canonical attrs for the substrate.
     */
    private fun emitSubstrateGather(
        step: String,
        name: String,
        operand: String,
        startIndices: String,
        operandType: DxirType,
        indicesType: DxirType,
        outType: DxirType,
    ) {
        // §0.4.113 / §0.4.132 — substrate-shape GATHER for any rank ≥ 1: scalar I32
        // index selects a slice along axis 0. Output is rank-(r-1) with shape
        // `operandType.dims.drop(1)`. The canonical stablehlo.gather attrs are:
        //   - offset_dims = (0 until r-1)            (all output axes are offsets)
        //   - collapsed_slice_dims = [0]             (axis 0 is collapsed by indexing)
        //   - start_index_map = [0]                  (single index targets axis 0)
        //   - index_vector_dim = 0                   (scalar index)
        //   - slice_sizes = [1, dim_1, …, dim_{r-1}] (window of size 1 on the
        //     indexed axis, full extent on the rest)
        require(operandType.rank >= 1) {
            "substrate GATHER operand must be rank ≥ 1; got rank=${operandType.rank}"
        }
        val expectedOutDims = operandType.dims.drop(1)
        require(outType.dims == expectedOutDims) {
            "substrate GATHER output shape ${outType.dims} doesn't match expected " +
                "$expectedOutDims for rank-${operandType.rank} operand ${operandType.dims}"
        }

        val offsetDims = (0 until operandType.rank - 1).toList()
        val sliceSizes = listOf(1) + operandType.dims.drop(1)

        emitGatherOp(
            step = step,
            name = name,
            operand = operand,
            startIndices = startIndices,
            operandType = operandType,
            indicesType = indicesType,
            outType = outType,
            offsetDims = offsetDims,
            collapsedSliceDims = listOf(0),
            startIndexMap = listOf(0),
            indexVectorDim = 0,
            sliceSizes = sliceSizes,
            indicesAreSorted = false,
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

    /**
     * §0.4.400 — EMBEDDING's fused scatter-add adjoint, deferred by §0.4.370 and
     * closed here: `stablehlo.scatter` with an ADD computation region over a
     * splat-zero `[V, D]` base. Rank-1 `[N]` indices with
     * `index_vector_dim = 1` (== indices rank, the implicit trailing-1 form the
     * SCATTER_ADD arm already uses for its scalar index); each `[N, D]` upstream
     * row is a window over operand axis 1 (`update_window_dims = [1]`) landing
     * at the vocab slot its index selects (`inserted_window_dims = [0]`,
     * `scatter_dims_to_operand_dims = [0]`). Collisions are the POINT — the same
     * vocab row embedded at several positions must accumulate — so
     * `unique_indices` is left unset and the region body is a real add, never
     * the §0.4.133 return-upd peephole (which is only sound for unique indices).
     */
    private fun emitEmbeddingGrad(
        step: String,
        name: String,
        indices: String,
        upstream: String,
        node: DxirOp,
        indicesType: DxirType,
        upstreamType: DxirType,
    ) {
        val outType = node.type
        require(outType.rank == 2) {
            "EMBEDDING_GRAD result must be rank-2 (vocab, dim); got ${outType.dims}"
        }
        require(indicesType.dtype is I32 || indicesType.dtype is I64) {
            "EMBEDDING_GRAD indices must be integer (I32 or I64); got ${indicesType.dtype}"
        }
        require(indicesType.rank == 1) {
            "EMBEDDING_GRAD indices must be rank-1; got ${indicesType.dims}"
        }
        val embedDim = outType.dims[1]
        require(upstreamType.dims == listOf(indicesType.dims[0], embedDim)) {
            "EMBEDDING_GRAD upstream shape ${upstreamType.dims} does not match expected " +
                "[${indicesType.dims[0]}, $embedDim] (indices ${indicesType.dims} ++ [$embedDim])"
        }

        val zeros = synth()
        out.appendLine("$step$zeros = stablehlo.constant dense<0.0> : ${outType.toMlir()}")

        val scalarT = "tensor<${mlirElementType(outType.dtype)}>"
        val dimNumbers = "#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], " +
            "scatter_dims_to_operand_dims = [0], index_vector_dim = 1>"
        val cur = synth(); val upd = synth(); val sum = synth()
        out.appendLine(
            """$step$name = "stablehlo.scatter"($zeros, $indices, $upstream) <{scatter_dimension_numbers = $dimNumbers}> ({""",
        )
        out.appendLine("$step ^bb0($cur: $scalarT, $upd: $scalarT):")
        out.appendLine("$step   $sum = stablehlo.add $cur, $upd : $scalarT")
        out.appendLine("$step   stablehlo.return $sum : $scalarT")
        out.appendLine(
            "$step }) : (${outType.toMlir()}, ${indicesType.toMlir()}, ${upstreamType.toMlir()}) -> ${outType.toMlir()}",
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

    /**
     * §0.4.396 — REVERSE → `stablehlo.reverse %x, dims = […]` (Phase C3). The
     * `dimensions` attr carries the user's literal axis positions; the op is
     * shape-preserving, so operand and result types coincide and the pretty
     * form's functional type spells both.
     */
    private fun emitReverse(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
        inputType: DxirType,
    ) {
        val axes = intListAttr(node, "dimensions")
        require(axes.isNotEmpty()) { "REVERSE requires at least one axis in 'dimensions'" }
        require(axes.toSet().size == axes.size) { "REVERSE axes $axes must be distinct" }
        for (ax in axes) {
            require(ax in 0 until inputType.rank) {
                "REVERSE axis $ax out of range for rank ${inputType.rank}"
            }
        }
        require(node.type.dims == inputType.dims) {
            "REVERSE is shape-preserving; got ${inputType.dims} -> ${node.type.dims}"
        }
        out.appendLine(
            "$step$name = stablehlo.reverse $x, dims = [${axes.joinToString(", ")}] " +
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
        // §0.4.393 — honour §0.4.359's polymorphic EMPTY form, which the interpreter
        // implements and whose comment already claimed this function matched it. It
        // did not: empty dims + equal rank + non-scalar input is the reduction
        // adjoints' "un-reduce stretch" ([N,1] → [N,K], emitted by the Max/Min/Tanh/
        // Softmax rules via `broadcastTo`), and the length check below rejected it,
        // so none of those gradients could be emitted at all. Mirroring the
        // interpreter's rule exactly — identity mapping in that case, empty
        // otherwise — keeps the two engines in agreement instead of one silently
        // supporting a form the other refuses.
        val inDims = inputType.dims
        val outDims = node.type.dims
        val inSize = if (inDims.isEmpty()) 1 else inDims.reduce(Int::times)
        val effectiveDims = if (bcastDims.isEmpty() && inSize != 1 && inDims.size == outDims.size) {
            inDims.indices.toList()
        } else {
            bcastDims
        }
        require(effectiveDims.size == inputType.rank) {
            "BROADCAST broadcast_dimensions length ${effectiveDims.size} must equal input rank " +
                "${inputType.rank} (shape ${inputType.dims} -> ${node.type.dims}); the empty form " +
                "requires a scalar input or an equal-rank keepdims input"
        }
        // §0.4.283 — StableHLO requires each broadcast_dimensions entry
        // in [0, output rank) and the entries to be unique. Without these
        // guards a malformed dxir silently emits invalid MLIR.
        val outputRank = node.type.rank
        effectiveDims.forEach {
            require(it in 0 until outputRank) {
                "BROADCAST broadcast_dimensions entry $it out of range [0, $outputRank); got $effectiveDims"
            }
        }
        require(effectiveDims.toSet().size == effectiveDims.size) {
            "BROADCAST broadcast_dimensions must be unique; got $effectiveDims"
        }
        out.appendLine(
            "$step$name = stablehlo.broadcast_in_dim $x, dims = [${effectiveDims.joinToString(", ")}] " +
                ": (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /** §0.4.360 — `stablehlo.select` over a Bool predicate tensor. */
    private fun emitWhere(step: String, name: String, ops: List<String>, node: DxirOp) {
        val predMlir = node.operands[0].type.toMlir()
        out.appendLine(
            "$step$name = stablehlo.select ${ops[0]}, ${ops[1]}, ${ops[2]} : $predMlir, ${node.type.toMlir()}",
        )
    }

    /** §0.4.360 — `stablehlo.compare` with the `direction` attr
     * (EQ/NE/LT/LE/GT/GE); FLOAT vs SIGNED comparison type from the
     * operand dtype (the emitStep spelling). Result is a Bool tensor. */
    private fun emitCompare(step: String, name: String, ops: List<String>, node: DxirOp) {
        val dir = node.attrs["direction"] as? String
            ?: error("COMPARE requires a `direction` attr (EQ/NE/LT/LE/GT/GE)")
        require(dir in setOf("EQ", "NE", "LT", "LE", "GT", "GE")) {
            "COMPARE: unknown direction `$dir`"
        }
        val inType = node.operands[0].type
        val cmpSuffix = when (inType.dtype) {
            is io.tlaloc.core.F32, is io.tlaloc.core.F64 -> "FLOAT"
            else -> "SIGNED"
        }
        out.appendLine(
            "$step$name = stablehlo.compare  $dir, ${ops[0]}, ${ops[1]},  $cmpSuffix : " +
                "(${inType.toMlir()}, ${node.operands[1].type.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /** §0.4.360 — `stablehlo.pad` with zero padding value, edge-only
     * (`low`/`high` List<Int> attrs; interior fixed at 0 in v1). */
    private fun emitPad(step: String, name: String, x: String, node: DxirOp, inputType: DxirType) {
        val low = intListAttr(node, "low")
        val high = intListAttr(node, "high")
        require(low.size == inputType.rank && high.size == inputType.rank) {
            "PAD attr lengths must match input rank ${inputType.rank}"
        }
        emitPadLine(step, name, x, inputType, node.type, low, high, List(inputType.rank) { 0 })
    }

    /**
     * One `stablehlo.pad`, with the syntax in a single place: [emitPad] (outer
     * padding only, interior fixed at 0) and §0.4.393's transposed-conv adjoints
     * (which need a genuine INTERIOR pad to dilate `x` by `lhs_dilation`) both go
     * through here.
     */
    private fun emitPadLine(
        step: String,
        name: String,
        x: String,
        inType: DxirType,
        outType: DxirType,
        low: List<Int>,
        high: List<Int>,
        interior: List<Int>,
    ) {
        val scalarMlir = "tensor<${mlirElementType(inType.dtype)}>"
        val zero = synth()
        out.appendLine("$step$zero = stablehlo.constant dense<0.0> : $scalarMlir")
        out.appendLine(
            "$step$name = stablehlo.pad $x, $zero, low = [${low.joinToString(", ")}], " +
                "high = [${high.joinToString(", ")}], interior = [${interior.joinToString(", ")}] : " +
                "(${inType.toMlir()}, $scalarMlir) -> ${outType.toMlir()}",
        )
    }

    // §0.4.374 — PAD_TO: zero-pad `value` (inputType) into the template shape
    // (node.type) at offset `low`; the trailing pad is derived from the concrete
    // template dims (`high[i] = node.type.dim[i] − low[i] − value.dim[i]`) — at
    // emit time dims are concrete, never sentinels.
    private fun emitPadTo(step: String, name: String, x: String, node: DxirOp, inputType: DxirType) {
        val low = intListAttr(node, "low")
        val rank = inputType.rank
        require(low.size == rank && node.type.rank == rank) {
            "PAD_TO attr/rank mismatch: low=${low.size}, value rank=$rank, template rank=${node.type.rank}"
        }
        val high = (0 until rank).map { i ->
            val h = node.type.dims[i] - low[i] - inputType.dims[i]
            require(h >= 0) {
                "PAD_TO axis $i: low ${low[i]} + value ${inputType.dims[i]} exceeds template ${node.type.dims[i]}"
            }
            h
        }
        val scalarMlir = "tensor<${mlirElementType(inputType.dtype)}>"
        val zero = synth()
        out.appendLine("$step$zero = stablehlo.constant dense<0.0> : $scalarMlir")
        out.appendLine(
            "$step$name = stablehlo.pad $x, $zero, low = [${low.joinToString(", ")}], " +
                "high = [${high.joinToString(", ")}], interior = [${List(rank) { 0 }.joinToString(", ")}] : " +
                "(${inputType.toMlir()}, $scalarMlir) -> ${node.type.toMlir()}",
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

    /**
     * Phase A2b — `SLICE_LIKE(value, thisTemplate, priorTemplate…)`, the CONCAT
     * adjoint: the window along attr `axis` starts at the sum of the prior
     * templates' axis extents and runs for `thisTemplate`'s, with every other axis
     * taken whole. Emit-time dims are always concrete, so both bounds fold to
     * literals and this is the same static `stablehlo.slice` [emitSlice] emits —
     * the templates exist for the host path, where the extents are only known at
     * runtime.
     */
    private fun emitSliceLike(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
    ) {
        val inputType = node.operands[0].type
        val thisType = node.operands[1].type
        val axis = intAttr(node, "axis")
        require(axis in inputType.dims.indices) {
            "SLICE_LIKE axis $axis outside the value's rank ${inputType.rank}"
        }
        require(thisType.rank == inputType.rank) {
            "SLICE_LIKE template rank ${thisType.rank} must equal the value's ${inputType.rank}"
        }
        val start = node.operands.drop(2).sumOf { it.type.dims[axis] }
        val len = thisType.dims[axis]
        require(start >= 0 && start + len <= inputType.dims[axis]) {
            "SLICE_LIKE window [$start, ${start + len}) exceeds the value's axis-$axis extent ${inputType.dims[axis]}"
        }
        out.appendLine(
            "$step$name = stablehlo.slice $x [" +
                inputType.dims.indices.joinToString(", ") { i ->
                    if (i == axis) "$start:${start + len}" else "0:${inputType.dims[i]}"
                } +
                "] : (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /**
     * §0.4.399 — `BROADCAST_LIKE(value, template)`, SUM_TO's forward twin and
     * VJP: broadcast `value` up to the template's shape under NumPy
     * right-alignment. Emit-time dims are always concrete, so it folds to a
     * static `stablehlo.broadcast_in_dim` with the identity right-aligned axis
     * map (value axis j → output axis `offset + j`) — the template's SSA value
     * goes unreferenced (it exists for the host path's runtime extents),
     * MLIR-legal and DCE'd downstream, the SLICE_LIKE precedent.
     */
    private fun emitBroadcastLike(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
    ) {
        val u = node.operands[0].type.dims
        val t = node.type.dims
        val ru = u.size
        val rt = t.size
        require(ru <= rt) { "BROADCAST_LIKE value rank $ru exceeds template rank $rt" }
        val offset = rt - ru
        for (i in 0 until ru) {
            require(u[i] == t[offset + i] || u[i] == 1) {
                "BROADCAST_LIKE value dim $i = ${u[i]} incompatible with template axis ${offset + i} = ${t[offset + i]}"
            }
        }
        val dims = (0 until ru).map { it + offset }
        out.appendLine(
            "$step$name = stablehlo.broadcast_in_dim $x, dims = [${dims.joinToString(", ")}] " +
                ": (${node.operands[0].type.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    /**
     * §0.4.399 — `SLICE_AT(value, template)` + attr `low`, PAD_TO's reverse
     * mirror and VJP: the window of the template's shape at literal offset
     * `low` per axis. Emit-time dims are always concrete, so the bounds fold to
     * literals and this is the same static `stablehlo.slice` [emitSlice] emits —
     * the template's SSA value goes unreferenced, as with [emitSliceLike].
     */
    private fun emitSliceAt(
        step: String,
        name: String,
        x: String,
        node: DxirOp,
    ) {
        val inputType = node.operands[0].type
        val low = intListAttr(node, "low")
        val rank = inputType.rank
        require(low.size == rank && node.type.rank == rank) {
            "SLICE_AT attr/rank mismatch: low=${low.size}, value rank=$rank, template rank=${node.type.rank}"
        }
        for (i in 0 until rank) {
            require(low[i] >= 0 && low[i] + node.type.dims[i] <= inputType.dims[i]) {
                "SLICE_AT axis $i: low ${low[i]} + template ${node.type.dims[i]} exceeds value ${inputType.dims[i]}"
            }
        }
        out.appendLine(
            "$step$name = stablehlo.slice $x [" +
                (0 until rank).joinToString(", ") { i ->
                    "${low[i]}:${low[i] + node.type.dims[i]}"
                } +
                "] : (${inputType.toMlir()}) -> ${node.type.toMlir()}",
        )
    }

    private fun synth(): String = "%s${nextSynth++}"
}
