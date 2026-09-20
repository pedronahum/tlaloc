package io.tlaloc.plugin

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.backend.common.lower.DeclarationIrBuilder
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.builders.IrBuilderWithScope
import org.jetbrains.kotlin.ir.builders.declarations.addValueParameter
import org.jetbrains.kotlin.ir.builders.declarations.buildFun
import org.jetbrains.kotlin.ir.builders.irBlockBody
import org.jetbrains.kotlin.ir.builders.irCall
import org.jetbrains.kotlin.ir.builders.irGet
import org.jetbrains.kotlin.ir.builders.irIfThenElse
import org.jetbrains.kotlin.ir.builders.irReturn
import org.jetbrains.kotlin.ir.builders.irTemporary
import org.jetbrains.kotlin.ir.declarations.IrDeclarationOrigin
import org.jetbrains.kotlin.ir.declarations.IrDeclarationParent
import org.jetbrains.kotlin.ir.declarations.IrParameterKind
import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.declarations.IrValueDeclaration
import org.jetbrains.kotlin.ir.declarations.IrValueParameter
import org.jetbrains.kotlin.ir.expressions.IrBlockBody
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrConstKind
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.IrFunctionExpression
import org.jetbrains.kotlin.ir.expressions.IrStatementOrigin
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstructorCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrFunctionExpressionImpl
import org.jetbrains.kotlin.ir.expressions.impl.fromSymbolOwner
import org.jetbrains.kotlin.ir.symbols.IrClassSymbol
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.UnsafeDuringIrConstructionAPI
import org.jetbrains.kotlin.ir.types.IrSimpleType
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.IrTypeArgument
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.impl.buildSimpleType
import org.jetbrains.kotlin.ir.types.typeOrNull
import org.jetbrains.kotlin.ir.types.typeWith
import org.jetbrains.kotlin.name.CallableId
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.name.SpecialNames

/**
 * Rebuilds a [DxirFunction]'s forward computation as a Kotlin IR lambda expression.
 *
 * Scope:
 * - Scalar primitives (Float / Double / Int / Long): each [DxirOp] maps to the matching
 *   `kotlin.{Float,Double,Int,Long}` arithmetic operator; [DxirConst]s become [IrConstImpl];
 *   [DxirParam]s become `irGet` of the matching lambda value parameter.
 * - Rank-1 `DTensor<Rank1<_>, F32>` — narrow widening landed in §0.4.10. Only exercised by
 *   `grad { x: DTensor<Rank1<_>, F32> -> x.sum() }` today: the gradient body is a scalar seed
 *   const + a `BROADCAST` op lowered as a call into `io.tlaloc.core.ops.broadcastLike`.
 *   The tensor IrType is harvested from the call site's [IrCall.type] (the function type's
 *   parameter arg) rather than rebuilt generically — generic erasure makes any phantom
 *   shape witness the same at runtime, so reusing the call-site IrType keeps
 *   [TlalocIrGenerationExtension]'s type-mismatch guard happy without re-deriving type args.
 *
 * Every intermediate (Const + Op) is materialised as a local `val` so the IR tree never
 * shares a node between parents — callers can reference the same DxirNode multiple times
 * (e.g. `x * x`) without running into IR node-sharing checks.
 *
 * Returns `null` (forcing the caller to leave the original call intact) if [fn] contains
 * anything outside this scope: rank > 1, unsupported op kinds, block args, nested regions,
 * multi-result ops, or a return that isn't one / two / three values.
 */
@OptIn(UnsafeDuringIrConstructionAPI::class)
internal class DxirToIrSynthesis(private val pluginContext: IrPluginContext) {

    /**
     * Per-[synthesise] state that threads call-site type context + the sole rank-1
     * template parameter (when one exists) through [irTypeFor] / [irOpFor] without
     * requiring every lowering arm to re-derive them. Scalar-only primals leave
     * [tensorIrType] and [tensorTemplateParam] null.
     *
     * §0.4.192 — Phase 0c-rectangular slice 1: [operandIrTypes] threads per-operand
     * `DxirNode.id → IrType` so future rank-2 ops with non-uniform shapes (rectangular
     * MATMUL: `Rank2<R, K> matmul Rank2<K, C>` produces `Rank2<R, C>` where R, K, C
     * differ) can resolve each operand's specific IrType. The map is empty by default
     * — when [tensorIrType] is set, callers fall back to it for any operand absent
     * from the map. Slice 2 will populate it from the call-site IrType arguments and
     * wire `irMatmul` / `irTranspose` to consult it instead of `tensorIrType`.
     */
    private data class SynthesisContext(
        val tensorIrType: IrType?,
        val tensorTemplateParam: IrValueParameter?,
        val operandIrTypes: Map<Int, IrType> = emptyMap(),
        // §0.4.197 — Phase 0c-rectangular slice 3b-2b: the function's tensor params
        // are needed by `irBroadcast`'s axis-matching path so it can synthesise
        // `param.dims[axis]` IR expressions when the BROADCAST target shape doesn't
        // match the single `tensorTemplateParam`'s shape (rectangular case).
        val fnParams: List<DxirParam> = emptyList(),
        val irParams: List<IrValueParameter> = emptyList(),
    )

    /**
     * §0.4.192 — Phase 0c-rectangular slice 1 helper. Returns the IrType for a
     * specific [DxirNode], preferring the per-operand map when present, falling back
     * to the call-site `tensorIrType` for any rank-1/2/3 F32 type, and `null` for
     * shapes outside the synthesis surface.
     *
     * Slice 1 doesn't populate the map yet (callers consistently fall back to
     * `tensorIrType`); behavior is bit-exact equivalent to the pre-§0.4.192 path.
     * Slice 2 wires the populator.
     */
    private fun irTypeForNode(node: io.tlaloc.ir.DxirNode, context: SynthesisContext): IrType? {
        context.operandIrTypes[node.id]?.let { return it }
        return irTypeFor(node.type, context)
    }

    /**
     * §0.4.194 — Phase 0c-rectangular slice 3a. Returns a fresh `IrSimpleType` shaped
     * like [original] but with [newArgs] substituted for `arguments`. Wraps Kotlin's
     * [buildSimpleType] DSL so callers don't need to reach into the impl package.
     * Used by [deriveResultIrType] to construct DTensor / Rank2 IrTypes with rearranged
     * shape arguments for TRANSPOSE / MATMUL outputs.
     */
    private fun reshapeIrSimpleType(original: IrSimpleType, newArgs: List<IrTypeArgument>): IrSimpleType =
        original.buildSimpleType { arguments = newArgs }

    /**
     * §0.4.194 — derive a `DxirOp`'s result IrType for the rectangular-shape ops:
     * [OpKind.TRANSPOSE] swaps the inner Rank2's two type-args; [OpKind.MATMUL]
     * combines LHS's first inner arg + RHS's last inner arg into a fresh Rank2.
     * For other ops returns null — callers fall back to [SynthesisContext.tensorIrType]
     * via [irTypeForNode].
     *
     * Operates only on rank-2 F32 surfaces (the slice-3a scope). The outer DTensor's
     * second type-arg (F32) is preserved from the operand.
     *
     * Returns null when:
     * - the dxir op isn't TRANSPOSE / MATMUL
     * - operand IrTypes aren't `IrSimpleType` (e.g., type parameters slipping through)
     * - the inner shape isn't a Rank2 with two type-args
     * - any required type-arg lookup hits a non-`IrTypeProjection` (star projection,
     *   etc.) — current Tlaloc surfaces don't emit those, but the guard keeps the
     *   helper safe if a future call site does.
     */
    private fun deriveResultIrType(
        op: DxirOp,
        operandIrTypes: Map<Int, IrType>,
        @Suppress("UNUSED_PARAMETER") fallback: IrType?,
    ): IrType? {
        // §0.4.364 — COMPARE is Bool-typed at the IR level but its runtime
        // value is the operands' F32 mask, so it propagates like elementwise.
        // §0.4.384 — rank-4 (NCHW conv/pool) surfaces derive separately: the
        // rank-2 TRANSPOSE swap and MATMUL combine below do not generalise to a
        // 4-permutation, and a conv output's spatial axes have no param-sourced
        // atom to take at all.
        if (op.type.rank == 4 && op.type.dtype == F32) {
            return deriveResultIrTypeRank4(op, operandIrTypes)
        }
        if (op.type.rank != 2) return null
        if (op.type.dtype != F32 && op.op != OpKind.COMPARE) return null
        return when (op.op) {
            OpKind.TRANSPOSE -> {
                if (op.operands.size != 1) return null
                // §0.4.197 — Phase 0c-rectangular slice 3b-2b: require operand IrType
                // explicitly in the map. Pre-§0.4.197 fell back to `tensorIrType` (=
                // first tensor param's IrType) for ANY missing operand — which poisons
                // rectangular MATMUL gradients because TRANSPOSE-of-an-unknown propagates
                // a's IrType where the operand's actual IrType is different. Backward
                // walk's solver fixes any remaining unknowns from the returns side.
                val operandSimple = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                deriveTransposedDTensor(operandSimple)
            }
            OpKind.MATMUL -> {
                if (op.operands.size != 2) return null
                val lhsIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val rhsIr = operandIrTypes[op.operands[1].id] as? IrSimpleType ?: return null
                deriveMatmulOutputDTensor(lhsIr, rhsIr)
            }
            // §0.4.198 — Phase 3 first slice: forward-propagate IrTypes through
            // elementwise unary ops (STEP / RELU / NEG / SQRT / EXP / LOG / SIN / COS /
            // ABS) — for each, output IrType equals operand IrType. Required for
            // gradient bodies that contain `STEP(matmul_output)` where matmul_output's
            // IrType is derived but STEP's wasn't, leaving downstream MUL/BROADCAST
            // without enough info to chain backward properly.
            OpKind.STEP, OpKind.RELU, OpKind.NEG,
            OpKind.SQRT, OpKind.EXP, OpKind.LOG,
            OpKind.SIN, OpKind.COS, OpKind.TAN, OpKind.ATAN, OpKind.ABS,
            OpKind.LGAMMA, OpKind.DIGAMMA, OpKind.TRIGAMMA, OpKind.POLYGAMMA,
            OpKind.TANH, OpKind.SIGMOID, OpKind.SIGN,
            // §0.4.368 — SOFTMAX is shape-preserving too (its output IrType
            // equals its operand's), so it forward-propagates like the unary
            // elementwise ops; the SoftmaxRule recomputes it in grad bodies.
            // §0.4.396 — REVERSE (flip) likewise: it permutes elements without
            // touching any extent, so output IrType = operand IrType exactly.
            OpKind.SOFTMAX, OpKind.REVERSE -> {
                if (op.operands.size != 1) return null
                operandIrTypes[op.operands[0].id]
            }
            // §0.4.198 — Forward-propagate through elementwise binary ops (ADD / SUB /
            // MUL / DIV) — output equals either operand's IrType (they must agree
            // shape-wise; the dxir guarantees that). Phase A5b adds POW, which is
            // elementwise on the same two shape-agreeing operands once the FIR has
            // splatted a literal exponent.
            // Phase A5c-2 — the operands need NOT agree any more: they broadcast. The
            // result has the WIDER operand's shape, so only a SAME-RANK operand's
            // IrType may stand in for it. Falling back to a rank-deficient operand
            // (the pre-fix behaviour, and what taking operand[0] unconditionally
            // amounts to when operand[0] is the narrow one) hands the result an
            // IrType with FEWER axes than its dxir type has — and a wrong-rank
            // IrType becomes a wrong-rank splat downstream rather than an error:
            // `mul(broadcast(1.0):[-1,-1], v:[-1])` typed from `v` made the seed
            // rank-1, so the gradient called `sumToLike([3], [2,3])`. Returning null
            // leaves the node for the backward solver, which propagates a result
            // IrType to same-rank operands only.
            OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW -> {
                if (op.operands.size != 2) return null
                op.operands.firstOrNull { it.type.rank == op.type.rank }
                    ?.let { operandIrTypes[it.id] }
            }
            // §0.4.364 — comparison surface: COMPARE's runtime mask carries the
            // operands' DTensor IrType; tensor CAST is a runtime identity; WHERE
            // takes its branches' IrType.
            OpKind.COMPARE -> {
                if (op.operands.size != 2) return null
                operandIrTypes[op.operands[0].id]
                    ?: operandIrTypes[op.operands[1].id]
            }
            OpKind.CAST -> {
                if (op.operands.size != 1) return null
                operandIrTypes[op.operands[0].id]
            }
            OpKind.WHERE -> {
                if (op.operands.size != 3) return null
                operandIrTypes[op.operands[1].id]
                    ?: operandIrTypes[op.operands[2].id]
            }
            // §0.4.373 — SUM_TO's result shape IS the template operand's shape
            // (operand[1]), so its IrType equals the template's. The value
            // operand (operand[0]) is star-projected in `sumToLike`, so its exact
            // IrType is irrelevant to synthesis.
            OpKind.SUM_TO -> {
                if (op.operands.size != 2) return null
                operandIrTypes[op.operands[1].id]
            }
            // §0.4.374 — PAD_TO's result shape IS the template operand's shape
            // (operand[1]), so its IrType equals the template's — same shape-only
            // template treatment as SUM_TO.
            OpKind.PAD_TO -> {
                if (op.operands.size != 2) return null
                operandIrTypes[op.operands[1].id]
            }
            // §0.4.399 — BROADCAST_LIKE's and SLICE_AT's result shapes ARE their
            // template operands' shapes (operand[1]), the same shape-only
            // template treatment as SUM_TO/PAD_TO.
            OpKind.BROADCAST_LIKE, OpKind.SLICE_AT -> {
                if (op.operands.size != 2) return null
                operandIrTypes[op.operands[1].id]
            }
            // §0.4.375 (Phase A4b) — RESHAPE that only INSERTS unit axes
            // (rank-increasing, e.g. `outerProduct`'s `[n]→[n,1]` / `[m]→[1,m]`
            // forward reshapes). The reshape-created size-1 axis carries no
            // param-sourced atom, so [deriveInsertedAxesDTensor] fills it with a
            // placeholder `Lit<Int>`; the operand's atoms fill the rest. This
            // types the RESHAPE so the downstream TRANSPOSE (MatmulRule's adjoint)
            // derives structurally instead of hitting `irOpFor returned null for
            // TRANSPOSE`. Only unit-axis insertions qualify (`insertedUnitAxes`
            // non-empty); rank-preserving / squeezing / general relayouts return
            // null and fall back as before.
            OpKind.RESHAPE -> {
                if (op.operands.size != 1) return null
                val operandIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val inserted = insertedUnitAxes(op.operands[0].type.dims, op.type.dims)
                    ?.takeIf { it.isNotEmpty() } ?: return null
                deriveInsertedAxesDTensor(operandIr, inserted, op.type.rank)
            }
            // Phase A2b — CONCAT's result keeps every non-axis atom of operand[0]
            // (the operands must agree off-axis) and takes a placeholder `Lit<Int>`
            // at the concat axis: that extent is the SUM of the operands' runtime
            // extents, which no param-sourced atom stands for. Exactly §0.4.375's
            // reasoning for a reshape-created unit axis, and safe for the same
            // reason — nothing reads the placeholder for a runtime-dim decision:
            // `concatPair` computes the extent, and the adjoint's `SLICE_LIKE`
            // reads its windows off template operands.
            OpKind.CONCAT -> {
                if (op.operands.size != 2) return null
                val operandIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val axis = (op.attrs["dimension"] as? Number)?.toInt() ?: return null
                val litAtom = litIntAtom() ?: return null
                val atoms = shapeAtomsOf(operandIr, op.type.rank) ?: return null
                rebuildShapeAtoms(operandIr, atoms.toMutableList().also { it[axis] = litAtom }, op.type.rank)
            }
            // Phase A2b — SLICE_LIKE's result shape IS its `thisTemplate`
            // (operand[1]), the same shape-only-template treatment SUM_TO and
            // PAD_TO get. §0.4.404 — PAD_LIKE identically: its result shape IS
            // its `outTemplate` (operand[1]).
            OpKind.SLICE_LIKE, OpKind.PAD_LIKE -> {
                if (op.operands.size < 2) return null
                operandIrTypes[op.operands[1].id]
            }
            // §0.4.400 — EMBEDDING [V,D] ⊗ [N] → [N,D]: the position atom from
            // the indices' Rank1, the feature atom from the table's Rank2 —
            // both param-sourced, no placeholder needed.
            OpKind.EMBEDDING -> {
                if (op.operands.size != 2) return null
                val tableIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val idxIr = operandIrTypes[op.operands[1].id] as? IrSimpleType ?: return null
                val tableAtoms = shapeAtomsOf(tableIr, 2) ?: return null
                val idxAtoms = shapeAtomsOf(idxIr, 1) ?: return null
                rebuildShapeAtoms(tableIr, listOf(idxAtoms[0], tableAtoms[1]), 2)
            }
            // §0.4.400 — EMBEDDING_GRAD's result IS its shape template's type
            // (operand[2], the primal table): the SUM_TO/PAD_TO treatment.
            OpKind.EMBEDDING_GRAD -> {
                if (op.operands.size != 3) return null
                operandIrTypes[op.operands[2].id]
            }
            else -> null
        }
    }

    /**
     * §0.4.384 — Phase A3b slice 1: rank-4 (NCHW) result-IrType derivation.
     *
     * - TRANSPOSE: permute the operand's four atoms by the `permutation` attr —
     *   the batch↔feature swap `[1,0,2,3]` that Conv2dRule's `dW = conv(Xᵀ, dYᵀ)`
     *   emits on both sides. The rank-2 [deriveTransposedDTensor] hard-codes the
     *   R↔C swap and cannot express it.
     * - CONV2D / CONV_TRANSPOSE2D: batch atom from the lhs, output-channel atom
     *   from the kernel's FIRST axis for CONV2D (OIHW `[Co, Ci, kh, kw]`) and its
     *   SECOND for CONV_TRANSPOSE2D (IOHW `[Ci, Co, kh, kw]` — the op contracts
     *   over axis 0 and emits axis 1). Both spatial axes take placeholder
     *   `Lit<Int>` atoms: their extents are a floor-division over runtime input
     *   extents and kernel sizes, so no param-sourced atom stands for them. That
     *   is §0.4.375's reasoning for a reshape-created unit axis and Phase A2b's
     *   for a concat axis, and it is safe for the same reason — nothing reads a
     *   placeholder for a runtime-dim decision: the conv host twins derive their
     *   own output extents from the operands' runtime `dims`.
     * - elementwise binaries/unaries: the same rank-agnostic propagation the
     *   rank-2 arms do (the forward transform's conv product rule emits a rank-4
     *   ADD of two convs; a relu/sum chain over a conv keeps propagating too).
     *
     * Pooling (MAXPOOL2D/AVGPOOL2D) follows the same shape once their slices land.
     */
    private fun deriveResultIrTypeRank4(op: DxirOp, operandIrTypes: Map<Int, IrType>): IrType? =
        when (op.op) {
            OpKind.TRANSPOSE -> {
                if (op.operands.size != 1) return null
                val perm = (op.attrs["permutation"] as? List<*>)?.map { (it as Number).toInt() }
                    ?: return null
                if (perm.size != 4 || perm.sorted() != listOf(0, 1, 2, 3)) return null
                val operandIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val atoms = shapeAtomsOf(operandIr, 4) ?: return null
                rebuildShapeAtoms(operandIr, perm.map { atoms[it] }, 4)
            }
            OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D -> {
                if (op.operands.size != 2) return null
                val lhsIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val rhsIr = operandIrTypes[op.operands[1].id] as? IrSimpleType ?: return null
                val lhsAtoms = shapeAtomsOf(lhsIr, 4) ?: return null
                val rhsAtoms = shapeAtomsOf(rhsIr, 4) ?: return null
                val litAtom = litIntAtom() ?: return null
                val cOutAxis = if (op.op == OpKind.CONV2D) 0 else 1
                rebuildShapeAtoms(
                    lhsIr,
                    listOf(lhsAtoms[0], rhsAtoms[cOutAxis], litAtom, litAtom),
                    4,
                )
            }
            OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT,
            OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT, OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT,
            -> {
                // §0.4.385 — the result IS the shape template's type (operand 2):
                // each adjoint produces a gradient shaped like the primal tensor it
                // differentiates w.r.t. No placeholder atoms needed, which is what
                // makes the fused spelling strictly easier to type than the
                // CONV_TRANSPOSE2D / TRANSPOSE chain it replaced. §0.4.391 — the
                // transposed-conv adjoints share the contract exactly.
                if (op.operands.size != 3) return null
                operandIrTypes[op.operands[2].id]
            }
            OpKind.AVGPOOL2D, OpKind.MAXPOOL2D -> {
                // §0.4.386 (avgpool), §0.4.389 (maxpool) — pooling keeps batch and
                // channels and floors the spatial extents, so the first two atoms
                // come from the input and the last two are placeholders (same
                // reasoning as the conv pair).
                if (op.operands.size != 1) return null
                val lhsIr = operandIrTypes[op.operands[0].id] as? IrSimpleType ?: return null
                val lhsAtoms = shapeAtomsOf(lhsIr, 4) ?: return null
                val litAtom = litIntAtom() ?: return null
                rebuildShapeAtoms(lhsIr, listOf(lhsAtoms[0], lhsAtoms[1], litAtom, litAtom), 4)
            }
            OpKind.AVGPOOL2D_GRAD, OpKind.MAXPOOL2D_GRAD -> {
                // §0.4.386 / §0.4.389 — the result IS operand 1's type: for avgpool
                // that operand is a shape-only template, for maxpool it is `x`
                // itself (a value operand, but still the tensor whose shape the
                // adjoint produces).
                if (op.operands.size < 2) return null
                operandIrTypes[op.operands[1].id]
            }
            OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW -> {
                if (op.operands.size != 2) return null
                op.operands.firstOrNull { it.type.rank == op.type.rank }
                    ?.let { operandIrTypes[it.id] }
            }
            OpKind.STEP, OpKind.RELU, OpKind.NEG,
            OpKind.SQRT, OpKind.EXP, OpKind.LOG,
            OpKind.SIN, OpKind.COS, OpKind.TAN, OpKind.ATAN, OpKind.ABS,
            OpKind.LGAMMA, OpKind.DIGAMMA, OpKind.TRIGAMMA, OpKind.POLYGAMMA,
            OpKind.TANH, OpKind.SIGMOID, OpKind.SIGN,
            // §0.4.396 — REVERSE is shape-preserving at any rank.
            OpKind.SOFTMAX, OpKind.REVERSE,
            -> {
                if (op.operands.size != 1) return null
                operandIrTypes[op.operands[0].id]
            }
            else -> null
        }

    /**
     * Given `DTensor<Rank2<R, C>, F32>` returns `DTensor<Rank2<C, R>, F32>`. Returns
     * null if the input isn't shaped like a 2-arg DTensor whose first arg is a 2-arg
     * Rank2.
     */
    private fun deriveTransposedDTensor(dtensor: IrSimpleType): IrSimpleType? {
        if (dtensor.arguments.size != 2) return null
        val rank2Outer = dtensor.arguments[0]
        val rank2Type = rank2Outer.typeOrNull as? IrSimpleType ?: return null
        if (rank2Type.arguments.size != 2) return null
        val swappedRank2 = reshapeIrSimpleType(
            rank2Type,
            listOf(rank2Type.arguments[1], rank2Type.arguments[0]),
        )
        // Replace the outer DTensor's first type-arg with the swapped Rank2. Reuse the
        // original `IrTypeArgument`'s variance by going through `buildSimpleType` →
        // `arguments` whose entries are IrTypeArguments. Since we're swapping the type
        // *inside* an existing projection, the cleanest path is to rebuild the
        // projection via `makeTypeProjection(type, variance)`.
        val originalProjection = rank2Outer as? org.jetbrains.kotlin.ir.types.IrTypeProjection
        val newRank2Projection = if (originalProjection != null) {
            org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(swappedRank2, originalProjection.variance)
        } else {
            org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(swappedRank2, org.jetbrains.kotlin.types.Variance.INVARIANT)
        }
        return reshapeIrSimpleType(dtensor, listOf(newRank2Projection, dtensor.arguments[1]))
    }

    /**
     * Given LHS `DTensor<Rank2<R, K>, F32>` and RHS `DTensor<Rank2<K, C>, F32>` returns
     * `DTensor<Rank2<R, C>, F32>`. The MATMUL output's outer DTensor layer is built
     * from LHS (preserves its annotations / nullability); the inner Rank2 takes its
     * first arg from LHS and its second arg from RHS. Returns null on shape mismatches.
     */
    private fun deriveMatmulOutputDTensor(lhs: IrSimpleType, rhs: IrSimpleType): IrSimpleType? {
        if (lhs.arguments.size != 2 || rhs.arguments.size != 2) return null
        val lhsRank2 = lhs.arguments[0].typeOrNull as? IrSimpleType ?: return null
        val rhsRank2 = rhs.arguments[0].typeOrNull as? IrSimpleType ?: return null
        if (lhsRank2.arguments.size != 2 || rhsRank2.arguments.size != 2) return null
        val combinedRank2 = reshapeIrSimpleType(
            lhsRank2,
            listOf(lhsRank2.arguments[0], rhsRank2.arguments[1]),
        )
        val originalLhsProjection = lhs.arguments[0] as? org.jetbrains.kotlin.ir.types.IrTypeProjection
        val variance = originalLhsProjection?.variance ?: org.jetbrains.kotlin.types.Variance.INVARIANT
        val newRank2Projection = org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(combinedRank2, variance)
        return reshapeIrSimpleType(lhs, listOf(newRank2Projection, lhs.arguments[1]))
    }

    /**
     * §0.4.197 — Backward-pass solve for a MATMUL's missing LHS operand. Given the
     * MATMUL's output IrType `Rank2<R, C>` and the known RHS IrType `Rank2<K, C>`,
     * the missing LHS must be `Rank2<R, K>` (since output.first = lhs.first = R and
     * lhs.last = rhs.first = K).
     *
     * Returns null on degenerate shapes (non-Rank2 inner, missing args). The output
     * DTensor wrapper is reused for the solved LHS so projections / nullability /
     * annotations stay consistent with the rest of the synthesised lambda.
     */
    private fun deriveMissingMatmulLhsDTensor(output: IrSimpleType, rhs: IrSimpleType): IrSimpleType? {
        if (output.arguments.size != 2 || rhs.arguments.size != 2) return null
        val outputRank2 = output.arguments[0].typeOrNull as? IrSimpleType ?: return null
        val rhsRank2 = rhs.arguments[0].typeOrNull as? IrSimpleType ?: return null
        if (outputRank2.arguments.size != 2 || rhsRank2.arguments.size != 2) return null
        val solvedRank2 = reshapeIrSimpleType(
            outputRank2,
            listOf(outputRank2.arguments[0], rhsRank2.arguments[0]),
        )
        val originalOutputProjection = output.arguments[0] as? org.jetbrains.kotlin.ir.types.IrTypeProjection
        val variance = originalOutputProjection?.variance ?: org.jetbrains.kotlin.types.Variance.INVARIANT
        val newRank2Projection = org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(solvedRank2, variance)
        return reshapeIrSimpleType(output, listOf(newRank2Projection, output.arguments[1]))
    }

    /**
     * §0.4.197 — Backward-pass solve for a MATMUL's missing RHS operand. Given the
     * MATMUL's output IrType `Rank2<R, C>` and the known LHS IrType `Rank2<R, K>`,
     * the missing RHS must be `Rank2<K, C>` (since lhs.last = rhs.first = K and
     * output.last = rhs.last = C).
     */
    private fun deriveMissingMatmulRhsDTensor(output: IrSimpleType, lhs: IrSimpleType): IrSimpleType? {
        if (output.arguments.size != 2 || lhs.arguments.size != 2) return null
        val outputRank2 = output.arguments[0].typeOrNull as? IrSimpleType ?: return null
        val lhsRank2 = lhs.arguments[0].typeOrNull as? IrSimpleType ?: return null
        if (outputRank2.arguments.size != 2 || lhsRank2.arguments.size != 2) return null
        val solvedRank2 = reshapeIrSimpleType(
            outputRank2,
            listOf(lhsRank2.arguments[1], outputRank2.arguments[1]),
        )
        val originalOutputProjection = output.arguments[0] as? org.jetbrains.kotlin.ir.types.IrTypeProjection
        val variance = originalOutputProjection?.variance ?: org.jetbrains.kotlin.types.Variance.INVARIANT
        val newRank2Projection = org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(solvedRank2, variance)
        return reshapeIrSimpleType(output, listOf(newRank2Projection, output.arguments[1]))
    }

    /**
     * §0.4.366 — given `DTensor<RankN<A0…An-1>, F32>` and result-indexed
     * [dropped] positions, returns `DTensor<RankM<kept atoms>, F32>` where
     * M = N − |dropped|. The backward-pass solve for a keepdims-unsqueeze
     * RESHAPE's operand: recovers the true squeezed rank so upstream
     * scalar-splat BROADCASTs axis-match at the correct arity.
     */
    private fun deriveDroppedAxesDTensor(dtensor: IrSimpleType, dropped: List<Int>): IrSimpleType? {
        if (dtensor.arguments.size != 2) return null
        val inner = dtensor.arguments[0].typeOrNull as? IrSimpleType ?: return null
        val keptTypes = inner.arguments
            .filterIndexed { i, _ -> i !in dropped }
            .map { it.typeOrNull ?: return null }
        if (keptTypes.size == inner.arguments.size || keptTypes.isEmpty()) return null
        val rankClassName = when (keptTypes.size) {
            1 -> "io/tlaloc/core/Rank1"
            2 -> "io/tlaloc/core/Rank2"
            3 -> "io/tlaloc/core/Rank3"
            // §0.4.384 — Phase A3b slice 1: the NCHW conv/pool ranks.
            4 -> "io/tlaloc/core/Rank4"
            else -> return null
        }
        val rankClass = pluginContext.referenceClass(ClassId.fromString(rankClassName)) ?: return null
        val newInner = rankClass.typeWith(keptTypes)
        val variance = (dtensor.arguments[0] as? org.jetbrains.kotlin.ir.types.IrTypeProjection)?.variance
            ?: org.jetbrains.kotlin.types.Variance.INVARIANT
        val proj = org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(newInner, variance)
        return reshapeIrSimpleType(dtensor, listOf(proj, dtensor.arguments[1]))
    }

    /**
     * §0.4.375 (Phase A4b) — inverse of [deriveDroppedAxesDTensor]: given a
     * lower-rank `DTensor<RankM<…>, F32>` and result-indexed [inserted]
     * unit-axis positions, returns `DTensor<RankN<…>, F32>` (N = M + |inserted|)
     * carrying a placeholder `Lit<Int>` atom at each inserted position and the
     * operand's atoms elsewhere. The reshape-created unit axes in
     * `outerProduct`'s adjoint (`[n]→[n,1]` / `[m]→[1,m]` forward, and the
     * `[n,1]→[n]` / `[1,m]→[m]` squeeze on the way out) carry no param-sourced
     * shape atom, so the `Lit<Int>` placeholder lets the downstream
     * TRANSPOSE/MATMUL derivations type structurally. The size-1 axis's static
     * atom is never read for a runtime-dim decision — unsqueeze/squeeze emit by
     * axis position, and the scalar-splat seed's shape is matched only against
     * the param-sourced atoms (the distinct `n`/`m` markers), never the
     * placeholder.
     */
    private fun deriveInsertedAxesDTensor(
        dtensor: IrSimpleType,
        inserted: List<Int>,
        resultRank: Int,
    ): IrSimpleType? {
        val existing = shapeAtomsOf(dtensor, resultRank - inserted.size) ?: return null
        if (inserted.toSet().size != inserted.size) return null
        if (inserted.any { it !in 0 until resultRank }) return null
        val litAtom = litIntAtom() ?: return null
        val insertedSet = inserted.toSet()
        val newAtoms = ArrayList<IrType>(resultRank)
        var srcIdx = 0
        for (pos in 0 until resultRank) {
            if (pos in insertedSet) {
                newAtoms += litAtom
            } else {
                newAtoms += existing.getOrNull(srcIdx++) ?: return null
            }
        }
        return rebuildShapeAtoms(dtensor, newAtoms, resultRank)
    }

    /**
     * Phase A2b — the shape atoms of a `DTensor<RankN<A0…>, F32>` IrType, or null if
     * [dtensor] is not shaped that way or does not carry exactly [rank] of them.
     */
    private fun shapeAtomsOf(dtensor: IrSimpleType, rank: Int): List<IrType>? {
        if (dtensor.arguments.size != 2) return null
        val inner = dtensor.arguments[0].typeOrNull as? IrSimpleType ?: return null
        val atoms = inner.arguments.map { it.typeOrNull ?: return null }
        return if (atoms.size == rank) atoms else null
    }

    /**
     * Phase A2b — rebuild [dtensor]'s shape argument as `Rank{rank}<newAtoms…>`,
     * keeping its dtype argument and its shape argument's variance. Shared by
     * [deriveInsertedAxesDTensor] (§0.4.375, which INSERTS placeholder atoms) and the
     * CONCAT arm (which REPLACES the concat axis's atom with one).
     */
    private fun rebuildShapeAtoms(dtensor: IrSimpleType, newAtoms: List<IrType>, rank: Int): IrSimpleType? {
        if (dtensor.arguments.size != 2) return null
        if (newAtoms.size != rank) return null
        val rankClassName = when (rank) {
            1 -> "io/tlaloc/core/Rank1"
            2 -> "io/tlaloc/core/Rank2"
            3 -> "io/tlaloc/core/Rank3"
            // §0.4.384 — Phase A3b slice 1: the NCHW conv/pool ranks. Rank5/Rank6
            // witnesses exist in `:core/Shape.kt` too; MaxPool2dRule's rank-6
            // upsample intermediates are what will need them (still deferred on the
            // single-representative `tensorIrType` generalisation).
            4 -> "io/tlaloc/core/Rank4"
            else -> return null
        }
        val rankClass = pluginContext.referenceClass(ClassId.fromString(rankClassName)) ?: return null
        val newInner = rankClass.typeWith(newAtoms)
        val variance = (dtensor.arguments[0] as? org.jetbrains.kotlin.ir.types.IrTypeProjection)?.variance
            ?: org.jetbrains.kotlin.types.Variance.INVARIANT
        val proj = org.jetbrains.kotlin.ir.types.impl.makeTypeProjection(newInner, variance)
        return reshapeIrSimpleType(dtensor, listOf(proj, dtensor.arguments[1]))
    }

    /** §0.4.375 — the placeholder `Lit<Int>` shape atom for reshape-created unit axes. */
    private fun litIntAtom(): IrType? {
        val litClass = pluginContext.referenceClass(ClassId.fromString("io/tlaloc/core/Lit")) ?: return null
        return litClass.typeWith(listOf(pluginContext.irBuiltIns.intType))
    }

    /**
     * §0.4.197 — Structural equivalence on shape-atom IrTypes. Two atoms are
     * equivalent when their classifiers match AND their type arguments recursively
     * match. Used by [matchBroadcastAxesToParams] to identify which (param, axis)
     * pair contributes each axis of a BROADCAST's target shape. `Sym` ≡ `Sym`,
     * `Lit<Int>` ≡ `Lit<Int>` but ≠ `Lit<Long>`, etc.
     */
    private fun shapeAtomEquivalent(a: IrType?, b: IrType?): Boolean {
        if (a == null || b == null) return false
        if (a === b) return true
        val aSimple = a as? IrSimpleType ?: return false
        val bSimple = b as? IrSimpleType ?: return false
        if (aSimple.classifier != bSimple.classifier) return false
        if (aSimple.arguments.size != bSimple.arguments.size) return false
        for (i in aSimple.arguments.indices) {
            if (!shapeAtomEquivalent(aSimple.arguments[i].typeOrNull, bSimple.arguments[i].typeOrNull)) return false
        }
        return true
    }

    /**
     * §0.4.197 — Match each axis of a BROADCAST's target Rank2 IrType to a
     * (param, axisIdx) pair where the param's inner-Rank2 atom at `axisIdx` is
     * structurally equivalent to the target axis atom. Returns null when ANY axis
     * fails to find a matching param-axis (BROADCAST falls back to the runtime-
     * tape path or `broadcastLike` template selection).
     *
     * Ambiguous matches (multiple params have an equivalent atom) pick the first
     * match — this is correct for the structural derivation since two equivalent
     * atoms will produce the same runtime dim value at the matched axis.
     */
    private fun matchBroadcastAxesToParams(
        targetIr: IrSimpleType,
        fnParams: List<DxirParam>,
        irParams: List<IrValueParameter>,
        paramIrTypeMap: Map<Int, IrType>,
    ): List<Pair<IrValueParameter, Int>>? {
        val innerRank2 = (targetIr.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: return null
        if (innerRank2.arguments.size != 2 && innerRank2.arguments.size != 1 && innerRank2.arguments.size != 3) return null
        val matched = mutableListOf<Pair<IrValueParameter, Int>>()
        for ((targetAxisIdx, axisArg) in innerRank2.arguments.withIndex()) {
            val targetAtom = axisArg.typeOrNull ?: return null
            // §0.4.197 — Two-pass match. First pass prefers same-axis match (target
            // axis i ↔ param axis i) so for SQUARE inputs (`Rank2<Sym, Sym>`) the
            // dims come from `[a.dims[0], a.dims[1]]` not `[a.dims[0], a.dims[0]]`.
            // Second pass falls back to ANY-axis match for the rectangular case
            // where target's axis atoms appear at different positions across params
            // (e.g., target axis 1 = `Lit<Long>` matches b's axis 1).
            var found: Pair<IrValueParameter, Int>? = null
            for ((paramIdx, p) in fnParams.withIndex()) {
                val paramIr = paramIrTypeMap[p.id] as? IrSimpleType ?: continue
                val paramInner = (paramIr.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: continue
                if (targetAxisIdx < paramInner.arguments.size &&
                    shapeAtomEquivalent(targetAtom, paramInner.arguments[targetAxisIdx].typeOrNull)
                ) {
                    found = irParams[paramIdx] to targetAxisIdx
                    break
                }
            }
            if (found == null) {
                for ((paramIdx, p) in fnParams.withIndex()) {
                    val paramIr = paramIrTypeMap[p.id] as? IrSimpleType ?: continue
                    val paramInner = (paramIr.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: continue
                    for ((axisIdx, paramAxisArg) in paramInner.arguments.withIndex()) {
                        if (shapeAtomEquivalent(targetAtom, paramAxisArg.typeOrNull)) {
                            found = irParams[paramIdx] to axisIdx
                            break
                        }
                    }
                    if (found != null) break
                }
            }
            if (found == null) return null
            matched += found
        }
        return matched
    }

    /**
     * §0.4.173 — names the FIRST gate that rejected the dxir during the most recent
     * [synthesise] call. Set by [reject] / [cancelWith] at every tagged return-null
     * site; cleared at the top of [synthesise]. Read by
     * [TlalocIrGenerationExtension] when the call returns null so the WARNING text
     * names the specific reason instead of the bare "DxirFunction falls outside the
     * scalar-primitive synthesis scope" message that landed in §0.4.171's bisection.
     * Mirrors the §0.4.169 → §0.4.172 diagnostic arc on the synthesis side.
     */
    var lastFailureReason: String? = null
        private set

    /** Tagged return-null helper — records [reason] before returning null. */
    private fun <T> reject(reason: String): T? {
        // Keep the FIRST reason (deepest gate) — later sites may pile on as the
        // null bubbles up through the call chain; the first stamp is the actionable
        // one.
        if (lastFailureReason == null) lastFailureReason = reason
        return null
    }

    /**
     * @param callTypeOverride §0.4.394 — Phase B2. When non-null, the
     *   `FunctionN<P0, …, Pn-1, R>` type used to harvest per-param and return
     *   IrTypes, INSTEAD of [originalCall]'s own type. The assembly intrinsics
     *   (`jacobian` / `hessian`) synthesise a 2-param seeded lambda
     *   (`jvp(x, dx)` / `hvp(x, v)`) at a call site whose own type is the
     *   1-param assembled function — the caller builds the seeded lambda's
     *   true function type and passes it here. [originalCall] still supplies
     *   source offsets.
     */
    fun synthesise(
        fn: DxirFunction,
        originalCall: IrCall,
        parent: IrDeclarationParent,
        callTypeOverride: IrSimpleType? = null,
    ): IrFunctionExpression? {
        lastFailureReason = null
        // Harvest the rank-1 DTensor IrType from the call site if any DxirParam is rank-1.
        // The call's type is `FunctionN<P0, …, Pn-1, R>` — the first rank-1 F32 DxirParam's
        // IrType matches `transformed.type.arguments[paramIdx].typeOrNull`. We only support
        // one distinct tensor IrType per function today (the single-rank-1-param case
        // covered by §0.4.10). Multiple tensor shapes would require a per-node map.
        val firstTensorParamIdx = fn.params.indexOfFirst { isAcceptedTensorType(it.type) }
        val tensorIrType: IrType? = if (firstTensorParamIdx < 0) null else run {
            val callType = callTypeOverride ?: originalCall.type as? IrSimpleType
                ?: return reject("call type ${originalCall.type} is not IrSimpleType")
            callType.arguments.getOrNull(firstTensorParamIdx)?.typeOrNull
                ?: return reject("tensor param at idx=$firstTensorParamIdx has no type arg on call type")
        }
        // §0.4.193 — Phase 0c-rectangular slice 2: populate per-param IrTypes from the
        // call site's `Function<P0, …, Pn-1, R>` argument list. Each rank-2/3 F32
        // DxirParam picks up its own specific IrType, so multi-param gradient bodies
        // can later resolve operand IrTypes per-DxirNode.id rather than collapsing all
        // tensor operands onto a single `tensorIrType`. For 1-param surfaces the map's
        // sole entry equals `tensorIrType`; for n-param surfaces (n ≥ 2) entries differ
        // when the params have distinct shapes.
        val callType = callTypeOverride ?: originalCall.type as? IrSimpleType
        val paramIrTypeMap = HashMap<Int, IrType>()
        if (callType != null) {
            for ((idx, p) in fn.params.withIndex()) {
                // §0.4.400 — index tensor params (embedding's rank-1 I32 indices)
                // harvest their call-site IrType too: `irTypeFor` has no fallback
                // for integer tensors, so the call site is their only source.
                if (!isAcceptedTensorType(p.type) && !isAcceptedIndexTensorType(p.type)) continue
                val argType = callType.arguments.getOrNull(idx)?.typeOrNull ?: continue
                paramIrTypeMap[p.id] = argType
            }
        }
        // §0.4.194 — Phase 0c-rectangular slice 3a: forward-pass derivation of result
        // IrTypes for OpKind.TRANSPOSE / OpKind.MATMUL on rank-2 F32 surfaces. For
        // each body op whose result IrType can be derived from its operands' IrTypes
        // via shape arithmetic (TRANSPOSE swaps R↔C; MATMUL combines lhs.first +
        // rhs.last), populate `paramIrTypeMap[op.id]` with a freshly-built
        // `IrSimpleType`. Other body ops (BROADCAST, ADD, etc.) leave their entries
        // unset; `irTypeForNode` falls back to `tensorIrType` for them — same as
        // pre-§0.4.194. Square surfaces are bit-exact equivalent (TRANSPOSE swap on
        // `Rank2<Sym, Sym>` yields the structurally same IrType; MATMUL combine
        // with all params sharing one shape produces the same IrType).
        for (n in fn.body) {
            if (n !is DxirOp) continue
            val derived = deriveResultIrType(n, paramIrTypeMap, tensorIrType) ?: continue
            paramIrTypeMap[n.id] = derived
        }
        // §0.4.186 — Phase 0c slice (b): widened from "scalar + rank-1 F32 only" to
        // "scalar + rank-1/2/3 F32". The same `broadcastLike<S>` helper handles all
        // accepted ranks via its generic shape parameter. Higher ranks + non-F32 dtypes
        // still fall back.
        for (p in fn.params) {
            // §0.4.400 — rank-1 I32 index params (embedding indices) are in scope
            // as non-differentiable pass-throughs.
            if (!p.type.isScalar && !isAcceptedTensorType(p.type) && !isAcceptedIndexTensorType(p.type)) {
                return reject("param '${p.name}' (id=${p.id}) has type ${p.type} outside scalar / rank-1-4 F32 / rank-1 I32 scope")
            }
        }
        for (n in fn.body) {
            // §0.4.364 — Bool tensor nodes (COMPARE and its consumers) are in
            // scope: the synthesis represents them as 0/1 F32 masks (see
            // [irCompare] / [irWhere] / [irCast]'s tensor arm).
            val boolTensorInScope = n.type.dtype == Bool && n.type.rank in 1..3
            // §0.4.400 — rank-1 I32 body nodes are in scope: the integer zero
            // const the reverse transform returns for a non-differentiable
            // index param (materialised via `intZerosLike`).
            if (isAcceptedIndexTensorType(n.type)) continue
            if (!n.type.isScalar && !isAcceptedTensorType(n.type) && !boolTensorInScope) {
                val opKind = (n as? DxirOp)?.op?.name ?: n::class.simpleName
                return reject("body node id=${n.id} ($opKind) has type ${n.type} outside scalar / rank-1-4 F32 scope")
            }
        }

        val context = SynthesisContext(
            tensorIrType = tensorIrType,
            tensorTemplateParam = null,
            operandIrTypes = paramIrTypeMap,
        )
        // §0.4.196 — Phase 0c-rectangular slice 3b-2a: per-param paramIrTypes from the
        // call-site populated map. For each tensor param, prefer `paramIrTypeMap[p.id]`
        // (= the call site's specific IrType for that param); fall back to `irTypeFor`
        // for scalars and edge cases. Pre-§0.4.196 every tensor param resolved to
        // `tensorIrType` (= the FIRST tensor param's IrType) — wrong for multi-param
        // surfaces with distinct shapes. Square surfaces remain bit-exact equivalent
        // (one shared IrType across all params).
        val paramIrTypes = fn.params.map { p ->
            paramIrTypeMap[p.id] ?: irTypeFor(p.type, context)
                ?: return reject("no IrType for param '${p.name}' type=${p.type}")
        }
        if (fn.returns.isEmpty() || fn.returns.size > 4) {
            return reject("returns.size=${fn.returns.size} outside [1, 4]")
        }
        // §0.4.196 — Phase 0c-rectangular slice 3b-2a: returnIrTypes derive from the
        // call-site Function<P0, …, Pn-1, R>'s R argument. Decompose Pair / Triple
        // when fn.returns.size ∈ {2, 3} so each component picks up its true IrType
        // independently rather than collapsing onto `tensorIrType`.
        //
        // For grad's Pair-return on multi-param SQUARE surfaces (both params share one
        // shape), R = Pair<a's IrType, a's IrType> — same as the pre-§0.4.196 fallback.
        // For RECTANGULAR (a: Rank2<R, K>, b: Rank2<K, C>), R = Pair<a's IrType,
        // b's IrType> — distinct components, slice-3b-2a's first correctness win.
        val returnIrTypes: List<IrType> = run {
            val callSiteR = callType?.arguments?.getOrNull(fn.params.size)?.typeOrNull as? IrSimpleType
            val decomposed = when (fn.returns.size) {
                1 -> callSiteR?.let { listOf<IrType>(it) }
                2, 3, 4 -> {
                    val components = callSiteR?.arguments?.mapNotNull { it.typeOrNull }
                    if (components != null && components.size == fn.returns.size) components else null
                }
                else -> null
            }
            if (decomposed != null) {
                decomposed
            } else {
                // Fallback for surfaces where call-site decomposition fails (no IrSimpleType,
                // unexpected component count). Match historical behaviour via `irTypeFor`.
                fn.returns.map { ret ->
                    irTypeFor(ret.type, context)
                        ?: return reject("no IrType for return id=${ret.id} type=${ret.type}")
                }
            }
        }
        // §0.4.197 — Phase 0c-rectangular slice 3b-2b: backward-pass IrType derivation.
        // Pre-populate `paramIrTypeMap` for each return DxirNode with its decomposed
        // IrType from `returnIrTypes`, then iterate the body in reverse: for each
        // MATMUL whose output IrType is known + exactly one operand IrType is unknown,
        // solve the missing operand via the matmul shape equation
        // `output: Rank2<R, C> = lhs: Rank2<R, K> · rhs: Rank2<K, C>`. Iterates to
        // fixpoint (one pass usually suffices). Propagates BROADCAST IrTypes (which
        // forward-derivation can't compute since BROADCAST's operand is scalar).
        for ((i, ret) in fn.returns.withIndex()) {
            // §0.4.400 — index-typed returns (the integer zero gradient of an
            // embedding-indices param) take their decomposed call-site IrType
            // too: `irTypeFor` has no integer-tensor fallback.
            if (paramIrTypeMap[ret.id] == null &&
                (isAcceptedTensorType(ret.type) || isAcceptedIndexTensorType(ret.type))
            ) {
                paramIrTypeMap[ret.id] = returnIrTypes[i]
            }
        }
        var changed = true
        while (changed) {
            changed = false
            for (n in fn.body.reversed()) {
                if (n !is DxirOp) continue
                when (n.op) {
                    OpKind.MATMUL -> {
                        if (n.operands.size != 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val lhsId = n.operands[0].id
                        val rhsId = n.operands[1].id
                        val lhsKnown = paramIrTypeMap[lhsId] as? IrSimpleType
                        val rhsKnown = paramIrTypeMap[rhsId] as? IrSimpleType
                        if (lhsKnown == null && rhsKnown != null) {
                            val solved = deriveMissingMatmulLhsDTensor(outputIr, rhsKnown)
                            if (solved != null) {
                                paramIrTypeMap[lhsId] = solved
                                changed = true
                            }
                        } else if (rhsKnown == null && lhsKnown != null) {
                            val solved = deriveMissingMatmulRhsDTensor(outputIr, lhsKnown)
                            if (solved != null) {
                                paramIrTypeMap[rhsId] = solved
                                changed = true
                            }
                        }
                    }
                    // §0.4.198 — Phase 3 first slice: backward propagate through
                    // elementwise binary ops. For ADD/SUB/MUL/DIV all operands and
                    // result share one IrType. If output known + one operand
                    // unknown, the unknown's IrType = output's. Phase A5b adds POW
                    // (same shape-agreeing elementwise contract).
                    // Phase A5c-2 — "share one IrType" now holds only for operands of
                    // the SAME RANK as the result: a rank-deficient operand is
                    // broadcast over new leading axes, so handing it the result's
                    // IrType would claim axes it does not have. Those keep whatever
                    // their own producers derive.
                    OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW -> {
                        if (n.operands.size != 2) continue
                        val outputIr = paramIrTypeMap[n.id] ?: continue
                        if (outputIr !is IrSimpleType) continue
                        for (operand in n.operands) {
                            if (paramIrTypeMap[operand.id] == null &&
                                isAcceptedTensorType(operand.type) &&
                                operand.type.rank == n.type.rank
                            ) {
                                paramIrTypeMap[operand.id] = outputIr
                                changed = true
                            }
                        }
                    }
                    // §0.4.198 — Backward propagate through elementwise unary ops
                    // (STEP / RELU / NEG / SQRT / EXP / LOG / SIN / COS / ABS).
                    // Output and operand share IrType; if output known + operand
                    // unknown, operand = output.
                    // §0.4.396 — REVERSE joins the shape-preserving set: its
                    // operand and result share one IrType (a flip moves
                    // elements, never extents).
                    OpKind.STEP, OpKind.RELU, OpKind.NEG,
                    OpKind.SQRT, OpKind.EXP, OpKind.LOG,
                    OpKind.SIN, OpKind.COS, OpKind.TAN, OpKind.ATAN, OpKind.ABS,
                    OpKind.LGAMMA, OpKind.DIGAMMA, OpKind.TRIGAMMA, OpKind.POLYGAMMA,
                    OpKind.TANH, OpKind.SIGMOID, OpKind.SIGN,
                    OpKind.SOFTMAX, OpKind.REVERSE -> {
                        if (n.operands.size != 1) continue
                        val outputIr = paramIrTypeMap[n.id] ?: continue
                        val operandId = n.operands[0].id
                        if (paramIrTypeMap[operandId] == null && isAcceptedTensorType(n.operands[0].type)) {
                            paramIrTypeMap[operandId] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.366 — backward propagate through the axis-reduction
                    // un-reduce chain (Phase A1). Stretch BROADCAST (equal rank):
                    // the keepdims-shaped operand takes the output's IrType — the
                    // size-1 axes' static atoms are cosmetically wrong but no
                    // synthesis decision reads them; runtime dims come from the
                    // unsqueeze helper. Scalar-splat BROADCAST operands stay
                    // untouched (rank differs). Keepdims-unsqueeze RESHAPE: the
                    // operand's TRUE squeezed rank is the output with the
                    // inserted size-1 axes dropped — this is what lets the
                    // scalar-splat BROADCAST further up the adjoint chain
                    // axis-match at rank 1 instead of inheriting the call-site
                    // rank-2 fallback and emitting a wrong-shaped splat.
                    OpKind.BROADCAST -> {
                        if (n.operands.size != 1) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val operand = n.operands[0]
                        if (operand.type.rank != n.type.rank) continue
                        if (paramIrTypeMap[operand.id] == null && isAcceptedTensorType(operand.type)) {
                            paramIrTypeMap[operand.id] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.373 — SUM_TO's output shape equals its template
                    // operand's shape (operand[1]). When the SUM_TO node's own
                    // IrType is known (it's typically a returned gradient) but the
                    // template operand's isn't yet, solve it as the output's.
                    OpKind.SUM_TO -> {
                        if (n.operands.size != 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[1].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[1].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                        // Phase A5c — and to the VALUE operand too, but only when the
                        // reduce is rank-preserving, i.e. the un-broadcast the binary
                        // VjpRules now emit. There the value IS the result shape, so
                        // the two share one IrType exactly as ADD/SUB/MUL/DIV do.
                        // Without this the chain breaks: a MUL feeding a SUM_TO lost
                        // its backward-solved IrType, its seed-BROADCAST operand fell
                        // back to `context.tensorIrType` (the rank-2 param
                        // representative) and splatted to the WRONG rank — the
                        // g2 axis-reduction gradient then called
                        // `times([2,2], [2])`. A rank-REDUCING SUM_TO (the genuine
                        // broadcasting case: `[N,C] → [N]`) must NOT propagate: its
                        // value really is bigger than the result, and its IrType
                        // comes from its own operands instead.
                        val valueId = n.operands[0].id
                        if (paramIrTypeMap[valueId] == null &&
                            isAcceptedTensorType(n.operands[0].type) &&
                            n.operands[0].type.rank == n.type.rank
                        ) {
                            paramIrTypeMap[valueId] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.374 — PAD_TO's output shape equals its template
                    // (operand[1]); solve the template's IrType from the PAD_TO
                    // node's own when it's a returned grad (mirror of SUM_TO).
                    OpKind.PAD_TO -> {
                        if (n.operands.size != 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[1].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[1].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.399 — BROADCAST_LIKE's / SLICE_AT's output shape equals
                    // its template (operand[1]); solve the template from the node's
                    // own when the adjoint is a returned grad (mirror of
                    // SUM_TO/PAD_TO). The value operand must NOT inherit it: for
                    // BROADCAST_LIKE it is generally smaller (the reduced upstream)
                    // and for SLICE_AT generally bigger (the padded upstream) —
                    // their IrTypes come from their own producers instead.
                    OpKind.BROADCAST_LIKE, OpKind.SLICE_AT -> {
                        if (n.operands.size != 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[1].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[1].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                    }
                    // Phase A2b — SLICE_LIKE's output shape is its `thisTemplate`
                    // (operand[1]), so solve the template from the node's own when the
                    // window is a returned grad (mirror of SUM_TO/PAD_TO). Neither the
                    // value operand (strictly bigger: it is the whole concat) nor the
                    // PRIOR templates (different windows again) may inherit it.
                    // §0.4.404 — PAD_LIKE identically: its output shape is its
                    // `outTemplate` (operand[1]); the value operand (strictly
                    // smaller: one window of it) and the priors must not inherit.
                    OpKind.SLICE_LIKE, OpKind.PAD_LIKE -> {
                        if (n.operands.size < 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[1].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[1].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.385 — the fused conv adjoints: the result IS the shape
                    // template (operand[2]), so solve the template from the node's
                    // own when the adjoint is a returned grad (mirror of
                    // SUM_TO/PAD_TO/SLICE_LIKE). Neither conv operand may inherit
                    // it — the upstream carries the OUTPUT's shape and the other
                    // operand is a different tensor again.
                    OpKind.CONV2D_DATA_ADJOINT, OpKind.CONV2D_KERNEL_ADJOINT,
                    OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT, OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT,
                    -> {
                        if (n.operands.size != 3) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[2].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[2].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                    }
                    // §0.4.386 — AVGPOOL2D_GRAD's template is operand[1]; same
                    // inversion (the upstream operand carries the OUTPUT's shape and
                    // must not inherit the result's). §0.4.389 — MAXPOOL2D_GRAD's
                    // operand[1] is `x`, which is likewise the result's shape (its
                    // third operand `y` carries the pooled shape and must not
                    // inherit either).
                    OpKind.AVGPOOL2D_GRAD, OpKind.MAXPOOL2D_GRAD -> {
                        if (n.operands.size < 2) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val templateId = n.operands[1].id
                        if (paramIrTypeMap[templateId] == null && isAcceptedTensorType(n.operands[1].type)) {
                            paramIrTypeMap[templateId] = outputIr
                            changed = true
                        }
                    }
                    OpKind.RESHAPE -> {
                        if (n.operands.size != 1) continue
                        val outputIr = paramIrTypeMap[n.id] as? IrSimpleType ?: continue
                        val operand = n.operands[0]
                        if (paramIrTypeMap[operand.id] != null || !isAcceptedTensorType(operand.type)) continue
                        val inserted = insertedUnitAxes(operand.type.dims, n.type.dims)
                        if (inserted == null) {
                            // §0.4.375 (Phase A4b) — SQUEEZE adjoint: the operand has
                            // MORE axes than the result (the reshape DROPPED unit axes,
                            // e.g. `outerProduct`'s adjoint `[n,1]→[n]` / `[1,m]→[m]` on
                            // the way out). Solve the operand's higher-rank IrType from
                            // the returned lower-rank grad by re-inserting a `Lit<Int>`
                            // at each dropped position (operand-indexed). This types the
                            // MATMUL result feeding the reshape, which in turn lets the
                            // MATMUL solver fill the scalar-seed's `[n,m]` IrType.
                            val dropped = insertedUnitAxes(n.type.dims, operand.type.dims)
                                ?.takeIf { it.isNotEmpty() } ?: continue
                            val solved = deriveInsertedAxesDTensor(outputIr, dropped, operand.type.rank) ?: continue
                            paramIrTypeMap[operand.id] = solved
                            changed = true
                            continue
                        }
                        if (inserted.isEmpty()) {
                            paramIrTypeMap[operand.id] = outputIr
                            changed = true
                            continue
                        }
                        val solved = deriveDroppedAxesDTensor(outputIr, inserted) ?: continue
                        paramIrTypeMap[operand.id] = solved
                        changed = true
                    }
                    else -> {}
                }
            }
        }

        // N = 1 → scalar lambda returning R.  N ∈ {2, 3} → lambda returning Pair<…> /
        // Triple<…>, matching the surface signatures of grad2 / valueAndGrad /
        // valueAndGrad2.  Boxing failure (missing kotlin.Pair / Triple symbol lookup)
        // falls back to `null` so the IR extension keeps the original call.
        val boxedReturnType: IrType = when (returnIrTypes.size) {
            1 -> returnIrTypes.single()
            2 -> pairClass()?.typeWith(returnIrTypes) ?: return reject("kotlin.Pair class symbol not found")
            3 -> tripleClass()?.typeWith(returnIrTypes) ?: return reject("kotlin.Triple class symbol not found")
            // §0.4.203 — Phase 3 fifth slice: 4-grad-output surface uses
            // `io.tlaloc.autograd.Quadruple` (which has lived in :autograd since
            // §0.4.134 for valueAndGrad3's value+3-grad return). Lifts the
            // pre-§0.4.203 cap that blocked CartPole's 4-weight NN gradient.
            4 -> quadrupleClass()?.typeWith(returnIrTypes)
                ?: return reject("io.tlaloc.autograd.Quadruple class symbol not found")
            else -> return reject("returnIrTypes.size=${returnIrTypes.size} outside [1, 4]")
        }

        val lambdaFun = pluginContext.irFactory.buildFun {
            name = SpecialNames.ANONYMOUS
            origin = IrDeclarationOrigin.LOCAL_FUNCTION_FOR_LAMBDA
            visibility = DescriptorVisibilities.LOCAL
            modality = Modality.FINAL
            returnType = boxedReturnType
            startOffset = originalCall.startOffset
            endOffset = originalCall.endOffset
        }.apply {
            this.parent = parent
        }

        val irParams: List<IrValueParameter> = fn.params.mapIndexed { i, p ->
            lambdaFun.addValueParameter(p.name, paramIrTypes[i])
        }

        // After params are materialised, capture the sole rank-1 tensor param (if any)
        // so BROADCAST / future reductions can reference it for runtime shape. Single
        // rank-1 param only for this slice — multi-tensor callers would need a richer
        // template-selection strategy keyed by DxirType equality.
        val tensorTemplateParam = if (firstTensorParamIdx >= 0) irParams[firstTensorParamIdx] else null
        val bodyContext = context.copy(
            tensorTemplateParam = tensorTemplateParam,
            fnParams = fn.params,
            irParams = irParams,
        )

        val body = buildBody(fn, lambdaFun, irParams, boxedReturnType, bodyContext)
            ?: return reject(lastFailureReason ?: "buildBody aborted (no specific gate stamped)")
        lambdaFun.body = body

        val functionType = pluginContext.irBuiltIns.functionN(fn.params.size).symbol
            .typeWith(paramIrTypes + boxedReturnType)

        return IrFunctionExpressionImpl(
            startOffset = originalCall.startOffset,
            endOffset = originalCall.endOffset,
            type = functionType,
            function = lambdaFun,
            origin = IrStatementOrigin.LAMBDA,
        )
    }

    private fun buildBody(
        fn: DxirFunction,
        lambdaFun: IrSimpleFunction,
        irParams: List<IrValueParameter>,
        boxedReturnType: IrType,
        context: SynthesisContext,
    ): IrBlockBody? {
        val builder = DeclarationIrBuilder(
            pluginContext,
            lambdaFun.symbol,
            lambdaFun.startOffset,
            lambdaFun.endOffset,
        )

        // Every DxirNode that can be used as an operand is represented by an IrValueDeclaration
        // (an IrValueParameter for params, an IrVariable for constants and op results). Using
        // `irGet` at each reference site keeps the tree node-disjoint even when the same dxir
        // value is referenced multiple times (e.g. `x * x`).
        val env = HashMap<Int, IrValueDeclaration>()
        for ((i, p) in fn.params.withIndex()) env[p.id] = irParams[i]

        return try {
            builder.irBlockBody {
                for (node in fn.body) {
                    val expr: IrExpression = when (node) {
                        is DxirConst -> irConstFor(node, context)
                            ?: cancelWith("irConstFor returned null for const id=${node.id} value=${node.value} type=${node.type}")
                        is DxirOp -> irOpFor(node, env, context)
                            ?: cancelWith("irOpFor returned null for ${node.op} id=${node.id} type=${node.type}")
                        else -> cancelWith("body node id=${node.id} is unsupported kind ${node::class.simpleName}")
                    }
                    // §0.4.194 — slice 3a: prefer the derived per-node IrType when one
                    // was populated by `synthesise()`'s body walk (TRANSPOSE / MATMUL
                    // outputs); fall back to `tensorIrType` for the rest. Square
                    // surfaces remain bit-exact since the derived IrTypes are
                    // structurally identical to the fallback.
                    val ty = irTypeForNode(node, context)
                        ?: cancelWith("no IrType for body node id=${node.id} type=${node.type}")
                    val v = irTemporary(
                        value = expr,
                        nameHint = "s${node.id}",
                        irType = ty,
                    )
                    env[node.id] = v
                }
                val returnExpr: IrExpression = when (fn.returns.size) {
                    1 -> irGet(env[fn.returns.single().id]
                        ?: cancelWith("return id=${fn.returns.single().id} not in env"))
                    2, 3, 4 -> {
                        val elementDecls = fn.returns.map {
                            env[it.id] ?: cancelWith("return id=${it.id} not in env")
                        }
                        val ctorSym = when (fn.returns.size) {
                            2 -> pairConstructor() ?: cancelWith("kotlin.Pair constructor symbol not found")
                            3 -> tripleConstructor() ?: cancelWith("kotlin.Triple constructor symbol not found")
                            // §0.4.203 — io.tlaloc.autograd.Quadruple ctor for 4-grad-output surface.
                            4 -> quadrupleConstructor() ?: cancelWith("io.tlaloc.autograd.Quadruple constructor symbol not found")
                            else -> cancelWith("returns.size=${fn.returns.size} reached the ctor switch unexpectedly")
                        }
                        val ctorCall = IrConstructorCallImpl.fromSymbolOwner(
                            startOffset = startOffset,
                            endOffset = endOffset,
                            type = boxedReturnType,
                            constructorSymbol = ctorSym,
                        )
                        // Unified argument list: the caller owns positional `arguments[i]`.
                        // fromSymbolOwner sizes the list from the constructor's parameter
                        // shape (Pair/Triple have no dispatch receiver, just N regulars).
                        elementDecls.forEachIndexed { i, decl ->
                            ctorCall.arguments[i] = irGet(decl)
                        }
                        ctorCall
                    }
                    else -> cancel()
                }
                +irReturn(returnExpr)
            }
        } catch (_: SynthesisAbort) {
            null
        }
    }

    private class SynthesisAbort : RuntimeException() {
        override fun fillInStackTrace(): Throwable = this
    }

    private fun cancel(): Nothing = throw SynthesisAbort()

    /** §0.4.173 — cancel after stamping [reason] so the caller's WARNING names the gate. */
    private fun cancelWith(reason: String): Nothing {
        if (lastFailureReason == null) lastFailureReason = reason
        throw SynthesisAbort()
    }

    private fun IrBuilderWithScope.irConstFor(node: DxirConst, context: SynthesisContext): IrExpression? {
        // §0.4.400 — the integer zero const the reverse transform returns for a
        // non-differentiable index param must be handled BEFORE the irTypeFor
        // gate below: integer tensor types have no `irTypeFor` fallback, so the
        // legacy first line would reject them unseen.
        if (!node.type.isScalar && isAcceptedIndexTensorType(node.type)) {
            return irIndexZerosConst(node, context)
        }
        val ty = irTypeFor(node.type, context) ?: return null
        val v = node.value
        // §0.4.186 — Phase 0c slice (b): rank-1/2/3 F32 consts route through
        // `broadcastLike(scalar, template)` since IR has no literal rank-N const op.
        // This handles the "unused-tensor-param zero gradient" case in
        // DxirReverseTransform (a Rank2 param whose gradient is the rank-2 zero const).
        if (!node.type.isScalar) {
            if (!isAcceptedTensorType(node.type)) return null
            val scalarValue = (v as? Number)?.toFloat() ?: return null
            val scalarConst = IrConstImpl(
                startOffset, endOffset,
                pluginContext.irBuiltIns.floatType,
                IrConstKind.Float, scalarValue,
            )

            // §0.4.200 — Phase 3 third slice: try axis-matching first (mirrors
            // §0.4.197's irBroadcast wiring). When the const's IrType has been
            // derived (forward elementwise propagation from §0.4.198) AND each
            // axis structurally matches some `(param, axisIdx)` pair, emit
            // `broadcastDimsRank{N}<S>(scalarValue, p0.dims[i0], ...)`. For
            // `tanh`/`sigmoid` gradient bodies the const-1.0 has the SAME shape
            // as the matmul output, not the param shape; the
            // `tensorTemplateParam` path was wrong for rectangular MATMUL +
            // tanh / sigmoid surfaces.
            val targetIrType = irTypeForNode(node, context) as? IrSimpleType
            if (targetIrType != null && context.fnParams.isNotEmpty()) {
                val axisMatches = matchBroadcastAxesToParams(
                    targetIrType,
                    context.fnParams,
                    context.irParams,
                    context.operandIrTypes,
                )
                if (axisMatches != null) {
                    val rank = axisMatches.size
                    val helperSym = broadcastDimsRankSymbol(rank) ?: return null
                    val shapeTypeArg = targetIrType.arguments.firstOrNull()?.typeOrNull ?: return null
                    val call = IrCallImpl.fromSymbolOwner(
                        startOffset = startOffset,
                        endOffset = endOffset,
                        type = targetIrType,
                        symbol = helperSym,
                    )
                    if (call.typeArguments.isNotEmpty()) {
                        call.typeArguments[0] = shapeTypeArg
                    }
                    call.arguments[0] = scalarConst
                    for ((i, match) in axisMatches.withIndex()) {
                        val (param, axisIdx) = match
                        val dimExpr = irParamDimAccess(param, axisIdx) ?: return null
                        call.arguments[i + 1] = dimExpr
                    }
                    return call
                }
            }

            // Fallback: existing `broadcastLike(v, tensorTemplateParam)` path.
            // Square surfaces and unused-tensor-param zero gradients rely on this.
            val template = context.tensorTemplateParam ?: return null
            val tensorIrType = context.tensorIrType as? IrSimpleType ?: return null
            val shapeTypeArg = tensorIrType.arguments.firstOrNull()?.typeOrNull ?: return null
            val helperSym = broadcastLikeSymbol() ?: return null
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = tensorIrType,
                symbol = helperSym,
            )
            call.typeArguments[0] = shapeTypeArg
            call.arguments[0] = scalarConst
            call.arguments[1] = irGet(template)
            return call
        }
        return when (node.type.dtype) {
            F32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Float, v as Float)
            F64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Double, v as Double)
            I32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Int, v as Int)
            I64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Long, v as Long)
            Bool -> null
        }
    }

    /**
     * Phase A5c-2 — the elementwise binaries that have a broadcasting host op in
     * `:core/ops/BroadcastOps.kt`. POW is absent on purpose: its tensor spelling
     * shares one shape parameter (`pow(other: DTensor<S, F32>)`), so it cannot be
     * called with two different shapes and keeps its shape-preserving symbol.
     */
    private val TENSOR_BROADCAST_BINARY_KINDS: Set<OpKind> =
        setOf(OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV)

    /**
     * Phase A5c-2 — tensor ADD/SUB/MUL/DIV → an IrCall to the matching
     * `:core/ops` broadcasting binary (`plusBroadcast(a, b)`, …). Those take
     * star-projected operands plus an explicit result-shape witness [R], which is
     * threaded from the derived result IrType exactly as `broadcastLike` /
     * `sumToLike` thread theirs.
     *
     * The witness comes from the node's own derived IrType, falling back to the
     * WIDER operand's — the result of a broadcast has the max-rank operand's shape,
     * which is why [deriveResultIrType]'s elementwise-binary arm propagates from the
     * higher-rank operand rather than blindly from operand[0].
     */
    private fun IrBuilderWithScope.irBroadcastBinary(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val name = when (op.op) {
            OpKind.ADD -> "plusBroadcast"
            OpKind.SUB -> "minusBroadcast"
            OpKind.MUL -> "timesBroadcast"
            OpKind.DIV -> "divBroadcast"
            else -> return null
        }
        val sym = opsTensorSymbol(name) ?: return null
        val lhsDecl = env[op.operands[0].id] ?: return null
        val rhsDecl = env[op.operands[1].id] ?: return null
        val wider = if (op.operands[0].type.rank >= op.operands[1].type.rank) op.operands[0] else op.operands[1]
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType)
            ?: (irTypeForNode(wider, context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeArg
        }
        call.arguments[0] = irGet(lhsDecl)
        call.arguments[1] = irGet(rhsDecl)
        return call
    }

    private fun IrBuilderWithScope.irOpFor(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.isMultiResult) return reject("op id=${op.id} ${op.op} is multi-result (types=${op.types})")
        // OpKind.IF is the only region-bearing op the synthesis scope accepts — emitted by
        // `DxirReverseTransform.handleIfAdjoint` (§0.4.23) with empty-body regions yielding
        // outer-scope adjoints. Primal-shape IFs with body ops in branches never survive to
        // synthesis (the reverse transform absorbs them via walkBranchReverse). Other
        // region-bearing ops (WHILE, MANUAL_COMPUTATION) still fall back.
        if (op.hasRegions) {
            if (op.op != OpKind.IF) return reject("op id=${op.id} ${op.op} has regions but isn't IF")
            return irIfOp(op, env, context)
        }
        // RELU and STEP don't lower to a stdlib operator; synthesise them from a primitive
        // `>` comparison + an if/else.  BROADCAST (rank-1 only) lowers to `broadcastLike`
        // in :core/ops. Everything else maps to the matching member op.
        if (op.op == OpKind.STEP) return irStep(op, env, context)
        if (op.op == OpKind.NOT) return irNot(op, env)
        if (op.op == OpKind.LAND) return irLand(op, env)
        if (op.op == OpKind.RELU) return irRelu(op, env, context)
        if (op.op == OpKind.BROADCAST) return irBroadcast(op, env, context)
        if (op.op == OpKind.SQRT) return irSqrt(op, env, context)
        // Phase A5b — a TENSOR POW joins the generic tensor-binary dispatch below
        // (`:core/ops pow`, elementwise base^exp, resolved by findTensorBinaryOp
        // exactly like ADD/SUB/MUL/DIV); [irPow] keeps the scalar `kotlin.math.pow`
        // path it has had since §0.4.52.
        if (op.op == OpKind.POW && !isAcceptedTensorType(op.type)) return irPow(op, env, context)
        if (op.op == OpKind.LOG) return irLog(op, env, context)
        if (op.op == OpKind.EXP) return irExp(op, env, context)
        if (op.op == OpKind.SIN) return irSin(op, env, context)
        if (op.op == OpKind.COS) return irCos(op, env, context)
        // §0.4.395 — Phase C2 trig tails (tensor via :core/ops, scalar via kotlin.math).
        if (op.op == OpKind.TAN) return irTan(op, env, context)
        if (op.op == OpKind.ATAN) return irAtan(op, env, context)
        // §0.4.402 — Phase C1 special functions (tensor via :core/ops, scalar via
        // the io.tlaloc.core extensions — no kotlin.math equivalent exists).
        if (op.op == OpKind.LGAMMA) return irSpecialUnary(op, env, context, "lgamma")
        if (op.op == OpKind.DIGAMMA) return irSpecialUnary(op, env, context, "digamma")
        if (op.op == OpKind.TRIGAMMA) return irSpecialUnary(op, env, context, "trigamma")
        // §0.4.405 — general polygamma: the literal `order` attr rides as an
        // extra Int const argument on the host call.
        if (op.op == OpKind.POLYGAMMA) return irPolygamma(op, env, context)
        if (op.op == OpKind.ABS) return irAbs(op, env, context)
        if (op.op == OpKind.CAST) return irCast(op, env, context)
        if (op.op == OpKind.COMPARE) return irCompare(op, env, context)
        if (op.op == OpKind.WHERE) return irWhere(op, env, context)
        if (op.op == OpKind.GATHER) return irGather(op, env, context)
        if (op.op == OpKind.SCATTER) return irScatter(op, env, context)
        if (op.op == OpKind.SCATTER_ADD) return irScatterAdd(op, env, context)
        if (op.op == OpKind.TRANSPOSE) return irTranspose(op, env, context)
        // §0.4.396 — REVERSE (flip along literal axes, Phase C3).
        if (op.op == OpKind.REVERSE) return irReverse(op, env, context)
        if (op.op == OpKind.MATMUL) return irMatmul(op, env, context)
        if (op.op == OpKind.TANH) return irTanh(op, env, context)
        if (op.op == OpKind.SIGMOID) return irSigmoid(op, env, context)
        if (op.op == OpKind.SIGN) return irSign(op, env, context)
        // §0.4.366 — reductions + the keepdims-unsqueeze RESHAPE (Phase A1).
        if (op.op == OpKind.SUM || op.op == OpKind.MEAN ||
            op.op == OpKind.MAX || op.op == OpKind.MIN
        ) {
            return irReduce(op, env, context)
        }
        if (op.op == OpKind.RESHAPE) return irReshape(op, env, context)
        if (op.op == OpKind.SOFTMAX) return irSoftmax(op, env, context)
        // §0.4.400 — Phase A3b: embedding and its fused scatter-add adjoint.
        if (op.op == OpKind.EMBEDDING) return irEmbedding(op, env, context)
        if (op.op == OpKind.EMBEDDING_GRAD) return irEmbeddingGrad(op, env, context)
        // §0.4.384 — Phase A3b slice 1: the NCHW conv pair.
        if (op.op == OpKind.CONV2D || op.op == OpKind.CONV_TRANSPOSE2D) return irConv(op, env, context)
        // §0.4.385 — the fused conv adjoints (runtime-solved padding).
        if (op.op == OpKind.CONV2D_DATA_ADJOINT || op.op == OpKind.CONV2D_KERNEL_ADJOINT) {
            return irConvAdjoint(op, env, context)
        }
        // §0.4.391 — the fused TRANSPOSED-conv adjoints (index inversion, no solve).
        if (op.op == OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT ||
            op.op == OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT
        ) {
            return irConvTransposeAdjoint(op, env, context)
        }
        // §0.4.386 — pooling and its fused adjoints. §0.4.389 covers maxpool too.
        if (op.op == OpKind.AVGPOOL2D || op.op == OpKind.MAXPOOL2D) return irPool(op, env, context)
        if (op.op == OpKind.AVGPOOL2D_GRAD || op.op == OpKind.MAXPOOL2D_GRAD) {
            return irPoolGrad(op, env, context)
        }
        if (op.op == OpKind.SUM_TO) return irSumTo(op, env, context)
        // §0.4.399 — the runtime-extent family's own adjoints (SUM_TO ⇄
        // BROADCAST_LIKE, PAD_TO ⇄ SLICE_AT): second-order reverse bodies.
        if (op.op == OpKind.BROADCAST_LIKE) return irBroadcastLike(op, env, context)
        if (op.op == OpKind.SLICE_AT) return irSliceAt(op, env, context)
        if (op.op == OpKind.SLICE) return irSlice(op, env, context)
        if (op.op == OpKind.PAD_TO) return irPadTo(op, env, context)
        // Phase A2b — concat and its runtime-extent window adjoint.
        if (op.op == OpKind.CONCAT) return irConcat(op, env, context)
        if (op.op == OpKind.SLICE_LIKE) return irSliceLike(op, env, context)
        // §0.4.404 — SLICE_LIKE's own adjoint (second-order reverse bodies
        // through a symbolic concat window).
        if (op.op == OpKind.PAD_LIKE) return irPadLike(op, env, context)
        // Phase A5c-2 — tensor ADD/SUB/MUL/DIV prefer the broadcasting host ops.
        // Under `grad {}`'s -1 sentinel dims two operands with the SAME static shape
        // can still be differently shaped at runtime (`[N,1]` and `[N,C]` are both
        // `Rank2<Sym, Lit<Int>>`), and rank-differing operands have no same-`S`
        // overload to call at all, so the shape-preserving `findTensorBinaryOp`
        // symbols below are only a fallback. Equal runtime dims take the host op's
        // flat-zip fast path, so nothing that worked before changes value or cost.
        if (op.op in TENSOR_BROADCAST_BINARY_KINDS && isAcceptedTensorType(op.type)) {
            irBroadcastBinary(op, env, context)?.let { return it }
        }

        val operandDecls = op.operands.mapIndexed { idx, o ->
            env[o.id] ?: return reject(
                "op id=${op.id} ${op.op} operand[$idx] id=${o.id} not in env " +
                    "(grad body never declared it — likely a clone-time leak)",
            )
        }
        // §0.4.42 — rank-1 ADD/SUB/MUL/DIV route through `:core/ops` tensor operators
        // (`DTensor.plus` etc., declared in HostOps.kt) rather than the primitive
        // `Float.plus`. `gradAccum`'s outer ADD accumulation for rank-1 gradient
        // contributions (e.g., multi-gather adjoints) hits this path. Scalar path
        // unchanged — `findBinaryOp` on a scalar `op.type` still resolves to the
        // primitive operator.
        // §0.4.189 — extends the §0.4.42 rank-1-only `findTensorBinaryOp` special case
        // to all `isAcceptedTensorType` ranks (1/2/3 F32). The same `:core/ops` tensor
        // operators (`DTensor.plus` etc.) handle any rank uniformly, so the dispatch
        // collapses to "tensor → findTensorBinaryOp; scalar → findBinaryOp".
        val symbol = when (op.op) {
            OpKind.ADD -> if (isAcceptedTensorType(op.type)) findTensorBinaryOp("plus") else findBinaryOp("plus", op.type, context)
            OpKind.SUB -> if (isAcceptedTensorType(op.type)) findTensorBinaryOp("minus") else findBinaryOp("minus", op.type, context)
            OpKind.MUL -> if (isAcceptedTensorType(op.type)) findTensorBinaryOp("times") else findBinaryOp("times", op.type, context)
            OpKind.DIV -> if (isAcceptedTensorType(op.type)) findTensorBinaryOp("div") else findBinaryOp("div", op.type, context)
            // Phase A5 — NEG was the one unary still routed unconditionally through
            // [findUnaryOp], which is keyed on `dxirType.dtype` alone: for a
            // TENSOR-typed NEG it resolved `kotlin.Float.unaryMinus` and then typed
            // the call as the DTensor result, so codegen emitted `checkcast Number`
            // → `floatValue` → `fneg` → `checkcast DTensor` — a ClassCastException
            // the first time the gradient ran. Reachable from any tensor SUB
            // (SubRule's `NEG(upstream)`), any tensor DIV (DivRule's `NEG(mul)`) and
            // CosRule's `-sin(x)`; no E2E surface exercised it before scalar mixing
            // landed. Mirrors the ADD/SUB/MUL/DIV dispatch directly above: tensor →
            // the `:core/ops` DTensor extension, scalar → the primitive member.
            OpKind.NEG -> if (isAcceptedTensorType(op.type)) {
                opsTensorSymbol("neg")
            } else {
                findUnaryOp("unaryMinus", op.type, context)
            }
            // Phase A5b — tensor POW. Only reached for accepted tensor types: the
            // scalar case returned through [irPow] above.
            OpKind.POW -> findTensorBinaryOp("pow")
            else -> return reject("op id=${op.id} ${op.op} type=${op.type} has no synthesis arm")
        } ?: return reject("no IR symbol for op id=${op.id} ${op.op} type=${op.type}")

        val resultType = irTypeFor(op.type, context) ?: return reject(
            "no IrType for result of op id=${op.id} ${op.op} type=${op.type}",
        )
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultType,
            symbol = symbol,
        )
        // The unified `arguments` list mirrors the callee's parameter shape: index 0 is the
        // dispatch receiver (always present for primitive member operators), followed by
        // regular params. `fromSymbolOwner` sizes the list from the symbol, so plain index
        // assignment replaces the placeholder nulls with our operand expressions.
        call.arguments[0] = irGet(operandDecls[0])
        if (operandDecls.size > 1) {
            call.arguments[1] = irGet(operandDecls[1])
        }
        return call
    }

    /**
     * STEP(x) → `if (x > 0) 1 else 0` for numeric result types, or bare `x > 0` when the
     * result is typed Bool (IF-predicate usage; §0.4.24). Synthesised via [irIfThenElse]
     * + the `>` primitive from [IrBuiltIns.greaterFunByOperandType] rather than any stdlib
     * operator — the `x == 0` case falls into the `else` branch and yields 0 / false,
     * matching the `stablehlo.compare GT` + `stablehlo.select` lowering in [:stablehlo].
     *
     * ReluRule (§0.4.7) emits STEP with the operand's numeric dtype; the FIR-side `if
     * (x > y)` lowering (§0.4.24) emits STEP with Bool dtype to feed an IF predicate.
     * Both cases share one Kotlin lowering — the operand's dtype drives the `>` lookup.
     */
    private fun IrBuilderWithScope.irStep(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        val operandDecl = env[op.operands[0].id] ?: return null
        val operandType = op.operands[0].type
        // §0.4.198 — Phase 3 first slice: tensor STEP path. When the operand and
        // result are rank-1/2/3 F32 (gradient bodies for tensor RELU primals via
        // ReluRule's `STEP(x) * upstream` emission), call the `:core/ops/step`
        // DTensor extension instead of synthesising a primitive `if (x > 0) 1 else 0`.
        // Scalar STEP path unchanged.
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(operandType)) {
            val operandIrType = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
            val operandShapeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
            val sym = stepTensorSymbol() ?: return null
            val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: operandIrType
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = resultIrType,
                symbol = sym,
            )
            // `fun <S : Shape> DTensor<S, F32>.step(): DTensor<S, F32>` — single
            // shape type-arg (extension receiver's S), no regular args.
            if (call.typeArguments.isNotEmpty()) {
                call.typeArguments[0] = operandShapeArg
            }
            call.arguments[0] = irGet(operandDecl)
            return call
        }
        val condition = greaterThanZero(operandDecl, operandType, context) ?: return null
        if (op.type.dtype == Bool) return condition
        val ty = irTypeFor(op.type, context) ?: return null
        val one = zeroOrOneConst(op.type, one = true, context) ?: return null
        val zero = zeroOrOneConst(op.type, one = false, context) ?: return null
        return irIfThenElse(ty, condition, one, zero)
    }

    /**
     * §0.4.198 — Resolves `io.tlaloc.core.ops.DTensor.step()` (the rank-1/2/3 F32
     * elementwise Heaviside step extension). Used by [irStep] when `OpKind.STEP`
     * has a tensor result type.
     */
    private fun stepTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("step"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * `OpKind.NOT` → Kotlin `!b`. Emitted by PhiCalculus F3 canonicalisation (§0.4.14)
     * when the if-branch order is swapped — the adjoint pipeline therefore sees NOT on
     * Bool-scalar predicates. Uses [IrBuiltIns.booleanNotSymbol] (the same symbol the
     * frontend resolves `!` to; avoids the `Boolean?.not()` overload ambiguity).
     */
    private fun IrBuilderWithScope.irNot(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
    ): IrExpression? {
        val operandDecl = env[op.operands[0].id] ?: return null
        val notSym = pluginContext.irBuiltIns.booleanNotSymbol
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = pluginContext.irBuiltIns.booleanType,
            symbol = notSym,
        )
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    /**
     * §0.4.55 — `OpKind.LAND(a, b)` → Kotlin `a and b` (infix `Boolean.and`). Emitted
     * by the break-hoist path in `lowerRawWhileLoop`'s branchless select: the `broke`
     * carried var's update uses LAND over Bool operands. We route to `kotlin.Boolean.and`
     * rather than `&&` so the dxir-to-IR mapping stays 1:1 with the dxir op (short-
     * circuit semantics don't matter here since both operands are cheap Bool values).
     */
    private fun IrBuilderWithScope.irLand(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
    ): IrExpression? {
        val lhsDecl = env[op.operands[0].id] ?: return null
        val rhsDecl = env[op.operands[1].id] ?: return null
        val sym = booleanAndSymbol() ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = pluginContext.irBuiltIns.booleanType,
            symbol = sym,
        )
        call.arguments[0] = irGet(lhsDecl)
        call.arguments[1] = irGet(rhsDecl)
        return call
    }

    private fun booleanAndSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            classId = ClassId(FqName("kotlin"), Name.identifier("Boolean")),
            callableName = Name.identifier("and"),
        )
        return pluginContext.referenceFunctions(callableId).firstOrNull()
    }

    /**
     * `OpKind.IF` → Kotlin `if (pred) thenYield else elseYield`. Accepts only the shape
     * `DxirReverseTransform.handleIfAdjoint` emits (§0.4.23): empty-body regions whose
     * single-terminator each yields an outer-scope SSA value. Primal-shape IFs with
     * branch-internal body ops never reach the synthesis layer (walkBranchReverse
     * absorbs them before SCT finishes).
     */
    private fun IrBuilderWithScope.irIfOp(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.regions.size != 2) return null
        val thenBlock = op.regions[0].blocks.singleOrNull() ?: return null
        val elseBlock = op.regions[1].blocks.singleOrNull() ?: return null
        if (thenBlock.body.isNotEmpty() || elseBlock.body.isNotEmpty()) return null
        if (thenBlock.terminator.size != 1 || elseBlock.terminator.size != 1) return null
        val predDecl = env[op.operands[0].id] ?: return null
        val thenDecl = env[thenBlock.terminator.single().id] ?: return null
        val elseDecl = env[elseBlock.terminator.single().id] ?: return null
        val resultTy = irTypeFor(op.type, context) ?: return null
        return irIfThenElse(resultTy, irGet(predDecl), irGet(thenDecl), irGet(elseDecl))
    }

    /**
     * RELU(x) → `if (x > 0) x else 0`.  Synthesised the same way as STEP but with the
     * then-branch returning the operand itself rather than a 1-constant. Required only
     * on the `valueAndGrad` / `valueAndGrad2` forward-prepend path; the bare `grad`
     * path for `x.relu()` doesn't clone RELU into the gradient body.
     */
    private fun IrBuilderWithScope.irRelu(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        val operandDecl = env[op.operands[0].id] ?: return null
        // §0.4.199 — Phase 3 second slice: tensor RELU path. When result is rank-1/2/3
        // F32, emit a call to `:core/ops/relu` (the DTensor extension); otherwise the
        // existing scalar `if (x > 0) x else 0` lowering applies. Forward RELU is
        // preserved in the gradient body when downstream rules read its output (e.g.,
        // `MatmulRule` reading `relu(matmul1).matmul(W2)`'s LHS for the inner matmul's
        // adjoint).
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            val operandIrType = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
            val operandShapeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
            val sym = reluTensorSymbol() ?: return null
            val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: operandIrType
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = resultIrType,
                symbol = sym,
            )
            if (call.typeArguments.isNotEmpty()) {
                call.typeArguments[0] = operandShapeArg
            }
            call.arguments[0] = irGet(operandDecl)
            return call
        }
        val ty = irTypeFor(op.type, context) ?: return null
        val zero = zeroOrOneConst(op.type, one = false, context) ?: return null
        val condition = greaterThanZero(operandDecl, op.type, context) ?: return null
        return irIfThenElse(ty, condition, irGet(operandDecl), zero)
    }

    /**
     * §0.4.199 — Resolves `io.tlaloc.core.ops.DTensor.relu()` (the rank-1/2/3 F32
     * elementwise RELU extension). Used by [irRelu] when `OpKind.RELU` has a tensor
     * result type.
     */
    private fun reluTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("relu"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.390 — resolves the DTensor `io.tlaloc.core.ops.sqrt` extension. Distinct
     * package from the scalar `io.tlaloc.core.sqrt` on Float/Double that
     * [sqrtSymbolFor] resolves, so the two never collide in `singleOrNull`.
     */
    private fun sqrtTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("sqrt"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.200 — Phase 3 third slice. `OpKind.TANH(x)` for tensor x → IrCall to
     * `:core/ops/tanh` (the DTensor extension). Mirrors [irRelu] / [irStep] /
     * [irSigmoid] rank-dispatch. Required by CartPole's `tanh(...)` chain in the
     * NN forward (and by any tanh-bearing primal more broadly).
     *
     * No scalar fallback — the existing scalar `tanh` lowering route through
     * `irUnaryMathCall(kotlin.math.tanh)` already handles scalar surfaces;
     * `irTanh` is only invoked for tensor TANH ops via the §0.4.200 dispatch
     * entry in [irOpFor].
     */
    private fun IrBuilderWithScope.irTanh(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            return tensorUnaryCall(op, env, context, tanhTensorSymbol())
        }
        // Scalar TANH falls through to irUnaryMathCall.
        return irUnaryMathCall(op, env, context, Name.identifier("tanh"))
    }

    /**
     * §0.4.200 — Phase 3 third slice. `OpKind.SIGMOID(x)` for tensor x → IrCall to
     * `:core/ops/sigmoid` (the DTensor extension). Same pattern as [irTanh].
     *
     * Phase A5b — scalar SIGMOID now synthesises too, via the new
     * `io.tlaloc.core.sigmoid` host extension ([irCoreScalarCall]). The §0.4.200
     * rejection ("no Tlaloc surface emits it scalarly") stopped holding once the
     * FIR grew the scalar `io.tlaloc.core.sigmoid` map entry; unlike TANH there is
     * no `kotlin.math` equivalent to route to.
     */
    private fun IrBuilderWithScope.irSigmoid(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            return tensorUnaryCall(op, env, context, sigmoidTensorSymbol())
        }
        return irCoreScalarCall(op, env, context, "sigmoid")
    }

    /**
     * §0.4.200 — Helper for tensor unary ops. Builds an `IrCall(symbol)` with the
     * operand's shape arg threaded through `typeArguments[0]`. Result IrType
     * derived via [irTypeForNode] (= operand IrType for unary ops, by the
     * elementwise propagation in [deriveResultIrType]).
     */
    private fun IrBuilderWithScope.tensorUnaryCall(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
        symbol: IrSimpleFunctionSymbol?,
    ): IrExpression? {
        val operandDecl = env[op.operands[0].id] ?: return null
        val sym = symbol ?: return null
        val operandIrType = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
        val operandShapeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: operandIrType
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = operandShapeArg
        }
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    private fun tanhTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("tanh"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    private fun sigmoidTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("sigmoid"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.204 — Phase 3 sixth slice. `OpKind.SIGN(x)` for tensor x → IrCall to
     * `:core/ops/sign` (the DTensor extension). Mirrors [irTanh] / [irSigmoid].
     * Scalar SIGN is rejected today (no Tlaloc scalar surface emits it).
     */
    private fun IrBuilderWithScope.irSign(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            return tensorUnaryCall(op, env, context, signTensorSymbol())
        }
        return null
    }

    private fun signTensorSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("sign"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * `BROADCAST(scalar_seed, target_type=f32[-1])` → `broadcastLike(seed, templateParam)`.
     *
     * The target shape comes from [SynthesisContext.tensorTemplateParam]'s runtime
     * `dims` (read inside the helper), not from any dxir-level dim value — [DxirType.dims]
     * is `-1` sentinel here per §0.4.10. Only the narrow scalar → rank-1 form emitted by
     * [io.tlaloc.ir.passes.VjpRegistry.SumRule] is supported: the operand must be scalar
     * and the output must be rank-1 F32.
     */
    private fun IrBuilderWithScope.irBroadcast(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        // Scalar-seed → rank-1/2/3 uniform broadcast. Dim values are irrelevant here
        // (the sentinel flows through unread); rank is gated by [isAcceptedTensorType].
        // §0.4.186 — widened from rank-1-only to rank-1/2/3 because broadcastLike's
        // generic shape parameter handles any rank uniformly.
        if (op.operands.size != 1 && op.operands.size != 2) return null
        val operand = op.operands[0]
        // §0.4.366 — equal-rank stretch arm (Phase A1): tile a keepdims-shaped
        // tensor back over its size-1 axes — the un-reduce that the axis
        // reduction VJP rules emit after their keepdims RESHAPE. Structural
        // gate mirrors the interpreter's stretch arm (same rank, each operand
        // dim 1 or equal). Dims are read at RUNTIME via `param.dims[i]`
        // (axis-matching, sentinel-safe) or a same-shaped template — never
        // baked from compile-time dims.
        if (!operand.type.isScalar) return irBroadcastStretch(op, operand, env, context)
        if (!isAcceptedTensorType(op.type)) return null

        val operandDecl = env[operand.id] ?: return null

        // Phase A5c-2 — an explicit shape-only template operand (operand[1], which
        // SumRule now attaches to a scalar seed) wins over every static guess below:
        // its RUNTIME dims are the target's, and that is the only sound source once
        // operands broadcast. Two params can share static atoms yet differ at runtime
        // (`[2,1]` and `[2,3]` are both `Rank2<Sym, Lit<Int>>`), and a rank-2 target
        // can axis-match a rank-1 param's axis outright — either way
        // [matchBroadcastAxesToParams] picked a shape the seed does not have, and the
        // wrong-shaped seed surfaced as a `sumToLike` / `elementwiseBroadcast`
        // IllegalArgumentException at run time.
        if (op.operands.size == 2) {
            val templateDecl = env[op.operands[1].id] ?: return null
            val templateIr = (irTypeForNode(op.operands[1], context) as? IrSimpleType)
                ?: (context.tensorIrType as? IrSimpleType)
            return irBroadcastLikeCall(operandDecl, templateDecl, templateIr)
        }

        // §0.4.367 — all-concrete targets (user-literal reshape shapes and
        // their splat seeds) need no structural matching at all: every dim
        // bakes as a const. This also sidesteps rank-lying static types for
        // concrete rank-changing chains (e.g. a splat to reshape(4)'s [4]).
        if (op.type.dims.all { it > 0 }) {
            val helperSym = broadcastDimsRankSymbol(op.type.rank) ?: return null
            val anyIr = (irTypeForNode(op, context) ?: context.tensorIrType) as? IrSimpleType
                ?: return null
            val shapeTypeArg = anyIr.arguments.firstOrNull()?.typeOrNull ?: return null
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = anyIr,
                symbol = helperSym,
            )
            if (call.typeArguments.isNotEmpty()) {
                call.typeArguments[0] = shapeTypeArg
            }
            call.arguments[0] = irGet(operandDecl)
            for (i in 0 until op.type.rank) {
                call.arguments[i + 1] = intConst(op.type.dims[i])
            }
            return call
        }

        // §0.4.197 — Phase 0c-rectangular slice 3b-2b: try axis-matching first.
        // When the BROADCAST's target IrType has been derived (slice 3a forward
        // pass for TRANSPOSE/MATMUL outputs OR slice 3b-2b's backward pass from
        // returns) AND each axis of its inner Rank2 atoms structurally matches
        // some `(param, axisIdx)` pair, build a `broadcastDimsRankN(v, p0.dims[i0],
        // …, pN-1.dims[iN-1])` call that constructs the runtime dims fresh from
        // the matched params. Falls back to the existing `broadcastLike(v,
        // tensorTemplateParam)` path when matching fails — covers the rank-1
        // sum-adjoint case where `tensorTemplateParam` is the rank-1 input itself.
        val targetIrType = irTypeForNode(op, context) as? IrSimpleType
        if (targetIrType != null && context.fnParams.isNotEmpty()) {
            val axisMatches = matchBroadcastAxesToParams(
                targetIrType,
                context.fnParams,
                context.irParams,
                context.operandIrTypes,
            )
            // §0.4.366 — the matched arity must equal the op's DXIR rank: for
            // squeezed-reduction targets the static IrType can be the call-site
            // fallback (a rank lie) and matching it would emit a wrong-shaped
            // splat. The backward solver derives true ranks for the un-reduce
            // chain; anything still mismatched falls through to the template
            // path or rejects — never a silently wrong shape.
            if (axisMatches != null && axisMatches.size == op.type.rank) {
                val rank = axisMatches.size
                val helperSym = broadcastDimsRankSymbol(rank) ?: return null
                // Type-arg: the result IrType's inner Rank2 (or Rank1/Rank3) — the
                // helper's `S : Shape` slot. We pass the WHOLE inner Shape (e.g.,
                // Rank2<R, C>) rather than its individual atoms because
                // `broadcastDimsRankN<S>(...)` has just one shape parameter.
                val shapeTypeArg = targetIrType.arguments.firstOrNull()?.typeOrNull
                    ?: return null
                val call = IrCallImpl.fromSymbolOwner(
                    startOffset = startOffset,
                    endOffset = endOffset,
                    type = targetIrType,
                    symbol = helperSym,
                )
                if (call.typeArguments.isNotEmpty()) {
                    call.typeArguments[0] = shapeTypeArg
                }
                call.arguments[0] = irGet(operandDecl)
                for ((i, match) in axisMatches.withIndex()) {
                    // §0.4.366/§0.4.367 — CONCRETE dxir dims (keepdims 1s,
                    // user-literal reshape targets) bake directly as consts;
                    // only -1 sentinels read a matched param's runtime dim.
                    // The static atom the matcher found for a concrete axis
                    // can be the call-site fallback's — a lie whose
                    // param.dims access would fetch the wrong extent.
                    if (op.type.dims[i] > 0) {
                        call.arguments[i + 1] = intConst(op.type.dims[i])
                        continue
                    }
                    val (param, axisIdx) = match
                    val dimExpr = irParamDimAccess(param, axisIdx) ?: return null
                    call.arguments[i + 1] = dimExpr
                }
                return call
            }
        }

        // Fallback: existing broadcastLike(v, template) path. Handles rank-1
        // SumRule's adjoint + rank-2 SQUARE surfaces where the template param
        // shares the target shape.
        val template = context.tensorTemplateParam ?: return null
        return irBroadcastLikeCall(operandDecl, template, context.tensorIrType as? IrSimpleType)
    }

    /**
     * `broadcastLike(v, template)` — splat the scalar [valueDecl] over the RUNTIME
     * shape of [templateDecl]. The callee is
     * `fun <S : Shape> broadcastLike(v: Float, template: DTensor<S, F32>)`, so its
     * single type argument is the template's shape and the call's type is the
     * template's IrType; [resultIrType] supplies both. `fromSymbolOwner` sizes
     * `arguments` from the callee's parameter shape — 2 regulars, no dispatch
     * receiver. Shared by the explicit-template arm of [irBroadcast] (Phase A5c-2)
     * and its param-template fallback.
     */
    private fun IrBuilderWithScope.irBroadcastLikeCall(
        valueDecl: IrValueDeclaration,
        templateDecl: IrValueDeclaration,
        resultIrType: IrSimpleType?,
    ): IrExpression? {
        val helperSym = broadcastLikeSymbol() ?: return null
        val ty = resultIrType ?: return null
        val shapeTypeArg = ty.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = helperSym,
        )
        call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        return call
    }

    /**
     * §0.4.197 — Resolves `io.tlaloc.core.ops.broadcastDimsRank{N}` for [rank] ∈ {1, 2, 3}.
     * Each delegate takes a `Float` value + N individual `Int` dim args and
     * forwards to the IntArray-taking [io.tlaloc.core.ops.broadcastDims].
     */
    private fun broadcastDimsRankSymbol(rank: Int): IrSimpleFunctionSymbol? {
        val name = when (rank) {
            1 -> "broadcastDimsRank1"
            2 -> "broadcastDimsRank2"
            3 -> "broadcastDimsRank3"
            else -> return null
        }
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.366 — the non-scalar half of [irBroadcast] (Phase A1): equal-rank
     * stretch of a keepdims-shaped tensor over its size-1 axes. Primary path
     * reuses [matchBroadcastAxesToParams] to read each target dim off a
     * structurally-matching param at runtime (`stretchToRankN(x, p.dims[i]…)`);
     * fallback is `stretchLike(x, template)` against the call-site template
     * param (runtime-validated in the host op — a shape mismatch fails loudly,
     * same looseness as the `broadcastLike` fallback above).
     */
    private fun IrBuilderWithScope.irBroadcastStretch(
        op: DxirOp,
        operand: io.tlaloc.ir.DxirNode,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (!isAcceptedTensorType(op.type) || !isAcceptedTensorType(operand.type)) return null
        if (operand.type.rank != op.type.rank) return null
        // §0.4.373 — a sentinel operand dim (≤ 0) can't be statically verified as
        // a valid stretch target; accept it and let the host `stretchTo` validate
        // at runtime (the same fail-loud looseness as the `stretchLike` fallback).
        // This admits the in-place size-1 stretch primal (`broadcastTo([1,C]→
        // [N,C])`) recomputed into a grad body, whose operand rides as -1 dims.
        val ok = operand.type.dims.indices.all {
            val od = operand.type.dims[it]
            od <= 0 || od == op.type.dims[it] || od == 1
        }
        if (!ok) return null
        val operandDecl = env[operand.id] ?: return null
        val targetIrType = (irTypeForNode(op, context) ?: context.tensorIrType) as? IrSimpleType
        // §0.4.373 — all-concrete target (a user-literal `broadcastTo(N, C)`):
        // every target dim bakes as a const via `stretchToRankN`, no param
        // matching needed — sentinel operand dims are validated at runtime by
        // `stretchTo`. Mirrors the scalar-splat all-concrete fast path in
        // [irBroadcast]. Un-reduce stretches (§0.4.366) keep SENTINEL targets
        // (the param's shape) so they skip this and take the param-match path.
        if (targetIrType != null && op.type.dims.all { it > 0 }) {
            val helperSym = stretchToRankSymbol(op.type.rank)
            val shapeTypeArg = targetIrType.arguments.firstOrNull()?.typeOrNull
            if (helperSym != null && shapeTypeArg != null) {
                val call = IrCallImpl.fromSymbolOwner(
                    startOffset = startOffset,
                    endOffset = endOffset,
                    type = targetIrType,
                    symbol = helperSym,
                )
                if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeTypeArg
                call.arguments[0] = irGet(operandDecl)
                for (i in 0 until op.type.rank) call.arguments[i + 1] = intConst(op.type.dims[i])
                return call
            }
        }
        // Phase A5c-3 — an explicit shape-only template operand (operand[1], which
        // the un-reduce rules attach whenever the target carries a sentinel) wins
        // over axis-matching the params: `stretchLike(x, template)` reads the target
        // extents off a value whose runtime shape IS the target. Axis-matching is a
        // guess from static atoms, and broadcasting made it unsafe — an un-reduce
        // target is often an INTERMEDIATE's shape (`(v * m).max(1)`'s adjoint
        // stretches back to the broadcast product), which no param need have.
        if (op.operands.size == 2) {
            val templateDecl = env[op.operands[1].id] ?: return null
            val templateIr = (irTypeForNode(op.operands[1], context) as? IrSimpleType)
                ?: (context.tensorIrType as? IrSimpleType)
            return irStretchLikeCall(operandDecl, templateDecl, templateIr)
        }
        if (targetIrType != null && context.fnParams.isNotEmpty()) {
            val axisMatches = matchBroadcastAxesToParams(
                targetIrType,
                context.fnParams,
                context.irParams,
                context.operandIrTypes,
            )
            if (axisMatches != null && axisMatches.size == op.type.rank) {
                val helperSym = stretchToRankSymbol(axisMatches.size) ?: return null
                val shapeTypeArg = targetIrType.arguments.firstOrNull()?.typeOrNull ?: return null
                val call = IrCallImpl.fromSymbolOwner(
                    startOffset = startOffset,
                    endOffset = endOffset,
                    type = targetIrType,
                    symbol = helperSym,
                )
                if (call.typeArguments.isNotEmpty()) {
                    call.typeArguments[0] = shapeTypeArg
                }
                call.arguments[0] = irGet(operandDecl)
                for ((i, match) in axisMatches.withIndex()) {
                    // Same concrete-dim override as the splat path above.
                    if (op.type.dims[i] > 0) {
                        call.arguments[i + 1] = intConst(op.type.dims[i])
                        continue
                    }
                    val (param, axisIdx) = match
                    call.arguments[i + 1] = irParamDimAccess(param, axisIdx) ?: return null
                }
                return call
            }
        }
        val template = context.tensorTemplateParam ?: return null
        return irStretchLikeCall(operandDecl, template, context.tensorIrType as? IrSimpleType)
    }

    /**
     * `stretchLike(x, template)` — tile [valueDecl]'s size-1 axes out to the RUNTIME
     * shape of [templateDecl]. The callee is
     * `fun <S : Shape> stretchLike(x: DTensor<*, F32>, template: DTensor<S, F32>)`,
     * so its single type argument and the call's type are both the template's shape
     * and [resultIrType] supplies them. Shared by [irBroadcastStretch]'s
     * explicit-template arm (Phase A5c-3) and its param-template fallback.
     */
    private fun IrBuilderWithScope.irStretchLikeCall(
        valueDecl: IrValueDeclaration,
        templateDecl: IrValueDeclaration,
        resultIrType: IrSimpleType?,
    ): IrExpression? {
        val helperSym = stretchLikeSymbol() ?: return null
        val ty = resultIrType ?: return null
        val shapeTypeArg = ty.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = helperSym,
        )
        call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        return call
    }

    /** §0.4.366 — resolves `io.tlaloc.core.ops.stretchToRank{N}` for rank ∈ {1, 2, 3}. */
    private fun stretchToRankSymbol(rank: Int): IrSimpleFunctionSymbol? {
        val name = when (rank) {
            1 -> "stretchToRank1"
            2 -> "stretchToRank2"
            3 -> "stretchToRank3"
            else -> return null
        }
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.366 — resolves `io.tlaloc.core.ops.stretchLike`. */
    private fun stretchLikeSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("stretchLike"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.373 — SUM_TO in gradient bodies: BroadcastRule's runtime-extent
     * unbroadcast adjoint for the in-place size-1 stretch. Calls the host twin
     * `sumToLike(value, template)`: `value` (operand[0]) is the upstream
     * gradient at the broadcast output shape, `template` (operand[1]) is the
     * primal broadcast input whose RUNTIME shape drives the reduction (its
     * values are never read). Result IrType = the template's IrType (SUM_TO's
     * output shape equals the template's), resolved via [irTypeForNode] —
     * mirror of [irBroadcastStretch]'s `stretchLike` fallback.
     */
    private fun IrBuilderWithScope.irSumTo(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val valueDecl = env[op.operands[0].id] ?: return null
        val templateDecl = env[op.operands[1].id] ?: return null
        val resultIrType = irTypeForNode(op, context) as? IrSimpleType
            ?: irTypeForNode(op.operands[1], context) as? IrSimpleType
            ?: return null
        val shapeTypeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val helperSym = sumToLikeSymbol() ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = helperSym,
        )
        call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        return call
    }

    /** §0.4.373 — resolves `io.tlaloc.core.ops.sumToLike`. */
    private fun sumToLikeSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("sumToLike"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.374 — the primal `slice` recomputed into a gradient body (it survives
     * when the sliced result feeds a downstream op, e.g. `slice(a) * b` — the
     * MulRule reads it as the primal operand). Calls the host
     * `x.slice(start, end, axis)` extension with (start, end, axis) recovered
     * from the `slice_*` attrs the FIR lowering stamped (all user literals,
     * sentinel-free). Result IrType = the SLICE node's own (resolved from the
     * returned-gradient position or backward-propagated through the elementwise
     * op that consumes it).
     */
    private fun IrBuilderWithScope.irSlice(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (!isAcceptedTensorType(operand.type) || op.type.dtype != F32) return null
        val start = (op.attrs["slice_start"] as? Number)?.toInt() ?: return null
        val end = (op.attrs["slice_end"] as? Number)?.toInt() ?: return null
        val axis = (op.attrs["slice_axis"] as? Number)?.toInt() ?: return null
        val operandDecl = env[operand.id] ?: return null
        val operandIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
        val shapeTypeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val resultIrType = irTypeForNode(op, context) as? IrSimpleType ?: return null
        val sym = sliceSymbol() ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(operandDecl)
        call.arguments[1] = intConst(start)
        call.arguments[2] = intConst(end)
        call.arguments[3] = intConst(axis)
        return call
    }

    /** §0.4.374 — resolves the `io.tlaloc.core.ops.slice(start, end, axis)` extension. */
    private fun sliceSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("slice"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.374 — SliceRule's runtime-extent adjoint (`slice`'s dual). Calls the
     * fixed-arity host `padToLikeRankN(value, template, l0..)`: `value`
     * (operand[0]) is the upstream gradient at the slice-output shape, `template`
     * (operand[1]) is the primal sliced operand whose RUNTIME shape drives the
     * trailing pad (`high[i] = template.dim[i] − low[i] − value.dim[i]`; its
     * values are never read). `low` is the user's slice start offsets (0 on the
     * non-sliced axes) baked as Int consts — mirror of [irSumTo] and the
     * `stretchToRankN` family. Result IrType = the template's (PAD_TO's output
     * shape equals the template's).
     */
    private fun IrBuilderWithScope.irPadTo(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val valueDecl = env[op.operands[0].id] ?: return null
        val templateDecl = env[op.operands[1].id] ?: return null
        @Suppress("UNCHECKED_CAST")
        val low = (op.attrs["low"] as? List<Int>) ?: return null
        val rank = op.type.rank
        if (low.size != rank) return null
        val resultIrType = irTypeForNode(op, context) as? IrSimpleType
            ?: irTypeForNode(op.operands[1], context) as? IrSimpleType
            ?: return null
        val shapeTypeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val helperSym = padToLikeRankSymbol(rank) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = helperSym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        for (i in 0 until rank) call.arguments[i + 2] = intConst(low[i])
        return call
    }

    /**
     * §0.4.399 — SumToRule's runtime-extent adjoint (SUM_TO's dual). Calls the
     * host twin `broadcastToLike(value, template)`: `value` (operand[0]) is the
     * upstream gradient at the SUM_TO-output shape, `template` (operand[1]) is
     * the primal value operand whose RUNTIME shape is the broadcast target (its
     * values are never read). Rank-polymorphic — no attrs to bake — so a single
     * symbol suffices, the [irSumTo] shape exactly. Result IrType = the
     * template's (BROADCAST_LIKE's output shape equals the template's).
     */
    private fun IrBuilderWithScope.irBroadcastLike(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val valueDecl = env[op.operands[0].id] ?: return null
        val templateDecl = env[op.operands[1].id] ?: return null
        val resultIrType = irTypeForNode(op, context) as? IrSimpleType
            ?: irTypeForNode(op.operands[1], context) as? IrSimpleType
            ?: return null
        val shapeTypeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val helperSym = broadcastToLikeSymbol() ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = helperSym,
        )
        call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        return call
    }

    /** §0.4.399 — resolves `io.tlaloc.core.ops.broadcastToLike`. */
    private fun broadcastToLikeSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("broadcastToLike"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.399 — PadToRule's runtime-extent adjoint (PAD_TO's dual). Calls the
     * fixed-arity host `sliceAtLikeRankN(value, template, l0..)`: `value`
     * (operand[0]) is the upstream gradient at the PAD_TO-output shape,
     * `template` (operand[1]) is the primal value operand whose RUNTIME shape
     * is the window's extent (its values are never read), and `low` is the
     * PAD_TO node's own literal offset baked as Int consts — the [irPadTo]
     * shape exactly. Result IrType = the template's.
     */
    private fun IrBuilderWithScope.irSliceAt(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val valueDecl = env[op.operands[0].id] ?: return null
        val templateDecl = env[op.operands[1].id] ?: return null
        @Suppress("UNCHECKED_CAST")
        val low = (op.attrs["low"] as? List<Int>) ?: return null
        val rank = op.type.rank
        if (low.size != rank) return null
        val resultIrType = irTypeForNode(op, context) as? IrSimpleType
            ?: irTypeForNode(op.operands[1], context) as? IrSimpleType
            ?: return null
        val shapeTypeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val helperSym = sliceAtLikeRankSymbol(rank) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = helperSym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(valueDecl)
        call.arguments[1] = irGet(templateDecl)
        for (i in 0 until rank) call.arguments[i + 2] = intConst(low[i])
        return call
    }

    /** §0.4.399 — resolves `io.tlaloc.core.ops.sliceAtLikeRank{1,2,3}`. */
    private fun sliceAtLikeRankSymbol(rank: Int): IrSimpleFunctionSymbol? {
        val name = when (rank) {
            1 -> "sliceAtLikeRank1"
            2 -> "sliceAtLikeRank2"
            3 -> "sliceAtLikeRank3"
            else -> return null
        }
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.374 — resolves `io.tlaloc.core.ops.padToLikeRank{1,2,3}`. */
    private fun padToLikeRankSymbol(rank: Int): IrSimpleFunctionSymbol? {
        val name = when (rank) {
            1 -> "padToLikeRank1"
            2 -> "padToLikeRank2"
            3 -> "padToLikeRank3"
            else -> return null
        }
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.366 — reductions in gradient bodies (Phase A1). Two arms:
     * scalar result → the no-arg `:core/ops` extension (`.sum()`/`.mean()`/
     * `.max()`/`.min()`) chained with `.toFloat()` (scalar dxir nodes ride
     * as Kotlin `Float` locals); axis result → the fixed-arity
     * `sumOver1/2`-family delegates with the axes baked as Int consts —
     * compile-time constants from `reduction_dims`, never runtime dims,
     * which may be symbolic sentinels. `keepDims` is derived structurally:
     * result rank == operand rank. First consumers: MaxRule/MinRule's `yRe`
     * recompute, and axis-reduction primal ops cloned into grad bodies.
     */
    /**
     * §0.4.368 — SOFTMAX in gradient bodies (Phase A3). SoftmaxRule recomputes
     * `y = SOFTMAX(x)` (the TanhRule convention) so a bare softmax in a user
     * lambda produces a SOFTMAX node in the grad body. Calls the shape-
     * preserving `:core/ops` host `.softmax(axis)` with the axis baked as an
     * Int const (a compile-time attr fact — never a runtime dim). The rest of
     * the softmax adjoint (SUM/BROADCAST-stretch/MUL/SUB) reuses the §0.4.366
     * axis-reduction synthesis arms unchanged.
     */
    private fun IrBuilderWithScope.irSoftmax(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (!isAcceptedTensorType(operand.type) || op.type.dtype != F32) return null
        val operandDecl = env[operand.id] ?: return null
        val axis = (op.attrs["axis"] as? Number)?.toInt() ?: (operand.type.rank - 1)
        val sym = softmaxSymbol() ?: return null
        val opIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
        val shapeTypeArg = opIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val resultIrType = irTypeForNode(op, context) ?: opIrType
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(operandDecl)
        call.arguments[1] = intConst(axis)
        return call
    }

    /**
     * §0.4.400 — the integer zero const the reverse transform returns for a
     * non-differentiable index param (§0.4.54's typed zero, reaching tensor
     * land for the first time). Materialised as `intZerosLike(param)` on the
     * function's index-typed param, whose RUNTIME dims are the only sound
     * shape source under -1 sentinels. Requires exactly ONE index-typed param
     * — with several, the const's sentinel-dimmed DxirType cannot say which
     * one it zeroes.
     */
    private fun IrBuilderWithScope.irIndexZerosConst(
        node: DxirConst,
        context: SynthesisContext,
    ): IrExpression? {
        if ((node.value as? Number)?.toDouble() != 0.0) return null
        val match = context.fnParams.withIndex()
            .filter { (_, p) -> isAcceptedIndexTensorType(p.type) }
            .singleOrNull() ?: return null
        val paramDecl = context.irParams.getOrNull(match.index) ?: return null
        val paramIr = (irTypeForNode(match.value, context) as? IrSimpleType)
            ?: (paramDecl.type as? IrSimpleType) ?: return null
        val shapeArg = paramIr.arguments.firstOrNull()?.typeOrNull ?: return null
        val sym = opsTensorSymbol("intZerosLike") ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = paramIr,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        call.arguments[0] = irGet(paramDecl)
        return call
    }

    /**
     * §0.4.400 — EMBEDDING in `grad {}` bodies (the primal, and its recompute
     * when the embedded rows feed a nonlinear consumer): `embedding(table,
     * indices)` in `:core/ops`. The host signature is `<V, D, N>` — vocab and
     * feature atoms from the table's Rank2, the position atom from the indices'
     * Rank1 — so all three type-args are dug out of the operand IrTypes the
     * way [irMatmul] digs `<R, K, C>`.
     */
    private fun IrBuilderWithScope.irEmbedding(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val tableDecl = env[op.operands[0].id] ?: return null
        val idxDecl = env[op.operands[1].id] ?: return null
        val tableIr = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
        val idxIr = irTypeForNode(op.operands[1], context) as? IrSimpleType ?: return null
        val tableAtoms = shapeAtomsOf(tableIr, 2) ?: return null
        val idxAtoms = shapeAtomsOf(idxIr, 1) ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType)
            ?: rebuildShapeAtoms(tableIr, listOf(idxAtoms[0], tableAtoms[1]), 2)
            ?: return null
        val sym = opsTensorSymbol("embedding") ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.size == 3) {
            call.typeArguments[0] = tableAtoms[0] // V
            call.typeArguments[1] = tableAtoms[1] // D
            call.typeArguments[2] = idxAtoms[0]   // N
        }
        call.arguments[0] = irGet(tableDecl)
        call.arguments[1] = irGet(idxDecl)
        return call
    }

    /**
     * §0.4.400 — EMBEDDING_GRAD → `:core/ops embeddingGrad(upstream, indices,
     * tableTemplate)`: EmbeddingRule's fused scatter-add adjoint. The dxir
     * operand order is (indices, upstream, template) — the §0.4.370 contract
     * plus the shape template — while the host twin leads with the upstream
     * (the [sumToLike] value-then-template convention). The template's RUNTIME
     * dims size the result (`[V, D]` is all -1 sentinels here), so the host
     * call forwards the cloned table operand as-is; its values are never read.
     */
    private fun IrBuilderWithScope.irEmbeddingGrad(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        val idxDecl = env[op.operands[0].id] ?: return null
        val upstreamDecl = env[op.operands[1].id] ?: return null
        val templateDecl = env[op.operands[2].id] ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType)
            ?: (irTypeForNode(op.operands[2], context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val sym = opsTensorSymbol("embeddingGrad") ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        call.arguments[0] = irGet(upstreamDecl)
        call.arguments[1] = irGet(idxDecl)
        call.arguments[2] = irGet(templateDecl)
        return call
    }

    /**
     * §0.4.384 — Phase A3b slice 1: CONV2D / CONV_TRANSPOSE2D → the fixed-arity
     * `:core/ops` twins (`conv2dGeneral` / `convTranspose2dGeneral`), which are
     * bit-exact against the interpreter's `evalConv2d` (pinned by
     * `DxirHostConvParityTest`).
     *
     * Every attr rides as a compile-time `Int`/`Boolean` const. That is sound here
     * — and it is the whole reason forward-mode conv works while reverse-mode does
     * not: a conv's OWN attrs (`window_strides`, `padding`, dilations) are literal
     * facts the FIR folded off the user's call, so they survive `grad {}`'s -1
     * sentinel dims unchanged. Conv2dRule's adjoint, by contrast, SOLVES its
     * padding from the primal's extents, which are exactly the symbolic values; it
     * now rejects them loudly rather than baking garbage.
     *
     * Groups have no host twin (v1 scope, matching the interpreter), so a
     * `feature_group_count`/`batch_group_count` above 1 rejects and falls back to
     * the runtime tape rather than silently convolving the wrong way.
     */
    private fun IrBuilderWithScope.irConv(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        if (!isAcceptedTensorType(op.type) || op.type.rank != 4) return null
        val fgc = (op.attrs["feature_group_count"] as? Number)?.toInt() ?: 1
        val bgc = (op.attrs["batch_group_count"] as? Number)?.toInt() ?: 1
        if (fgc != 1 || bgc != 1) {
            return reject("op id=${op.id} ${op.op} has groups $fgc/$bgc — no host twin (v1 is groups = 1)")
        }
        val transposed = op.op == OpKind.CONV_TRANSPOSE2D
        val sym = opsTensorSymbol(if (transposed) "convTranspose2dGeneral" else "conv2dGeneral") ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }

        fun intPair(key: String): List<Int> {
            val v = (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: return listOf(1, 1)
            return if (v.size == 2) v else listOf(1, 1)
        }
        val strides = intPair("window_strides")
        val lhsDil = intPair("lhs_dilation")
        val rhsDil = intPair("rhs_dilation")
        val rows = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
        val padTop = rows?.getOrNull(0)?.getOrNull(0) ?: 0
        val padBottom = rows?.getOrNull(0)?.getOrNull(1) ?: 0
        val padLeft = rows?.getOrNull(1)?.getOrNull(0) ?: 0
        val padRight = rows?.getOrNull(1)?.getOrNull(1) ?: 0
        val reversal = (op.attrs["window_reversal"] as? List<*>)?.map { it as Boolean }

        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        call.arguments[0] = irGet(decls[0])
        call.arguments[1] = irGet(decls[1])
        val ints = listOf(
            strides[0], strides[1],
            lhsDil[0], lhsDil[1],
            rhsDil[0], rhsDil[1],
            padTop, padBottom, padLeft, padRight,
        )
        ints.forEachIndexed { i, v -> call.arguments[i + 2] = intConst(v) }
        call.arguments[12] = boolConst(reversal?.getOrNull(0) ?: false)
        call.arguments[13] = boolConst(reversal?.getOrNull(1) ?: false)
        return call
    }

    /**
     * §0.4.385 — the fused conv adjoints → `:core/ops conv2dDataAdjoint` /
     * `conv2dKernelAdjoint`. Three tensor operands (the two conv operands plus the
     * shape-only template) and six Int attrs — and every one of those attrs is a
     * literal fact off the primal conv, so this arm reads no extent. The padding
     * solve happens inside the host twin against runtime `dims`, which is the whole
     * reason reverse-mode conv works under `grad {}`'s -1 sentinels at all.
     *
     * The result IrType is the TEMPLATE's, which is PAD_TO's trick and neither a
     * derivation nor a guess: the template is by construction the tensor whose shape
     * the adjoint produces. For conv that also dissolves the single-representative
     * `tensorIrType` blocker — a conv gradient body never mixes ranks, so every
     * rank-4 node takes its witness from a param.
     */
    private fun IrBuilderWithScope.irConvAdjoint(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        if (!isAcceptedTensorType(op.type) || op.type.rank != 4) return null
        val dataAdj = op.op == OpKind.CONV2D_DATA_ADJOINT
        val sym = opsTensorSymbol(if (dataAdj) "conv2dDataAdjoint" else "conv2dKernelAdjoint")
            ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }

        fun intPair(key: String): List<Int> {
            val v = (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: return listOf(1, 1)
            return if (v.size == 2) v else listOf(1, 1)
        }
        val strides = intPair("window_strides")
        val rhsDil = intPair("rhs_dilation")
        val rows = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
        // Only the primal's LOW padding matters: the high side is implied by the
        // upstream's runtime shape (the shape the primal's floor-division produced).
        val padTop = rows?.getOrNull(0)?.getOrNull(0) ?: 0
        val padLeft = rows?.getOrNull(1)?.getOrNull(0) ?: 0

        val resultIrType = (irTypeForNode(op.operands[2], context) as? IrSimpleType)
            ?: (irTypeForNode(op, context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        decls.forEachIndexed { i, decl -> call.arguments[i] = irGet(decl) }
        val ints = listOf(strides[0], strides[1], rhsDil[0], rhsDil[1], padTop, padLeft)
        ints.forEachIndexed { i, v -> call.arguments[i + 3] = intConst(v) }
        return call
    }

    /**
     * §0.4.391 — the fused TRANSPOSED-conv adjoints → `:core/ops
     * convTranspose2dDataAdjoint` / `convTranspose2dKernelAdjoint`. Same shape as
     * [irConvAdjoint], with two differences: the attr set is the transposed conv's
     * (so `lhs_dilation` and `window_reversal` ride along too — the index inversion
     * needs both), and there is no padding to solve, since the host twin inverts the
     * primal's tap equation per element.
     */
    private fun IrBuilderWithScope.irConvTransposeAdjoint(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        if (!isAcceptedTensorType(op.type) || op.type.rank != 4) return null
        val dataAdj = op.op == OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT
        val sym = opsTensorSymbol(
            if (dataAdj) "convTranspose2dDataAdjoint" else "convTranspose2dKernelAdjoint",
        ) ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }

        fun intPair(key: String): List<Int> {
            val v = (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: return listOf(1, 1)
            return if (v.size == 2) v else listOf(1, 1)
        }
        val strides = intPair("window_strides")
        val lhsDil = intPair("lhs_dilation")
        val rhsDil = intPair("rhs_dilation")
        val rows = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
        val padTop = rows?.getOrNull(0)?.getOrNull(0) ?: 0
        val padLeft = rows?.getOrNull(1)?.getOrNull(0) ?: 0
        val reversal = (op.attrs["window_reversal"] as? List<*>)?.map { it as Boolean }

        val resultIrType = (irTypeForNode(op.operands[2], context) as? IrSimpleType)
            ?: (irTypeForNode(op, context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        decls.forEachIndexed { i, decl -> call.arguments[i] = irGet(decl) }
        val ints = listOf(
            strides[0], strides[1],
            lhsDil[0], lhsDil[1],
            rhsDil[0], rhsDil[1],
            padTop, padLeft,
        )
        ints.forEachIndexed { i, v -> call.arguments[i + 3] = intConst(v) }
        call.arguments[11] = boolConst(reversal?.getOrNull(0) ?: false)
        call.arguments[12] = boolConst(reversal?.getOrNull(1) ?: false)
        return call
    }

    /**
     * §0.4.386 — `OpKind.AVGPOOL2D`, and (§0.4.389) `OpKind.MAXPOOL2D` →
     * `:core/ops avgPool2dGeneral` / `maxPool2dGeneral`, the host twins that are
     * bit-exact against the interpreter's `evalPool2d` (count_include_pad for the
     * average branch: the sum divides by the FULL window). A pooling primal reaches
     * a gradient body through the value stream — MaxPool2dRule recomputes `y` there
     * for its mask, and a `Σ p²`-style loss reads the pooled value — exactly as
     * CONV2D does. All attrs are literals.
     */
    private fun IrBuilderWithScope.irPool(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (!isAcceptedTensorType(op.type) || op.type.rank != 4) return null
        val sym = opsTensorSymbol(
            if (op.op == OpKind.MAXPOOL2D) "maxPool2dGeneral" else "avgPool2dGeneral",
        ) ?: return null
        val decl = env[op.operands[0].id] ?: return null

        fun intPair(key: String, def: List<Int>): List<Int> {
            val v = (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: return def
            return if (v.size == 2) v else def
        }
        val window = intPair("window", emptyList())
        if (window.size != 2) return reject("op id=${op.id} ${op.op} needs a `window` attr; got ${op.attrs}")
        val strides = intPair("window_strides", window)
        val rows = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }

        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        call.arguments[0] = irGet(decl)
        val ints = listOf(
            window[0], window[1], strides[0], strides[1],
            rows?.getOrNull(0)?.getOrNull(0) ?: 0, rows?.getOrNull(0)?.getOrNull(1) ?: 0,
            rows?.getOrNull(1)?.getOrNull(0) ?: 0, rows?.getOrNull(1)?.getOrNull(1) ?: 0,
        )
        ints.forEachIndexed { i, v -> call.arguments[i + 1] = intConst(v) }
        return call
    }

    /**
     * §0.4.386 — the fused pooling adjoints → `:core/ops avgPool2dGrad` and
     * (§0.4.389) `maxPool2dGrad`. Same shape as [irConvAdjoint]: literal attrs
     * only, the extent question answered at runtime by INVERTING the window against
     * the operands' real `dims`, and the result IrType taken from the tensor whose
     * shape the adjoint produces (operand 1) rather than derived.
     *
     * The two kinds differ in arity: avgpool's second operand is a shape-only
     * template, while maxpool reads `x`'s VALUES to find the window winners and
     * takes the recomputed pooled `y` as a third operand. Hence attrs start at
     * `decls.size` instead of a fixed index.
     */
    private fun IrBuilderWithScope.irPoolGrad(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        val isMax = op.op == OpKind.MAXPOOL2D_GRAD
        if (op.operands.size != if (isMax) 3 else 2) return null
        if (!isAcceptedTensorType(op.type) || op.type.rank != 4) return null
        val sym = opsTensorSymbol(if (isMax) "maxPool2dGrad" else "avgPool2dGrad") ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }

        fun intPair(key: String, def: List<Int>): List<Int> {
            val v = (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: return def
            return if (v.size == 2) v else def
        }
        val window = intPair("window", emptyList())
        if (window.size != 2) {
            return reject("op id=${op.id} ${op.op} needs a `window` attr; got ${op.attrs}")
        }
        val strides = intPair("window_strides", window)
        val rows = (op.attrs["padding"] as? List<*>)
            ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
        val padTop = rows?.getOrNull(0)?.getOrNull(0) ?: 0
        val padLeft = rows?.getOrNull(1)?.getOrNull(0) ?: 0

        val resultIrType = (irTypeForNode(op.operands[1], context) as? IrSimpleType)
            ?: (irTypeForNode(op, context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        decls.forEachIndexed { i, decl -> call.arguments[i] = irGet(decl) }
        val ints = listOf(window[0], window[1], strides[0], strides[1], padTop, padLeft)
        ints.forEachIndexed { i, v -> call.arguments[i + decls.size] = intConst(v) }
        return call
    }

    /**
     * Phase A2b — `OpKind.CONCAT(a, b)` → an IrCall to `:core/ops concatPair(axis, a, b)`.
     *
     * Exactly two operands: the FIR folds an n-ary user `concat`/`stack` into a
     * right-fold of binary CONCATs precisely so this arm never needs an `IrVararg`
     * (which the plugin cannot build — the documented reason for the `…RankN` shim
     * family). An IR-level n-ary CONCAT therefore has no synthesis path and falls
     * back to the tape; `DxirShapePlumbingTest` and the emitter tests exercise those
     * at the IR level, where they belong.
     */
    private fun IrBuilderWithScope.irConcat(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        if (!isAcceptedTensorType(op.type)) return null
        val sym = opsTensorSymbol("concatPair") ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }
        val axis = (op.attrs["dimension"] as? Number)?.toInt() ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        call.arguments[0] = intConst(axis)
        call.arguments[1] = irGet(decls[0])
        call.arguments[2] = irGet(decls[1])
        return call
    }

    /**
     * Phase A2b — `OpKind.SLICE_LIKE(value, thisTemplate, priorTemplate…)` → the
     * matching fixed-arity `:core/ops` twin (`sliceLikeStart` / `sliceLikeAfter{1,2,3}`),
     * selected by the PRIOR-template count. The axis rides as an Int const; every
     * extent is read off the templates at runtime, which is the whole point (a concat
     * operand's window offset is the cumulative sum of the prior operands' runtime
     * extents and does not exist at compile time). Bounded at 4 concat operands —
     * the FIR's fold-to-binary means user code only ever needs one prior.
     */
    private fun IrBuilderWithScope.irSliceLike(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        val priors = op.operands.size - 2
        if (priors !in 0..3) return null
        if (!isAcceptedTensorType(op.type)) return null
        val name = when (priors) {
            0 -> "sliceLikeStart"
            1 -> "sliceLikeAfter1"
            2 -> "sliceLikeAfter2"
            else -> "sliceLikeAfter3"
        }
        val sym = opsTensorSymbol(name) ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }
        val axis = (op.attrs["axis"] as? Number)?.toInt() ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType)
            ?: (irTypeForNode(op.operands[1], context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        decls.forEachIndexed { i, decl -> call.arguments[i] = irGet(decl) }
        call.arguments[decls.size] = intConst(axis)
        return call
    }

    /**
     * §0.4.404 — `OpKind.PAD_LIKE(value, outTemplate, priorTemplate…)` → the
     * matching fixed-arity `:core/ops` twin (`padLikeStart` / `padLikeAfter{1,2,3}`),
     * selected by the PRIOR-template count — the [irSliceLike] shape exactly, since
     * PAD_LIKE is SLICE_LIKE's transpose: the axis rides as an Int const; the offset
     * and target extent are read off the templates at runtime (a concat window's
     * offset is the cumulative sum of the prior operands' runtime extents and does
     * not exist at compile time). Result IrType = the outTemplate's (operand[1]).
     */
    private fun IrBuilderWithScope.irPadLike(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        val priors = op.operands.size - 2
        if (priors !in 0..3) return null
        if (!isAcceptedTensorType(op.type)) return null
        val name = when (priors) {
            0 -> "padLikeStart"
            1 -> "padLikeAfter1"
            2 -> "padLikeAfter2"
            else -> "padLikeAfter3"
        }
        val sym = opsTensorSymbol(name) ?: return null
        val decls = op.operands.map { env[it.id] ?: return null }
        val axis = (op.attrs["axis"] as? Number)?.toInt() ?: return null
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType)
            ?: (irTypeForNode(op.operands[1], context) as? IrSimpleType)
            ?: return null
        val shapeArg = resultIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) call.typeArguments[0] = shapeArg
        decls.forEachIndexed { i, decl -> call.arguments[i] = irGet(decl) }
        call.arguments[decls.size] = intConst(axis)
        return call
    }

    /** §0.4.368 — resolves the `io.tlaloc.core.ops.softmax(axis)` extension (one Regular param). */
    private fun softmaxSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("softmax"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    private fun IrBuilderWithScope.irReduce(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (!isAcceptedTensorType(operand.type)) return null
        val operandDecl = env[operand.id] ?: return null
        val opName = when (op.op) {
            OpKind.SUM -> "sum"
            OpKind.MEAN -> "mean"
            OpKind.MAX -> "max"
            OpKind.MIN -> "min"
            else -> return null
        }
        val rd = (op.attrs["reduction_dims"] as? List<*>)?.map { (it as Number).toInt() }
            ?: emptyList()
        val operandIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
        val shapeTypeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        if (op.type.isScalar) {
            val sym = reduceFullSymbol(opName) ?: return null
            val toFloatSym = toFloatSymbol() ?: return null
            val reduceCall = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = sym.owner.returnType,
                symbol = sym,
            )
            if (reduceCall.typeArguments.isNotEmpty()) {
                reduceCall.typeArguments[0] = shapeTypeArg
            }
            reduceCall.arguments[0] = irGet(operandDecl)
            val castCall = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = pluginContext.irBuiltIns.floatType,
                symbol = toFloatSym,
            )
            castCall.arguments[0] = reduceCall
            return castCall
        }
        // §0.4.390 — up to three axes (see [reduceOverSymbol]): a rank-4 NCHW
        // per-channel statistic reduces over the batch and both spatial axes.
        if (rd.isEmpty() || rd.size > 3) return null
        val keep = op.type.rank == operand.type.rank
        val sym = reduceOverSymbol(opName, rd.size) ?: return null
        val resultIrType = irTypeForNode(op, context) ?: irTypeFor(op.type, context) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(operandDecl)
        call.arguments[1] = intConst(rd[0])
        // The axis consts are positional and `keepDims` last, matching the
        // `{name}Over{N}` shim arities.
        when (rd.size) {
            1 -> call.arguments[2] = boolConst(keep)
            2 -> {
                call.arguments[2] = intConst(rd[1])
                call.arguments[3] = boolConst(keep)
            }
            else -> {
                call.arguments[2] = intConst(rd[1])
                call.arguments[3] = intConst(rd[2])
                call.arguments[4] = boolConst(keep)
            }
        }
        return call
    }

    /**
     * §0.4.366 — RESHAPE in gradient bodies, scoped to the keepdims
     * unsqueeze the axis-reduction rules emit: result dims must equal the
     * operand dims with size-1 axes INSERTED. The inserted positions are
     * structural compile-time facts (from `reduction_dims`), so they bake
     * as Int consts without touching possibly-sentinel dim values. General
     * relayout RESHAPE stays out of synthesis scope until Phase A2.
     */
    private fun IrBuilderWithScope.irReshape(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (!isAcceptedTensorType(operand.type) || op.type.dtype != F32) return null
        val operandDecl = env[operand.id] ?: return null
        if (op.type.dims == operand.type.dims) return irGet(operandDecl)
        val resultIrType = irTypeForNode(op, context) ?: irTypeFor(op.type, context) ?: return null
        val shapeTypeArg = (resultIrType as? IrSimpleType)?.arguments?.firstOrNull()?.typeOrNull
            ?: return null
        fun emit(sym: IrSimpleFunctionSymbol, ints: List<IrExpression>): IrExpression {
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = resultIrType,
                symbol = sym,
            )
            if (call.typeArguments.isNotEmpty()) {
                call.typeArguments[0] = shapeTypeArg
            }
            call.arguments[0] = irGet(operandDecl)
            for ((i, e) in ints.withIndex()) call.arguments[i + 1] = e
            return call
        }
        // Unsqueeze arm (§0.4.366): result = operand with size-1 axes inserted.
        // §0.4.390 widened the cap to three: `[C] → [1,C,1,1]` is how a per-channel
        // parameter becomes broadcastable against an NCHW tensor.
        val inserted = insertedUnitAxes(operand.type.dims, op.type.dims)
        if (inserted != null && inserted.isNotEmpty() && inserted.size <= 3) {
            val sym = unsqueezeSymbol(inserted.size) ?: return null
            return emit(sym, inserted.map { intConst(it) })
        }
        // §0.4.367 — squeeze arm: result = operand with size-1 axes DROPPED
        // (the adjoint of an unsqueeze, and the user `squeeze(axis)` primal).
        val dropped = insertedUnitAxes(op.type.dims, operand.type.dims)
        if (dropped != null && dropped.isNotEmpty() && dropped.size <= 3) {
            val sym = squeezeAxesSymbol(dropped.size) ?: return null
            return emit(sym, dropped.map { intConst(it) })
        }
        // §0.4.367 — general relayout (user `reshape(dims)` / `flatten` and
        // their adjoints): rank-1..4 targets via `reshapeToRankN`. Concrete
        // dims bake as consts; -1 sentinels read a structurally-matched
        // param's runtime dim (the broadcast paths' rule).
        if (op.type.rank !in 1..4) return null
        val sym = reshapeToRankSymbol(op.type.rank) ?: return null
        val allConcrete = op.type.dims.all { it > 0 }
        val matches = if (allConcrete) null else {
            val targetSimple = resultIrType as? IrSimpleType ?: return null
            if (context.fnParams.isEmpty()) return null
            val m = matchBroadcastAxesToParams(
                targetSimple, context.fnParams, context.irParams, context.operandIrTypes,
            )
            if (m == null || m.size != op.type.rank) return null
            m
        }
        val dimExprs = (0 until op.type.rank).map { i ->
            if (op.type.dims[i] > 0) intConst(op.type.dims[i])
            else {
                val (param, axisIdx) = matches!![i]
                irParamDimAccess(param, axisIdx) ?: return null
            }
        }
        return emit(sym, dimExprs)
    }

    /**
     * §0.4.366 — match [outDims] as [inDims] with size-1 axes inserted;
     * returns the inserted positions (result-indexed, ascending) or null if
     * the shapes don't relate that way. Ambiguity against input dims that are
     * themselves 1 resolves greedily — any valid assignment is runtime-
     * equivalent (the flat data is untouched either way).
     */
    private fun insertedUnitAxes(inDims: List<Int>, outDims: List<Int>): List<Int>? {
        if (outDims.size < inDims.size) return null
        val inserted = mutableListOf<Int>()
        var i = 0
        for (o in outDims.indices) {
            if (i < inDims.size && outDims[o] == inDims[i]) {
                i++
                continue
            }
            if (outDims[o] == 1) {
                inserted += o
                continue
            }
            return null
        }
        return if (i == inDims.size) inserted else null
    }

    /**
     * §0.4.366 — resolves the NO-ARG `:core/ops` reduction extension
     * (`sum`/`mean`/`max`/`min`); these names also carry the vararg axis
     * overloads, so filter to the overload with zero Regular parameters.
     */
    private fun reduceFullSymbol(name: String): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            sym.owner.parameters.none { it.kind == IrParameterKind.Regular }
        }
    }

    /** §0.4.366 — resolves `io.tlaloc.core.ops.{name}Over{axisCount}` (distinct names, no overloads). */
    private fun reduceOverSymbol(name: String, axisCount: Int): IrSimpleFunctionSymbol? {
        // §0.4.390 — three axes too: `mean(0, 2, 3)` is how training batchNorm takes
        // its per-channel statistics over an NCHW tensor.
        if (axisCount !in 1..3) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("${name}Over$axisCount"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.366 — resolves `io.tlaloc.core.ops.unsqueezeAxes{N}` for N ∈ {1, 2, 3}. */
    private fun unsqueezeSymbol(count: Int): IrSimpleFunctionSymbol? {
        // §0.4.390 — three inserted axes: `[C] → [1,C,1,1]`, the per-channel
        // parameter reshape an NCHW batchNorm needs.
        if (count !in 1..3) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("unsqueezeAxes$count"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.367 — resolves `io.tlaloc.core.ops.squeezeAxes{N}` for N ∈ {1, 2, 3}. */
    private fun squeezeAxesSymbol(count: Int): IrSimpleFunctionSymbol? {
        // §0.4.390 — three dropped axes: the adjoint of `[C] → [1,C,1,1]`.
        if (count !in 1..3) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("squeezeAxes$count"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.367 — resolves `io.tlaloc.core.ops.reshapeToRank{N}` for N ∈ {1, 2, 3, 4}. */
    private fun reshapeToRankSymbol(rank: Int): IrSimpleFunctionSymbol? {
        // §0.4.390 — rank 4 too, for the NCHW surfaces.
        if (rank !in 1..4) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("reshapeToRank$rank"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.367 — resolves `io.tlaloc.core.ops.transposePerm{N}` for N ∈ {2, 3}.
     * §0.4.384 — N = 4 too, for the batch↔feature swap Conv2dRule's `dW` emits.
     */
    private fun transposePermSymbol(rank: Int): IrSimpleFunctionSymbol? {
        if (rank !in 2..4) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("transposePerm$rank"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.396 — resolves `io.tlaloc.core.ops.flipAxes{N}` for N ∈ {1, 2, 3}
     * (the fixed-arity delegates of the vararg `flip`; the usual IrVararg
     * reason — see [transposePermSymbol]).
     */
    private fun flipAxesSymbol(count: Int): IrSimpleFunctionSymbol? {
        if (count !in 1..3) return null
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("flipAxes$count"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /** §0.4.366 — resolves `io.tlaloc.core.ops.toFloat` (the scalar-DTensor → Float bridge). */
    private fun toFloatSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("toFloat"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    private fun IrBuilderWithScope.intConst(v: Int): IrExpression = IrConstImpl(
        startOffset, endOffset, pluginContext.irBuiltIns.intType, IrConstKind.Int, v,
    )

    private fun IrBuilderWithScope.boolConst(v: Boolean): IrExpression = IrConstImpl(
        startOffset, endOffset, pluginContext.irBuiltIns.booleanType, IrConstKind.Boolean, v,
    )

    /**
     * §0.4.197 — Synthesise `param.dims[axis]` as an IR expression. Two-step IR call:
     * (1) `param.dims` (DTensor's val constructor property → property getter call);
     * (2) `IntArray.get(axis)` (primitive operator).
     *
     * Result type is `Int` — feeds `broadcastDimsRankN`'s individual `Int` args.
     */
    private fun IrBuilderWithScope.irParamDimAccess(
        param: IrValueParameter,
        axis: Int,
    ): IrExpression? {
        val dimsGetterSym = dtensorDimsGetter() ?: return null
        val intArrayGetSym = intArrayGetSymbol() ?: return null
        val intArrayType = pluginContext.referenceClass(ClassId.fromString("kotlin/IntArray"))
            ?.defaultType ?: return null
        val dimsCall = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = intArrayType,
            symbol = dimsGetterSym,
        )
        dimsCall.arguments[0] = irGet(param)
        val getCall = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = pluginContext.irBuiltIns.intType,
            symbol = intArrayGetSym,
        )
        getCall.arguments[0] = dimsCall
        getCall.arguments[1] = IrConstImpl(
            startOffset, endOffset,
            pluginContext.irBuiltIns.intType,
            IrConstKind.Int, axis,
        )
        return getCall
    }

    /**
     * §0.4.197 — Resolves the getter for `io.tlaloc.core.DTensor.dims` (a `val`
     * constructor property on `DTensor<S, T>`).
     */
    private fun dtensorDimsGetter(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            classId = ClassId(FqName("io.tlaloc.core"), Name.identifier("DTensor")),
            callableName = Name.identifier("dims"),
        )
        val prop = pluginContext.referenceProperties(callableId).singleOrNull() ?: return null
        return prop.owner.getter?.symbol
    }

    /**
     * §0.4.197 — Resolves `kotlin.IntArray.get(Int): Int` — the primitive operator
     * for `intArr[i]` reads.
     */
    private fun intArrayGetSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            classId = ClassId.fromString("kotlin/IntArray"),
            callableName = Name.identifier("get"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * `OpKind.SQRT(x)` → IrCall to `io.tlaloc.core.sqrt` (the Float / Double extension
     * declared in `:core/DScalar.kt`) for a scalar, or to the `io.tlaloc.core.ops.sqrt`
     * DTensor extension for a tensor.
     *
     * §0.4.390 — the tensor path. It was scalar-only ("the narrow scalar path is
     * sufficient for the D.1b brachistochrone port"), so ANY tensor sqrt in a gradient
     * body fell out of synthesis scope and silently dropped the whole function back to
     * the runtime tape. Training batchNorm's `√(ν+eps)` is the first body to need it;
     * the shape-preserving `DTensor<S, F32>.sqrt()` extension already existed, so this
     * mirrors [irRelu]'s tensor arm rather than adding host surface.
     */
    private fun IrBuilderWithScope.irSqrt(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operandDecl = env[op.operands[0].id] ?: return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            val operandIrType = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
            val operandShapeArg = operandIrType.arguments.firstOrNull()?.typeOrNull ?: return null
            val sym = sqrtTensorSymbol() ?: return null
            val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: operandIrType
            val call = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = resultIrType,
                symbol = sym,
            )
            if (call.typeArguments.isNotEmpty()) {
                call.typeArguments[0] = operandShapeArg
            }
            call.arguments[0] = irGet(operandDecl)
            return call
        }
        if (!op.type.isScalar) return null
        val ty = irTypeFor(op.type, context) ?: return null
        val sym = sqrtSymbolFor(op.type.dtype) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = sym,
        )
        // `Float.sqrt()` / `Double.sqrt()` have a single parameter — the extension
        // receiver — so `arguments[0]` is where the operand lands. `fromSymbolOwner`
        // sizes the list to the callee's parameter count regardless of kind.
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    /**
     * `OpKind.GATHER(arr, idx)` → IrCall to `io.tlaloc.core.ops.get` (the scalar-
     * index-into-rank-1 `operator fun get` declared in HostOps.kt). Result type is
     * scalar Float; uses the operand array's tensor IrType as the extension-
     * receiver type. Narrow rank-1-F32-only path today.
     */
    private fun IrBuilderWithScope.irGather(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val arrDecl = env[op.operands[0].id] ?: return null
        val idxDecl = env[op.operands[1].id] ?: return null
        if (op.operands[0].type.rank != 1 || op.operands[0].type.dtype != F32) return null
        val sym = gatherSymbol() ?: return null
        val resultTy = pluginContext.irBuiltIns.floatType
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultTy,
            symbol = sym,
        )
        // `operator fun <S : Shape> DTensor<S, F32>.get(i: Int): Float`:
        //   arguments[0] = extension receiver (the DTensor)
        //   arguments[1] = regular param `i` (Int)
        // Thread the call-site shape arg into the type-arg slot so the generic
        // `S : Shape` is concrete for the IR verifier — mirror irBroadcast's
        // handling of the same pattern.
        val tensorIrType = context.tensorIrType as? IrSimpleType
        val shapeTypeArg = tensorIrType?.arguments?.firstOrNull()?.typeOrNull
        if (shapeTypeArg != null && call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(arrDecl)
        call.arguments[1] = irGet(idxDecl)
        return call
    }

    /**
     * `OpKind.SCATTER(base, idx, value)` → IrCall to `io.tlaloc.core.ops.scatter`
     * (the non-destructive rank-1 slot-replace helper declared in HostOps.kt).
     * Emitted by the gradient body when GatherRule's adjoint propagates an upstream
     * scalar into a one-hot rank-1 vector (§0.4.41).
     */
    private fun IrBuilderWithScope.irScatter(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        val baseDecl = env[op.operands[0].id] ?: return null
        val idxDecl = env[op.operands[1].id] ?: return null
        val valueDecl = env[op.operands[2].id] ?: return null
        if (op.type.rank != 1 || op.type.dtype != F32) return null
        val sym = scatterSymbol() ?: return null
        val resultTy = irTypeFor(op.type, context) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultTy,
            symbol = sym,
        )
        // `fun <S : Shape> scatter(base: DTensor<S, F32>, i: Int, value: Float)`:
        //   top-level (no receivers), three regular params — arguments[0..2] are
        //   base / idx / value.
        val tensorIrType = context.tensorIrType as? IrSimpleType
        val shapeTypeArg = tensorIrType?.arguments?.firstOrNull()?.typeOrNull
        if (shapeTypeArg != null && call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(baseDecl)
        call.arguments[1] = irGet(idxDecl)
        call.arguments[2] = irGet(valueDecl)
        return call
    }

    /**
     * `OpKind.SCATTER_ADD(base, idx, value)` → IrCall to a `:core/ops` runtime
     * helper. Emits `scatterAddInPlace` (destructive, mutates base's buffer) when
     * the op carries the `"in_place": true` attr set by
     * `DxirReverseTransform.tagSingleUseScatterAdds` (§0.4.46); otherwise emits
     * `scatterAddInto` (non-destructive, copies base). Rank-1 F32 only.
     */
    private fun IrBuilderWithScope.irScatterAdd(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        val baseDecl = env[op.operands[0].id] ?: return null
        val idxDecl = env[op.operands[1].id] ?: return null
        val valueDecl = env[op.operands[2].id] ?: return null
        if (op.type.rank != 1 || op.type.dtype != F32) return null
        val inPlace = op.attrs["in_place"] == true
        val sym = if (inPlace) scatterAddInPlaceSymbol() else scatterAddSymbol()
        sym ?: return null
        val resultTy = irTypeFor(op.type, context) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultTy,
            symbol = sym,
        )
        val tensorIrType = context.tensorIrType as? IrSimpleType
        val shapeTypeArg = tensorIrType?.arguments?.firstOrNull()?.typeOrNull
        if (shapeTypeArg != null && call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(baseDecl)
        call.arguments[1] = irGet(idxDecl)
        call.arguments[2] = irGet(valueDecl)
        return call
    }

    private fun scatterAddSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("scatterAddInto"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    private fun scatterAddInPlaceSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("scatterAddInPlace"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * Resolves `io.tlaloc.core.ops.get` (`operator fun <S : Shape> DTensor<S, F32>.get(i: Int): Float`).
     */
    private fun gatherSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("get"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * Resolves `io.tlaloc.core.ops.scatter` (`fun <S : Shape> scatter(...)`).
     */
    private fun scatterSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("scatter"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * Resolves the rank-1 tensor binary operator (`DTensor.plus` / `minus` / etc.)
     * declared in `:core/ops/HostOps.kt`. Used by [irOpFor]'s rank-1 arms for
     * ADD/SUB/MUL/DIV — the primitive `Float.plus` path only handles scalars,
     * whereas gradAccum's outer accumulation for rank-1 primal params needs
     * elementwise-tensor-ADD.
     */
    private fun findTensorBinaryOp(opName: String): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(opName),
        )
        // §0.4.206 — `:core/ops/times` grew a second overload in that firing:
        // `DTensor<S, F32>.times(other: DTensor<S, F32>)` (elementwise tensor)
        // and `DTensor<S, F32>.times(scalar: Float)` (scalar-multiply). The
        // `singleOrNull()` lookup that worked through §0.4.205 then returned
        // null, so this filters to the tensor-by-tensor overload.
        //
        // Phase A5 — all four ops now also carry the scalar-mixing overloads
        // (`DTensor.plus(Float)` AND the scalar-on-the-left `Float.plus(DTensor)`),
        // so the filter has to name the receiver too: an extension receiver typed
        // `DTensor` plus exactly one regular parameter typed `DTensor`. Checking
        // only the regular parameter matched `Float.minus(DTensor)` as well — its
        // Float is the RECEIVER, not a parameter — and picking that overload
        // synthesized a call whose receiver slot held a DTensor, which the JVM
        // then rejected at runtime (`DTensor cannot be cast to Number`).
        //
        // Phase A5c-2 — and exactly ONE type parameter. The broadcasting overloads
        // (`<S1, S2> DTensor<S1, F32>.plus(other: DTensor<S2, F32>)` in
        // BroadcastOps.kt) satisfy every other clause, but they return
        // `DTensor<Shape, F32>`: the shape-preserving symbol is the one whose single
        // `S` is shared by receiver, parameter and result.
        val dtensorClass = pluginContext.referenceClass(ClassId.fromString("io/tlaloc/core/DTensor"))
        fun isDTensor(type: IrType?) = (type as? IrSimpleType)?.classifier == dtensorClass
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            if (sym.owner.typeParameters.size != 1) return@firstOrNull false
            val regular = params.filter { it.kind == IrParameterKind.Regular }
            if (regular.size != 1) return@firstOrNull false
            val receiver = params.firstOrNull { it.kind == IrParameterKind.ExtensionReceiver }
                ?: return@firstOrNull false
            isDTensor(receiver.type) && isDTensor(regular[0].type)
        }
    }

    /**
     * `OpKind.CAST(x)` → IrCall to the dtype-conversion member (e.g., `x.toFloat()`
     * on an Int operand). The conversion member is resolved against the operand's
     * Kotlin primitive class (`Int.toFloat`, `Long.toDouble`, etc.) via
     * [IrBuiltIns.primitiveIntegralIrTypes] + class-member search. Scalar-only; the
     * emitted IR mirrors what the user would have written at the source level.
     */
    private fun IrBuilderWithScope.irCast(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        // §0.4.364 — tensor CAST Bool→F32 (the comparison surface's mask
        // materialisation) is an identity at runtime: synthesis represents
        // Bool tensors as 0/1 F32 DTensors already ([irCompare] returns the
        // :core mask directly), so the cast forwards the operand's value.
        if (!op.type.isScalar) {
            val src = op.operands[0].type
            if (src.dims != op.type.dims) return null
            if (!((src.dtype == Bool && op.type.dtype == F32) || src.dtype == op.type.dtype)) {
                return null
            }
            val operandDecl = env[op.operands[0].id] ?: return null
            return irGet(operandDecl)
        }
        if (!op.operands[0].type.isScalar) return null
        val operandDecl = env[op.operands[0].id] ?: return null
        val srcDtype = op.operands[0].type.dtype
        val dstDtype = op.type.dtype
        if (srcDtype == dstDtype) return irGet(operandDecl)  // identity cast; rare but cheap
        val sym = castMemberSymbol(srcDtype, dstDtype) ?: return null
        val resultTy = irTypeFor(op.type, context) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultTy,
            symbol = sym,
        )
        // `Int.toFloat()` etc. have exactly one parameter — the dispatch receiver.
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    /**
     * §0.4.364 — `OpKind.COMPARE(a, b)` with a `direction` attr → IrCall to the
     * matching `:core/ops` comparison extension (`gt`/`ge`/`lt`/`le`/`eq`/`ne`,
     * HostOps.kt). The IR-level result is Bool; the runtime value is the 0/1 F32
     * mask those extensions return — the synthesis-wide convention for Bool
     * tensors (see [irCast]'s tensor arm and [irWhere]).
     */
    private fun IrBuilderWithScope.irCompare(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val fnName = when (op.attrs["direction"] as? String) {
            "GT" -> "gt"; "GE" -> "ge"; "LT" -> "lt"
            "LE" -> "le"; "EQ" -> "eq"; "NE" -> "ne"
            else -> return null
        }
        if (!isAcceptedTensorType(op.operands[0].type)) return null
        val lhsDecl = env[op.operands[0].id] ?: return null
        val rhsDecl = env[op.operands[1].id] ?: return null
        val lhsIrType = irTypeForNode(op.operands[0], context) as? IrSimpleType ?: return null
        val shapeArg = lhsIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        // §0.4.397 — the comparisons gained Float-scalar overloads (`a gt 1.0f`,
        // Phase A5c-3(iv)), so the uniquely-named `coreOpsSymbol` lookup would
        // return null. The FIR arm splats a scalar side before the IR ever sees
        // it, so synthesis always wants the tensor⊗tensor overload — exactly what
        // [findTensorBinaryOp]'s filter (DTensor receiver, one regular DTensor
        // param, one type parameter) selects.
        val sym = findTensorBinaryOp(fnName) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = lhsIrType,
            symbol = sym,
        )
        // `infix fun <S : Shape> DTensor<S, F32>.gt(other): DTensor<S, F32>` —
        // extension receiver at arguments[0], one regular param at [1].
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeArg
        }
        call.arguments[0] = irGet(lhsDecl)
        call.arguments[1] = irGet(rhsDecl)
        return call
    }

    /**
     * §0.4.364 — `OpKind.WHERE(pred, a, b)` → IrCall to
     * `io.tlaloc.core.ops.where` (top-level, HostOps.kt). Emitted both by the
     * forward lowering of user `where(...)` calls and by WhereRule's adjoint
     * (which routes the upstream through the same mask). `pred`'s runtime value
     * is the F32 mask per the Bool-as-F32 synthesis convention.
     */
    private fun IrBuilderWithScope.irWhere(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 3) return null
        if (!isAcceptedTensorType(op.type)) return null
        val predDecl = env[op.operands[0].id] ?: return null
        val aDecl = env[op.operands[1].id] ?: return null
        val bDecl = env[op.operands[2].id] ?: return null
        val aIrType = (irTypeForNode(op.operands[1], context) as? IrSimpleType)
            ?: (irTypeFor(op.type, context) as? IrSimpleType)
            ?: return null
        val shapeArg = aIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val sym = coreOpsSymbol("where") ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = aIrType,
            symbol = sym,
        )
        // `fun <S : Shape> where(pred, a, b)`: top-level, three regular params.
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeArg
        }
        call.arguments[0] = irGet(predDecl)
        call.arguments[1] = irGet(aDecl)
        call.arguments[2] = irGet(bDecl)
        return call
    }

    /** Resolves a uniquely-named top-level/extension function in `io.tlaloc.core.ops`. */
    private fun coreOpsSymbol(name: String): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * Resolves the Kotlin-primitive conversion member `<src>.to<Dst>()` (e.g.,
     * `Int.toFloat`, `Long.toDouble`). Returns null for unsupported dtype pairs
     * (Bool conversions, non-scalar, etc.).
     */
    private fun castMemberSymbol(src: DType, dst: DType): IrSimpleFunctionSymbol? {
        val srcCls = classSymbolFor(src) ?: return null
        val dstName = when (dst) {
            F32 -> "toFloat"
            F64 -> "toDouble"
            I32 -> "toInt"
            I64 -> "toLong"
            Bool -> return null
        }
        return srcCls.owner.declarations
            .asSequence()
            .filterIsInstance<IrSimpleFunction>()
            .firstOrNull { candidate ->
                if (candidate.name.asString() != dstName) return@firstOrNull false
                val regulars = candidate.parameters.filter { it.kind == IrParameterKind.Regular }
                val hasDispatch = candidate.parameters.any { it.kind == IrParameterKind.DispatchReceiver }
                hasDispatch && regulars.isEmpty()
            }?.symbol
    }

    /**
     * Resolves the `io.tlaloc.core.sqrt` overload matching [dtype]. Returns null for
     * non-F32/F64 dtypes (I32/I64/Bool sqrt is meaningless). The extension is declared
     * in `:core/DScalar.kt` as `fun Float.sqrt()` / `fun Double.sqrt()` (+ DScalar
     * variants we don't match here — those aren't reachable from the scalar Float/
     * Double DxirType path).
     */
    private fun sqrtSymbolFor(dtype: DType): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core"),
            callableName = Name.identifier("sqrt"),
        )
        val targetType = when (dtype) {
            F32 -> pluginContext.irBuiltIns.floatType
            F64 -> pluginContext.irBuiltIns.doubleType
            else -> return null
        }
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            params.size == 1 && params[0].type == targetType
        }
    }

    /**
     * §0.4.52 — `OpKind.POW(base, exp)` → IrCall to `kotlin.math.pow` (the
     * `Float.pow(Float): Float` / `Double.pow(Double): Double` extension). Emitted by
     * C6's closed-form geometric-sum lowering (`a^n`), which previously could not be
     * synthesised — the natural BGDHyperOpt kernel's coarsening went through SCT's
     * scalar-primitive synthesis check, rejected on POW, and fell back to runtime
     * tape. Scalar-only (F32 / F64); rank-1 POW would need the tensor extension.
     */
    private fun IrBuilderWithScope.irPow(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        if (!op.type.isScalar) return null
        val baseDecl = env[op.operands[0].id] ?: return null
        val expDecl = env[op.operands[1].id] ?: return null
        val ty = irTypeFor(op.type, context) ?: return null
        val sym = powSymbolFor(op.type.dtype) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = sym,
        )
        // `Float.pow(x: Float)` / `Double.pow(x: Double)`: extension-receiver base +
        // single value param for exponent.
        call.arguments[0] = irGet(baseDecl)
        call.arguments[1] = irGet(expDecl)
        return call
    }

    private fun powSymbolFor(dtype: DType): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("kotlin.math"),
            callableName = Name.identifier("pow"),
        )
        val targetType = when (dtype) {
            F32 -> pluginContext.irBuiltIns.floatType
            F64 -> pluginContext.irBuiltIns.doubleType
            else -> return null
        }
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            params.size == 2 && params[0].type == targetType && params[1].type == targetType
        }
    }

    /**
     * §0.4.53 — `OpKind.LOG(x)` → `kotlin.math.ln(x)`. Emitted by C6's closed-form
     * differentiation wrt a symbolic trip count: `d/dn (a^n) = a^n · ln(a)`. No LOG
     * support in synthesis previously because BGDHyperOpt's gradient was wrt the
     * hyperparameter (r, inside `a`) not wrt the iteration count; symbolic T changes
     * that. Scalar-only (F32 / F64).
     */
    private fun IrBuilderWithScope.irLog(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        // §0.4.368 — tensor LOG dispatches to `:core/ops/log` (like irTanh);
        // logSoftmax's grad body keeps a rank-2 LOG (`dw = log(softmax(x))`).
        if (op.operands.size == 1 &&
            isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)
        ) {
            return tensorUnaryCall(op, env, context, opsTensorSymbol("log"))
        }
        return irUnaryMathCall(op, env, context, Name.identifier("ln"))
    }

    /**
     * §0.4.53 — `OpKind.EXP(x)` → `kotlin.math.exp(x)`. Mirrors [irLog]; kept here so
     * any future C6/C7 closed form that differentiates into an `exp` term has a
     * synthesis path. §0.4.368 — tensor EXP dispatches to `:core/ops/exp`.
     */
    private fun IrBuilderWithScope.irExp(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size == 1 &&
            isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)
        ) {
            return tensorUnaryCall(op, env, context, opsTensorSymbol("exp"))
        }
        return irUnaryMathCall(op, env, context, Name.identifier("exp"))
    }

    /**
     * §0.4.368 — resolves a single-overload `:core/ops` callable by name: the tensor
     * unary extensions (`tanh`, `sigmoid`, `log`, `neg`, …) and, since Phase A5c-2,
     * the broadcasting binaries (`plusBroadcast`, `timesBroadcast`, …), which are
     * top-level and uniquely named so `singleOrNull()` holds for both.
     */
    private fun opsTensorSymbol(name: String): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier(name),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.166 — `OpKind.SIN(x)` → `kotlin.math.sin(x)`. CartPole Phase 0a primitive.
     * Scalar-only (F32 / F64). SinRule's adjoint emits `MUL(upstream, COS(x))`,
     * which routes through this synthesis arm + irCos for the COS.
     */
    private fun IrBuilderWithScope.irSin(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? = irUnaryMathCall(op, env, context, Name.identifier("sin"))

    /** §0.4.166 — `OpKind.COS(x)` → `kotlin.math.cos(x)`. Companion to [irSin]. */
    private fun IrBuilderWithScope.irCos(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? = irUnaryMathCall(op, env, context, Name.identifier("cos"))

    /**
     * §0.4.395 — `OpKind.TAN(x)`: tensor operands dispatch to `:core/ops/tan`
     * (the [irLog]/[irExp] pattern — TanRule's adjoint keeps a same-rank TAN
     * recompute in the gradient body), scalars to `kotlin.math.tan`.
     */
    private fun IrBuilderWithScope.irTan(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size == 1 &&
            isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)
        ) {
            return tensorUnaryCall(op, env, context, opsTensorSymbol("tan"))
        }
        return irUnaryMathCall(op, env, context, Name.identifier("tan"))
    }

    /**
     * §0.4.402 — Phase C1 special functions: `OpKind.LGAMMA` / `DIGAMMA` /
     * `TRIGAMMA`. Tensor operands dispatch to the `:core/ops` extension of the
     * same [name]; scalars to the `io.tlaloc.core` five-overload extension via
     * [irCoreScalarCall] (the §0.4.377 sigmoid path — none of these has a
     * `kotlin.math` equivalent). TRIGAMMA reaches here from gradient bodies
     * (DIGAMMA's adjoint/tangent emit it) and, since §0.4.405, from the user's
     * `polygamma(1)` spelling, which the FIR normalises to the TRIGAMMA op.
     */
    private fun IrBuilderWithScope.irSpecialUnary(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
        name: String,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            return tensorUnaryCall(op, env, context, opsTensorSymbol(name))
        }
        return irCoreScalarCall(op, env, context, name)
    }

    /**
     * §0.4.405 — `OpKind.POLYGAMMA(x)` with its literal `order` attr: the
     * [irSpecialUnary] shape plus one trailing Int const argument. Tensor
     * operands dispatch to `:core/ops/polygamma(n)`, scalars to the
     * `io.tlaloc.core.polygamma(n)` extension (both are single positional-Int
     * ops with no defaults — the K2 named-arg landmine). Reaches here both
     * from user `polygamma(n ≥ 2)` bodies and from gradient bodies (TRIGAMMA's
     * adjoint emits POLYGAMMA(2), POLYGAMMA(n)'s emits POLYGAMMA(n+1)).
     */
    private fun IrBuilderWithScope.irPolygamma(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val order = (op.attrs["order"] as? Number)?.toInt() ?: return null
        if (isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)) {
            val call = tensorUnaryCall(op, env, context, opsTensorSymbol("polygamma"))
                as? IrCallImpl ?: return null
            call.arguments[1] = intConst(order)
            return call
        }
        if (!op.type.isScalar) return null
        val operandDecl = env[op.operands[0].id] ?: return null
        val ty = irTypeFor(op.type, context) ?: return null
        val sym = coreScalarIntArgSymbolFor(op.type.dtype, Name.identifier("polygamma")) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = sym,
        )
        call.arguments[0] = irGet(operandDecl)
        call.arguments[1] = intConst(order)
        return call
    }

    /**
     * §0.4.405 — the `(receiver, Int)` sibling of [coreScalarSymbolFor]:
     * resolves the `io.tlaloc.core` scalar extension of [callable] whose
     * extension receiver matches the op's primitive dtype and whose single
     * regular parameter is `Int` (the polygamma order).
     */
    private fun coreScalarIntArgSymbolFor(dtype: DType, callable: Name): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core"),
            callableName = callable,
        )
        val targetType = when (dtype) {
            F32 -> pluginContext.irBuiltIns.floatType
            F64 -> pluginContext.irBuiltIns.doubleType
            else -> return null
        }
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            params.size == 2 &&
                params[0].kind == IrParameterKind.ExtensionReceiver &&
                params[0].type == targetType &&
                params[1].type == pluginContext.irBuiltIns.intType
        }
    }

    /** §0.4.395 — `OpKind.ATAN(x)`. Companion to [irTan]; `kotlin.math.atan` exists. */
    private fun IrBuilderWithScope.irAtan(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size == 1 &&
            isAcceptedTensorType(op.type) && isAcceptedTensorType(op.operands[0].type)
        ) {
            return tensorUnaryCall(op, env, context, opsTensorSymbol("atan"))
        }
        return irUnaryMathCall(op, env, context, Name.identifier("atan"))
    }

    /**
     * §0.4.167 — `OpKind.ABS(x)` → `kotlin.math.abs(x)`. CartPole Phase 0a-2 primitive.
     * Scalar-only (F32 / F64). AbsRule's adjoint emits `STEP(x) - STEP(-x)` — STEP
     * is synthesised separately via [irStep]; no special handling needed here.
     */
    private fun IrBuilderWithScope.irAbs(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? = irUnaryMathCall(op, env, context, Name.identifier("abs"))

    private fun IrBuilderWithScope.irUnaryMathCall(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
        callable: Name,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (!op.type.isScalar) return null
        val operandDecl = env[op.operands[0].id] ?: return null
        val ty = irTypeFor(op.type, context) ?: return null
        val sym = mathModuleSymbolFor(op.type.dtype, callable) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = sym,
        )
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    private fun mathModuleSymbolFor(dtype: DType, callable: Name): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("kotlin.math"),
            callableName = callable,
        )
        val targetType = when (dtype) {
            F32 -> pluginContext.irBuiltIns.floatType
            F64 -> pluginContext.irBuiltIns.doubleType
            else -> return null
        }
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            params.size == 1 && params[0].type == targetType
        }
    }

    /**
     * Phase A5b — the `io.tlaloc.core` sibling of [irUnaryMathCall]: an IrCall to a
     * receiver-only scalar extension declared in `:core/DScalar.kt`
     * (`Float.sigmoid()` / `Double.sigmoid()`) for a scalar-typed op. Needed for the
     * ops with no `kotlin.math` equivalent; the five-overload set each `:core` scalar
     * entry declares (Float, Double, FloatScalar, DoubleScalar, DScalar) is narrowed
     * by [coreScalarSymbolFor] to the one whose extension receiver is the op's
     * primitive dtype.
     */
    private fun IrBuilderWithScope.irCoreScalarCall(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
        name: String,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (!op.type.isScalar) return null
        val operandDecl = env[op.operands[0].id] ?: return null
        val ty = irTypeFor(op.type, context) ?: return null
        val sym = coreScalarSymbolFor(op.type.dtype, Name.identifier(name)) ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = ty,
            symbol = sym,
        )
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    private fun coreScalarSymbolFor(dtype: DType, callable: Name): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core"),
            callableName = callable,
        )
        val targetType = when (dtype) {
            F32 -> pluginContext.irBuiltIns.floatType
            F64 -> pluginContext.irBuiltIns.doubleType
            else -> return null
        }
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            val params = sym.owner.parameters
            params.size == 1 &&
                params[0].kind == IrParameterKind.ExtensionReceiver &&
                params[0].type == targetType
        }
    }

    private fun IrBuilderWithScope.greaterThanZero(
        operandDecl: IrValueDeclaration,
        operandType: DxirType,
        context: SynthesisContext,
    ): IrExpression? {
        val classSym = classSymbolFor(operandType.dtype) ?: return null
        val greaterSym = pluginContext.irBuiltIns.greaterFunByOperandType[classSym] ?: return null
        val zeroLhs = zeroOrOneConst(operandType, one = false, context) ?: return null
        val cmp = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = pluginContext.irBuiltIns.booleanType,
            symbol = greaterSym,
        )
        // greater* is a top-level primitive with no dispatch receiver — arguments are
        // both regular, at indices 0 and 1.
        cmp.arguments[0] = irGet(operandDecl)
        cmp.arguments[1] = zeroLhs
        return cmp
    }

    private fun IrBuilderWithScope.zeroOrOneConst(type: DxirType, one: Boolean, context: SynthesisContext): IrExpression? {
        val ty = irTypeFor(type, context) ?: return null
        return when (type.dtype) {
            F32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Float, if (one) 1.0f else 0.0f)
            F64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Double, if (one) 1.0 else 0.0)
            I32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Int, if (one) 1 else 0)
            I64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Long, if (one) 1L else 0L)
            Bool -> null
        }
    }

    private fun irTypeFor(type: DxirType, context: SynthesisContext): IrType? {
        if (type.isScalar) {
            return when (type.dtype) {
                F32 -> pluginContext.irBuiltIns.floatType
                F64 -> pluginContext.irBuiltIns.doubleType
                I32 -> pluginContext.irBuiltIns.intType
                I64 -> pluginContext.irBuiltIns.longType
                // §0.4.24 — Bool-scalar nodes materialise as Kotlin `Boolean` locals so
                // the IF-predicate path (STEP result feeding `OpKind.IF.operands[0]`)
                // lowers as a plain if/else. No Bool consts are emitted today; if a
                // future rule needs them, [irConstFor] / [zeroOrOneConst] must grow too.
                Bool -> pluginContext.irBuiltIns.booleanType
            }
        }
        // Rank-1 F32: use the call-site-harvested IrType (preserves the source-level shape
        // witness + any param/return type-arg machinery). §0.4.186 — rank-2/3 F32 also
        // route through the same call-site IrType: `broadcastLike<S>(v, template)` is
        // generic over `S : Shape`, so passing the rank-2/3 template + its IrType
        // produces a correctly-typed rank-2/3 result without any per-rank synthesis
        // surface widening. Other shapes / dtypes are out of scope — callers receive
        // `null` and fall back to the runtime tape path.
        if (isAcceptedTensorType(type)) return context.tensorIrType
        return null
    }

    private fun isRank1F32(type: DxirType): Boolean = type.rank == 1 && type.dtype == F32

    /**
     * §0.4.186 — Phase 0c slice (b). Widens the synthesis-side acceptance from "rank-1
     * F32 only" to "rank-1, rank-2, or rank-3 F32" so that gradient bodies for primals
     * with rank-2/3 inputs can route through the existing `broadcastLike` helper.
     *
     * §0.4.384 — Phase A3b slice 1 widens once more, to rank 4: the NCHW conv/pool
     * tensors. Rank 4 is safe to admit blanket-wide (rather than only for the conv
     * kinds) because every rank-dispatching arm resolves a `…RankN` host delegate by
     * name and returns null when it has no rank-4 entry — a missing delegate rejects
     * the function and falls back to the runtime tape, exactly as before, so the widen
     * cannot turn a rejection into wrong code. The `:core/ops` host ops that take a
     * generic `S : Shape` (`broadcastLike`, `stretchLike`, the elementwise binaries,
     * `sumToLike`, `padToLike`) read their runtime `dims` and are rank-agnostic
     * already. Rank 5+ and non-F32 dtypes still fall back.
     */
    private fun isAcceptedTensorType(type: DxirType): Boolean =
        type.dtype == F32 && type.rank in 1..4

    /**
     * §0.4.400 — the integer INDEX tensor scope: `embedding`'s rank-1 I32 index
     * vector, flowing through a `grad {}` lambda as a non-differentiable param.
     * Kept deliberately narrow (I32 rank-1, the only shape the host `embedding`
     * accepts) — the general integer-tensor synthesis story stays out of scope.
     */
    private fun isAcceptedIndexTensorType(type: DxirType): Boolean =
        type.dtype == I32 && type.rank == 1

    /**
     * Resolves `io.tlaloc.core.ops.broadcastLike` — the top-level extension function that
     * lowers [OpKind.BROADCAST] at the synthesis layer. Uses [CallableId] lookup
     * (top-level, no classId) and takes the single overload as of §0.4.10; if the helper
     * ever grows an overload set, the type-arg would need to be checked here.
     */
    private fun broadcastLikeSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("broadcastLike"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    /**
     * §0.4.189 — `OpKind.TRANSPOSE(a)` → IrCall to `io.tlaloc.core.ops.transpose`
     * (the Rank2 extension declared in HostOps.kt). Used by `MatmulRule`'s
     * gradient emission for the dA = upstream · B^T and dB = A^T · upstream chain.
     *
     * §0.4.193 — Phase 0c-rectangular slice 2: type-args now read from the operand's
     * specific IrType via [irTypeForNode] rather than the call-site `tensorIrType`.
     * For square-matrix surfaces the two are identical (one tensor IrType across all
     * operands); for multi-param surfaces with per-operand IrTypes populated in
     * `operandIrTypes`, `irTypeForNode` returns the operand-specific type. Slice 3
     * still needs op-result IrType derivation (TRANSPOSE: input `Rank2<R, C>` →
     * output `Rank2<C, R>`) before non-param operands (e.g., a TRANSPOSE feeding
     * a downstream op) can resolve correctly.
     */
    /**
     * §0.4.396 — `OpKind.REVERSE` → `flipAxes{N}(x, a0…)` (Phase C3). The
     * `dimensions` attr is a compile-time user literal, so each axis rides as
     * a positional Int const into the fixed-arity delegate selected by axis
     * count — the `transposePerm{N}` pattern. Shape-preserving: the result
     * IrType is the operand's own (both IrType solvers propagate it), and the
     * delegate's `S` type-arg is the operand's shape argument.
     */
    private fun IrBuilderWithScope.irReverse(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (operand.type.dtype != F32 || op.type.dtype != F32) return null
        if (!isAcceptedTensorType(operand.type)) return null
        val operandDecl = env[operand.id] ?: return null
        val axes = (op.attrs["dimensions"] as? List<*>)?.map { (it as Number).toInt() }
            ?: return null
        if (axes.isEmpty() || axes.toSet().size != axes.size) return null
        if (axes.any { it !in 0 until operand.type.rank }) return null
        if (op.type.rank != operand.type.rank) return null
        val sym = flipAxesSymbol(axes.size) ?: return null
        val opIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
        val shapeTypeArg = opIrType.arguments.firstOrNull()?.typeOrNull ?: return null
        val resultIrType = irTypeForNode(op, context) ?: context.tensorIrType ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.isNotEmpty()) {
            call.typeArguments[0] = shapeTypeArg
        }
        call.arguments[0] = irGet(operandDecl)
        for ((i, a) in axes.withIndex()) {
            call.arguments[i + 1] = intConst(a)
        }
        return call
    }

    private fun IrBuilderWithScope.irTranspose(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (operand.type.dtype != F32 || op.type.dtype != F32) return null
        val operandDecl = env[operand.id] ?: return null
        // §0.4.367 — non-swap spellings (Phase A2a). Identity perms and rank-1
        // transposes forward the operand; rank-2/3 general perms call the
        // fixed-arity `transposePermN` with the attr's compile-time constants.
        // The rank-2 swap (perm [1,0] or the attr-less MatmulRule emission)
        // keeps the precisely-typed `.transpose()` path below.
        val perm = (op.attrs["permutation"] as? List<*>)?.map { (it as Number).toInt() }
        if (!isAcceptedTensorType(operand.type)) return null
        if (operand.type.rank == 1 || (perm != null && perm == perm.indices.toList())) {
            return irGet(operandDecl)
        }
        if (perm != null && perm != listOf(1, 0)) {
            if (perm.size != operand.type.rank || op.type.rank != operand.type.rank) return null
            val permSym = transposePermSymbol(perm.size) ?: return null
            val opIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
            val shapeTypeArg = opIrType.arguments.firstOrNull()?.typeOrNull ?: return null
            val resultIrType = irTypeForNode(op, context) ?: context.tensorIrType ?: return null
            val permCall = IrCallImpl.fromSymbolOwner(
                startOffset = startOffset,
                endOffset = endOffset,
                type = resultIrType,
                symbol = permSym,
            )
            if (permCall.typeArguments.isNotEmpty()) {
                permCall.typeArguments[0] = shapeTypeArg
            }
            permCall.arguments[0] = irGet(operandDecl)
            for ((i, p) in perm.withIndex()) {
                permCall.arguments[i + 1] = intConst(p)
            }
            return permCall
        }
        if (operand.type.rank != 2 || op.type.rank != 2) return null
        val sym = transposeSymbol() ?: return null
        val operandIrType = irTypeForNode(operand, context) as? IrSimpleType ?: return null
        // §0.4.196 — slice 3b-2a: dig into the operand's outer DTensor → inner Rank2
        // → atomic shape atom args. typeArguments[0] (R) ← operand's inner-Rank2.first;
        // typeArguments[1] (C) ← operand's inner-Rank2.second. Pre-§0.4.196 set both
        // to the WHOLE Rank2<…> IrType — bound-violating (transpose's R/C are typed
        // `ShapeAtom`, not `Shape`) but tolerated by the IR verifier under erasure.
        // Atomic atoms make the substituted return type structurally consistent
        // with the dxir-derived result IrType (slice 3a's `deriveTransposedDTensor`).
        val operandInnerRank2 = (operandIrType.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: return null
        if (operandInnerRank2.arguments.size != 2) return null
        val operandRowAtom = operandInnerRank2.arguments[0].typeOrNull ?: return null
        val operandColAtom = operandInnerRank2.arguments[1].typeOrNull ?: return null
        // §0.4.194 — slice 3a: result IrType derived in `synthesise()`'s body walk via
        // [deriveResultIrType] / [deriveTransposedDTensor] (output Rank2's args are
        // swapped). Falls back to `context.tensorIrType` for ops the derivation
        // skipped (none today, but keeps the helper robust).
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: context.tensorIrType ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        // `fun <R, C> DTensor<Rank2<R, C>, F32>.transpose(): DTensor<Rank2<C, R>, F32>`:
        // arguments[0] = extension receiver. typeArguments[0] = R (operand row atom),
        // typeArguments[1] = C (operand col atom). After substitution, the return type
        // is `DTensor<Rank2<C, R>, F32>` — matches `deriveTransposedDTensor`'s output.
        if (call.typeArguments.size >= 2) {
            call.typeArguments[0] = operandRowAtom
            call.typeArguments[1] = operandColAtom
        }
        call.arguments[0] = irGet(operandDecl)
        return call
    }

    /**
     * §0.4.189 — `OpKind.MATMUL(a, b)` → IrCall to `io.tlaloc.core.ops.matmul`
     * (the Rank2 × Rank2 → Rank2 infix declared in HostOps.kt).
     *
     * §0.4.193 — Phase 0c-rectangular slice 2: per-operand type-arg reading. LHS
     * type-arg now derives from operand[0]'s IrType (via [irTypeForNode]); RHS
     * type-arg derives from operand[1]'s. For square surfaces both are identical
     * to `tensorIrType`. For rectangular surfaces with per-param `operandIrTypes`
     * populated, LHS and RHS pick up the correct distinct shape args.
     *
     * Result type still uses `tensorIrType`; slice 3 will derive the output shape
     * (combining LHS first type-arg + RHS last) so the generated MATMUL's call
     * type matches the dxir-level rank-2 output. Until then non-param operands
     * (TRANSPOSE results, BROADCAST results) fall back to `tensorIrType` and the
     * rectangular surface still requires the BROADCAST template gap to close
     * before it can ship end-to-end.
     */
    private fun IrBuilderWithScope.irMatmul(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 2) return null
        val lhs = op.operands[0]
        val rhs = op.operands[1]
        if (lhs.type.rank != 2 || lhs.type.dtype != F32) return null
        if (rhs.type.rank != 2 || rhs.type.dtype != F32) return null
        if (op.type.rank != 2 || op.type.dtype != F32) return null
        val lhsDecl = env[lhs.id] ?: return null
        val rhsDecl = env[rhs.id] ?: return null
        val sym = matmulSymbol() ?: return null
        val lhsIrType = irTypeForNode(lhs, context) as? IrSimpleType ?: return null
        val rhsIrType = irTypeForNode(rhs, context) as? IrSimpleType ?: return null
        // §0.4.196 — slice 3b-2a: dig into operand DTensor → inner Rank2 → atomic
        // shape atoms. matmul's signature is `<R, K, C: ShapeAtom>`. typeArgs:
        //   [0] = R (LHS row atom)
        //   [1] = K (LHS col atom = RHS row atom — both must agree; we pick LHS's
        //          for the IR call, with the IR verifier expected to substitute
        //          consistently because the dxir guarantees lhs.col == rhs.row)
        //   [2] = C (RHS col atom)
        // Pre-§0.4.196 set all three to the WHOLE Rank2<…> IrType — bound-violating
        // (R/K/C are typed `ShapeAtom`, not `Shape`) but tolerated by the IR verifier
        // under generic erasure. Atomic atoms make the substituted return type
        // structurally consistent with the dxir-derived result IrType (slice 3a's
        // `deriveMatmulOutputDTensor`).
        val lhsInnerRank2 = (lhsIrType.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: return null
        val rhsInnerRank2 = (rhsIrType.arguments.firstOrNull()?.typeOrNull as? IrSimpleType) ?: return null
        if (lhsInnerRank2.arguments.size != 2 || rhsInnerRank2.arguments.size != 2) return null
        val rAtom = lhsInnerRank2.arguments[0].typeOrNull ?: return null
        val kAtom = lhsInnerRank2.arguments[1].typeOrNull ?: return null
        val cAtom = rhsInnerRank2.arguments[1].typeOrNull ?: return null
        // §0.4.194 — slice 3a: result IrType derived in `synthesise()`'s body walk via
        // [deriveMatmulOutputDTensor] (output Rank2 = LHS first ⊕ RHS last). For
        // square the derived result is structurally identical to `tensorIrType`;
        // for rectangular it carries the correct combined shape so downstream ops
        // (in slice 3b) can resolve their operand IrTypes.
        val resultIrType = (irTypeForNode(op, context) as? IrSimpleType) ?: context.tensorIrType ?: return null
        val call = IrCallImpl.fromSymbolOwner(
            startOffset = startOffset,
            endOffset = endOffset,
            type = resultIrType,
            symbol = sym,
        )
        if (call.typeArguments.size >= 3) {
            call.typeArguments[0] = rAtom
            call.typeArguments[1] = kAtom
            call.typeArguments[2] = cAtom
        }
        call.arguments[0] = irGet(lhsDecl)
        call.arguments[1] = irGet(rhsDecl)
        return call
    }

    private fun transposeSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("transpose"),
        )
        // §0.4.367 — `transpose` gained the vararg-perm overload; the classic
        // rank-2 swap path wants the no-arg extension (zero Regular params).
        return pluginContext.referenceFunctions(callableId).firstOrNull { sym ->
            sym.owner.parameters.none { it.kind == IrParameterKind.Regular }
        }
    }

    private fun matmulSymbol(): IrSimpleFunctionSymbol? {
        val callableId = CallableId(
            packageName = FqName("io.tlaloc.core.ops"),
            callableName = Name.identifier("matmul"),
        )
        return pluginContext.referenceFunctions(callableId).singleOrNull()
    }

    private fun classSymbolFor(dtype: DType) = when (dtype) {
        F32 -> pluginContext.irBuiltIns.floatClass
        F64 -> pluginContext.irBuiltIns.doubleClass
        I32 -> pluginContext.irBuiltIns.intClass
        I64 -> pluginContext.irBuiltIns.longClass
        Bool -> null
    }

    private fun findBinaryOp(opName: String, dxirType: DxirType, context: SynthesisContext): IrSimpleFunctionSymbol? {
        val cls = (classSymbolFor(dxirType.dtype) ?: return null).owner
        val selfType = irTypeFor(dxirType, context) ?: return null
        return cls.declarations
            .asSequence()
            .filterIsInstance<IrSimpleFunction>()
            .firstOrNull { candidate ->
                if (candidate.name.asString() != opName) return@firstOrNull false
                val regulars = candidate.parameters.filter { it.kind == IrParameterKind.Regular }
                val hasDispatch = candidate.parameters.any { it.kind == IrParameterKind.DispatchReceiver }
                hasDispatch && regulars.size == 1 && regulars[0].type == selfType
            }?.symbol
    }

    private fun findUnaryOp(opName: String, dxirType: DxirType, @Suppress("UNUSED_PARAMETER") context: SynthesisContext): IrSimpleFunctionSymbol? {
        val cls = (classSymbolFor(dxirType.dtype) ?: return null).owner
        return cls.declarations
            .asSequence()
            .filterIsInstance<IrSimpleFunction>()
            .firstOrNull { candidate ->
                if (candidate.name.asString() != opName) return@firstOrNull false
                val regulars = candidate.parameters.filter { it.kind == IrParameterKind.Regular }
                val hasDispatch = candidate.parameters.any { it.kind == IrParameterKind.DispatchReceiver }
                hasDispatch && regulars.isEmpty()
            }?.symbol
    }

    private fun pairClass(): IrClassSymbol? =
        pluginContext.referenceClass(ClassId.fromString("kotlin/Pair"))

    private fun tripleClass(): IrClassSymbol? =
        pluginContext.referenceClass(ClassId.fromString("kotlin/Triple"))

    /**
     * §0.4.203 — `io.tlaloc.autograd.Quadruple<A, B, C, D>` (introduced §0.4.134
     * for `valueAndGrad3`'s `(value, dA, dB, dC)` return). Used by
     * [synthesise] when `fn.returns.size == 4` — typically a 4-grad-param surface
     * (CartPole's full NN with X + W1 + W2 + W3) or a `valueAndGrad3` that
     * synthesises through the plugin path.
     */
    private fun quadrupleClass(): IrClassSymbol? =
        pluginContext.referenceClass(ClassId.fromString("io/tlaloc/autograd/Quadruple"))

    private fun pairConstructor(): IrConstructorSymbol? =
        pluginContext.referenceConstructors(ClassId.fromString("kotlin/Pair")).singleOrNull()

    private fun tripleConstructor(): IrConstructorSymbol? =
        pluginContext.referenceConstructors(ClassId.fromString("kotlin/Triple")).singleOrNull()

    private fun quadrupleConstructor(): IrConstructorSymbol? =
        pluginContext.referenceConstructors(ClassId.fromString("io/tlaloc/autograd/Quadruple")).singleOrNull()

    @Suppress("unused")
    private val dummyParam: DxirParam? = null
}
