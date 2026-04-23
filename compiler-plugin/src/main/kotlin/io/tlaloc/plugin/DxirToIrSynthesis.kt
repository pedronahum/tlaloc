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
     */
    private data class SynthesisContext(
        val tensorIrType: IrType?,
        val tensorTemplateParam: IrValueParameter?,
    )

    fun synthesise(
        fn: DxirFunction,
        originalCall: IrCall,
        parent: IrDeclarationParent,
    ): IrFunctionExpression? {
        // Harvest the rank-1 DTensor IrType from the call site if any DxirParam is rank-1.
        // The call's type is `FunctionN<P0, …, Pn-1, R>` — the first rank-1 F32 DxirParam's
        // IrType matches `transformed.type.arguments[paramIdx].typeOrNull`. We only support
        // one distinct tensor IrType per function today (the single-rank-1-param case
        // covered by §0.4.10). Multiple tensor shapes would require a per-node map.
        val firstTensorParamIdx = fn.params.indexOfFirst { isRank1F32(it.type) }
        val tensorIrType: IrType? = if (firstTensorParamIdx < 0) null else run {
            val callType = originalCall.type as? IrSimpleType ?: return null
            callType.arguments.getOrNull(firstTensorParamIdx)?.typeOrNull ?: return null
        }
        // Reject any node whose rank is >1 or whose rank-1 dtype isn't F32. Rank-1 F32
        // and scalars are the full supported surface as of §0.4.10.
        val nodeTypes: Sequence<DxirType> = sequence {
            fn.params.forEach { yield(it.type) }
            fn.body.forEach { yield(it.type) }
        }
        for (t in nodeTypes) {
            if (!t.isScalar && !isRank1F32(t)) return null
        }

        val context = SynthesisContext(tensorIrType = tensorIrType, tensorTemplateParam = null)
        val paramIrTypes = fn.params.map { irTypeFor(it.type, context) ?: return null }
        if (fn.returns.isEmpty() || fn.returns.size > 3) return null
        val returnIrTypes = fn.returns.map { irTypeFor(it.type, context) ?: return null }

        // N = 1 → scalar lambda returning R.  N ∈ {2, 3} → lambda returning Pair<…> /
        // Triple<…>, matching the surface signatures of grad2 / valueAndGrad /
        // valueAndGrad2.  Boxing failure (missing kotlin.Pair / Triple symbol lookup)
        // falls back to `null` so the IR extension keeps the original call.
        val boxedReturnType: IrType = when (returnIrTypes.size) {
            1 -> returnIrTypes.single()
            2 -> pairClass()?.typeWith(returnIrTypes) ?: return null
            3 -> tripleClass()?.typeWith(returnIrTypes) ?: return null
            else -> return null
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
        val bodyContext = context.copy(tensorTemplateParam = tensorTemplateParam)

        val body = buildBody(fn, lambdaFun, irParams, boxedReturnType, bodyContext) ?: return null
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
                        is DxirConst -> irConstFor(node, context) ?: cancel()
                        is DxirOp -> irOpFor(node, env, context) ?: cancel()
                        else -> cancel()
                    }
                    val ty = irTypeFor(node.type, context) ?: cancel()
                    val v = irTemporary(
                        value = expr,
                        nameHint = "s${node.id}",
                        irType = ty,
                    )
                    env[node.id] = v
                }
                val returnExpr: IrExpression = when (fn.returns.size) {
                    1 -> irGet(env[fn.returns.single().id] ?: cancel())
                    2, 3 -> {
                        val elementDecls = fn.returns.map { env[it.id] ?: cancel() }
                        val ctorSym = when (fn.returns.size) {
                            2 -> pairConstructor() ?: cancel()
                            3 -> tripleConstructor() ?: cancel()
                            else -> cancel()
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

    private fun IrBuilderWithScope.irConstFor(node: DxirConst, context: SynthesisContext): IrExpression? {
        val ty = irTypeFor(node.type, context) ?: return null
        val v = node.value
        // Rank-1 DxirConst lowering is not exercised by any gradient body the SCT
        // transform emits today (SumRule's seed is scalar; MeanRule's `1/N` is scalar).
        // Rejecting here keeps the scalar-const path honest — if a future rule needs
        // rank-N constants, it grows a FloatArray-backed ctor call here.
        if (!node.type.isScalar) return null
        return when (node.type.dtype) {
            F32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Float, v as Float)
            F64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Double, v as Double)
            I32 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Int, v as Int)
            I64 -> IrConstImpl(startOffset, endOffset, ty, IrConstKind.Long, v as Long)
            Bool -> null
        }
    }

    private fun IrBuilderWithScope.irOpFor(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.isMultiResult) return null
        // OpKind.IF is the only region-bearing op the synthesis scope accepts — emitted by
        // `DxirReverseTransform.handleIfAdjoint` (§0.4.23) with empty-body regions yielding
        // outer-scope adjoints. Primal-shape IFs with body ops in branches never survive to
        // synthesis (the reverse transform absorbs them via walkBranchReverse). Other
        // region-bearing ops (WHILE, MANUAL_COMPUTATION) still fall back.
        if (op.hasRegions) {
            if (op.op != OpKind.IF) return null
            return irIfOp(op, env, context)
        }
        // RELU and STEP don't lower to a stdlib operator; synthesise them from a primitive
        // `>` comparison + an if/else.  BROADCAST (rank-1 only) lowers to `broadcastLike`
        // in :core/ops. Everything else maps to the matching member op.
        if (op.op == OpKind.STEP) return irStep(op, env, context)
        if (op.op == OpKind.NOT) return irNot(op, env)
        if (op.op == OpKind.RELU) return irRelu(op, env, context)
        if (op.op == OpKind.BROADCAST) return irBroadcast(op, env, context)
        if (op.op == OpKind.SQRT) return irSqrt(op, env, context)
        if (op.op == OpKind.POW) return irPow(op, env, context)
        if (op.op == OpKind.CAST) return irCast(op, env, context)
        if (op.op == OpKind.GATHER) return irGather(op, env, context)
        if (op.op == OpKind.SCATTER) return irScatter(op, env, context)
        if (op.op == OpKind.SCATTER_ADD) return irScatterAdd(op, env, context)

        val operandDecls = op.operands.map { env[it.id] ?: return null }
        // §0.4.42 — rank-1 ADD/SUB/MUL/DIV route through `:core/ops` tensor operators
        // (`DTensor.plus` etc., declared in HostOps.kt) rather than the primitive
        // `Float.plus`. `gradAccum`'s outer ADD accumulation for rank-1 gradient
        // contributions (e.g., multi-gather adjoints) hits this path. Scalar path
        // unchanged — `findBinaryOp` on a scalar `op.type` still resolves to the
        // primitive operator.
        val symbol = when (op.op) {
            OpKind.ADD -> if (op.type.rank == 1) findTensorBinaryOp("plus") else findBinaryOp("plus", op.type, context)
            OpKind.SUB -> if (op.type.rank == 1) findTensorBinaryOp("minus") else findBinaryOp("minus", op.type, context)
            OpKind.MUL -> if (op.type.rank == 1) findTensorBinaryOp("times") else findBinaryOp("times", op.type, context)
            OpKind.DIV -> if (op.type.rank == 1) findTensorBinaryOp("div") else findBinaryOp("div", op.type, context)
            OpKind.NEG -> findUnaryOp("unaryMinus", op.type, context)
            else -> return null
        } ?: return null

        val resultType = irTypeFor(op.type, context) ?: return null
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
        val condition = greaterThanZero(operandDecl, operandType, context) ?: return null
        if (op.type.dtype == Bool) return condition
        val ty = irTypeFor(op.type, context) ?: return null
        val one = zeroOrOneConst(op.type, one = true, context) ?: return null
        val zero = zeroOrOneConst(op.type, one = false, context) ?: return null
        return irIfThenElse(ty, condition, one, zero)
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
        val ty = irTypeFor(op.type, context) ?: return null
        val zero = zeroOrOneConst(op.type, one = false, context) ?: return null
        val condition = greaterThanZero(operandDecl, op.type, context) ?: return null
        return irIfThenElse(ty, condition, irGet(operandDecl), zero)
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
        // Only narrow scalar-seed → rank-1 uniform broadcast. Dim values are irrelevant
        // here (the sentinel flows through unread); rank is the thing we gate on.
        if (op.operands.size != 1) return null
        val operand = op.operands[0]
        if (!operand.type.isScalar) return null
        if (op.type.dtype != F32 || op.type.rank != 1) return null

        val operandDecl = env[operand.id] ?: return null
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
        // broadcastLike is `fun <S : Shape> broadcastLike(v: Float, template: DTensor<S, F32>)`.
        // Thread the call-site shape (`Rank1<Sym>` etc.) through the single type argument so
        // the IR verifier has a concrete S. fromSymbolOwner sizes `arguments` from the
        // callee's parameter shape — 2 regulars, no dispatch receiver.
        call.typeArguments[0] = shapeTypeArg
        call.arguments[0] = irGet(operandDecl)
        call.arguments[1] = irGet(template)
        return call
    }

    /**
     * `OpKind.SQRT(x)` → IrCall to `io.tlaloc.core.sqrt` (the Float / Double extension
     * declared in `:core/DScalar.kt`). Scalar-only today — rank-1 tensor sqrt would
     * need `io.tlaloc.core.ops.sqrt` (the DTensor extension) and tensor-IrType
     * threading; the narrow scalar path is sufficient for the D.1b brachistochrone
     * port that unblocks sqrt in scalar primal bodies.
     */
    private fun IrBuilderWithScope.irSqrt(
        op: DxirOp,
        env: Map<Int, IrValueDeclaration>,
        context: SynthesisContext,
    ): IrExpression? {
        if (op.operands.size != 1) return null
        if (!op.type.isScalar) return null
        val operandDecl = env[op.operands[0].id] ?: return null
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
        return pluginContext.referenceFunctions(callableId).singleOrNull()
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
        if (!op.type.isScalar || !op.operands[0].type.isScalar) return null
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
        // witness + any param/return type-arg machinery). Other shapes / dtypes are out of
        // scope — callers receive `null` and fall back to the runtime tape path.
        if (isRank1F32(type)) return context.tensorIrType
        return null
    }

    private fun isRank1F32(type: DxirType): Boolean = type.rank == 1 && type.dtype == F32

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

    private fun pairConstructor(): IrConstructorSymbol? =
        pluginContext.referenceConstructors(ClassId.fromString("kotlin/Pair")).singleOrNull()

    private fun tripleConstructor(): IrConstructorSymbol? =
        pluginContext.referenceConstructors(ClassId.fromString("kotlin/Triple")).singleOrNull()

    @Suppress("unused")
    private val dummyParam: DxirParam? = null
}
