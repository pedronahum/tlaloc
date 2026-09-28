package io.tlaloc.plugin

import org.jetbrains.kotlin.diagnostics.DiagnosticReporter
import org.jetbrains.kotlin.diagnostics.reportOn
import org.jetbrains.kotlin.fir.FirEvaluatorResult
import org.jetbrains.kotlin.fir.FirSession
import org.jetbrains.kotlin.fir.analysis.checkers.MppCheckerKind
import org.jetbrains.kotlin.fir.analysis.checkers.context.CheckerContext
import org.jetbrains.kotlin.fir.analysis.checkers.declaration.FirDeclarationChecker
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirFunctionCallChecker
import org.jetbrains.kotlin.fir.declarations.DirectDeclarationsAccess
import org.jetbrains.kotlin.fir.declarations.FirRegularClass
import org.jetbrains.kotlin.fir.expressions.FirDelegatedConstructorCall
import org.jetbrains.kotlin.fir.expressions.FirExpression
import org.jetbrains.kotlin.fir.expressions.FirExpressionEvaluator
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirLiteralExpression
import org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression
import org.jetbrains.kotlin.fir.expressions.FirVarargArgumentsExpression
import org.jetbrains.kotlin.fir.expressions.PrivateConstantEvaluatorAPI
import org.jetbrains.kotlin.fir.expressions.FirSpreadArgumentExpression
import org.jetbrains.kotlin.fir.expressions.arguments
import org.jetbrains.kotlin.fir.expressions.resolvedArgumentMapping
import org.jetbrains.kotlin.fir.references.toResolvedCallableSymbol
import org.jetbrains.kotlin.fir.resolve.providers.getRegularClassSymbolByClassId
import org.jetbrains.kotlin.fir.symbols.impl.FirConstructorSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirRegularClassSymbol
import org.jetbrains.kotlin.fir.types.ConeKotlinType
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.coneType
import org.jetbrains.kotlin.fir.types.resolvedType
import org.jetbrains.kotlin.fir.types.type
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.types.ConstantValueKind

/**
 * Compile-time checks for bounded dimensions (`io.tlaloc.core.Bounded<B>`, where `B` is a
 * `DimBound` object). See docs/design/bounded-dims.md.
 *
 * Kotlin's type checker already rejects mixing two bounds, or a bound and `Sym`, in one
 * operation: `DTensor`'s shape parameter is invariant. What it cannot see is a NUMBER,
 * so this checker compares the constant size arguments of a tensor-constructing call with
 * the bound of the axis they size.
 *
 * A bound's `max` is read from the delegating constructor call of a bound object declared
 * in the module being compiled. A bound from another compiled module has no constructor
 * arguments in FIR, so it is not checked here; the run-time check applies to it.
 */
internal object BoundedDims {
    val DIM_BOUND: ClassId = ClassId.fromString("io/tlaloc/core/DimBound")
    private const val BOUNDED = "io/tlaloc/core/Bounded"
    private const val NAMED = "io/tlaloc/core/Named"
    private const val DTENSOR = "io/tlaloc/core/DTensor"
    private val RANKS = mapOf(
        "io/tlaloc/core/Rank1" to 1, "io/tlaloc/core/Rank2" to 2, "io/tlaloc/core/Rank3" to 3,
        "io/tlaloc/core/Rank4" to 4, "io/tlaloc/core/Rank5" to 5, "io/tlaloc/core/Rank6" to 6,
    )

    /** The bound object of axis `i` of a shape type argument list, or null when the axis is not bounded. */
    fun boundedAxes(shapeType: ConeKotlinType): List<ClassId?>? {
        val rank = RANKS[shapeType.classId?.asString()] ?: return null
        return (0 until rank).map { i -> boundOf(shapeType.typeArguments.getOrNull(i)?.type) }
    }

    /** The bound object of one atom: `Bounded<B>`, or `Named<N, Bounded<B>>`. */
    fun boundOf(atom: ConeKotlinType?): ClassId? {
        val fqn = atom?.classId?.asString() ?: return null
        return when (fqn) {
            BOUNDED -> atom.typeArguments.firstOrNull()?.type?.classId
            NAMED -> boundOf(atom.typeArguments.getOrNull(1)?.type)
            else -> null
        }
    }

    /** The shape type of a `DTensor<S, T>`, or null. */
    fun dtensorShape(type: ConeKotlinType): ConeKotlinType? =
        if (type.classId?.asString() == DTENSOR) type.typeArguments.firstOrNull()?.type else null

    /** A bound object's `max`, when its declaration is in this module and passes a constant. */
    @OptIn(DirectDeclarationsAccess::class)
    fun boundMax(session: FirSession, bound: ClassId): Int? {
        val symbol: FirRegularClassSymbol = session.getRegularClassSymbolByClassId(bound) ?: return null
        val primary = symbol.declarationSymbols.filterIsInstance<FirConstructorSymbol>().firstOrNull { it.isPrimary }
            ?: return null
        val call = runCatching { primary.resolvedDelegatedConstructorCall }.getOrNull() ?: return null
        return delegatedMax(session, call)
    }

    fun delegatedMax(session: FirSession, call: FirDelegatedConstructorCall): Int? {
        val superId = call.constructedTypeRef.coneType.classId
        if (superId != DIM_BOUND) return null
        return call.argumentList.arguments.firstOrNull()?.let { constInt(session, it) }
    }

    /** The Int value of a compile-time constant expression, or null. */
    @OptIn(PrivateConstantEvaluatorAPI::class)
    fun constInt(session: FirSession, expr: FirExpression): Int? {
        val e = (expr as? FirNamedArgumentExpression)?.expression ?: expr
        val literal = e as? FirLiteralExpression
            ?: (runCatching { FirExpressionEvaluator.evaluateExpression(e, session) }.getOrNull()
                as? FirEvaluatorResult.Evaluated)?.result as? FirLiteralExpression
            ?: return null
        return when (literal.kind) {
            ConstantValueKind.Int, ConstantValueKind.IntegerLiteral, ConstantValueKind.Long ->
                (literal.value as? Number)?.toLong()?.takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt()
            else -> null
        }
    }
}

/**
 * `BOUNDED_DIM_EXCEEDED`: a call that builds a tensor whose type has a `Bounded<B>` axis,
 * with a constant size for that axis outside `1..B.max`.
 *
 * Which argument sizes which axis:
 * - the `Tensors` factories: `rows` and `cols` size axes 0 and 1, `d0`..`d5` axes 0..5,
 *   and a rank-1 factory's `data` sizes axis 0 when it is `floatArrayOf(...)`,
 *   `intArrayOf(...)`, `FloatArray(n)` or `IntArray(n)`;
 * - the `DTensor` constructor: its `dims` argument, when it is `intArrayOf(...)`;
 * - `io.tlaloc.autograd.specOf<S>(dtype, vararg fixedSizes)`: a fixed size that would land
 *   on a bounded axis is a count error, reported as `BOUNDED_SPEC_ARITY`.
 *
 * `BOUNDED_AXIS_MISMATCH`: the broadcasting elementwise operators of `io.tlaloc.core.ops`
 * given two operands whose right-aligned axes carry different bounds.
 */
internal class TlalocBoundedDimCallChecker : FirFunctionCallChecker(MppCheckerKind.Common) {
    private companion object {
        val BROADCASTING = setOf("plus", "minus", "times", "div", "plusBroadcast", "minusBroadcast", "timesBroadcast", "divBroadcast")
    }

    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirFunctionCall) {
        val symbol = expression.calleeReference.toResolvedCallableSymbol() ?: return
        val callableId = symbol.callableId ?: return
        val pkg = callableId.packageName.asString()
        val name = callableId.callableName.asString()
        when (pkg) {
            "io.tlaloc.core" -> Unit
            "io.tlaloc.core.ops" -> {
                if (callableId.classId == null && name in BROADCASTING) checkBroadcastOperands(expression, name)
                return
            }
            "io.tlaloc.autograd" -> {
                if (name == "specOf") checkSpecOf(expression)
                return
            }
            else -> return
        }
        val owner = callableId.classId?.asString()
        val isFactory = owner == "io/tlaloc/core/Tensors"
        val isConstructor = symbol is FirConstructorSymbol && owner == "io/tlaloc/core/DTensor"
        if (!isFactory && !isConstructor) return

        val shape = BoundedDims.dtensorShape(expression.resolvedType) ?: return
        val bounds = BoundedDims.boundedAxes(shape) ?: return
        if (bounds.all { it == null }) return
        val session = context.session

        val mapping = expression.resolvedArgumentMapping ?: return
        val sizes = HashMap<Int, Pair<String, Int>>() // axis -> (argument, value)
        for ((arg, param) in mapping) {
            val pname = param.name.asString()
            val axis = when {
                pname == "rows" -> 0
                pname == "cols" -> 1
                pname.length == 2 && pname[0] == 'd' && pname[1].isDigit() -> pname[1] - '0'
                else -> null
            }
            if (axis != null) {
                BoundedDims.constInt(session, arg)?.let { sizes[axis] = pname to it }
                continue
            }
            if (pname == "data" && bounds.size == 1) {
                arrayLength(session, arg)?.let { sizes[0] = "data" to it }
            }
            if (pname == "dims" && isConstructor) {
                arrayElements(session, arg)?.forEachIndexed { i, v -> if (v != null) sizes[i] = "dims[$i]" to v }
            }
        }
        for ((axis, sized) in sizes) {
            val bound = bounds.getOrNull(axis) ?: continue
            val max = BoundedDims.boundMax(session, bound) ?: continue
            val (argName, value) = sized
            if (value < 1 || value > max) {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.BOUNDED_DIM_EXCEEDED,
                    "axis $axis is Bounded<${bound.shortClassName.asString()}>, of size 1..$max, but " +
                        "`$argName` is $value",
                )
            }
        }
    }

    /**
     * The broadcasting elementwise operators take two unrelated shape parameters, so the type
     * checker accepts any pair of shapes. Axes are aligned from the right, as broadcasting
     * aligns them; two aligned axes bounded by different objects are an error. A bounded axis
     * against an unbounded one is not: it may be a size-1 axis that broadcasts.
     */
    context(context: CheckerContext, reporter: DiagnosticReporter)
    private fun checkBroadcastOperands(expression: FirFunctionCall, name: String) {
        val operands = buildList {
            expression.extensionReceiver?.let { add(it) }
            addAll(expression.arguments)
        }.mapNotNull { BoundedDims.dtensorShape(it.resolvedType)?.let(BoundedDims::boundedAxes) }
        if (operands.size != 2) return
        val (a, b) = operands
        val n = minOf(a.size, b.size)
        for (k in 1..n) {
            val ba = a[a.size - k] ?: continue
            val bb = b[b.size - k] ?: continue
            if (ba != bb) {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.BOUNDED_AXIS_MISMATCH,
                    "`$name` aligns axis ${a.size - k} of the left operand, Bounded<${ba.shortClassName.asString()}>, " +
                        "with axis ${b.size - k} of the right operand, Bounded<${bb.shortClassName.asString()}>; " +
                        "axes of different bounds have unrelated sizes",
                )
                return
            }
        }
    }

    context(context: CheckerContext, reporter: DiagnosticReporter)
    private fun checkSpecOf(expression: FirFunctionCall) {
        val shape = expression.typeArguments.firstOrNull()?.let { (it as? org.jetbrains.kotlin.fir.types.FirTypeProjectionWithVariance)?.typeRef?.coneType }
            ?: return
        val bounds = BoundedDims.boundedAxes(shape) ?: return
        val fixedAxes = bounds.count { it == null }
        val mapping = expression.resolvedArgumentMapping ?: return
        // No entry: no fixed size was passed. A spread (`*sizes`) has a count only at run time,
        // so the call is left to specOf's own check.
        val varargArg = mapping.entries.firstOrNull { it.value.name.asString() == "fixedSizes" }?.key
        val given = when (varargArg) {
            null -> emptyList()
            is FirVarargArgumentsExpression -> varargArg.arguments
            else -> return
        }
        if (given.any { it is FirSpreadArgumentExpression }) return
        if (given.size != fixedAxes) {
            reporter.reportOn(
                expression.source,
                TlalocErrors.BOUNDED_SPEC_ARITY,
                "the shape has ${bounds.size} axes, ${bounds.size - fixedAxes} of them bounded, so it takes " +
                    "$fixedAxes fixed size(s), one per unbounded axis in order; ${given.size} given",
            )
            return
        }
        given.forEachIndexed { i, e ->
            val v = BoundedDims.constInt(context.session, e) ?: return@forEachIndexed
            if (v < 1) {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.BOUNDED_SPEC_ARITY,
                    "fixed size ${i + 1} is $v; a fixed size must be at least 1",
                )
            }
        }
    }

    private fun arrayLength(session: FirSession, arg: FirExpression): Int? {
        val call = arg as? FirFunctionCall ?: return null
        val id = call.calleeReference.toResolvedCallableSymbol()?.callableId ?: return null
        val fn = id.callableName.asString()
        return when {
            id.packageName.asString() == "kotlin" && (fn == "floatArrayOf" || fn == "intArrayOf") ->
                (call.arguments.singleOrNull() as? FirVarargArgumentsExpression)?.arguments?.size ?: call.arguments.size
            id.classId?.asString() == "kotlin/FloatArray" || id.classId?.asString() == "kotlin/IntArray" ->
                call.arguments.firstOrNull()?.let { BoundedDims.constInt(session, it) }
            else -> null
        }
    }

    private fun arrayElements(session: FirSession, arg: FirExpression): List<Int?>? {
        val call = arg as? FirFunctionCall ?: return null
        val id = call.calleeReference.toResolvedCallableSymbol()?.callableId ?: return null
        if (id.packageName.asString() != "kotlin" || id.callableName.asString() != "intArrayOf") return null
        val elements = (call.arguments.singleOrNull() as? FirVarargArgumentsExpression)?.arguments ?: call.arguments
        return elements.map { BoundedDims.constInt(session, it) }
    }
}

/** `BOUNDED_DIM_INVALID`: `object X : DimBound(n)` with a constant `n` below 1. */
internal class TlalocDimBoundDeclarationChecker : FirDeclarationChecker<FirRegularClass>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    @OptIn(DirectDeclarationsAccess::class)
    override fun check(declaration: FirRegularClass) {
        if (declaration.superTypeRefs.none { it.coneType.classId == BoundedDims.DIM_BOUND }) return
        val primary = declaration.symbol.declarationSymbols.filterIsInstance<FirConstructorSymbol>()
            .firstOrNull { it.isPrimary } ?: return
        val call = primary.resolvedDelegatedConstructorCall ?: return
        val max = BoundedDims.delegatedMax(context.session, call) ?: return
        if (max < 1) {
            reporter.reportOn(
                declaration.source,
                TlalocErrors.BOUNDED_DIM_INVALID,
                "${declaration.name.asString()} passes $max to DimBound; a bound must be at least 1",
            )
        }
    }
}
