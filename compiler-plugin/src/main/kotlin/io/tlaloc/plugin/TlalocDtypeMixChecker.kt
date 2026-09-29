package io.tlaloc.plugin

import org.jetbrains.kotlin.diagnostics.DiagnosticReporter
import org.jetbrains.kotlin.diagnostics.reportOn
import org.jetbrains.kotlin.fir.analysis.checkers.MppCheckerKind
import org.jetbrains.kotlin.fir.analysis.checkers.context.CheckerContext
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirFunctionCallChecker
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression
import org.jetbrains.kotlin.fir.expressions.arguments
import org.jetbrains.kotlin.fir.references.FirResolvedErrorReference
import org.jetbrains.kotlin.fir.resolve.providers.symbolProvider
import org.jetbrains.kotlin.fir.references.FirResolvedNamedReference
import org.jetbrains.kotlin.fir.types.ConeKotlinType
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.resolvedType
import org.jetbrains.kotlin.fir.types.type

/**
 * `DTYPE_UNSUPPORTED`: a call to an operation that has no F64 version (see [F32_ONLY]) with
 * F64 operands.
 *
 * `DTYPE_MISMATCH`: a call that does not resolve because its receiver and arguments are
 * `DTensor`s of two float dtypes (`F32` and `F64`). Tlaloc has no implicit promotion, so
 * no overload takes both; Kotlin reports the candidates it tried, and this names the
 * dtypes, at the same call.
 */
internal class TlalocDtypeMixChecker : FirFunctionCallChecker(MppCheckerKind.Common) {

    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirFunctionCall) {
        // K2 resolves a single-candidate call with mismatched arguments to a
        // FirResolvedErrorReference, a subtype of FirResolvedNamedReference.
        val ref = expression.calleeReference
        if (ref is FirResolvedNamedReference && ref !is FirResolvedErrorReference) return
        val operands = listOfNotNull(expression.explicitReceiver) +
            expression.argumentList.arguments.map { (it as? FirNamedArgumentExpression)?.expression ?: it }
        val dtypes = operands.mapNotNull { floatDtypeOf(it.resolvedType) }.distinct()
        val name = ref.name.asString()
        // A Float or FloatScalar next to an F64 tensor is reported too; next to an F32
        // tensor it is not this checker's business (F32 programs keep Kotlin's message).
        val scalarF32 = operands.any { scalarF32(it.resolvedType) }
        if ("F64" in dtypes && "F32" !in dtypes && scalarF32 && isCoreOp(ref.name)) {
            reporter.reportOn(
                expression.source,
                TlalocErrors.DTYPE_MISMATCH,
                "`$name` is applied to a DTensor<…, F64> and a Float value. Tlaloc does not " +
                    "convert between float dtypes implicitly; use a Double (or DoubleScalar)",
            )
            return
        }
        if (dtypes == listOf("F64") && name in F32_ONLY) {
            reporter.reportOn(
                expression.source,
                TlalocErrors.DTYPE_UNSUPPORTED,
                "`$name` exists for F32 tensors only; there is no F64 version",
            )
            return
        }
        if (dtypes.size < 2) return
        reporter.reportOn(
            expression.source,
            TlalocErrors.DTYPE_MISMATCH,
            "`$name` is applied to DTensor<…, F32> and DTensor<…, F64> operands. Tlaloc does " +
                "not convert between float dtypes implicitly; make both operands the same dtype",
        )
    }

    private companion object {
        /** The `io.tlaloc.core.ops` operations that have no F64 overload. */
        val F32_ONLY = setOf(
            "embedding", "embeddingGrad", "sparseMatmul", "sparseMatmulTransposed", "sparseMatmulValuesAdjoint",
        )
    }

    /**
     * True when [name] is a top-level function of `io.tlaloc.core.ops`: the Float branch
     * above speaks only for Tlaloc's own operations, so a user function that fails to
     * resolve for another reason keeps Kotlin's message alone.
     */
    context(context: CheckerContext)
    private fun isCoreOp(name: org.jetbrains.kotlin.name.Name): Boolean =
        context.session.symbolProvider
            .getTopLevelCallableSymbols(org.jetbrains.kotlin.name.FqName("io.tlaloc.core.ops"), name)
            .isNotEmpty()

    private fun scalarF32(type: ConeKotlinType): Boolean =
        type.classId?.asString() in setOf("kotlin/Float", "io/tlaloc/core/FloatScalar")

    private fun floatDtypeOf(type: ConeKotlinType): String? {
        if (type.classId?.asString() != "io/tlaloc/core/DTensor") return null
        return when (type.typeArguments.getOrNull(1)?.type?.classId?.asString()) {
            "io/tlaloc/core/F32" -> "F32"
            "io/tlaloc/core/F64" -> "F64"
            else -> null
        }
    }
}
