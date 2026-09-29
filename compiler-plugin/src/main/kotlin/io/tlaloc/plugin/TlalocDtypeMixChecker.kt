package io.tlaloc.plugin

import org.jetbrains.kotlin.diagnostics.DiagnosticReporter
import org.jetbrains.kotlin.diagnostics.reportOn
import org.jetbrains.kotlin.fir.analysis.checkers.MppCheckerKind
import org.jetbrains.kotlin.fir.analysis.checkers.context.CheckerContext
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirFunctionCallChecker
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression
import org.jetbrains.kotlin.fir.expressions.arguments
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
        if (expression.calleeReference is FirResolvedNamedReference) return
        val operands = listOfNotNull(expression.explicitReceiver) +
            expression.argumentList.arguments.map { (it as? FirNamedArgumentExpression)?.expression ?: it }
        val dtypes = operands.mapNotNull { floatDtypeOf(it.resolvedType) }.distinct()
        val name = expression.calleeReference.name.asString()
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

    private fun floatDtypeOf(type: ConeKotlinType): String? {
        if (type.classId?.asString() != "io/tlaloc/core/DTensor") return null
        return when (type.typeArguments.getOrNull(1)?.type?.classId?.asString()) {
            "io/tlaloc/core/F32" -> "F32"
            "io/tlaloc/core/F64" -> "F64"
            else -> null
        }
    }
}
