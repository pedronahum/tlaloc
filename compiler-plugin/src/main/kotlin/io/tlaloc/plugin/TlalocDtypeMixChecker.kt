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
        if (dtypes.size < 2) return
        val name = expression.calleeReference.name.asString()
        reporter.reportOn(
            expression.source,
            TlalocErrors.DTYPE_MISMATCH,
            "`$name` is applied to DTensor<…, F32> and DTensor<…, F64> operands. Tlaloc does " +
                "not convert between float dtypes implicitly; make both operands the same dtype",
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
