package io.tlaloc.plugin

import io.tlaloc.ir.pretty
import org.jetbrains.kotlin.diagnostics.DiagnosticReporter
import org.jetbrains.kotlin.diagnostics.reportOn
import org.jetbrains.kotlin.fir.analysis.checkers.MppCheckerKind
import org.jetbrains.kotlin.fir.analysis.checkers.context.CheckerContext
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirFunctionCallChecker
import org.jetbrains.kotlin.fir.expressions.FirAnonymousFunctionExpression
import org.jetbrains.kotlin.fir.expressions.FirExpression
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression
import org.jetbrains.kotlin.fir.references.toResolvedCallableSymbol

object TlalocIntrinsicCallChecker : FirFunctionCallChecker(MppCheckerKind.Common) {
    private val intrinsicNames: Set<String> = setOf(
        "io.tlaloc.autograd.grad",
        "io.tlaloc.autograd.grad2",
        "io.tlaloc.autograd.valueAndGrad",
        "io.tlaloc.autograd.valueAndGrad2",
    )

    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirFunctionCall) {
        val symbol = expression.calleeReference.toResolvedCallableSymbol() ?: return
        val callableId = symbol.callableId ?: return
        if (callableId.classId != null) return
        val fqn = "${callableId.packageName.asString()}.${callableId.callableName.asString()}"
        if (fqn !in intrinsicNames) return

        val lambda = extractLambdaArgument(expression)
        if (lambda == null) {
            reporter.reportOn(expression.source, TlalocErrors.INTRINSIC_CALL, fqn)
            return
        }

        val loweredName = "${callableId.callableName.asString()}_body"
        when (val result = FirLambdaToDxirLowering.lower(loweredName, lambda.anonymousFunction)) {
            is FirLambdaToDxirLowering.Result.Success -> {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.LAMBDA_LOWERED,
                    result.fn.pretty().trimEnd(),
                )
                val src = expression.source
                if (src != null) {
                    TlalocLoweringHandoff.record(src.startOffset, src.endOffset, result.fn)
                }
            }
            is FirLambdaToDxirLowering.Result.Failure -> {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.LAMBDA_UNSUPPORTED,
                    result.reason,
                )
            }
        }
    }

    private fun extractLambdaArgument(call: FirFunctionCall): FirAnonymousFunctionExpression? {
        for (arg in call.argumentList.arguments) {
            val unwrapped = unwrap(arg)
            if (unwrapped is FirAnonymousFunctionExpression) return unwrapped
        }
        return null
    }

    private fun unwrap(expr: FirExpression): FirExpression =
        if (expr is FirNamedArgumentExpression) expr.expression else expr
}
