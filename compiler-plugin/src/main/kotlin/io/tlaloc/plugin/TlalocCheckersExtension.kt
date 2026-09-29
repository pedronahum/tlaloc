package io.tlaloc.plugin

import org.jetbrains.kotlin.fir.FirSession
import org.jetbrains.kotlin.fir.analysis.checkers.declaration.DeclarationCheckers
import org.jetbrains.kotlin.fir.analysis.checkers.declaration.FirDeclarationChecker
import org.jetbrains.kotlin.fir.analysis.checkers.expression.ExpressionCheckers
import org.jetbrains.kotlin.fir.declarations.FirRegularClass
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirFunctionCallChecker
import org.jetbrains.kotlin.fir.analysis.extensions.FirAdditionalCheckersExtension

class TlalocCheckersExtension(
    session: FirSession,
    private val options: TlalocPluginOptions = TlalocPluginOptions(),
    private val handoff: TlalocLoweringHandoff = TlalocLoweringHandoff(),
) : FirAdditionalCheckersExtension(session) {
    override val expressionCheckers: ExpressionCheckers = object : ExpressionCheckers() {
        override val functionCallCheckers: Set<FirFunctionCallChecker> =
            setOf(TlalocIntrinsicCallChecker(options, handoff), TlalocBoundedDimCallChecker(), TlalocDtypeMixChecker())
    }
    override val declarationCheckers: DeclarationCheckers = object : DeclarationCheckers() {
        override val regularClassCheckers: Set<FirDeclarationChecker<FirRegularClass>> =
            setOf(TlalocDimBoundDeclarationChecker())
    }
}
