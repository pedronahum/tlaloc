package io.tlaloc.plugin

import org.jetbrains.kotlin.fir.expressions.FirExpression
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.resolvedArgumentMapping
import org.jetbrains.kotlin.fir.references.toResolvedCallableSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirFunctionSymbol

/**
 * The call's arguments in the order of the parameters they bind to.
 *
 * FIR keeps the argument list in source order, and Kotlin accepts named arguments in
 * any order, so code that reads arguments by position reads them from here. A named
 * argument keeps its [org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression]
 * wrapper. A parameter left to its default value has no entry, so a position is only
 * meaningful for functions whose defaulted parameters come last. An unresolved call
 * keeps its source order.
 */
internal fun argumentsInParameterOrder(call: FirFunctionCall): List<FirExpression> {
    val mapping = call.resolvedArgumentMapping ?: return call.argumentList.arguments
    val symbol = call.calleeReference.toResolvedCallableSymbol() as? FirFunctionSymbol<*>
        ?: return call.argumentList.arguments
    val position = symbol.valueParameterSymbols.withIndex().associate { (i, p) -> p.name to i }
    return mapping.entries.sortedBy { (_, p) -> position[p.name] ?: Int.MAX_VALUE }.map { it.key }
}
