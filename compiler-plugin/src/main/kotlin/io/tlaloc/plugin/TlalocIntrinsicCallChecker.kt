package io.tlaloc.plugin

import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.validateDxirShapes
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
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.coneType

/**
 * §0.4.499 — was an `object`; now a class carrying this compilation's
 * [TlalocPluginOptions]. Two things hang off them:
 *
 *  - [TlalocPluginOptions.dumpLoweredIr] gates the lowered-dxir dump
 *    ([TlalocErrors.LAMBDA_LOWERED] and [TlalocErrors.INTRINSIC_CALL]). Until
 *    §0.4.499 both fired unconditionally, so ONE `grad {}` put two IR dumps in
 *    every consumer's build log — and made the plugin unusable under `-Werror`.
 *  - [TlalocPluginOptions.strictLowering] decides the severity of a lowering
 *    FAILURE: [TlalocErrors.LAMBDA_NOT_LOWERABLE] (error, the default) versus
 *    [TlalocErrors.LAMBDA_UNSUPPORTED] (warning, the opt-out).
 */
class TlalocIntrinsicCallChecker(
    private val options: TlalocPluginOptions = TlalocPluginOptions(),
) : FirFunctionCallChecker(MppCheckerKind.Common) {
    private val intrinsicNames: Set<String> = setOf(
        "io.tlaloc.autograd.grad",
        "io.tlaloc.autograd.grad2",
        "io.tlaloc.autograd.valueAndGrad",
        "io.tlaloc.autograd.valueAndGrad2",
        // §0.4.424 — the three-argument reverse spellings.
        "io.tlaloc.autograd.grad3",
        "io.tlaloc.autograd.valueAndGrad3",
        // §0.4.372 — forward-mode (Phase B1). §0.4.387 — its two-argument forms.
        "io.tlaloc.autograd.jvp",
        "io.tlaloc.autograd.valueAndJvp",
        "io.tlaloc.autograd.jvp2",
        "io.tlaloc.autograd.valueAndJvp2",
        // §0.4.394 — Phase B2: assembly intrinsics over the seeded transforms.
        // §0.4.406 — their two-argument forms.
        "io.tlaloc.autograd.jacobian",
        "io.tlaloc.autograd.hessian",
        "io.tlaloc.autograd.jacobian2",
        "io.tlaloc.autograd.hessian2",
        // §0.4.412 — the reverse-assembled (tall) Jacobian. §0.4.424 — its
        // two-argument form.
        "io.tlaloc.autograd.jacobianReverse",
        "io.tlaloc.autograd.jacobianReverse2",
        // §0.4.398 — the seeded-cotangent user surface. §0.4.406 — its
        // two-argument forms.
        "io.tlaloc.autograd.vjp",
        "io.tlaloc.autograd.valueAndVjp",
        "io.tlaloc.autograd.vjp2",
        "io.tlaloc.autograd.valueAndVjp2",
    )

    /** §0.4.372 — the forward-mode intrinsics probe differentiability with the
     * forward transform (JVP), not the reverse one. §0.4.394 — `jacobian`
     * assembles forward columns, so it probes the same way (its lambda returns
     * a TENSOR, which the reverse probe would reject outright). */
    private val forwardIntrinsics: Set<String> =
        setOf("jvp", "valueAndJvp", "jvp2", "valueAndJvp2", "jacobian", "jacobian2")

    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirFunctionCall) {
        val symbol = expression.calleeReference.toResolvedCallableSymbol() ?: return
        val callableId = symbol.callableId ?: return
        if (callableId.classId != null) return
        val fqn = "${callableId.packageName.asString()}.${callableId.callableName.asString()}"
        if (fqn !in intrinsicNames) return

        val lambda = extractLambdaArgument(expression)
        if (lambda == null) {
            // §0.4.499 — no lambda literal to lower. The call stays as written, so
            // the `io.tlaloc.autograd` fallback body runs and throws at the first
            // call: under the default that is a refusal, by name, at compile time.
            // The informational spelling survives behind `dumpLoweredIr`.
            if (options.strictLowering) {
                reporter.reportOn(
                    expression.source,
                    TlalocErrors.LAMBDA_NOT_LOWERABLE,
                    "the argument to `$fqn` is not a lambda literal — the plugin lowers the " +
                        "BODY of a `{ }` written at the call site; a function reference, a " +
                        "variable holding a lambda, or a lambda built elsewhere has no body here " +
                        "to lower",
                )
            } else if (options.dumpLoweredIr) {
                reporter.reportOn(expression.source, TlalocErrors.INTRINSIC_CALL, fqn)
            }
            return
        }

        // §0.4.499 — `io.tlaloc.autograd.grad` / `grad2` / `grad3` / `valueAndGrad*`
        // are OVERLOADED: the compile-time INTRINSIC (GradIntrinsics.kt) and the
        // runtime Tracer-capture TAPE (Grad.kt) share those names, and Kotlin's
        // overload resolution picks between them by the lambda's parameter types.
        // This checker matches on the FQN alone, so it sees both. A Tracer-typed
        // lambda is the tape overload — the documented plugin-free route, never
        // something the plugin was asked to lower — so it is not diagnosed here at
        // all. Until §0.4.499 every such call drew a spurious
        // "could not lower lambda: ... unsupported type io.tlaloc.autograd.Tracer"
        // warning; with the refusal promoted to an error it would have broken that
        // route outright.
        if (isTracerLambda(lambda)) return

        val loweredName = "${callableId.callableName.asString()}_body"
        when (val result = FirLambdaToDxirLowering.lower(loweredName, lambda.anonymousFunction)) {
            is FirLambdaToDxirLowering.Result.Success -> {
                // §0.4.499 — developer introspection, OFF by default. This dump plus
                // the IR extension's handoff dump were the two warnings every
                // successful `grad {}` used to put in a consumer's build log.
                if (options.dumpLoweredIr) {
                    reporter.reportOn(
                        expression.source,
                        TlalocErrors.LAMBDA_LOWERED,
                        result.fn.pretty().trimEnd(),
                    )
                }
                // §0.4.353 — Meta-stage ergonomics: validate the lowered body
                // NOW, so shape and differentiability failures are red
                // squiggles at the call site instead of runtime surprises.
                // (1) Conservative static shape check (silent on symbolic dims).
                val shapeErrors = validateDxirShapes(result.fn)
                for (err in shapeErrors) {
                    reporter.reportOn(expression.source, TlalocErrors.TENSOR_SHAPE_MISMATCH, err)
                }
                // (2) The gradient this intrinsic requests, computed
                // symbolically at check time. Scope: bodies the raw reverse
                // transform handles directly (straight-line + IF). Loop-bearing
                // bodies go through the runtime's PhiCalculus + Symja-engine
                // coarsening pipeline (TlalocIrGenerationExtension §0.4.24/33)
                // — running that per keystroke is not check-time material, so
                // they keep their runtime backstop.
                val hasLoopRegions = result.fn.body.any {
                    it is io.tlaloc.ir.DxirOp && it.regions.isNotEmpty() && it.op != io.tlaloc.ir.OpKind.IF
                }
                if (shapeErrors.isEmpty() && !hasLoopRegions) {
                    val name = callableId.callableName.asString()
                    val probe: () -> Unit = when {
                        // §0.4.394 — `hessian` is forward-OVER-reverse, so the
                        // check-time probe composes both transforms exactly as
                        // the IR extension will. §0.4.406 — `hessian2` likewise.
                        name == "hessian" || name == "hessian2" -> {
                            { DxirForwardTransform.apply(DxirReverseTransform.apply(result.fn)) }
                        }
                        // §0.4.398 — the seeded-cotangent intrinsics probe the
                        // SEEDED reverse transform: their lambda may return a
                        // tensor, which the default probe's scalar gate would
                        // reject; seedAsParam is exactly what the IR extension
                        // runs, so the red squiggle matches the real lowering.
                        // §0.4.406 — the two-argument spellings likewise (the
                        // transform is arity-agnostic).
                        // §0.4.412 — `jacobianReverse` assembles seeded reverse
                        // pullbacks, so it probes exactly like `vjp` (its lambda
                        // may return a tensor; seedAsParam is what the IR
                        // extension runs).
                        // §0.4.424 — `jacobianReverse2` likewise (the transform
                        // is arity-agnostic).
                        name == "vjp" || name == "valueAndVjp" ||
                            name == "vjp2" || name == "valueAndVjp2" ||
                            name == "jacobianReverse" || name == "jacobianReverse2" -> {
                            {
                                DxirReverseTransform.apply(
                                    result.fn,
                                    includeForward = name.startsWith("valueAnd"),
                                    seedAsParam = true,
                                )
                            }
                        }
                        name in forwardIntrinsics -> {
                            // §0.4.407 — the forward transform carries the IF direct
                            // arm now, so IF-bearing bodies are probed like any other
                            // (the §0.4.403 skip is lifted). The only remaining gate
                            // is a loop region NESTED inside an IF branch —
                            // hasLoopRegions above sees only the top level, and those
                            // bodies keep their runtime backstop (PhiCalculus per
                            // keystroke is not check-time material, same reasoning as
                            // the top-level loop gate).
                            if (hasNestedNonIfRegions(result.fn.body)) {
                                ({ })
                            } else {
                                ({ DxirForwardTransform.apply(result.fn) })
                            }
                        }
                        else -> {
                            { DxirReverseTransform.apply(result.fn) }
                        }
                    }
                    runCatching { probe() }.onFailure { t ->
                        reporter.reportOn(
                            expression.source,
                            TlalocErrors.NOT_DIFFERENTIABLE,
                            t.message ?: t::class.simpleName ?: "unknown",
                        )
                    }
                }
                val src = expression.source
                if (src != null) {
                    TlalocLoweringHandoff.record(src.startOffset, src.endOffset, result.fn)
                }
            }
            is FirLambdaToDxirLowering.Result.Failure -> {
                // §0.4.353 — named-axis misuse is a type error (error severity).
                // §0.4.499 — and so, by default, is every OTHER lowering failure:
                // there is no working fallback behind it. The lowering's own verbatim
                // reason is carried through either way, so the ~208 named
                // `LoweringException` sites stay distinguishable instead of
                // collapsing to one generic string. `strictLowering=false` restores
                // the pre-alpha warning.
                val factory = when {
                    result.namedIndex -> TlalocErrors.NAMED_INDEX_MISMATCH
                    options.strictLowering -> TlalocErrors.LAMBDA_NOT_LOWERABLE
                    else -> TlalocErrors.LAMBDA_UNSUPPORTED
                }
                reporter.reportOn(expression.source, factory, result.reason)
            }
        }
    }

    /** §0.4.499 — true when any of the lambda's parameters is an
     * `io.tlaloc.autograd.Tracer`, i.e. this call resolved to the runtime
     * tape overload rather than the compile-time intrinsic. */
    private fun isTracerLambda(lambda: FirAnonymousFunctionExpression): Boolean =
        runCatching {
            lambda.anonymousFunction.valueParameters.any { p ->
                p.returnTypeRef.coneType.classId?.asFqNameString() == TRACER_FQN
            }
        }.getOrDefault(false)

    private fun extractLambdaArgument(call: FirFunctionCall): FirAnonymousFunctionExpression? {
        for (arg in call.argumentList.arguments) {
            val unwrapped = unwrap(arg)
            if (unwrapped is FirAnonymousFunctionExpression) return unwrapped
        }
        return null
    }

    private fun unwrap(expr: FirExpression): FirExpression =
        if (expr is FirNamedArgumentExpression) expr.expression else expr

    /** §0.4.407 — true when any op in [nodes] (recursing through IF branch
     * bodies) carries regions and is NOT an IF: a WHILE nested inside an IF
     * branch, the one region shape the raw forward transform still refuses
     * and the extension's PhiCalculus pipeline may yet lower. */
    private fun hasNestedNonIfRegions(nodes: List<io.tlaloc.ir.DxirNode>): Boolean = nodes.any { n ->
        n is io.tlaloc.ir.DxirOp && (
            (n.regions.isNotEmpty() && n.op != io.tlaloc.ir.OpKind.IF) ||
                n.regions.any { r -> r.blocks.any { b -> hasNestedNonIfRegions(b.body) } }
            )
    }

    private companion object {
        const val TRACER_FQN = "io.tlaloc.autograd.Tracer"
    }
}
