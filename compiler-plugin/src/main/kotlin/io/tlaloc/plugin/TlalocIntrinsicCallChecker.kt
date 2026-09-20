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

object TlalocIntrinsicCallChecker : FirFunctionCallChecker(MppCheckerKind.Common) {
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
                // §0.4.353 — named-axis misuse is a type error (error severity);
                // everything else stays a warning (the runtime tape still runs it).
                reporter.reportOn(
                    expression.source,
                    if (result.namedIndex) TlalocErrors.NAMED_INDEX_MISMATCH else TlalocErrors.LAMBDA_UNSUPPORTED,
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
}
