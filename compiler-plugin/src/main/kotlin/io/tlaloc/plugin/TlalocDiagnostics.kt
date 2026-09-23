package io.tlaloc.plugin

import org.jetbrains.kotlin.com.intellij.psi.PsiElement
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactory1
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactoryToRendererMap
import org.jetbrains.kotlin.diagnostics.KtDiagnosticsContainer
import org.jetbrains.kotlin.diagnostics.SourceElementPositioningStrategies
import org.jetbrains.kotlin.diagnostics.rendering.BaseDiagnosticRendererFactory
import org.jetbrains.kotlin.diagnostics.rendering.CommonRenderers
import org.jetbrains.kotlin.diagnostics.error1
import org.jetbrains.kotlin.diagnostics.warning1

object TlalocErrors : KtDiagnosticsContainer() {
    /** Fires for every recognised intrinsic call that we did *not* attempt to lower. */
    val INTRINSIC_CALL: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** Fires with the pretty-printed dxir when the lambda body was successfully lowered. */
    val LAMBDA_LOWERED: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * Fires with a human-readable reason when the lambda body couldn't be lowered
     * AND the compilation opted out of the §0.4.499 refusal
     * (`strictLowering=false`). Warning severity: the call stays unrewritten and
     * the `io.tlaloc.autograd` fallback body runs — which throws at the first
     * call. See [LAMBDA_NOT_LOWERABLE] for the default.
     */
    val LAMBDA_UNSUPPORTED: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * §0.4.499 — the DEFAULT severity for a lambda body the lowering refused: an
     * ERROR at the call site, carrying the lowering's own verbatim reason (one of
     * ~208 named `LoweringException` sites in `FirLambdaToDxirLowering`).
     *
     * Why an error and not a warning: leaving the call unrewritten does not
     * degrade to a slower-but-correct path — it leaves `io.tlaloc.autograd`'s
     * `pluginMissing` fallback in place, which throws `IllegalStateException` at
     * the FIRST CALL, at runtime, telling the user to add a compiler plugin that
     * is already applied. A compile-time refusal that names the construct is the
     * same information, hours earlier.
     */
    val LAMBDA_NOT_LOWERABLE: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * Layer 1 (§0.4.241+) — fires when a binary tensor op's operands carry
     * incompatible named-index structure: disjoint named axes that can neither
     * contract nor broadcast, or a shared name with conflicting symbolic dims.
     * The payload is the rendered offending mismatch (e.g. "expected Named<Batch, B>
     * but got Named<SeqLen, T>").
     */
    val NAMED_INDEX_MISMATCH: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** §0.4.353 — concrete tensor-shape violation in a successfully lowered
     * grad body (e.g. matmul contract-dim mismatch on literal dims). Error
     * severity: the program cannot execute. */
    val TENSOR_SHAPE_MISMATCH: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** §0.4.353 — the reverse-mode transform, run at CHECK time on the
     * lowered body, failed: the gradient this call requests cannot be
     * computed. The payload is the transform's reason (e.g. a missing VJP
     * rule). Error severity — the same failure would otherwise surface at
     * runtime. */
    val NOT_DIFFERENTIABLE: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * §0.4.514 — an unexpected exception inside the plugin while handling a recognised
     * intrinsic call (anything other than the lowering's own named refusals). ERROR by
     * default; see [INTERNAL_ERROR_WARNING] for `strictLowering=false`.
     */
    val INTERNAL_ERROR: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** §0.4.514 — [INTERNAL_ERROR] under `strictLowering=false`: the call is left as
     * written and the fallback body throws at the first call. */
    val INTERNAL_ERROR_WARNING: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    override fun getRendererFactory(): BaseDiagnosticRendererFactory = TlalocRendererFactory
}

object TlalocRendererFactory : BaseDiagnosticRendererFactory() {
    override val MAP: KtDiagnosticFactoryToRendererMap by KtDiagnosticFactoryToRendererMap("Tlaloc") { map ->
        map.put(
            TlalocErrors.INTRINSIC_CALL,
            "Tlaloc intrinsic ''{0}'' recognised (runtime-tape path; the K2 plugin will replace this with a dxir transform in a later step).",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.LAMBDA_LOWERED,
            "Tlaloc lowered lambda to dxir:\n{0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.LAMBDA_UNSUPPORTED,
            "Tlaloc could not lower lambda: {0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.LAMBDA_NOT_LOWERABLE,
            "Tlaloc could not lower this lambda at compile time: {0}\n" +
                "The call is therefore NOT rewritten, and the io.tlaloc.autograd fallback body " +
                "would throw at the first call instead of returning a gradient. Rewrite the body " +
                "within the supported surface (docs/GETTING_STARTED.md), use the Tracer-capture " +
                "API (io.tlaloc.autograd.gradWithScalars) for this one, or — to take that runtime " +
                "failure deliberately — pass " +
                "-P plugin:io.tlaloc.plugin:strictLowering=false, which turns this back into a warning.",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.INTERNAL_ERROR,
            "{0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.INTERNAL_ERROR_WARNING,
            "{0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.NAMED_INDEX_MISMATCH,
            "Tlaloc named-index mismatch: {0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.TENSOR_SHAPE_MISMATCH,
            "Tlaloc tensor shape mismatch: {0}",
            CommonRenderers.STRING,
        )
        map.put(
            TlalocErrors.NOT_DIFFERENTIABLE,
            "Tlaloc cannot differentiate this body: {0}",
            CommonRenderers.STRING,
        )
    }
}
