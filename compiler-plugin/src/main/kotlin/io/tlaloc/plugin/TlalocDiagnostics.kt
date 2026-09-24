package io.tlaloc.plugin

import org.jetbrains.kotlin.com.intellij.psi.PsiElement
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactory1
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactoryToRendererMap
import org.jetbrains.kotlin.diagnostics.KtDiagnosticsContainer
import org.jetbrains.kotlin.diagnostics.KtSourcelessDiagnosticFactory
import org.jetbrains.kotlin.diagnostics.SourceElementPositioningStrategies
import org.jetbrains.kotlin.diagnostics.rendering.BaseDiagnosticRendererFactory
import org.jetbrains.kotlin.diagnostics.rendering.BaseSourcelessDiagnosticRendererFactory
import org.jetbrains.kotlin.diagnostics.rendering.CommonRenderers
import org.jetbrains.kotlin.diagnostics.error1
import org.jetbrains.kotlin.diagnostics.errorWithoutSource
import org.jetbrains.kotlin.diagnostics.warning1
import org.jetbrains.kotlin.diagnostics.warningWithoutSource

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
     * AND the compilation opted out of the default refusal
     * (`strictLowering=false`). Warning severity: the call stays unrewritten and
     * the `io.tlaloc.autograd` fallback body runs — which throws at the first
     * call. See [LAMBDA_NOT_LOWERABLE] for the default.
     */
    val LAMBDA_UNSUPPORTED: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * The DEFAULT severity for a lambda body the lowering refused: an
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
     * Fires when a binary tensor op's operands carry
     * incompatible named-index structure: disjoint named axes that can neither
     * contract nor broadcast, or a shared name with conflicting symbolic dims.
     * The payload is the rendered offending mismatch (e.g. "expected Named<Batch, B>
     * but got Named<SeqLen, T>").
     */
    val NAMED_INDEX_MISMATCH: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** Concrete tensor-shape violation in a successfully lowered
     * grad body (e.g. matmul contract-dim mismatch on literal dims). Error
     * severity: the program cannot execute. */
    val TENSOR_SHAPE_MISMATCH: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** The reverse-mode transform, run at CHECK time on the
     * lowered body, failed: the gradient this call requests cannot be
     * computed. The payload is the transform's reason (e.g. a missing VJP
     * rule). Error severity — the same failure would otherwise surface at
     * runtime. */
    val NOT_DIFFERENTIABLE: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /**
     * An unexpected exception inside the plugin while handling a recognised
     * intrinsic call (anything other than the lowering's own named refusals). ERROR by
     * default; see [INTERNAL_ERROR_WARNING] for `strictLowering=false`.
     */
    val INTERNAL_ERROR: KtDiagnosticFactory1<String> by error1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** [INTERNAL_ERROR] under `strictLowering=false`: the call is left as
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
                "failure deliberately — call `strictLowering.set(false)` in the `tlaloc` block of the build script (or pass " +
                "-P plugin:io.tlaloc.plugin:strictLowering=false), which turns this back into a warning.",
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

/**
 * The IR phase's warnings, reported by [TlalocIrReporter] at the intrinsic call they
 * concern. Each carries its complete message as the single argument, and each name
 * can be given to `@Suppress` to silence it. The IR phase's errors are in
 * [TlalocIrSourcelessErrors], out of `@Suppress`'s reach.
 */
internal object TlalocIrErrors : KtDiagnosticsContainer() {
    /** The call cannot be rewritten into a gradient and is left as written, so it
     * throws at its first call. Reported under `strictLowering=false`; by default the
     * same message is an error. */
    val IR_LOWERING_REFUSED_WARNING: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** An unexpected exception inside the IR phase while rewriting a call, under
     * `strictLowering=false`; by default the same message is an error. */
    val IR_INTERNAL_ERROR_WARNING: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    /** A pipeline step failed and the call was compiled from a less processed input. */
    val IR_DEGRADED: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
        SourceElementPositioningStrategies.DEFAULT,
    )

    override fun getRendererFactory(): BaseDiagnosticRendererFactory = TlalocIrRendererFactory
}

internal object TlalocIrRendererFactory : BaseDiagnosticRendererFactory() {
    override val MAP: KtDiagnosticFactoryToRendererMap by KtDiagnosticFactoryToRendererMap("TlalocIr") { map ->
        map.put(TlalocIrErrors.IR_LOWERING_REFUSED_WARNING, "{0}", CommonRenderers.STRING)
        map.put(TlalocIrErrors.IR_INTERNAL_ERROR_WARNING, "{0}", CommonRenderers.STRING)
        map.put(TlalocIrErrors.IR_DEGRADED, "{0}", CommonRenderers.STRING)
    }
}

/**
 * The IR phase's errors, and its warnings that have no call to point at (a failure
 * covering a whole file, or a call the IR phase never found). They carry a file and,
 * when known, a line and column as a plain location. Having no source element, they
 * cannot be silenced with `@Suppress`.
 */
internal object TlalocIrSourcelessErrors : KtDiagnosticsContainer() {
    val IR_ERROR_NO_SOURCE: KtSourcelessDiagnosticFactory by errorWithoutSource()
    val IR_WARNING_NO_SOURCE: KtSourcelessDiagnosticFactory by warningWithoutSource()

    override fun getRendererFactory(): BaseDiagnosticRendererFactory = TlalocIrSourcelessRendererFactory
}

internal object TlalocIrSourcelessRendererFactory : BaseSourcelessDiagnosticRendererFactory() {
    override val MAP: KtDiagnosticFactoryToRendererMap by KtDiagnosticFactoryToRendererMap("TlalocIrSourceless") { map ->
        map.put(TlalocIrSourcelessErrors.IR_ERROR_NO_SOURCE, MESSAGE_PLACEHOLDER)
        map.put(TlalocIrSourcelessErrors.IR_WARNING_NO_SOURCE, MESSAGE_PLACEHOLDER)
    }
}
