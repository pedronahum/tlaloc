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

    /** Fires with a human-readable reason when the lambda body couldn't be lowered. */
    val LAMBDA_UNSUPPORTED: KtDiagnosticFactory1<String> by warning1<PsiElement, String>(
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
