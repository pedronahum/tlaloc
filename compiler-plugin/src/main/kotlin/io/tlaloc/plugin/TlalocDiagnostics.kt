package io.tlaloc.plugin

import org.jetbrains.kotlin.com.intellij.psi.PsiElement
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactory1
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactoryToRendererMap
import org.jetbrains.kotlin.diagnostics.KtDiagnosticsContainer
import org.jetbrains.kotlin.diagnostics.SourceElementPositioningStrategies
import org.jetbrains.kotlin.diagnostics.rendering.BaseDiagnosticRendererFactory
import org.jetbrains.kotlin.diagnostics.rendering.CommonRenderers
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
    }
}
