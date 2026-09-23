package io.tlaloc.plugin

import org.jetbrains.kotlin.fir.analysis.extensions.FirAdditionalCheckersExtension
import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrar

/**
 * §0.4.499 — [options] arrives here as a value rather than a global, so the FIR
 * checker's diagnostic severities belong to THIS compilation and cannot be flipped
 * by another module compiled concurrently in the same Kotlin daemon.
 */
class TlalocFirExtensionRegistrar(
    private val options: TlalocPluginOptions = TlalocPluginOptions(),
    private val handoff: TlalocLoweringHandoff = TlalocLoweringHandoff(),
) : FirExtensionRegistrar() {
    override fun ExtensionRegistrarContext.configurePlugin() {
        +FirAdditionalCheckersExtension.Factory { session ->
            TlalocCheckersExtension(session, options, handoff)
        }
    }
}
