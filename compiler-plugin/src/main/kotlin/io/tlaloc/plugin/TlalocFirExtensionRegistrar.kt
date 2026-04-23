package io.tlaloc.plugin

import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrar

class TlalocFirExtensionRegistrar : FirExtensionRegistrar() {
    override fun ExtensionRegistrarContext.configurePlugin() {
        +::TlalocCheckersExtension
    }
}
