package io.tlaloc.plugin

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.compiler.plugin.CompilerPluginRegistrar
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrarAdapter

@OptIn(ExperimentalCompilerApi::class)
class TlalocCompilerPluginRegistrar : CompilerPluginRegistrar() {
    override val pluginId: String = TlalocCommandLineProcessor.PLUGIN_ID

    override val supportsK2: Boolean = true

    override fun ExtensionStorage.registerExtensions(configuration: CompilerConfiguration) {
        // §0.4.503 (Tier 3, item 2) — THE VERSION GUARD RUNS FIRST, before a single
        // extension is registered, and that ordering is the whole design. Every K2
        // internal this plugin touches is reached from inside an extension: the FIR
        // checker, the FIR→dxir lowering, the IR generation extension. If the running
        // compiler's internals have moved, the failure happens once one of those is
        // constructed or invoked — deep inside `compileKotlin`, as a NoSuchMethodError
        // naming JetBrains classes. Refusing here, in the one method whose signature is
        // a published plugin API, is the difference between a sentence the user can act
        // on and a stack trace they cannot.
        //
        // Reported as an ERROR (which fails the compilation) rather than thrown, per the
        // house rule: a clean refusal, not a crash. `GuardReporter` owns the channel: the
        // compiler's own CLI diagnostic API when the running compiler has it, the message
        // collector when it does not — the guard's whole job is to run inside compilers
        // this plugin was NOT built against, so its reporting cannot assume the API it
        // was compiled against is there.
        val runningKotlin = KotlinVersionGuard.detectRunningCompilerVersion()
        if (!runVersionGuard(configuration, runningKotlin)) return
        try {
            registerTlalocExtensions(configuration)
        } catch (e: LinkageError) {
            // Reached only past the guard's opt-out: on a compiler inside the supported
            // range every one of these classes links. Reported as an ERROR that repeats
            // the versions and the opt-out, rather than left to crash the compiler: a
            // crash prints a stack trace and nothing else, and a Kotlin 2.3 CLI does not
            // print the guard's warning once the compilation has an error either.
            GuardReporter.report(
                configuration,
                CompilerMessageSeverity.ERROR,
                "Tlaloc's K2 compiler plugin could not register its extensions in Kotlin " +
                    "$runningKotlin: it is built against Kotlin " +
                    "${KotlinVersionGuard.COMPILED_AGAINST} (supported: " +
                    "${KotlinVersionGuard.supportedRangeDescription}) and was loaded anyway " +
                    "because unsafeAllowUnsupportedKotlin is set. The running compiler lacks " +
                    "an API it needs (${e.javaClass.simpleName}: ${e.message}).",
            )
        }
    }

    private fun ExtensionStorage.registerTlalocExtensions(configuration: CompilerConfiguration) {

        // §0.4.499 — the per-compilation knobs, resolved once and handed to BOTH
        // halves of the plugin (FIR checker + IR extension) as a value.
        val options = TlalocPluginOptions(
            dumpLoweredIr = configuration.get(TlalocCommandLineProcessor.DUMP_LOWERED_IR_KEY) ?: false,
            strictLowering = configuration.get(TlalocCommandLineProcessor.STRICT_LOWERING_KEY) ?: true,
        )
        // §0.4.514 — ONE handoff table per compilation, shared by the FIR checker that
        // writes it and the IR extension that reads it. registerExtensions runs once per
        // compilation, so two modules compiled concurrently in one Kotlin daemon each get
        // their own table and cannot read, overwrite or clear each other's entries.
        val handoff = TlalocLoweringHandoff()
        FirExtensionRegistrarAdapter.registerExtension(TlalocFirExtensionRegistrar(options, handoff))
        // §0.4.450 — the readable-reverse dump options (see TlalocCommandLineProcessor):
        // the dir form implies the message form.
        val dumpDir = configuration.get(TlalocCommandLineProcessor.DUMP_GRAD_SOURCE_DIR_KEY)
        val dump = (configuration.get(TlalocCommandLineProcessor.DUMP_GRAD_SOURCE_KEY) ?: false) ||
            dumpDir != null
        IrGenerationExtension.registerExtension(
            TlalocIrGenerationExtension(
                dumpGradSource = dump,
                dumpGradSourceDir = dumpDir,
                options = options,
                handoff = handoff,
            ),
        )
    }

    internal companion object {
        /**
         * The guard step of [registerExtensions], with the running compiler's version
         * as a parameter so a test can drive the refusal through a real compilation of
         * the compiler this plugin is built against. Returns true when the extensions
         * may be registered.
         */
        fun runVersionGuard(configuration: CompilerConfiguration, found: String?): Boolean {
            val allowUnsupportedKotlin =
                configuration.get(TlalocCommandLineProcessor.UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_KEY) ?: false
            return KotlinVersionGuard.check(
                found = found,
                allowUnsupported = allowUnsupportedKotlin,
            ) { severity, text -> GuardReporter.report(configuration, severity, text) }
        }
    }
}
