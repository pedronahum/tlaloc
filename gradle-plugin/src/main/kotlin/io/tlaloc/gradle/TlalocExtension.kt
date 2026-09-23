package io.tlaloc.gradle

import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.Property

/**
 * The `tlaloc { }` block. Each property is one option of the Tlaloc compiler plugin
 * and is passed to every Kotlin/JVM compilation of the project. The defaults are the
 * compiler plugin's own.
 */
abstract class TlalocExtension {
    /**
     * Refuse a `grad { }` / `jvp { }` / `vjp { }` lambda the plugin cannot compile, at
     * compile time, with the reason. Default `true`. With `false` the refusal is a
     * warning and the call throws `IllegalStateException` when it runs.
     */
    abstract val strictLowering: Property<Boolean>

    /**
     * Print each synthesized reverse-mode gradient as Kotlin source, as a compiler INFO
     * message headed by the lambda's source location. Default `false`.
     */
    abstract val dumpGradSource: Property<Boolean>

    /**
     * Also write each synthesized gradient as a `.kt` file, named after the lambda's
     * source location, under this directory. Setting it implies [dumpGradSource].
     * Unset by default.
     */
    abstract val dumpGradSourceDir: DirectoryProperty

    /**
     * Print the lowered Tlaloc IR of every recognised intrinsic lambda. Default `false`.
     * Part of this output is a compiler WARNING, so do not combine it with `-Werror`.
     */
    abstract val dumpLoweredIr: Property<Boolean>

    /**
     * Register the compiler plugin even when the Kotlin compiler is outside the
     * version range the plugin was built against. Default `false`, which is a compile
     * error naming both versions. The plugin reads unstable compiler internals, so a
     * crash inside the compiler is the expected result of setting it.
     */
    abstract val unsafeAllowUnsupportedKotlin: Property<Boolean>

    internal fun applyConventions() {
        strictLowering.convention(true)
        dumpGradSource.convention(false)
        dumpLoweredIr.convention(false)
        unsafeAllowUnsupportedKotlin.convention(false)
    }
}
