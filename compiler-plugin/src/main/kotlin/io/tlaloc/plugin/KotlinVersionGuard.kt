package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.config.KotlinCompilerVersion

/**
 * **The Kotlin version guard.**
 *
 * The problem it solves, stated plainly: this plugin is compiled against
 * `kotlin-compiler-embeddable` [COMPILED_AGAINST] and reaches into 40 distinct
 * `org.jetbrains.kotlin.fir.*` packages (119 `org.jetbrains.kotlin.*` imports in
 * total). None of that is a stable API. JetBrains moves K2's FIR and IR internals
 * between feature releases without deprecation, which is their prerogative — the
 * consequence for a consumer is that running this plugin on a different Kotlin
 * feature release produces a `NoSuchMethodError`, `NoSuchFieldError` or
 * `AbstractMethodError` from the middle of `compileKotlin`, with a stack trace
 * full of JetBrains package names and nothing at all pointing at Tlaloc.
 *
 * The guard turns that into a refusal that names the Kotlin version and the
 * supported range before any extension is registered.
 *
 * ## The supported range, and why it is that shape
 *
 * Kotlin's release train numbers FEATURE releases by tens in the third
 * component — 2.4.0, 2.4.10, 2.4.20 are three different feature releases — and
 * BUGFIX releases by ones above them: 2.4.21, 2.4.22 are bugfixes of 2.4.20.
 * Internal API moves at feature releases, not at bugfix releases. So the
 * supported range is the whole bugfix family of the release this plugin was
 * built against and nothing else:
 *
 *   built against 2.4.20  ⇒  supported 2.4.20 … 2.4.29
 *
 * That is deliberately neither "exactly 2.4.20" (which would break every user
 * the day a bugfix release lands, for no reason) nor "any 2.4.x" (which would
 * wave 2.4.0 and 2.4.10 through, and those are different feature releases with
 * different internals). [supportedRangeDescription] renders it for humans and
 * the README / docs/GETTING_STARTED.md / docs/COMPATIBILITY.md quote the same
 * two numbers.
 *
 * ## The escape hatch
 *
 * [TlalocCommandLineProcessor.UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION] downgrades
 * the refusal to a warning and registers the extensions anyway. It exists because
 * "refuse loudly" should not mean "refuse absolutely": a user on a fresh Kotlin
 * feature release may well find that everything works, and there is no reason to
 * make them fork the plugin to find out. The refusal message names the option.
 */
object KotlinVersionGuard {

    /**
     * The `kotlin-compiler-embeddable` version this plugin is compiled against.
     *
     * PINNED TO THE VERSION CATALOG, not to a copy of it:
     * `KotlinVersionGuardTest.compiledAgainstMatchesTheVersionCatalog` reads
     * `libs.versions.kotlin` through a system property the `:compiler-plugin`
     * test task sets, and fails if the two disagree. So a Kotlin bump in
     * `gradle/libs.versions.toml` that forgets this constant is a red test, not
     * a guard that silently refuses the compiler the repository itself uses.
     */
    const val COMPILED_AGAINST: String = "2.4.20"

    /** A parsed Kotlin version. [suffix] is everything after the numeric triple. */
    data class Version(
        val major: Int,
        val minor: Int,
        val patch: Int,
        val suffix: String?,
    ) {
        /**
         * The feature-release family: `patch / 10`. 2.4.20 and 2.4.27 share a
         * family (20..29); 2.4.10 does not. See the class KDoc for why this, and
         * not the patch number, is the unit of API compatibility.
         */
        val featureFamily: Int get() = patch / 10
    }

    /** What the guard decided. Rendered by [message]; acted on by [check]. */
    sealed interface Verdict {
        /** The running compiler is inside the supported range. */
        data object Supported : Verdict

        /** Parsed fine, wrong release. [found] is the raw string. */
        data class Unsupported(val found: String, val version: Version) : Verdict

        /**
         * Could not be parsed as a Kotlin version at all — including the case
         * where the compiler did not report one. Refused rather than assumed
         * good: a guard that passes whatever it cannot read is not a guard.
         */
        data class Unreadable(val found: String?) : Verdict
    }

    /** e.g. `2.4.20 through 2.4.29`. */
    val supportedRangeDescription: String
        get() {
            val v = parse(COMPILED_AGAINST)
                ?: error(
                    "KotlinVersionGuard.COMPILED_AGAINST is '$COMPILED_AGAINST', which is " +
                        "not a parseable Kotlin version. This is a Tlaloc bug, not a user error.",
                )
            val lo = v.featureFamily * 10
            return "${v.major}.${v.minor}.$lo through ${v.major}.${v.minor}.${lo + 9}"
        }

    /**
     * Parse `major.minor.patch[-suffix]`. Returns null for anything else —
     * including a two-component version, which Kotlin does not publish.
     */
    fun parse(raw: String?): Version? {
        if (raw == null) return null
        val text = raw.trim()
        if (text.isEmpty()) return null
        val dash = text.indexOf('-')
        val numeric = if (dash >= 0) text.substring(0, dash) else text
        val suffix = if (dash >= 0) text.substring(dash + 1).takeIf { it.isNotEmpty() } else null
        val parts = numeric.split('.')
        if (parts.size != 3) return null
        val nums = parts.map { it.toIntOrNull() ?: return null }
        if (nums.any { it < 0 }) return null
        return Version(nums[0], nums[1], nums[2], suffix)
    }

    /** The decision, as a pure function of the version string. */
    fun verdict(found: String?): Verdict {
        val running = parse(found) ?: return Verdict.Unreadable(found)
        val target = parse(COMPILED_AGAINST)!!
        val compatible = running.major == target.major &&
            running.minor == target.minor &&
            running.featureFamily == target.featureFamily
        return if (compatible) Verdict.Supported else Verdict.Unsupported(found!!.trim(), running)
    }

    /**
     * The text a user sees. Null for [Verdict.Supported] — a working build stays
     * silent.
     */
    fun message(verdict: Verdict, allowUnsupported: Boolean): String? = when (verdict) {
        Verdict.Supported -> null
        is Verdict.Unsupported -> head(verdict.found) + tail(allowUnsupported)
        is Verdict.Unreadable ->
            "Tlaloc's K2 compiler plugin could not determine the running Kotlin compiler's " +
                "version (org.jetbrains.kotlin.config.KotlinCompilerVersion reported " +
                "${verdict.found?.let { "'$it'" } ?: "nothing"}). It is built against Kotlin " +
                "$COMPILED_AGAINST and depends on K2 FIR/IR internals that move between " +
                "Kotlin feature releases, so it cannot confirm it is safe to run here. " +
                tail(allowUnsupported)
    }

    private fun head(found: String): String =
        "Tlaloc's K2 compiler plugin is built against Kotlin $COMPILED_AGAINST and the " +
            "running Kotlin compiler is $found. Supported: $supportedRangeDescription. " +
            "The plugin reads K2's FIR and IR internals (40 org.jetbrains.kotlin.fir " +
            "packages), which are not a stable API and change between Kotlin FEATURE " +
            "releases — the third version component in steps of ten. Running it here " +
            "would most likely fail with a NoSuchMethodError or AbstractMethodError from " +
            "inside the compiler, naming JetBrains classes and not Tlaloc. "

    private fun tail(allowUnsupported: Boolean): String = if (allowUnsupported) {
        "Proceeding anyway because -P plugin:${TlalocCommandLineProcessor.PLUGIN_ID}:" +
            "${TlalocCommandLineProcessor.UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION.optionName}" +
            "=true was passed. Any crash from inside the plugin below this line is expected, " +
            "not a bug — reproduce it on Kotlin $COMPILED_AGAINST before reporting it."
    } else {
        "REFUSING to register, rather than crashing later with no explanation. Either use " +
            "Kotlin $COMPILED_AGAINST, or use a Tlaloc release built against your Kotlin — " +
            "docs/COMPATIBILITY.md states the toolchain this one is pinned to. To try it " +
            "anyway, set `tlaloc { unsafeAllowUnsupportedKotlin.set(true) }` or pass " +
            "-P plugin:${TlalocCommandLineProcessor.PLUGIN_ID}:" +
            "${TlalocCommandLineProcessor.UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION.optionName}=true."
    }

    /**
     * The running compiler's version.
     *
     * Reads `KotlinCompilerVersion.VERSION`, which is a Java `static final String`
     * assigned in a static initialiser (NOT a `ConstantValue` attribute), so it is
     * the RUNNING compiler's version and not the one this plugin compiled against.
     * That distinction is the whole point of the guard, so the fallback reads the
     * same answer straight out of the compiler jar's `META-INF/compiler.version`
     * resource — which needs no Kotlin API at all, and is therefore the one lookup
     * that cannot itself be broken by the API drift being detected.
     */
    fun detectRunningCompilerVersion(): String? =
        runCatching { KotlinCompilerVersion.VERSION }.getOrNull()
            ?: runCatching {
                KotlinVersionGuard::class.java
                    .getResourceAsStream(KotlinCompilerVersion.VERSION_FILE_PATH)
                    ?.use { it.readBytes().decodeToString().trim() }
                    ?.takeIf { it.isNotEmpty() }
            }.getOrNull()

    /**
     * Run the guard. Returns true when the plugin's extensions may be registered.
     *
     * [report] is the sink ([GuardReporter] in production); it is a parameter
     * so the whole decision — verdict, severity and text — is unit-testable without
     * a second Kotlin compiler on the machine, which is the one thing a test on
     * this box cannot conjure.
     */
    fun check(
        found: String?,
        allowUnsupported: Boolean,
        report: (CompilerMessageSeverity, String) -> Unit,
    ): Boolean {
        val verdict = verdict(found)
        val text = message(verdict, allowUnsupported) ?: return true
        report(
            if (allowUnsupported) CompilerMessageSeverity.WARNING else CompilerMessageSeverity.ERROR,
            text,
        )
        return allowUnsupported
    }
}
