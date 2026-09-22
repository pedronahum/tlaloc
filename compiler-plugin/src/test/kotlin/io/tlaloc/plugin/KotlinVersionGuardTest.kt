package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.503 (Tier 3, item 2) — the Kotlin version guard, pinned.
 *
 * The defect being closed: this plugin imports 40 `org.jetbrains.kotlin.fir.*`
 * packages — internal K2 API with no stability contract — and until now NOTHING
 * checked which compiler it was running inside. A user on 2.2.x or 2.4.x got a raw
 * `NoSuchMethodError` from the middle of `compileKotlin`, naming JetBrains classes
 * and never Tlaloc.
 *
 * What these tests can and cannot certify, stated rather than implied. They cannot
 * summon a second Kotlin compiler onto this machine, so no test here runs the plugin
 * under 2.2.x. What they DO certify is every decision the guard makes as a pure
 * function of a version string ([KotlinVersionGuard.verdict], [KotlinVersionGuard.check]),
 * and — the link that would otherwise be assumed — that the version the guard
 * actually reads at run time from the running compiler is the one this repository
 * builds with, and that the constant it compares against is the version catalog's
 * and not a second copy of it.
 */
class KotlinVersionGuardTest {

    // ---------------- the constant is not a second copy of the version ----------------

    @Test
    fun `COMPILED_AGAINST matches the version catalog`() {
        val fromCatalog = System.getProperty("tlaloc.kotlin.version")
        assertNotNull(
            fromCatalog,
            "the :compiler-plugin test task must pass -Dtlaloc.kotlin.version=<libs.versions.kotlin>; " +
                "without it this test proves nothing",
        )
        assertEquals(
            fromCatalog, KotlinVersionGuard.COMPILED_AGAINST,
            "KotlinVersionGuard.COMPILED_AGAINST is a SECOND copy of the Kotlin version. If it " +
                "drifts from gradle/libs.versions.toml the guard refuses the compiler this " +
                "repository itself builds with. Update both, and the range quoted in README.md, " +
                "docs/GETTING_STARTED.md and docs/COMPATIBILITY.md.",
        )
    }

    @Test
    fun `the guard does not refuse the compiler this repository builds with`() {
        val running = KotlinVersionGuard.detectRunningCompilerVersion()
        assertNotNull(running, "KotlinCompilerVersion reported no version at all")
        assertNotNull(
            KotlinVersionGuard.parse(running),
            "the running compiler reported '$running', which the guard cannot parse",
        )
        assertEquals(
            KotlinVersionGuard.Verdict.Supported, KotlinVersionGuard.verdict(running),
            "the guard refuses the running compiler ($running). Either the Kotlin version was " +
                "bumped without widening the guard, or the range rule is wrong.",
        )
        assertNull(
            KotlinVersionGuard.message(KotlinVersionGuard.verdict(running), allowUnsupported = false),
            "a supported version must produce NO message — §0.4.499's rule is that a working " +
                "build is silent",
        )
    }

    // ---------------- the range rule ----------------

    @Test
    fun `the whole bugfix family of the built-against feature release is supported`() {
        // Kotlin numbers bugfix releases by ones above a feature release: 2.3.21 and
        // 2.3.29 are bugfixes OF 2.3.20 and do not move K2 internals.
        for (patch in 20..29) {
            val v = "2.3.$patch"
            assertEquals(
                KotlinVersionGuard.Verdict.Supported, KotlinVersionGuard.verdict(v),
                "$v is a bugfix release of 2.3.20 and must be accepted — refusing it would " +
                    "break every user the day JetBrains ships a patch",
            )
        }
    }

    @Test
    fun `a different feature release in the same minor is refused`() {
        // The trap this rule exists for: 2.3.0 and 2.3.10 look like "2.3.x" and are
        // NOT the same feature release as 2.3.20.
        for (v in listOf("2.3.0", "2.3.9", "2.3.10", "2.3.19", "2.3.30", "2.3.31")) {
            val verdict = KotlinVersionGuard.verdict(v)
            assertTrue(
                verdict is KotlinVersionGuard.Verdict.Unsupported,
                "$v is a different Kotlin FEATURE release than 2.3.20 and must be refused; got $verdict",
            )
        }
    }

    @Test
    fun `a different minor or major is refused`() {
        for (v in listOf("2.2.20", "2.4.20", "1.9.20", "3.0.20")) {
            assertTrue(
                KotlinVersionGuard.verdict(v) is KotlinVersionGuard.Verdict.Unsupported,
                "$v must be refused",
            )
        }
    }

    @Test
    fun `a pre-release of the supported feature release is accepted`() {
        for (v in listOf("2.3.20-RC", "2.3.20-RC2", "2.3.21-Beta1", "2.3.20-dev-1234")) {
            assertEquals(
                KotlinVersionGuard.Verdict.Supported, KotlinVersionGuard.verdict(v),
                "$v carries the supported numeric triple; the suffix is a build channel, not a " +
                    "different set of internals",
            )
        }
    }

    @Test
    fun `an unreadable version is refused rather than assumed good`() {
        for (v in listOf(null, "", "   ", "2.3", "2.3.x", "banana", "2.3.20.1", "-1.2.3")) {
            val verdict = KotlinVersionGuard.verdict(v)
            assertTrue(
                verdict is KotlinVersionGuard.Verdict.Unreadable,
                "'$v' is not a readable Kotlin version; a guard that waves through what it " +
                    "cannot read is not a guard. Got $verdict",
            )
        }
    }

    @Test
    fun `parse splits the numeric triple from the suffix`() {
        assertEquals(KotlinVersionGuard.Version(2, 3, 20, null), KotlinVersionGuard.parse("2.3.20"))
        assertEquals(KotlinVersionGuard.Version(2, 3, 20, "RC2"), KotlinVersionGuard.parse("2.3.20-RC2"))
        assertEquals(2, KotlinVersionGuard.parse("2.3.20")!!.featureFamily)
        assertEquals(1, KotlinVersionGuard.parse("2.3.19")!!.featureFamily)
        assertEquals(0, KotlinVersionGuard.parse("2.3.9")!!.featureFamily)
    }

    @Test
    fun `the supported range reads as two concrete numbers`() {
        assertEquals("2.3.20 through 2.3.29", KotlinVersionGuard.supportedRangeDescription)
    }

    // ---------------- the refusal itself ----------------

    @Test
    fun `an unsupported version is an ERROR that names both versions and the opt-out`() {
        val reported = mutableListOf<Pair<CompilerMessageSeverity, String>>()
        val mayRegister = KotlinVersionGuard.check(
            found = "2.2.20",
            allowUnsupported = false,
        ) { severity, text -> reported += severity to text }

        assertFalse(mayRegister, "an unsupported Kotlin version must stop the plugin registering")
        assertEquals(1, reported.size, "exactly one diagnostic; got $reported")
        val (severity, text) = reported.single()
        assertEquals(CompilerMessageSeverity.ERROR, severity, "a refusal is an ERROR, not a warning")
        // By name, all four of them: what it found, what it was built against, the
        // range, and the flag that overrides it.
        assertTrue("2.2.20" in text, "the message must name the version it FOUND: $text")
        assertTrue("2.3.20" in text, "the message must name the version it was BUILT AGAINST: $text")
        assertTrue("2.3.20 through 2.3.29" in text, "the message must name the RANGE: $text")
        assertTrue(
            "unsafeAllowUnsupportedKotlin" in text,
            "the message must name the opt-out option: $text",
        )
        assertTrue("Tlaloc" in text, "the message must say whose refusal this is: $text")
    }

    @Test
    fun `an unreadable version is an ERROR that says so instead of naming a fake version`() {
        val reported = mutableListOf<Pair<CompilerMessageSeverity, String>>()
        val mayRegister =
            KotlinVersionGuard.check(found = null, allowUnsupported = false) { s, t -> reported += s to t }
        assertFalse(mayRegister)
        val (severity, text) = reported.single()
        assertEquals(CompilerMessageSeverity.ERROR, severity)
        assertTrue("could not determine" in text, "the message must say what it could not do: $text")
        assertTrue("KotlinCompilerVersion" in text, "…and where it looked: $text")
        assertTrue("unsafeAllowUnsupportedKotlin" in text, "…and the opt-out: $text")
    }

    @Test
    fun `the opt-out downgrades the refusal to a WARNING and lets registration proceed`() {
        val reported = mutableListOf<Pair<CompilerMessageSeverity, String>>()
        val mayRegister = KotlinVersionGuard.check(
            found = "2.4.0",
            allowUnsupported = true,
        ) { severity, text -> reported += severity to text }

        assertTrue(mayRegister, "the opt-out must let the plugin register")
        val (severity, text) = reported.single()
        assertEquals(
            CompilerMessageSeverity.WARNING, severity,
            "with the opt-out the refusal becomes a warning — the user asked for this",
        )
        assertTrue("2.4.0" in text, "the warning still names the version: $text")
        assertTrue(
            "expected, not a bug" in text,
            "the warning must say that a later crash is the expected outcome: $text",
        )
        assertFalse(
            "REFUSING" in text,
            "the opt-out path must not claim to be refusing when it is proceeding: $text",
        )
    }

    @Test
    fun `a supported version reports nothing even with the opt-out set`() {
        val reported = mutableListOf<Pair<CompilerMessageSeverity, String>>()
        for (allow in listOf(false, true)) {
            val mayRegister = KotlinVersionGuard.check(
                found = KotlinVersionGuard.COMPILED_AGAINST,
                allowUnsupported = allow,
            ) { severity, text -> reported += severity to text }
            assertTrue(mayRegister)
        }
        assertTrue(
            reported.isEmpty(),
            "the escape hatch must not become a source of noise on a supported compiler; got $reported",
        )
    }
}
