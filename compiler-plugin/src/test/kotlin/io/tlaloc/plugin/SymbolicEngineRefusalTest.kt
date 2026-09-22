package io.tlaloc.plugin

import io.tlaloc.ir.passes.SymbolicEngines
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.503 (Tier 3, item 3) — the compiler plugin's half of "Symja is optional now".
 *
 * `:ir` declares matheclipse-core `compileOnly`, so ABSENT is the default state on a
 * consumer's classpath. The plugin's old behaviour — `catch (_: Throwable) { null }`
 * around `SymjaEngine()`, then coarsen engine-free — would turn that default into a
 * silent loss of the C6–C9 corollaries, which is precisely the silent degradation the
 * house rules forbid.
 *
 * What is pinned here is the DECISION, as a pure function of its three inputs, and the
 * text it produces. What is NOT pinned, and is stated rather than glossed: no test in
 * this repository compiles a loop-bearing `grad {}` against a classpath with no Symja
 * on it. Symja is on `:compiler-plugin`'s test classpath deliberately (removing it
 * would make every coarsening test here pass by doing nothing), and the K2 harness
 * shares the test JVM's classpath, so the absent case cannot be staged without a
 * second compiler process. Recorded as a 🧪 row in docs/ALPHA_PLAN.md.
 */
class SymbolicEngineRefusalTest {

    @Test
    fun symjaIsOnThisModulesTestClasspath() {
        // The tripwire. :ir's `compileOnly` means this arrives only because
        // compiler-plugin/build.gradle.kts asks for it explicitly. If that line is ever
        // removed, every coarsening test in this module would quietly stop exercising
        // C6-C9 and stay green — so this failing first is the point.
        assertTrue(
            SymbolicEngines.symjaAvailable,
            "Symja must be on :compiler-plugin's test classpath, or the C6-C9 corollaries " +
                "stop firing and the coarsening tests certify nothing",
        )
    }

    @Test
    fun noRefusalWhenSymjaIsPresent() {
        assertNull(
            TlalocIrGenerationExtension.missingSymbolicEngineMessage(
                fnName = "f",
                symjaAvailable = true,
                loopSurvivedCoarsening = true,
            ),
            "with an engine present, a rejected loop-bearing primal is a Tlaloc limitation " +
                "and must not be blamed on a missing dependency",
        )
    }

    @Test
    fun noRefusalWhenNoLoopSurvivedCoarsening() {
        assertNull(
            TlalocIrGenerationExtension.missingSymbolicEngineMessage(
                fnName = "f",
                symjaAvailable = false,
                loopSurvivedCoarsening = false,
            ),
            "most programs never need the CAS. A reverse-transform refusal on a loop-free " +
                "primal has nothing to do with Symja, and saying so would send the user to " +
                "add a dependency that cannot help.",
        )
    }

    @Test
    fun theRefusalNamesTheFunctionTheDependencyTheLicenceAndTheLine() {
        val text = TlalocIrGenerationExtension.missingSymbolicEngineMessage(
            fnName = "hmcLeapfrog",
            symjaAvailable = false,
            loopSurvivedCoarsening = true,
        )
        assertNotNull(text, "absent engine + surviving loop is the case that must refuse")
        assertTrue("hmcLeapfrog" in text, "the function by name: $text")
        assertTrue("WHILE" in text, "what it found in the primal: $text")
        assertTrue("C6-C9" in text, "which rewrites would have needed the engine: $text")
        assertTrue("org.matheclipse:matheclipse-core:" in text, "the coordinate: $text")
        assertTrue("LGPL-3.0" in text, "the licence: $text")
        assertTrue(
            "kotlinCompilerPluginClasspath(\"org.matheclipse:matheclipse-core:" in text,
            "the one line to add: $text",
        )
        // §0.4.507 — this assertion used to demand `implementation(...)`, and so did the twin in
        // :ir's SymjaOptionalDependencyTest. Both certified advice that could not work: the CAS is
        // resolved against the COMPILER PLUGIN's class loader (this very extension is the only
        // production caller), so a jar on the consumer's runtime classpath is invisible to the
        // probe and the reader would have added the line and seen the identical refusal again.
        // Two gates agreeing on a false claim is why this one is now stated in the negative too.
        assertFalse(
            "implementation(\"org.matheclipse" in text,
            "must not send the reader to their own runtime classpath: $text",
        )
    }
}
