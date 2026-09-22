package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * §0.4.503 (Tier 3, item 3) — Symja as an OPTIONAL dependency, pinned.
 *
 * Until this commit `ir/build.gradle.kts` declared `implementation(libs.symja.core)` in
 * `jvmMain`, which made `org.matheclipse:matheclipse-core` — 8.3 MB, **LGPL-3.0**, with
 * its own transitive tree — a mandatory runtime dependency of every consumer of `:ir`,
 * and therefore of `:autograd`, `:nn` and `:stablehlo`. It is `compileOnly` now.
 *
 * Two things have to be true for that to be a fix rather than a regression, and both
 * are tested here:
 *
 *  1. **Nothing quietly stopped being tested.** Symja is still on the TEST classpath, so
 *     the C6–C9 corollaries, the bake-off and the L-sweep certify what they always did.
 *     [symjaIsPresentOnThisTestClasspath] is the tripwire: if a future build change drops
 *     the test dependency too, this fails instead of the whole coarsening suite silently
 *     degrading to engine-free and staying green.
 *  2. **Absence refuses by name.** [SymbolicEngines.probe] is exercised against a class
 *     loader that genuinely cannot see Symja, and the message names the coordinate, the
 *     licence, and the one line that adds it back.
 *
 * [PhiCalculus.containsLoop] is here too because it is the other half of the same
 * mechanism: it is the predicate that decides whether a body ever needed the CAS.
 */
class SymjaOptionalDependencyTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    // ---------------- 1. the dependency is still on the test classpath ----------------

    @Test
    fun symjaIsPresentOnThisTestClasspath() {
        assertTrue(
            SymbolicEngines.symjaAvailable,
            "Symja must stay on :ir's TEST classpath. Making it `compileOnly` in jvmMain " +
                "must not reduce what the coarsening suite certifies — without it C6-C9 stop " +
                "firing and every engine-backed test would pass by doing nothing.",
        )
        assertNotNull(
            SymbolicEngines.symjaOrNull(),
            "symjaOrNull() must hand back a real engine when Symja is present",
        )
        assertTrue(
            SymbolicEngines.symja("a test asked for an engine").version.startsWith("symja-"),
            "symja(need) must return a SymjaEngine, not something else",
        )
    }

    @Test
    fun theAdvertisedCoordinateIsTheVersionTheBuildResolves() {
        val catalogVersion = System.getProperty("tlaloc.symja.version")
        assertNotNull(
            catalogVersion,
            ":ir's test task must pass -Dtlaloc.symja.version=<libs.versions.symja>",
        )
        assertEquals(
            "org.matheclipse:matheclipse-core:$catalogVersion",
            SymbolicEngines.SYMJA_COORDINATE,
            "the refusal message tells a user to add a dependency LINE. If that line names a " +
                "different version than the one Tlaloc is built and tested against, the advice " +
                "is wrong. Update SymbolicEngines.SYMJA_COORDINATE with the catalog.",
        )
    }

    // ---------------- 2. absence is detected, and refuses by name ----------------

    /** A loader that can see everything except Symja — absence, manufactured. */
    private class WithoutSymja(parent: ClassLoader) : ClassLoader(parent) {
        override fun loadClass(name: String, resolve: Boolean): Class<*> {
            if (name.startsWith("org.matheclipse.")) throw ClassNotFoundException(name)
            return super.loadClass(name, resolve)
        }
    }

    @Test
    fun theProbeSaysAbsentWhenTheClassLoaderCannotSeeSymja() {
        val loader = WithoutSymja(SymbolicEngines::class.java.classLoader)
        assertFalse(
            SymbolicEngines.probe(loader),
            "the probe must report ABSENT for a loader that throws ClassNotFoundException " +
                "for org.matheclipse.* — this is the state a consumer who did not add the " +
                "optional dependency is in",
        )
        assertTrue(
            SymbolicEngines.probe(SymbolicEngines::class.java.classLoader),
            "…and PRESENT for the loader that can see it, or the test above proves nothing",
        )
    }

    @Test
    fun theProbeSaysAbsentForTheBootstrapLoader() {
        // Belt and braces on the probe's own polarity: the platform class loader has
        // never contained a computer algebra system.
        assertFalse(
            SymbolicEngines.probe(ClassLoader.getPlatformClassLoader()),
            "the platform class loader has never contained a computer algebra system; if the " +
                "probe says PRESENT here it is not reading the loader it was handed",
        )
    }

    @Test
    fun theAbsenceMessageNamesTheDependencyTheLicenceAndTheOneLineToAdd() {
        val message = SymbolicEngines.absenceMessage("A test needed to close a loop")
        assertTrue("A test needed to close a loop" in message, "the CALLER's need: $message")
        assertTrue("org.matheclipse:matheclipse-core:" in message, "the coordinate: $message")
        assertTrue("LGPL-3.0" in message, "the licence — this is the fact a policy reviewer needs: $message")
        assertTrue(
            "kotlinCompilerPluginClasspath(\"org.matheclipse:matheclipse-core:" in message,
            "the literal build line, copy-pasteable: $message",
        )
        // §0.4.507 — the regression tripwire, and the reason this assertion is inverted.
        // Until §0.4.507 this message named an `implementation` dependency. That CANNOT work:
        // `SymbolicEngines.symjaAvailable` probes `SymbolicEngines::class.java.classLoader`, and
        // the only production caller is the compiler plugin's `TlalocIrGenerationExtension`, so
        // the loader being asked is the PLUGIN's. A jar on the consumer's own runtime classpath
        // is invisible to it, and the reader would have followed the advice and seen no change.
        // Advice that cannot work is worse than no advice, so the wrong line is now pinned out.
        assertFalse(
            "implementation(\"org.matheclipse" in message,
            "must not send the reader to their own runtime classpath: $message",
        )
        assertTrue("optional" in message.lowercase(), "that it is optional and why: $message")
        assertTrue(
            "does not modify or redistribute" in message,
            "the linking-only statement that makes Apache-2.0 + LGPL-3.0 sound: $message",
        )
    }

    @Test
    fun theProbeTreatsAHalfPresentSymjaAsAbsent() {
        // A LinkageError (wrong Symja version, missing transitive dependency) is not
        // "present" in any useful sense, and swallowing it into `true` would produce a
        // NoClassDefFoundError later from a caller that had been told everything was fine.
        val loader = object : ClassLoader(SymbolicEngines::class.java.classLoader) {
            override fun loadClass(name: String, resolve: Boolean): Class<*> {
                if (name == SymbolicEngines.SYMJA_PROBE_CLASS) throw NoClassDefFoundError(name)
                return super.loadClass(name, resolve)
            }
        }
        assertFalse(SymbolicEngines.probe(loader), "a LinkageError from the probe means ABSENT")
    }

    // ---------------- 3. containsLoop — the predicate that decides "needed the CAS" ----------------

    private fun loopFreeFunction() = DxirBuilder.function("loopFree") {
        val x = param("x", f32s)
        val y = op(OpKind.MUL, listOf(x, x), f32s)
        listOf(op(OpKind.ADD, listOf(y, const(1f, f32s)), f32s))
    }

    private fun whileFunction() = DxirBuilder.function("withWhile") {
        val p = param("p", f32s)
        val n = param("n", i32s)
        val zero = const(0, i32s)
        val w = whileOp(
            inits = listOf(p, zero),
            cond = { args ->
                yields(op(OpKind.STEP, listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)), boolS))
            },
            body = { args ->
                yields(
                    op(OpKind.MUL, listOf(const(2f, f32s), args[0]), f32s),
                    op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s),
                )
            },
        )
        listOf(w.result(0))
    }

    @Test
    fun containsLoopIsFalseWithoutOneAndTrueWithOne() {
        assertFalse(
            PhiCalculus.containsLoop(loopFreeFunction()),
            "a straight-line body never needed a symbolic engine, and must not be blamed on one",
        )
        assertTrue(
            PhiCalculus.containsLoop(whileFunction()),
            "a WHILE is exactly the shape only the engine-backed corollaries can close",
        )
    }

    @Test
    fun containsLoopSeesAWhileNestedInsideAnIfRegion() {
        // §0.4.128's LoopInvariant rewrite produces WHILEs inside IF arms; a predicate
        // that only looked at the top-level body would miss them.
        val fn = DxirBuilder.function("nested") {
            val p = param("p", f32s)
            val pred = op(OpKind.STEP, listOf(p), boolS)
            val ifOp = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val inner = whileOp(
                        inits = listOf(p, const(0, i32s)),
                        cond = { args ->
                            yields(
                                op(
                                    OpKind.STEP,
                                    listOf(op(OpKind.SUB, listOf(const(3, i32s), args[1]), i32s)),
                                    boolS,
                                ),
                            )
                        },
                        body = { args ->
                            yields(
                                op(OpKind.MUL, listOf(const(2f, f32s), args[0]), f32s),
                                op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s),
                            )
                        },
                    )
                    yields(inner.result(0))
                },
                elseRegion = region { yields(p) },
            )
            listOf(ifOp.result(0))
        }
        assertTrue(
            PhiCalculus.containsLoop(fn),
            "containsLoop must recurse; a WHILE the predicate cannot see is a refusal that " +
                "never fires",
        )
    }

    @Test
    fun containsLoopIsFalseAfterC5UnrollsAConcreteTripCount() {
        // The asymmetry the refusal depends on: a concrete-n loop is closed by C5, which
        // is ENGINE-FREE. So `examples/differentiable-physics` — whose loop bound is a
        // `const val` — keeps working with no Symja on the classpath at all, and must
        // never see the refusal.
        val fn = DxirBuilder.function("concreteTrip") {
            val p = param("p", f32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(p, zero),
                cond = { args ->
                    yields(
                        op(
                            OpKind.STEP,
                            listOf(op(OpKind.SUB, listOf(const(3, i32s), args[1]), i32s)),
                            boolS,
                        ),
                    )
                },
                body = { args ->
                    yields(
                        op(OpKind.MUL, listOf(const(2f, f32s), args[0]), f32s),
                        op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s),
                    )
                },
            )
            listOf(w.result(0))
        }
        assertTrue(PhiCalculus.containsLoop(fn), "…before coarsening, the loop is there")
        val engineFree = PhiCalculus.apply(fn, engine = null)
        assertFalse(
            PhiCalculus.containsLoop(engineFree),
            "C5 unrolls a concrete trip count with no engine, so no loop survives and the " +
                "missing-CAS refusal must not fire",
        )
    }
}
