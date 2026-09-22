package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.SymbolicEngines
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.507 — the third Symja tripwire, added because the count was wrong.
 *
 * §0.4.503 made Symja `compileOnly` in `:ir`, so it leaves every published POM, and kept it on
 * the three test classpaths that certify coarsening: `:ir`'s `jvmTest`, `:compiler-plugin`'s
 * `testImplementation`, and `:benchmarks`' `jvmTest` (`benchmarks/build.gradle.kts`). Two of
 * those three had a tripwire — [io.tlaloc.ir.passes.SymjaOptionalDependencyTest] and
 * `io.tlaloc.plugin.SymbolicEngineRefusalTest` each assert the CAS is present on their own
 * module's test classpath — while README.md and docs/ALPHA_PLAN.md both claimed three.
 *
 * This is the missing one. It exists because of the failure mode a `compileOnly` dependency
 * creates: dropping `implementation(libs.symja.core)` from this module would not break the
 * build, it would silently turn every coarsening benchmark here into a measurement of the
 * engine-free fallback path — a green suite measuring the wrong thing. That is precisely the
 * "nothing silently degrades" rule, applied to a benchmark rather than to a user's program.
 */
class SymjaBenchmarkClasspathTest {
    @Test
    fun symjaIsOnTheBenchmarkTestClasspath() {
        assertTrue(
            SymbolicEngines.symjaAvailable,
            "the :benchmarks jvmTest classpath must carry Symja, or every coarsening benchmark " +
                "in this module silently measures the engine-free fallback instead of the CAS " +
                "path it claims to measure. Restore `implementation(libs.symja.core)` in " +
                "benchmarks/build.gradle.kts.",
        )
    }
}
