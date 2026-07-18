plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.jvm) apply false
}

group = "io.tlaloc"
version = "0.0.1-SNAPSHOT"

// §0.4.355 — packaging: every consumable module publishes to Maven under
// io.tlaloc:<module>:0.0.1-SNAPSHOT. `./gradlew publishToMavenLocal` is the
// onboarding entry point (docs/GETTING_STARTED.md; examples/quickstart is a
// standalone consumer project resolving from mavenLocal). :benchmarks is a
// test harness, not a library — excluded. Maven Central release wiring
// (signing, Sonatype) is a deliberate follow-up.
allprojects {
    group = "io.tlaloc"
    version = "0.0.1-SNAPSHOT"
}
subprojects {
    if (name != "benchmarks") {
        apply(plugin = "maven-publish")
    }
}

// §0.4.41 — make `./gradlew test` run every subproject's tests, not just those
// where a `test` task exists at the subproject level. The root `test` lifecycle
// task historically only picked up `:compiler-plugin:test` (the plain-JVM module);
// KMP modules (`:core`, `:ir`, `:autograd`, `:stablehlo`) expose their tests via
// `jvmTest` / `allTests` instead of `test`, and so were silently skipped. Aliasing
// `test` to `check` ensures the root task covers the whole suite — in particular,
// `./gradlew test` now catches regressions in `:ir`'s PhiCalculus / interpreter
// tests that pre-§0.4.41 slipped through (the §0.4.40 C5-counter-relax introduced
// three latent failures in PhiCalculusC7/8/9 that this change would have caught).
tasks.register("test") {
    dependsOn(subprojects.map { it.tasks.named("check") })
}

// Round-trip tests in :stablehlo (and future :runtime-iree) shell out to
// `stablehlo-translate`, `sdy-opt`, and `iree-compile`, resolving them via
// /usr/bin/which against the inherited PATH. On Apple Silicon those binaries
// live under /opt/homebrew/bin — which is on PATH for interactive shells (via
// the brew shellenv line in ~/.zshrc) but NOT for non-interactive shells (CI,
// Claude Code's tool harness, IDE-spawned gradle daemons). When PATH is
// missing /opt/homebrew/bin the tests' assumeTrue(...) silently self-skip,
// turning a real toolchain regression into a green build.
//
// Inject /opt/homebrew/bin into every Test task's environment when the dir
// exists, so any way of invoking gradle (CLI, IDE, CI) gets the binaries.
// No-op on Linux/Windows where the directory doesn't exist.
subprojects {
    tasks.withType<Test>().configureEach {
        val brewBin = file("/opt/homebrew/bin")
        if (brewBin.isDirectory) {
            val currentPath = System.getenv("PATH") ?: ""
            if (!currentPath.split(":").contains(brewBin.absolutePath)) {
                environment("PATH", "${brewBin.absolutePath}:$currentPath")
            }
        }
    }
}
