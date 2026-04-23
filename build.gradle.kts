plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.jvm) apply false
}

group = "io.tlaloc"
version = "0.0.1-SNAPSHOT"

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
