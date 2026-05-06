rootProject.name = "tlaloc"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

// Layer 2.5 §0.4.245+ — vendored Netflix/maestro under third-party/maestro/
// participates as a Gradle composite build. A single `./gradlew build` at
// the Tlaloc root drives both Tlaloc's modules and Maestro's module tree;
// `:maestro-tlaloc` (added in L2.5.1) consumes Tlaloc's :maestro jar via
// composite-build dependency substitution without staging through a local
// Maven repo. See docs/vendoring.md for the upgrade procedure.
//
// `name = "vendored-maestro"` overrides the included build's root project
// name (which is `maestro` upstream) to avoid colliding with Tlaloc's own
// `:maestro` module. From the Tlaloc command line, vendored Maestro tasks
// are invoked as `./gradlew :vendored-maestro:<task>`.
includeBuild("third-party/maestro") {
    name = "vendored-maestro"
}

include(":core")
include(":ir")
include(":autograd")
include(":stablehlo")
include(":compiler-plugin")
include(":benchmarks")
include(":maestro")
include(":runtime-iree")
