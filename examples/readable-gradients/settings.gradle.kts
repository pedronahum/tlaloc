// §0.4.488 — examples/readable-gradients: a STANDALONE Gradle build (deliberately
// not included in the root settings) consuming the published io.github.pedronahum:* artifacts
// from mavenLocal, exactly as an external user's project would.
//
// Run from the repo root:
//   ./gradlew publishToMavenLocal
//   ./gradlew -p examples/readable-gradients run
rootProject.name = "tlaloc-readable-gradients"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
