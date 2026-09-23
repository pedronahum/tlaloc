// §0.4.485 — a STANDALONE Gradle build (deliberately NOT included in the repo
// root's settings.gradle.kts) that consumes the published io.github.pedronahum:* artifacts
// from mavenLocal, exactly as an external user's project would.
//
// Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/named-indices run
rootProject.name = "tlaloc-named-indices"

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
