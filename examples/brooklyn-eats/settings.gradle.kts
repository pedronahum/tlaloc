// A standalone Gradle build consuming the published io.github.pedronahum:* artifacts
// (0.1.0-alpha04 on Maven Central; mavenLocal is searched first). Run from the repo root:
//   ./gradlew -p examples/brooklyn-eats build
rootProject.name = "tlaloc-brooklyn-eats"

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
