// A STANDALONE Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/differentiable-physics run
rootProject.name = "tlaloc-differentiable-physics"

pluginManagement {
    repositories {
        // The Tlaloc Gradle plugin is published to Maven, not the Gradle Plugin Portal.
        mavenLocal()
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
