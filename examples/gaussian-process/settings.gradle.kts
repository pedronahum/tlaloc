// A STANDALONE Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/gaussian-process run
rootProject.name = "tlaloc-gaussian-process"

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
