// A STANDALONE Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal -x test && ./gradlew -p examples/per-example-gradients run
rootProject.name = "tlaloc-per-example-gradients"

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
