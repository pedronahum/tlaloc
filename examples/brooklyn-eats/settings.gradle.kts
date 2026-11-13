// A standalone Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal. Run from the repo root:
//   ./gradlew publishToMavenLocal -x test && ./gradlew -p examples/brooklyn-eats build
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
