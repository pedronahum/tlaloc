// A STANDALONE Gradle build consuming the published io.tlaloc:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/differentiable-physics run
rootProject.name = "tlaloc-differentiable-physics"

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
