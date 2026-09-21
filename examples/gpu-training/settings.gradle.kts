// §0.4.486 — the GPU training example: a STANDALONE Gradle build (deliberately
// not included in the root settings) consuming the published io.tlaloc:*
// artifacts from mavenLocal, exactly as an external user would.
// Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/gpu-training run
rootProject.name = "tlaloc-gpu-training"

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
