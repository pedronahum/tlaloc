// Fine-tune a Hugging Face model on the GPU from Kotlin: a STANDALONE Gradle
// build (not included in the root settings) consuming the published
// io.github.pedronahum:* artifacts from mavenLocal, as an external user would.
// Run from the repo root:
//   ./gradlew publishToMavenLocal -x test && ./gradlew -p examples/fine-tune run
rootProject.name = "tlaloc-fine-tune"

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
