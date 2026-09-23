// §0.4.487 — the GPU inference example: a STANDALONE Gradle build (deliberately
// not included in the root settings) consuming the published io.github.pedronahum:*
// artifacts from mavenLocal, exactly as an external user would.
//
//   ./gradlew publishToMavenLocal                      # at the repo root
//   ./gradlew -p examples/gpu-inference export         # half one: Kotlin writes
//   python3 examples/gpu-inference/serve.py …          # half two: Python serves
rootProject.name = "tlaloc-gpu-inference"

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
