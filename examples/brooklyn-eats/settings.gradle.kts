// A standalone Gradle build consuming the published io.github.pedronahum:* artifacts
// (0.1.0-alpha05 on Maven Central; mavenLocal is searched first). Run from the repo root:
//   ./gradlew -p examples/brooklyn-eats build
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

// The build compiles and runs on JDK 25 (kotlin { jvmToolchain(25) }). When no JDK 25 is
// installed, Gradle downloads one through the Foojay toolchain resolver.
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

rootProject.name = "tlaloc-brooklyn-eats"

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
