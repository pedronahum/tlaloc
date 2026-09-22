// §0.4.355 — the Tlaloc onboarding quickstart: a STANDALONE Gradle build
// (deliberately not included in the root settings) consuming the published
// io.tlaloc:* artifacts from mavenLocal, exactly as an external user would.
// Run from the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/quickstart run
// or use scripts/onboarding-smoke.sh, which does both.
rootProject.name = "tlaloc-quickstart"

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

// §0.4.503 — `runOnJdk21` needs a JDK 21 launcher, and Gradle's auto-detection does
// not look in `~/.local/jdks`. The location comes from an environment variable so no
// machine-specific path is committed:
//
//     export JDK21_HOME=/path/to/jdk-21
//
// Nothing else in this project needs it; `run` uses the toolchain JDK 25.
