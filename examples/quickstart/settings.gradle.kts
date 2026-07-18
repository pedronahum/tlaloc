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
