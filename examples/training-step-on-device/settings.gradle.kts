// A STANDALONE Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal -x test && ./gradlew -p examples/training-step-on-device run
rootProject.name = "tlaloc-training-step-on-device"

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
