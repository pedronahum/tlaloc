// A STANDALONE Gradle build consuming the published io.github.pedronahum:* artifacts from
// mavenLocal, exactly as an external user would. Run from the repo root:
//   ./gradlew publishToMavenLocal -x test && ./gradlew -p examples/spark-inference run
rootProject.name = "tlaloc-spark-inference"

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
