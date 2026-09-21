// §0.4.489 — the TPU example: a STANDALONE Gradle build (deliberately not
// included in the root settings) consuming the published io.tlaloc:* artifacts
// from mavenLocal, exactly as an external user would.
//
// From the repo root:
//   ./gradlew publishToMavenLocal && ./gradlew -p examples/tpu run
//
// On a machine without a TPU that prints the host half and then a named skip.
// On a Cloud TPU VM with TLALOC_PJRT_PLUGIN_PATH pointing at libtpu.so it runs
// the device half too. See README.md — and read its first paragraph before you
// believe anything: nothing in this example has ever executed on a TPU.
rootProject.name = "tlaloc-tpu"

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
