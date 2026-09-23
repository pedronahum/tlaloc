plugins {
    `java-platform`
}

// tlaloc-bom: one version for every Tlaloc artifact.
//
//   implementation(platform("io.github.pedronahum:tlaloc-bom:<version>"))
//   implementation("io.github.pedronahum:tlaloc-autograd")
//
// The constraints are written from the module list below. For each Kotlin
// Multiplatform module both coordinates are constrained: `tlaloc-<m>` (what a
// Gradle consumer names) and `tlaloc-<m>-jvm` (what a Maven consumer resolves).
// `verifyBomCoverage` checks the generated POM against the artifact ids every
// other module actually publishes, so a module added to the build without a line
// here fails `check`.
val kmpModules = listOf(
    "core", "ir", "autograd", "nn", "stablehlo", "maestro",
    "runtime-pjrt", "runtime-cuda", "runtime-iree", "kptx",
)
val jvmModules = listOf("compiler-plugin", "gradle-plugin")

dependencies {
    constraints {
        val v = project.version.toString()
        val g = project.group.toString()
        kmpModules.forEach {
            api("$g:tlaloc-$it:$v")
            api("$g:tlaloc-$it-jvm:$v")
        }
        jvmModules.forEach { api("$g:tlaloc-$it:$v") }
    }
}

publishing {
    publications {
        create<MavenPublication>("bom") {
            from(components["javaPlatform"])
        }
    }
}
