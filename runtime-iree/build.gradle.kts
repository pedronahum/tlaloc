import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // §0.4.503 — Stays at 25 deliberately, and NOT because of FFM — this backend shells out to
    // `iree-compile` / `iree-run-module` and imports no `java.lang.foreign`. It could
    // be lowered; nothing in this repository certifies an IREE run on a JDK 21, so
    // lowering it would be publishing an unchecked claim. Recorded as a ⬜ row in
    // docs/ALPHA_PLAN.md rather than done silently.
    jvmToolchain(25)

    jvm {
        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_25)
                }
            }
        }
    }

    sourceSets {
        jvmMain {
            dependencies {
                // §0.4.287 — runOnIree bridge needs DType (:core), DxirFunction
                // and DxirType (:ir), and the StableHLO emit pipeline
                // (:stablehlo). The §0.4.284 subprocess facade was dep-free; the
                // bridge is the layering boundary where IREE meets the rest of
                // the Tlaloc compiler. Mirrors :stablehlo's commonMain deps.
                implementation(project(":core"))
                implementation(project(":ir"))
                implementation(project(":stablehlo"))
            }
        }
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
