import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
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
