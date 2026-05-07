import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    jvmToolchain(21)

    jvm {
        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_21)
                }
            }
        }
    }

    sourceSets {
        jvmMain {
            dependencies {
                // Mirrors :runtime-iree's main-side dep set: DxirFunction +
                // DxirType (:ir), DType (:core), and the StableHLO emit
                // pipeline (:stablehlo) so runOnPjrt can take a
                // DxirFunction directly.
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
