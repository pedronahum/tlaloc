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
                // Mirrors :runtime-iree's main-side dep set: DxirFunction +
                // DxirType (:ir), DType (:core), and the StableHLO emit
                // pipeline (:stablehlo) so runOnPjrt can take a
                // DxirFunction directly.
                implementation(project(":core"))
                implementation(project(":ir"))
                implementation(project(":stablehlo"))
                // KPTX v1.4 — kernel dispatch inside PJRT executables needs
                // the CUDA driver bindings (cuModuleLoadData / cuLaunchKernel).
                api(project(":runtime-cuda"))
            }
        }
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
                // §0.4.344 — the backward-kernel oracle test registers
                // DSL-emitted PTX from the :kptx production kernel library.
                implementation(project(":kptx"))
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    // §0.4.303 — FFM bindings to libpjrt_c_api.so. JDK 22+ gates native
    // downcalls behind `--enable-native-access`. FFM became stable in JDK 22
    // (JEP 454) so `--enable-preview` is no longer required as of the §0.4.311
    // JDK 25 bump.
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
