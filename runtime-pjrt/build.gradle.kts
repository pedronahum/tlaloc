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
    // §0.4.303 — FFM bindings to libpjrt_c_api.so. JDK 21+ gates native
    // downcalls behind `--enable-native-access`; FFM is preview status in
    // JDK 21 so the API itself is unrestricted only with `--enable-preview`.
    // Both flags drop when we migrate to JDK 22 (FFM stable) — the JDK
    // bump is the §0.4.30N follow-up that simplifies this list.
    jvmArgs("--enable-preview", "--enable-native-access=ALL-UNNAMED")
}
