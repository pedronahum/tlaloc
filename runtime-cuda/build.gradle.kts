import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // §0.4.503 — FFM (JEP 454) again: `CudaDriverFfm` binds libcuda with `Linker`/`Arena`/
    // `MemorySegment`. Stays at 25 — see `tlalocJvmTargets` in the root build.
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
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
                // KPTX v2.1 (§0.4.338) — PtxIrDriverJitTest feeds IR-emitted
                // PTX through the driver JIT.
                implementation(project(":kptx"))
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    // KPTX v1.2 — FFM bindings to libcuda.so.1. Same native-access gate as
    // :runtime-pjrt (JEP 454, stable since JDK 22).
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
