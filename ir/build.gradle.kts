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
        commonMain {
            dependencies {
                implementation(project(":core"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        // Stage B coarsening — Symja (org.matheclipse:matheclipse-core, LGPL-3.0) is the
        // default symbolic engine per docs/STAGE_B_PLAN.md §5. JVM-only because Symja
        // depends on java.* stdlib classes; commonMain (and Kotlin/Native, JS, WASM
        // targets in the future) stays Symja-free since SymbolicEngine.kt is interface
        // -only with opaque sealed handles.
        jvmMain {
            dependencies {
                implementation(libs.symja.core)
            }
        }
        jvmTest {
            dependencies {
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
