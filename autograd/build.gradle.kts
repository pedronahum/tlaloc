import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // §0.4.503 — BUILD on JDK 25, EMIT Java 21 bytecode. The two numbers differ on
    // purpose. §0.4.311 moved the whole repository to JDK 25 for ONE reason: the
    // Foreign Function & Memory API went stable in JEP 454, and Tlaloc's PJRT and
    // CUDA bindings are FFM. That reason applies to `:runtime-pjrt`, `:runtime-cuda`
    // and `:kptx` — and to nothing in this module. Emitting 25 bytecode here made
    // JDK 25 a floor for every consumer of the LIBRARY, a cost with no matching
    // benefit. The authoritative per-module table, and the gate that reads the
    // emitted .class files back, live in the root build.gradle.kts
    // (`tlalocJvmTargets` / `verifyJvmTarget`).
    jvmToolchain(25)

    jvm {
        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_21)
                }
            }
        }
        // `-Xjdk-release=21` on the MAIN compilation is what makes "this jar runs on
        // a JDK 21" a checked claim rather than a bytecode-version coincidence.
        // Without it the compiler still resolves against JDK 25's class library, so a
        // call to a JDK 22+ method would compile happily into 21 bytecode and then
        // fail at run time with NoSuchMethodError — the exact silent degradation the
        // house rules forbid. MAIN only: the test compilations run on the toolchain
        // JDK and are free to use everything it has.
        compilations.named("main").configure {
            compileTaskProvider.configure {
                compilerOptions {
                    freeCompilerArgs.add("-Xjdk-release=21")
                }
            }
        }
    }

    sourceSets {
        commonMain {
            dependencies {
                implementation(project(":core"))
                implementation(project(":ir"))
            }
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
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
