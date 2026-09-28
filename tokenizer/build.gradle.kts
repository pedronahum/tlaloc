import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // Built on JDK 25, emitting Java 21 bytecode, like :core. The per-module
    // table and the gate that reads the .class files back are in the root
    // build.gradle.kts (`tlalocJvmTargets` / `verifyJvmTarget`).
    jvmToolchain(25)

    jvm {
        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_21)
                }
            }
        }
        // `-Xjdk-release=21` on main: a JDK 22+ API call fails the compile
        // instead of failing a JDK 21 consumer at run time.
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
                // parseJson reads tokenizer.json; no :core type appears in this module's API.
                implementation(project(":core"))
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
    // The golden tests keep five loaded tokenizers; reading Gemma 4's or Muse
    // Glimmer's tokenizer.json holds about 200 MB of parse tree for a moment.
    maxHeapSize = "2g"
}
