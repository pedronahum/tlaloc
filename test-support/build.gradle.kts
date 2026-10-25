import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

// Test code shared by other modules' tests: the op catalogue (one entry per OpKind,
// with samples and the surfaces each op supports), recorded goldens and the tolerance
// table. Publishes nothing (tlalocUnpublishedModules in the root build).
kotlin {
    jvmToolchain(25)

    // Java 21, with -Xjdk-release=21 on main: the 21-targeted modules' suites use this
    // module and run on a JDK 21 in CI.
    jvm {
        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_21)
                }
            }
        }
        compilations.named("main").configure {
            compileTaskProvider.configure {
                compilerOptions {
                    freeCompilerArgs.add("-Xjdk-release=21")
                }
            }
        }
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":core"))
                api(project(":ir"))
                // Goldens skips a test through JUnit's TestAbortedException.
                implementation(kotlin("test-junit5"))
            }
        }
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotest.runner.junit5)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
