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
            dependencies {}
        }
        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        jvmTest {
            dependencies {
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.property)
                implementation(libs.kotest.runner.junit5)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
