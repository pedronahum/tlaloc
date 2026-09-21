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
    // §0.4.478 (H3c-1). Gradle forks a Test JVM at a 512 MB default heap, and
    // HfLlamaCheckpointTest's lane B decodes REAL tensors out of a 2.2 GB
    // TinyLlama checkpoint — `model.embed_tokens.weight` alone is 32000x2048
    // bf16, a 131 MB ShortArray decoded from a 131 MB read buffer. 2 GB leaves
    // room for that pair without making the number a tuning knob. REJECTED:
    // probing only small tensors to stay inside the default — the probe that
    // matters most is the LAST element of the LARGEST tensor, because it is
    // the one that fails when the data_offsets base or a short read is wrong.
    // §0.4.479 (H3c-2) raised it again, to 8 GB, for a different reason:
    // HfLlamaRealDecodeParityTest STAGES a real checkpoint's weights as f32
    // FloatArrays. Two real TinyLlama layers plus the [32000, 2048] embedding
    // table and the untied head is ~880 MB of staged f32, and each transposed
    // Linear exists twice for the duration of its transpose. REJECTED:
    // certifying against a tiny-random model to stay inside 2 GB — §0.4.478
    // already paid for a real checkpoint precisely so the numbers would be a
    // real model's, and a parity claim against a stub certifies the stub.
    maxHeapSize = "8g"
}
