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
                // §0.4.274 — Phase 3 step 1 of the dual-track Llama-decoder
                // benchmark plan needs :stablehlo emit coverage on the
                // LlamaDecoderPrimal. Test-only dependency; the production
                // :benchmarks code stays on :core + :ir.
                implementation(project(":stablehlo"))
                // §0.4.286 — IREE compile probe of the LlamaDecoder MLIR:
                // first-contact assertion that the CPU baseline pipeline
                // produces valid IREE input (exit=0 from iree-compile via
                // IreeRuntime.compile). Test-only dependency.
                implementation(project(":runtime-iree"))
                // §0.4.308 — pure-Kotlin PJRT-XLA-CUDA benchmark via
                // PjrtSession (the FFM equivalent of the §0.4.299 Python
                // spike script). Test-only dependency.
                implementation(project(":runtime-pjrt"))
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    // §0.4.308 — :benchmarks:jvmTest now exercises PjrtSession (FFM bindings
    // via java.lang.foreign). Same flag `:runtime-pjrt` uses; FFM became
    // stable in JDK 22 (JEP 454) so `--enable-preview` was dropped at §0.4.311.
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}

// §0.4.236 — `./gradlew :benchmarks:dumpHarnessResults` runs
// `HeadToHeadHarnessMain.kt` against the jvmTest classpath, producing
// `build/harness-results-tlaloc.{csv,json}` for the user's cross-framework
// comparison workflow (Python references + `harness/python/aggregate.py`).
tasks.register<JavaExec>("dumpHarnessResults") {
    group = "verification"
    description = "Run HeadToHeadHarnessMain to dump Tlaloc-side harness results to build/"
    // The jvmTest compilation's classpath gives us everything: compiled main
    // sources of dependencies, compiled test sources of :benchmarks itself,
    // plus the test runtime libraries.
    dependsOn("jvmTestClasses")
    val jvmTest = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath = files(jvmTest.runtimeDependencyFiles) +
        files(jvmTest.output.allOutputs) +
        files(jvmTest.compileDependencyFiles)
    mainClass.set("io.tlaloc.benchmarks.HeadToHeadHarnessMainKt")
    // Pass through harness configuration env vars from the Gradle invoker.
    environment(
        "TLALOC_HARNESS_OUTPUT_DIR",
        System.getenv("TLALOC_HARNESS_OUTPUT_DIR")
            ?: layout.buildDirectory.get().asFile.absolutePath,
    )
    System.getenv("TLALOC_HARNESS_WARMUP")?.let { environment("TLALOC_HARNESS_WARMUP", it) }
    System.getenv("TLALOC_HARNESS_MEASURED")?.let { environment("TLALOC_HARNESS_MEASURED", it) }
}
