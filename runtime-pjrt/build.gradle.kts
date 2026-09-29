import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // §0.4.503 — This module IS the reason §0.4.311 moved to JDK 25: the PJRT C API is bound
    // with the Foreign Function & Memory API, stable since JEP 454 (JDK 22) and
    // final in 25. It stays at 25 while the library modules drop to 21 — see the
    // `tlalocJvmTargets` table in the root build.gradle.kts.
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
                // :ir is api: runOnPjrt and PjrtSession take a DxirFunction.
                implementation(project(":core"))
                api(project(":ir"))
                implementation(project(":stablehlo"))
                // KPTX v1.4 — kernel dispatch inside PJRT executables needs
                // the CUDA driver bindings (cuModuleLoadData / cuLaunchKernel).
                api(project(":runtime-cuda"))
                // §0.4.471 (H4) — the kernel LIBRARY moves to the main side.
                // KptxKernelRegistry deliberately takes PTX *text*, so it never
                // needed :kptx; but claiming means the serving path itself must
                // register a chain for a name the compiler emitted
                // (KptxPagedAttention), and that needs the kernel sources.
                // :kptx is pure-Kotlin PTX construction — no native surface.
                implementation(project(":kptx"))
                // ServingModel reads a serving artifact's manifest with
                // :maestro's parser rather than a second copy of the schema.
                // implementation: no :maestro type is in ServingModel's API.
                implementation(project(":maestro"))
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
                // §0.4.458 (G1d) — the mixed-precision smoke test captures a
                // real :nn model step and runs its gradient function on the
                // compiled lane (the product path, not a hand-built twin).
                implementation(project(":autograd"))
                implementation(project(":nn"))
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
    // PjrtQwen3GreedyParityTest stages Qwen3-0.6B's weights in f32 (2.4 GB).
    maxHeapSize = "12g"
}

// The device test suite as a directory that runs with a JDK and nothing else:
// the compiled test classes and resources, their runtime classpath, the JUnit
// console launcher, the fixtures the tests read by relative path, and
// run-tests.sh. Built here and copied to a machine (a Cloud TPU VM) that then
// needs no Gradle, no dependency download and no compile. The jars are JVM
// bytecode, so an aarch64 build runs on an x86_64 VM.
val junitConsole by configurations.creating
dependencies {
    junitConsole("org.junit.platform:junit-platform-console:1.13.4")
    junitConsole("org.junit.platform:junit-platform-reporting:1.13.4")
}

val tpuBundle by tasks.registering(Sync::class) {
    group = "verification"
    description = "Writes build/device-test-bundle/: the device tests runnable with a JDK alone."
    val test = kotlin.jvm().compilations.getByName("test")
    into(layout.buildDirectory.dir("device-test-bundle"))
    from(test.runtimeDependencyFiles) { into("lib") }
    from(junitConsole) { into("lib") }
    from(tasks.named("jvmJar")) { into("lib") }
    from(test.output.allOutputs) { into("runtime-pjrt/classes") }
    from(rootProject.file("ir/src/jvmTest/resources/io/tlaloc/ir/inference")) {
        include("*.json")
        into("ir/src/jvmTest/resources/io/tlaloc/ir/inference")
    }
    from(rootProject.file("scripts/device-test-bundle/run-tests.sh"))
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}
