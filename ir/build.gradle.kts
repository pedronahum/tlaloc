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
                // api, not implementation: :core's DType and shapes appear in this
                // module's public signatures, so a consumer of it alone can name them.
                api(project(":core"))
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
                // §0.4.503 (Tier 3, item 3) — `compileOnly`, not `implementation`.
                //
                // matheclipse-core is an 8.3 MB LGPL-3.0 jar with its own transitive
                // tree, and until now it was a MANDATORY runtime dependency of every
                // consumer of :ir — and so of :autograd, :nn and :stablehlo — whether
                // or not they ever differentiated a loop-bearing body. Two costs, one
                // of them not ours to impose: the download, and an LGPL artifact
                // arriving in a stranger's dependency graph unannounced.
                //
                // compileOnly keeps :ir compiling (SymjaEngine ships unchanged) and
                // keeps the dependency OUT of the published POM. The runtime seam is
                // io.tlaloc.ir.passes.SymbolicEngines, which probes for the class and
                // refuses by name — naming the one line to add and the licence — when
                // a body genuinely needs the CAS and Symja is absent.
                compileOnly(libs.symja.core)
            }
        }
        jvmTest {
            dependencies {
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
                // §0.4.503 — Symja stays on the TEST classpath, so the whole Stage B
                // coarsening suite (C6–C9, the bake-off, the L-sweep) certifies exactly
                // the same behaviour it certified before this change. Making a dependency
                // optional must not quietly reduce what is tested.
                implementation(libs.symja.core)
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
    // §0.4.503 (Tier 3, item 3) — the Symja coordinate is quoted verbatim in
    // SymbolicEngines.SYMJA_COORDINATE, because a refusal that says "add a dependency"
    // without saying which one is not a refusal by name. This hands the test the
    // catalog's version so the two cannot drift.
    systemProperty("tlaloc.symja.version", libs.versions.symja.get())
}
