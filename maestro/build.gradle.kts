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
        // The exporters' `main` entry points. A separate compilation so the
        // published jar carries the export API and no command-line programs;
        // it sees the main compilation and its runtime classpath, not jvmTest's.
        // Nothing publishes it.
        compilations.create("tools") {
            associateWith(this@jvm.compilations.getByName("main"))
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
                // api, not implementation: :core, :ir and :autograd types appear in
                // this module's public signatures (program { } takes a body over
                // autograd's Tracer), so a consumer of it alone can name them.
                api(project(":core"))
                api(project(":ir"))
                api(project(":autograd"))
                implementation(project(":stablehlo"))
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
    // §0.4.480 (H3c-3). Gradle forks a Test JVM at a 512 MB default heap, and
    // HfLlamaServingArtifactTest EXPORTS a real TinyLlama-1.1B: each staged
    // weight is read as bf16, widened to f32, transposed (so it exists twice
    // for the duration) and serialised to a ByteArray before it is written.
    // The embedding table alone is 32000x2048 = 262 MiB of f32 plus its byte
    // buffer. 2 GB is the peak-per-tensor budget with headroom; the export is
    // deliberately ONE TENSOR AT A TIME (`stageWeight` is a callback, not a
    // `List<FloatArray>`) so 4.2 GiB of weights never has to be resident —
    // which is why this number is 2 GB and not :ir's 8 GB.
    maxHeapSize = "2g"
}

/**
 * §0.4.477 (H7) — `exportServingArtifact`: the sanctioned way to write a
 * serving artifact directory outside a test.
 *
 * INFERENCE_SERVING_AUDIT.md §5 carried this as WRITTEN-BUT-UNCERTIFIED
 * item 3: the reference exporter's `main` has existed since H3a and is the
 * documented entry point, but nothing assembled its classpath, so the only
 * *certified* way to obtain an artifact was to be inside
 * `ServingArtifactExportRunTest`. That is fine for a test and useless for a
 * runbook: the live-vLLM certification needs a directory a shell can point
 * `$TLALOC_SERVING_ARTIFACT` at.
 *
 *     ./gradlew :maestro:exportServingArtifact -PoutDir=/abs/path
 *
 * The classpath is the `tools` compilation's: jvmMain's output and runtime
 * classpath plus the `main` entry point, which is kept out of the published jar.
 * NOT jvmTest's. The distinction is the point: if the exporter needed a test
 * fixture to run, the artifact would be a test fixture, and H3a's whole
 * claim (the directory is the deployment) would be a claim about the test
 * source set.
 */
tasks.register<JavaExec>("exportServingArtifact") {
    group = "tlaloc"
    description = "Write the reference serving artifact (manifest + StableHLO bodies) to -PoutDir"
    val tools = kotlin.jvm().compilations.getByName("tools")
    dependsOn(tools.compileTaskProvider)
    classpath(tools.output.allOutputs, tools.runtimeDependencyFiles)
    mainClass.set("io.tlaloc.maestro.serving.ExportReferenceServingArtifactKt")
    // §0.4.503 — the launcher is a KNOB now, defaulting to the JDK this repository
    // builds with. `-PexportJdk=21` is the certification lane for `:maestro`'s new
    // 21 bytecode target: the export then runs on a JDK 21 against the jvmMain
    // runtime classpath (`:core`, `:ir`, `:autograd`, `:stablehlo` — all 21), so a
    // JDK 22+ dependency anywhere in that graph surfaces as an
    // UnsupportedClassVersionError instead of as a claim in a README. It is a knob
    // and not simply 21 because this task is a documented runbook command
    // (docs/TPU_RUNBOOK.md, INFERENCE_SERVING_AUDIT.md) and making it need a second
    // JDK that Gradle cannot auto-detect would be a regression for its actual users.
    // Requires `JDK21_HOME` (see gradle.properties' installations.fromEnv line).
    javaLauncher.set(
        javaToolchains.launcherFor {
            languageVersion.set(
                JavaLanguageVersion.of(
                    (project.findProperty("exportJdk") as String?)?.toIntOrNull() ?: 25,
                ),
            )
        },
    )
    argumentProviders.add(
        CommandLineArgumentProvider {
            listOf(
                (project.findProperty("outDir") as String?)
                    ?: layout.buildDirectory.dir("serving-artifact").get().asFile.absolutePath,
            )
        },
    )
}

/**
 * §0.4.480 (H3c-3) — `exportLlamaServingArtifact`: the same act for a REAL
 * HuggingFace Llama checkpoint.
 *
 *     ./gradlew :maestro:exportLlamaServingArtifact \
 *         -PckptDir=$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
 *         -PoutDir=/tmp/tlaloc-llama-artifact [-PnumLayers=2]
 *
 * Same classpath rule as [exportServingArtifact] and for the same reason.
 * The heap is raised because the export TRANSPOSES: §0.4.479's host-side
 * `[out, in] -> [in, out]` pass holds a tensor and its transpose at once,
 * and TinyLlama's embedding table is 262 MiB staged as f32.
 */
tasks.register<JavaExec>("exportLlamaServingArtifact") {
    group = "tlaloc"
    description = "Write a serving artifact for a real HF Llama checkpoint (-PckptDir) to -PoutDir"
    val tools = kotlin.jvm().compilations.getByName("tools")
    dependsOn(tools.compileTaskProvider)
    classpath(tools.output.allOutputs, tools.runtimeDependencyFiles)
    mainClass.set("io.tlaloc.maestro.serving.ExportLlamaServingArtifactKt")
    maxHeapSize = "8g"
    // §0.4.503 — `-PexportJdk=<n>`, as on [exportServingArtifact].
    javaLauncher.set(
        javaToolchains.launcherFor {
            languageVersion.set(
                JavaLanguageVersion.of(
                    (project.findProperty("exportJdk") as String?)?.toIntOrNull() ?: 25,
                ),
            )
        },
    )
    argumentProviders.add(
        CommandLineArgumentProvider {
            fun p(name: String) = (project.findProperty(name) as String?) ?: ""
            listOf(
                p("ckptDir").ifBlank {
                    throw GradleException(
                        "exportLlamaServingArtifact needs -PckptDir=<an HF checkpoint directory>",
                    )
                },
                p("outDir").ifBlank {
                    layout.buildDirectory.dir("llama-serving-artifact").get().asFile.absolutePath
                },
                p("numLayers"), p("maxBatch"), p("maxContext"), p("blockSize"), p("numBlocks"),
            )
        },
    )
}
