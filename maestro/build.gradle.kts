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
                implementation(project(":autograd"))
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
 * item 3: `ReferenceDecodeGraphKt.main` has existed since H3a and is the
 * documented entry point, but nothing assembled its classpath, so the only
 * *certified* way to obtain an artifact was to be inside
 * `ServingArtifactExportRunTest`. That is fine for a test and useless for a
 * runbook: the live-vLLM certification needs a directory a shell can point
 * `$TLALOC_SERVING_ARTIFACT` at.
 *
 *     ./gradlew :maestro:exportServingArtifact -PoutDir=/abs/path
 *
 * The classpath is the jvmMain compilation's runtime classpath — NOT
 * jvmTest's. The distinction is the point: if the exporter needed a test
 * fixture to run, the artifact would be a test fixture, and H3a's whole
 * claim (the directory is the deployment) would be a claim about the test
 * source set.
 */
val exportServingArtifact by tasks.registering(JavaExec::class) {
    group = "tlaloc"
    description = "Write the reference serving artifact (manifest + StableHLO bodies) to -PoutDir"
    val jvmMain = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMain.compileTaskProvider)
    classpath(jvmMain.output.allOutputs, jvmMain.runtimeDependencyFiles)
    mainClass.set("io.tlaloc.maestro.serving.ReferenceDecodeGraphKt")
    javaLauncher.set(
        javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) },
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
val exportLlamaServingArtifact by tasks.registering(JavaExec::class) {
    group = "tlaloc"
    description = "Write a serving artifact for a real HF Llama checkpoint (-PckptDir) to -PoutDir"
    val jvmMain = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMain.compileTaskProvider)
    classpath(jvmMain.output.allOutputs, jvmMain.runtimeDependencyFiles)
    mainClass.set("io.tlaloc.maestro.serving.HfLlamaServingExportKt")
    maxHeapSize = "8g"
    javaLauncher.set(
        javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) },
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
