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
