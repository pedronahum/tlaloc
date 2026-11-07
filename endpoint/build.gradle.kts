import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    // Built on JDK 25, emitting Java 21 bytecode, like :tokenizer.
    jvmToolchain(25)

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
                // The endpoint's API takes HfTokenizer; parseJson reads requests.
                api(project(":tokenizer"))
                implementation(project(":core"))
            }
        }
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotest.assertions.core)
                implementation(libs.kotest.runner.junit5)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    maxHeapSize = "2g"
}

/**
 * The OpenAI-compatible endpoint in front of a decoupled speculative Tlaloc model in Triton:
 *
 *     ./gradlew :endpoint:run --args="--model ripple --tokenizer <checkpoint dir> --triton localhost:8001 --port 8100"
 */
tasks.register<JavaExec>("run") {
    group = "tlaloc"
    description = "Serve a Triton-hosted Tlaloc model over an OpenAI-compatible HTTP endpoint"
    val main = kotlin.jvm().compilations.getByName("main")
    dependsOn(main.compileTaskProvider)
    classpath(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("io.tlaloc.endpoint.MainKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    maxHeapSize = providers.gradleProperty("endpointHeap").getOrElse("4g")
    standardInput = System.`in`
}

/**
 * `installEndpoint`: the endpoint's jars and a launcher, `build/endpoint/bin/tlaloc-endpoint`, to run
 * it without Gradle (a serving process outlives builds):
 *
 *     ./gradlew :endpoint:installEndpoint && endpoint/build/endpoint/bin/tlaloc-endpoint --model ripple --tokenizer <dir>
 */
val installEndpoint = tasks.register<Sync>("installEndpoint") {
    group = "tlaloc"
    description = "Install the OpenAI-compatible endpoint with a launcher script under build/endpoint"
    val main = kotlin.jvm().compilations.getByName("main")
    val jar = tasks.named<Jar>("jvmJar")
    into(layout.buildDirectory.dir("endpoint"))
    from(jar) { into("lib") }
    from(main.runtimeDependencyFiles) { into("lib") }
    val launcher = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) }
    val java = launcher.map { it.executablePath.asFile.absolutePath }
    inputs.property("java", java)
    doLast {
        val bin = destinationDir.resolve("bin").apply { mkdirs() }
        val script = bin.resolve("tlaloc-endpoint")
        script.writeText(
            "#!/bin/sh\n" +
                "# The Tlaloc OpenAI-compatible endpoint (written by :endpoint:installEndpoint).\n" +
                "here=\$(cd \"\$(dirname \"\$0\")/..\" && pwd)\n" +
                "exec \"\${JAVA:-${java.get()}}\" \${ENDPOINT_JAVA_OPTS:--Xmx4g} -cp \"\$here/lib/*\" io.tlaloc.endpoint.MainKt \"\$@\"\n",
        )
        script.setExecutable(true)
    }
}
