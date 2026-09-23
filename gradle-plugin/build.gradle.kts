import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-gradle-plugin`
}

// The Gradle plugin runs inside the consumer's Gradle daemon, which may be on a
// JDK 21, so it is Java 21 bytecode (tlalocJvmTargets in the root build). It is
// compiled by the repository's JDK 25 toolchain with -Xjdk-release=21, the same
// arrangement as the library modules.
kotlin {
    jvmToolchain(25)
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
        freeCompilerArgs.add("-Xjdk-release=21")
        // A consumer's build script is compiled by its Gradle's embedded Kotlin,
        // which on Gradle 8 is older than the catalog's. Kotlin reads metadata one
        // language version ahead, so 2.1 metadata stays readable by an embedded 2.0.
        languageVersion.set(KotlinVersion.KOTLIN_2_1)
        apiVersion.set(KotlinVersion.KOTLIN_2_1)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
    withSourcesJar()
}

dependencies {
    // Provided by the consumer's build: the plugin requires the Kotlin JVM or
    // Multiplatform Gradle plugin to be applied, and Gradle carries the stdlib.
    compileOnly("org.jetbrains.kotlin:kotlin-gradle-plugin-api:${libs.versions.kotlin.get()}")
    compileOnly(kotlin("stdlib"))

    // ProjectBuilder tests apply the real Kotlin Gradle plugin and read what it wired.
    testImplementation("org.jetbrains.kotlin:kotlin-gradle-plugin:${libs.versions.kotlin.get()}")
    testImplementation(kotlin("test"))
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}

gradlePlugin {
    plugins {
        create("tlaloc") {
            id = "io.github.pedronahum.tlaloc"
            implementationClass = "io.tlaloc.gradle.TlalocGradlePlugin"
            displayName = "Tlaloc"
            description = "Applies the Tlaloc K2 compiler plugin (grad { } lowering) to every " +
                "Kotlin/JVM compilation, at the version of this Gradle plugin."
        }
    }
}

// The compiler plugin's coordinates, baked in at build time so the Gradle plugin
// always asks for the tlaloc-compiler-plugin built alongside it.
val generatedVersionDir = layout.buildDirectory.dir("generated/tlaloc-version/kotlin")
// The compiler plugin id is read from the compiler plugin's own source, so the two
// cannot drift apart.
val compilerPluginIdSource =
    rootProject.file("compiler-plugin/src/main/kotlin/io/tlaloc/plugin/TlalocCommandLineProcessor.kt")
val generateTlalocVersion = tasks.register("generateTlalocVersion") {
    val outDir = generatedVersionDir
    val group = project.group.toString()
    val version = project.version.toString()
    val idSource = compilerPluginIdSource
    inputs.property("group", group)
    inputs.property("version", version)
    inputs.file(idSource).withPropertyName("compilerPluginIdSource")
    outputs.dir(outDir)
    doLast {
        val ids = Regex("""const val PLUGIN_ID: String = "([^"]+)"""")
            .findAll(idSource.readText()).map { it.groupValues[1] }.toList()
        if (ids.size != 1) {
            throw GradleException(
                "generateTlalocVersion: expected one `const val PLUGIN_ID: String = \"…\"` in " +
                    "${idSource.path}, found ${ids.size}. The Gradle plugin reads the compiler " +
                    "plugin id from there.",
            )
        }
        val compilerPluginId = ids.single()
        val file = outDir.get().file("io/tlaloc/gradle/TlalocBuildInfo.kt").asFile
        file.parentFile.mkdirs()
        file.writeText(
            """
            |package io.tlaloc.gradle
            |
            |/** Generated at build time by :gradle-plugin:generateTlalocVersion. */
            |internal object TlalocBuildInfo {
            |    const val GROUP: String = "$group"
            |    const val VERSION: String = "$version"
            |    const val COMPILER_PLUGIN_ARTIFACT: String = "tlaloc-compiler-plugin"
            |    const val COMPILER_PLUGIN_ID: String = "$compilerPluginId"
            |}
            |
            """.trimMargin(),
        )
    }
}

kotlin.sourceSets.named("main") {
    kotlin.srcDir(generateTlalocVersion)
}

tasks.named("sourcesJar") { dependsOn(generateTlalocVersion) }
