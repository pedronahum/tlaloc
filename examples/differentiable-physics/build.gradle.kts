import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

// Where the compiler writes the gradient it derived, as Kotlin source.
val gradientSourceDir: Directory = layout.buildDirectory.dir("gradients").get()

dependencies {
    implementation("io.tlaloc:core:0.1.0-alpha01")
    implementation("io.tlaloc:ir:0.1.0-alpha01")
    implementation("io.tlaloc:autograd:0.1.0-alpha01")
    // The K2 compiler plugin: it is what turns the `grad2 { }` block — loop and
    // all — into synthesized gradient code at compile time.
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.1.0-alpha01")
}

// `dumpGradSourceDir` implies `dumpGradSource=true`: every reverse-gradient
// lambda the plugin synthesises is ALSO rendered as Kotlin and written here,
// named after the lambda's source location. Act [4] reads it back and prints it.
tasks.named<KotlinCompile>("compileKotlin") {
    compilerOptions.freeCompilerArgs.addAll(
        "-P",
        "plugin:io.tlaloc.plugin:dumpGradSourceDir=${gradientSourceDir.asFile.absolutePath}",
    )
    outputs.dir(gradientSourceDir)
}

application {
    mainClass.set("MainKt")
}

tasks.named<JavaExec>("run") {
    systemProperty("tlaloc.example.gradientSourceDir", gradientSourceDir.asFile.absolutePath)
}
