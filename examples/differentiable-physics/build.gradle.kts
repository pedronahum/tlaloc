import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "2.3.20"
    // Puts the Tlaloc K2 compiler plugin on the compilation: it is what turns the
    // `grad2 { }` block, loop and all, into synthesized gradient code at compile time.
    id("io.github.pedronahum.tlaloc") version "0.1.0-alpha01"
    application
}

kotlin {
    jvmToolchain(25)
}

// Where the compiler writes the gradient it derived, as Kotlin source.
val gradientSourceDir: Directory = layout.buildDirectory.dir("gradients").get()

dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:0.1.0-alpha01"))
    implementation("io.github.pedronahum:tlaloc-core")
    implementation("io.github.pedronahum:tlaloc-ir")
    implementation("io.github.pedronahum:tlaloc-autograd")
}

// `dumpGradSourceDir` implies `dumpGradSource`: every reverse-gradient lambda the
// plugin synthesises is also rendered as Kotlin and written here, named after the
// lambda's source location. Act [4] reads it back and prints it.
tlaloc {
    dumpGradSourceDir.set(gradientSourceDir)
}

// Declaring the dump dir as an output keeps it in sync with the compilation.
tasks.named<KotlinCompile>("compileKotlin") {
    outputs.dir(gradientSourceDir)
}

application {
    mainClass.set("MainKt")
}

tasks.named<JavaExec>("run") {
    systemProperty("tlaloc.example.gradientSourceDir", gradientSourceDir.asFile.absolutePath)
}
