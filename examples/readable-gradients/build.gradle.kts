import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

// Where the compiler writes the gradients it derived, as Kotlin source.
val gradientSourceDir: Directory = layout.buildDirectory.dir("gradients").get()

sourceSets {
    // A SECOND source set, compiled from the generated gradient. Its whole point
    // is what it does NOT get: no `printedCompilerPluginClasspath` line below, so
    // the Tlaloc plugin is absent from this compilation. If the printed gradient
    // compiles here, it is ordinary Kotlin over `:core` — not plugin magic.
    create("printed")
}

kotlin.sourceSets.named("printed") {
    kotlin.srcDir(gradientSourceDir)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha01")
    implementation("io.github.pedronahum:tlaloc-autograd:0.1.0-alpha01")
    // The K2 compiler plugin — for the MAIN source set only.
    kotlinCompilerPluginClasspath("io.github.pedronahum:tlaloc-compiler-plugin:0.1.0-alpha01")

    // The printed gradient needs nothing but the host tensor library.
    "printedImplementation"("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
}

// §0.4.450's plugin CLI option. `dumpGradSourceDir` implies `dumpGradSource=true`:
// every reverse-gradient lambda the plugin synthesises is ALSO rendered as Kotlin
// and written to this directory, named after the lambda's source location.
tasks.named<KotlinCompile>("compileKotlin") {
    compilerOptions.freeCompilerArgs.addAll(
        "-P",
        "plugin:io.tlaloc.plugin:dumpGradSourceDir=${gradientSourceDir.asFile.absolutePath}",
    )
    // Declaring the dump dir as an output keeps it in sync with the compilation:
    // delete it and Gradle re-runs the compile that regenerates it.
    outputs.dir(gradientSourceDir)
}

// The dump is a by-product of compiling main, so main must compile first.
tasks.named<KotlinCompile>("compilePrintedKotlin") {
    dependsOn(tasks.named("compileKotlin"))
}

application {
    mainClass.set("MainKt")
}

tasks.named<JavaExec>("run") {
    dependsOn(tasks.named("printedClasses"))
    // Main reflects into the separately-compiled printed gradient and calls it.
    classpath += sourceSets["printed"].output
    systemProperty("tlaloc.example.gradientSourceDir", gradientSourceDir.asFile.absolutePath)
    systemProperty(
        "tlaloc.example.printedClassesDir",
        layout.buildDirectory.dir("classes/kotlin/printed").get().asFile.absolutePath,
    )
}
