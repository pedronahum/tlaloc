import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    // §0.4.503 — the supported consumer configuration, spelled out in the one project
    // that exists to be copied. BUILD on JDK 25, EMIT Java 21 bytecode:
    //
    //   * the toolchain must be 25, because Kotlin runs a compiler plugin inside the
    //     compiler's own JVM and `io.tlaloc:compiler-plugin` is 25 bytecode. This is
    //     the honest boundary of the §0.4.503 split: your BUILD machine needs a JDK 25
    //     to use `grad { }` at all.
    //   * the TARGET can be 21, because every library module Tlaloc publishes except
    //     the runtime backends is 21 bytecode now. So what you ship runs on a JDK 21.
    //
    // `runOnJdk21` below is the proof, not the promise: it runs THIS program — a
    // compile-time-synthesized gradient and all — on a real JDK 21.
    jvmToolchain(25)

    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

java {
    // Kotlin's jvmTarget alone leaves the Java compile and the `java` component at the
    // toolchain's 25; without these two the jar would be a 21/25 mixture and
    // `runOnJdk21` would fail on a class nobody wrote.
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

sourceSets {
    // A source set that is SUPPOSED to fail. It is not part of `build`, `check`
    // or `run` — nothing depends on it — so the quickstart still runs green.
    // `./gradlew -p examples/quickstart shapeError` compiles it on purpose, to
    // show you the compiler rejecting a shape bug.
    create("shapeError")
}

dependencies {
    implementation("io.tlaloc:core:0.1.0-alpha01")
    implementation("io.tlaloc:ir:0.1.0-alpha01")
    implementation("io.tlaloc:autograd:0.1.0-alpha01")
    // The K2 compiler plugin: rewrites `grad { }` calls into synthesized
    // gradient code at compile time (and gives you compile-time shape /
    // differentiability errors in the IDE).
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.1.0-alpha01")

    // The failing source set needs the same libraries and the same plugin —
    // the point is that it fails on its MERITS, not for want of a dependency.
    "shapeErrorImplementation"("io.tlaloc:core:0.1.0-alpha01")
    "shapeErrorImplementation"("io.tlaloc:ir:0.1.0-alpha01")
    "shapeErrorImplementation"("io.tlaloc:autograd:0.1.0-alpha01")
    "kotlinCompilerPluginClasspathShapeError"("io.tlaloc:compiler-plugin:0.1.0-alpha01")
}

application {
    mainClass.set("MainKt")
}

// Main.kt prints the offending source, so it needs to find it.
tasks.named<JavaExec>("run") {
    systemProperty(
        "tlaloc.example.shapeErrorSource",
        layout.projectDirectory.file("src/shapeError/kotlin/ShapeError.kt").asFile.absolutePath,
    )
}

/**
 * §0.4.503 (Tier 3, item 1) — **run the quickstart on a real JDK 21.**
 *
 * This is the certification behind the README's per-module JDK table. The program is
 * compiled by a JDK 25 toolchain (the compiler plugin needs one) into Java 21
 * bytecode, and then executed by a JDK 21 launcher against the published
 * `io.tlaloc:core`, `:ir` and `:autograd` jars. If any of those jars — or the
 * gradient the plugin synthesized into this one — carried a JDK 22+ API or 25
 * bytecode, this task fails with `UnsupportedClassVersionError` or
 * `NoSuchMethodError` instead of printing a gradient.
 *
 *     export JDK21_HOME=/path/to/jdk-21     # gradle.properties reads this
 *     ./gradlew -p examples/quickstart runOnJdk21
 *
 * `scripts/jdk21-smoke.sh` is the same thing from the repository root, including the
 * `publishToMavenLocal` that has to precede it. Gradle cannot auto-detect a JDK in
 * `~/.local/jdks`, so the env var is how the location stays off this file.
 */
val runOnJdk21 by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Run the quickstart on a JDK 21 launcher (needs \$JDK21_HOME)"
    val main = kotlin.target.compilations.getByName("main")
    dependsOn(main.compileTaskProvider)
    classpath(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("MainKt")
    javaLauncher.set(
        javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(21)) },
    )
    systemProperty(
        "tlaloc.example.shapeErrorSource",
        layout.projectDirectory.file("src/shapeError/kotlin/ShapeError.kt").asFile.absolutePath,
    )
}

/**
 * Compile the deliberately-broken source set. THIS TASK IS MEANT TO FAIL:
 * a green build here would mean Tlaloc had stopped catching the bug.
 */
tasks.register("shapeError") {
    group = "verification"
    description = "Compile a program with mismatched named axes. Expected to FAIL."
    dependsOn("compileShapeErrorKotlin")
    doLast {
        throw GradleException(
            "compileShapeErrorKotlin SUCCEEDED — it was supposed to fail. " +
                "Tlaloc did not reject the mismatched named axes in src/shapeError/kotlin/ShapeError.kt.",
        )
    }
}
