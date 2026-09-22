plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
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
