plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

sourceSets {
    // A source set that is SUPPOSED to fail. Nothing depends on it, so `run`
    // and `build` stay green; the `shapeError` task below compiles it on
    // purpose to show the two world-boundary violations being rejected.
    create("shapeError")
}

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    // `program { }` / `workflow { }` and the Maestro descriptor live here.
    implementation("io.tlaloc:maestro:0.0.1-SNAPSHOT")
    // No compiler plugin: `program { }` captures its body with the :autograd
    // runtime tracer, not a plugin lowering. The plugin's role in the four
    // worlds is enforcing scope discipline at compile time, which this example
    // demonstrates with a source set that must not compile.

    "shapeErrorImplementation"("io.tlaloc:core:0.0.1-SNAPSHOT")
    "shapeErrorImplementation"("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    "shapeErrorImplementation"("io.tlaloc:maestro:0.0.1-SNAPSHOT")
}

application {
    mainClass.set("MainKt")
}

tasks.named<JavaExec>("run") {
    systemProperty(
        "tlaloc.example.shapeErrorSource",
        layout.projectDirectory.file("src/shapeError/kotlin/WorldErrors.kt").asFile.absolutePath,
    )
}

/**
 * Compile the deliberately-broken source set. THIS TASK IS MEANT TO FAIL:
 * a green build here would mean the world boundaries had stopped holding.
 */
tasks.register("shapeError") {
    group = "verification"
    description = "Compile two world-boundary violations. Expected to FAIL."
    dependsOn("compileShapeErrorKotlin")
    doLast {
        throw GradleException(
            "compileShapeErrorKotlin SUCCEEDED — it was supposed to fail. " +
                "The four-worlds boundaries in src/shapeError/kotlin/WorldErrors.kt were not enforced.",
        )
    }
}
