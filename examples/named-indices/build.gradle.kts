plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

sourceSets {
    // A source set that is SUPPOSED to fail. Nothing depends on it, so `run`
    // and `build` stay green; `./gradlew -p examples/named-indices shapeError`
    // compiles it on purpose to show the type checker refusing a shape bug.
    create("shapeError")
}

dependencies {
    // Named axes live entirely in :core's type system. No compiler plugin is
    // needed for THIS example: the contraction rule is expressed in Kotlin's
    // own generics, so Kotlin's native type checker is the one rejecting bad
    // programs. (The plugin's job is lowering `grad { }` bodies — see
    // examples/quickstart.)
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")

    // Deliberately the SAME dependency and, deliberately, still no plugin: the
    // failing file fails on Kotlin's own type rules.
    "shapeErrorImplementation"("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
}

application {
    mainClass.set("MainKt")
}

tasks.named<JavaExec>("run") {
    systemProperty(
        "tlaloc.example.shapeErrorSource",
        layout.projectDirectory.file("src/shapeError/kotlin/ShapeError.kt").asFile.absolutePath,
    )
}

/**
 * Compile the deliberately-broken source set. THIS TASK IS MEANT TO FAIL:
 * a green build here would mean the named-axis rule had stopped holding.
 */
tasks.register("shapeError") {
    group = "verification"
    description = "Compile a contraction over axes that share no name. Expected to FAIL."
    dependsOn("compileShapeErrorKotlin")
    doLast {
        throw GradleException(
            "compileShapeErrorKotlin SUCCEEDED — it was supposed to fail. " +
                "Kotlin did not reject the mismatched named axes in src/shapeError/kotlin/ShapeError.kt.",
        )
    }
}
