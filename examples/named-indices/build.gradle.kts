plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    // Named axes live entirely in :core's type system. No compiler plugin is
    // needed for THIS example: the contraction rule is expressed in Kotlin's
    // own generics, so Kotlin's native type checker is the one rejecting bad
    // programs. (The plugin's job is lowering `grad { }` bodies — see
    // examples/quickstart.)
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
}

application {
    mainClass.set("MainKt")
}
