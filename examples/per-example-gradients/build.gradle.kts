plugins {
    kotlin("jvm") version "2.4.20"
    // Puts the Tlaloc K2 compiler plugin on the compilation: it turns `vmap { grad { } }`
    // into batched gradient code at compile time.
    id("io.github.pedronahum.tlaloc") version "0.1.0-alpha02"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:0.1.0-alpha02"))
    implementation("io.github.pedronahum:tlaloc-core")
    implementation("io.github.pedronahum:tlaloc-autograd")
}

application {
    mainClass.set("MainKt")
}
