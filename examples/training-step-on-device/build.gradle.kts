plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha03")
    implementation("io.github.pedronahum:tlaloc-nn:0.1.0-alpha03")
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha03")
    // PjrtSession and StepBenchmark. Without a GPU the device lanes skip by name.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha03")
}

application {
    mainClass.set("MainKt")
    // The PJRT plugin is loaded through java.lang.foreign.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}
