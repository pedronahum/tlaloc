plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-autograd:0.1.0-alpha02")
    // CausalLM, HfCausalLm, AdamW, capture, crossEntropy.
    implementation("io.github.pedronahum:tlaloc-nn:0.1.0-alpha02")
    // PjrtSession: XLA compiles the captured graphs and runs them on the GPU.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha02")
}

application {
    mainClass.set("MainKt")
    // FFM downcalls into the PJRT plugin need native access. The heap holds the
    // f32 weights, AdamW's two moments and a gradient, twice over while a step builds the new ones: about 7 x 2.4 GB.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx32g")
}
