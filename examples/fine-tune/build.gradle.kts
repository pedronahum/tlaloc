plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha03")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha03")
    implementation("io.github.pedronahum:tlaloc-autograd:0.1.0-alpha03")
    // CausalLM, HfCausalLm, AdamW, withDeviceAdamW, capture, crossEntropy.
    implementation("io.github.pedronahum:tlaloc-nn:0.1.0-alpha03")
    // toStablehlo: the training step as StableHLO, its parameters and moments donated.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha03")
    // HfTokenizer: the checkpoint's tokenizer.json, with the ids transformers gives.
    implementation("io.github.pedronahum:tlaloc-tokenizer:0.1.0-alpha03")
    // PjrtSession: XLA compiles the captured graphs and runs them on the GPU.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha03")
}

application {
    mainClass.set("MainKt")
    // FFM downcalls into the PJRT plugin need native access. The heap holds the
    // f32 weights (2.4 GB) and their copies on the way to and from the device;
    // AdamW's moments live on the device.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx32g")
}
