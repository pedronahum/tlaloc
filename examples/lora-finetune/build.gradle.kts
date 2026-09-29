plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    // LoRA, HfLoraAdapter and the frozen-parameter capture are in this checkout,
    // not in 0.1.0-alpha02: publish it to mavenLocal first.
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-autograd:0.1.0-alpha02")
    // CausalLM, HfCausalLm, Lora, HfLoraAdapter, AdamW, capture, crossEntropy.
    implementation("io.github.pedronahum:tlaloc-nn:0.1.0-alpha02")
    // HfTokenizer: the checkpoint's tokenizer.json, with the ids transformers gives.
    implementation("io.github.pedronahum:tlaloc-tokenizer:0.1.0-alpha02")
    // PjrtSession: XLA compiles the captured graphs and runs them on the GPU.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha02")
}

application {
    mainClass.set("MainKt")
    // FFM downcalls into the PJRT plugin need native access. The heap holds the
    // f32 base weights (2.4 GB), the merged copy written at the end, and the
    // one-hot targets.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx24g")
}
