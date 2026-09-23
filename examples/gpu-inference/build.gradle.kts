plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha01")
    // The exporter and the manifest: ServingArtifactWriter, ServingManifest,
    // ReferenceDecodeGraph, HfLlamaServingExport. This example's Kotlin half is
    // ~100 lines around these, and the whole point of the example is that the
    // Kotlin half STOPS once the directory exists.
    implementation("io.github.pedronahum:tlaloc-maestro:0.1.0-alpha01")
    // StableHLO emission — the exporter writes the bodies through it.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha01")

    // NOTE what is NOT here: `io.github.pedronahum:tlaloc-runtime-pjrt`. The exporting process
    // never executes anything, never loads a PJRT plugin and never touches the
    // GPU. Nothing in this build needs a driver, which is why half one of this
    // example runs on any laptop.
}

application {
    mainClass.set("ExportKt")
    // The exporter holds a real checkpoint's embedding table and its transpose
    // at once when it stages a Llama (262 MiB as f32 on TinyLlama). Irrelevant
    // for the reference model; free to ask for.
    applicationDefaultJvmArgs = listOf("-Xmx8g")
}

// Half one is `./gradlew -p examples/gpu-inference run`, which writes the
// reference decode graph's artifact to build/artifact and needs no checkpoint
// and no GPU. Point it at a real HuggingFace Llama instead with:
//
//   ./gradlew -p examples/gpu-inference run \
//     --args="--checkpoint $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0"
