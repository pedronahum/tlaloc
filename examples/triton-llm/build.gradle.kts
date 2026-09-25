plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    // HfCheckpoint, DecodeBucketPolicy: read a HuggingFace checkpoint and
    // choose the compiled shapes.
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha02")
    // HfServingExport writes the serving artifact; TritonModelRepository
    // turns it into a Triton model directory.
    implementation("io.github.pedronahum:tlaloc-maestro:0.1.0-alpha02")
    // StableHLO emission: the exporter writes every entry through it.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha02")

    // Not here: io.github.pedronahum:tlaloc-runtime-pjrt. The export never
    // runs a graph. Triton runs it, through libtriton_tlaloc.so.
}

application {
    mainClass.set("ExportKt")
    // The export transposes each Linear on the host and holds a tensor and
    // its transpose at once. Muse Glimmer's weights are streamed in blocks,
    // so 8 GiB covers all three models.
    applicationDefaultJvmArgs = listOf("-Xmx8g")
}
