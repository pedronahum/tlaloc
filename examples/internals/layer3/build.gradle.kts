plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha06")
    // DXIR, the pattern recognizers, the VJP coarsener, the kernel registry
    // and the KV-quant pass.
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha06")
    // The StableHLO emitter — this is what makes the per-target divergence
    // visible as actual MLIR text.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha06")
    // The backend-matrix populator (the manifest fragment a cluster reads).
    implementation("io.github.pedronahum:tlaloc-maestro:0.1.0-alpha06")
}

application {
    mainClass.set("MainKt")
}
