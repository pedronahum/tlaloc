plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
    // DXIR, the pattern recognizers, the VJP coarsener, the kernel registry
    // and the KV-quant pass.
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha01")
    // The StableHLO emitter — this is what makes the per-target divergence
    // visible as actual MLIR text.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha01")
    // The backend-matrix populator (the manifest fragment a cluster reads).
    implementation("io.github.pedronahum:tlaloc-maestro:0.1.0-alpha01")
}

application {
    mainClass.set("MainKt")
}
