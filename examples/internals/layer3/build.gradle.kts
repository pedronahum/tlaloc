plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    // DXIR, the pattern recognizers, the VJP coarsener, the kernel registry
    // and the KV-quant pass.
    implementation("io.tlaloc:ir:0.0.1-SNAPSHOT")
    // The StableHLO emitter — this is what makes the per-target divergence
    // visible as actual MLIR text.
    implementation("io.tlaloc:stablehlo:0.0.1-SNAPSHOT")
    // The backend-matrix populator (the manifest fragment a cluster reads).
    implementation("io.tlaloc:maestro:0.0.1-SNAPSHOT")
}

application {
    mainClass.set("MainKt")
}
