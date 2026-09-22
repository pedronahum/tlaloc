plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:ir:0.0.1-SNAPSHOT")
    // The Tracer: `capture` traces the model's forward through it.
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    // Dense / ReluLayer / Sequential / Adam / capture — the model layer.
    implementation("io.tlaloc:nn:0.0.1-SNAPSHOT")
    // StableHLO emission of the captured gradient graph. `:runtime-pjrt` pulls
    // it in transitively, but the example names it because the example's story
    // IS the emission.
    implementation("io.tlaloc:stablehlo:0.0.1-SNAPSHOT")
    // PjrtSession + PjrtBinaries: compile the emitted StableHLO with XLA and
    // execute it on the GPU. Pure JVM at resolution time — this dependency
    // costs nothing on a machine with no CUDA, and the example self-skips the
    // GPU lane there.
    implementation("io.tlaloc:runtime-pjrt:0.0.1-SNAPSHOT")

    // NOTE: no `kotlinCompilerPluginClasspath` here. The compiler plugin lowers
    // `grad { }` LAMBDAS (see examples/quickstart); a MODEL's gradient comes
    // from `:nn`'s `capture`, which runs the same compiler AD pass
    // (DxirReverseTransform) over a traced graph instead of over a lambda body.
    // Same engine, different front door — this example is the model front door.
}

application {
    mainClass.set("MainKt")
    // The PJRT plugin is loaded through java.lang.foreign (FFM). JDK 22+ gates
    // native downcalls behind this flag; without it the GPU lane throws on the
    // first PjrtSession. Harmless on a machine with no GPU.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}
