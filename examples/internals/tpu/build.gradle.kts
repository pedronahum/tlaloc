plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-core:0.1.0-alpha01")
    // The IR: this example builds its graphs by hand, so the shape of the
    // program is visible in the source instead of hidden behind a model API.
    implementation("io.github.pedronahum:tlaloc-ir:0.1.0-alpha01")
    // StableHLO emission — the text handed to the TPU compiler is written to
    // build/ so you can read exactly what crossed the boundary.
    implementation("io.github.pedronahum:tlaloc-stablehlo:0.1.0-alpha01")
    // PjrtSession + PjrtBinaries + PjrtTarget.Tpu: plugin resolution, client
    // creation, compile, buffer staging, execute. Pure JVM at resolution time;
    // this dependency costs nothing on a machine with no accelerator, which is
    // why the host half of this example runs on a laptop.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha01")

    // NOTE: no `kotlinCompilerPluginClasspath`. The compiler plugin lowers
    // `grad { }` lambdas (see examples/quickstart); this example differentiates
    // an IR function directly with DxirReverseTransform — the same pass the
    // plugin runs, reached through its other front door.
}

application {
    mainClass.set("MainKt")
    // The PJRT plugin is loaded through java.lang.foreign (FFM). JDK 22+ gates
    // native downcalls behind this flag; without it the device half throws on
    // the first PjrtSession. Harmless on a machine with no TPU.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

// Scale the workload up on real hardware:
//   ./gradlew -p examples/tpu run --args="--seq 2048 --dim 512"
// The host cross-check is a plain JVM interpreter, so it is the part that gets
// slow — that is the honest cost of checking a device against a reference.
