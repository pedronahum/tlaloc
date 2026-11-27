plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

// -PtlalocVersion (run.sh: TLALOC_VERSION) builds against a locally published Tlaloc instead.
val tlaloc = (findProperty("tlalocVersion") as String?) ?: "0.1.0-alpha06"

dependencies {
    // JsonValue / parseJson / jsonQuote (io.tlaloc.core.io): the JSON used across Tlaloc.
    implementation("io.github.pedronahum:tlaloc-core:$tlaloc")
    implementation("io.github.pedronahum:tlaloc-ir:$tlaloc")
    implementation("io.github.pedronahum:tlaloc-autograd:$tlaloc")
    // HfCausalLm, Lora, AdamW, capture, crossEntropy, distillationLoss.
    implementation("io.github.pedronahum:tlaloc-nn:$tlaloc")
    implementation("io.github.pedronahum:tlaloc-stablehlo:$tlaloc")
    // HfTokenizer, ChatTemplate.TWIL / RIPPLE and their output parsers.
    implementation("io.github.pedronahum:tlaloc-tokenizer:$tlaloc")
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:$tlaloc")
    // OpenAI-compatible endpoint with tools, in front of Triton.
    implementation("io.github.pedronahum:tlaloc-endpoint:$tlaloc")
    // Storage: one SQLite file with FTS5.
    implementation("org.xerial:sqlite-jdbc:3.50.3.0")

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("brooklyn.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx24g")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
