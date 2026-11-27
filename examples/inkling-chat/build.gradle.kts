plugins {
    kotlin("jvm") version "2.4.20"
    application
}

kotlin {
    jvmToolchain(25)
}

// -PtlalocVersion builds against a locally published Tlaloc (./gradlew publishToMavenLocal
// -PtlalocVersion=...) instead.
val tlaloc = (findProperty("tlalocVersion") as String?) ?: "0.1.0-alpha06"

dependencies {
    // InklingModel: the checkpoint on the GPU, chunked prefill and decode through PJRT
    // (brings tlaloc-nn's InklingLoop and InklingCheckpoint).
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:$tlaloc")
    // HfTokenizer, ChatTemplate.INKLING and InklingOutputParser.
    implementation("io.github.pedronahum:tlaloc-tokenizer:$tlaloc")
    // OpenAiEndpoint and SequenceBackend: inkling-serve's OpenAI-compatible server.
    implementation("io.github.pedronahum:tlaloc-endpoint:$tlaloc")
}

application {
    mainClass.set("InklingChatKt")
    // FFM downcalls into the PJRT plugin need native access. The heap holds one layer's experts
    // at a time while they are dequantized and uploaded.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx24g")
}

// ./gradlew -p examples/inkling-chat serve --args="<snapshot dir> [--port 8100] [--context 8192]"
tasks.register<JavaExec>("serve") {
    group = "application"
    description = "Serves Inkling-2-Mini behind the OpenAI-compatible endpoint, in this JVM"
    mainClass.set("InklingServeKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-Xmx24g")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
}
