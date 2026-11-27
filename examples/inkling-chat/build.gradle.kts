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
    // at a time while they are converted and uploaded.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx12g")
}

// XLA's device pool may grow to this share of the GPU's memory (TLALOC_PJRT_MEMORY_FRACTION;
// Tlaloc's default is 0.5). 0.3 holds the quantized experts, the caches at 8,192 and a 256-token
// chunk on a 128 GB DGX Spark.
val memoryFraction = System.getenv("TLALOC_PJRT_MEMORY_FRACTION") ?: "0.3"
// Compiled programs are kept on disk (TLALOC_PJRT_CACHE_DIR): each chunk size and span compiles once
// per machine instead of at every start.
val cacheDir = System.getenv("TLALOC_PJRT_CACHE_DIR") ?: "${System.getProperty("user.home")}/.cache/tlaloc/inkling-executables"
tasks.named<JavaExec>("run") {
    environment("TLALOC_PJRT_MEMORY_FRACTION", memoryFraction)
    environment("TLALOC_PJRT_CACHE_DIR", cacheDir)
}

// ./gradlew -p examples/inkling-chat serve --args="<snapshot dir> [--port 8100] [--context 8192]"
tasks.register<JavaExec>("serve") {
    group = "application"
    description = "Serves Inkling-2-Mini behind the OpenAI-compatible endpoint, in this JVM"
    mainClass.set("InklingServeKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-Xmx12g")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    environment("TLALOC_PJRT_MEMORY_FRACTION", memoryFraction)
    environment("TLALOC_PJRT_CACHE_DIR", cacheDir)
}
