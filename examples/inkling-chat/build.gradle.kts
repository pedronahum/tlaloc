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
    // The serving manifest inkling-export writes (InklingServingExport).
    implementation("io.github.pedronahum:tlaloc-maestro:$tlaloc")
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

// ./gradlew -p examples/inkling-chat export --args="<snapshot dir> <artifact dir> [--contexts 2048,8192] [--slots 4]"
tasks.register<JavaExec>("export") {
    group = "application"
    description = "Writes Inkling-2-Mini as a serving artifact for the Triton backend"
    mainClass.set("InklingExportKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("-Xmx12g")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
}

// ./gradlew -p examples/inkling-chat endpoint --args="--model inkling --served-name inkling-2-mini --tokenizer <snapshot dir> --triton localhost:8131 --triton-http localhost:8130 --port 8140"
tasks.register<JavaExec>("endpoint") {
    group = "application"
    description = "tlaloc-endpoint's OpenAI API in front of Triton serving the exported artifact"
    mainClass.set("io.tlaloc.endpoint.MainKt")
    classpath = sourceSets["main"].runtimeClasspath
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
}
