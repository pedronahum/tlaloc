// Plain Java: no Kotlin plugin. The Tlaloc jars are ordinary JVM libraries.
plugins {
    java
    application
}

java {
    // :runtime-pjrt is Java 25 bytecode (it binds PJRT with the FFM API).
    toolchain { languageVersion = JavaLanguageVersion.of(25) }
}

dependencies {
    // ServingModel and ServingExport are in this checkout, not in 0.1.0-alpha02:
    // publish it to mavenLocal first.
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha02")
    // HfTokenizer: text to ids and back, from the checkpoint's tokenizer.json.
    implementation("io.github.pedronahum:tlaloc-tokenizer:0.1.0-alpha02")
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
}

application {
    mainClass.set("JavaInference")
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "-Xmx8g")
}
