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
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    // The K2 compiler plugin: rewrites `grad { }` calls into synthesized
    // gradient code at compile time (and gives you compile-time shape /
    // differentiability errors in the IDE).
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.0.1-SNAPSHOT")
}

application {
    mainClass.set("MainKt")
}
