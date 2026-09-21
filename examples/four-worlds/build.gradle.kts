plugins {
    kotlin("jvm") version "2.3.20"
    application
}

kotlin {
    jvmToolchain(25)
}

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    // `program { }` / `workflow { }` and the Maestro descriptor live here.
    implementation("io.tlaloc:maestro:0.0.1-SNAPSHOT")
    // No compiler plugin: `program { }` captures its body with the :autograd
    // runtime tracer, not a plugin lowering. The plugin's role in the four
    // worlds is enforcing scope discipline at compile time, which this example
    // demonstrates by pointing at code that does not compile.
}

application {
    mainClass.set("MainKt")
}
