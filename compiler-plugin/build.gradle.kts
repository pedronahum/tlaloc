import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

// §0.4.355 — the KMP modules get publications from the multiplatform plugin
// automatically; this plain-JVM module declares its own.
publishing {
    publications {
        create<org.gradle.api.publish.maven.MavenPublication>("maven") {
            from(components["java"])
        }
    }
}

kotlin {
    jvmToolchain(25)

    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_25)
        freeCompilerArgs.addAll("-Xcontext-parameters")
    }
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

dependencies {
    compileOnly(libs.kotlin.compiler.embeddable)
    implementation(project(":ir"))
    implementation(project(":core"))

    testImplementation(kotlin("test"))
    testImplementation(libs.kotlin.compiler.embeddable)
    testImplementation(libs.kotest.assertions.core)
    testImplementation(libs.kotest.runner.junit5)
    // §0.4.58 — real :autograd on the test classpath so the Tracer-surface
    // integration test can prove the plugin falls back and the Tracer-capture route
    // (§0.4.446: the same compiler engine, traced at runtime) produces the correct
    // gradient end-to-end (not via a broken-stub sentinel).
    testImplementation(project(":autograd"))
}

evaluationDependsOn(":ir")
evaluationDependsOn(":core")
evaluationDependsOn(":autograd")

val irJvmJar = project(":ir").tasks.named<Jar>("jvmJar")
val coreJvmJar = project(":core").tasks.named<Jar>("jvmJar")
val autogradJvmJar = project(":autograd").tasks.named<Jar>("jvmJar")

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    dependsOn(tasks.jar, irJvmJar, coreJvmJar, autogradJvmJar)
    systemProperty("tlaloc.plugin.jar", tasks.jar.flatMap { it.archiveFile }.get().asFile.absolutePath)
    systemProperty("tlaloc.ir.jar", irJvmJar.flatMap { it.archiveFile }.get().asFile.absolutePath)
    systemProperty("tlaloc.core.jar", coreJvmJar.flatMap { it.archiveFile }.get().asFile.absolutePath)
}
