import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        freeCompilerArgs.addAll("-Xcontext-parameters")
    }
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
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
}

evaluationDependsOn(":ir")
evaluationDependsOn(":core")

val irJvmJar = project(":ir").tasks.named<Jar>("jvmJar")
val coreJvmJar = project(":core").tasks.named<Jar>("jvmJar")

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    dependsOn(tasks.jar, irJvmJar, coreJvmJar)
    systemProperty("tlaloc.plugin.jar", tasks.jar.flatMap { it.archiveFile }.get().asFile.absolutePath)
    systemProperty("tlaloc.ir.jar", irJvmJar.flatMap { it.archiveFile }.get().asFile.absolutePath)
    systemProperty("tlaloc.core.jar", coreJvmJar.flatMap { it.archiveFile }.get().asFile.absolutePath)
}
