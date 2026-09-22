import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

// §0.4.355 — the KMP modules get publications from the multiplatform plugin
// automatically; this plain-JVM module declares its own.
//
// §0.4.498 — and the KMP modules also got a sources jar for free, which this one
// did not: `from(components["java"])` publishes the binary jar and nothing else.
// Maven Central requires sources AND javadoc per artifact, so this module was the
// single publication in the repo that could never have passed validation. The
// javadoc jar is attached centrally (root build.gradle.kts, Dokka HTML); the
// sources jar has to be asked for here, because it is the java component that
// carries it.
java {
    withSourcesJar()
}

publishing {
    publications {
        create<org.gradle.api.publish.maven.MavenPublication>("maven") {
            from(components["java"])
        }
    }
}

kotlin {
    // §0.4.503 — the plugin STAYS at 25 while the library modules drop to 21, and the
    // consequence has to be stated rather than implied. Kotlin loads a compiler plugin
    // inside the compiler's own JVM, which for a Gradle build is the toolchain JDK. So
    // a jar of 25 bytecode here means: BUILDING code that contains `grad { }` requires
    // a JDK 25 on the build machine, even though the code it produces, and every
    // library module it links against, runs on a JDK 21.
    //
    // That is the honest reading of the split, and it is narrower than "a consumer on
    // JDK 21 can use grad { }". The supported consumer configuration is
    // `jvmToolchain(25)` + `jvmTarget = JVM_21` — exactly what this repository now
    // does to itself — and it is spelled out in the README's Requirements section,
    // in docs/GETTING_STARTED.md §0 and in docs/ALPHA_PLAN.md's Tier 3 table.
    // Whether the plugin itself could be lowered to 21 was NOT decided here: the
    // split was handed down with `compiler-plugin -> 25` in it.
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
    // §0.4.503 (Tier 3, item 3) — Symja used to arrive here TRANSITIVELY, through
    // :ir's runtime elements. Now that :ir declares it `compileOnly` it does not, and
    // without this line every coarsening test in this module would quietly run
    // engine-free — the C6–C9 corollaries would stop firing and the suite would still
    // be green, which is the worst possible outcome of making a dependency optional.
    testImplementation(libs.symja.core)
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
    // §0.4.503 (Tier 3, item 2) — the version catalog's Kotlin version, handed to
    // `KotlinVersionGuardTest` so the guard's `COMPILED_AGAINST` constant is pinned to
    // the ONE place the version is really declared. Without this the guard would be a
    // second copy of the number, and the failure mode of a second copy is a Kotlin bump
    // that makes the plugin refuse the very compiler this repository builds with.
    systemProperty("tlaloc.kotlin.version", libs.versions.kotlin.get())
}
