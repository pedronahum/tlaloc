import javax.xml.parsers.DocumentBuilderFactory
import org.gradle.api.publish.maven.tasks.GenerateMavenPom
import org.gradle.plugins.signing.Sign
import org.w3c.dom.Element

plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.dokka) apply false
}

group = "io.tlaloc"

// §0.4.498 — 0.0.1-SNAPSHOT was never a release coordinate; it was a placeholder
// that outlived 140 sections. The first named version is an ALPHA, deliberately:
// docs/COMPATIBILITY.md says in writing what that word buys a consumer and what
// it does not, and CHANGELOG.md starts its record here.
version = "0.1.0-alpha01"

// §0.4.355 — every consumable module publishes to Maven under
// io.tlaloc:<module>:<version>. `./gradlew publishToMavenLocal` is the
// onboarding entry point (docs/GETTING_STARTED.md; examples/quickstart is a
// standalone consumer project resolving from mavenLocal). :benchmarks is a
// test harness, not a library — excluded.
//
// §0.4.498 — the follow-up that §0.4.355 named as "deliberate" lands here:
// Central-mandatory POM metadata, a Dokka javadoc jar per publication,
// in-memory GPG signing, and the Central deploy repository. What is NOT done,
// and cannot be from this machine: an actual release. There are no credentials
// here and a Central upload is irreversible, so the release step is WRITTEN AND
// DRY-RUNNABLE, not certified — docs/RELEASING.md is the procedure and
// docs/ALPHA_PLAN.md carries the honest status.
allprojects {
    group = "io.tlaloc"
    version = rootProject.version
}

// One line per published module. Central requires a <description>; an empty or
// copy-pasted one is how a POM passes validation and still tells a consumer
// nothing. A module missing from this map fails the build by name rather than
// publishing with a placeholder.
val moduleDescriptions = mapOf(
    "core" to
        "Typed tensors for Kotlin: rank, dtype and axis names live in the type system, " +
        "so a transposed weight is a compile error.",
    "ir" to
        "DXIR — Tlaloc's differentiable IR, its reference interpreter, and the " +
        "phi-calculus coarsener that rewrites loops before differentiation.",
    "autograd" to
        "The grad / forward / higher-order transformation surface, and the Tracer that " +
        "captures a DXIR function at runtime. One AD engine, no value tape.",
    "nn" to
        "Layers, initializers, losses, optimizers and parameter handling over Tlaloc's " +
        "typed tensors, with DiffKT's model surface as the parity target.",
    "stablehlo" to
        "StableHLO and Shardy emission from DXIR: the portable artifact Tlaloc programs " +
        "compile to.",
    "compiler-plugin" to
        "The Tlaloc K2 compiler plugin: it rewrites grad { } into synthesized gradient " +
        "code at compile time and turns shape and differentiability misuse into compile errors.",
    "maestro" to
        "Graph orchestration and serving-artifact export, including the Netflix Maestro " +
        "step type that runs a Tlaloc graph as a workflow step.",
    "runtime-iree" to
        "IREE execution backend for Tlaloc's StableHLO artifacts.",
    "runtime-pjrt" to
        "PJRT execution backend for Tlaloc's StableHLO artifacts, bound from Kotlin with " +
        "the Foreign Function & Memory API — no JNI.",
    "runtime-cuda" to
        "CUDA driver bindings via the Foreign Function & Memory API — no JNI.",
    "kptx" to
        "KPTX: a Kotlin DSL, emitter, parser and ISA table for NVIDIA PTX, plus the kernel " +
        "library built with it.",
)

val tlalocUrl = "https://github.com/pedronahum/tlaloc"

// Signing and Central credentials come from Gradle properties first, then the
// environment. BOTH are optional and their absence is a no-op: a contributor's
// `./gradlew publishToMavenLocal` and the GitHub build lane have neither, and
// neither may be made to fail because of that.
val signingKey: Provider<String> =
    providers.gradleProperty("signingInMemoryKey")
        .orElse(providers.environmentVariable("SIGNING_IN_MEMORY_KEY"))
val signingKeyPassword: Provider<String> =
    providers.gradleProperty("signingInMemoryKeyPassword")
        .orElse(providers.environmentVariable("SIGNING_IN_MEMORY_KEY_PASSWORD"))
        .orElse("")
val centralUsername: Provider<String> =
    providers.gradleProperty("centralUsername")
        .orElse(providers.environmentVariable("CENTRAL_USERNAME"))
val centralPassword: Provider<String> =
    providers.gradleProperty("centralPassword")
        .orElse(providers.environmentVariable("CENTRAL_PASSWORD"))

subprojects {
    if (name == "benchmarks") return@subprojects

    apply(plugin = "maven-publish")
    apply(plugin = "signing")
    // Dokka HTML, not Dokka Javadoc: the Javadoc format explicitly does not
    // support Kotlin Multiplatform projects (kotlinlang.org/docs/dokka-javadoc.html)
    // and ten of these eleven modules are KMP. Central validates that a
    // -javadoc.jar EXISTS, not that it contains Javadoc-flavoured HTML, and
    // shipping Dokka HTML under that classifier is what KMP libraries on Central
    // do. Named in docs/ALPHA_PLAN.md so it is not a silent substitution.
    apply(plugin = "org.jetbrains.dokka")

    val moduleName = name
    val moduleDescription = moduleDescriptions[name]
        ?: throw GradleException(
            "§0.4.498: module ':$name' publishes to Maven but has no entry in " +
                "moduleDescriptions in the root build.gradle.kts. Maven Central requires a " +
                "<description>; add one there (or exclude the module from publishing) rather " +
                "than letting it publish without one.",
        )

    val javadocJar = tasks.register<Jar>("dokkaJavadocJar") {
        group = "documentation"
        description = "Package this module's Dokka HTML as the javadoc artifact Central requires"
        archiveClassifier.set("javadoc")
        from(tasks.named("dokkaGeneratePublicationHtml"))
    }

    extensions.configure<PublishingExtension> {
        publications.withType<MavenPublication>().configureEach {
            artifact(javadocJar)
            pom {
                name.set("Tlaloc :: $moduleName")
                description.set(moduleDescription)
                url.set(tlalocUrl)
                inceptionYear.set("2026")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                        distribution.set("repo")
                    }
                }
                developers {
                    developer {
                        id.set("pedronahum")
                        name.set("Pedro N. Rodriguez")
                        url.set("https://github.com/pedronahum")
                    }
                }
                scm {
                    url.set(tlalocUrl)
                    connection.set("scm:git:$tlalocUrl.git")
                    developerConnection.set("scm:git:ssh://git@github.com/pedronahum/tlaloc.git")
                }
                issueManagement {
                    system.set("GitHub Issues")
                    url.set("$tlalocUrl/issues")
                }
            }
        }

        repositories {
            // Declared unconditionally so `./gradlew publishAllPublicationsToCentralRepository
            // --dry-run` shows the real task graph. Credentials are nullable on
            // purpose: with none set the task exists and fails at the wire with
            // 401 instead of failing the whole build at configuration time.
            maven {
                name = "central"
                url = uri(
                    if (rootProject.version.toString().endsWith("SNAPSHOT")) {
                        "https://central.sonatype.com/repository/maven-snapshots/"
                    } else {
                        "https://ossrh-staging-api.central.sonatype.com/service/local/staging/deploy/maven2/"
                    },
                )
                credentials {
                    username = centralUsername.orNull
                    password = centralPassword.orNull
                }
            }
        }
    }

    extensions.configure<SigningExtension> {
        if (signingKey.isPresent) {
            useInMemoryPgpKeys(signingKey.get(), signingKeyPassword.get())
            sign(extensions.getByType<PublishingExtension>().publications)
        }
    }

    // The KMP-plus-signing ordering hazard: Gradle's publish tasks consume the
    // .asc files without declaring the dependency, which is a hard error since
    // Gradle 8. Harmless when no key is set (the Sign collection is empty).
    tasks.withType<AbstractPublishToMaven>().configureEach {
        dependsOn(tasks.withType<Sign>())
    }

    // §0.4.498 — the gate that makes "Central-ready" a checked claim instead of
    // a commit message. It reads the POMs this build actually generates and
    // asserts the six elements Central rejects a bundle for omitting, plus the
    // javadoc and sources artifacts. Wired into `check`, so `./gradlew test`
    // (aliased to check below) fails if a future module publishes bare.
    val pomTasks = tasks.withType<GenerateMavenPom>()
    val pomFiles = provider { pomTasks.map { it.destination } }
    val publicationJavadocArtifacts = provider {
        extensions.getByType<PublishingExtension>().publications
            .withType(MavenPublication::class.java)
            .associate { pub -> pub.name to pub.artifacts.map { it.classifier } }
    }
    val sourcesJarTaskNames = provider { tasks.names.filter { it.endsWith("sourcesJar") } }
    val modulePath = path
    val verifyPomMetadata = tasks.register("verifyPomMetadata") {
        group = "verification"
        description = "Assert every generated POM carries the metadata Maven Central mandates"
        dependsOn(pomTasks)
        val poms = pomFiles
        val javadocArtifacts = publicationJavadocArtifacts
        val sourcesTasks = sourcesJarTaskNames
        doLast {
            val required = listOf("name", "description", "url", "licenses", "developers", "scm")
            val generated = poms.get()
            if (generated.isEmpty()) {
                throw GradleException(
                    "$modulePath: verifyPomMetadata found no generated POM. The module applies " +
                        "maven-publish but declares no publication — it would ship nothing.",
                )
            }
            generated.forEach { pom ->
                // Parsed, not grepped. `text.contains("<name>")` would be
                // satisfied by the <name> inside <licenses><license>, and
                // "<url>" by the one inside <scm> — so a POM missing the two
                // elements Central most often rejects would pass a substring
                // check. Only DIRECT children of <project> count.
                val root = DocumentBuilderFactory.newInstance()
                    .also { it.isNamespaceAware = false }
                    .newDocumentBuilder()
                    .parse(pom)
                    .documentElement
                val topLevel = root.childNodes.let { children ->
                    (0 until children.length).mapNotNull { children.item(it) as? Element }
                }
                val present = topLevel.filter { it.textContent.isNotBlank() }.map { it.tagName }
                val missing = required.filterNot { it in present }
                if (missing.isNotEmpty()) {
                    throw GradleException(
                        "$modulePath: ${pom.name} is missing (or has empty) Central-mandatory " +
                            "POM element(s) ${missing.joinToString(", ")}. Central rejects the " +
                            "bundle. Fix the pom { } block in the root build.gradle.kts. " +
                            "Present top-level elements: ${topLevel.joinToString(", ") { it.tagName }}",
                    )
                }
            }
            javadocArtifacts.get().forEach { (publication, classifiers) ->
                if (!classifiers.contains("javadoc")) {
                    throw GradleException(
                        "$modulePath: publication '$publication' has no javadoc artifact " +
                            "(classifiers: ${classifiers.joinToString(", ").ifEmpty { "none" }}). " +
                            "Central requires one per artifact.",
                    )
                }
            }
            if (sourcesTasks.get().isEmpty()) {
                throw GradleException(
                    "$modulePath: no *sourcesJar task exists, so no sources artifact is " +
                        "published. Central requires one per artifact.",
                )
            }
        }
    }
    tasks.named("check") { dependsOn(verifyPomMetadata) }
}

// §0.4.503 — THE JVM TARGET SPLIT, and the gate that makes it a checked fact.
//
// §0.4.311 put the whole repository on JDK 25 for one stated reason: the Foreign
// Function & Memory API went stable in JEP 454, and Tlaloc's PJRT and CUDA
// bindings are FFM with no JNI. That reason is real — and it applies to three
// modules. Emitting 25 bytecode everywhere else made JDK 25 a hard floor for
// every consumer of `:core`, `:nn` or `:stablehlo`, months ahead of where most
// shops are, in exchange for nothing.
//
// This map is the authoritative split. It is not what configures the modules —
// each module sets its own `jvmTarget` (and the 21 ones additionally compile with
// `-Xjdk-release=21`, so a JDK 22+ API is a compile error rather than a run-time
// NoSuchMethodError). This map is what CHECKS them: `verifyJvmTarget` opens the
// module's jar and reads the major version out of every `.class` byte stream.
// A module missing from the map fails configuration by name, the same way
// `moduleDescriptions` above works, so a new module cannot arrive untargeted.
//
// Class-file major versions: Java 21 = 65, Java 25 = 69 (major = release + 44).
val tlalocJvmTargets = mapOf(
    // The library surface. A consumer can put these on a JDK 21 runtime.
    "core" to 21,
    "ir" to 21,
    "autograd" to 21,
    "nn" to 21,
    "stablehlo" to 21,
    // §0.4.503 decision, asked for explicitly: `:maestro` goes to 21. It imports no
    // `java.lang.foreign` anywhere in `jvmMain`, its four project dependencies are
    // all 21 now, and the serving story it exports is a DIRECTORY (a manifest plus
    // StableHLO bodies) read at serve time by a framework-free ctypes-PJRT Python
    // process — there is no JVM, and therefore no FFM, on the serving side at all.
    // `-Xjdk-release=21` is the proof it needs nothing newer; `exportServingArtifact`
    // now runs on a JDK 21 launcher, which is the proof it works there.
    "maestro" to 21,
    // FFM (JEP 454) — the original and only reason for the 25 floor.
    "runtime-pjrt" to 25,
    "runtime-cuda" to 25,
    "kptx" to 25,
    // Not FFM. See the comment in each module's build file for why each stays at 25.
    "runtime-iree" to 25,
    "compiler-plugin" to 25,
    "benchmarks" to 25,
)

subprojects {
    val expectedTarget = tlalocJvmTargets[name]
        ?: throw GradleException(
            "§0.4.503: module ':$name' has no entry in tlalocJvmTargets in the root " +
                "build.gradle.kts. Every module's JVM bytecode target is a published, " +
                "per-module fact (README Requirements, docs/GETTING_STARTED.md §0); add " +
                "':$name' to that map — and to the docs — rather than letting it inherit " +
                "a number nobody chose.",
        )
    val expectedMajor = expectedTarget + 44
    val modulePath = path
    val moduleName = name

    afterEvaluate {
        // KMP modules jar their JVM output as `jvmJar`; `:compiler-plugin` is a plain
        // JVM module and calls it `jar`. The JAR is read (not the classes directory)
        // because the jar is what a consumer resolves.
        //
        // `:benchmarks` is the exception and needs naming rather than skipping: it has
        // NO jvmMain source at all — every line of it lives in `jvmTest` — so its jar is
        // genuinely empty, and an empty jar passes a bytecode check by vacuity. For that
        // module the gate reads the JVM TEST classes instead, which is where its code
        // actually is. Nothing is exempt; the thing being read is just different.
        val jarTaskName = if (tasks.names.contains("jvmJar")) "jvmJar" else "jar"
        val jarTask = tasks.named<Jar>(jarTaskName)
        val jarFile = jarTask.flatMap { it.archiveFile }
        val codeIsTestOnly = moduleName == "benchmarks"
        val testClassesDir = layout.buildDirectory.dir("classes/kotlin/jvm/test")
        val verifyJvmTarget = tasks.register("verifyJvmTarget") {
            group = "verification"
            description =
                "Assert every class this module emits is Java $expectedTarget bytecode"
            if (codeIsTestOnly) {
                dependsOn("jvmTestClasses")
                inputs.dir(testClassesDir).withPropertyName("moduleTestClasses")
            } else {
                inputs.file(jarFile).withPropertyName("moduleJar")
            }
            doLast {
                val offenders = linkedMapOf<String, Int>()
                var classes = 0
                fun judge(name: String, header: ByteArray) {
                    classes++
                    val major = ((header[6].toInt() and 0xFF) shl 8) or (header[7].toInt() and 0xFF)
                    if (major != expectedMajor) offenders[name] = major
                }
                val what: String
                if (codeIsTestOnly) {
                    val dir = testClassesDir.get().asFile
                    what = dir.absolutePath
                    dir.walkTopDown().filter { it.isFile && it.name.endsWith(".class") }
                        .forEach { f ->
                            val header = ByteArray(8)
                            f.inputStream().use { it.readNBytes(header, 0, 8) }
                            judge(f.relativeTo(dir).path, header)
                        }
                } else {
                    val jar = jarFile.get().asFile
                    what = jar.name
                    java.util.zip.ZipFile(jar).use { zip ->
                        for (entry in zip.entries()) {
                            if (!entry.name.endsWith(".class")) continue
                            // A multi-release jar deliberately carries several versions of
                            // the same class; Tlaloc publishes none, and if one ever appears
                            // this gate would be the wrong place to judge it.
                            if (entry.name.startsWith("META-INF/versions/")) continue
                            val header = ByteArray(8)
                            zip.getInputStream(entry).use { it.readNBytes(header, 0, 8) }
                            judge(entry.name, header)
                        }
                    }
                }
                if (classes == 0) {
                    throw GradleException(
                        "$modulePath: verifyJvmTarget found no .class files in $what. " +
                            "An empty input passes a bytecode check by vacuity, so this is a " +
                            "failure, not a pass.",
                    )
                }
                if (offenders.isNotEmpty()) {
                    throw GradleException(
                        "$modulePath: tlalocJvmTargets says ':$moduleName' emits Java " +
                            "$expectedTarget bytecode (class-file major $expectedMajor), but " +
                            "${offenders.size} of $classes classes in $what disagree. " +
                            "First few: " +
                            offenders.entries.take(5).joinToString(", ") {
                                "${it.key} is major ${it.value} (Java ${it.value - 44})"
                            } +
                            ". Fix the module's jvmTarget, or change the map and the docs " +
                            "that publish the number (README Requirements, " +
                            "docs/GETTING_STARTED.md, docs/ALPHA_PLAN.md).",
                    )
                }
            }
        }
        tasks.named("check") { dependsOn(verifyJvmTarget) }
    }
}

// §0.4.41 — make `./gradlew test` run every subproject's tests, not just those
// where a `test` task exists at the subproject level. The root `test` lifecycle
// task historically only picked up `:compiler-plugin:test` (the plain-JVM module);
// KMP modules (`:core`, `:ir`, `:autograd`, `:stablehlo`) expose their tests via
// `jvmTest` / `allTests` instead of `test`, and so were silently skipped. Aliasing
// `test` to `check` ensures the root task covers the whole suite — in particular,
// `./gradlew test` now catches regressions in `:ir`'s PhiCalculus / interpreter
// tests that pre-§0.4.41 slipped through (the §0.4.40 C5-counter-relax introduced
// three latent failures in PhiCalculusC7/8/9 that this change would have caught).
tasks.register("test") {
    dependsOn(subprojects.map { it.tasks.named("check") })
}

// Round-trip tests in :stablehlo (and future :runtime-iree) shell out to
// `stablehlo-translate`, `sdy-opt`, and `iree-compile`, resolving them via
// /usr/bin/which against the inherited PATH. On Apple Silicon those binaries
// live under /opt/homebrew/bin — which is on PATH for interactive shells (via
// the brew shellenv line in ~/.zshrc) but NOT for non-interactive shells (CI,
// Claude Code's tool harness, IDE-spawned gradle daemons). When PATH is
// missing /opt/homebrew/bin the tests' assumeTrue(...) silently self-skip,
// turning a real toolchain regression into a green build.
//
// Inject /opt/homebrew/bin into every Test task's environment when the dir
// exists, so any way of invoking gradle (CLI, IDE, CI) gets the binaries.
// No-op on Linux/Windows where the directory doesn't exist.
subprojects {
    tasks.withType<Test>().configureEach {
        val brewBin = file("/opt/homebrew/bin")
        if (brewBin.isDirectory) {
            val currentPath = System.getenv("PATH") ?: ""
            if (!currentPath.split(":").contains(brewBin.absolutePath)) {
                environment("PATH", "${brewBin.absolutePath}:$currentPath")
            }
        }
    }
}
