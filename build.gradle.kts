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
