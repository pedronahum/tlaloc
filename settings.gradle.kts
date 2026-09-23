rootProject.name = "tlaloc"

// §0.4.504 (Tier 3, CI matrix) — THE KOTLIN VERSION OVERRIDE, and why it exists.
//
// This plugin reads 40 `org.jetbrains.kotlin.fir.*` packages of internal K2 API
// (119 `org.jetbrains.kotlin.*` imports). DIFFKTX_SPEC.md §16 lists "K2 plugin API
// changes across Kotlin releases" as a HIGH-likelihood, HIGH-impact risk and names
// the mitigation in three words: "CI against Kotlin EAP". Nothing implemented it
// until here, and nothing COULD: the Kotlin version is declared once, in
// `gradle/libs.versions.toml`, and a CI lane that wanted a different one had no
// way to ask for it short of rewriting a tracked file with `sed`.
//
//   ./gradlew <tasks> -PtlalocKotlinVersion=2.4.20      (or TLALOC_KOTLIN_VERSION=…)
//
// WHAT IT ACTUALLY MOVES, measured rather than assumed (§0.4.504, locally, against
// 2.4.20 and 2.3.10). The rule is in the root build.gradle.kts and it rewrites every
// `org.jetbrains.kotlin:*` DEPENDENCY. Two things follow, and the second one was a
// surprise:
//
//   * `:compiler-plugin` compiles against THAT `kotlin-compiler-embeddable`'s
//     internal API, and its 83 in-process K2 harnesses run inside THAT compiler.
//     This is the signal the lane is for.
//   * so does everything else — because in Kotlin 2.x the Kotlin Gradle plugin runs
//     compilations through `kotlin-build-tools-impl`, which is in the same group. So
//     the COMPILER swaps for every module while the Kotlin Gradle PLUGIN stays at
//     the catalog's version. Verified by the compiler's own output:
//     `-PtlalocKotlinVersion=2.4.20` produces "The argument '-Xcontext-parameters'
//     is redundant for the current language version 2.4".
//
// WHAT IT CANNOT MOVE: the Kotlin Gradle plugin itself. A `resolutionStrategy`
// rule in `pluginManagement` cannot do it in THIS repository, because the root
// build declares both Kotlin plugins `apply false` and so loads KGP onto the root
// classpath before any module asks for it. Gradle then refuses the module's request:
//
//     Error resolving plugin [id: 'org.jetbrains.kotlin.multiplatform', version: '2.4.20']
//     > The request for this plugin could not be satisfied because the plugin is
//       already on the classpath with a different version (2.4.20)
//
// — naming 2.4.20 on both sides of a "different version" complaint. `useVersion` and
// `useModule` were both tried and both hit it. The consequence is stated rather than
// papered over: this lane tests our source and our plugin against a new COMPILER, not
// against a new KGP. docs/ALPHA_PLAN.md carries the row.
//
// WHAT IT DELIBERATELY DOES NOT DO: rewrite `libs.versions.kotlin`. The catalog
// stays the declared, supported toolchain, so `KotlinVersionGuard.COMPILED_AGAINST`
// keeps being pinned to it (`KotlinVersionGuardTest.COMPILED_AGAINST matches the
// version catalog`) and the override cannot make that pin vacuous. Under an
// override, `KotlinVersionGuardTest.the guard does not refuse the compiler this
// repository builds with` is EXPECTED to fail — that failure is the guard reporting
// the foreign compiler it was written for, and `.github/workflows/kotlin-next.yml`
// treats it as the signal rather than as a break.
//
// Absent the property this file behaves exactly as it did before: same two
// repositories, no resolution rules at all.
pluginManagement {
    val kotlinOverride: String? =
        (
            settings.startParameter.projectProperties["tlalocKotlinVersion"]
                ?: System.getenv("TLALOC_KOTLIN_VERSION")
            )?.trim()?.takeIf { it.isNotEmpty() }

    repositories {
        gradlePluginPortal()
        mavenCentral()
        if (kotlinOverride != null) {
            // Kotlin Betas and RCs are published to Maven Central (2.4.20-Beta1 is
            // there); `-dev-` bootstrap builds are NOT, and only this repository
            // carries them. Added ONLY under an override so the default build's
            // repository set — and therefore its supply chain — is unchanged.
            maven {
                name = "kotlin-eap"
                url = uri("https://maven.pkg.jetbrains.space/kotlin/p/kotlin/eap")
            }
        }
    }

}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        val kotlinOverride: String? =
            (
                settings.startParameter.projectProperties["tlalocKotlinVersion"]
                    ?: System.getenv("TLALOC_KOTLIN_VERSION")
                )?.trim()?.takeIf { it.isNotEmpty() }
        if (kotlinOverride != null) {
            maven {
                name = "kotlin-eap"
                url = uri("https://maven.pkg.jetbrains.space/kotlin/p/kotlin/eap")
            }
        }
    }
}

// Layer 2.5 §0.4.245+ — vendored Netflix/maestro under third-party/maestro/
// participates as a Gradle composite build. A single `./gradlew build` at
// the Tlaloc root drives both Tlaloc's modules and Maestro's module tree;
// `:maestro-tlaloc` (added in L2.5.1) consumes Tlaloc's :maestro jar via
// composite-build dependency substitution without staging through a local
// Maven repo. See docs/vendoring.md for the upgrade procedure.
//
// `name = "vendored-maestro"` overrides the included build's root project
// name (which is `maestro` upstream) to avoid colliding with Tlaloc's own
// `:maestro` module. From the Tlaloc command line, vendored Maestro tasks
// are invoked as `./gradlew :vendored-maestro:<task>`.
includeBuild("third-party/maestro") {
    name = "vendored-maestro"
}

include(":core")
include(":ir")
include(":autograd")
include(":nn")
include(":stablehlo")
include(":compiler-plugin")
include(":benchmarks")
include(":maestro")
include(":runtime-iree")
include(":runtime-pjrt")
include(":runtime-cuda")
include(":kptx")
include(":gradle-plugin")
include(":bom")
