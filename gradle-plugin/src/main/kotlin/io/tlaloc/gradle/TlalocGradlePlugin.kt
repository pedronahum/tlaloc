package io.tlaloc.gradle

import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project

/**
 * `plugins { id("io.github.pedronahum.tlaloc") }`: puts the Tlaloc compiler plugin, at
 * this Gradle plugin's own version, on every Kotlin/JVM compilation of the project and
 * passes it the options of the `tlaloc { }` block ([TlalocExtension]).
 *
 * Requires the Kotlin JVM or Multiplatform Gradle plugin, in either order. Kotlin/JS,
 * Native, Wasm and Android compilations are skipped with a warning naming them: Tlaloc
 * is published for the JVM only. Multiplatform metadata compilations are skipped
 * silently.
 *
 * This class does not reference the Kotlin Gradle plugin, so it loads without it and can
 * say that it is missing; [TlalocKotlinSubplugin] does the work once a Kotlin plugin is
 * applied.
 */
class TlalocGradlePlugin : Plugin<Project> {

    override fun apply(target: Project) {
        val extension = target.extensions.create(EXTENSION_NAME, TlalocExtension::class.java)
        extension.applyConventions()
        var kotlinApplied = false
        KOTLIN_PLUGIN_IDS.forEach { id ->
            target.pluginManager.withPlugin(id) {
                if (!kotlinApplied) {
                    kotlinApplied = true
                    target.pluginManager.apply(TlalocKotlinSubplugin::class.java)
                }
            }
        }
        target.afterEvaluate { project ->
            if (!kotlinApplied) {
                throw GradleException(
                    "Plugin '$PLUGIN_ID' was applied to ${project.displayName}, which applies " +
                        "no Kotlin Gradle plugin, so there is no Kotlin compilation to put the " +
                        "Tlaloc compiler plugin on. Apply kotlin(\"jvm\") or " +
                        "kotlin(\"multiplatform\") as well.",
                )
            }
        }
    }

    companion object {
        const val PLUGIN_ID: String = "io.github.pedronahum.tlaloc"
        const val EXTENSION_NAME: String = "tlaloc"
        private val KOTLIN_PLUGIN_IDS = listOf(
            "org.jetbrains.kotlin.jvm",
            "org.jetbrains.kotlin.multiplatform",
        )
    }
}
