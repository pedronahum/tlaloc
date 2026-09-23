package io.tlaloc.gradle

import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project

/**
 * `plugins { id("io.github.pedronahum.tlaloc") }`: puts the Tlaloc compiler plugin, at
 * this Gradle plugin's own version, on every Kotlin/JVM compilation of the project and
 * passes it the options of the `tlaloc { }` block ([TlalocExtension]).
 *
 * Requires the Kotlin JVM or Multiplatform Gradle plugin, in either order, declared
 * where this plugin's classes can see it (the same plugins { } block, or a parent
 * project's). In a Multiplatform build, Kotlin/JS, Native, Wasm and Android compilations
 * are skipped with a warning naming them: Tlaloc is published for the JVM only.
 * Metadata compilations are skipped silently.
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
                    applyKotlinSubplugin(target, id)
                }
            }
        }
        target.afterEvaluate { project ->
            if (!kotlinApplied) {
                throw GradleException(
                    "Plugin '$PLUGIN_ID' was applied to ${project.displayName}, which applies " +
                        "neither the Kotlin JVM plugin nor the Kotlin Multiplatform plugin, so " +
                        "there is no Kotlin/JVM compilation to put the Tlaloc compiler plugin on. " +
                        "Apply kotlin(\"jvm\") or kotlin(\"multiplatform\") as well.",
                )
            }
        }
    }

    // TlalocKotlinSubplugin implements an interface of the Kotlin Gradle plugin, so
    // loading it needs that plugin on this plugin's classloader. Gradle gives each
    // plugins { } block its own classloader, parented by the enclosing project's: a
    // Kotlin plugin declared only in a subproject is invisible to a Tlaloc plugin
    // declared in the root.
    private fun applyKotlinSubplugin(target: Project, kotlinPluginId: String) {
        try {
            target.pluginManager.apply(TlalocKotlinSubplugin::class.java)
        } catch (e: NoClassDefFoundError) {
            throw GradleException(
                "Plugin '$PLUGIN_ID' cannot use the Kotlin Gradle plugin ('$kotlinPluginId') " +
                    "that ${target.displayName} applies: the Kotlin plugin is declared in a " +
                    "plugins { } block the Tlaloc plugin's classes cannot see (missing class " +
                    "${e.message?.replace('/', '.')}). Declare both plugins in the same " +
                    "plugins { } block, for example kotlin(\"jvm\") version \"<version>\" " +
                    "apply false next to id(\"$PLUGIN_ID\") in the root build script.",
                e,
            )
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
