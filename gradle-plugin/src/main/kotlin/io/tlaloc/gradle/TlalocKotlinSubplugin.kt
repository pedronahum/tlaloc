package io.tlaloc.gradle

import org.gradle.api.Project
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.plugin.FilesOptionKind
import org.jetbrains.kotlin.gradle.plugin.FilesSubpluginOption
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilation
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilerPluginSupportPlugin
import org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType
import org.jetbrains.kotlin.gradle.plugin.SubpluginArtifact
import org.jetbrains.kotlin.gradle.plugin.SubpluginOption

/**
 * The Kotlin Gradle plugin's side of [TlalocGradlePlugin], applied by it once a Kotlin
 * plugin is present. Maps [TlalocExtension] to the compiler plugin's options.
 */
internal class TlalocKotlinSubplugin : KotlinCompilerPluginSupportPlugin {

    override fun apply(target: Project) = Unit

    override fun isApplicable(kotlinCompilation: KotlinCompilation<*>): Boolean =
        when (kotlinCompilation.platformType) {
            KotlinPlatformType.jvm -> true
            KotlinPlatformType.common -> false
            else -> {
                kotlinCompilation.target.project.logger.warn(
                    "Tlaloc: compiler plugin not applied to Kotlin compilation " +
                        "'${kotlinCompilation.name}' of target '${kotlinCompilation.target.name}' " +
                        "(platform ${kotlinCompilation.platformType.name}). Tlaloc supports " +
                        "Kotlin/JVM compilations only.",
                )
                false
            }
        }

    override fun getCompilerPluginId(): String = TlalocBuildInfo.COMPILER_PLUGIN_ID

    override fun getPluginArtifact(): SubpluginArtifact = SubpluginArtifact(
        groupId = TlalocBuildInfo.GROUP,
        artifactId = TlalocBuildInfo.COMPILER_PLUGIN_ARTIFACT,
        version = TlalocBuildInfo.VERSION,
    )

    override fun applyToCompilation(
        kotlinCompilation: KotlinCompilation<*>,
    ): Provider<List<SubpluginOption>> {
        val project = kotlinCompilation.target.project
        val extension = project.extensions.getByType(TlalocExtension::class.java)
        // Each compilation writes into its own subdirectory, named after its default
        // source set (main, test; jvmMain, jvmTest in Multiplatform), which is an
        // output of its compile task: two compilations cannot share one output
        // directory, and an output is stored in and restored from the build cache.
        val gradSourceDir = extension.dumpGradSourceDir.map { it.dir(kotlinCompilation.defaultSourceSet.name) }
        kotlinCompilation.compileTaskProvider.configure { task ->
            task.outputs.dir(gradSourceDir).optional().withPropertyName(GRAD_SOURCE_OUTPUT_PROPERTY)
        }
        return project.provider {
            buildList {
                add(SubpluginOption("strictLowering", extension.strictLowering.get().toString()))
                add(SubpluginOption("dumpGradSource", extension.dumpGradSource.get().toString()))
                add(SubpluginOption("dumpLoweredIr", extension.dumpLoweredIr.get().toString()))
                add(
                    SubpluginOption(
                        "unsafeAllowUnsupportedKotlin",
                        extension.unsafeAllowUnsupportedKotlin.get().toString(),
                    ),
                )
                // A files option is left out of the compile task's inputs, so the
                // absolute path does not enter the build-cache key; the directory is
                // tracked as the output registered above.
                gradSourceDir.orNull?.let {
                    add(FilesSubpluginOption("dumpGradSourceDir", listOf(it.asFile), FilesOptionKind.INTERNAL))
                }
            }
        }
    }

    internal companion object {
        /** The compile task's output property that holds the dumped gradient sources. */
        const val GRAD_SOURCE_OUTPUT_PROPERTY: String = "tlalocGradSourceDir"
    }
}
