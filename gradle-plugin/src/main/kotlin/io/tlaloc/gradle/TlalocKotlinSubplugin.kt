package io.tlaloc.gradle

import org.gradle.api.Project
import org.gradle.api.provider.Provider
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
                extension.dumpGradSourceDir.orNull?.let {
                    add(SubpluginOption("dumpGradSourceDir", it.asFile.absolutePath))
                }
            }
        }
    }
}
