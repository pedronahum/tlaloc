package io.tlaloc.gradle

import org.gradle.api.Project
import org.gradle.api.internal.project.ProjectInternal
import org.gradle.testfixtures.ProjectBuilder
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.SubpluginOption
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The Gradle plugin, applied with the real Kotlin Gradle plugin in a ProjectBuilder
 * project: what it puts on the compile tasks, which compilations it skips, and what it
 * refuses. `scripts/onboarding-smoke.sh` runs the same plugin end to end from mavenLocal.
 */
class TlalocGradlePluginTest {

    private fun project(configure: Project.() -> Unit): Project {
        val project = ProjectBuilder.builder().build()
        project.configure()
        (project as ProjectInternal).evaluate()
        return project
    }

    private fun tlalocOptions(project: Project, task: String): List<SubpluginOption> {
        val compile = project.tasks.getByName(task) as KotlinCompile
        val byPlugin = compile.pluginOptions.get().flatMap { it.allOptions().entries }
        return byPlugin.filter { it.key == TlalocBuildInfo.COMPILER_PLUGIN_ID }
            .flatMap { it.value }
    }

    private fun Throwable.messages(): String =
        generateSequence(this) { it.cause }.joinToString(" | ") { it.message.orEmpty() }

    @Test
    fun theDefaultsReachTheCompilerAndThePluginArtifactIsThisVersion() {
        val p = project {
            pluginManager.apply("org.jetbrains.kotlin.jvm")
            pluginManager.apply(TlalocGradlePlugin::class.java)
        }
        val options = tlalocOptions(p, "compileKotlin").associate { it.key to it.value }
        assertEquals(
            mapOf(
                "strictLowering" to "true",
                "dumpGradSource" to "false",
                "dumpLoweredIr" to "false",
                "unsafeAllowUnsupportedKotlin" to "false",
            ),
            options,
        )
        val classpath = p.configurations.getByName("kotlinCompilerPluginClasspathMain")
        val coordinates = classpath.allDependencies.map { "${it.group}:${it.name}:${it.version}" }
        assertTrue(
            "${TlalocBuildInfo.GROUP}:tlaloc-compiler-plugin:${TlalocBuildInfo.VERSION}" in coordinates,
            "kotlinCompilerPluginClasspathMain holds $coordinates",
        )
    }

    @Test
    fun everyTlalocOptionIsPassedOnToEveryJvmCompilation() {
        val p = project {
            pluginManager.apply(TlalocGradlePlugin::class.java) // before Kotlin: order must not matter
            pluginManager.apply("org.jetbrains.kotlin.jvm")
            extensions.getByType(TlalocExtension::class.java).apply {
                strictLowering.set(false)
                dumpGradSource.set(true)
                dumpLoweredIr.set(true)
                unsafeAllowUnsupportedKotlin.set(true)
                dumpGradSourceDir.set(layout.buildDirectory.dir("gradients"))
            }
        }
        val expectedDir = p.layout.buildDirectory.dir("gradients").get().asFile.absolutePath
        for (task in listOf("compileKotlin", "compileTestKotlin")) {
            val options = tlalocOptions(p, task)
            assertEquals(
                mapOf(
                    "strictLowering" to "false",
                    "dumpGradSource" to "true",
                    "dumpLoweredIr" to "true",
                    "unsafeAllowUnsupportedKotlin" to "true",
                    "dumpGradSourceDir" to expectedDir,
                ),
                options.associate { it.key to it.value },
                task,
            )
        }
    }

    @Test
    fun theCompilerPluginIdIsTheCompilerPluginsOwn() {
        // Read at build time from TlalocCommandLineProcessor.PLUGIN_ID.
        assertEquals("io.tlaloc.plugin", TlalocBuildInfo.COMPILER_PLUGIN_ID)
    }

    @Test
    fun aJsCompilationIsSkippedAndTheJvmOneIsNot() {
        val p = project {
            pluginManager.apply("org.jetbrains.kotlin.multiplatform")
            pluginManager.apply(TlalocGradlePlugin::class.java)
            extensions.getByType(KotlinMultiplatformExtension::class.java).apply {
                jvm()
                js { nodejs() }
            }
        }
        val kotlin = p.extensions.getByType(KotlinMultiplatformExtension::class.java)
        val subplugin = TlalocKotlinSubplugin()
        val jvmMain = kotlin.targets.getByName("jvm").compilations.getByName("main")
        val jsMain = kotlin.targets.getByName("js").compilations.getByName("main")
        assertTrue(subplugin.isApplicable(jvmMain))
        assertFalse(subplugin.isApplicable(jsMain))
        assertFalse(subplugin.isApplicable(kotlin.targets.getByName("metadata").compilations.getByName("main")))
        assertTrue(tlalocOptions(p, "compileKotlinJvm").isNotEmpty())
    }

    @Test
    fun withoutAKotlinPluginTheBuildIsRefusedByName() {
        val error = runCatching {
            project { pluginManager.apply(TlalocGradlePlugin::class.java) }
        }.exceptionOrNull() ?: fail("a project with no Kotlin plugin was accepted")
        assertTrue(
            "applies neither the Kotlin JVM plugin nor the Kotlin Multiplatform plugin" in error.messages(),
            error.messages(),
        )
    }

    @Test
    fun theExtensionIsNamedTlaloc() {
        val p = project {
            pluginManager.apply("org.jetbrains.kotlin.jvm")
            pluginManager.apply(TlalocGradlePlugin::class.java)
        }
        assertNotNull(p.extensions.findByName("tlaloc"))
        assertNotNull(p.extensions.getByType(KotlinJvmProjectExtension::class.java))
    }
}
