package io.tlaloc.gradle

import org.gradle.testkit.runner.BuildResult
import org.gradle.testkit.runner.GradleRunner
import org.gradle.testkit.runner.TaskOutcome
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * `tlaloc { dumpGradSourceDir }` in real consumer builds (Gradle TestKit): each
 * compilation writes into its own subdirectory, the build cache restores the dumped
 * files, and a compile task with the option set is relocatable, i.e. a copy of the
 * project at another absolute path takes its result from the cache.
 *
 * The consumers resolve Tlaloc from the build-local repository that
 * gradle-plugin/build.gradle.kts publishes to before this test runs.
 */
class DumpGradSourceBuildCacheTest {

    private val repo: File = File(
        System.getProperty("tlaloc.functionalTest.repo")
            ?: fail("system property tlaloc.functionalTest.repo is not set; run through :gradle-plugin:test"),
    )
    private val kotlinVersion: String = System.getProperty("tlaloc.functionalTest.kotlinVersion")
        ?: fail("system property tlaloc.functionalTest.kotlinVersion is not set; run through :gradle-plugin:test")

    private val work: File = createTempDirectory("tlaloc-dump-cache").toFile()
    private val cacheDir = File(work, "build-cache")

    @AfterTest
    fun cleanUp() {
        work.deleteRecursively()
    }

    /** Writes a consumer project at [dir]. [tlalocBlock] configures the dump directory. */
    private fun consumer(dir: File, tlalocBlock: String, extra: String = "") {
        File(dir, "src/main/kotlin").mkdirs()
        File(dir, "src/test/kotlin").mkdirs()
        val repoUri = repo.toURI()
        File(dir, "settings.gradle.kts").writeText(
            """
            rootProject.name = "consumer"
            pluginManagement {
                repositories {
                    maven { url = uri("$repoUri") }
                    gradlePluginPortal()
                    mavenCentral()
                }
            }
            dependencyResolutionManagement {
                repositories {
                    maven { url = uri("$repoUri") }
                    mavenCentral()
                }
            }
            buildCache {
                local { directory = file("${cacheDir.absolutePath.replace("\\", "/")}") }
            }
            """.trimIndent(),
        )
        File(dir, "build.gradle.kts").writeText(
            """
            plugins {
                id("io.github.pedronahum.tlaloc") version "${TlalocBuildInfo.VERSION}"
                kotlin("jvm") version "$kotlinVersion"
            }
            dependencies {
                implementation("${TlalocBuildInfo.GROUP}:tlaloc-core:${TlalocBuildInfo.VERSION}")
                implementation("${TlalocBuildInfo.GROUP}:tlaloc-autograd:${TlalocBuildInfo.VERSION}")
            }
            $tlalocBlock
            $extra
            """.trimIndent(),
        )
        File(dir, "src/main/kotlin/Cube.kt").writeText(
            """
            import io.tlaloc.autograd.grad

            fun cubeSlope(x: Float): Float = grad { v: Float -> v * v * v }(x)
            """.trimIndent(),
        )
        File(dir, "src/test/kotlin/Square.kt").writeText(
            """
            import io.tlaloc.autograd.grad

            fun squareSlope(x: Float): Float = grad { v: Float -> v * v }(x)
            """.trimIndent(),
        )
    }

    private fun build(dir: File, vararg args: String): BuildResult =
        GradleRunner.create()
            .withProjectDir(dir)
            .withArguments(*args, "--build-cache", "--stacktrace")
            .forwardOutput()
            .build()

    private fun dumped(dir: File): Map<String, String> =
        dir.walkTopDown().filter { it.isFile && it.name.endsWith(".kt") }
            .associate { it.relativeTo(dir).invariantSeparatorsPath to it.readText() }

    private val pluginDump = """
        tlaloc {
            dumpGradSourceDir.set(layout.buildDirectory.dir("gradients"))
        }
    """.trimIndent()

    @Test
    fun mainAndTestWriteToSeparateDirectoriesThatTheCacheRestoresAndRelocates() {
        val first = File(work, "a/consumer")
        consumer(first, pluginDump)

        val built = build(first, "compileKotlin", "compileTestKotlin")
        assertEquals(TaskOutcome.SUCCESS, built.task(":compileKotlin")?.outcome)
        assertEquals(TaskOutcome.SUCCESS, built.task(":compileTestKotlin")?.outcome)
        val gradients = File(first, "build/gradients")
        val written = dumped(gradients)
        val mainFiles = written.keys.filter { it.startsWith("main/") }
        val testFiles = written.keys.filter { it.startsWith("test/") }
        assertEquals(written.keys, (mainFiles + testFiles).toSet(), "files outside main/ and test/: $written")
        assertTrue(mainFiles.singleOrNull()?.startsWith("main/Cube_kt_") == true, "main/ holds $mainFiles")
        assertTrue(testFiles.singleOrNull()?.startsWith("test/Square_kt_") == true, "test/ holds $testFiles")

        // A clean build takes both compilations, and the dumped files, from the cache.
        File(first, "build").deleteRecursively()
        val restored = build(first, "compileKotlin", "compileTestKotlin")
        assertEquals(TaskOutcome.FROM_CACHE, restored.task(":compileKotlin")?.outcome)
        assertEquals(TaskOutcome.FROM_CACHE, restored.task(":compileTestKotlin")?.outcome)
        assertEquals(written, dumped(gradients))

        // The same project at another absolute path (a different depth, too) hits the
        // same cache entries.
        val second = File(work, "elsewhere/nested/consumer")
        consumer(second, pluginDump)
        val relocated = build(second, "compileKotlin", "compileTestKotlin")
        assertEquals(TaskOutcome.FROM_CACHE, relocated.task(":compileKotlin")?.outcome)
        assertEquals(TaskOutcome.FROM_CACHE, relocated.task(":compileTestKotlin")?.outcome)
        assertEquals(written, dumped(File(second, "build/gradients")))
    }

    /**
     * The check above can fail: the same dump directory passed as a raw compiler
     * argument puts its absolute path into the compile task's inputs, and the copy of
     * the project at another path misses the cache.
     */
    @Test
    fun anAbsolutePathInTheCompilerArgumentsDoesNotRelocate() {
        val rawDump = """
            kotlin {
                compilerOptions.freeCompilerArgs.addAll(
                    "-P",
                    "plugin:io.tlaloc.plugin:dumpGradSourceDir=" +
                        layout.buildDirectory.dir("gradients").get().asFile.absolutePath,
                )
            }
        """.trimIndent()
        val first = File(work, "a/consumer")
        consumer(first, "", rawDump)
        assertEquals(TaskOutcome.SUCCESS, build(first, "compileKotlin").task(":compileKotlin")?.outcome)
        assertTrue(dumped(File(first, "build/gradients")).isNotEmpty(), "the raw option wrote no file")

        val second = File(work, "elsewhere/nested/consumer")
        consumer(second, "", rawDump)
        val relocated = build(second, "compileKotlin")
        assertNotEquals(TaskOutcome.FROM_CACHE, relocated.task(":compileKotlin")?.outcome)
    }
}
