package io.tlaloc.plugin

import java.io.File
import java.nio.file.Files
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The version guard inside a Kotlin compiler this plugin was NOT built against.
 *
 * [KotlinVersionGuardTest] pins every decision the guard makes as a function of a
 * version string. This class pins the part a pure function cannot: that the plugin
 * jar this build produces, loaded by a real compiler of the previous feature release
 * (`libs.versions.previousKotlin`, the family 0.1.0-alpha01 supports), gets as far
 * as the guard, and that the guard's report reaches the user as a compiler ERROR
 * naming both versions — rather than as a `NoClassDefFoundError` or
 * `NoSuchMethodError` from the reporting call itself. The guard reports through a
 * Kotlin 2.4 API that a 2.3 compiler does not have; this is the test that proves the
 * fallback in [GuardReporter] is taken there.
 *
 * Each compiler runs as its own JVM (`K2JVMCompiler` main class) with its own
 * dependencies, resolved by the `foreignKotlinCompiler` / `currentKotlinCompiler`
 * configurations in `compiler-plugin/build.gradle.kts`. The same harness run on the
 * catalog's compiler is the control: it must compile the same file with no Tlaloc
 * message at all, so a refusal seen under the previous compiler is caused by the
 * version and not by the harness.
 */
class ForeignCompilerGuardTest {

    private data class Run(val exit: Int, val output: String)

    private val current: String = prop("tlaloc.kotlin.version")
    private val previous: String = prop("tlaloc.kotlin.previousVersion")

    private fun prop(name: String): String =
        System.getProperty(name) ?: fail("system property $name is not set; run through :compiler-plugin:test")

    private fun pluginClasspath(): String = listOf(
        prop("tlaloc.plugin.jar"),
        prop("tlaloc.ir.jar"),
        prop("tlaloc.core.jar"),
    ).joinToString(",") // -Xplugin takes a comma-separated list

    private fun compileWith(compilerClasspath: String, vararg pluginOptions: String): Run {
        val dir = Files.createTempDirectory("tlaloc-foreign-compiler").toFile()
        try {
            val src = File(dir, "Hello.kt").apply { writeText("fun main() { println(\"hello\") }\n") }
            val stdlib = compilerClasspath.split(File.pathSeparator)
                .firstOrNull { File(it).name.matches(Regex("kotlin-stdlib-\\d.*\\.jar")) }
                ?: fail("no kotlin-stdlib jar on the compiler classpath: $compilerClasspath")
            val java = File(System.getProperty("java.home"), "bin/java").absolutePath
            val command = mutableListOf(
                java, "--enable-native-access=ALL-UNNAMED", "-cp", compilerClasspath,
                "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
                "-no-stdlib", "-no-reflect", "-classpath", stdlib,
                "-Xplugin=${pluginClasspath()}",
                "-d", File(dir, "out").absolutePath,
            )
            for (opt in pluginOptions) {
                command += listOf("-P", "plugin:${TlalocCommandLineProcessor.PLUGIN_ID}:$opt")
            }
            command += src.absolutePath
            val process = ProcessBuilder(command).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertTrue(process.waitFor(5, TimeUnit.MINUTES), "the compiler process did not finish")
            return Run(process.exitValue(), output)
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun assertNoLinkageFailure(run: Run) {
        // Fully qualified: the refusal's own prose mentions NoSuchMethodError by name.
        val markers = listOf(
            "java.lang.NoClassDefFoundError", "java.lang.NoSuchMethodError",
            "java.lang.NoSuchFieldError", "java.lang.AbstractMethodError", "exception:",
        )
        for (marker in markers) {
            assertFalse(
                marker in run.output,
                "the compiler reported '$marker' — the plugin failed to link instead of refusing cleanly:\n${run.output}",
            )
        }
    }

    @Test
    fun `the previous Kotlin feature release is refused by name, as a compiler error`() {
        assertNotEquals(
            KotlinVersionGuard.Verdict.Supported, KotlinVersionGuard.verdict(previous),
            "previousKotlin ($previous) is inside the supported range, so this test would certify nothing",
        )
        val run = compileWith(prop("tlaloc.foreignCompiler.classpath"))
        assertNoLinkageFailure(run)
        assertNotEquals(0, run.exit, "a refusal must fail the compilation:\n${run.output}")
        assertTrue(
            Regex("""(?m)^error: [Tt]laloc's K2 compiler plugin is built against Kotlin \Q$current\E and the running Kotlin compiler is \Q$previous\E""")
                .containsMatchIn(run.output),
            "the $previous compiler must print the guard's refusal as an error, naming both versions:\n${run.output}",
        )
        assertTrue("REFUSING" in run.output, "the refusal must say it is refusing:\n${run.output}")
    }

    @Test
    fun `under the previous release the opt-out gets past the guard and fails as an error, not a crash`() {
        val run = compileWith(prop("tlaloc.foreignCompiler.classpath"), "unsafeAllowUnsupportedKotlin=true")
        assertFalse("REFUSING" in run.output, "the opt-out must not claim to refuse:\n${run.output}")
        // Past the opt-out, registering the FIR extension does not link against the
        // previous release. That must arrive as a compiler error naming both versions and
        // the opt-out — not as a crash, and not as a bare stack trace. (The guard's own
        // warning is reported too, but the 2.3 CLI prints no warnings once there is an
        // error, so the error carries the same facts.)
        assertNoLinkageFailure(run)
        assertNotEquals(0, run.exit, "the unlinkable plugin must fail the compilation:\n${run.output}")
        assertTrue(
            Regex(
                """(?m)^error: [Tt]laloc's K2 compiler plugin could not register its extensions in Kotlin \Q$previous\E: """ +
                    """it is built against Kotlin \Q$current\E .*unsafeAllowUnsupportedKotlin is set""",
            ).containsMatchIn(run.output),
            "the link failure must be reported as an error naming both versions and the opt-out:\n${run.output}",
        )
    }

    @Test
    fun `control - the catalog's compiler in the same harness compiles with no Tlaloc message`() {
        val run = compileWith(prop("tlaloc.currentCompiler.classpath"))
        assertNoLinkageFailure(run)
        assertEquals(0, run.exit, "the $current compiler must compile the file:\n${run.output}")
        assertFalse("Tlaloc" in run.output, "a supported compiler must not print a Tlaloc message:\n${run.output}")
    }
}
