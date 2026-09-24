@file:OptIn(ExperimentalCompilerApi::class, CompilerConfiguration.Internals::class)

package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.ExitCode
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.compiler.plugin.CommandLineProcessor
import org.jetbrains.kotlin.compiler.plugin.CompilerPluginRegistrar
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.cli.common.CLIConfigurationKeys
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.diagnostics.Severity
import org.jetbrains.kotlin.diagnostics.impl.DiagnosticsCollectorImpl
import org.jetbrains.kotlin.config.Services
import java.io.File
import java.nio.file.Files
import java.util.jar.JarEntry
import java.util.jar.JarOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The guard's refusal travelling through a real compilation by the compiler this
 * plugin is built against.
 *
 * On a supported compiler the guard never refuses, so the reporting path it takes
 * there — `CompilerConfiguration.report` with the CLI's plugin-initialisation
 * diagnostic, see [GuardReporter] — is otherwise never exercised by a compile. This
 * test loads [ForcedVersionGuardRegistrar] as a compiler plugin (a jar holding only
 * its service file; the class itself comes from the test classpath) and has it run
 * the production guard step, [TlalocCompilerPluginRegistrar.runVersionGuard], with a
 * version string chosen by the test. [ForeignCompilerGuardTest] covers the other
 * direction: this plugin inside a real older compiler.
 */
class GuardReportingTest {

    private data class Message(val severity: CompilerMessageSeverity, val text: String)

    private fun compileWithForcedVersion(
        found: String?,
        vararg options: String,
    ): Pair<ExitCode, List<Message>> {
        ForcedVersionGuardRegistrar.found = found
        ForcedVersionGuardRegistrar.mayRegister = null
        val dir = Files.createTempDirectory("tlaloc-guard-reporting").toFile()
        try {
            val pluginJar = File(dir, "forced-guard.jar")
            JarOutputStream(pluginJar.outputStream()).use { jar ->
                jar.putNextEntry(JarEntry("META-INF/services/${CompilerPluginRegistrar::class.java.name}"))
                jar.write("${ForcedVersionGuardRegistrar::class.java.name}\n".toByteArray())
                jar.closeEntry()
                // The production option parser, so `-P plugin:io.tlaloc.plugin:…` reaches
                // the configuration the guard step reads.
                jar.putNextEntry(JarEntry("META-INF/services/${CommandLineProcessor::class.java.name}"))
                jar.write("${TlalocCommandLineProcessor::class.java.name}\n".toByteArray())
                jar.closeEntry()
            }
            val src = File(dir, "Hello.kt").apply { writeText("fun main() { println(\"hello\") }\n") }
            val messages = mutableListOf<Message>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean = messages.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    messages += Message(severity, message)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(src.absolutePath)
                pluginClasspaths = arrayOf(pluginJar.absolutePath)
                destination = File(dir, "out").absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
                pluginOptions = options
                    .map { "plugin:${TlalocCommandLineProcessor.PLUGIN_ID}:$it" }
                    .toTypedArray()
            }
            val exit = K2JVMCompiler().exec(collector, Services.EMPTY, args)
            return exit to messages
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `a refusal inside a real compile is an ERROR naming both versions and fails the build`() {
        val (exit, messages) = compileWithForcedVersion("2.3.20")
        assertEquals(false, ForcedVersionGuardRegistrar.mayRegister, "the forced registrar never ran the guard")
        assertEquals(ExitCode.COMPILATION_ERROR, exit, "a refusal must fail the compilation; got $messages")
        val refusal = messages.filter { "Tlaloc's K2 compiler plugin" in it.text }
        assertEquals(1, refusal.size, "exactly one refusal must reach the collector; got $messages")
        assertEquals(CompilerMessageSeverity.ERROR, refusal.single().severity)
        assertTrue(
            "built against Kotlin ${KotlinVersionGuard.COMPILED_AGAINST} and the running Kotlin compiler is 2.3.20" in
                refusal.single().text,
            "the refusal must name both versions: ${refusal.single().text}",
        )
    }

    @Test
    fun `with the opt-out the same refusal is a WARNING and the compilation succeeds`() {
        val (exit, messages) = compileWithForcedVersion("2.5.0-Beta1", "unsafeAllowUnsupportedKotlin=true")
        assertEquals(true, ForcedVersionGuardRegistrar.mayRegister, "the opt-out must let the extensions register")
        assertEquals(ExitCode.OK, exit, "a downgraded refusal must not fail the compilation; got $messages")
        val warning = messages.filter { "Tlaloc's K2 compiler plugin" in it.text }
        assertEquals(1, warning.size, "exactly one guard message must reach the collector; got $messages")
        // Kotlin 2.4's CLI renders COMPILER_PLUGIN_INITIALIZATION_WARNING as a
        // STRONG_WARNING (not hidden by -nowarn); the message collector fallback
        // reports a plain WARNING. Either is a warning; neither fails the build.
        assertTrue(
            warning.single().severity in setOf(CompilerMessageSeverity.WARNING, CompilerMessageSeverity.STRONG_WARNING),
            "the opt-out must downgrade the refusal to a warning; got ${warning.single().severity}",
        )
        assertTrue(
            "running Kotlin compiler is 2.5.0-Beta1" in warning.single().text &&
                "unsafeAllowUnsupportedKotlin=true was passed" in warning.single().text,
            "the warning must name the version and the opt-out: ${warning.single().text}",
        )
    }

    @Test
    fun `control - the same harness with the supported version compiles with no message`() {
        val (exit, messages) = compileWithForcedVersion(KotlinVersionGuard.COMPILED_AGAINST)
        assertEquals(true, ForcedVersionGuardRegistrar.mayRegister, "the forced registrar never ran the guard")
        assertEquals(ExitCode.OK, exit, "the supported version must compile; got $messages")
        assertFalse(messages.any { "Tlaloc" in it.text }, "a supported compiler stays silent; got $messages")
    }

    @Test
    fun `under the built-against compiler the guard reports through the CLI diagnostic API`() {
        val diagnostics = DiagnosticsCollectorImpl()
        val configuration = CompilerConfiguration().apply {
            put(CLIConfigurationKeys.DIAGNOSTICS_COLLECTOR, diagnostics)
        }
        val channel = GuardReporter.report(configuration, CompilerMessageSeverity.ERROR, "probe text")
        assertEquals(GuardReporter.Channel.CLI_DIAGNOSTIC, channel)
        val reported = diagnostics.diagnostics.single()
        assertEquals("COMPILER_PLUGIN_INITIALIZATION_ERROR", reported.factoryName)
        assertEquals(Severity.ERROR, reported.severity)
        assertEquals("probe text", reported.renderMessage())
    }

    @Test
    fun `without the CLI diagnostics collector the guard falls back to the message collector`() {
        val collected = mutableListOf<Message>()
        val configuration = CompilerConfiguration().apply {
            @OptIn(MessageCollectorAccess::class)
            put(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY, object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean = collected.isNotEmpty()
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    collected += Message(severity, message)
                }
            })
        }
        val channel = GuardReporter.report(configuration, CompilerMessageSeverity.ERROR, "probe text")
        assertEquals(GuardReporter.Channel.MESSAGE_COLLECTOR, channel)
        assertEquals(listOf(Message(CompilerMessageSeverity.ERROR, "probe text")), collected)
    }

    @Test
    fun `with no channel at all a refusal throws instead of going unheard`() {
        val thrown = assertFailsWith<IllegalStateException> {
            GuardReporter.report(CompilerConfiguration(), CompilerMessageSeverity.ERROR, "probe text")
        }
        assertEquals("probe text", thrown.message)
    }
}

/** Service-loaded by [GuardReportingTest]; runs only the production guard step. */
@OptIn(ExperimentalCompilerApi::class)
class ForcedVersionGuardRegistrar : CompilerPluginRegistrar() {
    override val pluginId: String = "io.tlaloc.plugin.test.forced-guard"
    override val supportsK2: Boolean = true

    override fun ExtensionStorage.registerExtensions(configuration: CompilerConfiguration) {
        mayRegister = TlalocCompilerPluginRegistrar.runVersionGuard(configuration, found)
    }

    companion object {
        @Volatile var found: String? = null
        @Volatile var mayRegister: Boolean? = null
    }
}
