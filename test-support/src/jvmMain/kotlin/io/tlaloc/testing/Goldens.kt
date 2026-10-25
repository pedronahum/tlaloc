package io.tlaloc.testing

import org.opentest4j.TestAbortedException
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * The oracle venv's Python (JAX, PyTorch), found at an environment variable, else
 * `~/.local/venvs/iree/bin/python`.
 */
object PythonOracle {

    private val found = HashMap<String, String?>()

    /** `TLALOC_NO_ORACLE=1` hides the oracle, so a run replays the recordings as CI does. */
    val disabled: Boolean
        get() = (System.getenv("TLALOC_NO_ORACLE") ?: System.getProperty("TLALOC_NO_ORACLE")) == "1"

    /** A Python that imports every module in [modules], or null. */
    fun python(vararg modules: String, envVar: String = "TLALOC_JAX_PYTHON"): String? = if (disabled) null else synchronized(found) {
        val key = "$envVar:${modules.joinToString(",")}"
        found.getOrPut(key) {
            val candidates = listOfNotNull(
                System.getenv(envVar),
                System.getenv("TLALOC_ORACLE_PYTHON"),
                System.getProperty("user.home")?.let { "$it/.local/venvs/iree/bin/python" },
            )
            candidates.firstOrNull { p ->
                Files.isExecutable(Path.of(p)) && run(p, modules.joinToString("\n") { "import $it" }) != null
            }
        }
    }

    /** Runs [code] with [python] (`-I`, isolated); its output, or null when it fails. */
    fun run(python: String, code: String, timeoutSeconds: Long = 600): String? = runCatching {
        val p = ProcessBuilder(python, "-I", "-c", code).redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        if (p.waitFor(timeoutSeconds, TimeUnit.SECONDS) && p.exitValue() == 0) out else null
    }.getOrNull()

    /**
     * Runs a repository script ([script], under harness/python) with [args]; its output,
     * or null when it fails. Not isolated: the harness scripts import their neighbours.
     */
    fun runScript(python: String, script: Path, args: List<String>, timeoutSeconds: Long = 600): String? = runCatching {
        val p = ProcessBuilder(listOf(python, script.toString()) + args).redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        if (p.waitFor(timeoutSeconds, TimeUnit.SECONDS) && p.exitValue() == 0) out else null
    }.getOrNull()
}

/**
 * Reference outputs recorded from an oracle, so a test that compares against JAX or
 * PyTorch runs where that oracle is not installed (GitHub runners).
 *
 * A golden is keyed by the SHA-256 of what produced it (the oracle program and its
 * inputs, as the caller spells them), so a change to either makes the recording stale.
 * Each golden file's own SHA-256 is pinned in `goldens.pin`, and a file that does not
 * match its pin is refused.
 *
 * [arrays]:
 * - with the oracle, runs it; when a recording exists for the same key, the live result
 *   must agree with it (a changed oracle shows up). With `TLALOC_RECORD_GOLDENS=1` the
 *   live result is written and pinned.
 * - without the oracle, returns the recording; a test without one is skipped, and one
 *   whose recording is stale fails.
 *
 * Files live in `test-support/src/jvmMain/resources/goldens/<suite>/<case>.golden`,
 * plain text: a header, then each array as `name count` and one line of values.
 */
object Goldens {

    private const val HEADER = "# tlaloc golden v1"
    private const val PIN = "goldens/goldens.pin"

    val recording: Boolean
        get() = (System.getenv("TLALOC_RECORD_GOLDENS") ?: System.getProperty("TLALOC_RECORD_GOLDENS")) == "1"

    /** The repository root: the nearest directory up from the working directory with settings.gradle.kts. */
    internal val root: File by lazy {
        generateSequence(File("").absoluteFile) { it.parentFile }.firstOrNull { File(it, "settings.gradle.kts").isFile }
            ?: error("Goldens: no settings.gradle.kts above ${File("").absolutePath}")
    }

    private val sourceDir: File get() = File(root, "test-support/src/jvmMain/resources")

    fun key(vararg parts: String): String = sha256(parts.joinToString("\u0000").toByteArray())

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    /** A recording: named arrays, or one [text] (an oracle's output file) after them. */
    data class Golden(val key: String, val oracle: String, val arrays: Map<String, DoubleArray>, val text: String? = null)

    private fun path(suite: String, case: String): String {
        require(Regex("[a-z0-9_-]+").matches(suite) && Regex("[a-z0-9_-]+").matches(case)) {
            "Goldens: suite and case are lower-case words ($suite/$case)"
        }
        return "goldens/$suite/$case.golden"
    }

    private fun pins(text: String?): Map<String, String> =
        text.orEmpty().lines().filter { it.isNotBlank() && !it.startsWith("#") }.associate { line ->
            val (sha, p) = line.trim().split(Regex("\\s+"), limit = 2)
            p to sha
        }

    private fun resource(p: String): ByteArray? =
        Goldens::class.java.classLoader.getResourceAsStream(p)?.use { it.readBytes() }

    /** The recording for [suite]/[case], checked against its pin; null when there is none. */
    fun read(suite: String, case: String): Golden? {
        val p = path(suite, case)
        val bytes = resource(p) ?: return null
        val pinned = pins(resource(PIN)?.decodeToString())[p]
            ?: throw AssertionError("Goldens: $p is not in $PIN; record it with TLALOC_RECORD_GOLDENS=1")
        val actual = sha256(bytes)
        if (actual != pinned) throw AssertionError("Goldens: $p has SHA-256 $actual, pinned $pinned")
        return parse(bytes.decodeToString(), p)
    }

    private fun parse(text: String, p: String): Golden {
        val lines = text.lines()
        require(lines.firstOrNull() == HEADER) { "Goldens: $p does not start with '$HEADER'" }
        var key = ""
        var oracle = ""
        val arrays = LinkedHashMap<String, DoubleArray>()
        var text: String? = null
        var i = 1
        while (i < lines.size) {
            val l = lines[i]
            when {
                l == "text" -> {
                    text = lines.drop(i + 1).joinToString("\n")
                    i = lines.size
                }
                l.startsWith("key ") -> key = l.removePrefix("key ")
                l.startsWith("oracle ") -> oracle = l.removePrefix("oracle ")
                l.startsWith("array ") -> {
                    val (_, name, n) = l.split(" ")
                    val values = if (n.toInt() == 0) DoubleArray(0) else lines[i + 1].split(",").map { it.toDouble() }.toDoubleArray()
                    require(values.size == n.toInt()) { "Goldens: $p array $name has ${values.size} values, header says $n" }
                    arrays[name] = values
                    i++
                }
            }
            i++
        }
        return Golden(key, oracle, arrays, text)
    }

    private fun format(g: Golden): String = buildString {
        appendLine(HEADER)
        appendLine("key ${g.key}")
        appendLine("oracle ${g.oracle}")
        for ((name, a) in g.arrays) {
            require(Regex("[A-Za-z0-9_.]+").matches(name)) { "Goldens: array name '$name'" }
            appendLine("array $name ${a.size}")
            appendLine(a.joinToString(",") { it.toString() })
        }
        if (g.text != null) {
            appendLine("text")
            append(g.text)
        }
    }

    /** Writes [g] into the source tree and updates the pin. */
    fun write(suite: String, case: String, g: Golden) {
        val p = path(suite, case)
        val file = File(sourceDir, p)
        file.parentFile.mkdirs()
        val text = format(g)
        file.writeText(text)
        val pinFile = File(sourceDir, PIN)
        val pins = pins(if (pinFile.isFile) pinFile.readText() else null).toSortedMap()
        pins[p] = sha256(text.toByteArray())
        pinFile.writeText(
            "# SHA-256 of each golden file (test-support Goldens).\n" +
                pins.entries.joinToString("") { (k, v) -> "$v  $k\n" },
        )
    }

    /**
     * The oracle's arrays for [suite]/[case]. [key] identifies what produces them
     * ([Goldens.key] of the oracle program and inputs); [oracle] names the oracle and
     * its version; [live] runs it and returns null when it is not installed.
     */
    fun arrays(
        suite: String,
        case: String,
        key: String,
        oracle: String,
        live: () -> Map<String, DoubleArray>?,
    ): Map<String, DoubleArray> {
        val recorded = read(suite, case)
        val result = live()
        if (result != null) {
            if (recording) {
                write(suite, case, Golden(key, oracle, result))
            } else if (recorded != null && recorded.key == key) {
                for ((name, want) in recorded.arrays) {
                    val got = result[name] ?: throw AssertionError("Goldens: $suite/$case: the oracle no longer returns '$name'")
                    val err = scaledError(got, want)
                    if (err > 1e-6) {
                        throw AssertionError(
                            "Goldens: $suite/$case '$name': the oracle ($oracle) differs from the recording " +
                                "(${recorded.oracle}) by $err; record again with TLALOC_RECORD_GOLDENS=1 if the change is expected",
                        )
                    }
                }
            }
            return result
        }
        if (recorded == null) throw TestAbortedException("no oracle and no recording for $suite/$case")
        if (recorded.key != key) {
            throw AssertionError(
                "Goldens: $suite/$case is stale (the oracle program or its inputs changed); record it again " +
                    "with the oracle installed and TLALOC_RECORD_GOLDENS=1",
            )
        }
        return recorded.arrays
    }

    /**
     * Whether [item] (an input's identity, for example the SHA-256 of a module's text) passed
     * a check only an oracle can run: [live] runs it and returns null when the oracle is not
     * installed. Items that passed are recorded in the set [suite]/[case] with
     * `TLALOC_RECORD_GOLDENS=1`; without the oracle, an item passed when the set holds it,
     * and null means it was never recorded.
     */
    fun verified(suite: String, case: String, item: String, live: () -> Boolean?): Boolean? {
        val result = live()
        if (result != null) {
            if (recording && result) {
                synchronized(this) {
                    // The source file, not the classpath copy: that one is the set as it was
                    // when this JVM started, without the items recorded since.
                    val source = File(sourceDir, path(suite, case))
                    val current = if (source.isFile) parse(source.readText(), source.path).text else null
                    val set = current.orEmpty().lines().filter { it.isNotBlank() }.toSortedSet()
                    if (set.add(item)) write(suite, case, Golden("verified-set", "", emptyMap(), set.joinToString("\n", postfix = "\n")))
                }
            }
            return result
        }
        val set = read(suite, case)?.text.orEmpty().lines().toSet()
        return if (item in set) true else null
    }

    /** SHA-256 of [text] as lower-case hex. */
    fun sha256Of(text: String): String = sha256(text.toByteArray())

    /**
     * The text [live] returns (an oracle's output file), or its recording: the text
     * counterpart of [arrays]. A live result must equal a recording of the same key
     * byte for byte unless recording.
     */
    fun text(suite: String, case: String, key: String, oracle: String, live: () -> String?): String {
        val recorded = read(suite, case)
        val result = live()
        if (result != null) {
            if (recording) {
                write(suite, case, Golden(key, oracle, emptyMap(), result))
            } else if (recorded != null && recorded.key == key && recorded.text != result) {
                throw AssertionError(
                    "Goldens: $suite/$case: the oracle ($oracle) output differs from the recording " +
                        "(${recorded.oracle}); record again with TLALOC_RECORD_GOLDENS=1 if the change is expected",
                )
            }
            return result
        }
        if (recorded == null) throw TestAbortedException("no oracle and no recording for $suite/$case")
        if (recorded.key != key) {
            throw AssertionError(
                "Goldens: $suite/$case is stale (the oracle script or its inputs changed); record it again " +
                    "with the oracle installed and TLALOC_RECORD_GOLDENS=1",
            )
        }
        return recorded.text ?: throw AssertionError("Goldens: $suite/$case holds arrays, not text")
    }

    /** [key] of [script]'s source, [extra], and every file in [inputs] (name and bytes, sorted by name). */
    fun keyOf(script: Path, inputs: Path?, extra: String = ""): String {
        val parts = mutableListOf(Files.readString(script), extra)
        inputs?.toFile()?.listFiles()?.sortedBy { it.name }?.forEach { f ->
            parts += f.name
            parts += sha256(f.readBytes())
        }
        return key(*parts.toTypedArray())
    }

    /**
     * The output of a harness script run by a Python with [modules], or its recording.
     * [run] runs the script with the Python it is given and returns the output text;
     * [inputs] (a directory the test has written) and [extra] (the script's arguments)
     * enter the key with the script's source.
     */
    fun script(
        suite: String,
        case: String,
        script: Path,
        inputs: Path?,
        extra: String,
        vararg modules: String,
        envVar: String = "TLALOC_TORCH_PYTHON",
        run: (String) -> String,
    ): String {
        val py = PythonOracle.python(*modules, envVar = envVar)
        val oracle = py?.let { p ->
            PythonOracle.run(p, modules.joinToString("; ") { "import $it" } + "; print(" +
                modules.joinToString(" + ' ' + ") { "'$it ' + $it.__version__" } + ")")?.trim()?.lines()?.last()
        } ?: modules.joinToString(" ")
        return text(suite, case, keyOf(script, inputs, extra), oracle) { py?.let(run) }
    }

    /**
     * The comma-separated numbers on the last line [code] prints, from a Python that
     * imports [modules]: run live when one is installed, else the recording of
     * [suite]/[case]. The key is the program itself, which carries its inputs, so inputs
     * computed in Kotlin must be the same bits on every platform: `StrictMath.sin`, not
     * `kotlin.math.sin` (`Math.sin` may differ in the last bit between x86 and arm64).
     */
    fun python(suite: String, case: String, code: String, vararg modules: String = arrayOf("jax")): DoubleArray {
        val py = PythonOracle.python(*modules)
        val oracle = py?.let { p ->
            PythonOracle.run(p, modules.joinToString("; ") { "import $it" } + "; print(" +
                modules.joinToString(" + ' ' + ") { "'$it ' + $it.__version__" } + ")")?.trim()?.lines()?.last()
        } ?: modules.joinToString(" ")
        return arrays(suite, case, key(code), oracle) {
            py?.let { p ->
                val out = PythonOracle.run(p, code) ?: throw AssertionError("Goldens: the oracle program for $suite/$case failed")
                mapOf("r" to out.trim().lines().last().split(",").map { it.trim().toDouble() }.toDoubleArray())
            }
        }.getValue("r")
    }
}
