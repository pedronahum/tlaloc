package io.tlaloc.benchmarks

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.238 — end-to-end integration coverage between the JVM-side harness
 * dump and the Python aggregator. Catches a class of silent bugs that
 * neither `HeadToHeadHarnessAllTest` (JVM-side) nor `aggregate_test.py`
 * (Python-side) covers: **JSON shape divergence between the two halves**.
 *
 * Concrete bug class: if someone renames `HeadToHeadResult.toJsonString()`'s
 * `forwardValue` field to `forward_value`, all the Kotlin tests still pass
 * (they assert on the JSON string content via `contains`), and all the
 * Python aggregator tests still pass (they use synthetic fixtures with the
 * old shape). Only this test — which runs the JVM dump, then invokes
 * `aggregate.py` against the produced JSON — would fail.
 *
 * **Toolchain handling**: the test gracefully skips if `python3` is not
 * available on `PATH`. This respects the /loop's "no toolchain installs"
 * rule: most dev/CI environments have Python preinstalled, but Tlaloc
 * itself doesn't require it. When Python isn't available, the test prints
 * a skip message and returns without asserting on aggregator behaviour.
 */
class HeadToHeadHarnessE2ETest {

    @Test
    fun jvmDumpRoundTripsThroughPythonAggregator() {
        if (!isPython3Available()) {
            println("[harness-e2e] python3 not available — skipping aggregator round-trip test")
            return
        }

        // Locate aggregate.py relative to project root. Walk up from the
        // test's working directory until we find harness/python/aggregate.py.
        val aggregator = findAggregatePy()
        if (aggregator == null) {
            println("[harness-e2e] harness/python/aggregate.py not found — skipping")
            return
        }

        val tmpDir = Files.createTempDirectory("tlaloc-harness-e2e").toFile()
        try {
            // Run the JVM-side dump with a small warmup/measured to keep
            // the test fast (~3 seconds).
            HeadToHeadHarnessRunner.runAllAndDump(
                warmup = 20,
                measured = 50,
                outputDir = tmpDir,
            )

            // Verify the JVM dump produced both files.
            val jsonFile = File(tmpDir, "harness-results-tlaloc.json")
            val csvFile = File(tmpDir, "harness-results-tlaloc.csv")
            assertTrue(jsonFile.exists(), "JVM dump should have produced ${jsonFile.absolutePath}")
            assertTrue(csvFile.exists(), "JVM dump should have produced ${csvFile.absolutePath}")

            // Run the aggregator against the dump.
            val result = ProcessBuilder("python3", aggregator.absolutePath, "--input", tmpDir.absolutePath)
                .redirectErrorStream(false)
                .start()
            val stdout = result.inputStream.bufferedReader().readText()
            val stderr = result.errorStream.bufferedReader().readText()
            val exitCode = result.waitFor()

            assertEquals(
                0, exitCode,
                "aggregator should exit 0; stderr was:\n$stderr\nstdout:\n$stdout",
            )

            // The aggregator's output should be a valid Markdown table mentioning
            // every harness inhabitant by name (since they all came from the JVM dump).
            val expectedNames = listOf(
                "qwop-avatar-step",
                "bgd-hyperopt-outer-loop-K3",
                "hookean-spring-scalar-N10",
                "brachistochrone-compound-velocity-N5",
                "hmc-logistic-regression-n4-d2",
                "cartpole-phase1-onestep",
            )
            for (name in expectedNames) {
                assertTrue(
                    stdout.contains(name),
                    "aggregator stdout should mention $name; got:\n$stdout",
                )
            }

            // Cross-framework cells should show `—` since only the JVM dump exists.
            assertTrue(
                stdout.contains("—"),
                "aggregator should show `—` cells when only Tlaloc data is present",
            )

            // The JSON-shape contract: `forwardValue` is the field name produced
            // by `HeadToHeadResult.toJsonString()` and consumed by the aggregator.
            // If either side renames it, this test fails (the aggregator would
            // raise a KeyError trying to read 'forwardValue').
            val jsonText = jsonFile.readText()
            assertTrue(
                jsonText.contains("\"forwardValue\""),
                "JVM-produced JSON should contain `forwardValue` field — " +
                    "if this fails, the aggregator's parser will likely fail next",
            )

            println("[harness-e2e] aggregator round-trip succeeded; ${stdout.lines().size} lines of Markdown")
        } finally {
            tmpDir.deleteRecursively()
        }
    }

    private fun isPython3Available(): Boolean {
        return try {
            val proc = ProcessBuilder("python3", "--version")
                .redirectErrorStream(true)
                .start()
            proc.waitFor() == 0
        } catch (_: Throwable) {
            false
        }
    }

    private fun findAggregatePy(): File? {
        // Walk up from the current working directory looking for
        // harness/python/aggregate.py. The :benchmarks subproject runs
        // tests from $projectRoot/benchmarks, so harness/python is at
        // ../harness/python relative to that.
        var dir: File? = File("").absoluteFile
        repeat(5) {
            val candidate = File(dir, "harness/python/aggregate.py")
            if (candidate.exists()) return candidate
            dir = dir?.parentFile
        }
        return null
    }
}
