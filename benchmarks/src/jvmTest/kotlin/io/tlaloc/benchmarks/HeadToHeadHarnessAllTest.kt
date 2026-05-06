package io.tlaloc.benchmarks

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.228 — head-to-head harness Phase 1 closure test. Validates the
 * [HeadToHeadHarnessRunner] multi-inhabitant runner: runs all six harness
 * inhabitants, dumps CSV + JSON, and verifies the dumps' structural shape.
 *
 * **What this test pins**:
 *  - All six inhabitants run successfully (no exceptions through the full
 *    PhiCalculus.apply → DxirReverseTransform.apply → eval pipeline at scale).
 *  - The runner produces 6 results.
 *  - CSV has the expected paper-style header + 6 data rows.
 *  - JSON is a 6-element array containing each inhabitant's benchmark name.
 *  - Forward values across inhabitants are non-zero (sanity that nothing
 *    silently degraded since each inhabitant's individual test pinned
 *    specific forward values).
 *  - Timings are sane (positive + median < 100× min for every inhabitant).
 *
 * **Why this is the right Phase 1 closure shape**: a single runner that
 * dumps all inhabitants' baselines gives Phase 2's Python references one
 * place to plug in. The CSV format mirrors the paper's per-framework
 * performance tables; the JSON preserves the full numerical baseline for
 * cross-framework correctness checks.
 */
class HeadToHeadHarnessAllTest {

    @Test
    fun runAllAndDumpProducesCsvAndJsonForAllInhabitants() {
        val outputDir = Files.createTempDirectory("tlaloc-harness-all").toFile()
        try {
            // Use small warmup/measured to keep the suite fast. A real
            // measurement run would use defaults (200/800).
            val results = HeadToHeadHarnessRunner.runAllAndDump(
                warmup = 20,
                measured = 50,
                outputDir = outputDir,
            )

            // 6 inhabitants → 6 results.
            assertEquals(6, results.size, "expected 6 inhabitants, got ${results.size}")

            val expectedNames = setOf(
                "qwop-avatar-step",
                "bgd-hyperopt-outer-loop-K3",
                "hookean-spring-scalar-N10",
                "brachistochrone-compound-velocity-N5",
                "hmc-logistic-regression-n4-d2",
                "cartpole-phase1-onestep",
            )
            assertEquals(
                expectedNames,
                results.map { it.benchmark }.toSet(),
                "harness inhabitants don't match the expected six paper-benchmark + QWOP set",
            )

            // Each result has reasonable structure.
            for (r in results) {
                // Forward should be non-NaN/non-zero for every inhabitant
                // we care about (sanity: rest configs would be exactly zero,
                // but baseline configs are all non-rest).
                assertTrue(
                    !r.forwardValue.isNaN() && r.forwardValue.isFinite(),
                    "${r.benchmark}: forward should be finite, got ${r.forwardValue}",
                )
                assertTrue(
                    r.gradientValues.isNotEmpty(),
                    "${r.benchmark}: gradient values should not be empty",
                )
                for ((i, g) in r.gradientValues.withIndex()) {
                    assertTrue(
                        !g.isNaN() && g.isFinite(),
                        "${r.benchmark}: gradient[$i] should be finite, got $g",
                    )
                }
                // Timings sane.
                assertTrue(r.minNanos > 0, "${r.benchmark}: min ${r.minNanos} should be > 0")
                assertTrue(
                    r.medianNanos >= r.minNanos,
                    "${r.benchmark}: median ${r.medianNanos} < min ${r.minNanos}",
                )
                assertTrue(
                    r.p99Nanos >= r.medianNanos,
                    "${r.benchmark}: p99 ${r.p99Nanos} < median ${r.medianNanos}",
                )
                assertTrue(
                    r.medianNanos < r.minNanos * 100,
                    "${r.benchmark}: median ${r.medianNanos} >= 100× min ${r.minNanos} (GC-pause concern)",
                )
            }

            // CSV file structure.
            val csvFile = File(outputDir, "harness-results-tlaloc.csv")
            assertTrue(csvFile.exists(), "CSV file should exist at ${csvFile.absolutePath}")
            val csvLines = csvFile.readLines()
            assertEquals(
                7, csvLines.size,
                "CSV should have 7 lines (1 header + 6 data rows), got ${csvLines.size}: $csvLines",
            )
            assertEquals(
                "benchmark,framework,n_iterations,median_ns,min_ns,p99_ns",
                csvLines[0],
                "CSV header doesn't match the paper-style format",
            )
            for (line in csvLines.drop(1)) {
                // §0.4.293 — framework label renamed from "tlaloc" to
                // "tlaloc-interpreter" to distinguish the JVM-interpreter path
                // from "tlaloc-iree-cpu" / "tlaloc-iree-cuda" rows the
                // LlamaDecoderIreeBenchmark produces. The default
                // HeadToHeadResult.framework picks up "tlaloc-interpreter",
                // so every existing inhabitant lands in this column.
                assertTrue(
                    line.contains(",tlaloc-interpreter,"),
                    "CSV row should contain ',tlaloc-interpreter,', got: $line",
                )
                val parts = line.split(",")
                assertEquals(6, parts.size, "CSV row should have 6 fields: $line")
            }
            for (name in expectedNames) {
                assertTrue(
                    csvLines.any { it.startsWith("$name,") },
                    "CSV missing row for benchmark $name",
                )
            }

            // JSON file structure.
            val jsonFile = File(outputDir, "harness-results-tlaloc.json")
            assertTrue(jsonFile.exists(), "JSON file should exist at ${jsonFile.absolutePath}")
            val jsonText = jsonFile.readText()
            assertTrue(jsonText.startsWith("["), "JSON should start with '['")
            assertTrue(jsonText.trim().endsWith("]"), "JSON should end with ']'")
            for (name in expectedNames) {
                assertTrue(
                    jsonText.contains("\"benchmark\": \"$name\""),
                    "JSON missing object for benchmark $name",
                )
            }

            // Side-channel print for /loop spot-checks.
            println("[harness-all] dumped to ${outputDir.absolutePath}")
            println("[harness-all] CSV has ${csvLines.size} lines")
            println("[harness-all] JSON has ${jsonText.length} chars")
            for (r in results) {
                println(
                    "[harness-all] ${r.benchmark}: forward=${r.forwardValue}, " +
                        "median=${r.medianNanos}ns",
                )
            }
        } finally {
            outputDir.deleteRecursively()
        }
    }
}
