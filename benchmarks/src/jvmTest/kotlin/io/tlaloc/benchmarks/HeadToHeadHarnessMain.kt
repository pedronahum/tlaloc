package io.tlaloc.benchmarks

import java.io.File

/**
 * §0.4.236 — JVM-side dump-path entrypoint. Drives [HeadToHeadHarnessRunner.runAllAndDump]
 * with a configurable output directory + warmup/measured counts so the user
 * can produce `harness-results-tlaloc.{csv,json}` via a single Gradle task
 * (`./gradlew :benchmarks:dumpHarnessResults`).
 *
 * The existing [HeadToHeadHarnessAllTest] writes to `Files.createTempDirectory(...)`
 * and deletes the result on tear-down — useful for verifying the runner
 * works, useless for the user's actual cross-framework workflow. This `main`
 * fills that gap.
 *
 * **Environment variables** (read once at start):
 *   - `TLALOC_HARNESS_OUTPUT_DIR` — output directory (default: `build/`).
 *   - `TLALOC_HARNESS_WARMUP` — warmup iterations (default: 200).
 *   - `TLALOC_HARNESS_MEASURED` — measured iterations (default: 800).
 *
 * On success, prints the absolute paths of the written files + a brief
 * per-inhabitant summary. Useful for `aggregate.py` chaining via:
 *
 * ```bash
 * ./gradlew :benchmarks:dumpHarnessResults
 * python harness/python/run_pytorch.py
 * python harness/python/run_jax.py
 * python harness/python/aggregate.py
 * ```
 */
fun main(args: Array<String>) {
    val outputDir = File(System.getenv("TLALOC_HARNESS_OUTPUT_DIR") ?: "build")
    val warmup = System.getenv("TLALOC_HARNESS_WARMUP")?.toIntOrNull() ?: 200
    val measured = System.getenv("TLALOC_HARNESS_MEASURED")?.toIntOrNull() ?: 800

    println("[harness-main] output=${outputDir.absolutePath}")
    println("[harness-main] warmup=$warmup measured=$measured")

    val results = HeadToHeadHarnessRunner.runAllAndDump(
        warmup = warmup,
        measured = measured,
        outputDir = outputDir,
    )

    println()
    println("[harness-main] results written:")
    println("  ${File(outputDir, "harness-results-tlaloc.csv").absolutePath}")
    println("  ${File(outputDir, "harness-results-tlaloc.json").absolutePath}")
    println()
    println("[harness-main] per-inhabitant summary:")
    for (r in results) {
        println(
            "  ${r.benchmark}: forward=${"%.6g".format(r.forwardValue)}, " +
                "median=${r.medianNanos}ns, n_grads=${r.gradientValues.size}",
        )
    }
}
