package io.tlaloc.runtime.pjrt

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * [BenchRegistry]: its cases are well formed (here, without a device); every case that
 * is not skipped runs one step on the device; and with `TLALOC_BENCH_OUT` set, the
 * measured cases are timed with [StepBenchmark] and written there as JSON lines
 * (`scripts/bench-registry.sh`).
 */
class BenchRegistryTest {

    @Test
    fun casesAreWellFormed() {
        val names = BenchRegistry.cases.map { it.name }
        assertTrue(names.toSet().size == names.size, "duplicate case names: $names")
        for (c in BenchRegistry.cases) {
            assertTrue(Regex("[a-z0-9_]+").matches(c.name), "case name '${c.name}'")
            assertTrue(c.describe.isNotBlank(), "${c.name} has no description")
            when (val p = c.policy) {
                is BenchRegistry.Policy.SmokeOnly -> assertTrue(p.reason.isNotBlank(), "${c.name}: smoke-only without a reason")
                is BenchRegistry.Policy.Skip -> assertTrue(p.reason.isNotBlank(), "${c.name}: skipped without a reason")
                BenchRegistry.Policy.Measure -> Unit
            }
        }
    }

    @Test
    fun everyCaseThatIsNotSkippedRunsOneStep() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val failures = mutableListOf<String>()
        TestBackend.session().use { session ->
            for (c in BenchRegistry.cases.filter { it.policy !is BenchRegistry.Policy.Skip }) {
                try {
                    val step = c.setup(session)
                    try {
                        step.run()
                    } finally {
                        step.close()
                    }
                } catch (t: Throwable) {
                    failures += "${c.name}: ${t.message?.lineSequence()?.first()}"
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }

    @Test
    fun measure() {
        val out = System.getenv("TLALOC_BENCH_OUT")
        assumeTrue(!out.isNullOrBlank(), "TLALOC_BENCH_OUT is not set (scripts/bench-registry.sh sets it)")
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val lines = ArrayList<String>()
        // The negative control: TLALOC_BENCH_PERTURB_MS adds a wait to every step, which the
        // comparison must report as a regression (scripts/bench-registry.sh --perturb).
        val perturbNs = ((System.getenv("TLALOC_BENCH_PERTURB_MS")?.toDoubleOrNull() ?: 0.0) * 1e6).toLong()
        fun pause() {
            if (perturbNs <= 0) return
            val until = System.nanoTime() + perturbNs
            while (System.nanoTime() < until) Thread.onSpinWait()
        }
        TestBackend.session().use { session ->
            for (c in BenchRegistry.cases.filter { it.policy == BenchRegistry.Policy.Measure }) {
                val step = c.setup(session)
                try {
                    // An idle GB10 sits at a few hundred MHz; two seconds of the step first
                    // bring the clocks up, so the timed steps do not measure the ramp.
                    val until = System.nanoTime() + 2_000_000_000L
                    while (System.nanoTime() < until) step.run()
                    val r = StepBenchmark(minWarmup = 10, maxWarmup = 40, timedSteps = 50)
                        .run(compileCount = { session.cacheSize }, peakBytes = { session.memoryStats().second }) { step.run(); pause() }
                    lines += r.toJson(c.name)
                    println("[bench] ${r.toJson(c.name)}")
                } finally {
                    step.close()
                }
            }
        }
        File(out!!).writeText(lines.joinToString("\n", postfix = "\n"))
    }
}
