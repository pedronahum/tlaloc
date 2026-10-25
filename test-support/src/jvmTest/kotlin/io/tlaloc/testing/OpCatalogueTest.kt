package io.tlaloc.testing

import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertTrue

class OpCatalogueTest {

    @Test
    fun everyOpKindHasAnEntry() {
        val missing = OpCatalogue.missing()
        assertTrue(missing.isEmpty(), "OpKinds without an OpCatalogue entry: $missing")
    }

    @Test
    fun entriesAreWellFormed() {
        val problems = mutableListOf<String>()
        for (e in OpCatalogue.entries.values) {
            if (e.samples.isEmpty() && e.noSample.isNullOrBlank()) problems += "${e.kind}: no samples and no noSample reason"
            if (e.samples.isNotEmpty() && e.noSample != null) problems += "${e.kind}: samples and a noSample reason"
            for ((s, why) in e.excluded) if (why.isBlank()) problems += "${e.kind}: $s excluded without a reason"
            val names = e.samples.map { it.name }
            if (names.toSet().size != names.size) problems += "${e.kind}: duplicate sample names $names"
        }
        assertTrue(problems.isEmpty(), problems.joinToString("\n"))
    }

    @Test
    fun samplesBuild() {
        val problems = mutableListOf<String>()
        for (e in OpCatalogue.entries.values) for (s in e.samples) {
            try {
                SampleFunctions(s, io.tlaloc.core.F32).loss
            } catch (t: Throwable) {
                problems += "${e.kind}/${s.name}: ${t.message}"
            }
        }
        assertTrue(problems.isEmpty(), problems.joinToString("\n"))
    }

    @Test
    fun tolerancesSuggestTheTightestPassingValues() {
        val (atol, rtol) = Tolerances.suggest(doubleArrayOf(0.5, 10.1), doubleArrayOf(0.4, 10.0))
        assertTrue(kotlin.math.abs(atol - 0.12) < 1e-12, "atol $atol")
        assertTrue(kotlin.math.abs(rtol - 0.012) < 1e-12, "rtol $rtol")
        val msg = runCatching { Tolerances.assertClose(doubleArrayOf(1.0, 2.5), doubleArrayOf(1.0, 2.0), 1e-3, "x") }
            .exceptionOrNull()?.message ?: ""
        assertTrue("worst at [1]" in msg && "would pass with" in msg, msg)
        check(OpKind.entries.isNotEmpty())
    }
}
