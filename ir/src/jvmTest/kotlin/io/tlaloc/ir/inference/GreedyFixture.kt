package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A greedy fixture (`{"p<i>": [prompt, ids, margin]}`) against the one a test computed: the
 * prompts and ids exactly, the margins within [tolerance]. The JVM's `exp`, `sin` and `cos` may
 * differ by an ulp between machines, so the margin's last digits are not the fixture's.
 */
internal fun assertGreedyFixture(expected: String, actual: String, tolerance: Double = 1e-6) {
    val e = parseJson(expected) as JsonObject
    val a = parseJson(actual) as JsonObject
    assertEquals(e.fields.keys, a.fields.keys)
    for (key in e.fields.keys) {
        val (ep, eIds, eMargin) = (e[key] as JsonArray).elements
        val (ap, aIds, aMargin) = (a[key] as JsonArray).elements
        assertEquals(ep, ap, "$key: prompt")
        assertEquals(eIds, aIds, "$key: greedy ids")
        val em = (eMargin as JsonNumber).value
        val am = (aMargin as JsonNumber).value
        assertTrue(abs(em - am) <= tolerance, "$key: margin $am, fixture $em")
    }
}
