package io.tlaloc.ir

import io.tlaloc.core.F32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Verifier + pretty-print tests for [DxirType.axisNames] (Layer 1 §0.4.241+).
 *
 * Backwards-compat is the load-bearing invariant: existing two-arg
 * `DxirType(F32, listOf(...))` constructions must continue to behave
 * identically. Named-axis behaviour is purely additive.
 */
class DxirTypeNamedAxesTest {

    @Test
    fun unnamedTypeStillFormatsAsBefore() {
        assertEquals("f32", DxirType(F32, emptyList()).toString())
        assertEquals("f32[4]", DxirType(F32, listOf(4)).toString())
        assertEquals("f32[2,3,4]", DxirType(F32, listOf(2, 3, 4)).toString())
    }

    @Test
    fun emptyAxisNamesIsBackwardCompatible() {
        // Explicitly passing an empty axisNames list must equal the old
        // two-arg construction — including data-class equality / hashCode.
        val old = DxirType(F32, listOf(2, 3))
        val new = DxirType(F32, listOf(2, 3), emptyList())
        assertEquals(old, new)
        assertEquals(old.hashCode(), new.hashCode())
        assertFalse(old.hasNamedAxes)
        assertFalse(new.hasNamedAxes)
    }

    @Test
    fun fullyNamedTypeRendersNamesInPrettyPrint() {
        val t = DxirType(F32, listOf(8, 64), listOf("batch", "seq"))
        assertEquals("f32[8@batch,64@seq]", t.toString())
        assertTrue(t.hasNamedAxes)
        assertEquals("batch", t.axisNameOrNull(0))
        assertEquals("seq", t.axisNameOrNull(1))
    }

    @Test
    fun mixedNamedAndUnnamedAxesRender() {
        // Half-named types are valid — entries may be null to mark a single
        // axis as unnamed within an otherwise-named type.
        val t = DxirType(F32, listOf(8, 16, 32), listOf("batch", null, "dim"))
        assertEquals("f32[8@batch,16,32@dim]", t.toString())
        assertTrue(t.hasNamedAxes)
        assertEquals("batch", t.axisNameOrNull(0))
        assertNull(t.axisNameOrNull(1))
        assertEquals("dim", t.axisNameOrNull(2))
    }

    @Test
    fun verifierRejectsAxisNamesShorterThanDims() {
        assertFails {
            DxirType(F32, listOf(8, 16, 32), listOf("batch", "seq"))
        }
    }

    @Test
    fun verifierRejectsAxisNamesLongerThanDims() {
        assertFails {
            DxirType(F32, listOf(8, 16), listOf("a", "b", "c"))
        }
    }

    @Test
    fun verifierRejectsAxisNamesOnScalar() {
        // Scalar (rank 0) can't carry axis names.
        assertFails {
            DxirType(F32, emptyList(), listOf("anything"))
        }
    }

    @Test
    fun namedAndUnnamedTypesAreDistinct() {
        // Data-class equality must distinguish a named type from its unnamed
        // counterpart, so the SDY emitter doesn't accidentally treat them as
        // interchangeable in a Map<DxirType, _> lookup.
        val unnamed = DxirType(F32, listOf(8, 64))
        val named = DxirType(F32, listOf(8, 64), listOf("batch", "seq"))
        assertTrue(unnamed != named)
    }

    @Test
    fun axisNameOrNullOutOfRangeReturnsNull() {
        val t = DxirType(F32, listOf(8), listOf("batch"))
        assertNull(t.axisNameOrNull(-1))
        assertNull(t.axisNameOrNull(5))
    }

    @Test
    fun copyPreservesAxisNames() {
        // The data-class `copy` should carry axisNames through verbatim — this
        // matters because lots of IR code rebuilds types via `.copy(dims = ...)`.
        val t = DxirType(F32, listOf(8, 64), listOf("batch", "seq"))
        val same = t.copy(dtype = F32)
        assertEquals(t, same)
        assertEquals(listOf("batch", "seq"), same.axisNames)
    }
}
