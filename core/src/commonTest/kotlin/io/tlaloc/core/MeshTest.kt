package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class MeshTest {

    @Test
    fun meshOfBuildsNamedAxes() {
        val m = MeshSpec.of("data" to 8, "model" to 4)
        assertEquals(listOf(MeshAxis("data", 8), MeshAxis("model", 4)), m.axes)
        assertEquals(32, m.totalSize())
        assertEquals(8, m.size("data"))
        assertEquals(4, m.size("model"))
    }

    @Test
    fun meshAxisLookup() {
        val m = MeshSpec.of("x" to 2, "y" to 3)
        assertEquals(MeshAxis("x", 2), m.axis("x"))
        assertNull(m.axis("missing"))
    }

    @Test
    fun duplicateMeshAxisNamesRejected() {
        assertFailsWith<IllegalArgumentException> {
            MeshSpec("bad", listOf(MeshAxis("x", 2), MeshAxis("x", 4)))
        }
    }

    @Test
    fun meshAxisSizeMustBePositive() {
        assertFailsWith<IllegalArgumentException> { MeshAxis("x", 0) }
        assertFailsWith<IllegalArgumentException> { MeshAxis("x", -1) }
    }

    @Test
    fun onSpecRequiresAtLeastOneAxis() {
        assertFailsWith<IllegalArgumentException> { Spec.On(axes = emptyList()) }
    }

    @Test
    fun specPrettyPrintsReplicatedAndOn() {
        assertEquals("_", Spec.Replicated.toString())
        assertEquals("{data}", Spec.On(listOf("data")).toString())
        assertEquals("{data,model}?#p1", Spec.On(listOf("data", "model"), open = true, priority = 1).toString())
    }

    @Test
    fun validatePassesForDivisibleShape() {
        val mesh = MeshSpec.of("data" to 8, "model" to 4)
        val spec = partitionSpec(Spec.On.of("data"), Spec.On.of("model"))
        mesh.validate(spec, intArrayOf(32, 16))
    }

    @Test
    fun validateRejectsRankMismatch() {
        val mesh = MeshSpec.of("data" to 2)
        val spec = partitionSpec(Spec.On.of("data"), Spec.Replicated)
        assertFailsWith<IllegalArgumentException> { mesh.validate(spec, intArrayOf(8)) }
    }

    @Test
    fun validateRejectsUnknownAxis() {
        val mesh = MeshSpec.of("data" to 2)
        val spec = partitionSpec(Spec.On.of("ghost"))
        assertFailsWith<IllegalStateException> { mesh.validate(spec, intArrayOf(8)) }
    }

    @Test
    fun validateRejectsNonDivisibleDim() {
        val mesh = MeshSpec.of("data" to 4)
        val spec = partitionSpec(Spec.On.of("data"))
        assertFailsWith<IllegalArgumentException> { mesh.validate(spec, intArrayOf(10)) }
    }

    @Test
    fun validateCombinedAxesMultiply() {
        val mesh = MeshSpec.of("x" to 2, "y" to 3)
        val spec = partitionSpec(Spec.On.of("x", "y"))
        mesh.validate(spec, intArrayOf(12))
        assertFailsWith<IllegalArgumentException> { mesh.validate(spec, intArrayOf(8)) }
    }
}
