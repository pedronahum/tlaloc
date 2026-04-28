package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — type-level smoke tests for the phantom-typed
 * [Mesh] family. Mirrors `NamedTest` from Layer 1.
 */

private object UserMeshAxis : MeshDim {
    override val name = "user-axis"
}

class MeshTypesTest {

    @Test
    fun commonMeshDimsCarryExpectedNames() {
        assertEquals("data", DataAxis.name)
        assertEquals("model", ModelAxis.name)
        assertEquals("pipeline", PipelineAxis.name)
        assertEquals("tensor", TensorAxis.name)
        assertEquals("expert", ExpertAxis.name)
        assertEquals("replica", ReplicaAxis.name)
    }

    @Test
    fun meshDimSingletonsAreObjects() {
        assertSame(DataAxis, DataAxis)
        assertSame(ModelAxis, ModelAxis)
    }

    @Test
    fun userDefinedMeshDimIsAllowed() {
        // Non-sealed: user code can declare additional axes.
        assertEquals("user-axis", UserMeshAxis.name)
        assertTrue(UserMeshAxis is MeshDim)
    }

    @Test
    fun phantomMeshTypesCompose() {
        val m0 = Mesh0
        val m1 = Mesh1<DataAxis>()
        val m2 = Mesh2<DataAxis, ModelAxis>()
        val m3 = Mesh3<DataAxis, ModelAxis, PipelineAxis>()
        val m4 = Mesh4<DataAxis, ModelAxis, PipelineAxis, TensorAxis>()
        assertTrue(m0 is Mesh)
        assertTrue(m1 is Mesh)
        assertTrue(m2 is Mesh)
        assertTrue(m3 is Mesh)
        assertTrue(m4 is Mesh)
    }

    @Test
    fun mesh0IsADataObject() {
        // Mesh0 is a data object — single-instance, deterministic equality.
        assertSame(Mesh0, Mesh0)
        assertEquals(Mesh0, Mesh0)
    }
}
