package io.tlaloc.maestro

import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.DTensor
import io.tlaloc.core.DataAxis
import io.tlaloc.core.F32
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Mesh1
import io.tlaloc.core.ModelAxis
import io.tlaloc.core.Rank1
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.Tlaloc
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — workflow composition + reshard insertion tests.
 *
 * v1 contract under test:
 * - A two-step workflow types correctly: stepB consumes stepA's output
 *   handle directly; Kotlin's checker enforces the type match at compose
 *   time (compile-time errors live in the compile-fail tests).
 * - Cross-mesh transitions produce a [WorkflowEdge] with [ReshardKind.Mesh].
 * - Named-axis-mismatched transitions produce [ReshardKind.Transpose].
 * - Same-mesh same-axis transitions produce [ReshardKind.None].
 * - Workflow execution (via the recorded shims) produces correct results.
 */
class WorkflowTest {

    @Test
    fun twoStepLinearWorkflowExecutesEndToEnd() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f))
        val reluStep = Tlaloc.program("relu_step", initial, Mesh0) { x -> x.relu() }
        val sumStep = Tlaloc.program("sum_step", initial, Mesh0) { x -> (x * x).sum() }

        val wf = Tlaloc.workflow("two_step") {
            val initialHandle = seed(initial, Mesh0)
            val activated = step(reluStep, initialHandle)
            val loss = step(sumStep, activated)
            // Materialize loss for assertion outside the workflow body.
            @Suppress("UNCHECKED_CAST")
            val tensorOut = loss.ref.payload as DTensor<ScalarShape, F32>
            assertTrue(
                abs(tensorOut.hostF32().single() - 10f) < 1e-5f,
                "expected 10.0 (sum of squared relu), got ${tensorOut.hostF32().single()}",
            )
        }

        assertEquals(2, wf.steps.size)
        assertEquals(1, wf.edges.size)
        assertEquals(ReshardKind.None, wf.edges.single().reshardKind)
        assertEquals("relu_step", wf.edges.single().fromStep)
        assertEquals("sum_step", wf.edges.single().toStep)
    }

    @Test
    fun crossMeshTransitionProducesMeshReshardEdge() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f))
        val onMesh0 = Tlaloc.program("on_mesh0", initial, Mesh0) { x -> x.relu() }
        val onMesh1 = Tlaloc.program("on_mesh1", initial, Mesh1<DataAxis>()) { x -> (x * x).sum() }

        val wf = Tlaloc.workflow("cross_mesh") {
            val seedH = seed(initial, Mesh0)
            val activated = step(onMesh0, seedH)
            // The next step expects Mesh1; the prior produced on Mesh0.
            // v1 type system tolerates this at the BufferHandle<*, *>
            // erasure boundary (the workflow builder records the metadata
            // mismatch instead of failing the type checker). Layer 4
            // tightens this with type-level mesh refinement.
            @Suppress("UNCHECKED_CAST")
            val activatedAsMesh1 = activated as io.tlaloc.core.BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh1<DataAxis>>
            step(onMesh1, activatedAsMesh1)
        }

        assertEquals(2, wf.steps.size)
        assertEquals(1, wf.edges.size)
        val edge = wf.edges.single()
        assertEquals(ReshardKind.Mesh, edge.reshardKind, "cross-mesh transition must record a Mesh reshard")
        assertEquals("Mesh0", edge.fromMesh)
        assertEquals("Mesh1", edge.toMesh)
    }

    @Test
    fun samemeshTransitionWithMatchingAxesIsPassThrough() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val a = Tlaloc.program("a", initial, Mesh1<DataAxis>()) { x -> x.relu() }
        val b = Tlaloc.program("b", initial, Mesh1<DataAxis>()) { x -> x.sum() }

        val wf = Tlaloc.workflow("same_mesh") {
            val s = seed(initial, Mesh1<DataAxis>())
            step(b, step(a, s))
        }
        assertEquals(1, wf.edges.size)
        assertTrue(wf.edges.single().isPassThrough, "same-mesh same-axes must be pass-through")
    }

    @Test
    fun threeStepLinearWorkflowProducesTwoEdges() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, 1f, 1f))
        val a = Tlaloc.program("a", initial, Mesh0) { x -> x.relu() }
        val b = Tlaloc.program("b", initial, Mesh0) { x -> x * x }
        val c = Tlaloc.program("c", initial, Mesh0) { x -> x.sum() }

        val wf = Tlaloc.workflow("three_step") {
            step(c, step(b, step(a, seed(initial, Mesh0))))
        }
        assertEquals(3, wf.steps.size)
        assertEquals(2, wf.edges.size)
        assertEquals(listOf("a" to "b", "b" to "c"), wf.edges.map { it.fromStep to it.toStep })
        assertTrue(wf.edges.all { it.isPassThrough })
    }

    @Test
    fun seedProducesLiveBufferHandle() {
        val v = Tensors.f32Vector<Sym>(floatArrayOf(7f))
        val wf = Tlaloc.workflow("seed_only") {
            val h = seed(v, Mesh0)
            assertTrue(h.isLive)
            assertEquals(1, h.ref.refCount)
        }
        assertEquals(0, wf.steps.size)
        assertEquals(0, wf.edges.size)
    }
}
