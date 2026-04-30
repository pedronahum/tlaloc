@file:Suppress("DEPRECATION")
// MaestroDescriptor and StubExecutor are deprecated as of §0.4.249 (Layer 2.5.5).
// This test class keeps coverage on the deprecated surface during the migration
// window — silencing the warnings is intentional and scoped to this file only.

package io.tlaloc.maestro

import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.DTensor
import io.tlaloc.core.DataAxis
import io.tlaloc.core.F32
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Mesh1
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.Tlaloc
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — Maestro descriptor + stub executor end-to-end tests.
 *
 * v1 contract under test:
 * - Emitted descriptor has the canonical Maestro JSON shape
 *   (`{properties, workflow: {id, name, steps: [{step: {...}}]}}`).
 * - Each step is shaped as a Kubernetes-type Maestro step with our three
 *   Tlaloc-specific params (`image`, `tlaloc_artifact_uri`, `tlaloc_manifest`).
 * - Cross-mesh edges produce inline reshard steps with `tlaloc-reshard:*` image.
 * - Stub executor runs a Workflow end-to-end, returning the final output.
 */
class MaestroDescriptorTest {

    @Test
    fun emittedDescriptorHasCanonicalMaestroShape() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f))
        val s1 = Tlaloc.program("encode", initial, Mesh0) { x -> x.relu() }
        val s2 = Tlaloc.program("loss", initial, Mesh0) { x -> (x * x).sum() }
        val wf = Tlaloc.workflow("test_wf") {
            step(s2, step(s1, seed(initial, Mesh0)))
        }

        val json = MaestroDescriptor.emit(wf)

        // Top-level Maestro shape
        assertTrue("\"properties\":" in json, "must include properties")
        assertTrue("\"workflow\":" in json, "must include workflow")
        assertTrue("\"id\":\"tlaloc_test_wf\"" in json, "must include workflow id")
        assertTrue("\"name\":\"test_wf\"" in json, "must include workflow name")

        // Each step inside `steps[]` must be wrapped in `{"step": {...}}`.
        assertTrue("\"steps\":[{\"step\":" in json, "steps array must wrap each step")

        // Every step must be type "Kubernetes" (Maestro's only public-extension surface)
        assertEquals(2, "\"type\":\"Kubernetes\"".toRegex().findAll(json).count())

        // Tlaloc-specific params must appear on each step.
        for (key in listOf("image", "tlaloc_artifact_uri", "tlaloc_manifest")) {
            assertTrue(
                "\"$key\":{\"value\":" in json,
                "step params must include $key; got:\n$json",
            )
        }

        // Transition shape: the second step's transition is an empty {} (terminal).
        assertTrue("\"transition\":{}" in json, "terminal step's transition must be empty")
        // The first step's transition points to the next.
        assertTrue("\"successors\":{\"loss\":\"true\"}" in json)
    }

    @Test
    fun crossMeshEdgeProducesInlineReshardStep() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val a = Tlaloc.program("a", initial, Mesh0) { x -> x.relu() }
        val b = Tlaloc.program("b", initial, Mesh1<DataAxis>()) { x -> x.sum() }
        val wf = Tlaloc.workflow("xmesh") {
            val seed = seed(initial, Mesh0)
            val activated = step(a, seed)
            @Suppress("UNCHECKED_CAST")
            val asMesh1 = activated as io.tlaloc.core.BufferHandle<DTensor<io.tlaloc.core.Rank1<Sym>, F32>, Mesh1<DataAxis>>
            step(b, asMesh1)
        }
        val json = MaestroDescriptor.emit(wf)
        assertTrue(
            "\"id\":\"reshard_a_to_b\"" in json,
            "cross-mesh edge must produce a reshard step; got:\n$json",
        )
        assertTrue("tlaloc-reshard" in json, "reshard step must use the tlaloc-reshard image")
        assertTrue("\"reshard_kind\":{\"value\":\"Mesh\"" in json)
        assertTrue("\"from_mesh\":{\"value\":\"Mesh0\"" in json)
        assertTrue("\"to_mesh\":{\"value\":\"Mesh1\"" in json)
        // Sanity: a -> reshard -> b chain
        assertTrue("\"successors\":{\"reshard_a_to_b\":\"true\"}" in json)
        assertTrue("\"successors\":{\"b\":\"true\"}" in json)
    }

    @Test
    fun stubExecutorRunsLinearWorkflowEndToEnd() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f))
        val activate = Tlaloc.program("activate", initial, Mesh0) { x -> x.relu() }
        val score = Tlaloc.program("score", initial, Mesh0) { x -> (x * x).sum() }
        val wf = Tlaloc.workflow("linear") {
            step(score, step(activate, seed(initial, Mesh0)))
        }

        val seedHandle = io.tlaloc.core.BufferHandle<DTensor<io.tlaloc.core.Rank1<Sym>, F32>, Mesh0>(
            io.tlaloc.core.HandleRef(nativeId = 99L, payload = initial),
        )
        val out: io.tlaloc.core.BufferHandle<DTensor<ScalarShape, F32>, Mesh0> = StubExecutor().run(wf, seedHandle)
        @Suppress("UNCHECKED_CAST")
        val tensorOut = out.ref.payload as DTensor<ScalarShape, F32>
        assertTrue(
            abs(tensorOut.hostF32().single() - 10f) < 1e-5f,
            "stub executor must produce 10.0 (sum of squared relu); got ${tensorOut.hostF32().single()}",
        )
    }

    @Test
    fun differentWorkflowsProduceDifferentDescriptors() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f))
        val a = Tlaloc.program("a", initial, Mesh0) { x -> x.relu() }
        val b = Tlaloc.program("b", initial, Mesh0) { x -> x * x }
        val wf1 = Tlaloc.workflow("one_step") { step(a, seed(initial, Mesh0)) }
        val wf2 = Tlaloc.workflow("two_step") { step(b, step(a, seed(initial, Mesh0))) }
        assertNotEquals(MaestroDescriptor.emit(wf1), MaestroDescriptor.emit(wf2))
    }

    @Test
    fun customRuntimeImageIsHonored() {
        val initial = Tensors.f32Vector<Sym>(floatArrayOf(1f))
        val a = Tlaloc.program("a", initial, Mesh0) { x -> x.relu() }
        val wf = Tlaloc.workflow("simple") { step(a, seed(initial, Mesh0)) }
        val json = MaestroDescriptor.emit(wf, runtimeImage = "tlaloc-runtime:0.99-test")
        assertTrue("\"value\":\"tlaloc-runtime:0.99-test\"" in json)
    }
}
