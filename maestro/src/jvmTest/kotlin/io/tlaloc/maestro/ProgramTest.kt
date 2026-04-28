package io.tlaloc.maestro

import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.DataAxis
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Mesh1
import io.tlaloc.core.Rank1
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
 * Layer 2 §0.4.243+ — `program { }` builder + content-addressed hash +
 * manifest round-trip tests.
 *
 * v1 contract under test:
 * - The builder produces a [MaestroStep] with a non-empty body and a
 *   correctly-populated manifest (input + output type descriptors,
 *   mesh requirement, SHA-256 hash).
 * - Two semantically identical programs produce identical [bodyHash]es
 *   (content-addressing).
 * - The shim's BufferHandle round-trip returns the right tensor value.
 * - Manifest JSON round-trips through `toJson` / `fromJson`.
 */
class ProgramTest {

    @Test
    fun programBuildsArtifactWithManifestAndBody() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f))
        val step = Tlaloc.program(
            name = "relu_sum",
            input = input,
            mesh = Mesh0,
        ) { x ->
            (x.relu() * x.relu()).sum()
        }

        assertEquals("relu_sum", step.name)
        assertEquals("relu_sum", step.manifest.name)
        assertTrue(step.body.isNotEmpty(), "StableHLO body bytes must be non-empty")
        assertEquals(64, step.manifest.bodyHash.length, "SHA-256 hex must be 64 chars")
        assertEquals(1, step.manifest.inputs.size)
        assertEquals(1, step.manifest.outputs.size)
        assertEquals("f32", step.manifest.inputs.single().dtype)
        assertEquals(listOf(4), step.manifest.inputs.single().dims)
        // Output is scalar (sum reduces to ScalarShape).
        assertEquals(emptyList<Int>(), step.manifest.outputs.single().dims)
    }

    @Test
    fun shimReproducesForwardComputationViaBufferHandle() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f, -4f))
        val step = Tlaloc.program("relu_sum", input, Mesh0) { x ->
            (x.relu() * x.relu()).sum()
        }
        // Wrap input in a BufferHandle, invoke step, materialize output.
        val handleIn = input.handleOn(Mesh0)
        val handleOut = step(handleIn)
        // relu([1,-2,3,-4]) = [1,0,3,0]; squared = [1,0,9,0]; sum = 10
        @Suppress("UNCHECKED_CAST")
        val outTensor = handleOut.ref.payload as DTensor<ScalarShape, F32>
        assertTrue(
            abs(outTensor.hostF32().single() - 10f) < 1e-5f,
            "expected 10.0, got ${outTensor.hostF32().single()}",
        )
    }

    @Test
    fun semanticallyIdenticalProgramsHaveIdenticalBodyHashes() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f))
        val step1 = Tlaloc.program("sum_sq", input, Mesh0) { x -> (x * x).sum() }
        val step2 = Tlaloc.program("sum_sq", input, Mesh0) { x -> (x * x).sum() }
        assertEquals(
            step1.manifest.bodyHash, step2.manifest.bodyHash,
            "two semantically identical programs must have identical bodyHash",
        )
    }

    @Test
    fun differentBodiesProduceDifferentHashes() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f, 4f))
        val sumSq = Tlaloc.program("sum_sq", input, Mesh0) { x -> (x * x).sum() }
        val justSum = Tlaloc.program("just_sum", input, Mesh0) { x -> x.sum() }
        assertNotEquals(sumSq.manifest.bodyHash, justSum.manifest.bodyHash)
    }

    @Test
    fun manifestJsonRoundTripsExactly() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val step = Tlaloc.program("identity", input, Mesh0) { x -> x }
        val json = step.manifest.toJson()
        val parsed = ProgramManifest.fromJson(json)
        assertEquals(step.manifest, parsed, "manifest must round-trip exactly through JSON")
    }

    @Test
    fun manifestJsonContainsExpectedKeys() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val step = Tlaloc.program("identity", input, Mesh0) { x -> x }
        val json = step.manifest.toJson()
        for (key in listOf("name", "inputs", "outputs", "meshRequirement", "bodyHash", "shardingSpec", "backendMatrix")) {
            assertTrue(
                "\"$key\":" in json,
                "manifest JSON must include key \"$key\":; got:\n$json",
            )
        }
    }

    @Test
    fun stepWithMesh1CarriesMeshRequirement() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val step = Tlaloc.program(
            name = "data_parallel_step",
            input = input,
            mesh = Mesh1<DataAxis>(),
        ) { x -> x.sum() }
        assertEquals("Mesh1", step.manifest.meshRequirement)
    }

    @Test
    fun manifestPreservesAxisNames() {
        val input = Tensors.f32Vector<Sym>(floatArrayOf(1f))
        val step = Tlaloc.program("noop", input, Mesh0) { x -> x }
        assertEquals(emptyList<String?>(), step.manifest.inputs.single().axisNames)
        assertEquals(emptyList<String?>(), step.manifest.outputs.single().axisNames)
    }

    @Test
    fun handleOnWrapsTensorAsBufferHandle() {
        val tensor = Tensors.f32Vector<Sym>(floatArrayOf(7f, 8f))
        val handle = tensor.handleOn(Mesh0)
        assertTrue(handle.isLive)
        assertSame(tensor, handle.ref.payload)
    }

    private fun assertSame(expected: Any?, actual: Any?) {
        assertEquals(expected, actual, "expected same instance")
    }
}
