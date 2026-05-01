package io.tlaloc.maestro

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.258+ — `BackendTarget` JSON round-trip + populator
 * pipeline tests.
 *
 * Pin: structured JSON shape, null handling on optional fields, the
 * populator's per-target tuple (kernel name + kv quant dtype + cost).
 */
class BackendMatrixTest {

    @Test
    fun backendTargetRoundTripsThroughJson() {
        val bt = BackendTarget(
            vendor = "nvidia",
            arch = "h100",
            kernelName = "flash_attn_v3",
            kvQuantDtype = "fp8_e4m3",
            costMicroseconds = 12.5,
        )
        val json = bt.toJson()
        val parsed = BackendTarget.fromJson(json)
        assertEquals(bt, parsed)
    }

    @Test
    fun backendTargetRoundTripsWithNulls() {
        // CPU_GENERIC + decompose path: no kernel, no quant, no cost.
        val bt = BackendTarget(vendor = "tlaloc", arch = "cpu_generic")
        val json = bt.toJson()
        assertTrue("\"kernelName\":null" in json)
        assertTrue("\"kvQuantDtype\":null" in json)
        assertTrue("\"costMicroseconds\":null" in json)
        val parsed = BackendTarget.fromJson(json)
        assertEquals(bt, parsed)
        assertNull(parsed.kernelName)
        assertNull(parsed.kvQuantDtype)
        assertNull(parsed.costMicroseconds)
    }

    @Test
    fun manifestRoundTripsWithStructuredBackendMatrix() {
        val manifest = ProgramManifest(
            name = "encode",
            inputs = listOf(TypeDescriptor("f32", listOf(8))),
            outputs = listOf(TypeDescriptor("f32", listOf(8))),
            meshRequirement = "Mesh0",
            bodyHash = "deadbeef",
            backendMatrix = listOf(
                BackendTarget("nvidia", "h100", "flash_attn_v3", "fp8_e4m3", 5.0),
                BackendTarget("aws", "trainium2", "nki_flash_attention", null, 7.2),
                BackendTarget("tlaloc", "cpu_generic"),
            ),
        )
        val parsed = ProgramManifest.fromJson(manifest.toJson())
        assertEquals(manifest, parsed)
        assertEquals(3, parsed.backendMatrix.size)
        assertEquals("flash_attn_v3", parsed.backendMatrix[0].kernelName)
        assertNull(parsed.backendMatrix[2].kernelName, "CPU entry has no kernel")
    }

    @Test
    fun emptyBackendMatrixStaysBackwardsCompatible() {
        // The default-empty backendMatrix that pre-L3.5 callers produce
        // still serialises + parses identically.
        val manifest = ProgramManifest(
            name = "noop",
            inputs = emptyList(),
            outputs = emptyList(),
            meshRequirement = "Mesh0",
            bodyHash = "0",
        )
        val json = manifest.toJson()
        assertTrue("\"backendMatrix\":[]" in json)
        val parsed = ProgramManifest.fromJson(json)
        assertEquals(emptyList<BackendTarget>(), parsed.backendMatrix)
    }

    // -------------------------------------------------------------------
    // Populator pipeline tests
    // -------------------------------------------------------------------

    private val qType = DxirType(F32, listOf(8, 4))
    private val kType = DxirType(F32, listOf(4, 8))
    private val vType = DxirType(F32, listOf(8, 4))
    private val sType = DxirType(F32, listOf(8, 8))
    private val oType = DxirType(F32, listOf(8, 4))

    private fun rawAttention() = DxirBuilder.function("attn") {
        val q = param("Q", qType)
        val k = param("K", kType)
        val v = param("V", vType)
        val qk = op(OpKind.MATMUL, listOf(q, k), sType)
        val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
        val out = op(OpKind.MATMUL, listOf(sm, v), oType)
        listOf(out)
    }

    @Test
    fun populateEmptyTargetListIsEmpty() {
        assertEquals(emptyList<BackendTarget>(), populateBackendMatrix(rawAttention(), emptyList()))
    }

    @Test
    fun populateAcrossThreeTargetsProducesThreeRows() {
        val matrix = populateBackendMatrix(
            rawAttention(),
            listOf(KernelTarget.NVIDIA_H100, KernelTarget.GOOGLE_TPU_V5E, KernelTarget.CPU_GENERIC),
        )
        assertEquals(3, matrix.size)

        val h100 = matrix[0]
        assertEquals("nvidia", h100.vendor)
        assertEquals("h100", h100.arch)
        assertEquals("flash_attn_v3", h100.kernelName)
        assertNull(h100.kvQuantDtype, "no KV quant requested")
        assertNotNull(h100.costMicroseconds)

        val tpu = matrix[1]
        assertEquals("google", tpu.vendor)
        assertEquals("tpu_pallas_flash_attention", tpu.kernelName)

        val cpu = matrix[2]
        assertEquals("tlaloc", cpu.vendor)
        assertEquals("cpu_generic", cpu.arch)
        // CPU forces decompose → no kernel was annotated.
        assertNull(cpu.kernelName, "CPU forces decompose; no kernel name")
    }

    @Test
    fun populateWithFp8KvQuantAcceptsOnlyOnSupportingTargets() {
        val matrix = populateBackendMatrix(
            rawAttention(),
            listOf(
                KernelTarget.NVIDIA_H100,    // FP8 ✓
                KernelTarget.NVIDIA_A100,    // FP8 ✗
                KernelTarget.AWS_TRAINIUM2,  // FP8 ✓
                KernelTarget.GOOGLE_TPU_V5E, // FP8 ✗ (v5e doesn't have FP8)
            ),
            kvQuant = KvQuantConfig.FP8_PER_HEAD,
        )
        assertEquals("fp8_e4m3", matrix[0].kvQuantDtype, "H100 supports FP8")
        assertNull(matrix[1].kvQuantDtype, "A100 declined FP8")
        assertEquals("fp8_e4m3", matrix[2].kvQuantDtype, "Trainium2 supports FP8")
        assertNull(matrix[3].kvQuantDtype, "TPU v5e declined FP8")
    }

    @Test
    fun populateWithInt8KvQuantAcceptedAcrossLine() {
        // INT8 is universally supported across our kernel matrix.
        val matrix = populateBackendMatrix(
            rawAttention(),
            listOf(
                KernelTarget.NVIDIA_H100,
                KernelTarget.NVIDIA_A100,
                KernelTarget.AMD_MI300X,
                KernelTarget.GOOGLE_TPU_V5E,
                KernelTarget.AWS_TRAINIUM2,
            ),
            kvQuant = KvQuantConfig.INT8_PER_TENSOR,
        )
        for (row in matrix) {
            assertEquals("int8", row.kvQuantDtype, "${row.vendor}/${row.arch} should accept int8")
        }
    }

    @Test
    fun costMicrosecondsOrdersByDevice() {
        val matrix = populateBackendMatrix(
            rawAttention(),
            listOf(KernelTarget.NVIDIA_H100, KernelTarget.NVIDIA_A100, KernelTarget.CPU_GENERIC),
        )
        val tH100 = matrix[0].costMicroseconds!!
        val tA100 = matrix[1].costMicroseconds!!
        val tCpu = matrix[2].costMicroseconds!!
        assertTrue(tH100 < tA100, "H100 ($tH100 us) faster than A100 ($tA100 us)")
        assertTrue(tA100 < tCpu, "A100 ($tA100 us) faster than CPU ($tCpu us)")
    }

    @Test
    fun functionWithoutRecognizedPatternsStillProducesEntries() {
        val plain = DxirBuilder.function("plain") {
            val x = param("x", DxirType(F32, listOf(64)))
            val y = op(OpKind.RELU, listOf(x), DxirType(F32, listOf(64)))
            listOf(y)
        }
        val matrix = populateBackendMatrix(plain, listOf(KernelTarget.NVIDIA_H100))
        assertEquals(1, matrix.size)
        // No COARSENED → no kernel attached → kernelName null.
        assertNull(matrix.single().kernelName)
        assertNotNull(matrix.single().costMicroseconds, "cost still reportable")
    }

    @Test
    fun populatedMatrixSerialisesIntoManifest() {
        val matrix = populateBackendMatrix(
            rawAttention(),
            listOf(KernelTarget.NVIDIA_H100, KernelTarget.AWS_TRAINIUM2),
            kvQuant = KvQuantConfig.FP8_PER_HEAD,
        )
        val manifest = ProgramManifest(
            name = "attn",
            inputs = emptyList(),
            outputs = emptyList(),
            meshRequirement = "Mesh0",
            bodyHash = "0",
            backendMatrix = matrix,
        )
        val parsed = ProgramManifest.fromJson(manifest.toJson())
        assertEquals(matrix, parsed.backendMatrix, "populated matrix survives JSON round-trip")
    }
}
