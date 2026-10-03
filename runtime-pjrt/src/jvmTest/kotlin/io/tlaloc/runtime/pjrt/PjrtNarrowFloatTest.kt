package io.tlaloc.runtime.pjrt

import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * What the backend's XLA does with the narrow float types the quantized
 * checkpoints store: bytes reinterpreted as f8E4M3FN, and bytes reinterpreted
 * as pairs of f4E2M1FN (NVFP4's element type), both widened to f32. Pins the
 * codes' values and the order of the two FP4 values in a byte.
 */
class PjrtNarrowFloatTest {

    private fun run(mlir: String, bytes: IntArray, count: Int): FloatArray = TestBackend.session().use { s ->
        val out = s.executeStablehlo(mlir, listOf(s.bufferFromHostI32(bytes, listOf(bytes.size))))
        out.single().toFloatArray(count)
    }

    @Test
    fun f8E4M3FnBytesWidenToTheirValues() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val bytes = intArrayOf(0x38, 0x40, 0xB8, 0x7E, 0x01, 0x00, 0x3C)
        val n = bytes.size
        val mlir = """
            func.func @main(%a: tensor<${n}xi32>) -> tensor<${n}xf32> {
              %b = stablehlo.convert %a : (tensor<${n}xi32>) -> tensor<${n}xui8>
              %c = stablehlo.bitcast_convert %b : (tensor<${n}xui8>) -> tensor<${n}xf8E4M3FN>
              %d = stablehlo.convert %c : (tensor<${n}xf8E4M3FN>) -> tensor<${n}xf32>
              return %d : tensor<${n}xf32>
            }
        """.trimIndent()
        assertEquals(listOf(1f, 2f, -1f, 448f, 1f / 512f, 0f, 1.5f), run(mlir, bytes, n).toList())
    }

    @Test
    fun aByteHoldsTwoF4E2M1FnValuesLowNibbleFirst() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // 0x21: low nibble 1 (0.5), high nibble 2 (1.0); 0x7F: 0xF (-6), 0x7 (6); 0x95: 5 (3), 9 (-0.5).
        val bytes = intArrayOf(0x21, 0xF7, 0x95)
        val n = bytes.size
        val mlir = """
            func.func @main(%a: tensor<${n}xi32>) -> tensor<${2 * n}xf32> {
              %b = stablehlo.convert %a : (tensor<${n}xi32>) -> tensor<${n}xui8>
              %c = stablehlo.bitcast_convert %b : (tensor<${n}xui8>) -> tensor<${n}x2xf4E2M1FN>
              %d = stablehlo.convert %c : (tensor<${n}x2xf4E2M1FN>) -> tensor<${n}x2xf32>
              %e = stablehlo.reshape %d : (tensor<${n}x2xf32>) -> tensor<${2 * n}xf32>
              return %e : tensor<${2 * n}xf32>
            }
        """.trimIndent()
        assertEquals(listOf(0.5f, 1f, 6f, -6f, 3f, -0.5f), run(mlir, bytes, 2 * n).toList())
    }
}
