package io.tlaloc.runtime.pjrt

import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * Whether a weight dequantized in the graph is read narrow: a 27B-sized MLP
 * projection ([17408, 5120]) times 1 and 4 rows, with the weight in bf16, in
 * FP8 (f8e4m3 bytes and one scale) and in NVFP4 (packed f4e2m1 pairs, an f8e4m3
 * scale per 16 and a global scale), each dequantized to bf16 in front of the
 * dot. Prints milliseconds per call and the bandwidth that time implies for
 * the weight's stored bytes. Runs with TLALOC_QUANT_BENCH=1.
 */
class PjrtQuantGemvBenchTest {

    private val n = 17408
    private val h = 5120

    private fun dot(rows: Int, w: String) =
        "    %y = stablehlo.dot_general %x, $w, contracting_dims = [1] x [1] : (tensor<${rows}x${h}xbf16>, tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32>\n" +
            "    return %y : tensor<${rows}x${n}xf32>\n  }"

    private fun bf16(rows: Int) = "func.func @main(%x: tensor<${rows}x${h}xbf16>, %w: tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32> {\n" + dot(rows, "%w")

    private fun fp8(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h}xui8>, %s: tensor<f32>) -> tensor<${rows}x${n}xf32> {
    %f8 = stablehlo.bitcast_convert %c : (tensor<${n}x${h}xui8>) -> tensor<${n}x${h}xf8E4M3FN>
    %f = stablehlo.convert %f8 : (tensor<${n}x${h}xf8E4M3FN>) -> tensor<${n}x${h}xf32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<${n}x${h}xf32>
    %m = stablehlo.multiply %f, %sb : tensor<${n}x${h}xf32>
    %w = stablehlo.convert %m : (tensor<${n}x${h}xf32>) -> tensor<${n}x${h}xbf16>
""".trimStart() + dot(rows, "%w")

    private fun nvfp4(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h / 2}xui8>, %s: tensor<${n}x${h / 16}xui8>, %g: tensor<f32>) -> tensor<${rows}x${n}xf32> {
    %f4 = stablehlo.bitcast_convert %c : (tensor<${n}x${h / 2}xui8>) -> tensor<${n}x${h / 2}x2xf4E2M1FN>
    %f4f = stablehlo.convert %f4 : (tensor<${n}x${h / 2}x2xf4E2M1FN>) -> tensor<${n}x${h / 2}x2xf32>
    %v = stablehlo.reshape %f4f : (tensor<${n}x${h / 2}x2xf32>) -> tensor<${n}x${h}xf32>
    %s8 = stablehlo.bitcast_convert %s : (tensor<${n}x${h / 16}xui8>) -> tensor<${n}x${h / 16}xf8E4M3FN>
    %sf = stablehlo.convert %s8 : (tensor<${n}x${h / 16}xf8E4M3FN>) -> tensor<${n}x${h / 16}xf32>
    %gb = stablehlo.broadcast_in_dim %g, dims = [] : (tensor<f32>) -> tensor<${n}x${h / 16}xf32>
    %sg = stablehlo.multiply %sf, %gb : tensor<${n}x${h / 16}xf32>
    %sb = stablehlo.broadcast_in_dim %sg, dims = [0, 1] : (tensor<${n}x${h / 16}xf32>) -> tensor<${n}x${h / 16}x16xf32>
    %sr = stablehlo.reshape %sb : (tensor<${n}x${h / 16}x16xf32>) -> tensor<${n}x${h}xf32>
    %m = stablehlo.multiply %v, %sr : tensor<${n}x${h}xf32>
    %w = stablehlo.convert %m : (tensor<${n}x${h}xf32>) -> tensor<${n}x${h}xbf16>
""".trimStart() + dot(rows, "%w")

    /** The scale after the dot: the weight path is one widening. */
    private fun fp8After(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h}xui8>, %s: tensor<f32>) -> tensor<${rows}x${n}xf32> {
    %f8 = stablehlo.bitcast_convert %c : (tensor<${n}x${h}xui8>) -> tensor<${n}x${h}xf8E4M3FN>
    %w = stablehlo.convert %f8 : (tensor<${n}x${h}xf8E4M3FN>) -> tensor<${n}x${h}xbf16>
    %d = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1] : (tensor<${rows}x${h}xbf16>, tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<${rows}x${n}xf32>
    %y = stablehlo.multiply %d, %sb : tensor<${rows}x${n}xf32>
    return %y : tensor<${rows}x${n}xf32>
  }"""

    /** NVFP4 with the group scales applied in bf16 in front of the dot. */
    private fun nvfp4Bf16(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h / 2}xui8>, %s: tensor<${n}x${h / 16}xui8>, %g: tensor<f32>) -> tensor<${rows}x${n}xf32> {
    %f4 = stablehlo.bitcast_convert %c : (tensor<${n}x${h / 2}xui8>) -> tensor<${n}x${h / 2}x2xf4E2M1FN>
    %f4b = stablehlo.convert %f4 : (tensor<${n}x${h / 2}x2xf4E2M1FN>) -> tensor<${n}x${h / 2}x2xbf16>
    %v = stablehlo.reshape %f4b : (tensor<${n}x${h / 2}x2xbf16>) -> tensor<${n}x${h}xbf16>
    %s8 = stablehlo.bitcast_convert %s : (tensor<${n}x${h / 16}xui8>) -> tensor<${n}x${h / 16}xf8E4M3FN>
    %sf = stablehlo.convert %s8 : (tensor<${n}x${h / 16}xf8E4M3FN>) -> tensor<${n}x${h / 16}xbf16>
    %sb = stablehlo.broadcast_in_dim %sf, dims = [0, 1] : (tensor<${n}x${h / 16}xbf16>) -> tensor<${n}x${h / 16}x16xbf16>
    %sr = stablehlo.reshape %sb : (tensor<${n}x${h / 16}x16xbf16>) -> tensor<${n}x${h}xbf16>
    %w = stablehlo.multiply %v, %sr : tensor<${n}x${h}xbf16>
    %d = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1] : (tensor<${rows}x${h}xbf16>, tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32>
    %gb = stablehlo.broadcast_in_dim %g, dims = [] : (tensor<f32>) -> tensor<${rows}x${n}xf32>
    %y = stablehlo.multiply %d, %gb : tensor<${rows}x${n}xf32>
    return %y : tensor<${rows}x${n}xf32>
  }"""

    /** int8 codes with a scale per output channel after the dot (NVFP4 requantized). */
    private fun int8Channel(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h}xui8>, %s: tensor<${n}xf32>) -> tensor<${rows}x${n}xf32> {
    %i8 = stablehlo.bitcast_convert %c : (tensor<${n}x${h}xui8>) -> tensor<${n}x${h}xi8>
    %w = stablehlo.convert %i8 : (tensor<${n}x${h}xi8>) -> tensor<${n}x${h}xbf16>
    %d = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1] : (tensor<${rows}x${h}xbf16>, tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [1] : (tensor<${n}xf32>) -> tensor<${rows}x${n}xf32>
    %y = stablehlo.multiply %d, %sb : tensor<${rows}x${n}xf32>
    return %y : tensor<${rows}x${n}xf32>
  }"""

    /** int4 codes (two per byte) with bf16 group scales per 16 in front of the dot. */
    private fun int4Group(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h / 2}xui8>, %s: tensor<${n}x${h / 16}xbf16>) -> tensor<${rows}x${n}xf32> {
    %i4 = stablehlo.bitcast_convert %c : (tensor<${n}x${h / 2}xui8>) -> tensor<${n}x${h / 2}x2xi4>
    %ib = stablehlo.convert %i4 : (tensor<${n}x${h / 2}x2xi4>) -> tensor<${n}x${h / 2}x2xbf16>
    %v = stablehlo.reshape %ib : (tensor<${n}x${h / 2}x2xbf16>) -> tensor<${n}x${h}xbf16>
    %sb = stablehlo.broadcast_in_dim %s, dims = [0, 1] : (tensor<${n}x${h / 16}xbf16>) -> tensor<${n}x${h / 16}x16xbf16>
    %sr = stablehlo.reshape %sb : (tensor<${n}x${h / 16}x16xbf16>) -> tensor<${n}x${h}xbf16>
    %w = stablehlo.multiply %v, %sr : tensor<${n}x${h}xbf16>
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1] : (tensor<${rows}x${h}xbf16>, tensor<${n}x${h}xbf16>) -> tensor<${rows}x${n}xf32>
    return %y : tensor<${rows}x${n}xf32>
  }"""

    /**
     * NVFP4 with the activations quantized too (W4A4): each run of 16 input
     * values gets an e4m3 scale amax/6 and is rounded to e2m1, and XLA's
     * block-scaled dot multiplies the two. The weight bytes are bitcast to
     * e2m1 pairs, as staged.
     */
    private fun nvfp4BlockScaled(rows: Int) = """
func.func @main(%x: tensor<${rows}x${h}xbf16>, %c: tensor<${n}x${h / 2}xui8>, %s: tensor<${n}x${h / 16}xui8>, %g: tensor<f32>) -> tensor<${rows}x${n}xf32> {
    %f4 = stablehlo.bitcast_convert %c : (tensor<${n}x${h / 2}xui8>) -> tensor<${n}x${h / 2}x2xf4E2M1FN>
    %wq = stablehlo.reshape %f4 : (tensor<${n}x${h / 2}x2xf4E2M1FN>) -> tensor<${n}x${h}xf4E2M1FN>
    %ws = stablehlo.bitcast_convert %s : (tensor<${n}x${h / 16}xui8>) -> tensor<${n}x${h / 16}xf8E4M3FN>
    %xf = stablehlo.convert %x : (tensor<${rows}x${h}xbf16>) -> tensor<${rows}x${h}xf32>
    %xg = stablehlo.reshape %xf : (tensor<${rows}x${h}xf32>) -> tensor<${rows}x${h / 16}x16xf32>
    %ab = stablehlo.abs %xg : tensor<${rows}x${h / 16}x16xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %am = stablehlo.reduce(%ab init: %z) applies stablehlo.maximum across dimensions = [2] : (tensor<${rows}x${h / 16}x16xf32>, tensor<f32>) -> tensor<${rows}x${h / 16}xf32>
    %six = stablehlo.constant dense<0.16666667> : tensor<${rows}x${h / 16}xf32>
    %tiny = stablehlo.constant dense<1.0e-30> : tensor<${rows}x${h / 16}xf32>
    %sc0 = stablehlo.multiply %am, %six : tensor<${rows}x${h / 16}xf32>
    %sc1 = stablehlo.maximum %sc0, %tiny : tensor<${rows}x${h / 16}xf32>
    %xs = stablehlo.convert %sc1 : (tensor<${rows}x${h / 16}xf32>) -> tensor<${rows}x${h / 16}xf8E4M3FN>
    %xsf = stablehlo.convert %xs : (tensor<${rows}x${h / 16}xf8E4M3FN>) -> tensor<${rows}x${h / 16}xf32>
    %xsb = stablehlo.broadcast_in_dim %xsf, dims = [0, 1] : (tensor<${rows}x${h / 16}xf32>) -> tensor<${rows}x${h / 16}x16xf32>
    %xd = stablehlo.divide %xg, %xsb : tensor<${rows}x${h / 16}x16xf32>
    %xr = stablehlo.reshape %xd : (tensor<${rows}x${h / 16}x16xf32>) -> tensor<${rows}x${h}xf32>
    %xq = stablehlo.convert %xr : (tensor<${rows}x${h}xf32>) -> tensor<${rows}x${h}xf4E2M1FN>
    %d = stablehlo.custom_call @__op${'$'}block_scaled_dot(%xq, %wq, %xs, %ws) : (tensor<${rows}x${h}xf4E2M1FN>, tensor<${n}x${h}xf4E2M1FN>, tensor<${rows}x${h / 16}xf8E4M3FN>, tensor<${n}x${h / 16}xf8E4M3FN>) -> tensor<${rows}x${n}xf32>
    %gb = stablehlo.broadcast_in_dim %g, dims = [] : (tensor<f32>) -> tensor<${rows}x${n}xf32>
    %y = stablehlo.multiply %d, %gb : tensor<${rows}x${n}xf32>
    return %y : tensor<${rows}x${n}xf32>
  }"""

    @Test
    fun dequantizedWeightsAgainstBf16() {
        assumeTrue(System.getenv("TLALOC_QUANT_BENCH") == "1", "set TLALOC_QUANT_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rnd = Random(3)
        TestBackend.session().use { s ->
            for (rows in listOf(1, 4)) {
                val x = s.bufferFromHostBf16(ShortArray(rows * h) { (0x3c00 + rnd.nextInt(256)).toShort() }, listOf(rows, h))
                // Codes that are finite in every format (no NaN patterns).
                fun bytes(count: Int) = ByteArray(count) { (rnd.nextInt(0x70)).toByte() }
                val cases = listOf(
                    Triple("bf16", bf16(rows), listOf(s.bufferFromHostBf16(ShortArray(n * h) { 0x3c00 }, listOf(n, h)))),
                    Triple("fp8", fp8(rows), listOf(s.bufferFromHostU8(bytes(n * h), listOf(n, h)), s.bufferFromHostF32(floatArrayOf(0.01f), emptyList()))),
                    Triple("nvfp4", nvfp4(rows), listOf(
                        s.bufferFromHostU8(bytes(n * h / 2), listOf(n, h / 2)), s.bufferFromHostU8(bytes(n * h / 16), listOf(n, h / 16)),
                        s.bufferFromHostF32(floatArrayOf(0.01f), emptyList()),
                    )),
                )
                    .plus(listOf(
                        Triple("fp8, scale after", fp8After(rows), listOf(s.bufferFromHostU8(bytes(n * h), listOf(n, h)), s.bufferFromHostF32(floatArrayOf(0.01f), emptyList()))),
                        Triple("nvfp4, bf16 scales", nvfp4Bf16(rows), listOf(
                            s.bufferFromHostU8(bytes(n * h / 2), listOf(n, h / 2)), s.bufferFromHostU8(bytes(n * h / 16), listOf(n, h / 16)),
                            s.bufferFromHostF32(floatArrayOf(0.01f), emptyList()),
                        )),
                        Triple("int8, channel scale after", int8Channel(rows), listOf(s.bufferFromHostU8(bytes(n * h), listOf(n, h)), s.bufferFromHostF32(FloatArray(n) { 0.01f }, listOf(n)))),
                        Triple("nvfp4 W4A4, block-scaled dot", nvfp4BlockScaled(rows), listOf(
                            s.bufferFromHostU8(bytes(n * h / 2), listOf(n, h / 2)), s.bufferFromHostU8(bytes(n * h / 16), listOf(n, h / 16)),
                            s.bufferFromHostF32(floatArrayOf(0.01f), emptyList()),
                        )),
                        Triple("int4, bf16 group scales", int4Group(rows), listOf(
                            s.bufferFromHostU8(bytes(n * h / 2), listOf(n, h / 2)), s.bufferFromHostBf16(ShortArray(n * h / 16) { 0x3c00 }, listOf(n, h / 16)),
                        )),
                    ))
                val stored = mapOf(
                    "bf16" to 2.0 * n * h, "fp8" to 1.0 * n * h, "nvfp4" to 0.5 * n * h + n * h / 16.0,
                    "fp8, scale after" to 1.0 * n * h, "nvfp4, bf16 scales" to 0.5 * n * h + n * h / 16.0,
                    "int8, channel scale after" to 1.0 * n * h, "int4, bf16 group scales" to 0.5 * n * h + n * h / 8.0,
                    "nvfp4 W4A4, block-scaled dot" to 0.5 * n * h + n * h / 16.0,
                )
                for ((name, mlir, ws) in cases) {
                    s.prepareStablehlo(mlir)
                    repeat(5) { s.executeStablehlo(mlir, listOf(x) + ws).forEach { it.close() } }
                    val iters = 50
                    val t0 = System.nanoTime()
                    repeat(iters) { s.executeStablehlo(mlir, listOf(x) + ws).forEach { it.close() } }
                    val ms = (System.nanoTime() - t0) / 1e6 / iters
                    println("[quant-gemv] rows=$rows $name: %.3f ms per call, %.0f GB/s of stored weight".format(ms, stored[name]!! / ms / 1e6))
                }
            }
        }
    }
}
