package io.tlaloc.stablehlo

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I8
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfModelFamily
import io.tlaloc.ir.inference.WeightQuant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Text pins for int8 weights: the `i8` element type, and a quantized
 * projection lowered as a `stablehlo.convert` of the int8 parameter to the
 * compute dtype feeding the `dot_general`, then a multiply by the scales. The
 * weights stay int8 in memory; the widening is part of the program, where XLA
 * fuses it into the matmul.
 */
class Int8WeightEmitTest {

    private val small = HfDecoderConfig(
        architecture = "Qwen3ForCausalLM", modelType = "qwen3",
        hiddenSize = 16, intermediateSize = 24, numLayers = 2,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 1e6, maxPositionEmbeddings = 64,
        tieWordEmbeddings = true, attentionBias = false, torchDtype = "bfloat16",
        ropeScalingType = null, family = HfModelFamily.Qwen3,
    )

    private fun emit(config: HfDecoderConfig): String {
        val model = config.toDecodeModelShape(numBlocks = 4, blockSize = 4)
        return HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, 16)), config).toStablehlo()
    }

    @Test
    fun i8TypeSpellsMlirI8() {
        assertEquals("tensor<16x24xi8>", DxirType(I8, listOf(16, 24)).toMlir())
    }

    @Test
    fun aQuantizedProjectionWidensTheCodesIntoTheDotAndScalesItsOutput() {
        for (dtype in listOf(BF16, F32)) {
            val config = small.copy(weightDType = dtype, weightQuant = WeightQuant.INT8)
            val mlir = emit(config)
            val el = if (dtype == BF16) "bf16" else "f32"
            // Seven Linears per layer: q, k, v, o, gate, up, down.
            val widen = Regex("""stablehlo\.convert %\w+ : \(tensor<(\d+)x(\d+)xi8>\) -> tensor<\1x\2x$el>""")
            assertEquals(7 * config.numLayers, widen.findAll(mlir).count(), "one widening per quantized Linear ($el)")
            assertTrue("tensor<16x24xi8>" in mlir && "tensor<24x16xi8>" in mlir, "gate/up and down as int8 parameters")
            // The scales are f32 [out] parameters, broadcast over the rows.
            assertTrue(Regex("""%\w+: tensor<24xf32>""").containsMatchIn(mlir), "an f32 [24] scale parameter")
            // The embedding table keeps the weight dtype.
            assertTrue("tensor<29x16x$el>" in mlir, "the embedding table stays $el")
        }
        // Without quantization nothing is int8.
        assertTrue("xi8>" !in emit(small.copy(weightDType = BF16)))
    }
}
