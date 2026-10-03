package io.tlaloc.ir.inference

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * NVIDIA's NVFP4 Qwen3.8-27B (ModelOpt: FP8 projections in the attention and
 * Gated DeltaNet layers, NVFP4 MLPs) read back through [HfCheckpoint]'s
 * dequantization, against the bf16 Qwen3.8-27B it was quantized from. A wrong
 * nibble order, scale layout or scale product would put the error far past
 * what the formats themselves cost. Skips without both checkpoints.
 */
class QuantizedCheckpointReadTest {

    private fun snapshot(repo: String): Path? {
        val root = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub", "models--" + repo.replace("/", "--"), "snapshots")
        if (!Files.isDirectory(root)) return null
        return Files.list(root).use { s -> s.toList().firstOrNull { Files.isRegularFile(it.resolve("config.json")) } }
    }

    private fun relErr(a: FloatArray, b: FloatArray): Double {
        var num = 0.0
        var den = 0.0
        for (i in a.indices) {
            val d = (a[i] - b[i]).toDouble()
            num += d * d
            den += b[i].toDouble() * b[i]
        }
        return sqrt(num / den)
    }

    @Test
    fun dequantizedWeightsStayWithinTheirFormatsErrorOfTheBf16Weights() {
        val q = snapshot("nvidia/Qwen3.8-27B-NVFP4")
        val b = snapshot("Qwen/Qwen3.8-27B")
        assumeTrue(q != null && b != null, "needs nvidia/Qwen3.8-27B-NVFP4 and Qwen/Qwen3.8-27B in the HuggingFace cache")
        HfCheckpoint.open(q!!).use { qc ->
            HfCheckpoint.open(b!!).use { bc ->
                val fp8 = DecoderWeightRole.Layer(0, DecoderLayerPart.IN_PROJ_QKV)
                val nvfp4 = DecoderWeightRole.Layer(0, DecoderLayerPart.GATE_PROJ)
                assertTrue(qc.storesQuantized(fp8) && qc.storesQuantized(nvfp4) && !bc.storesQuantized(nvfp4))
                val eFp8 = relErr(qc.load(fp8).toF32Array(), bc.load(fp8).toF32Array())
                val eFp4 = relErr(qc.load(nvfp4).toF32Array(), bc.load(nvfp4).toF32Array())
                println("dequantized against bf16: FP8 in_proj_qkv relative error $eFp8, NVFP4 gate_proj $eFp4")
                assertTrue(eFp8 < 0.05, "FP8 relative error $eFp8")
                assertTrue(eFp4 < 0.15, "NVFP4 relative error $eFp4")
            }
        }
    }
}
