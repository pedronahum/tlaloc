package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [InklingLoop.forward] against transformers 5.24.0's modeling code on `tiny_inkling` (written by
 * `harness/python/inkling_modeling_fixture.py`: six layers in blocks of two, layers 2-3 looped
 * three times, full attention at layers 3 and 5, a window of 8 and a relative extent of 6 shorter
 * than the 19-token prompt, log scaling from position 3, clipping thresholds that clip). The logits
 * are compared at every position.
 */
class InklingLoopTest {
    private val dir: Path = Path.of("src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")

    private fun logits(ck: InklingCheckpoint, weights: Map<String, DTensor<*, *>>, ids: IntArray): FloatArray {
        val keys = weights.keys.toList()
        var out: Tracer<*>? = null
        captureN(listOf(DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(ids.size), I32)) + weights.values) { leaves ->
            val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[1 + i] })
            InklingLoop.forward(ck.config, leaves[0], w).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    @Test
    fun theTinyCheckpointReadsAsLooped() {
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            assertEquals(listOf(true, true, true, false, true, false), c.sliding)
            assertEquals(Triple(2, 4, 3), Triple(c.loopStart, c.loopEnd, c.passes))
            assertEquals(5, c.gateRank)
            assertEquals(128, c.unpaddedVocabSize)
        }
    }

    /** NVFP4 and MXFP8 expert stacks decoded as transformers' `Inkling2LoopDequantizeExperts` decodes them (to bf16). */
    @Test
    fun quantizedExpertsDecodeAsTransformersDoes() {
        val want = parseJson(Files.readString(dir.resolve("quantized_experts.json"))) as JsonObject
        val e = (want["experts"] as JsonNumber).value.toInt()
        val rows = (want["rows"] as JsonNumber).value.toInt()
        val cols = (want["cols"] as JsonNumber).value.toInt()
        io.tlaloc.core.io.SafetensorsFile.open(dir.resolve("quantized_experts.safetensors")).use { f ->
            fun load(n: String) = if (n in f.names) f.load(n) else null
            for ((name, key) in listOf("nv.w" to "nvfp4", "mx.w" to "mxfp8")) {
                val got = InklingCheckpoint.dequantizeExperts(name, e, rows, cols, ::load)
                val exp = (want[key] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }
                assertEquals(exp.size, got.size)
                // The op returns the model's dtype, bf16 here (as the released model loads): compare at bf16.
                for (i in exp.indices) {
                    assertEquals(exp[i], io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(got[i])), "$key element $i")
                }
            }
        }
    }

    @Test
    fun logitsMatchTheModelingCode() {
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        InklingCheckpoint(dir).use { ck ->
            val weights = ck.hostWeights()
            for (p in (fixture["prompts"] as JsonArray).elements) {
                p as JsonObject
                val ids = (p["ids"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
                val want = (p["logits"] as JsonArray).elements.flatMap { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
                val got = logits(ck, weights, ids)
                assertEquals(want.size, got.size)
                var worst = 0.0
                for (i in want.indices) worst = maxOf(worst, abs(want[i] - got[i]))
                println("tiny_inkling ${ids.size} tokens: worst logit difference $worst")
                assertTrue(worst < 2e-4, "worst logit difference $worst over ${ids.size} tokens")
            }
        }
    }
}
