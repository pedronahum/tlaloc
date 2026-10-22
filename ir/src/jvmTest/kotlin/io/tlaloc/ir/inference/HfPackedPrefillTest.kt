package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * Packed prefill entries ([DecodeGraphSpec.packedTokens]) on the tiny Qwen3.5-MoE checkpoint (Gated
 * DeltaNet, gated attention, experts): two prompts of 5 and 11 tokens, right-aligned in a
 * two-sequence entry of 16 tokens each, and the same call packed into 16 rows (no padding) and
 * 20 (four padding rows). The logits and every pool after the call are bit for bit the unpacked
 * entry's: a row's arithmetic does not depend on the rows beside it.
 */
class HfPackedPrefillTest {

    private val dir: Path = Path.of(javaClass.getResource("tiny_qwen3_5_moe/config.json")!!.toURI()).parent

    @Test
    fun aPackedPrefillEqualsThePaddedOne() {
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject
        val base = ((fixture["prompts"] as JsonArray).elements[0] as JsonObject)["promptTokens"] as JsonArray
        val tokens0 = base.elements.map { (it as JsonNumber).value.toInt() }
        val prompts = listOf(tokens0.take(5), (tokens0 + tokens0 + tokens0).take(11))
        val bs = 4
        val context = 32
        val t = 16
        val b = 2
        val m = context / bs
        val model = config.toDecodeModelShape(numBlocks = 1 + b * m, blockSize = bs, stateSlots = 2)
        val bucket = DecodeBucket(b, context)
        val padded = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = t)

        // The right-aligned rows, as a server lays them out (DecodePadding: token 0, position 0, slot -1).
        val tok = FloatArray(b * t)
        val pos = FloatArray(b * t)
        val slot = FloatArray(b * t) { -1f }
        val tables = FloatArray(b * m) { (1 + it).toFloat() }
        val lens = FloatArray(b)
        for ((r, p) in prompts.withIndex()) {
            val pad = t - p.size
            for (j in p.indices) {
                tok[r * t + pad + j] = p[j].toFloat()
                pos[r * t + pad + j] = j.toFloat()
                slot[r * t + pad + j] = ((1 + r * m + j / bs) * bs + j % bs).toFloat()
            }
            lens[r] = p.size.toFloat()
        }
        val live = (0 until b * t).filter { slot[it] >= 0f }
        fun pools() = padded.inputs.drop(padded.kvPoolInputBase).take(padded.numPools).map { FloatArray(it.type.dims.fold(1) { a, c -> a * c }) }
        val common = listOf(tok, pos, tables, lens, slot, floatArrayOf(0f, 1f))
        val want = DxirInterpreter.evalFunction(HfDecoderGraph.build(padded, config), common + pools() + weights)

        for (p in listOf(16, 20)) {
            val spec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = t, packedTokens = p)
            val packIndex = FloatArray(p) { if (it < live.size) live[it].toFloat() else -1f }
            val unpackIndex = FloatArray(b * t) { i -> live.indexOf(i).toFloat() }
            val got = DxirInterpreter.evalFunction(
                HfDecoderGraph.build(spec, config), common + listOf(packIndex, unpackIndex) + pools() + weights,
            )
            assertEquals(want.size, got.size)
            for (i in want.indices) assertContentEquals(want[i], got[i], "packed into $p rows: output $i")
            println("[packed-prefill] ${live.size} live tokens packed into $p rows: the logits and all ${want.size - 1} pools equal the padded entry's")
        }
    }
}
