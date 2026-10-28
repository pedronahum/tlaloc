package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirRemat
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.capture
import io.tlaloc.nn.captureCausalLm
import io.tlaloc.nn.crossEntropy
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The LM head's cross-entropy in chunks ([captureCausalLm]) on the device: a step whose logits
 * are large next to the model (4,096 tokens, a 32,768-token vocabulary: 512 MiB of f32 logits),
 * each step in its own session. Chunked, the forward pass holds one chunk's logits at a time;
 * rematerialized with [DxirRemat.SAVE_ALL], so does the backward pass, and the step's peak falls
 * to under a third of the full loss's, with the same gradients.
 */
class PjrtLmHeadLossMemoryTest {

    private val config = CausalLmConfig(
        vocabSize = 32_768, dModel = 64, numLayers = 1, numHeads = 2, numKvHeads = 1, ffHidden = 128, initStd = 0.05f,
    )
    private val batch = 4
    private val seq = 1024
    private val chunk = 512

    private fun peak(fn: DxirFunction, inputs: List<FloatArray>): Pair<List<FloatArray>, Long> =
        TestBackend.session().use { s ->
            val out = s.runOn(fn, inputs)
            out to (s.memoryStats().second ?: error("the plugin reports no peak"))
        }

    @Test
    fun chunksWithRematLowerThePeakAndKeepTheGradient() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val model = CausalLM.llama(config, RandomKey.fromSeed(2))
        val ids = IntArray(batch * seq) { (it * 7919 + 13) % config.vocabSize }
        val next = IntArray(ids.size) { if (it % seq < 16) -100 else (ids[it] * 31 + 5) % config.vocabSize }
        val idsT = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32)
        val nextT = DTensor<Shape, I32>(HostI32Storage(next), intArrayOf(batch, seq), I32)
        val inputs = listOf(FloatArray(ids.size) { ids[it].toFloat() }, FloatArray(next.size) { next[it].toFloat() }) +
            model.parameters.map { it.tensor.hostF32() }

        val full = capture(model, listOf(idsT), targets = listOf(nextT)) { logits, t -> crossEntropy(logits, t[0]) }
        val chunked = captureCausalLm(model, idsT, nextT, chunk)
        val (want, fullPeak) = peak(full.gradient, inputs)
        val (_, chunkedPeak) = peak(chunked.gradient, inputs)
        val (got, rematPeak) = peak(chunked.withRemat(DxirRemat.SAVE_ALL).gradient, inputs)

        val mib = 1024.0 * 1024.0
        println(
            "PjrtLmHeadLossMemoryTest: peak %.0f MiB full, %.0f MiB in chunks of %d, %.0f MiB in chunks with remat"
                .format(fullPeak / mib, chunkedPeak / mib, chunk, rematPeak / mib),
        )
        // XLA runs f32 dots at TF32 by default: compare against each output's scale.
        for (k in want.indices) {
            val scale = maxOf(1e-6f, want[k].maxOf { abs(it) })
            val worst = want[k].indices.maxOf { abs(want[k][it] - got[k][it]) } / scale
            assertTrue(worst <= 2e-3f, "result $k differs by $worst of its scale")
        }
        // On the GB10: 1,684 MiB full, 788 MiB in chunks, 385 MiB in chunks with remat.
        assertTrue(rematPeak < fullPeak / 3, "chunked remat peak $rematPeak should be under a third of $fullPeak")
    }
}
