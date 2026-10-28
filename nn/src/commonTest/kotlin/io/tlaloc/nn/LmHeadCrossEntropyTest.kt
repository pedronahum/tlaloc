package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirRemat
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Cross-entropy against I32 ids, and the LM head's cross-entropy computed in chunks. */
class LmHeadCrossEntropyTest {

    private val config = CausalLmConfig(
        vocabSize = 24, dModel = 16, numLayers = 2, numHeads = 2, numKvHeads = 1, ffHidden = 32, initStd = 0.1f,
    )
    private val batch = 2
    private val seq = 5

    private val ids = IntArray(batch * seq) { (5 * it + 1) % config.vocabSize }
    // Two ignored positions, as a prompt's would be.
    private val next = IntArray(ids.size) { if (it % seq == 0) -100 else (ids[it] + 3) % config.vocabSize }
    private val idsT = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32)
    private val nextT = DTensor<Shape, I32>(HostI32Storage(next), intArrayOf(batch, seq), I32)
    private val oneHotT: DTensor<*, F32> = oneHot(next, config.vocabSize, intArrayOf(batch, seq), ignoreIndex = -100)

    private fun close(want: StepResult, got: StepResult, tol: Double) {
        assertTrue(abs(want.loss - got.loss) <= tol * maxOf(1f, abs(want.loss)), "loss ${got.loss} against ${want.loss}")
        assertEquals(want.gradients.keys, got.gradients.keys)
        for ((key, g) in want.gradients) {
            val a = g.hostF32()
            val b = got.gradients.getValue(key).hostF32()
            val scale = maxOf(1e-6f, a.maxOf { abs(it) })
            for (i in a.indices) assertTrue(abs(a[i] - b[i]) <= tol * scale, "d $key[$i]: ${b[i]} against ${a[i]}")
        }
    }

    @Test
    fun idsGiveTheOneHotLossAndGradients() {
        for (tied in listOf(false, true)) {
            val model = CausalLM.llama(config.copy(tiedEmbeddings = tied), RandomKey.fromSeed(4))
            val dense = capture(model, listOf(idsT), targets = listOf(oneHotT)) { logits, t -> crossEntropy(logits, t[0]) }
                .run(model, listOf(idsT, oneHotT))
            val sparse = capture(model, listOf(idsT), targets = listOf(nextT)) { logits, t -> crossEntropy(logits, t[0]) }
                .run(model, listOf(idsT, nextT))
            close(dense, sparse, 1e-6)
        }
    }

    @Test
    fun theChunkedLossIsTheFullLoss() {
        for (tied in listOf(false, true)) {
            val model = CausalLM.llama(config.copy(tiedEmbeddings = tied), RandomKey.fromSeed(7))
            val full = capture(model, listOf(idsT), targets = listOf(nextT)) { logits, t -> crossEntropy(logits, t[0]) }
                .run(model, listOf(idsT, nextT))
            // 10 rows in chunks of 4: the last chunk is short.
            for (chunk in listOf(4, 10, 64)) {
                val step = captureCausalLm(model, idsT, nextT, chunk)
                close(full, step.run(model, listOf(idsT, nextT)), 1e-5)
            }
        }
    }

    @Test
    fun rematRecomputesEachChunkAndKeepsTheGradient() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(9))
        val step = captureCausalLm(model, idsT, nextT, chunk = 4)
        val plain = step.run(model, listOf(idsT, nextT))
        val chunks = 3
        assertEquals(chunks, step.primal.body.count { it is DxirOp && it.op == OpKind.LOG_SOFTMAX })
        for (policy in listOf(DxirRemat.SAVE_ALL, DxirRemat.SAVE_MARKED)) {
            val remat = step.withRemat(policy)
            // The backward pass computes each chunk's log-probabilities again.
            val logSoftmax = remat.gradient.body.count { it is DxirOp && it.op == OpKind.LOG_SOFTMAX }
            assertEquals(2 * chunks, logSoftmax, "$policy")
            val r = remat.run(model, listOf(idsT, nextT))
            assertEquals(plain.loss, r.loss)
            for ((key, g) in plain.gradients) assertContentEquals(g.hostF32(), r.gradients.getValue(key).hostF32(), key)
        }
    }
}
