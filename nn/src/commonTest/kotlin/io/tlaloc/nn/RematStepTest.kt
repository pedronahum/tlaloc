package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirRemat
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** [CapturedStep.withRemat] on a transformer step: the same loss and gradients, bit for bit. */
class RematStepTest {

    private val config = CausalLmConfig(
        vocabSize = 16, dModel = 16, numLayers = 2, numHeads = 2, numKvHeads = 1, ffHidden = 32, initStd = 0.1f,
    )

    @Test
    fun aRematerializedTransformerStepComputesTheSameGradients() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(4))
        val ids = IntArray(2 * 4) { (5 * it + 1) % config.vocabSize }
        val next = IntArray(ids.size) { (ids[it] + 3) % config.vocabSize }
        val idsT = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(2, 4), I32)
        val targets: DTensor<*, F32> = oneHot(next, config.vocabSize, intArrayOf(2, 4))
        val step = capture(model, listOf(idsT), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }

        val plain = step.run(model, listOf(idsT, targets))
        for (policy in listOf(DxirRemat.SAVE_MARKED, DxirRemat.SAVE_DOTS, DxirRemat.SAVE_NOTHING)) {
            val remat = step.withRemat(policy)
            assertTrue(remat.gradient.body.any { it is DxirOp && it.op == OpKind.OPTIMIZATION_BARRIER })
            val r = remat.run(model, listOf(idsT, targets))
            assertEquals(plain.loss, r.loss)
            for ((key, g) in plain.gradients) {
                assertContentEquals((g.storage as io.tlaloc.core.HostF32Storage).data, (r.gradients.getValue(key).storage as io.tlaloc.core.HostF32Storage).data, key)
            }
        }
        // CausalLM marks each block's output: under SAVE_MARKED those are what the backward reads.
        val marked = step.gradient.body.filterIsInstance<DxirOp>().count { it.attrs[DxirRemat.SAVE_ATTR] == true }
        assertEquals(config.numLayers, marked)
        // The device optimizer composes with it.
        assertEquals(1 + 3 * step.parameterKeys.size, step.withRemat().withDeviceAdamW(AdamW()).function.returns.size)
    }
}
