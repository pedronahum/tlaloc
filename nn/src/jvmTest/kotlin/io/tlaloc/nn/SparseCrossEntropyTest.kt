package io.tlaloc.nn

import io.tlaloc.autograd.captureN
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * [lmHeadSparseCrossEntropy] against torch (`harness/python/sparse_ce_fixture.py`): 10 rows in
 * chunks of 3, rows with a teacher's top-4, rows with the true token alone and rows without a
 * target; the loss and its gradients for the hidden states and the head (the compiler's reverse
 * transform through the interpreter).
 */
class SparseCrossEntropyTest {
    private val fixture = parseJson(
        Files.readString(Paths.get(javaClass.getResource("/io/tlaloc/nn/sparse_ce/fixture.json")!!.toURI())),
    ) as JsonObject

    private fun doubles(k: String) = (fixture[k] as JsonArray).elements.map { (it as JsonNumber).raw.toDouble() }
    private fun dim(k: String) = ((fixture["dims"] as JsonObject)[k] as JsonNumber).raw.toInt()

    @Test
    fun lossAndGradientsEqualTorch() {
        val (b, t, d, v, j) = listOf(dim("b"), dim("t"), dim("d"), dim("v"), dim("j"))
        val chunk = (fixture["chunk"] as JsonNumber).raw.toInt()
        val hidden = doubles("hidden").map { it.toFloat() }.toFloatArray()
        val head = doubles("head").map { it.toFloat() }.toFloatArray()
        val ids = doubles("ids").map { it.toInt() }.toIntArray()
        val weights = doubles("weights").map { it.toFloat() }.toFloatArray()
        val ins = listOf<DTensor<*, *>>(
            DTensor<Shape, F32>(HostF32Storage(hidden), intArrayOf(b, t, d), F32),
            DTensor<Shape, F32>(HostF32Storage(head), intArrayOf(d, v), F32),
            DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(b, t, j), I32),
            DTensor<Shape, F32>(HostF32Storage(weights), intArrayOf(b, t, j), F32),
        )
        val fn = captureN(ins) { l -> lmHeadSparseCrossEntropy(l[0], l[1], l[2], l[3], chunk) }
        val values = listOf(hidden, head, ids.map { it.toFloat() }.toFloatArray(), weights)
        val loss = DxirInterpreter.evalFunction(fn, values).single()[0]
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), values)
        val wantLoss = (fixture["loss"] as JsonNumber).raw.toDouble()
        assertTrue(abs(loss - wantLoss) < 1e-5 * maxOf(1.0, abs(wantLoss)), "loss $loss against torch $wantLoss")
        for ((k, name) in listOf(0 to "d_hidden", 1 to "d_head")) {
            val want = doubles(name)
            val scale = want.maxOf { abs(it) }
            val worst = want.indices.maxOf { abs(grads[k][it] - want[it]) } / scale
            println("sparse CE: $name worst ${worst} of scale $scale")
            assertTrue(worst < 1e-5, "$name differs from torch by $worst of its scale")
        }
    }
}
