package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.nn.MixtureOfTopKAttention
import io.tlaloc.nn.capture
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * A [MixtureOfTopKAttention] training step on the device against the interpreter: the
 * landmark top-k (`chlo.top_k`), the batched gathers of keys and values and their
 * scatter-add adjoints (`stablehlo.gather` / `stablehlo.scatter` with batching dims).
 */
class PjrtMixtureOfTopKAttentionTest {

    @Test
    fun trainingStepOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val batch = 2
        val seq = 16
        val dModel = 16
        val model = MixtureOfTopKAttention(
            dModel, numHeads = 2, landmarks = 4, topK = 4, key = RandomKey.fromSeed(21), initStd = 0.3f,
            tokenGrid = 4 to 4, landmarkGrid = 2 to 2,
        )
        val keys = RandomKey.fromSeed(4).split(2)
        val x = uniformFloats(keys[0], batch * seq * dModel).map { 2f * it - 1f }.toFloatArray()
        val w = uniformFloats(keys[1], batch * seq * dModel).map { 2f * it - 1f }.toFloatArray()
        val input = DTensor<Shape, F32>(HostF32Storage(x), intArrayOf(batch, seq, dModel), F32)
        val step = capture(model, listOf(input)) { out: Tracer<Shape> ->
            (out * out.constant<Shape>(w, out.dims)).sum()
        }
        val values = listOf(x) + step.parameterTensors(model).map { it.hostF32() }
        val host = DxirInterpreter.evalFunction(step.gradient, values)
        // The matmuls run at TF32 on the GB10 (defaultDotRelTolerance); for these inputs
        // the landmark and routing choices come out as the interpreter's.
        TestBackend.session().use { session ->
            val device = session.runOnHost(step.gradient, values).map { it as FloatArray }
            for ((i, h) in host.withIndex()) {
                val scale = maxOf(1f, h.maxOf { abs(it) })
                val err = h.indices.maxOf { abs(device[i][it] - h[it]) } / scale
                assertTrue(err < TestBackend.defaultDotRelTolerance, "output $i: device against interpreter, scaled error $err")
            }
            println("[pjrt] MiTA step: loss device ${device[0][0]}, interpreter ${host[0][0]}")
        }
    }

    @Test
    fun selectionOpsAreExactOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // loss = Σ w ⊙ take(x, take(table, topK(s, 2))): integer gathers, a float gather and
        // its scatter-add with repeated indices, all exact.
        val xT = DxirType(F32, listOf(2, 5, 3))
        val fn = DxirBuilder.function("sel") {
            val x = param("x", xT)
            val s = param("s", DxirType(F32, listOf(2, 3, 5)))
            val route = param("route", DxirType(I32, listOf(2, 4)))
            val w = param("w", DxirType(F32, listOf(2, 8, 3)))
            val top = op(OpKind.TOP_K, listOf(s), DxirType(I32, listOf(2, 3, 2)), attrs = mapOf("k" to 2))
            val perRow = op(OpKind.TAKE_ALONG, listOf(top, route), DxirType(I32, listOf(2, 4, 2)))
            val flat = op(OpKind.RESHAPE, listOf(perRow), DxirType(I32, listOf(2, 8)))
            val g = op(OpKind.TAKE_ALONG, listOf(x, flat), DxirType(F32, listOf(2, 8, 3)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(g, w), DxirType(F32, listOf(2, 8, 3)))), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(fn, includeForward = true)
        val x = FloatArray(30) { (it % 7) - 3f }
        val s = FloatArray(30) { ((it * 11) % 13).toFloat() }
        val route = floatArrayOf(2f, 0f, 2f, 1f, 1f, 1f, 0f, 2f)
        val w = FloatArray(48) { 0.25f * ((it % 5) - 2) }
        val host = DxirInterpreter.evalFunction(grad, listOf(x, s, route, w))
        TestBackend.session().use { session ->
            val device = session.runOnHost(grad, listOf(x, s, IntArray(8) { route[it].toInt() }, w))
            assertContentEquals(host[0], device[0] as FloatArray, "loss")
            assertContentEquals(host[1], device[1] as FloatArray, "d x")
            assertContentEquals(host[4], device[4] as FloatArray, "d w")
        }
    }
}
