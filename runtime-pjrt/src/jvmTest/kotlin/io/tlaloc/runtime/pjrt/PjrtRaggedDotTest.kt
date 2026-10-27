package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.RaggedDotAttrs
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.testing.Tolerances
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * RAGGED_DOT through XLA (`chlo.ragged_dot`, both modes) against the interpreter: the products
 * and the gradient of a grouped two-layer MLP, the shape of a mixture of experts' training step
 * once its rows are sorted by expert.
 */
class PjrtRaggedDotTest {

    private val m = 96
    private val h = 32
    private val inter = 48
    private val sizes = intArrayOf(20, 0, 41, 30) // 5 rows in no group
    private val g = sizes.size
    private val r = Random(9)
    private fun values(size: Int, scale: Float = 1f) = FloatArray(size) { (r.nextFloat() - 0.5f) * scale }
    private val sizesF = FloatArray(g) { sizes[it].toFloat() }

    private fun check(fn: DxirFunction, inputs: List<FloatArray>, tol: Double) {
        val want = DxirInterpreter.evalFunction(fn, inputs)
        val got = TestBackend.session().use { it.runOn(fn, inputs) }
        for (i in want.indices) {
            Tolerances.assertClose(DoubleArray(got[i].size) { got[i][it].toDouble() }, DoubleArray(want[i].size) { want[i][it].toDouble() }, tol, "${fn.name} result $i")
        }
    }

    @Test
    fun bothFormsMatchTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fn = DxirBuilder.function("ragged") {
            val x = param("x", DxirType(F32, listOf(m, h)))
            val w = param("w", DxirType(F32, listOf(g, h, inter)))
            val dy = param("dy", DxirType(F32, listOf(m, inter)))
            val s = param("sizes", DxirType(I32, listOf(g)))
            listOf(
                op(OpKind.RAGGED_DOT, listOf(x, w, s), DxirType(F32, listOf(m, inter))),
                op(OpKind.RAGGED_DOT, listOf(x, dy, s), DxirType(F32, listOf(g, h, inter)), attrs = mapOf("ragged" to RaggedDotAttrs.CONTRACTION)),
            )
        }
        check(fn, listOf(values(m * h), values(g * h * inter), values(m * inter), sizesF), 1e-5)
    }

    @Test
    fun aGroupedMlpsGradientMatchesTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // Σ (tanh(x·up[e])·down[e])² over each row's group e.
        val loss = DxirBuilder.function("grouped_mlp") {
            val x = param("x", DxirType(F32, listOf(m, h)))
            val up = param("up", DxirType(F32, listOf(g, h, inter)))
            val down = param("down", DxirType(F32, listOf(g, inter, h)))
            val s = param("sizes", DxirType(I32, listOf(g)))
            val a = op(OpKind.TANH, listOf(op(OpKind.RAGGED_DOT, listOf(x, up, s), DxirType(F32, listOf(m, inter)))), DxirType(F32, listOf(m, inter)))
            val y = op(OpKind.RAGGED_DOT, listOf(a, down, s), DxirType(F32, listOf(m, h)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(loss, includeForward = true, inputOnlyTrailingParams = 1)
        check(grad, listOf(values(m * h), values(g * h * inter, 0.3f), values(g * inter * h, 0.3f), sizesF), 1e-4)
    }
}
