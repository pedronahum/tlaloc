@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirVmapTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * vmap through StableHLO on PJRT: the batched function `DxirVmapTransform` produces is
 * emitted and run on the device, and its outputs must equal the per-example function
 * evaluated on the reference interpreter for each example and stacked. F32 with
 * full-precision dots (no TF32) and F64, batch sizes 1, 7 and 64. Skips by name without
 * a PJRT plugin or a device.
 */
class PjrtVmapTest {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    private fun DxirBuilder.c(dt: DType, v: Double): DxirNode = const(if (dt == F64) v else v.toFloat(), t(dt))

    private fun input(param: Int, example: Int, size: Int): DoubleArray {
        var s = (param * 7919L + example * 104729L + 12345L) and 0x7fffffffL
        return DoubleArray(size) {
            s = (s * 1103515245L + 12345L) and 0x7fffffffL
            ((s % 20000L) - 10000L) / 7000.0
        }
    }

    private fun size(fn: DxirFunction, p: Int) = fn.params[p].type.dims.fold(1) { a, d -> a * d }

    /** vmap on the device against stacking the interpreter's per-example results. */
    private fun check(build: (DType) -> DxirFunction, batched: List<Boolean>, tolerance: Double) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session(portableF32Dots = true).use { session ->
            for (dt in listOf(F32, F64)) for (batch in listOf(1, 7, 64)) {
                val fn = build(dt)
                val vfn = DxirVmapTransform.apply(fn, batched, batch)
                val per = (0 until batch).map { e ->
                    fn.params.indices.map { p -> input(p, if (batched[p]) e else 0, size(fn, p)) }
                }
                val stackedIn = fn.params.indices.map { p ->
                    if (batched[p]) per.flatMap { it[p].asList() }.toDoubleArray() else per[0][p]
                }
                val want: List<List<DoubleArray>>
                val got: List<DoubleArray>
                if (dt == F64) {
                    want = per.map { DxirInterpreterF64.evalFunction(fn, it) }
                    got = session.runOnF64(vfn, stackedIn)
                } else {
                    fun f(d: DoubleArray) = FloatArray(d.size) { d[it].toFloat() }
                    fun d(f: FloatArray) = DoubleArray(f.size) { f[it].toDouble() }
                    want = per.map { ex -> DxirInterpreter.evalFunction(fn, ex.map(::f)).map(::d) }
                    got = session.runOn(vfn, stackedIn.map(::f)).map(::d)
                }
                for (r in fn.returns.indices) {
                    val stacked = want.flatMap { it[r].asList() }
                    assertEquals(stacked.size, got[r].size, "${fn.name} $dt batch $batch output $r size")
                    val scale = maxOf(1.0, stacked.maxOf { abs(it) })
                    val worst = stacked.indices.maxOf { abs(got[r][it] - stacked[it]) } / scale
                    assertTrue(
                        worst <= tolerance,
                        "${fn.name} $dt batch $batch output $r: largest difference $worst of the largest " +
                            "magnitude, above $tolerance",
                    )
                }
            }
        }
    }

    @Test
    fun `elementwise and broadcasting ops`() = check(
        build = { dt ->
            DxirBuilder.function("elementwise") {
                val x = param("x", t(dt, 3, 4))
                val bias = param("bias", t(dt, 4))
                val a = op(OpKind.TANH, listOf(x), t(dt, 3, 4))
                val e = op(OpKind.EXP, listOf(op(OpKind.ADD, listOf(a, bias), t(dt, 3, 4))), t(dt, 3, 4))
                val s = op(OpKind.SUM, listOf(x), t(dt))
                val scaled = op(OpKind.MUL, listOf(s, e), t(dt, 3, 4))
                val two = op(
                    OpKind.BROADCAST, listOf(c(dt, 2.0)), t(dt, 3, 4),
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
                listOf(op(OpKind.DIV, listOf(scaled, two), t(dt, 3, 4)), op(OpKind.NEG, listOf(bias), t(dt, 4)))
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-5,
    )

    @Test
    fun `reductions, softmax and shape ops`() = check(
        build = { dt ->
            DxirBuilder.function("shapes") {
                val x = param("x", t(dt, 2, 3, 4))
                val sm = op(OpKind.SOFTMAX, listOf(x), t(dt, 2, 3, 4), attrs = mapOf("axis" to 1))
                val mx = op(OpKind.MAX, listOf(sm), t(dt, 2, 4), attrs = mapOf("reduction_dims" to listOf(1)))
                val tr = op(OpKind.TRANSPOSE, listOf(mx), t(dt, 4, 2), attrs = mapOf("permutation" to listOf(1, 0)))
                val flat = op(OpKind.RESHAPE, listOf(tr), t(dt, 8))
                val sl = op(
                    OpKind.SLICE, listOf(flat), t(dt, 3),
                    attrs = mapOf("start_indices" to listOf(2), "limit_indices" to listOf(5), "strides" to listOf(1)),
                )
                val mean = op(OpKind.MEAN, listOf(x), t(dt))
                listOf(sl, mean)
            }
        },
        batched = listOf(true),
        tolerance = 1e-5,
    )

    @Test
    fun `matmuls with batched and unbatched operands and a cross-entropy loss`() = check(
        build = { dt ->
            DxirBuilder.function("mlp") {
                val x = param("x", t(dt, 1, 4))
                val oneHot = param("oneHot", t(dt, 1, 3))
                val w1 = param("w1", t(dt, 4, 5))
                val w2 = param("w2", t(dt, 5, 3))
                val h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w1), t(dt, 1, 5))), t(dt, 1, 5))
                val logits = op(OpKind.MATMUL, listOf(h, w2), t(dt, 1, 3))
                val logp = op(OpKind.LOG, listOf(op(OpKind.SOFTMAX, listOf(logits), t(dt, 1, 3))), t(dt, 1, 3))
                val loss = op(
                    OpKind.NEG,
                    listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(oneHot, logp), t(dt, 1, 3))), t(dt))),
                    t(dt),
                )
                listOf(loss, logits)
            }
        },
        batched = listOf(true, true, false, false),
        tolerance = 1e-5,
    )
}
