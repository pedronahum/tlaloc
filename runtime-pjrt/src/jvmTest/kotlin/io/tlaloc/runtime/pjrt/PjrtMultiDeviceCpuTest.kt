package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.testing.Tolerances
import io.tlaloc.testing.inline
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Sharded and collective programs executed by [PjrtSession] itself on four host devices of
 * XLA's CPU plugin (`scripts/pjrt-cpu-plugin/build.sh`), or on four TPU chips
 * ([MultiDeviceBackend]). Each result is checked against what one device computes in the
 * interpreter. Skips when neither is there.
 */
class PjrtMultiDeviceCpuTest {

    private val n = 4
    private val all = listOf((0 until n).toList())
    private val r = Random(5)

    private fun values(k: Int) = FloatArray(k) { r.nextFloat() * 2 - 1 }

    private fun <T> session(block: (PjrtSession) -> T): T = MultiDeviceBackend.session(n, block = block)

    private fun interp(fn: DxirFunction, inputs: List<FloatArray>): List<FloatArray> = DxirInterpreter.evalFunction(fn, inputs)

    private fun assertClose(got: FloatArray, want: FloatArray, tol: Double, what: String) =
        Tolerances.assertClose(
            DoubleArray(got.size) { got[it].toDouble() },
            DoubleArray(want.size) { want[it].toDouble() },
            tol,
            what,
        )

    @Test
    fun theClientHasTheRequestedDevicesAndPlatform() = session { s ->
        if (!TestBackend.isTpu) assertEquals(n, s.devices.size)
        assertEquals(MultiDeviceBackend.platform, s.platformName())
    }

    @Test
    fun allReduceSumsOverFourDevices() = session { s ->
        val t = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("ar4") {
            val x = param("x", t)
            listOf(op(OpKind.ALL_REDUCE, listOf(x), t, attrs = mapOf("replica_groups" to all)))
        }
        val xs = List(n) { values(5) }
        val out = s.runPerDevice(fn, listOf(xs)).single()
        val want = FloatArray(5) { i -> xs.sumOf { it[i].toDouble() }.toFloat() }
        for ((d, got) in out.withIndex()) assertClose(got, want, 1e-6, "device $d")
    }

    @Test
    fun allReduceOverTwoGroups() = session { s ->
        val t = DxirType(F32, listOf(3))
        val groups = listOf(listOf(0, 2), listOf(1, 3))
        val fn = DxirBuilder.function("ar2x2") {
            val x = param("x", t)
            listOf(op(OpKind.ALL_REDUCE, listOf(x), t, attrs = mapOf("replica_groups" to groups)))
        }
        val xs = List(n) { values(3) }
        val out = s.runPerDevice(fn, listOf(xs)).single()
        for (d in 0 until n) {
            val g = groups.first { d in it }
            assertClose(out[d], FloatArray(3) { i -> g.sumOf { xs[it][i].toDouble() }.toFloat() }, 1e-6, "device $d")
        }
    }

    @Test
    fun allGatherThenReduceScatter() = session { s ->
        val shard = DxirType(F32, listOf(2, 3))
        val whole = DxirType(F32, listOf(2 * n, 3))
        val fn = DxirBuilder.function("ag_rs4") {
            val x = param("x", shard)
            val g = op(OpKind.ALL_GATHER, listOf(x), whole, attrs = mapOf("replica_groups" to all, "dimension" to 0))
            val sc = op(OpKind.REDUCE_SCATTER, listOf(g), shard, attrs = mapOf("replica_groups" to all, "dimension" to 0))
            listOf(g, sc)
        }
        val xs = List(n) { values(6) }
        val (gathered, scattered) = s.runPerDevice(fn, listOf(xs))
        val concat = xs.flatMap { it.asIterable() }.toFloatArray()
        for (d in 0 until n) {
            assertClose(gathered[d], concat, 0.0, "all_gather on device $d")
            assertClose(scattered[d], FloatArray(6) { i -> n * concat[d * 6 + i] }, 1e-6, "reduce_scatter on device $d")
        }
    }

    @Test
    fun allToAllAndCollectivePermuteMoveTheDevicesValues() = session { s ->
        // x [4, 3] per device: rows split four ways, the chunks concatenated along the columns.
        val a2a = DxirBuilder.function("a2a4") {
            val x = param("x", DxirType(F32, listOf(4, 3)))
            listOf(op(OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, listOf(1, 12)), attrs = mapOf("replica_groups" to all, "split_dimension" to 0, "concat_dimension" to 1)))
        }
        val xs = List(n) { values(12) }
        val moved = s.runPerDevice(a2a, listOf(xs)).single()
        for (d in 0 until n) {
            // Device d receives row d of every device, in device order.
            assertClose(moved[d], FloatArray(12) { i -> xs[i / 3][d * 3 + i % 3] }, 0.0, "all_to_all on device $d")
        }
        val ring = (0 until n).map { listOf(it, (it + 1) % n) }
        val shift = DxirBuilder.function("cp4") {
            val x = param("x", DxirType(F32, listOf(5)))
            listOf(op(OpKind.COLLECTIVE_PERMUTE, listOf(x), x.type, attrs = mapOf("source_target_pairs" to ring.dropLast(1))))
        }
        val ys = List(n) { values(5) }
        val shifted = s.runPerDevice(shift, listOf(ys)).single()
        assertClose(shifted[0], FloatArray(5), 0.0, "device 0 is no target")
        for (d in 1 until n) assertClose(shifted[d], ys[d - 1], 0.0, "collective_permute on device $d")
    }

    /**
     * The gradient of every device's `Σ f(x)·w` with respect to its x, against the adjoint
     * identity `Σ_d ⟨f(x)_d, w_d⟩ = Σ_d ⟨x_d, g_d⟩`: the per-device gradients are the transpose of
     * the collective applied to the cotangents, values that only cross-device semantics give.
     */
    private fun checkAdjoint(s: PjrtSession, what: String, inDims: List<Int>, outDims: List<Int>, collective: DxirBuilder.(io.tlaloc.ir.DxirNode) -> io.tlaloc.ir.DxirNode) {
        val forward = DxirBuilder.function("fwd") {
            val x = param("x", DxirType(F32, inDims))
            listOf(collective(x))
        }
        val loss = DxirBuilder.function("loss") {
            val x = param("x", DxirType(F32, inDims))
            val w = param("w", DxirType(F32, outDims))
            val y = collective(x)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, w), y.type)), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1)
        val inSize = inDims.fold(1) { a, b -> a * b }
        val outSize = outDims.fold(1) { a, b -> a * b }
        val xs = List(n) { values(inSize) }
        val ws = List(n) { values(outSize) }
        val ys = s.runPerDevice(forward, listOf(xs)).single()
        val gs = s.runPerDevice(grad, listOf(xs, ws)).single()
        var lhs = 0.0
        var rhs = 0.0
        for (d in 0 until n) {
            for (i in 0 until outSize) lhs += ys[d][i].toDouble() * ws[d][i]
            for (i in 0 until inSize) rhs += xs[d][i].toDouble() * gs[d][i]
        }
        assertTrue(kotlin.math.abs(lhs - rhs) <= 1e-5 * maxOf(1.0, kotlin.math.abs(lhs)), "$what: <f(x), w> = $lhs but <x, f*(w)> = $rhs")
    }

    @Test
    fun theCollectivesGradientsAreTheirAdjoints() = session { s ->
        checkAdjoint(s, "all_gather", listOf(2, 3), listOf(2 * n, 3)) { x ->
            op(OpKind.ALL_GATHER, listOf(x), DxirType(F32, listOf(2 * n, 3)), attrs = mapOf("replica_groups" to all, "dimension" to 0))
        }
        checkAdjoint(s, "reduce_scatter", listOf(3, 2 * n), listOf(3, 2)) { x ->
            op(OpKind.REDUCE_SCATTER, listOf(x), DxirType(F32, listOf(3, 2)), attrs = mapOf("replica_groups" to all, "dimension" to 1))
        }
        checkAdjoint(s, "all_gather in two groups", listOf(3), listOf(6)) { x ->
            op(OpKind.ALL_GATHER, listOf(x), DxirType(F32, listOf(6)), attrs = mapOf("replica_groups" to listOf(listOf(0, 2), listOf(1, 3)), "dimension" to 0))
        }
        checkAdjoint(s, "all_to_all", listOf(4, 3), listOf(1, 12)) { x ->
            op(OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, listOf(1, 12)), attrs = mapOf("replica_groups" to all, "split_dimension" to 0, "concat_dimension" to 1))
        }
        checkAdjoint(s, "all_to_all on one axis", listOf(8, 2), listOf(8, 2)) { x ->
            op(OpKind.ALL_TO_ALL, listOf(x), DxirType(F32, listOf(8, 2)), attrs = mapOf("replica_groups" to all, "split_dimension" to 0, "concat_dimension" to 0))
        }
        checkAdjoint(s, "collective_permute", listOf(5), listOf(5)) { x ->
            op(OpKind.COLLECTIVE_PERMUTE, listOf(x), x.type, attrs = mapOf("source_target_pairs" to listOf(listOf(0, 1), listOf(1, 3), listOf(3, 2))))
        }
    }

    @Test
    fun dataParallelGradientEqualsTheFullBatchGradient() = session { s ->
        val rows = 3
        fun loss(k: Int) = DxirBuilder.function("loss$k") {
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val x = param("x", DxirType(F32, listOf(k, 4)))
            val y = param("y", DxirType(F32, listOf(k, 2)))
            val h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(k, 2)))), DxirType(F32, listOf(k, 2)))
            val d = op(OpKind.SUB, listOf(h, y), DxirType(F32, listOf(k, 2)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(d, d), d.type)), DxirType(F32, emptyList())))
        }
        val shardGrad = DxirReverseTransform.apply(loss(rows), inputOnlyTrailingParams = 2)
        val step = DxirBuilder.function("dp_step") {
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val x = param("x", DxirType(F32, listOf(rows, 4)))
            val y = param("y", DxirType(F32, listOf(rows, 2)))
            val g = inline(shardGrad, listOf(w, x, y)).single()
            listOf(op(OpKind.ALL_REDUCE, listOf(g), g.type, attrs = mapOf("replica_groups" to all)))
        }
        val w = values(8)
        val xs = List(n) { values(rows * 4) }
        val ys = List(n) { values(rows * 2) }
        val out = s.runPerDevice(step, listOf(List(n) { w }, xs, ys)).single()
        val fullGrad = DxirReverseTransform.apply(loss(rows * n), inputOnlyTrailingParams = 2)
        val want = interp(fullGrad, listOf(w, xs.flatMap { it.asIterable() }.toFloatArray(), ys.flatMap { it.asIterable() }.toFloatArray())).single()
        for ((d, got) in out.withIndex()) assertClose(got, want, 1e-5, "all-reduced gradient on device $d")
    }

    @Test
    fun aBatchShardedProgramAndItsGradientMatchOneDevice() = session { s ->
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", n)))
        val byRows = DxirSharding("m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("data"))), DxirDimSharding(emptyList())))
        val primal = DxirBuilder.function("sharded_loss") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 4)), byRows)
            val w = param("w", DxirType(F32, listOf(4, 3)))
            val y = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(8, 3)))), DxirType(F32, listOf(8, 3)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(primal, includeForward = true)
        val x = values(32)
        val w = values(12)
        val got = s.runSharded(grad, listOf(x, w))
        val want = interp(grad, listOf(x, w))
        for (i in want.indices) assertClose(got[i], want[i], 1e-5, "result $i")
    }

    @Test
    fun aTwoAxisMeshShardsBothOperandsOfAMatmul() = session { s ->
        // x [8, 6] rows on "a", w [6, 4] columns on "b": a 2 x 2 mesh.
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("a", 2), DxirMeshAxis("b", 2)))
        val rowsOnA = DxirSharding("m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("a"))), DxirDimSharding(emptyList())))
        val colsOnB = DxirSharding("m", listOf(DxirDimSharding(emptyList()), DxirDimSharding(listOf(DxirAxisRef.Full("b")))))
        val fn = DxirBuilder.function("mm2d") {
            declareMesh(mesh)
            val x = param("x", DxirType(F32, listOf(8, 6)), rowsOnA)
            val w = param("w", DxirType(F32, listOf(6, 4)), colsOnB)
            listOf(op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(8, 4))))
        }
        val x = values(48)
        val w = values(24)
        assertClose(s.runSharded(fn, listOf(x, w)).single(), interp(fn, listOf(x, w)).single(), 1e-5, "matmul")
    }
}
