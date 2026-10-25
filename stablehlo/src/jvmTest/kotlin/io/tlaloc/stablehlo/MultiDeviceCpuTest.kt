package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.testing.CpuDevices
import io.tlaloc.testing.Tolerances
import io.tlaloc.testing.inline
import kotlin.random.Random
import kotlin.test.Test

/**
 * Emitted collectives and sharded programs executed on four devices: XLA:CPU's host
 * platform split in four, through jaxlib ([CpuDevices]; recorded, so this runs where
 * jax is not installed). Each result is checked against what one device computes in
 * the interpreter.
 */
class MultiDeviceCpuTest {

    private val devices = 4
    private val all = listOf((0 until devices).toList())
    private val r = Random(5)

    private fun values(n: Int) = DoubleArray(n) { (r.nextDouble() * 2 - 1).toFloat().toDouble() }

    @Test
    fun allReduceSumsOverFourDevices() {
        val t = DxirType(F32, listOf(5))
        val fn = DxirBuilder.function("ar4") {
            val x = param("x", t)
            listOf(op(OpKind.ALL_REDUCE, listOf(x), t, attrs = mapOf("replica_groups" to all)))
        }
        val xs = List(devices) { values(5) }
        val out = CpuDevices.runPerDevice("multi-device", "all_reduce", fn, fn.toStablehlo(), devices, listOf(xs)).single()
        val want = DoubleArray(5) { i -> xs.sumOf { it[i] } }
        for ((d, got) in out.withIndex()) Tolerances.assertClose(got, want, 1e-6, "device $d")
    }

    @Test
    fun allReduceOverTwoGroups() {
        val t = DxirType(F32, listOf(3))
        val groups = listOf(listOf(0, 2), listOf(1, 3))
        val fn = DxirBuilder.function("ar2x2") {
            val x = param("x", t)
            listOf(op(OpKind.ALL_REDUCE, listOf(x), t, attrs = mapOf("replica_groups" to groups)))
        }
        val xs = List(devices) { values(3) }
        val out = CpuDevices.runPerDevice("multi-device", "all_reduce_groups", fn, fn.toStablehlo(), devices, listOf(xs)).single()
        for (d in 0 until devices) {
            val g = groups.first { d in it }
            Tolerances.assertClose(out[d], DoubleArray(3) { i -> g.sumOf { xs[it][i] } }, 1e-6, "device $d")
        }
    }

    @Test
    fun allGatherThenReduceScatter() {
        val shard = DxirType(F32, listOf(2, 3))
        val whole = DxirType(F32, listOf(2 * devices, 3))
        val fn = DxirBuilder.function("ag_rs4") {
            val x = param("x", shard)
            val g = op(OpKind.ALL_GATHER, listOf(x), whole, attrs = mapOf("replica_groups" to all, "dimension" to 0))
            val s = op(OpKind.REDUCE_SCATTER, listOf(g), shard, attrs = mapOf("replica_groups" to all, "dimension" to 0))
            listOf(g, s)
        }
        val xs = List(devices) { values(6) }
        val (gathered, scattered) = CpuDevices.runPerDevice("multi-device", "all_gather_reduce_scatter", fn, fn.toStablehlo(), devices, listOf(xs))
        val concat = xs.flatMap { it.asIterable() }.toDoubleArray()
        for (d in 0 until devices) {
            Tolerances.assertClose(gathered[d], concat, 0.0 + 1e-12, "all_gather on device $d")
            // Every device gathered the same rows; scattering their sum gives device d's
            // block of devices × the gathered rows.
            Tolerances.assertClose(scattered[d], DoubleArray(6) { i -> devices * concat[d * 6 + i] }, 1e-6, "reduce_scatter on device $d")
        }
    }

    @Test
    fun dataParallelGradientEqualsTheFullBatchGradient() {
        // loss(w; x, y) = Σ (tanh(x·w) − y)² over a batch; each device holds 3 rows.
        val rows = 3
        fun loss(n: Int) = DxirBuilder.function("loss$n") {
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val x = param("x", DxirType(F32, listOf(n, 4)))
            val y = param("y", DxirType(F32, listOf(n, 2)))
            val h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(n, 2)))), DxirType(F32, listOf(n, 2)))
            val d = op(OpKind.SUB, listOf(h, y), DxirType(F32, listOf(n, 2)))
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
        val xs = List(devices) { values(rows * 4) }
        val ys = List(devices) { values(rows * 2) }
        val out = CpuDevices.runPerDevice(
            "multi-device", "data_parallel_gradient", step, step.toStablehlo(), devices,
            listOf(List(devices) { w }, xs, ys),
        ).single()

        val fullGrad = DxirReverseTransform.apply(loss(rows * devices), inputOnlyTrailingParams = 2)
        val want = DxirInterpreter.evalFunction(
            fullGrad,
            listOf(w, xs.flatMap { it.asIterable() }.toDoubleArray(), ys.flatMap { it.asIterable() }.toDoubleArray())
                .map { a -> FloatArray(a.size) { a[it].toFloat() } },
        ).single().let { a -> DoubleArray(a.size) { a[it].toDouble() } }
        for ((d, got) in out.withIndex()) Tolerances.assertClose(got, want, 1e-5, "all-reduced gradient on device $d")
    }

    @Test
    fun aBatchShardedProgramAndItsGradientMatchOneDevice() {
        // x sharded on its rows over a 4-device mesh, w replicated: XLA partitions the
        // matmul and the reduction, and the gradient's sharding follows x's.
        val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", devices)))
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
        val mlir = grad.toStablehlo()
        kotlin.test.assertTrue("sdy.sharding = #sdy.sharding<@m, [{\"data\"}, {}]>" in mlir, "x carries its row sharding:\n$mlir")
        val got = CpuDevices.runGlobal("multi-device", "sharded_gradient", grad, mlir, devices, listOf(x, w))
        val want = DxirInterpreter.evalFunction(grad, listOf(x, w).map { a -> FloatArray(a.size) { a[it].toFloat() } })
        for (i in want.indices) {
            Tolerances.assertClose(got[i], DoubleArray(want[i].size) { want[i][it].toDouble() }, 1e-5, "result $i")
        }
    }
}
