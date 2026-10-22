package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * The emitted `stablehlo.all_reduce` (flattened ids: a channel and `use_global_device_ids`)
 * compiles and runs on one device, under the single-device options and under the multi-device
 * form TorchTPU compiles collectives with ([CompileOptions.spmd]) over that one device. A sum
 * over a group of one is the identity. More than one device is not available here. The same
 * for `stablehlo.all_gather` and `stablehlo.reduce_scatter`.
 */
class PjrtCollectiveCompileTest {

    @Test
    fun allReduceRunsUnderBothCompileForms() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val v = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("ar_one") {
            val x = param("x", v)
            val r = op(OpKind.ALL_REDUCE, listOf(x), v, attrs = mapOf<String, Any>("replica_groups" to listOf(listOf(0))))
            listOf(op(OpKind.ADD, listOf(r, x), v))
        }
        val x = floatArrayOf(1f, -2f, 3.5f, 0f)
        for (opts in listOf(CompileOptions.DEFAULT, CompileOptions.spmd(listOf(0L)))) {
            val out = PjrtSession(target = PjrtTarget.Cuda, compileOptions = opts, diskCache = null).use {
                it.runOn(fn, listOf(x))
            }
            assertContentEquals(floatArrayOf(2f, -4f, 7f, 0f), out[0], "$opts")
        }
    }

    @Test
    fun allGatherAndReduceScatterRunUnderBothCompileForms() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val v = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("ag_rs_one") {
            val x = param("x", v)
            val one = mapOf<String, Any>("replica_groups" to listOf(listOf(0)), "dimension" to 1)
            val g = op(OpKind.ALL_GATHER, listOf(x), v, attrs = one)
            val s = op(OpKind.REDUCE_SCATTER, listOf(g), v, attrs = mapOf<String, Any>("dimension" to 0))
            listOf(op(OpKind.ADD, listOf(s, x), v))
        }
        val text = fn.toStablehlo()
        assertTrue("stablehlo.all_gather" in text && "stablehlo.reduce_scatter" in text, "both collectives are emitted")
        val x = floatArrayOf(1f, -2f, 3.5f, 0f, 5f, 6f)
        val want = FloatArray(6) { 2 * x[it] }
        assertContentEquals(want, io.tlaloc.ir.passes.DxirInterpreter.evalFunction(fn, listOf(x))[0], "interpreter")
        for (opts in listOf(CompileOptions.DEFAULT, CompileOptions.spmd(listOf(0L)))) {
            val out = PjrtSession(target = PjrtTarget.Cuda, compileOptions = opts, diskCache = null).use {
                it.runOn(fn, listOf(x))
            }
            assertContentEquals(want, out[0], "$opts")
        }
    }
}
