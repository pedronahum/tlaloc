package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.core.floatToF8e4m3fn
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * A decode step over e4m3fn KV pools on the device against the interpreter:
 * the step writes one token per sequence (values past ±448 and between codes
 * among them) into pools that already hold a context, then attends over it.
 * Both sides store the same rounded values, so the outputs differ only by the
 * order of f32 sums.
 */
class PjrtFp8KvCacheTest {

    @Test
    fun fp8PoolsDecodeAsTheInterpreterDoes() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val seqs = 3; val heads = 4; val kvHeads = 2; val dim = 8
        val blocks = 16; val bs = 4; val maxBlocks = 4
        val poolType = DxirType(F8E4M3FN, listOf(blocks, bs, kvHeads, dim))
        val tokType = DxirType(F32, listOf(seqs, kvHeads, dim))
        val qType = DxirType(F32, listOf(seqs, heads, dim))
        val fn = DxirBuilder.function("fp8_kv_step") {
            val kc = param("kc", poolType)
            val vc = param("vc", poolType)
            val newK = param("newK", tokType)
            val newV = param("newV", tokType)
            val slots = param("slots", DxirType(I32, listOf(seqs)))
            val q = param("q", qType)
            val table = param("table", DxirType(I32, listOf(seqs, maxBlocks)))
            val lens = param("lens", DxirType(I32, listOf(seqs)))
            val kc2 = op(OpKind.KV_CACHE_WRITE, listOf(kc, newK, slots), poolType)
            val vc2 = op(OpKind.KV_CACHE_WRITE, listOf(vc, newV, slots), poolType)
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, kc2, vc2, table, lens), qType, mapOf("scale" to 0.35)))
        }
        val rnd = java.util.Random(7)
        val poolSize = blocks * bs * kvHeads * dim
        val kCodes = ByteArray(poolSize) { floatToF8e4m3fn((rnd.nextGaussian() * 2).toFloat()) }
        val vCodes = ByteArray(poolSize) { floatToF8e4m3fn((rnd.nextGaussian() * 2).toFloat()) }
        val newK = FloatArray(seqs * kvHeads * dim) { (rnd.nextGaussian() * 3).toFloat() }
        val newV = FloatArray(seqs * kvHeads * dim) { (rnd.nextGaussian() * 3).toFloat() }
        newV[0] = 1000f; newV[5] = -600f; newK[3] = 449.5f
        val q = FloatArray(seqs * heads * dim) { (rnd.nextGaussian()).toFloat() }
        // Sequence s owns pages 1 + 4s .. 4 + 4s and has lens[s] positions after the write.
        val table = IntArray(seqs * maxBlocks) { 1 + it }
        val lens = intArrayOf(5, 11, 16)
        val slots = IntArray(seqs) { s -> table[s * maxBlocks + (lens[s] - 1) / bs] * bs + (lens[s] - 1) % bs }
        fun f(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val want = DxirInterpreter.evalFunction(
            fn,
            listOf(
                FloatArray(poolSize) { f8e4m3fnToFloat(kCodes[it]) }, FloatArray(poolSize) { f8e4m3fnToFloat(vCodes[it]) },
                newK, newV, f(slots), q, f(table), f(lens),
            ),
        )[0]
        TestBackend.session().use { s ->
            val got = s.runOnHost(fn, listOf(kCodes, vCodes, newK, newV, slots, q, table, lens)).single() as FloatArray
            var worst = 0.0
            for (i in want.indices) worst = maxOf(worst, abs((got[i] - want[i]).toDouble()) / maxOf(1.0, abs(want[i].toDouble())))
            println("[pjrt-fp8-kv] fp8 pools: largest difference from the interpreter $worst on ${TestBackend.target}")
            assertTrue(worst < 1e-4, "fp8 KV decode step differs from the interpreter by $worst")
        }
    }
}
