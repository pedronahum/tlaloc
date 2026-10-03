package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.core.I8
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

/**
 * An e4m3fn KV pool: KV_CACHE_WRITE takes f32 keys and values, clamps them to
 * ±448 and rounds them to nearest even; PAGED_ATTENTION reads the stored
 * values back. Expected values are worked by hand.
 */
class Fp8KvCacheTest {

    private val blocks = 4
    private val bs = 2
    private val dim = 3
    private val poolType = DxirType(F8E4M3FN, listOf(blocks, bs, 1, dim))
    private val tokType = DxirType(F32, listOf(2, 1, dim))
    private val slotType = DxirType(I32, listOf(2))

    @Test
    fun aWriteStoresClampedRoundedValuesThatAttentionReadsBack() {
        val qType = DxirType(F32, listOf(2, 1, dim))
        val fn = DxirBuilder.function("fp8Kv") {
            val kc = param("kc", poolType)
            val vc = param("vc", poolType)
            val newK = param("newK", tokType)
            val newV = param("newV", tokType)
            val slots = param("slots", slotType)
            val q = param("q", qType)
            val table = param("table", DxirType(I32, listOf(2, 2)))
            val lens = param("lens", DxirType(I32, listOf(2)))
            val kc2 = op(OpKind.KV_CACHE_WRITE, listOf(kc, newK, slots), poolType)
            val vc2 = op(OpKind.KV_CACHE_WRITE, listOf(vc, newV, slots), poolType)
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, kc2, vc2, table, lens), qType, mapOf("scale" to 1.0)))
        }
        val pool = FloatArray(blocks * bs * dim)
        // 0.3 -> 1.25 * 2^-2, 0.1 -> 1.625 * 2^-4, -2.6 -> -2.5, beyond ±448 -> ±448.
        val newV = floatArrayOf(0.3f, -1000f, 7f, 0.1f, 500f, -2.6f)
        val want = floatArrayOf(0.3125f, -448f, 7f, 0.1015625f, 448f, -2.5f)
        val got = DxirInterpreter.evalFunction(
            fn,
            listOf(
                pool, pool, FloatArray(6) { 0.5f }, newV, floatArrayOf(2f, 4f), FloatArray(6) { 1f },
                floatArrayOf(1f, 0f, 2f, 0f), floatArrayOf(1f, 1f),
            ),
        )[0]
        // One live position: the softmax weight is exactly 1, so attention returns the stored V.
        assertContentEquals(want, got)
    }

    @Test
    fun onlyAWiderFloatConvertsIntoAnFp8Pool() {
        fun write(cache: DxirType, newKv: DxirType) = DxirBuilder.function("w") {
            val c = param("c", cache)
            val n = param("n", newKv)
            val s = param("s", slotType)
            listOf(op(OpKind.KV_CACHE_WRITE, listOf(c, n, s), cache))
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                write(poolType, DxirType(I8, tokType.dims)),
                listOf(FloatArray(24), FloatArray(6), floatArrayOf(0f, 1f)),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                write(DxirType(F32, poolType.dims), DxirType(F8E4M3FN, tokType.dims)),
                listOf(FloatArray(24), FloatArray(6), floatArrayOf(0f, 1f)),
            )
        }
    }
}
