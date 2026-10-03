package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * An e4m3fn KV pool in StableHLO: the write clamps the f32 tokens to ±448 and
 * converts them before the scatter; attention gathers codes and widens the
 * gathered window, not the pool.
 */
class Fp8KvCacheEmitTest {

    private val poolType = DxirType(F8E4M3FN, listOf(4, 2, 1, 3))
    private val qType = DxirType(F32, listOf(2, 1, 3))

    private val mlir = DxirBuilder.function("fp8Kv") {
        val kc = param("kc", poolType)
        val vc = param("vc", poolType)
        val newK = param("newK", DxirType(F32, listOf(2, 1, 3)))
        val slots = param("slots", DxirType(I32, listOf(2)))
        val q = param("q", qType)
        val table = param("table", DxirType(I32, listOf(2, 2)))
        val lens = param("lens", DxirType(I32, listOf(2)))
        val kc2 = op(OpKind.KV_CACHE_WRITE, listOf(kc, newK, slots), poolType)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, kc2, vc, table, lens), qType, mapOf("scale" to 1.0)))
    }.toStablehlo()

    @Test
    fun theWriteClampsAndConvertsBeforeTheScatter() {
        assertTrue("stablehlo.constant dense<-448.0> : tensor<f32>" in mlir, mlir)
        assertTrue(Regex("""stablehlo\.clamp %\w+, %\w+, %\w+ : tensor<2x1x3xf32>""").containsMatchIn(mlir), mlir)
        assertTrue(
            Regex("""stablehlo\.convert %\w+ : \(tensor<2x1x3xf32>\) -> tensor<2x1x3xf8E4M3FN>""").containsMatchIn(mlir),
            mlir,
        )
        assertTrue(Regex("""-> tensor<8x1x3xf8E4M3FN>\n""").containsMatchIn(mlir), "the scatter stays in the pool's dtype")
    }

    @Test
    fun attentionWidensTheGatheredWindow() {
        // Two windows (keys and values), each gathered as codes and then widened.
        val widen = Regex("""stablehlo\.convert %\w+ : \(tensor<2x4x1x3xf8E4M3FN>\) -> tensor<2x4x1x3xf32>""")
        assertTrue(widen.findAll(mlir).count() == 2, mlir)
        assertTrue("(tensor<4x2x1x3xf8E4M3FN>) -> tensor<4x2x1x3xf32>" !in mlir, "the pool itself is never widened")
    }
}
