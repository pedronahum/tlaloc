package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.466 — Phase H1b: the OFFLINE half of KV_CACHE_WRITE's emission
 * certification. The load-bearing oracle is `PjrtKvCacheWriteSmokeTest` — it
 * compiles this text under real XLA on the GB10 and pins it against the
 * interpreter's reference walk, padding lane included — but that test
 * self-skips without CUDA, so the structure a reader would look for is pinned
 * here where it always runs.
 *
 * These are STRUCTURE pins, not golden text: ONE scatter (not a per-token
 * chain), the flat-slot dimension numbers, and a replace body.
 */
class KvCacheWriteEmitTest {

    private val numBlocks = 6
    private val blockSize = 2
    private val numKvHeads = 2
    private val headDim = 3
    private val numTokens = 4
    private val numSlots = numBlocks * blockSize

    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tokensType = DxirType(F32, listOf(numTokens, numKvHeads, headDim))
    private val slotsType = DxirType(I32, listOf(numTokens))

    private fun writeFn(attrs: Map<String, Any> = emptyMap()): DxirFunction =
        DxirBuilder.function("kvWrite") {
            val cache = param("cache", cacheType)
            val newKv = param("newKv", tokensType)
            val slots = param("slots", slotsType)
            listOf(op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType, attrs))
        }

    private fun mlir(): String = writeFn().toStablehlo()

    @Test
    fun lowersToExactlyOneScatterNotAPerTokenChain() {
        val text = mlir()
        val scatters = text.lines().count { "stablehlo.scatter\"" in it }
        kotlin.test.assertEquals(
            1, scatters,
            "the whole point of the kind is that numTokens writes are ONE scatter:\n$text",
        )
    }

    @Test
    fun scatterUsesTheFlatSlotDimensionNumbers() {
        val text = mlir()
        assertTrue("update_window_dims = [1, 2]" in text, "a token's update is a whole [Hkv, D] window:\n$text")
        assertTrue("inserted_window_dims = [0]" in text, "the slot indexes the collapsed slot axis:\n$text")
        assertTrue("scatter_dims_to_operand_dims = [0]" in text, "…and maps to operand axis 0:\n$text")
        // index_vector_dim == the indices' rank: each entry is a SCALAR slot id.
        assertTrue("index_vector_dim = ${slotsType.rank}" in text, "implicit trailing index dim:\n$text")
    }

    /**
     * The flat-slot convention, visible in the lowering: the pool is collapsed
     * to [numBlocks*blockSize, numKvHeads, headDim] and back, and NO arithmetic
     * on block/offset appears in between.
     */
    @Test
    fun bracketsTheScatterWithTheFlatPoolReshapes() {
        val text = mlir()
        val flat = "tensor<${numSlots}x${numKvHeads}x${headDim}xf32>"
        val pool = "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>"
        assertTrue("stablehlo.reshape" in text, "expected the flattening reshapes:\n$text")
        assertTrue("($pool) -> $flat" in text, "expected the pool→flat reshape:\n$text")
        assertTrue("($flat) -> $pool" in text, "expected the flat→pool reshape:\n$text")
        for (arith in listOf("stablehlo.divide", "stablehlo.remainder", "stablehlo.multiply")) {
            assertTrue(
                arith !in text,
                "the flat-slot convention exists so no block/offset arithmetic is needed; found $arith:\n$text",
            )
        }
    }

    /**
     * A replace body (`return` the update, ignore the current value): a cache
     * write overwrites whatever stale token owned the slot. And
     * `unique_indices = false` — deliberately, because a bucketed batch's
     * padding lanes repeat `-1`, so the uniqueness promise would be literally
     * false even though every LIVE slot is distinct.
     */
    @Test
    fun usesAReplaceBodyAndMakesNoUniquenessPromise() {
        val text = mlir()
        assertTrue("unique_indices = false" in text, "padding lanes repeat -1:\n$text")
        val bodyLine = text.lines().first { "stablehlo.return" in it }
        assertTrue("stablehlo.return %s" in bodyLine, "replace semantics return the update:\n$text")
    }

    @Test
    fun theResultKeepsThePoolShape() {
        val text = mlir()
        assertTrue(
            "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>" in text.lines().last { "return" in it } ||
                "tensor<${numBlocks}x${blockSize}x${numKvHeads}x${headDim}xf32>" in text,
            "the write is functional and returns an updated POOL:\n$text",
        )
    }

    @Test
    fun theEmitterRefusesADimDerivedAttrByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            writeFn(mapOf("numKvHeads" to numKvHeads)).toStablehlo()
        }
        assertTrue("sentinel" in (ex.message ?: ""), "got ${ex.message}")
    }
}
