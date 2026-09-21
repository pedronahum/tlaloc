package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.466 — Phase H1b: KV_CACHE_WRITE's single-scatter StableHLO emission
 * compiles under real XLA on the GB10 and agrees with the interpreter's
 * reference walk.
 *
 * This is the emission's load-bearing oracle, and it carries one thing the
 * offline structure pins cannot: the PADDING LANE. The interpreter skips a
 * negative slot explicitly (`slot < 0 → continue`); the emission instead
 * relies on StableHLO's rule that a scatter whose target index is out of
 * bounds has its update IGNORED. Those are two entirely different mechanisms
 * arriving at the same pool, so running them against each other is the only
 * honest way to know the spec rule holds on this backend — and if a future
 * XLA ever clamped instead of dropping, the padding token would land in slot 0
 * and this test would say so loudly.
 *
 * `slotMapping` rides as an I32 CONST rather than a param because
 * [PjrtSession.runOn] is an all-F32 param lane (§0.4.354's per-lane single
 * dtype). Harness limit, not a shape limit — H3's serving path feeds slot
 * mappings as device buffers through the manifest.
 *
 * Tolerance: EXACT. A cache write performs no arithmetic — every output
 * element is a copy of an input element — so the usual GPU-vs-host float floor
 * does not apply and a mismatch of any size would mean a value went somewhere
 * it was not sent.
 */
class PjrtKvCacheWriteSmokeTest {

    private val numBlocks = 6
    private val blockSize = 2
    private val numKvHeads = 2
    private val headDim = 4
    private val numTokens = 4

    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tokensType = DxirType(F32, listOf(numTokens, numKvHeads, headDim))

    // Token 2 is PADDING (-1). The live slots are deliberately unsorted and
    // land two tokens in the SAME BLOCK (slots 6 and 7 are block 3), with
    // slot 11 the very last slot of the pool.
    private val slotMapping = intArrayOf(7, 6, -1, 11)

    private fun writeFn(): DxirFunction = DxirBuilder.function("kv_write_gpu") {
        val cache = param("cache", cacheType)
        val newKv = param("newKv", tokensType)
        val slots = const(
            FloatArray(slotMapping.size) { slotMapping[it].toFloat() },
            DxirType(I32, listOf(numTokens)),
        )
        listOf(op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType))
    }

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    @Test
    fun kvCacheWriteEmissionMatchesInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val cache = pseudo(numBlocks * blockSize * numKvHeads * headDim, 211)
        val newKv = pseudo(numTokens * numKvHeads * headDim, 223)
        val fn = writeFn()
        val want = DxirInterpreter.evalFunction(fn, listOf(cache, newKv))[0]

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(fn, listOf(cache, newKv)).single()
            kotlin.test.assertEquals(want.size, got.size, "updated pool size")
            for (i in want.indices) {
                assertTrue(
                    want[i] == got[i],
                    "kv cache write lane $i: GPU ${got[i]} vs interpreter ${want[i]} — a write " +
                        "performs no arithmetic, so any difference means a value moved",
                )
            }
            // The padding lane, stated rather than merely covered: token 2's
            // slot is -1, so slot 0 of the pool must still hold its ORIGINAL
            // value. A clamping backend would have put token 2 there.
            val slot0 = 0
            assertTrue(
                got[slot0] == cache[slot0],
                "the -1 padding token must write NOTHING: slot 0 lane 0 was ${cache[slot0]} " +
                    "and is now ${got[slot0]} — the backend clamped an out-of-bounds scatter " +
                    "index instead of dropping the update",
            )
            println(
                "[pjrt-kvwrite] single-scatter KV_CACHE_WRITE emission agrees EXACTLY with the " +
                    "interpreter on GB10 (unsorted slots, two tokens in one block, -1 padding lane " +
                    "dropped by the out-of-bounds rule)",
            )
        }
    }
}
