package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ShardedCheckpointTest {

    private val rows = 6
    private val cols = 8
    private val w = FloatArray(rows * cols) { it * 0.5f - 3f }
    private val bias = FloatArray(cols) { it.toFloat() }

    private fun f32(name: String, data: FloatArray, dims: IntArray) = SafetensorsTensor(name, F32, dims, HostF32Storage(data))

    /** The block of [w] at rows [r0, r0 + nr) and columns [c0, c0 + nc). */
    private fun block(r0: Int, nr: Int, c0: Int, nc: Int) = FloatArray(nr * nc) { w[(r0 + it / nc) * cols + c0 + it % nc] }

    /** w tiled 2 × 2 over four ranks (blocks of 3 × 4); the bias replicated on every rank. */
    private fun save(dir: Path, ranks: List<Int> = listOf(0, 1, 2, 3)) {
        for (rank in ranks) {
            val (r0, c0) = (rank / 2) * 3 to (rank % 2) * 4
            ShardedCheckpoint.writeShard(
                dir, rank,
                listOf(
                    ShardedCheckpoint.ShardTensor(f32("w", block(r0, 3, c0, 4), intArrayOf(3, 4)), listOf(r0, c0), listOf(rows, cols)),
                    ShardedCheckpoint.ShardTensor(f32("bias", bias.copyOf(), intArrayOf(cols)), listOf(0), listOf(cols)),
                ),
            )
        }
    }

    @Test
    fun aTiledTensorLoadsWholeOrUnderAnotherLayout(@TempDir dir: Path) {
        save(dir)
        ShardedCheckpoint.open(dir).use { ck ->
            assertEquals(setOf("w", "bias"), ck.names)
            assertEquals(listOf(rows, cols), ck.globalDims("w"))
            assertContentEquals(w, (ck.load("w").storage as HostF32Storage).data)
            assertContentEquals(bias, (ck.load("bias").storage as HostF32Storage).data)
            // Column halves, as a 2-device column layout would read them: each spans two shards.
            for (c0 in listOf(0, 4)) {
                val half = ck.load("w", listOf(0, c0), listOf(rows, 4))
                assertContentEquals(intArrayOf(rows, 4), half.dims)
                assertContentEquals(block(0, rows, c0, 4), (half.storage as HostF32Storage).data)
            }
            // A block across all four shards.
            assertContentEquals(block(1, 4, 2, 5), (ck.load("w", listOf(1, 2), listOf(4, 5)).storage as HostF32Storage).data)
        }
    }

    @Test
    fun bf16BlocksRoundTripBitForBit(@TempDir dir: Path) {
        val bits = ShortArray(4 * 6) { (0x3F80 + it).toShort() }
        for (rank in 0..1) {
            val part = bits.copyOfRange(rank * 12, rank * 12 + 12)
            ShardedCheckpoint.writeShard(
                dir, rank,
                listOf(ShardedCheckpoint.ShardTensor(SafetensorsTensor("e", BF16, intArrayOf(2, 6), HostBf16Storage(part)), listOf(rank * 2, 0), listOf(4, 6))),
            )
        }
        ShardedCheckpoint.open(dir).use { ck ->
            val e = ck.load("e")
            assertEquals(BF16, e.dtype)
            assertContentEquals(bits, (e.storage as HostBf16Storage).data)
        }
    }

    @Test
    fun aRegionNoShardHoldsIsRefused(@TempDir dir: Path) {
        save(dir, ranks = listOf(0, 1, 2))
        ShardedCheckpoint.open(dir).use { ck ->
            // Rows 0..2 are complete; rows 3..5, columns 4..7 (rank 3) are not written.
            assertContentEquals(block(0, 3, 0, cols), (ck.load("w", listOf(0, 0), listOf(3, cols)).storage as HostF32Storage).data)
            val e = assertFailsWith<IllegalArgumentException> { ck.load("w") }
            assertTrue("no shard holds" in e.message!!, e.message)
        }
    }
}
