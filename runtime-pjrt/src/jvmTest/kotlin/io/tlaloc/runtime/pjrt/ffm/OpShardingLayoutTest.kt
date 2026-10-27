package io.tlaloc.runtime.pjrt.ffm

import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OpShardingLayoutTest {

    /** An OpSharding written field by field, packed repeated varints as protoc writes them. */
    private fun proto(
        type: Int,
        tileDims: List<Int> = emptyList(),
        devices: List<Int> = emptyList(),
        replicateOnLast: Boolean = false,
        iotaReshape: List<Int> = emptyList(),
        iotaPerm: List<Int> = emptyList(),
    ): ByteArray {
        val out = ByteArrayOutputStream()
        fun varint(v: Long) {
            var x = v
            while (x >= 0x80) { out.write(((x and 0x7f) or 0x80).toInt()); x = x ushr 7 }
            out.write(x.toInt())
        }
        fun packed(field: Int, vs: List<Int>) {
            if (vs.isEmpty()) return
            val body = ByteArrayOutputStream()
            for (v in vs) {
                var x = v.toLong()
                while (x >= 0x80) { body.write(((x and 0x7f) or 0x80).toInt()); x = x ushr 7 }
                body.write(x.toInt())
            }
            varint((field shl 3 or 2).toLong()); varint(body.size().toLong()); out.write(body.toByteArray())
        }
        if (type != 0) { varint(1 shl 3); varint(type.toLong()) }
        packed(3, tileDims)
        packed(4, devices)
        if (replicateOnLast) { varint(6 shl 3); varint(1) }
        packed(9, iotaReshape)
        packed(10, iotaPerm)
        return out.toByteArray()
    }

    private val whole = FloatArray(4 * 6) { it.toFloat() } // [4, 6]
    private val dims = listOf(4, 6)

    @Test
    fun replicatedGivesEveryDeviceTheWholeArray() {
        val l = OpShardingLayout.parse(proto(0), rank = 2, deviceCount = 4)
        assertTrue(l.isReplicated)
        for (d in 0 until 4) assertContentEquals(whole, l.slice(whole, dims, d))
    }

    @Test
    fun rowShardingSplitsTheFirstDimension() {
        val l = OpShardingLayout.parse(proto(3, tileDims = listOf(4, 1), devices = listOf(0, 1, 2, 3)), 2, 4)
        assertEquals(listOf(1, 6), l.shardDims(dims))
        assertContentEquals(FloatArray(6) { 12f + it }, l.slice(whole, dims, 2))
        assertContentEquals(whole, l.assemble(List(4) { l.slice(whole, dims, it) }, dims))
    }

    @Test
    fun twoDimensionalTilesFollowTheDeviceList() {
        // tiles [2, 2], devices listed in reverse: device 3 holds tile (0, 0).
        val l = OpShardingLayout.parse(proto(3, tileDims = listOf(2, 2), devices = listOf(3, 2, 1, 0)), 2, 4)
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 6f, 7f, 8f), l.slice(whole, dims, 3))
        assertContentEquals(floatArrayOf(15f, 16f, 17f, 21f, 22f, 23f), l.slice(whole, dims, 0))
        assertContentEquals(whole, l.assemble(List(4) { l.slice(whole, dims, it) }, dims))
    }

    @Test
    fun replicateOnLastTileDimSharesATileBetweenDevices() {
        // columns in 2, each half on two devices: tile dims [1, 2, 2].
        val l = OpShardingLayout.parse(proto(3, tileDims = listOf(1, 2, 2), devices = listOf(0, 1, 2, 3), replicateOnLast = true), 2, 4)
        assertEquals(listOf(4, 3), l.shardDims(dims))
        assertContentEquals(l.slice(whole, dims, 0), l.slice(whole, dims, 1))
        assertContentEquals(l.slice(whole, dims, 2), l.slice(whole, dims, 3))
        assertContentEquals(whole, l.assemble(List(4) { l.slice(whole, dims, it) }, dims))
    }

    @Test
    fun iotaTileAssignmentWithATranspose() {
        // devices = iota(4).reshape(2, 2).transpose(1, 0) = [0, 2, 1, 3].
        assertEquals(listOf(0, 2, 1, 3), OpShardingLayout.iotaDevices(listOf(2, 2), listOf(1, 0), 4))
        val l = OpShardingLayout.parse(proto(3, tileDims = listOf(4, 1), iotaReshape = listOf(2, 2), iotaPerm = listOf(1, 0)), 2, 4)
        // tile 1 is on device 2.
        assertContentEquals(FloatArray(6) { 6f + it }, l.slice(whole, dims, 2))
    }

    @Test
    fun unsupportedKindsAreRefusedByName() {
        assertTrue("MAXIMAL" in assertFailsWith<IllegalArgumentException> { OpShardingLayout.parse(proto(1), 2, 4) }.message!!)
        assertTrue("type 4" in assertFailsWith<IllegalArgumentException> { OpShardingLayout.parse(proto(4), 2, 4) }.message!!)
        val l = OpShardingLayout.parse(proto(3, tileDims = listOf(4, 1), devices = listOf(0, 1, 2, 3)), 2, 4)
        assertTrue("does not divide" in assertFailsWith<IllegalArgumentException> { l.shardDims(listOf(6, 2)) }.message!!)
    }
}
