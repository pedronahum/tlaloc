package io.tlaloc.runtime.pjrt.ffm

/**
 * Where each device's piece of an array lives, read from a serialized `xla.OpSharding`
 * (`xla/xla_data.proto`), the form the PJRT shardings extension reports for every
 * parameter and output of a compiled SPMD program.
 *
 * [tileCounts] has one entry per array dimension: how many pieces that dimension is cut
 * into. [tileOf] gives, for each device in the program's device order, its tile index
 * along each dimension. Devices in a replication subgroup share a tile.
 */
class OpShardingLayout internal constructor(
    val tileCounts: List<Int>,
    private val tileOf: List<List<Int>>,
) {
    val deviceCount: Int get() = tileOf.size

    /** True when [other] puts the same tile on every device. */
    internal fun sameAs(other: OpShardingLayout): Boolean = tileCounts == other.tileCounts && tileOf == other.tileOf

    /** True when every device holds the whole array. */
    val isReplicated: Boolean get() = tileCounts.all { it == 1 }

    /** The shape of one device's piece of an array of [dims]. Each dimension must divide evenly. */
    fun shardDims(dims: List<Int>): List<Int> {
        require(dims.size == tileCounts.size) { "array rank ${dims.size} != sharding rank ${tileCounts.size}" }
        return dims.indices.map { d ->
            require(dims[d] % tileCounts[d] == 0) {
                "dimension $d of $dims does not divide into ${tileCounts[d]} pieces (padded shards are not supported)"
            }
            dims[d] / tileCounts[d]
        }
    }

    /** Device [device]'s piece of the row-major array [whole] of [dims]. */
    fun slice(whole: FloatArray, dims: List<Int>, device: Int): FloatArray {
        val shard = shardDims(dims)
        val out = FloatArray(shard.fold(1) { a, b -> a * b })
        forEachElement(dims, shard, tileOf[device]) { local, global -> out[local] = whole[global] }
        return out
    }

    /** The whole row-major array of [dims], from one piece per device in device order. */
    fun assemble(pieces: List<FloatArray>, dims: List<Int>): FloatArray {
        require(pieces.size == deviceCount) { "${pieces.size} pieces for $deviceCount devices" }
        val shard = shardDims(dims)
        val out = FloatArray(dims.fold(1) { a, b -> a * b })
        for ((device, piece) in pieces.withIndex()) {
            forEachElement(dims, shard, tileOf[device]) { local, global -> out[global] = piece[local] }
        }
        return out
    }

    private inline fun forEachElement(dims: List<Int>, shard: List<Int>, tile: List<Int>, f: (Int, Int) -> Unit) {
        val rank = dims.size
        val n = shard.fold(1) { a, b -> a * b }
        val idx = IntArray(rank)
        for (local in 0 until n) {
            var rem = local
            for (d in rank - 1 downTo 0) {
                idx[d] = rem % shard[d]
                rem /= shard[d]
            }
            var global = 0
            for (d in 0 until rank) global = global * dims[d] + tile[d] * shard[d] + idx[d]
            f(local, global)
        }
    }

    companion object {
        private const val REPLICATED = 0
        private const val MAXIMAL = 1
        private const val OTHER = 3

        /**
         * The layout of an array of rank [rank] under the serialized OpSharding [proto], over
         * [deviceCount] devices. REPLICATED, and OTHER (tiled, with `replicate_on_last_tile_dim`
         * or REPLICATED `last_tile_dims`, explicit or iota tile assignment) are read. MAXIMAL,
         * MANUAL, TUPLE and the other subgroup kinds are refused by name.
         */
        fun parse(proto: ByteArray, rank: Int, deviceCount: Int): OpShardingLayout {
            val p = ProtoReader(proto)
            var type = REPLICATED
            val tileDims = mutableListOf<Long>()
            val tileDevices = mutableListOf<Long>()
            var replicateOnLast = false
            val lastTileDims = mutableListOf<Long>()
            val iotaReshape = mutableListOf<Long>()
            val iotaPerm = mutableListOf<Long>()
            while (p.hasMore()) {
                val (field, wire) = p.tag()
                when (field) {
                    1 -> type = p.varint().toInt()
                    3 -> p.repeatedVarint(wire, tileDims)
                    4 -> p.repeatedVarint(wire, tileDevices)
                    6 -> replicateOnLast = p.varint() != 0L
                    8 -> p.repeatedVarint(wire, lastTileDims)
                    9 -> p.repeatedVarint(wire, iotaReshape)
                    10 -> p.repeatedVarint(wire, iotaPerm)
                    else -> p.skip(wire)
                }
            }
            if (type == REPLICATED) return OpShardingLayout(List(rank) { 1 }, List(deviceCount) { List(rank) { 0 } })
            require(type != MAXIMAL) { "OpSharding MAXIMAL (one device holds the array) is not supported" }
            require(type == OTHER) { "OpSharding type $type is not supported (only REPLICATED and tiled)" }
            require(lastTileDims.all { it == REPLICATED.toLong() }) {
                "OpSharding last_tile_dims $lastTileDims: only REPLICATED subgroups are supported"
            }
            val dims = tileDims.map { it.toInt() }
            val subgroups = (if (replicateOnLast) 1 else 0) + lastTileDims.size
            require(dims.size == rank + subgroups) {
                "OpSharding tile assignment $dims does not match an array of rank $rank with $subgroups subgroup dims"
            }
            val total = dims.fold(1) { a, b -> a * b }
            require(total == deviceCount) { "OpSharding tiles $total devices, the program has $deviceCount" }
            val devices: List<Int> = if (tileDevices.isNotEmpty()) {
                tileDevices.map { it.toInt() }
            } else {
                iotaDevices(iotaReshape.map { it.toInt() }, iotaPerm.map { it.toInt() }, total)
            }
            require(devices.size == total) { "OpSharding lists ${devices.size} devices for $total tiles" }
            val tileOf = arrayOfNulls<List<Int>>(deviceCount)
            for ((flat, device) in devices.withIndex()) {
                var rem = flat
                val coords = IntArray(dims.size)
                for (d in dims.indices.reversed()) {
                    coords[d] = rem % dims[d]
                    rem /= dims[d]
                }
                require(device in 0 until deviceCount && tileOf[device] == null) { "OpSharding device $device repeated or out of range" }
                tileOf[device] = coords.take(rank)
            }
            return OpShardingLayout(dims.take(rank), tileOf.map { it!! })
        }

        /** `iota(prod(reshape)).reshape(reshape).transpose(perm).flatten()`. */
        internal fun iotaDevices(reshape: List<Int>, perm: List<Int>, total: Int): List<Int> {
            val shape = reshape.ifEmpty { listOf(total) }
            val order = perm.ifEmpty { shape.indices.toList() }
            val permuted = order.map { shape[it] }
            val strides = IntArray(shape.size).also { s ->
                var acc = 1
                for (d in shape.indices.reversed()) { s[d] = acc; acc *= shape[d] }
            }
            return List(total) { flat ->
                var rem = flat
                var src = 0
                for (k in permuted.indices.reversed()) {
                    val c = rem % permuted[k]
                    rem /= permuted[k]
                    src += c * strides[order[k]]
                }
                src
            }
        }
    }
}

/** The protobuf wire format, enough of it to read an OpSharding. */
internal class ProtoReader(private val bytes: ByteArray) {
    private var pos = 0

    fun hasMore() = pos < bytes.size

    fun varint(): Long {
        var result = 0L
        var shift = 0
        while (true) {
            val b = bytes[pos++].toInt() and 0xff
            result = result or ((b and 0x7f).toLong() shl shift)
            if (b < 0x80) return result
            shift += 7
        }
    }

    fun tag(): Pair<Int, Int> = varint().let { (it ushr 3).toInt() to (it and 7).toInt() }

    fun repeatedVarint(wire: Int, into: MutableList<Long>) {
        if (wire == 2) {
            val length = varint().toInt()
            val end = pos + length
            while (pos < end) into += varint()
        } else {
            into += varint()
        }
    }

    fun skip(wire: Int) {
        when (wire) {
            0 -> varint()
            1 -> pos += 8
            2 -> { val length = varint().toInt(); pos += length }
            5 -> pos += 4
            else -> error("protobuf wire type $wire is not supported")
        }
    }
}
