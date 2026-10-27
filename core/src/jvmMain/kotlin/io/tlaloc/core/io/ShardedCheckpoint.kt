package io.tlaloc.core.io

import java.nio.file.Files
import java.nio.file.Path

/**
 * A checkpoint of sharded tensors: each rank writes the blocks it holds to its own safetensors
 * file, and a reader assembles any block of any tensor from the files that hold it, reading only
 * those bytes. A tensor saved under one layout (rows over 4 devices) can be loaded under another
 * (columns over 2), or whole.
 *
 * A shard file is an ordinary safetensors file named `shard-<rank>.safetensors`. Each tensor in
 * it is the block at [Block.offsets] of a tensor of [Block.globalDims]; the file's metadata
 * records both under `tlaloc.block.<name>` as `offsets;globalDims` (comma-separated). Blocks of
 * one tensor may overlap (replicas); a reader takes each element from the first shard, in rank
 * order, that holds it, and refuses a request that some element of no shard covers.
 */
object ShardedCheckpoint {

    /** Where a shard's tensor sits in the whole tensor. */
    data class Block(val offsets: List<Int>, val globalDims: List<Int>)

    /** One tensor a rank holds: its local block ([tensor]) and where the block sits. */
    class ShardTensor(val tensor: SafetensorsTensor, val offsets: List<Int>, val globalDims: List<Int>) {
        init {
            val d = tensor.dims.toList()
            require(offsets.size == d.size && globalDims.size == d.size && d.indices.all { offsets[it] >= 0 && offsets[it] + d[it] <= globalDims[it] }) {
                "ShardedCheckpoint: block ${tensor.name} ${d} at $offsets does not fit $globalDims"
            }
        }
    }

    private const val PREFIX = "tlaloc.block."

    fun shardFile(dir: Path, rank: Int): Path = dir.resolve("shard-$rank.safetensors")

    /** Writes [rank]'s blocks to `dir/shard-<rank>.safetensors`. */
    fun writeShard(dir: Path, rank: Int, tensors: List<ShardTensor>): Path {
        require(rank >= 0) { "ShardedCheckpoint: rank must be >= 0, got $rank" }
        val metadata = tensors.associate {
            PREFIX + it.tensor.name to it.offsets.joinToString(",") + ";" + it.globalDims.joinToString(",")
        }
        return SafetensorsFileWriter.write(shardFile(dir, rank), tensors.map { it.tensor }, metadata)
    }

    /** The shards in [dir], in rank order. Close it after use. */
    fun open(dir: Path): Reader {
        val files = Files.list(dir).use { s ->
            s.filter { Regex("shard-(\\d+)\\.safetensors").matches(it.fileName.toString()) }.toList()
        }.sortedBy { it.fileName.toString().removePrefix("shard-").removeSuffix(".safetensors").toInt() }
        require(files.isNotEmpty()) { "ShardedCheckpoint: no shard-<rank>.safetensors in $dir" }
        return Reader(files.map { SafetensorsFile.open(it) })
    }

    class Reader internal constructor(private val shards: List<SafetensorsFile>) : AutoCloseable {

        /** Each tensor's blocks: the shard index and the block. */
        private val blocks: Map<String, List<Pair<Int, Block>>> = buildMap<String, MutableList<Pair<Int, Block>>> {
            for ((i, f) in shards.withIndex()) for (name in f.names) {
                val meta = f.metadata[PREFIX + name] ?: throw JsonException("ShardedCheckpoint: ${f.path} has no block for '$name'")
                val (off, glob) = meta.split(";").map { part -> if (part.isEmpty()) emptyList() else part.split(",").map { it.toInt() } }
                getOrPut(name) { ArrayList() } += i to Block(off, glob)
            }
        }

        val names: Set<String> get() = blocks.keys

        /** The whole tensor's dims. */
        fun globalDims(name: String): List<Int> = blocksOf(name).first().second.globalDims

        private fun blocksOf(name: String) = blocks[name] ?: throw JsonException("ShardedCheckpoint: no tensor '$name' (have ${blocks.keys})")

        /** The whole tensor. */
        fun load(name: String): LoadedTensor = load(name, List(globalDims(name).size) { 0 }, globalDims(name))

        /** The block of [name] at [offsets] with [extents], read from the shards that hold it. */
        fun load(name: String, offsets: List<Int>, extents: List<Int>): LoadedTensor {
            val parts = blocksOf(name)
            val glob = parts.first().second.globalDims
            val rank = glob.size
            require(offsets.size == rank && extents.size == rank && (0 until rank).all { offsets[it] >= 0 && extents[it] >= 1 && offsets[it] + extents[it] <= glob[it] }) {
                "ShardedCheckpoint: block $offsets + $extents is outside '$name' $glob"
            }
            val first = shards[parts.first().first].entry(name)
            val width = (first.byteLength / maxOf(1L, first.elementCount)).toInt()
            val count = extents.fold(1L) { a, d -> a * d }
            val out = ByteArray(Math.toIntExact(count * width))
            val filled = BooleanArray(count.toInt())
            val outStrides = strides(extents)
            for ((s, block) in parts) {
                val entry = shards[s].entry(name)
                require(entry.wireDType == first.wireDType && block.globalDims == glob) { "ShardedCheckpoint: the blocks of '$name' disagree on dtype or shape" }
                val local = entry.dims
                // The intersection of this block with the request, in global coordinates.
                val lo = IntArray(rank) { maxOf(offsets[it], block.offsets[it]) }
                val hi = IntArray(rank) { minOf(offsets[it] + extents[it], block.offsets[it] + local[it]) }
                if ((0 until rank).any { lo[it] >= hi[it] }) continue
                val localStrides = strides(local)
                // One read per run along the last axis.
                val outer = (0 until rank - 1).map { hi[it] - lo[it] }
                val run = if (rank == 0) 1 else hi[rank - 1] - lo[rank - 1]
                val total = outer.fold(1) { a, b -> a * b }
                val idx = IntArray(maxOf(0, rank - 1))
                val buf = ByteArray(run * width)
                for (r in 0 until total) {
                    var rem = r
                    for (k in rank - 2 downTo 0) { idx[k] = lo[k] + rem % outer[k]; rem /= outer[k] }
                    var src = 0L
                    var dst = 0L
                    for (k in 0 until rank - 1) {
                        src += (idx[k] - block.offsets[k]).toLong() * localStrides[k]
                        dst += (idx[k] - offsets[k]).toLong() * outStrides[k]
                    }
                    if (rank > 0) {
                        src += (lo[rank - 1] - block.offsets[rank - 1]).toLong()
                        dst += (lo[rank - 1] - offsets[rank - 1]).toLong()
                    }
                    // Skip elements an earlier shard already gave.
                    if ((0 until run).all { filled[(dst + it).toInt()] }) continue
                    shards[s].readBytes(name, src * width, buf)
                    for (j in 0 until run) {
                        val at = (dst + j).toInt()
                        if (filled[at]) continue
                        System.arraycopy(buf, j * width, out, at * width, width)
                        filled[at] = true
                    }
                }
            }
            val missing = filled.indexOfFirst { !it }
            require(missing < 0) { "ShardedCheckpoint: no shard holds element $missing of block $offsets + $extents of '$name'" }
            return Safetensors.decode(SafetensorsEntry(name, first.wireDType, extents, 0, out.size.toLong()), out)
        }

        override fun close() = shards.forEach { it.close() }

        private fun strides(dims: List<Int>): LongArray {
            val s = LongArray(dims.size)
            var acc = 1L
            for (k in dims.indices.reversed()) { s[k] = acc; acc *= dims[k] }
            return s
        }
    }
}
