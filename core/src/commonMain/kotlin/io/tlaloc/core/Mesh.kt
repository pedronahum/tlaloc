package io.tlaloc.core

data class MeshAxis(val name: String, val size: Int) {
    init {
        require(name.isNotEmpty()) { "mesh axis name must be non-empty" }
        require(size > 0) { "mesh axis size must be positive: $size" }
    }
}

/**
 * Runtime device-mesh specification — the *value-level* description of a
 * cluster topology (name + axes + sizes). Used by sharding utilities and
 * the IR's `DxirMesh` / `DxirSharding` layer.
 *
 * Layer 2 (§0.4.243+) introduces a parallel **phantom-typed** [Mesh]
 * sealed interface for compile-time tracking of where a [BufferHandle]
 * lives. The two are intentionally separate: [MeshSpec] is the runtime
 * value; [Mesh] is the type-level placement marker. Renamed from `Mesh`
 * → `MeshSpec` to free up the bare name for the phantom-typed marker.
 */
class MeshSpec(val name: String, val axes: List<MeshAxis>) {

    init {
        require(name.isNotEmpty()) { "mesh name must be non-empty" }
        val dupes = axes.groupingBy { it.name }.eachCount().filter { it.value > 1 }
        require(dupes.isEmpty()) { "duplicate mesh axis names: ${dupes.keys}" }
    }

    fun axis(name: String): MeshAxis? = axes.firstOrNull { it.name == name }

    fun size(name: String): Int =
        axis(name)?.size ?: error("mesh '${this.name}' has no axis '$name'")

    fun totalSize(): Int = axes.fold(1) { acc, a -> acc * a.size }

    override fun toString(): String = "MeshSpec($name, [${axes.joinToString { "${it.name}=${it.size}" }}])"

    companion object {
        fun of(vararg axes: Pair<String, Int>, name: String = "default"): MeshSpec =
            MeshSpec(name, axes.map { MeshAxis(it.first, it.second) })
    }
}

sealed class Spec {
    data object Replicated : Spec() {
        override fun toString(): String = "_"
    }

    data class On(
        val axes: List<String>,
        val open: Boolean = false,
        val priority: Int? = null,
    ) : Spec() {
        init {
            require(axes.isNotEmpty()) { "On-spec must reference at least one axis" }
            require(priority == null || priority >= 0) { "priority must be >= 0" }
        }

        override fun toString(): String {
            val axesStr = axes.joinToString(",")
            val openStr = if (open) "?" else ""
            val prioStr = priority?.let { "#p$it" } ?: ""
            return "{$axesStr}$openStr$prioStr"
        }

        companion object {
            fun of(vararg axes: String, open: Boolean = false, priority: Int? = null): On =
                On(axes.toList(), open, priority)
        }
    }

    data class SubOn(val axis: String, val preSize: Int, val size: Int) : Spec() {
        init {
            require(preSize > 0) { "preSize must be positive" }
            require(size > 0) { "size must be positive" }
        }

        override fun toString(): String = "{$axis($preSize,$size)}"
    }
}

typealias PartitionSpec = List<Spec>

fun partitionSpec(vararg specs: Spec): PartitionSpec = specs.toList()

fun MeshSpec.validate(spec: PartitionSpec, tensorDims: IntArray) {
    require(spec.size == tensorDims.size) {
        "PartitionSpec length ${spec.size} does not match tensor rank ${tensorDims.size}"
    }
    val axisSizes = axes.associate { it.name to it.size }
    spec.forEachIndexed { dim, s ->
        when (s) {
            Spec.Replicated -> {}
            is Spec.On -> {
                val product = s.axes.fold(1) { acc, a ->
                    val sz = axisSizes[a]
                        ?: error("mesh '${this.name}' has no axis '$a' (referenced on dim $dim)")
                    acc * sz
                }
                require(tensorDims[dim] % product == 0) {
                    "dim $dim size ${tensorDims[dim]} not divisible by shard product $product"
                }
            }
            is Spec.SubOn -> {
                val full = axisSizes[s.axis]
                    ?: error("mesh '${this.name}' has no axis '${s.axis}' (referenced on dim $dim)")
                require(s.preSize * s.size <= full) {
                    "sub-axis ${s.axis}($s.preSize,$s.size) exceeds axis size $full"
                }
                require(tensorDims[dim] % s.size == 0) {
                    "dim $dim size ${tensorDims[dim]} not divisible by sub-axis size ${s.size}"
                }
            }
        }
    }
}
