package io.tlaloc.ir

import io.tlaloc.core.MeshSpec
import io.tlaloc.core.PartitionSpec
import io.tlaloc.core.Spec

data class DxirMeshAxis(val name: String, val size: Int)

data class DxirMesh(val name: String, val axes: List<DxirMeshAxis>) {
    fun axis(name: String): DxirMeshAxis? = axes.firstOrNull { it.name == name }
}

sealed class DxirAxisRef {
    abstract val name: String

    data class Full(override val name: String) : DxirAxisRef() {
        override fun toString(): String = name
    }

    data class Sub(override val name: String, val preSize: Int, val size: Int) : DxirAxisRef() {
        override fun toString(): String = "$name($preSize,$size)"
    }
}

data class DxirDimSharding(
    val axes: List<DxirAxisRef>,
    val closed: Boolean = true,
    val priority: Int? = null,
) {
    override fun toString(): String {
        val axesStr = axes.joinToString(",")
        val openMark = if (!closed) "?" else ""
        val prioStr = priority?.let { "#p$it" } ?: ""
        return "{$axesStr}$openMark$prioStr"
    }
}

data class DxirSharding(
    val meshName: String,
    val dimShardings: List<DxirDimSharding>,
    val replicated: List<DxirAxisRef> = emptyList(),
) {
    override fun toString(): String {
        val dims = dimShardings.joinToString(",")
        val rep = if (replicated.isEmpty()) "" else " rep=[${replicated.joinToString(",")}]"
        return "@$meshName[$dims]$rep"
    }
}

fun MeshSpec.toDxir(): DxirMesh = DxirMesh(name, axes.map { DxirMeshAxis(it.name, it.size) })

fun PartitionSpec.toDxir(meshName: String): DxirSharding = DxirSharding(
    meshName = meshName,
    dimShardings = map { s ->
        when (s) {
            Spec.Replicated -> DxirDimSharding(axes = emptyList(), closed = true)
            is Spec.On -> DxirDimSharding(
                axes = s.axes.map { DxirAxisRef.Full(it) },
                closed = !s.open,
                priority = s.priority,
            )
            is Spec.SubOn -> DxirDimSharding(
                axes = listOf(DxirAxisRef.Sub(s.axis, s.preSize, s.size)),
                closed = true,
            )
        }
    },
)
