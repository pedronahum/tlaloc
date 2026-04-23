package io.tlaloc.stablehlo

import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirSharding

/** Module-scope mesh declaration. Example: `sdy.mesh @data_par = <["data"=8, "model"=4]>` */
internal fun DxirMesh.toSdyDecl(): String {
    val axes = axes.joinToString(", ") { "\"${it.name}\"=${it.size}" }
    return "sdy.mesh @$name = <[$axes]>"
}

/**
 * SDY attribute form used by `sdy.sharding_constraint` and the `<@mesh, [...]>` tail of
 * `in_shardings`/`out_shardings` in `sdy.manual_computation`. Example:
 *   `<@mesh, [{"data"}, {"model"?}p1], replicated={"other"}>`
 */
internal fun DxirSharding.toSdyAttr(): String = buildString {
    append("<@$meshName, [")
    append(dimShardings.joinToString(", ") { it.toSdy() })
    append("]")
    if (replicated.isNotEmpty()) {
        append(", replicated={${replicated.joinToString(", ") { it.toSdy() }}}")
    }
    append(">")
}

private fun DxirDimSharding.toSdy(): String {
    val axesPart = axes.joinToString(", ") { it.toSdy() }
    val openMarker = if (!closed) {
        if (axes.isEmpty()) "?" else ", ?"
    } else {
        ""
    }
    val prioritySuffix = priority?.let { "p$it" } ?: ""
    return "{$axesPart$openMarker}$prioritySuffix"
}

internal fun DxirAxisRef.toSdy(): String = when (this) {
    is DxirAxisRef.Full -> "\"$name\""
    is DxirAxisRef.Sub -> "\"$name\":($preSize)$size"
}
