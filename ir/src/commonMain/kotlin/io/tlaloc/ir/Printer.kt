package io.tlaloc.ir

fun DxirModule.pretty(): String = buildString {
    appendLine("module {")
    meshes.forEach { m -> appendLine("  " + m.prettyLine()) }
    functions.forEach { fn ->
        append(fn.pretty(indent = "  "))
    }
    append("}")
}

fun DxirFunction.pretty(indent: String = ""): String = buildString {
    val paramList = params.joinToString(", ") { p ->
        val s = p.sharding?.let { " $it" }.orEmpty()
        "%${p.id}: ${p.type}$s"
    }
    val returnTypes = returns.joinToString(", ") { it.type.toString() }
    appendLine("${indent}fn $name($paramList) -> $returnTypes {")
    val step = "$indent  "
    meshes.forEach { appendLine(step + it.prettyLine()) }
    body.forEach { appendLine(step + it.prettyLine()) }
    val returnIds = returns.joinToString(", ") { nodeRef(it) }
    appendLine("${step}return $returnIds")
    appendLine("${indent}}")
}

private fun nodeRef(node: DxirNode): String = when (node) {
    is DxirOpResult -> "%${node.source.id}#${node.index}"
    else -> "%${node.id}"
}

private fun DxirMesh.prettyLine(): String =
    "mesh @$name = [${axes.joinToString(",") { "${it.name}=${it.size}" }}]"

private fun DxirNode.prettyLine(indent: String = ""): String {
    val tail = sharding?.let { " $it" }.orEmpty()
    return when (this) {
        is DxirParam -> "%$id = param $name : $type$tail"
        is DxirConst -> "%$id = const $value : $type$tail"
        is DxirOp -> {
            val args = operands.joinToString(", ") { nodeRef(it) }
            val attrStr = if (attrs.isEmpty()) "" else
                " {${attrs.entries.joinToString(", ") { "${it.key}=${it.value}" }}}"
            val typeStr = if (isMultiResult) {
                "(${types.joinToString(", ") { it.toString() }})"
            } else {
                type.toString()
            }
            val lhs = if (isMultiResult) "%$id:$numResults" else "%$id"
            val header = "$lhs = ${op.name.lowercase()}($args)$attrStr : $typeStr$tail"
            if (regions.isEmpty()) {
                header
            } else {
                buildString {
                    appendLine("$header {")
                    regions.forEachIndexed { i, region ->
                        if (i > 0) appendLine()
                        append(prettyRegion(region, "$indent  "))
                    }
                    append("$indent}")
                }
            }
        }
        is DxirCall -> {
            val argList = args.joinToString(", ") { nodeRef(it) }
            "%$id = call ${callee.name}($argList) : $type$tail"
        }
        is DxirOpResult -> error("DxirOpResult should not appear directly in body")
        is DxirBlockArg -> error("DxirBlockArg should not appear directly at function body level")
    }
}

private fun prettyRegion(region: DxirRegion, indent: String): String = buildString {
    region.blocks.forEach { block ->
        val argList = block.args.joinToString(", ") { "%${it.id}: ${it.type}" }
        appendLine("${indent}block ($argList) {")
        val step = "$indent  "
        block.body.forEach { appendLine(step + it.prettyLine(step)) }
        val y = block.terminator.joinToString(", ") { nodeRef(it) }
        appendLine("${step}yield $y")
        appendLine("$indent}")
    }
}
