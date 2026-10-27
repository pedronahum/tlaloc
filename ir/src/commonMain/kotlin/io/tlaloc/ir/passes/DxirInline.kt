package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBlock
import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirRegion

/**
 * Copies a function's body into another function: [fn]'s parameters are bound positionally to
 * [args], every op and constant (region blocks included) is copied with a fresh id from [nextId],
 * and the copies are appended to [out] in [fn]'s order. Returns the copies of [fn]'s returns.
 *
 * Calling it twice on the same function gives two independent copies, so a program can hold
 * several instances of one step (gradient accumulation over microbatches).
 */
object DxirInline {

    fun inline(fn: DxirFunction, args: List<DxirNode>, nextId: () -> Int, out: MutableList<DxirNode>): List<DxirNode> {
        require(args.size == fn.params.size) {
            "DxirInline: '${fn.name}' takes ${fn.params.size} parameters, got ${args.size} arguments"
        }
        val map = HashMap<Int, DxirNode>()
        for ((p, a) in fn.params.zip(args)) map[p.id] = a
        for (node in fn.body) out += copy(node, map, nextId)
        return fn.returns.map { mapped(it, map) }
    }

    /** The largest node id in [fn], region blocks included. */
    fun maxId(fn: DxirFunction): Int {
        var max = -1
        fun visit(n: DxirNode) {
            if (n.id > max) max = n.id
            if (n is DxirOp) {
                for (r in n.regions) for (b in r.blocks) {
                    b.args.forEach(::visit)
                    b.body.forEach(::visit)
                }
            }
        }
        fn.params.forEach(::visit)
        fn.body.forEach(::visit)
        return max
    }

    private fun mapped(n: DxirNode, map: Map<Int, DxirNode>): DxirNode = when (n) {
        is DxirOpResult -> {
            val source = map[n.source.id] ?: error("DxirInline: %${n.source.id} is used before it is defined")
            (source as DxirOp).result(n.index)
        }
        else -> map[n.id] ?: error("DxirInline: %${n.id} is used before it is defined")
    }

    private fun copy(node: DxirNode, map: HashMap<Int, DxirNode>, nextId: () -> Int): DxirNode {
        val c = when (node) {
            is DxirConst -> DxirConst(nextId(), node.value, node.type, node.sharding)
            is DxirOp -> DxirOp(
                nextId(), node.op, node.operands.map { mapped(it, map) }, node.attrs, node.types, node.sharding,
                node.regions.map { r -> DxirRegion(r.blocks.map { b -> copyBlock(b, map, nextId) }) },
            )
            is DxirParam, is DxirBlockArg, is DxirOpResult -> error("DxirInline: %${node.id} cannot appear in a body")
            else -> error("DxirInline: unsupported node ${node::class.simpleName} %${node.id}")
        }
        map[node.id] = c
        return c
    }

    private fun copyBlock(b: DxirBlock, map: HashMap<Int, DxirNode>, nextId: () -> Int): DxirBlock {
        val args = b.args.map { a -> DxirBlockArg(nextId(), a.type, a.sharding).also { map[a.id] = it } }
        val body = b.body.map { copy(it, map, nextId) }
        return DxirBlock(args, body, b.terminator.map { mapped(it, map) })
    }
}
