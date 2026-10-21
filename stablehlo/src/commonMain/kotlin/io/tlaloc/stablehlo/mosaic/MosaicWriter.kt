package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * Writes a Mosaic module in MLIR's generic op form, either as Pallas hands it to Mosaic
 * ([serialized] false) or as `mosaic-serde{serialize=true}` leaves it ([serialized] true):
 * every op name prefixed with `stable_mosaic.` and the module carrying
 * `stable_mosaic.version`. libtpu reads the serialized form as MLIR text as well as bytecode
 * (TorchTPU's `mosaic_custom_call.h`), so a serialized module needs no jaxlib.
 *
 * Ops write their inherent attributes in full (`fastmath = #arith.fastmath<none>`, a store's
 * `add` and `operandSegmentSizes`), as serde does, so the two forms differ only in names and
 * the module attribute. `harness/python/check_kmosaic_forms.py` checks both: the
 * unserialized module against Pallas's (equal after `cse`), and the serialized one against
 * serde's output for it.
 */
@ExperimentalTlalocApi
class MosaicWriter(val serialized: Boolean, val version: Int = 9) {

    /** An SSA value and its type. */
    data class V(val name: String, val type: String)

    private val sb = StringBuilder()
    private var next = 0
    private var indent = "    "

    private fun opName(n: String) = if (serialized) "stable_mosaic.$n" else n

    /** Appends one op; returns its results. [attrs] are written in the order given. */
    fun op(
        name: String,
        operands: List<V>,
        resultTypes: List<String>,
        attrs: List<Pair<String, String>> = emptyList(),
        regions: List<MosaicWriter.() -> Unit> = emptyList(),
    ): List<V> {
        val results = resultTypes.map { V("%${next++}", it) }
        sb.append(indent)
        if (results.isNotEmpty()) sb.append(results.joinToString(", ") { it.name }).append(" = ")
        sb.append('"').append(opName(name)).append("\"(").append(operands.joinToString(", ") { it.name }).append(')')
        if (regions.isNotEmpty()) {
            sb.append(" (")
            regions.forEachIndexed { i, body ->
                if (i > 0) sb.append(", ")
                sb.append("{\n")
                val saved = indent
                indent = "$saved  "
                body()
                indent = saved
                sb.append(indent).append('}')
            }
            sb.append(')')
        }
        if (attrs.isNotEmpty()) sb.append(" {").append(attrs.joinToString(", ") { "${it.first} = ${it.second}" }).append('}')
        sb.append(" : (").append(operands.joinToString(", ") { it.type }).append(") -> ")
        sb.append(if (resultTypes.size == 1) resultTypes[0] else "(${resultTypes.joinToString(", ")})")
        sb.append('\n')
        return results
    }

    fun op1(name: String, operands: List<V>, resultType: String, attrs: List<Pair<String, String>> = emptyList()): V =
        op(name, operands, listOf(resultType), attrs)[0]

    // The ops the Tlaloc kernels use.
    fun indexConst(v: Int) = op1("arith.constant", emptyList(), "index", listOf("value" to "$v : index"))
    fun i32Const(v: Int) = op1("arith.constant", emptyList(), "i32", listOf("value" to "$v : i32"))
    fun f32Const(bits: String, type: String = "f32") = op1("arith.constant", emptyList(), type, listOf("value" to bits))
    fun float(name: String, a: V, b: V? = null) =
        op1(name, listOfNotNull(a, b), a.type, listOf("fastmath" to "#arith.fastmath<none>"))
    fun cmpi(predicate: Int, a: V, b: V) = op1("arith.cmpi", listOf(a, b), "i1", listOf("predicate" to "$predicate : i64"))
    fun extui(a: V, to: String) = op1("arith.extui", listOf(a), to)
    fun cast(name: String, a: V, to: String) = op1(name, listOf(a), to)
    fun broadcast(a: V, to: String) = op1("vector.broadcast", listOf(a), to)
    fun load(ref: V, indices: List<V>, vector: String) = op1("vector.load", listOf(ref) + indices, vector)
    fun store(value: V, ref: V, indices: List<V>) {
        op(
            "tpu.vector_store", listOf(value, ref) + indices, emptyList(),
            listOf("add" to "false", "operandSegmentSizes" to "array<i32: 1, 1, ${indices.size}, 0>", "strides" to "array<i32>"),
        )
    }
    fun matmul(a: V, b: V, acc: V, dimensionNumbers: String) =
        op1(
            "tpu.matmul", listOf(a, b, acc), acc.type,
            listOf("dimension_numbers" to dimensionNumbers, "transpose_lhs" to "false", "transpose_rhs" to "false"),
        )
    fun ifThen(cond: V, then: MosaicWriter.() -> Unit) {
        // Pallas's `pl.when` leaves an else block that only yields.
        op(
            "scf.if", listOf(cond), emptyList(),
            regions = listOf({ then(); op("scf.yield", emptyList(), emptyList()) }, { op("scf.yield", emptyList(), emptyList()) }),
        )
    }
    fun ret(values: List<V> = emptyList()) { op("func.return", values, emptyList()) }

    /** A function: [args] are its argument types; [attrs] its attributes besides function_type and sym_name. */
    fun function(name: String, args: List<String>, results: List<String>, attrs: List<Pair<String, String>>, body: MosaicWriter.(List<V>) -> Unit) {
        val params = args.map { V("%${next++}", it) }
        sb.append("  \"").append(opName("func.func")).append("\"() ({\n")
        sb.append("  ^bb0(").append(params.joinToString(", ") { "${it.name}: ${it.type}" }).append("):\n")
        val saved = indent
        indent = "    "
        body(params)
        indent = saved
        val fnType = "(${args.joinToString(", ")}) -> " + (if (results.size == 1) results[0] else "(${results.joinToString(", ")})")
        val all = (attrs + listOf("function_type" to fnType, "sym_name" to "\"$name\"")).sortedBy { it.first }
        sb.append("  }) {").append(all.joinToString(", ") { "${it.first} = ${it.second}" }).append("} : () -> ()\n")
    }

    /** The module text. */
    fun module(): String = buildString {
        append(if (serialized) "module attributes {stable_mosaic.version = $version : i64} {\n" else "module {\n")
        append(sb)
        append("}\n")
    }

    companion object {
        /** `arith.cmpi` predicates. */
        const val EQ = 0
        const val NE = 1
    }
}
