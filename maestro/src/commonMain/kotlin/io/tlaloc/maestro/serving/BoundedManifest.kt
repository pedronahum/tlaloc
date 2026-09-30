package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson

/**
 * The manifest of a **bounded-program artifact** (`tlaloc-bounded.json`): one StableHLO body per
 * bucket of a program whose inputs have bounded axes (docs/design/bounded-dims.md).
 *
 * ```
 *   <artifact>/
 *     tlaloc-bounded.json          this manifest
 *     bodies/<bodyHash>.mlir       one StableHLO module per bucket, entry `main`
 *     programs/<entryId>.json      one ProgramManifest per bucket
 * ```
 *
 * A runtime picks, for each bound, the smallest bucket that holds the request's size; pads
 * every `DATA` input with [paddingValue] along its bounded axes; fills each `VALID_MASK` input
 * (`[bucket]`, 1 for a real position) and `VALID_LENGTH` input (scalar, the real size); runs the
 * entry of those buckets; and slices the output's bounded axes back to the real sizes.
 *
 * It is a separate file and schema from `tlaloc-serving.json`, whose readers are unchanged.
 * Unlike those readers, [fromJson] refuses unknown keys, so a field added later cannot be
 * skipped by an old reader without a version bump.
 */
data class BoundedManifest(
    val name: String,
    val bounds: List<BoundDecl>,
    val inputs: List<TensorDecl>,
    val outputs: List<TensorDecl>,
    val entries: List<BoundedEntry>,
    val paddingCheck: PaddingCheck,
    val paddingValue: Double = 0.0,
    val schemaVersion: String = SCHEMA_VERSION,
) {
    init {
        require(schemaVersion == SCHEMA_VERSION) {
            "BoundedManifest: schemaVersion '$schemaVersion' is not one this reader knows ($SCHEMA_VERSION); " +
                "refusing it rather than reading the fields it recognises"
        }
        require(name.isNotBlank()) { "BoundedManifest: name must not be blank" }
        require(bounds.isNotEmpty()) { "BoundedManifest: no bounds" }
        val boundNames = bounds.map { it.name }
        require(boundNames.toSet().size == boundNames.size) { "BoundedManifest: bound names repeat: $boundNames" }
        for (t in inputs + outputs) {
            for (a in t.axes) {
                val b = a.bound ?: continue
                require(b in boundNames) { "BoundedManifest: ${t.name} uses bound '$b', which is not declared" }
            }
            if (t.bound != null) require(t.bound in boundNames) { "BoundedManifest: ${t.name} names bound '${t.bound}', which is not declared" }
        }
        require(inputs.any { it.role == ROLE_DATA }) { "BoundedManifest: no DATA input" }
        require(outputs.isNotEmpty() && outputs.all { it.role == ROLE_DATA }) {
            "BoundedManifest: outputs must be DATA tensors"
        }
        for (t in inputs) {
            when (t.role) {
                ROLE_DATA -> require(t.bound == null) { "BoundedManifest: DATA input ${t.name} names a bound" }
                ROLE_VALID_MASK -> require(t.bound != null && t.dtype == "f32" && t.axes == listOf(AxisDecl(bound = t.bound))) {
                    "BoundedManifest: VALID_MASK input ${t.name} must be f32 [bound] of the bound it names"
                }
                ROLE_VALID_LENGTH -> require(t.bound != null && t.dtype == "f32" && t.axes.isEmpty()) {
                    "BoundedManifest: VALID_LENGTH input ${t.name} must be an f32 scalar naming its bound"
                }
            }
        }
        // Every combination of buckets has exactly one entry, so a size within the bounds always
        // has somewhere to run.
        val want = bounds.fold(listOf(emptyMap<String, Int>())) { acc, b -> acc.flatMap { m -> b.buckets.map { m + (b.name to it) } } }
        val got = entries.map { it.sizes }
        require(got.toSet().size == got.size) { "BoundedManifest: two entries share sizes" }
        require(got.toSet() == want.toSet()) {
            "BoundedManifest: entries cover ${got.size} bucket combinations; the bucket lists give ${want.size} " +
                "(missing ${want.toSet() - got.toSet()}, extra ${got.toSet() - want.toSet()})"
        }
        for (e in entries) {
            require(insideArtifact(e.bodyPath) && insideArtifact(e.programPath)) {
                "BoundedManifest: entry ${e.id}'s paths (${e.bodyPath}, ${e.programPath}) must be relative paths inside the artifact"
            }
        }
        require(paddingCheck.maxDifference <= paddingCheck.tolerance) {
            "BoundedManifest: the recorded padding check failed (${paddingCheck.maxDifference} > ${paddingCheck.tolerance})"
        }
    }

    /** The bound declaration named [name]. */
    fun bound(name: String): BoundDecl = bounds.firstOrNull { it.name == name }
        ?: throw IllegalArgumentException("BoundedManifest '${this.name}': no bound '$name'")

    /** The entry for exact bucket sizes. */
    fun entryFor(sizes: Map<String, Int>): BoundedEntry = entries.firstOrNull { it.sizes == sizes }
        ?: throw IllegalArgumentException("BoundedManifest '$name': no entry for $sizes")

    /** The smallest bucket of each bound that holds the given real sizes. */
    fun bucketsFor(sizes: Map<String, Int>): Map<String, Int> {
        require(sizes.keys == bounds.map { it.name }.toSet()) {
            "BoundedManifest '$name': sizes for ${sizes.keys} given, the bounds are ${bounds.map { it.name }}"
        }
        return bounds.associate { b ->
            val n = sizes.getValue(b.name)
            require(n in 1..b.max) { "BoundedManifest '$name': ${b.name} = $n is outside 1..${b.max}" }
            b.name to b.buckets.first { it >= n }
        }
    }

    fun toJson(): String = buildString {
        append("{")
        append("\"schemaVersion\":").append(jsonStr(schemaVersion)).append(',')
        append("\"name\":").append(jsonStr(name)).append(',')
        append("\"bounds\":").append(bounds.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"padding\":{\"value\":").append(num(paddingValue)).append("},")
        append("\"inputs\":").append(inputs.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"outputs\":").append(outputs.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"entries\":").append(entries.joinToString(",", "[", "]") { it.toJson() }).append(',')
        append("\"paddingCheck\":").append(paddingCheck.toJson())
        append("}")
    }

    companion object {
        const val SCHEMA_VERSION: String = "tlaloc-bounded-v1"
        const val FILE_NAME: String = "tlaloc-bounded.json"
        const val ROLE_DATA = "DATA"
        const val ROLE_VALID_MASK = "VALID_MASK"
        const val ROLE_VALID_LENGTH = "VALID_LENGTH"
        val ROLES = setOf(ROLE_DATA, ROLE_VALID_MASK, ROLE_VALID_LENGTH)

        fun fromJson(text: String): BoundedManifest {
            val o = parseJson(text) as? JsonObject ?: throw JsonException("BoundedManifest: top level is not a JSON object")
            // The version first, so an artifact of another version is refused for that and not
            // for a key it may legitimately have.
            val version = o.str("schemaVersion")
            if (version != SCHEMA_VERSION) {
                throw JsonException(
                    "BoundedManifest: schemaVersion '$version' is not one this reader knows ($SCHEMA_VERSION)",
                )
            }
            o.onlyKeys("manifest", "schemaVersion", "name", "bounds", "padding", "inputs", "outputs", "entries", "paddingCheck")
            val padding = o.obj("padding").also { it.onlyKeys("padding", "value") }
            return BoundedManifest(
                schemaVersion = version,
                name = o.str("name"),
                bounds = o.arr("bounds").objects("bounds").map { BoundDecl.fromJson(it) },
                paddingValue = padding.num("value"),
                inputs = o.arr("inputs").objects("inputs").map { TensorDecl.fromJson(it) },
                outputs = o.arr("outputs").objects("outputs").map { TensorDecl.fromJson(it) },
                entries = o.arr("entries").objects("entries").map { BoundedEntry.fromJson(it) },
                paddingCheck = PaddingCheck.fromJson(o.obj("paddingCheck")),
            )
        }
    }
}

/** A bound: its name (the `DimBound` object's simple name), its max, and its buckets. */
data class BoundDecl(val name: String, val max: Int, val buckets: List<Int>) {
    init {
        require(name.isNotBlank()) { "BoundDecl: blank name" }
        require(max >= 1) { "BoundDecl $name: max $max below 1" }
        require(buckets.isNotEmpty() && buckets.first() >= 1 && buckets.zipWithNext().all { (a, b) -> a < b }) {
            "BoundDecl $name: buckets $buckets are not strictly ascending from at least 1"
        }
        require(buckets.last() == max) { "BoundDecl $name: the last bucket ${buckets.last()} is not the max $max" }
    }

    fun toJson(): String = "{\"name\":${jsonStr(name)},\"max\":$max,\"buckets\":${buckets.joinToString(",", "[", "]")}}"

    companion object {
        fun fromJson(o: JsonObject): BoundDecl {
            o.onlyKeys("bound", "name", "max", "buckets")
            return BoundDecl(o.str("name"), o.int("max"), o.arr("buckets").asIntList("buckets"))
        }
    }
}

/** One axis: a fixed [size], or a [bound] name. Exactly one is set. */
data class AxisDecl(val size: Int? = null, val bound: String? = null) {
    init {
        require((size == null) != (bound == null)) { "AxisDecl: exactly one of size and bound, got size=$size bound=$bound" }
        require(size == null || size >= 1) { "AxisDecl: size $size below 1" }
    }

    fun toJson(): String = if (size != null) "{\"size\":$size}" else "{\"bound\":${jsonStr(bound!!)}}"

    companion object {
        fun fromJson(o: JsonObject): AxisDecl {
            o.onlyKeys("axis", "size", "bound")
            return when {
                o["size"] != null && o["bound"] == null -> AxisDecl(size = o.int("size"))
                o["bound"] != null && o["size"] == null -> AxisDecl(bound = o.str("bound"))
                else -> throw JsonException("axis: exactly one of 'size' and 'bound' is required, got ${o.fields.keys}")
            }
        }
    }
}

/** An input or output: name, role, dtype, axes, and for a mask or length the bound it describes. */
data class TensorDecl(
    val name: String,
    val role: String,
    val dtype: String,
    val axes: List<AxisDecl>,
    val bound: String? = null,
) {
    init {
        require(role in BoundedManifest.ROLES) { "TensorDecl $name: unknown role '$role'; known ${BoundedManifest.ROLES}" }
        require(dtype in setOf("f32", "i32")) { "TensorDecl $name: dtype '$dtype' is not f32 or i32" }
    }

    /** The dims when each bound has the given size. */
    fun dimsAt(sizes: Map<String, Int>): List<Int> = axes.map { it.size ?: sizes.getValue(it.bound!!) }

    fun toJson(): String = buildString {
        append("{\"name\":").append(jsonStr(name))
        append(",\"role\":").append(jsonStr(role))
        append(",\"dtype\":").append(jsonStr(dtype))
        append(",\"axes\":").append(axes.joinToString(",", "[", "]") { it.toJson() })
        if (bound != null) append(",\"bound\":").append(jsonStr(bound))
        append("}")
    }

    companion object {
        fun fromJson(o: JsonObject): TensorDecl {
            o.onlyKeys("tensor", "name", "role", "dtype", "axes", "bound")
            val role = o.str("role")
            if (role !in BoundedManifest.ROLES) {
                throw JsonException("tensor '${o.str("name")}': unknown role '$role'; known ${BoundedManifest.ROLES}")
            }
            return TensorDecl(
                name = o.str("name"),
                role = role,
                dtype = o.str("dtype"),
                axes = o.arr("axes").objects("axes").map { AxisDecl.fromJson(it) },
                bound = (o["bound"] as? JsonString)?.value,
            )
        }
    }
}

/** One bucket combination's program. */
data class BoundedEntry(
    val id: String,
    val sizes: Map<String, Int>,
    val bodyPath: String,
    val bodyHash: String,
    val programPath: String,
    val entryPoint: String = "main",
) {
    fun toJson(): String = buildString {
        append("{\"id\":").append(jsonStr(id))
        append(",\"sizes\":").append(sizes.entries.joinToString(",", "{", "}") { "${jsonStr(it.key)}:${it.value}" })
        append(",\"entryPoint\":").append(jsonStr(entryPoint))
        append(",\"bodyPath\":").append(jsonStr(bodyPath))
        append(",\"bodyHash\":").append(jsonStr(bodyHash))
        append(",\"programPath\":").append(jsonStr(programPath))
        append("}")
    }

    companion object {
        fun fromJson(o: JsonObject): BoundedEntry {
            o.onlyKeys("entry", "id", "sizes", "entryPoint", "bodyPath", "bodyHash", "programPath")
            return BoundedEntry(
                id = o.str("id"),
                sizes = o.obj("sizes").intMap("sizes"),
                entryPoint = o.str("entryPoint"),
                bodyPath = o.str("bodyPath"),
                bodyHash = o.str("bodyHash"),
                programPath = o.str("programPath"),
            )
        }
    }
}

/** What the exporter checked: padded against exact results at [sizes], in the interpreter. */
data class PaddingCheck(val sizes: List<Map<String, Int>>, val maxDifference: Double, val tolerance: Double) {
    fun toJson(): String =
        "{\"sizes\":${sizes.joinToString(",", "[", "]") { m -> m.entries.joinToString(",", "{", "}") { "${jsonStr(it.key)}:${it.value}" } }}," +
            "\"maxDifference\":${num(maxDifference)},\"tolerance\":${num(tolerance)}}"

    companion object {
        fun fromJson(o: JsonObject): PaddingCheck {
            o.onlyKeys("paddingCheck", "sizes", "maxDifference", "tolerance")
            return PaddingCheck(
                sizes = o.arr("sizes").objects("paddingCheck.sizes").map { it.intMap("paddingCheck.sizes") },
                maxDifference = o.num("maxDifference"),
                tolerance = o.num("tolerance"),
            )
        }
    }
}

// --- helpers ------------------------------------------------------------

/** A relative path with no `..` component: every file of an artifact is inside it. */
private fun insideArtifact(path: String): Boolean =
    path.isNotEmpty() && !path.startsWith("/") && path.split('/').none { it == ".." }

private fun num(v: Double): String {
    require(v.isFinite()) { "BoundedManifest: $v cannot be written as JSON" }
    return v.toString()
}

private fun JsonObject.onlyKeys(what: String, vararg allowed: String) {
    val extra = fields.keys - allowed.toSet()
    if (extra.isNotEmpty()) {
        throw JsonException("BoundedManifest: $what has unknown key(s) $extra; this reader knows ${allowed.toList()}")
    }
}

private fun JsonObject.int(key: String): Int =
    (this[key] as? JsonNumber)?.asInt(key) ?: throw JsonException("field '$key' is not a number")

private fun JsonObject.num(key: String): Double =
    (this[key] as? JsonNumber)?.value ?: throw JsonException("field '$key' is not a number")

private fun JsonObject.intMap(what: String): Map<String, Int> =
    fields.entries.associate { (k, v) -> k to ((v as? JsonNumber)?.asInt("$what.$k") ?: throw JsonException("$what.$k is not a number")) }

private fun JsonArray.objects(what: String): List<JsonObject> =
    elements.mapIndexed { i, e -> e as? JsonObject ?: throw JsonException("$what[$i] is not an object") }
