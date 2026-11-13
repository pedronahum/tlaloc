package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.parseJson

// The tools' OpenAI-style definitions and a validator for the JSON Schema subset they use
// (type, properties, required, additionalProperties, enum, minimum/maximum, minLength/maxLength,
// pattern, items, minItems/maxItems, anyOf; description, default and examples are annotations).

class ToolDef(val name: String, val description: String, val parameters: Map<String, Any?>) {
    /** `{"type":"function","function":{name, description, parameters}}`, as the endpoint takes it. */
    fun openAi(): Map<String, Any?> = linkedMapOf(
        "type" to "function",
        "function" to linkedMapOf("name" to name, "description" to description, "parameters" to parameters),
    )

    val schema: JsonObject by lazy { parseJson(toJson(parameters)) as JsonObject }
}

private fun obj(vararg props: Pair<String, Map<String, Any?>>, required: List<String> = emptyList(), extra: Map<String, Any?> = emptyMap()): Map<String, Any?> =
    linkedMapOf<String, Any?>("type" to "object", "properties" to linkedMapOf(*props))
        .also { if (required.isNotEmpty()) it["required"] = required }
        .also { it["additionalProperties"] = false }
        .also { it.putAll(extra) }

private fun str(desc: String, vararg more: Pair<String, Any?>) = linkedMapOf<String, Any?>("type" to "string", "description" to desc, *more)
private fun int(desc: String, min: Int, max: Int, default: Int? = null) = linkedMapOf<String, Any?>(
    "type" to "integer", "description" to desc, "minimum" to min, "maximum" to max,
).also { if (default != null) it["default"] = default }

private val ID = str("Restaurant id from search_restaurants: a DOHMH CAMIS number (\"50060692\") or \"osm:node/123\" for a place with no DOHMH permit.", "minLength" to 1, "maxLength" to 64)
private val SINCE = str("Only records on or after this date (YYYY-MM-DD).", "pattern" to "^\\d{4}-\\d{2}-\\d{2}$")

private val NEAR = linkedMapOf<String, Any?>(
    "type" to "object",
    "description" to "A point: lat and lon, or an address or place name in Brooklyn to geocode.",
    "properties" to linkedMapOf(
        "lat" to linkedMapOf("type" to "number", "minimum" to 40.4, "maximum" to 41.0),
        "lon" to linkedMapOf("type" to "number", "minimum" to -74.3, "maximum" to -73.6),
        "address" to str("Street address, intersection or place name.", "minLength" to 2, "maxLength" to 200),
    ),
    "additionalProperties" to false,
    "anyOf" to listOf(mapOf("required" to listOf("lat", "lon")), mapOf("required" to listOf("address"))),
)

val TOOL_DEFS: List<ToolDef> = listOf(
    ToolDef(
        "search_restaurants",
        "Search Brooklyn restaurants, cafes and bars by name or keywords, cuisine, neighborhood, distance, health grade and opening hours. Returns ids for the other tools.",
        obj(
            "query" to str("Words to match in the name, cuisine or address.", "maxLength" to 200),
            "cuisine" to str("Cuisine, e.g. \"thai\", \"pizza\", \"coffee\".", "maxLength" to 60),
            "neighborhood" to str("Brooklyn neighborhood, e.g. \"Park Slope\", \"Bed-Stuy\".", "maxLength" to 80),
            "near" to NEAR,
            "radius_m" to int("Radius around near, in metres.", 50, 10_000, 1_000),
            "min_grade" to linkedMapOf("type" to "string", "enum" to listOf("A", "B", "C"), "description" to "Lowest acceptable DOHMH letter grade; places without a letter grade are excluded."),
            "open_at" to str("Local New York time the place must be open, YYYY-MM-DDTHH:MM, or \"now\" (live mode only).", "pattern" to "^(now|\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2})$"),
            "limit" to int("Maximum results.", 1, 50, 10),
        ),
    ),
    ToolDef(
        "get_restaurant",
        "Details of one restaurant: names, address, cuisine, phone, website, hours, neighborhood, current DOHMH grade and latest inspection, Google rating, and counts of inspections, complaints and reviews.",
        obj("id" to ID, required = listOf("id")),
    ),
    ToolDef(
        "get_inspections",
        "DOHMH inspections of one restaurant, newest first: date, type, score, grade and the violations cited (code, critical flag, description).",
        obj("id" to ID, "since" to SINCE, "limit" to int("Maximum inspections.", 1, 50, 10), required = listOf("id")),
    ),
    ToolDef(
        "get_complaints",
        "NYC 311 food complaints (food poisoning, food establishment) linked to a restaurant, or near a point. Give id or near.",
        obj(
            "id" to ID, "near" to NEAR, "radius_m" to int("Radius around near, in metres.", 10, 2_000, 100),
            "since" to SINCE, "limit" to int("Maximum complaints.", 1, 50, 10),
            extra = mapOf("anyOf" to listOf(mapOf("required" to listOf("id")), mapOf("required" to listOf("near")))),
        ),
    ),
    ToolDef(
        "get_reviews",
        "Google Local reviews of a restaurant (UCSD dataset, up to 2021): average rating and review texts, optionally only those matching a query.",
        obj(
            "id" to ID, "query" to str("Words the review text must contain, e.g. \"brunch\", \"rude\".", "maxLength" to 200),
            "limit" to int("Maximum reviews.", 1, 20, 5), required = listOf("id"),
        ),
    ),
    ToolDef(
        "get_menu",
        "The restaurant's menu as priced lines of text, read from its website or menu page: the Internet Archive capture of the site nearest the data cutoff, with its date.",
        obj("id" to ID, required = listOf("id")),
    ),
    ToolDef(
        "geocode",
        "Coordinates and Brooklyn neighborhood of an address or place name.",
        obj("address" to str("Address, intersection or place name.", "minLength" to 2, "maxLength" to 200), required = listOf("address")),
    ),
    ToolDef(
        "web_search",
        "Web search (title, url, snippet) for facts the other tools do not hold, such as news about a restaurant.",
        obj("query" to str("Search query.", "minLength" to 2, "maxLength" to 300), "limit" to int("Maximum results.", 1, 10, 5), required = listOf("query")),
    ),
)

val TOOL_BY_NAME: Map<String, ToolDef> = TOOL_DEFS.associateBy { it.name }

private val KNOWN_KEYWORDS = setOf(
    "type", "properties", "required", "additionalProperties", "enum", "minimum", "maximum", "minLength",
    "maxLength", "pattern", "items", "minItems", "maxItems", "anyOf", "description", "default", "examples",
)
private val TYPES = setOf("object", "array", "string", "number", "integer", "boolean", "null")

/** Problems with a schema itself: unknown keywords or types, required names that are not properties, bad defaults. */
fun checkSchema(s: JsonValue, path: String = "$"): List<String> {
    if (s !is JsonObject) return listOf("$path: a schema must be an object")
    val out = ArrayList<String>()
    for (k in s.fields.keys) if (k !in KNOWN_KEYWORDS) out += "$path: unsupported keyword '$k'"
    when (val t = s["type"]) {
        null -> {}
        is JsonString -> if (t.value !in TYPES) out += "$path: unknown type '${t.value}'"
        is JsonArray -> t.elements.forEach { if (it !is JsonString || it.value !in TYPES) out += "$path: bad type entry" }
        else -> out += "$path: type must be a string or array"
    }
    val props = s["properties"]
    if (props != null) {
        if (props !is JsonObject) out += "$path: properties must be an object"
        else props.fields.forEach { (k, v) -> out += checkSchema(v, "$path.properties.$k") }
    }
    (s["required"] as? JsonArray)?.elements?.forEach { r ->
        val name = (r as? JsonString)?.value
        if (name == null) out += "$path: required entries must be strings"
        else if (props is JsonObject && name !in props.fields && s["anyOf"] == null) out += "$path: required '$name' is not a property"
    }
    (s["anyOf"] as? JsonArray)?.elements?.forEachIndexed { i, sub ->
        out += checkSchema(sub, "$path.anyOf[$i]")
        ((sub as? JsonObject)?.get("required") as? JsonArray)?.elements?.forEach { r ->
            val name = (r as? JsonString)?.value
            if (props is JsonObject && name !in props.fields) out += "$path.anyOf[$i]: required '$name' is not a property"
        }
    }
    s["items"]?.let { out += checkSchema(it, "$path.items") }
    (s["pattern"] as? JsonString)?.let { p -> runCatching { Regex(p.value) }.onFailure { out += "$path: bad pattern" } }
    s["default"]?.let { d -> validate(s.copyWithout("default"), d, "$path.default").forEach { out += it } }
    return out
}

private fun JsonObject.copyWithout(k: String) = JsonObject(fields - k)

private fun typeOf(v: JsonValue): String = when (v) {
    is JsonObject -> "object"
    is JsonArray -> "array"
    is JsonString -> "string"
    is JsonNumber -> if (v.raw.toLongOrNull() != null || (v.value == Math.rint(v.value) && v.value.isFinite())) "integer" else "number"
    is JsonBool -> "boolean"
    JsonNull -> "null"
}

private fun typeMatches(want: String, v: JsonValue): Boolean {
    val t = typeOf(v)
    return want == t || (want == "number" && t == "integer")
}

/** Errors of [v] against [schema] (empty when valid). Paths are JSONPath-like. */
fun validate(schema: JsonObject, v: JsonValue, path: String = "$"): List<String> {
    val out = ArrayList<String>()
    when (val t = schema["type"]) {
        is JsonString -> if (!typeMatches(t.value, v)) return listOf("$path: expected ${t.value}, got ${typeOf(v)}")
        is JsonArray -> if (t.elements.none { typeMatches((it as JsonString).value, v) }) return listOf("$path: type ${typeOf(v)} not allowed")
        else -> {}
    }
    (schema["enum"] as? JsonArray)?.let { e ->
        if (e.elements.none { canonicalJson(it) == canonicalJson(v) }) out += "$path: not one of ${canonicalJson(e)}"
    }
    if (v is JsonNumber) {
        (schema["minimum"] as? JsonNumber)?.let { if (v.value < it.value) out += "$path: below minimum ${it.raw}" }
        (schema["maximum"] as? JsonNumber)?.let { if (v.value > it.value) out += "$path: above maximum ${it.raw}" }
    }
    if (v is JsonString) {
        val n = v.value.codePointCount(0, v.value.length)
        (schema["minLength"] as? JsonNumber)?.let { if (n < it.value) out += "$path: shorter than ${it.raw}" }
        (schema["maxLength"] as? JsonNumber)?.let { if (n > it.value) out += "$path: longer than ${it.raw}" }
        (schema["pattern"] as? JsonString)?.let { if (!Regex(it.value).containsMatchIn(v.value)) out += "$path: does not match ${it.value}" }
    }
    if (v is JsonArray) {
        (schema["minItems"] as? JsonNumber)?.let { if (v.size < it.value) out += "$path: fewer than ${it.raw} items" }
        (schema["maxItems"] as? JsonNumber)?.let { if (v.size > it.value) out += "$path: more than ${it.raw} items" }
        (schema["items"] as? JsonObject)?.let { s -> v.elements.forEachIndexed { i, x -> out += validate(s, x, "$path[$i]") } }
    }
    if (v is JsonObject) {
        val props = (schema["properties"] as? JsonObject)?.fields.orEmpty()
        (schema["required"] as? JsonArray)?.elements?.forEach { r ->
            val k = (r as JsonString).value
            if (k !in v.fields) out += "$path: missing required '$k'"
        }
        if ((schema["additionalProperties"] as? JsonBool)?.value == false) {
            for (k in v.fields.keys) if (k !in props) out += "$path: unexpected property '$k'"
        }
        for ((k, x) in v.fields) (props[k] as? JsonObject)?.let { out += validate(it, x, "$path.$k") }
    }
    (schema["anyOf"] as? JsonArray)?.let { alts ->
        if (alts.elements.none { validate(it as JsonObject, v, path).isEmpty() }) {
            out += "$path: matches none of anyOf " + alts.elements.joinToString(" | ") { canonicalJson(it) }
        }
    }
    return out
}
