package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson

// Parsers for the four sources. Each takes the raw text a source returns and produces plain
// records; nothing here touches the network or the database.

/** An OpenStreetMap node, way or relation with a food amenity tag. */
data class OsmPlace(
    val osmId: String, // "node/123", "way/456"
    val name: String?,
    val amenity: String,
    val cuisine: String?,
    val lat: Double,
    val lon: Double,
    val houseNumber: String?,
    val street: String?,
    val postcode: String?,
    val phone: String?,
    val website: String?,
    val menuUrl: String?,
    val openingHours: String?,
    val tags: Map<String, String>,
)

/** One row of DOHMH Restaurant Inspection Results (one violation of one inspection). */
data class InspectionRow(
    val camis: String,
    val dba: String?,
    val building: String?,
    val street: String?,
    val zipcode: String?,
    val phone: String?,
    val cuisine: String?,
    val inspectionDate: String?, // yyyy-MM-dd; 1900-01-01 means not yet inspected
    val action: String?,
    val violationCode: String?,
    val violationDescription: String?,
    val criticalFlag: String?,
    val score: Int?,
    val grade: String?,
    val gradeDate: String?,
    val inspectionType: String?,
    val lat: Double?,
    val lon: Double?,
    val bbl: String?,
    val nta: String?,
)

/** One NYC 311 service request. */
data class Complaint(
    val uniqueKey: String,
    val createdDate: String?,
    val closedDate: String?,
    val complaintType: String?,
    val descriptor: String?,
    val descriptor2: String?,
    val locationType: String?,
    val incidentAddress: String?,
    val incidentZip: String?,
    val bbl: String?,
    val status: String?,
    val resolution: String?,
    val lat: Double?,
    val lon: Double?,
)

/** A Google Local business from the UCSD metadata file. */
data class GPlace(
    val gmapId: String,
    val name: String,
    val address: String?,
    val lat: Double,
    val lon: Double,
    val categories: List<String>,
    val avgRating: Double?,
    val numReviews: Int?,
    val price: String?,
    val state: String?,
    val description: String?,
    val hours: String?,
)

/** A Google Local review. Reviewer names and ids are not kept. */
data class GReview(val gmapId: String, val timeMs: Long, val rating: Int?, val text: String?, val response: String?)

internal fun JsonObject.s(k: String): String? = when (val v = this[k]) {
    is JsonString -> v.value.takeIf { it.isNotBlank() }
    is JsonNumber -> v.raw
    else -> null
}

internal fun JsonObject.d(k: String): Double? = when (val v = this[k]) {
    is JsonNumber -> v.value
    is JsonString -> v.value.toDoubleOrNull()
    else -> null
}

/** Coordinates of 0 or outside a broad NYC box are treated as missing. */
internal fun validNyc(lat: Double?, lon: Double?): Boolean =
    lat != null && lon != null && lat in 40.4..41.0 && lon in -74.3..-73.6

/** Parses an Overpass `out center tags` JSON response. Elements without a usable coordinate are dropped. */
fun parseOverpass(text: String): List<OsmPlace> {
    val root = parseJson(text) as JsonObject
    val out = ArrayList<OsmPlace>()
    for (e in root.arr("elements").elements) {
        e as JsonObject
        val type = e.s("type") ?: continue
        val id = e.s("id") ?: continue
        val center = e["center"] as? JsonObject
        val lat = e.d("lat") ?: center?.d("lat") ?: continue
        val lon = e.d("lon") ?: center?.d("lon") ?: continue
        val tags = (e["tags"] as? JsonObject)?.fields
            ?.mapNotNull { (k, v) -> (v as? JsonString)?.let { k to it.value } }?.toMap() ?: emptyMap()
        val amenity = tags["amenity"] ?: continue
        out += OsmPlace(
            osmId = "$type/$id",
            name = tags["name"],
            amenity = amenity,
            cuisine = tags["cuisine"],
            lat = lat, lon = lon,
            houseNumber = tags["addr:housenumber"],
            street = tags["addr:street"],
            postcode = tags["addr:postcode"],
            phone = tags["phone"] ?: tags["contact:phone"],
            website = tags["website"] ?: tags["contact:website"],
            menuUrl = tags["website:menu"] ?: tags["menu:url"] ?: tags["menu"]?.takeIf { it.startsWith("http") },
            openingHours = tags["opening_hours"],
            tags = tags,
        )
    }
    return out
}

private fun day(s: String?): String? = s?.takeIf { it.length >= 10 }?.substring(0, 10)

/** Parses one page of Socrata rows from `43nn-pn8j`. */
fun parseInspections(text: String): List<InspectionRow> =
    (parseJson(text) as JsonArray).elements.mapNotNull { v ->
        val o = v as JsonObject
        val camis = o.s("camis") ?: return@mapNotNull null
        val lat = o.d("latitude")
        val lon = o.d("longitude")
        val ok = validNyc(lat, lon)
        InspectionRow(
            camis = camis,
            dba = o.s("dba"),
            building = o.s("building"),
            street = o.s("street"),
            zipcode = o.s("zipcode"),
            phone = o.s("phone"),
            cuisine = o.s("cuisine_description"),
            inspectionDate = day(o.s("inspection_date")),
            action = o.s("action"),
            violationCode = o.s("violation_code"),
            violationDescription = o.s("violation_description"),
            criticalFlag = o.s("critical_flag"),
            score = o.s("score")?.toIntOrNull(),
            grade = o.s("grade"),
            gradeDate = day(o.s("grade_date")),
            inspectionType = o.s("inspection_type"),
            lat = if (ok) lat else null,
            lon = if (ok) lon else null,
            bbl = o.s("bbl"),
            nta = o.s("nta"),
        )
    }

/** Parses one page of Socrata rows from `erm2-nwe9`. */
fun parseComplaints(text: String): List<Complaint> =
    (parseJson(text) as JsonArray).elements.mapNotNull { v ->
        val o = v as JsonObject
        val key = o.s("unique_key") ?: return@mapNotNull null
        val lat = o.d("latitude")
        val lon = o.d("longitude")
        val ok = validNyc(lat, lon)
        Complaint(
            uniqueKey = key,
            createdDate = o.s("created_date"),
            closedDate = o.s("closed_date"),
            complaintType = o.s("complaint_type"),
            descriptor = o.s("descriptor"),
            descriptor2 = o.s("descriptor_2")?.takeIf { it != "N/A" },
            locationType = o.s("location_type"),
            incidentAddress = o.s("incident_address"),
            incidentZip = o.s("incident_zip"),
            bbl = o.s("bbl"),
            status = o.s("status"),
            resolution = o.s("resolution_description"),
            lat = if (ok) lat else null,
            lon = if (ok) lon else null,
        )
    }

/** Category words that mark a food or drink business (whole words, lower case). */
private val FOOD_CATEGORY = Regex(
    "\\b(restaurant|cafe|café|coffee|bar|pub|pizza|pizzeria|bakery|deli|delicatessen|diner|bistro|" +
        "brewpub|gastropub|ice cream|dessert|juice|bubble tea|tea house|bagel|donut|doughnut|sandwich|" +
        "hamburger|burger|taco|taqueria|sushi|noodle|ramen|buffet|steak house|grill|eatery|food court|" +
        "fast food|brasserie|tavern|beer hall|wine bar|cocktail|creperie|patisserie|cupcake|chicken wings)\\b",
)
private val NOT_FOOD_CATEGORY = Regex("\\b(supply|supplies|equipment|wholesale|wholesaler|distributor|supplier|repair|consultant)\\b")

/** True when one of [categories] names a food or drink business. */
fun isFoodCategory(categories: List<String>): Boolean = categories.any { c ->
    val l = c.lowercase()
    FOOD_CATEGORY.containsMatchIn(l) && !NOT_FOOD_CATEGORY.containsMatchIn(l)
}

private val ZIP = Regex("\\bNY (\\d{5})\\b")

/** The 5-digit ZIP in a Google Local address, or null. */
fun zipOf(address: String?): String? = address?.let { ZIP.find(it)?.groupValues?.get(1) }

/** Brooklyn ZIP codes all start with 112. */
fun isBrooklynAddress(address: String?): Boolean {
    if (address == null) return false
    val z = zipOf(address)
    return (z != null && z.startsWith("112")) || address.contains(", Brooklyn, NY")
}

/**
 * Parses one line of `meta-New_York.json.gz`. Returns null for lines that fail to parse
 * (the file is Python output and may hold non-JSON tokens such as NaN).
 */
fun parseGPlace(line: String): GPlace? {
    val o = try { parseJson(line) as? JsonObject } catch (_: IllegalArgumentException) { null } ?: return null
    val id = o.s("gmap_id") ?: return null
    val name = o.s("name") ?: return null
    val lat = o.d("latitude") ?: return null
    val lon = o.d("longitude") ?: return null
    val cats = (o["category"] as? JsonArray)?.elements?.mapNotNull { (it as? JsonString)?.value } ?: emptyList()
    val hours = (o["hours"] as? JsonArray)?.elements?.mapNotNull { h ->
        val a = (h as? JsonArray)?.elements ?: return@mapNotNull null
        a.mapNotNull { (it as? JsonString)?.value }.joinToString(" ")
    }?.joinToString("; ")
    return GPlace(
        gmapId = id, name = name, address = o.s("address"), lat = lat, lon = lon,
        categories = cats, avgRating = o.d("avg_rating"), numReviews = o.d("num_of_reviews")?.toInt(),
        price = o.s("price"), state = o.s("state"), description = o.s("description"), hours = hours,
    )
}

private val GMAP_ID_IN_LINE = Regex("\"gmap_id\":\\s*\"([^\"]+)\"")

/** The gmap_id of a review line without parsing the whole line (for the stream filter). */
fun gmapIdOfLine(line: String): String? = GMAP_ID_IN_LINE.find(line, maxOf(0, line.length - 200))?.groupValues?.get(1)
    ?: GMAP_ID_IN_LINE.find(line)?.groupValues?.get(1)

/** Parses one line of `review-New_York.json.gz`, dropping the reviewer's name and id. */
fun parseGReview(line: String): GReview? {
    val o = try { parseJson(line) as? JsonObject } catch (_: IllegalArgumentException) { null } ?: return null
    val id = o.s("gmap_id") ?: return null
    val time = (o["time"] as? JsonNumber)?.value?.toLong() ?: return null
    val resp = (o["resp"] as? JsonObject)?.s("text")
    return GReview(id, time, o.d("rating")?.toInt(), o.s("text"), resp)
}

/** Split "Name, 123 Main St, Brooklyn, NY 11201" into house number and street. */
fun houseAndStreetOf(address: String?): Pair<String, String>? {
    if (address == null) return null
    for (part in address.split(", ")) {
        val m = Regex("^(\\d+[A-Za-z]?(?:-\\d+)?)\\s+(.+)$").find(part.trim()) ?: continue
        return m.groupValues[1] to m.groupValues[2]
    }
    return null
}
