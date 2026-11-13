package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson

// Brooklyn neighbourhoods: NYC's 2020 Neighborhood Tabulation Areas (Socrata 9nt8-h7nd), with
// their polygons. Restaurants get the NTA whose polygon holds them; a neighbourhood named in a
// question resolves to one or more NTAs by name.

const val NTA_DATASET = "9nt8-h7nd"
/** Brooklyn rows in a fixed order, so the file (pinned by SHA-256 in manifest.json) is the same on every fetch. */
const val NTA_URL = "$SOCRATA/$NTA_DATASET.json?boroname=Brooklyn&\$order=nta2020&\$limit=500"

/** One NTA. [polygons]: each polygon is a list of rings, each ring a list of (lon, lat). */
class Neighborhood(
    val code: String,
    val name: String,
    val cdta: String?,
    /** 0 residential; 6 special; 7 cemetery; 9 park. */
    val type: String?,
    val polygons: List<List<List<DoubleArray>>>,
) {
    val minLat = polygons.minOf { p -> p.minOf { r -> r.minOf { it[1] } } }
    val maxLat = polygons.maxOf { p -> p.maxOf { r -> r.maxOf { it[1] } } }
    val minLon = polygons.minOf { p -> p.minOf { r -> r.minOf { it[0] } } }
    val maxLon = polygons.maxOf { p -> p.maxOf { r -> r.maxOf { it[0] } } }

    /** Average of the outer rings' vertices: a point near the middle, not a true centroid. */
    val center: Pair<Double, Double> by lazy {
        val pts = polygons.flatMap { it.first() }
        pts.sumOf { it[1] } / pts.size to pts.sumOf { it[0] } / pts.size
    }

    /** Even-odd rule over every ring of each polygon (holes included). */
    fun contains(lat: Double, lon: Double): Boolean {
        if (lat < minLat || lat > maxLat || lon < minLon || lon > maxLon) return false
        return polygons.any { poly ->
            var inside = false
            for (ring in poly) {
                var j = ring.size - 1
                for (i in ring.indices) {
                    val xi = ring[i][0]; val yi = ring[i][1]
                    val xj = ring[j][0]; val yj = ring[j][1]
                    if ((yi > lat) != (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside
                    j = i
                }
            }
            inside
        }
    }

    fun geometryJson(): String = polygons.joinToString(",", "[", "]") { p ->
        p.joinToString(",", "[", "]") { r -> r.joinToString(",", "[", "]") { "[${it[0]},${it[1]}]" } }
    }
}

private fun JsonArray.ring(): List<DoubleArray> = elements.map { pt ->
    val a = pt as JsonArray
    doubleArrayOf((a[0] as JsonNumber).value, (a[1] as JsonNumber).value)
}

/** Parses Socrata rows of `9nt8-h7nd` (MultiPolygon `the_geom`). Rows without a geometry are dropped. */
fun parseNtas(text: String): List<Neighborhood> =
    (parseJson(text) as JsonArray).elements.mapNotNull { v ->
        val o = v as JsonObject
        val code = o.s("nta2020") ?: return@mapNotNull null
        val name = o.s("ntaname") ?: return@mapNotNull null
        val geom = o["the_geom"] as? JsonObject ?: return@mapNotNull null
        val coords = geom["coordinates"] as? JsonArray ?: return@mapNotNull null
        val polys = when (geom.s("type")) {
            "MultiPolygon" -> coords.elements.map { p -> (p as JsonArray).elements.map { (it as JsonArray).ring() } }
            "Polygon" -> listOf(coords.elements.map { (it as JsonArray).ring() })
            else -> return@mapNotNull null
        }
        Neighborhood(code, name, o.s("cdtaname"), o.s("ntatype"), polys)
    }

/** Parses the `geom` column written by [Neighborhood.geometryJson]. */
fun parseGeometry(text: String): List<List<List<DoubleArray>>> =
    (parseJson(text) as JsonArray).elements.map { p -> (p as JsonArray).elements.map { (it as JsonArray).ring() } }

/** The residential NTA holding the point, else any NTA holding it (parks, cemeteries), else null. */
fun neighborhoodOf(ntas: List<Neighborhood>, lat: Double?, lon: Double?): Neighborhood? {
    if (lat == null || lon == null) return null
    val hits = ntas.filter { it.contains(lat, lon) }
    return hits.firstOrNull { it.type == "0" } ?: hits.firstOrNull()
}

/** Common names that are not a part of an NTA name. */
private val NEIGHBORHOOD_ALIASES = mapOf(
    "bed stuy" to "bedford stuyvesant", "bedstuy" to "bedford stuyvesant",
    "dumbo" to "downtown brooklyn dumbo boerum hill", "downtown" to "downtown brooklyn",
    "ditmas" to "ditmas park", "plg" to "prospect lefferts gardens", "lefferts" to "prospect lefferts gardens",
    "south slope" to "windsor terrace south slope", "sea gate" to "coney island sea gate",
    "williamsburg" to "williamsburg", "carroll gardens" to "carroll gardens",
)

private fun nkey(s: String): String =
    java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFD).replace(Regex("\\p{M}"), "")
        .lowercase().replace(Regex("\\([^)]*\\)"), " ").replace(Regex("[^a-z0-9]+"), " ").trim()

/**
 * NTAs whose name has [query] as one of its hyphen-separated parts or as a whole-word phrase,
 * ignoring "(East)" / "(West)" qualifiers: "Park Slope" -> BK0602; "Bushwick" -> both
 * Bushwick NTAs; "Gowanus" -> Carroll Gardens-Cobble Hill-Gowanus-Red Hook.
 */
fun resolveNeighborhood(ntas: List<Neighborhood>, query: String): List<Neighborhood> {
    var q = nkey(query).removePrefix("the ").removeSuffix(" brooklyn").trim()
    q = NEIGHBORHOOD_ALIASES[q] ?: q
    if (q.isEmpty()) return emptyList()
    val exactCode = ntas.filter { it.code.equals(query.trim(), ignoreCase = true) }
    if (exactCode.isNotEmpty()) return exactCode
    val parts = ntas.filter { n -> n.name.split('-').any { nkey(it) == q } || nkey(n.name) == q }
    if (parts.isNotEmpty()) return parts
    return ntas.filter { n -> " ${nkey(n.name)} ".contains(" $q ") }
}
