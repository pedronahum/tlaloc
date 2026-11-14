package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeParseException

// The agent's tools over build/data/brooklyn.db.
//
//   snapshot  the database only: no network, and results depend only on the database file
//             (the snapshot date is "today"; get_menu returns the menu read at ingest from the
//             Internet Archive snapshot of the place's site nearest the cutoff; geocode and
//             web_search use offline stand-ins built from the snapshot)
//   live      current data: DOHMH and 311 from the Socrata API, Nominatim, the restaurant's
//             own site, SearXNG; reviews have no live source. The database is never written.
//   record    live, and every result is stored in recordings.db keyed by tool and arguments
//   replay    only what record stored; a call that was not recorded returns an error

enum class ToolMode { SNAPSHOT, LIVE, RECORD, REPLAY;
    val online: Boolean get() = this == LIVE || this == RECORD
    companion object {
        fun parse(s: String): ToolMode = entries.firstOrNull { it.name.equals(s, true) }
            ?: throw IllegalArgumentException("mode must be one of snapshot, live, record, replay")
    }
}

class ToolError(message: String) : Exception(message)

const val NYC_TZ = "America/New_York"
const val SEARXNG_DEFAULT = "http://127.0.0.1:8890"
private const val DOHMH_URL = "$SOCRATA/43nn-pn8j.json"
private const val NYC311_URL = "$SOCRATA/erm2-nwe9.json"
private const val NOMINATIM = "https://nominatim.openstreetmap.org/search"
/** West, north, east, south: a box around Kings County for Nominatim. */
private const val BROOKLYN_VIEWBOX = "-74.06,40.74,-73.83,40.56"

/** One restaurants row (plus its Google Local categories). */
class Rest(
    val id: String, val camis: String?, val osmId: String?, val gmapId: String?, val name: String, val dba: String?,
    val osmName: String?, val gName: String?, val cuisine: String?, val osmCuisine: String?, val amenity: String?,
    val address: String?, val zip: String?, val phone: String?, val lat: Double?, val lon: Double?,
    val website: String?, val menuUrl: String?, val osmHours: String?, val grade: String?, val gradeDate: String?,
    val latestInspection: String?, val latestScore: Int?, val gRating: Double?, val gNumReviews: Int?,
    val gPrice: String?, val gState: String?, val gHours: String?, val nta2020: String?, val neighborhood: String?,
    val gCategories: String?,
) {
    /** Lower-case cuisine words from DOHMH, OSM, Google categories and the OSM amenity. */
    val cuisineText: String by lazy {
        listOfNotNull(cuisine, osmCuisine, gCategories, amenity).joinToString(" ").lowercase()
            .replace(Regex("[^a-z0-9]+"), " ").trim()
    }
    val hours: Pair<WeekHours, String>? by lazy {
        parseOsmHours(osmHours)?.let { it to "osm" } ?: parseGoogleHours(gHours)?.let { it to "google-local-2021" }
    }
}

private val CUISINE_SYNONYMS = mapOf(
    "coffee" to listOf("coffee", "cafe"), "cafe" to listOf("cafe", "coffee"), "bbq" to listOf("bbq", "barbecue", "barbeque"),
    "barbecue" to listOf("barbecue", "bbq", "barbeque"), "burgers" to listOf("burger", "hamburger", "burgers"),
    "burger" to listOf("burger", "hamburger", "burgers"), "tacos" to listOf("taco", "tacos", "mexican", "taqueria"),
    "doughnuts" to listOf("donut", "donuts", "doughnut"), "donuts" to listOf("donut", "donuts", "doughnut"),
    "bakery" to listOf("bakery", "bakeries", "bagels", "pastry"), "tea" to listOf("tea", "bubble tea"),
    "bar" to listOf("bar", "pub", "bars", "cocktail", "wine bar"), "pub" to listOf("pub", "bar", "gastropub"),
    "ice cream" to listOf("ice cream", "frozen desserts", "dessert"), "vegan" to listOf("vegan", "vegetarian"),
)

/** True when [query] names one of [r]'s cuisines (whole words; simple plurals; a few synonyms). */
fun cuisineMatches(r: Rest, query: String): Boolean {
    val q = query.lowercase().replace(Regex("[^a-z0-9]+"), " ").trim()
    if (q.isEmpty()) return true
    val alts = (CUISINE_SYNONYMS[q] ?: listOf(q)) + listOf(q.removeSuffix("s"), q + "s")
    val text = " ${r.cuisineText} "
    return alts.any { a -> text.contains(" $a ") }
}

class ToolServer(
    val dbPath: Path,
    val mode: ToolMode,
    private val netFactory: () -> Net = { HttpNet() },
    val recordingsPath: Path = dbPath.resolveSibling("recordings.db"),
    val searxngUrl: String = System.getenv("SEARXNG_URL") ?: SEARXNG_DEFAULT,
    private val liveClock: () -> LocalDateTime = { LocalDateTime.now(ZoneId.of(NYC_TZ)) },
) : AutoCloseable {
    private val db: Connection = DriverManager.getConnection("jdbc:sqlite:file:$dbPath?mode=ro")

    /** True once an HTTP client has been built (never in snapshot or replay mode). */
    var netConstructed = false
        private set
    private val net: Net by lazy {
        check(mode.online) { "no network in $mode mode" }
        netConstructed = true
        netFactory()
    }
    private val robots = Robots()
    private var recordings: Connection? = null

    val snapshotDate: String = scalar("SELECT value FROM meta WHERE key='snapshot_date'") ?: error("database has no snapshot_date")
    val neighborhoods: List<Neighborhood> = query("SELECT code, name, cdta, type, geom FROM neighborhoods ORDER BY code").map {
        Neighborhood(it[0] as String, it[1] as String, it[2] as String?, it[3] as String?, parseGeometry(it[4] as String))
    }
    val restaurants: List<Rest> = query(
        "SELECT r.id, r.camis, r.osm_id, r.gmap_id, r.name, r.dba, r.osm_name, r.g_name, r.cuisine, r.osm_cuisine, r.amenity, " +
            "r.address, r.zipcode, r.phone, r.lat, r.lon, r.website, r.menu_url, r.opening_hours, r.grade, r.grade_date, " +
            "r.latest_inspection, r.latest_score, r.g_rating, r.g_num_reviews, r.g_price, r.g_state, r.g_hours, r.nta2020, " +
            "r.neighborhood, g.categories FROM restaurants r LEFT JOIN g_places g ON g.gmap_id = r.gmap_id ORDER BY r.id",
    ).map { c ->
        Rest(
            c[0] as String, c[1] as String?, c[2] as String?, c[3] as String?, c[4] as String, c[5] as String?, c[6] as String?,
            c[7] as String?, c[8] as String?, c[9] as String?, c[10] as String?, c[11] as String?, c[12] as String?,
            c[13] as String?, (c[14] as Number?)?.toDouble(), (c[15] as Number?)?.toDouble(), c[16] as String?, c[17] as String?,
            c[18] as String?, c[19] as String?, c[20] as String?, c[21] as String?, (c[22] as Number?)?.toInt(),
            (c[23] as Number?)?.toDouble(), (c[24] as Number?)?.toInt(), c[25] as String?, c[26] as String?, c[27] as String?,
            c[28] as String?, c[29] as String?, c[30] as String?,
        )
    }
    private val byId = restaurants.associateBy { it.id }
    private val byOsm = restaurants.filter { it.osmId != null }.associateBy { it.osmId!! }

    override fun close() {
        db.close()
        recordings?.close()
    }

    // ------------------------------------------------------------------ dispatch

    /** Runs tool [name] on [argumentsJson] and returns its JSON result (an object; `error` on failure). */
    fun call(name: String, argumentsJson: String): String {
        val def = TOOL_BY_NAME[name] ?: return toJson(mapOf("error" to "unknown tool '$name'", "tools" to TOOL_DEFS.map { it.name }))
        val args = try {
            parseJson(argumentsJson.ifBlank { "{}" }) as? JsonObject ?: return toJson(mapOf("error" to "arguments must be a JSON object"))
        } catch (e: Exception) {
            return toJson(mapOf("error" to "arguments are not valid JSON: ${e.message}"))
        }
        val problems = validate(def.schema, args)
        if (problems.isNotEmpty()) return toJson(mapOf("error" to "invalid arguments", "details" to problems))
        val key = canonicalJson(args)
        if (mode == ToolMode.REPLAY) {
            return recorded(name, key) ?: toJson(mapOf("error" to "no recording of $name $key", "mode" to "replay"))
        }
        val result = try {
            toJson(dispatch(name, args))
        } catch (e: ToolError) {
            toJson(mapOf("error" to e.message))
        } catch (e: java.io.IOException) {
            toJson(mapOf("error" to "network: ${e.message ?: e.javaClass.simpleName}"))
        }
        if (mode == ToolMode.RECORD) record(name, key, result)
        return result
    }

    private fun dispatch(name: String, a: JsonObject): Map<String, Any?> = when (name) {
        "search_restaurants" -> searchRestaurants(a)
        "get_restaurant" -> getRestaurant(rest(a.str("id")))
        "get_inspections" -> getInspections(rest(a.str("id")), a.optString("since"), a.optInt("limit") ?: 10)
        "get_complaints" -> getComplaints(a)
        "get_reviews" -> getReviews(rest(a.str("id")), a.optString("query"), a.optInt("limit") ?: 5)
        "get_menu" -> getMenu(rest(a.str("id")))
        "geocode" -> geocode(a.str("address"))
        "web_search" -> webSearch(a.str("query"), a.optInt("limit") ?: 5)
        else -> throw ToolError("unknown tool '$name'")
    }

    /** The restaurant with [id]: a restaurants id, an OSM id with or without "osm:", or a CAMIS. */
    fun rest(id: String): Rest {
        val t = id.trim()
        return byId[t] ?: byId["osm:$t"] ?: byOsm[t.removePrefix("osm:")]
            ?: throw ToolError("unknown restaurant id '$id'; ids come from search_restaurants")
    }

    private val modeLabel get() = if (mode.online) "live" else "snapshot $snapshotDate"
    private fun src(what: String) = "$what ($modeLabel)"
    /** Data that has no live source in any mode: OSM tags, matches, Google Local. */
    private fun snap(what: String) = "$what (snapshot $snapshotDate)"
    private fun today(): LocalDate = if (mode.online) liveClock().toLocalDate() else LocalDate.parse(snapshotDate)

    // ------------------------------------------------------------------ search_restaurants

    private fun searchRestaurants(a: JsonObject): Map<String, Any?> {
        val limit = a.optInt("limit") ?: 10
        val query = a.optString("query")
        var cands: List<Rest>
        var rank: Map<String, Int>? = null
        if (query != null) {
            val ids = ftsIds(query)
            rank = ids.withIndex().associate { it.value to it.index }
            cands = ids.mapNotNull { byId[it] }
        } else {
            cands = restaurants
        }
        val applied = LinkedHashMap<String, Any?>()
        a.optString("cuisine")?.let { c -> cands = cands.filter { cuisineMatches(it, c) }; applied["cuisine"] = c }
        a.optString("neighborhood")?.let { n ->
            val ns = resolveNeighborhood(neighborhoods, n)
            if (ns.isEmpty()) throw ToolError("unknown Brooklyn neighborhood '$n'")
            val codes = ns.map { it.code }.toSet()
            cands = cands.filter { it.nta2020 in codes }
            applied["neighborhood"] = ns.map { it.name }
        }
        var center: Map<String, Any?>? = null
        val dist = HashMap<String, Double>()
        a.optObject("near")?.let { near ->
            val (lat, lon, label) = point(near)
            val radius = (a.optInt("radius_m") ?: 1000).toDouble()
            cands = cands.filter { r ->
                if (r.lat == null || r.lon == null) return@filter false
                val d = distanceM(lat, lon, r.lat, r.lon)
                dist[r.id] = d
                d <= radius
            }
            center = linkedMapOf("lat" to round5(lat), "lon" to round5(lon), "from" to label, "radius_m" to radius.toInt())
        }
        a.optString("min_grade")?.let { g ->
            val ok = "ABC".substring(0, "ABC".indexOf(g) + 1)
            cands = cands.filter { it.grade != null && it.grade.length == 1 && it.grade in ok }
            applied["min_grade"] = g
        }
        var unknownHours = 0
        var at: LocalDateTime? = null
        a.optString("open_at")?.let { s ->
            at = if (s == "now") {
                if (!mode.online) throw ToolError("open_at 'now' needs live mode; give a time, e.g. ${snapshotDate}T19:30")
                liveClock()
            } else try { LocalDateTime.parse(s) } catch (e: DateTimeParseException) { throw ToolError("open_at: ${e.message}") }
            val before = cands.size
            cands = cands.filter { it.hours != null }
            unknownHours = before - cands.size
            cands = cands.filter { it.hours!!.first.isOpen(at!!) }
            applied["open_at"] = at.toString()
        }
        cands = when {
            rank != null -> cands.sortedWith(compareBy({ rank[it.id] }, { dist[it.id] ?: 0.0 }))
            dist.isNotEmpty() -> cands.sortedWith(compareBy({ dist[it.id] }, { it.id }))
            else -> cands.sortedWith(compareBy({ it.name.lowercase() }, { it.id }))
        }
        val top = cands.take(limit)
        var liveNote: String? = null
        val live = if (mode.online) try {
            liveGrades(top.mapNotNull { it.camis })
        } catch (e: Exception) {
            liveNote = "live grade refresh failed (${e.message ?: e.javaClass.simpleName}); grades are from the snapshot"
            emptyMap()
        } else emptyMap()
        val results = top.map { r ->
            val lg = r.camis?.let { live[it] }
            linkedMapOf(
                "id" to r.id, "name" to r.name, "cuisine" to (r.cuisine ?: r.osmCuisine?.replace(';', ',')?.replace('_', ' ')),
                "address" to r.address, "neighborhood" to r.neighborhood,
                "grade" to (lg?.get("grade") ?: r.grade), "grade_date" to (lg?.get("grade_date") ?: r.gradeDate),
                "rating" to r.gRating, "distance_m" to dist[r.id]?.let { Math.round(it).toInt() },
                "open" to at?.let { true }, "hours" to if (at != null) r.hours?.second else null,
            )
        }
        return linkedMapOf(
            "count" to cands.size, "results" to results,
            "query" to query, "filters" to applied.ifEmpty { null }, "near" to center,
            "excluded_unknown_hours" to if (at != null) unknownHours else null, "note" to liveNote,
            "as_of" to if (mode.online) today().toString() else snapshotDate,
            "sources" to listOfNotNull(
                snap("restaurants: DOHMH permits, OpenStreetMap, Google Local 2021"),
                if (mode.online && live.isNotEmpty()) "grades of these results: DOHMH 43nn-pn8j (live)" else null,
            ),
        )
    }

    /** Restaurant ids for a free-text query, best first (FTS5 bm25; every word, then any word). */
    fun ftsIds(query: String, max: Int = 2000): List<String> {
        val words = Regex("[\\p{L}\\p{N}]+").findAll(query.lowercase()).map { it.value }.toList()
        val toks = words.filter { it.length > 1 }.ifEmpty { words }.distinct()
        if (toks.isEmpty()) return emptyList()
        fun run(op: String): List<String> = query(
            "SELECT id FROM restaurants_fts WHERE restaurants_fts MATCH ? ORDER BY bm25(restaurants_fts, 0.0, 10.0, 2.0, 1.0), id LIMIT $max",
            toks.joinToString(" $op ") { "\"" + it.replace("\"", "") + "\"*" },
        ).map { it[0] as String }
        return run("AND").ifEmpty { if (toks.size > 1) run("OR") else emptyList() }
    }

    /** lat, lon and how they were obtained, for a `near` argument. */
    private fun point(near: JsonObject): Triple<Double, Double, String> {
        val lat = near.optNumber("lat")
        val lon = near.optNumber("lon")
        if (lat != null && lon != null) return Triple(lat, lon, "coordinates")
        val address = near.optString("address") ?: throw ToolError("near needs lat and lon, or address")
        @Suppress("UNCHECKED_CAST")
        val hits = geocode(address)["results"] as List<Map<String, Any?>>
        val h = hits.firstOrNull() ?: throw ToolError("could not geocode '$address'")
        return Triple((h["lat"] as Number).toDouble(), (h["lon"] as Number).toDouble(), "geocode: ${h["display_name"]} (${h["precision"]})")
    }

    // ------------------------------------------------------------------ get_restaurant

    private fun getRestaurant(r: Rest): Map<String, Any?> {
        val n = linkedMapOf<String, Any?>()
        if (r.camis != null) {
            val since = today().minusYears(1).toString()
            n["inspections"] = scalarLong("SELECT count(DISTINCT inspection_date) FROM inspections WHERE camis=? AND inspection_date > '1900-01-01'", r.camis)
            n["critical_violations_last_12_months"] = scalarLong(
                "SELECT count(*) FROM inspections WHERE camis=? AND critical_flag='Critical' AND inspection_date >= ?", r.camis, since,
            )
            n["complaints_311"] = scalarLong("SELECT count(*) FROM complaints WHERE camis=?", r.camis)
        }
        if (r.gmapId != null) n["reviews"] = scalarLong("SELECT count(*) FROM reviews WHERE gmap_id=?", r.gmapId)
        val menu = query("SELECT status, snapshot FROM menus WHERE restaurant_id=? ORDER BY url LIMIT 1", r.id).firstOrNull()
        val out = linkedMapOf<String, Any?>(
            "id" to r.id, "name" to r.name,
            "other_names" to listOfNotNull(r.dba?.let(::titleCase), r.osmName, r.gName).distinct().filter { it != r.name }.ifEmpty { null },
            "camis" to r.camis, "osm_id" to r.osmId,
            "cuisine" to r.cuisine, "osm_cuisine" to r.osmCuisine, "type" to r.amenity,
            "address" to r.address, "neighborhood" to r.neighborhood, "lat" to r.lat?.let(::round5), "lon" to r.lon?.let(::round5),
            "phone" to r.phone, "website" to r.website, "menu_url" to r.menuUrl,
            "opening_hours" to (r.osmHours ?: r.gHours), "hours_source" to when {
                r.osmHours != null -> "osm"; r.gHours != null -> "google-local-2021"; else -> null
            },
            "grade" to r.grade, "grade_date" to r.gradeDate, "latest_inspection" to r.latestInspection, "latest_score" to r.latestScore,
            "google_rating" to r.gRating, "google_reviews" to r.gNumReviews, "price" to r.gPrice,
            "counts" to n.ifEmpty { null },
            "archived_menu" to menu?.let { linkedMapOf("status" to it[0], "archived_at" to it[1]) },
        )
        val sources = mutableListOf(
            src("DOHMH 43nn-pn8j").takeIf { r.camis != null }, snap("OpenStreetMap").takeIf { r.osmId != null },
            "Google Local (UCSD, to 2021)".takeIf { r.gmapId != null },
        )
        if (mode.online && r.camis != null) {
            val rows = liveInspectionRows(r.camis, null)
            val latest = rows.filter { it.inspectionDate != null && it.inspectionDate > "1900-01-01" }.maxByOrNull { it.inspectionDate!! }
            val graded = rows.filter { it.grade != null && it.gradeDate != null }.maxByOrNull { it.gradeDate!! }
            out["grade"] = graded?.grade
            out["grade_date"] = graded?.gradeDate
            out["latest_inspection"] = latest?.inspectionDate
            out["latest_score"] = rows.filter { it.score != null && it.inspectionDate == latest?.inspectionDate }.maxOfOrNull { it.score!! }
            out["live_dohmh_rows"] = rows.size
            if (rows.isEmpty()) out["note"] = "DOHMH returned no rows for this CAMIS today (closed or permit lapsed?)"
        }
        out["sources"] = sources.filterNotNull()
        return out
    }

    // ------------------------------------------------------------------ get_inspections

    private fun getInspections(r: Rest, since: String?, limit: Int): Map<String, Any?> {
        val camis = r.camis ?: return linkedMapOf(
            "id" to r.id, "name" to r.name, "inspections" to emptyList<Any>(),
            "note" to "no DOHMH permit is linked to this place, so there are no inspection records for it",
        )
        val rows: List<InspectionRow> = if (mode.online) liveInspectionRows(camis, since) else query(
            "SELECT inspection_date, inspection_type, action, violation_code, violation_description, critical_flag, score, grade, grade_date " +
                "FROM inspections WHERE camis=? AND inspection_date >= ? ORDER BY inspection_date DESC, violation_code",
            camis, since ?: "0000",
        ).map {
            InspectionRow(
                camis, null, null, null, null, null, null, it[0] as String?, it[2] as String?, it[3] as String?, it[4] as String?,
                it[5] as String?, (it[6] as Number?)?.toInt(), it[7] as String?, it[8] as String?, it[1] as String?, null, null, null, null,
            )
        }
        val visits = rows.filter { it.inspectionDate != null && it.inspectionDate > "1900-01-01" }
            .groupBy { it.inspectionDate!! to (it.inspectionType ?: "") }
            .entries.sortedWith(compareByDescending<Map.Entry<Pair<String, String>, List<InspectionRow>>> { it.key.first }.thenBy { it.key.second })
        val out = visits.take(limit).map { (k, rs) ->
            linkedMapOf(
                "date" to k.first, "type" to k.second.ifEmpty { null },
                "action" to rs.firstNotNullOfOrNull { it.action }?.let { shorten(it, 120) },
                "score" to rs.firstNotNullOfOrNull { it.score }, "grade" to rs.firstNotNullOfOrNull { it.grade },
                "grade_date" to rs.firstNotNullOfOrNull { it.gradeDate },
                "violations" to rs.filter { it.violationCode != null }.distinctBy { it.violationCode }.sortedBy { it.violationCode }.map {
                    linkedMapOf("code" to it.violationCode, "critical" to (it.criticalFlag == "Critical"), "text" to it.violationDescription?.let { d -> shorten(d, 160) })
                },
            )
        }
        return linkedMapOf(
            "id" to r.id, "name" to r.name, "camis" to camis, "count" to visits.size, "inspections" to out,
            "not_yet_inspected" to if (visits.isEmpty() && rows.isNotEmpty()) true else null,
            "since" to since, "source" to src("DOHMH Restaurant Inspection Results 43nn-pn8j"),
        )
    }

    private fun liveInspectionRows(camis: String, since: String?): List<InspectionRow> {
        val where = "camis='${camis.filter { it.isDigit() }}'" + (since?.let { " AND inspection_date >= '${it}T00:00:00'" } ?: "")
        val url = "$DOHMH_URL?\$where=${Http.enc(where)}&\$order=${Http.enc("inspection_date DESC")}&\$limit=1000"
        val resp = net.get(url, "application/json")
        if (resp.status != 200) throw ToolError("DOHMH API returned HTTP ${resp.status}")
        return parseInspections(resp.text)
    }

    /** Latest letter grade per CAMIS from the live API (one request). */
    private fun liveGrades(camis: List<String>): Map<String, Map<String, String?>> {
        if (camis.isEmpty()) return emptyMap()
        val ids = camis.map { c -> c.filter { it.isDigit() } }.joinToString(",") { "'$it'" }
        val url = "$DOHMH_URL?\$select=${Http.enc("camis,grade,grade_date")}&\$where=${Http.enc("camis in($ids) AND grade IS NOT NULL")}&\$limit=5000"
        val resp = net.get(url, "application/json")
        if (resp.status != 200) throw ToolError("DOHMH API returned HTTP ${resp.status}")
        return parseInspections(resp.text).filter { it.gradeDate != null }.groupBy { it.camis }
            .mapValues { (_, rs) -> rs.maxBy { it.gradeDate!! }.let { mapOf("grade" to it.grade, "grade_date" to it.gradeDate) } }
    }

    // ------------------------------------------------------------------ get_complaints

    private fun getComplaints(a: JsonObject): Map<String, Any?> {
        val since = a.optString("since")
        val limit = a.optInt("limit") ?: 10
        val id = a.optString("id")
        val r = id?.let { rest(it) }
        var lat: Double? = null
        var lon: Double? = null
        var radius = (a.optInt("radius_m") ?: 100).toDouble()
        var basis: String
        val near = a.optObject("near")
        if (r != null && r.camis != null && near == null) {
            basis = "linked to CAMIS ${r.camis} by address or tax lot"
        } else if (near != null) {
            val p = point(near); lat = p.first; lon = p.second
            basis = "within ${radius.toInt()} m of ${round5(p.first)}, ${round5(p.second)} (${p.third})"
        } else {
            lat = r!!.lat ?: throw ToolError("this place has no DOHMH permit and no coordinates")
            lon = r.lon
            radius = minOf(radius, 25.0)
            basis = "no DOHMH permit linked; complaints within ${radius.toInt()} m of the place"
        }
        val rows: List<Map<String, Any?>> = if (mode.online) liveComplaints(r?.camis?.takeIf { near == null }, lat, lon, radius, since)
        else if (lat == null) query(
            "SELECT unique_key, created_date, closed_date, complaint_type, descriptor, descriptor_2, incident_address, incident_zip, " +
                "status, resolution, camis, link_rule FROM complaints WHERE camis=? AND created_date >= ? ORDER BY created_date DESC, unique_key",
            r!!.camis, since ?: "0000",
        ).map(::complaintRow) else {
            val dLat = radius / 111_000.0
            val dLon = radius / (111_000.0 * Math.cos(Math.toRadians(lat)))
            query(
                "SELECT unique_key, created_date, closed_date, complaint_type, descriptor, descriptor_2, incident_address, incident_zip, " +
                    "status, resolution, camis, link_rule, lat, lon FROM complaints WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? " +
                    "AND created_date >= ? ORDER BY created_date DESC, unique_key",
                lat - dLat, lat + dLat, lon!! - dLon, lon + dLon, since ?: "0000",
            ).filter { distanceM(lat, lon, (it[12] as Number).toDouble(), (it[13] as Number).toDouble()) <= radius }.map(::complaintRow)
        }
        return linkedMapOf(
            "id" to r?.id, "name" to r?.name, "basis" to basis, "count" to rows.size, "complaints" to rows.take(limit),
            "since" to since, "source" to src("NYC 311 erm2-nwe9, Food Poisoning and Food Establishment" + if (mode.online) "" else ", from ${LocalDate.parse(snapshotDate).minusYears(3)}"),
        )
    }

    private fun complaintRow(c: List<Any?>): Map<String, Any?> = linkedMapOf(
        "key" to c[0], "created" to (c[1] as String?)?.take(10), "closed" to (c[2] as String?)?.take(10),
        "type" to c[3], "descriptor" to c[4], "detail" to c[5], "address" to c[6], "zip" to c[7], "status" to c[8],
        "resolution" to (c[9] as String?)?.let { shorten(it, 200) }, "camis" to c[10], "link" to c[11],
    )

    private fun liveComplaints(camis: String?, lat: Double?, lon: Double?, radius: Double, since: String?): List<Map<String, Any?>> {
        val types = "complaint_type in(${COMPLAINT_TYPES.joinToString(",") { "'$it'" }})"
        val sinceW = since?.let { " AND created_date >= '${it}T00:00:00'" } ?: ""
        val est = camis?.let { query("SELECT building, street, zipcode, bbl FROM establishments WHERE camis=?", it).firstOrNull() }
        val where = if (camis != null) {
            val bbl = (est?.get(3) as String?)?.filter { it.isDigit() }
                ?: throw ToolError("no tax lot (BBL) for CAMIS $camis to query 311 by")
            "bbl='$bbl' AND $types$sinceW"
        } else "within_circle(location, $lat, $lon, ${radius.toInt()}) AND $types$sinceW"
        val url = "$NYC311_URL?\$where=${Http.enc(where)}&\$order=${Http.enc("created_date DESC")}&\$limit=200"
        val resp = net.get(url, "application/json")
        if (resp.status != 200) throw ToolError("311 API returned HTTP ${resp.status}")
        val estKey = est?.let { addressKey(it[0] as String?, it[1] as String?) }
        return parseComplaints(resp.text).map { x ->
            val hs = x.incidentAddress?.let { s -> s.indexOf(' ').takeIf { it > 0 }?.let { s.substring(0, it) to s.substring(it + 1) } }
            val link = when {
                camis == null -> null
                estKey != null && addressKey(hs?.first, hs?.second) == estKey -> "address"
                else -> "same tax lot"
            }
            complaintRow(
                listOf(
                    x.uniqueKey, x.createdDate, x.closedDate, x.complaintType, x.descriptor, x.descriptor2, x.incidentAddress,
                    x.incidentZip, x.status, x.resolution, camis?.takeIf { link == "address" }, link,
                ),
            )
        }
    }

    // ------------------------------------------------------------------ get_reviews

    private fun getReviews(r: Rest, q: String?, limit: Int): Map<String, Any?> {
        val gmap = r.gmapId ?: return linkedMapOf(
            "id" to r.id, "name" to r.name, "reviews" to emptyList<Any>(),
            "note" to "no Google Local business is matched to this place",
        )
        val stats = query("SELECT count(*), avg(rating), min(time_ms), max(time_ms), sum(text IS NOT NULL) FROM reviews WHERE gmap_id=?", gmap).first()
        val hist = query("SELECT rating, count(*) FROM reviews WHERE gmap_id=? GROUP BY rating ORDER BY rating DESC", gmap)
            .associate { (it[0] as Number?)?.toString() to (it[1] as Number).toInt() }
        val rows = if (q != null) {
            val toks = Regex("[\\p{L}\\p{N}]+").findAll(q.lowercase()).map { it.value }.toList()
            if (toks.isEmpty()) emptyList() else query(
                "SELECT r.time_ms, r.rating, r.text, r.response FROM reviews r WHERE r.gmap_id = ? AND r.rowid IN " +
                    "(SELECT rowid FROM reviews_fts WHERE reviews_fts MATCH ?) ORDER BY r.time_ms DESC LIMIT 200",
                gmap, toks.joinToString(" OR ") { "\"$it\"*" },
            )
        } else query(
            "SELECT time_ms, rating, text, response FROM reviews WHERE gmap_id=? AND text IS NOT NULL ORDER BY time_ms DESC LIMIT ?",
            gmap, limit,
        )
        return linkedMapOf(
            "id" to r.id, "name" to r.name, "google_name" to r.gName,
            "count" to (stats[0] as Number).toInt(), "with_text" to (stats[4] as Number?)?.toInt(),
            "average_rating" to (stats[1] as Number?)?.toDouble()?.let { Math.round(it * 100) / 100.0 },
            "ratings" to hist, "first" to (stats[2] as Number?)?.let { day(it.toLong()) }, "last" to (stats[3] as Number?)?.let { day(it.toLong()) },
            "query" to q, "matching" to if (q != null) rows.size else null,
            "reviews" to rows.take(limit).map {
                linkedMapOf(
                    "date" to day((it[0] as Number).toLong()), "rating" to (it[1] as Number?)?.toInt(),
                    "text" to (it[2] as String?)?.let { t -> shorten(t, 400) }, "owner_response" to (it[3] as String?)?.let { t -> shorten(t, 200) },
                )
            },
            "source" to "Google Local reviews, UCSD McAuley lab dataset (to 2021); no live source" +
                if (mode.online) "" else "; snapshot $snapshotDate",
        )
    }

    private fun day(ms: Long): String = Instant.ofEpochMilli(ms).atZone(ZoneOffset.UTC).toLocalDate().toString()

    // ------------------------------------------------------------------ get_menu

    private fun getMenu(r: Rest): Map<String, Any?> {
        val target = (r.menuUrl ?: r.website)?.let { normalizeStartUrl(it) }
        val base = linkedMapOf<String, Any?>("id" to r.id, "name" to r.name, "website" to r.website, "menu_url" to r.menuUrl)
        if (target == null) {
            base["menu"] = null
            base["reason"] = "no website or menu URL in OpenStreetMap"
            base["source"] = src("OpenStreetMap")
            return base
        }
        if (mode.online) {
            val m = fetchLiveMenu(net, robots, target)
            base["menu"] = m.text
            base["status"] = m.status
            base["items"] = m.items
            base["page"] = m.url
            base["fetched_at"] = Instant.now().truncatedTo(java.time.temporal.ChronoUnit.SECONDS).toString()
            m.text ?: run { base["reason"] = menuReason(m.status) }
            base["source"] = "restaurant web site, live"
            return base
        }
        val row = query(
            "SELECT page_url, snapshot, archive_url, text, status, items FROM menus WHERE restaurant_id=? ORDER BY url LIMIT 1", r.id,
        ).firstOrNull()
        if (row == null) {
            base["menu"] = null
            base["reason"] = "the archived site was not looked up for this snapshot (quick run, or the place has no OSM website)"
            base["source"] = src("menus from the Internet Archive")
            return base
        }
        base["menu"] = row[3]
        base["status"] = row[4]
        base["items"] = (row[5] as Number?)?.toInt()
        base["page"] = row[0]
        base["archived_at"] = row[1]
        base["archive_url"] = row[2]
        if (row[3] == null) base["reason"] = menuReason(row[4] as String)
        base["source"] = "Internet Archive snapshot of the restaurant's site, on or before ${scalar("SELECT value FROM meta WHERE key='menu_cutoff'") ?: "the cutoff"}; snapshot $snapshotDate"
        return base
    }

    private fun menuReason(status: String): String = when (status) {
        "no_prices" -> "the page has no priced menu lines (often a menu rendered by JavaScript or an image)"
        "pdf" -> "the menu is a PDF, which is not extracted"
        "not_html" -> "the page is not HTML"
        "robots" -> "the site's robots.txt disallows fetching this page"
        "no_snapshot" -> "the Internet Archive has no capture of the site on or before the cutoff"
        "excluded" -> "the Internet Archive does not serve captures of this site"
        "empty" -> "the Internet Archive's capture of the page is empty"
        "undecodable" -> "the archived page is in an encoding this example does not read (not gzip)"
        else -> "fetch failed: $status"
    }

    // ------------------------------------------------------------------ geocode

    fun geocode(address: String): Map<String, Any?> {
        if (mode.online) return nominatim(address)
        val results = offlineGeocode(address)
        return linkedMapOf(
            "query" to address, "results" to results,
            "source" to "offline gazetteer from the snapshot (DOHMH and OSM addresses, NTA polygons, restaurant names); snapshot $snapshotDate",
        )
    }

    private fun nominatim(address: String): Map<String, Any?> {
        val a = ordinalStreets(address)
        val q = if (Regex("(?i)brooklyn|\\bny\\b|new york|\\b112\\d\\d\\b").containsMatchIn(a)) a else "$a, Brooklyn, NY"
        val url = "$NOMINATIM?q=${Http.enc(q)}&format=jsonv2&limit=3&countrycodes=us&viewbox=$BROOKLYN_VIEWBOX&bounded=1"
        val resp = net.get(url, "application/json")
        if (resp.status != 200) throw ToolError("Nominatim returned HTTP ${resp.status}")
        val hits = (parseJson(resp.text) as JsonArray).elements.map { it as JsonObject }.map { o ->
            val lat = o.d("lat")!!
            val lon = o.d("lon")!!
            linkedMapOf(
                "lat" to round5(lat), "lon" to round5(lon), "display_name" to o.s("display_name"),
                "precision" to (o.s("addresstype") ?: o.s("type")), "osm" to o.s("osm_type")?.let { "$it/${o.s("osm_id")}" },
                "neighborhood" to neighborhoodOf(neighborhoods, lat, lon)?.name,
            )
        }
        return linkedMapOf(
            "query" to address, "results" to hits,
            "source" to "Nominatim (OpenStreetMap), live; data (c) OpenStreetMap contributors, ODbL",
        )
    }

    /** Address, street, ZIP, neighbourhood or restaurant name, from the snapshot alone. */
    fun offlineGeocode(address: String): List<Map<String, Any?>> {
        val first = address.split(',').first().trim()
        val m = Regex("^(\\d+[A-Za-z]?(?:-\\d+)?)\\s+(.+)$").find(first)
        fun hit(lat: Double, lon: Double, name: String, precision: String) = linkedMapOf<String, Any?>(
            "lat" to round5(lat), "lon" to round5(lon), "display_name" to name, "precision" to precision,
            "neighborhood" to neighborhoodOf(neighborhoods, lat, lon)?.name,
        )
        if (m != null) {
            val house = m.groupValues[1]
            val key = addressKey(house, m.groupValues[2])
            if (key != null) {
                val street = key.substringAfter('|')
                val pts = addressPoints()
                pts[key]?.let { (lat, lon, label) -> return listOf(hit(lat, lon, label, "address")) }
                val num = house.takeWhile { it.isDigit() }.toIntOrNull()
                val same = pts.entries.filter { it.key.substringAfter('|') == street }
                if (same.isNotEmpty() && num != null) {
                    val best = same.minBy { e ->
                        val n = e.key.substringBefore('|').takeWhile { it.isDigit() }.toIntOrNull() ?: Int.MAX_VALUE / 2
                        Math.abs(n - num) * 2 + (if ((n - num) % 2 != 0) 1 else 0)
                    }
                    return listOf(hit(best.value.first, best.value.second, best.value.third, "street (nearest known number)"))
                }
            }
        }
        Regex("\\b(112\\d\\d)\\b").find(address)?.let { z ->
            val zip = z.groupValues[1]
            val pts = query("SELECT avg(lat), avg(lon), count(*) FROM establishments WHERE zipcode=? AND lat IS NOT NULL", zip).first()
            if ((pts[2] as Number).toInt() > 0) {
                return listOf(hit((pts[0] as Number).toDouble(), (pts[1] as Number).toDouble(), "ZIP $zip", "zip"))
            }
        }
        val ns = resolveNeighborhood(neighborhoods, first)
        if (ns.isNotEmpty()) return ns.map { hit(it.center.first, it.center.second, "${it.name}, Brooklyn", "neighborhood") }
        val ids = ftsIds(first, 3)
        return ids.mapNotNull { byId[it] }.filter { it.lat != null }.map { hit(it.lat!!, it.lon!!, "${it.name}, ${it.address ?: "Brooklyn"}", "place") }
    }

    private var addrCache: Map<String, Triple<Double, Double, String>>? = null

    /** addressKey -> point and label, from DOHMH establishments and OSM places with an address. */
    private fun addressPoints(): Map<String, Triple<Double, Double, String>> = addrCache ?: run {
        val out = HashMap<String, Triple<Double, Double, String>>()
        for (row in query("SELECT housenumber, street, lat, lon, postcode FROM osm_places WHERE housenumber IS NOT NULL AND street IS NOT NULL")) {
            val k = addressKey(row[0] as String?, row[1] as String?) ?: continue
            out[k] = Triple((row[2] as Number).toDouble(), (row[3] as Number).toDouble(), "${row[0]} ${row[1]}, Brooklyn, NY" + (row[4]?.let { " $it" } ?: ""))
        }
        for (row in query("SELECT building, street, lat, lon, zipcode FROM establishments WHERE lat IS NOT NULL ORDER BY camis")) {
            val k = addressKey(row[0] as String?, row[1] as String?) ?: continue
            out[k] = Triple((row[2] as Number).toDouble(), (row[3] as Number).toDouble(), titleCase("${row[0]} ${row[1]}") + ", Brooklyn, NY" + (row[4]?.let { " $it" } ?: ""))
        }
        addrCache = out
        out
    }

    // ------------------------------------------------------------------ web_search

    private fun webSearch(q: String, limit: Int): Map<String, Any?> {
        if (!mode.online) {
            val hits = ftsIds(q, limit).mapNotNull { byId[it] }.map { r ->
                linkedMapOf(
                    "title" to r.name, "url" to (r.website ?: r.osmId?.let { "https://www.openstreetmap.org/$it" }),
                    "snippet" to listOfNotNull(r.cuisine ?: r.osmCuisine, r.address, r.grade?.let { "DOHMH grade $it" }).joinToString(" · "),
                    "id" to r.id,
                )
            }
            return linkedMapOf(
                "query" to q, "results" to hits,
                "source" to "offline stand-in: the snapshot's restaurant index (no web access in snapshot mode); snapshot $snapshotDate",
            )
        }
        val url = "${searxngUrl.trimEnd('/')}/search?q=${Http.enc(q)}&format=json&language=en-US&safesearch=1"
        val resp = try { net.get(url, "application/json") } catch (e: java.io.IOException) {
            throw ToolError("web search unavailable: SearXNG at $searxngUrl did not answer (examples/brooklyn-eats/searxng/searxng.sh start)")
        }
        if (resp.status != 200) throw ToolError("SearXNG returned HTTP ${resp.status}")
        val results = ((parseJson(resp.text) as JsonObject)["results"] as? JsonArray)?.elements.orEmpty().map { it as JsonObject }
        return linkedMapOf(
            "query" to q,
            "results" to results.take(limit).map { o ->
                linkedMapOf(
                    "title" to o.s("title"), "url" to o.s("url"), "snippet" to o.s("content")?.let { shorten(it, 300) },
                    "published" to o.s("publishedDate")?.take(10),
                    "engines" to (o["engines"] as? JsonArray)?.elements?.mapNotNull { (it as? JsonString)?.value },
                )
            },
            "source" to "SearXNG (self-hosted metasearch), live",
        )
    }

    // ------------------------------------------------------------------ recordings

    private fun recordingsDb(): Connection = recordings ?: run {
        if (mode == ToolMode.REPLAY && !Files.exists(recordingsPath)) throw ToolError("no recordings file at $recordingsPath")
        val c = DriverManager.getConnection("jdbc:sqlite:$recordingsPath")
        c.createStatement().use {
            it.execute("CREATE TABLE IF NOT EXISTS recordings(tool TEXT, args TEXT, result TEXT, recorded_at TEXT, PRIMARY KEY(tool, args))")
        }
        recordings = c
        c
    }

    private fun recorded(tool: String, args: String): String? = try {
        recordingsDb().prepareStatement("SELECT result FROM recordings WHERE tool=? AND args=?").use { ps ->
            ps.setString(1, tool); ps.setString(2, args)
            ps.executeQuery().use { rs -> if (rs.next()) rs.getString(1) else null }
        }
    } catch (e: ToolError) {
        null
    }

    private fun record(tool: String, args: String, result: String) {
        recordingsDb().prepareStatement("INSERT OR REPLACE INTO recordings VALUES(?,?,?,?)").use { ps ->
            ps.setString(1, tool); ps.setString(2, args); ps.setString(3, result); ps.setString(4, Instant.now().toString())
            ps.executeUpdate()
        }
    }

    // ------------------------------------------------------------------ SQL helpers

    fun query(sql: String, vararg params: Any?): List<List<Any?>> = db.prepareStatement(sql).use { ps ->
        params.forEachIndexed { i, p ->
            when (p) {
                null -> ps.setNull(i + 1, java.sql.Types.NULL)
                is String -> ps.setString(i + 1, p)
                is Int -> ps.setInt(i + 1, p)
                is Long -> ps.setLong(i + 1, p)
                is Double -> ps.setDouble(i + 1, p)
                else -> ps.setString(i + 1, p.toString())
            }
        }
        ps.executeQuery().use { rs ->
            val n = rs.metaData.columnCount
            buildList { while (rs.next()) add((1..n).map { rs.getObject(it) }) }
        }
    }

    private fun scalar(sql: String, vararg p: Any?): String? = query(sql, *p).firstOrNull()?.get(0)?.toString()
    private fun scalarLong(sql: String, vararg p: Any?): Long = (query(sql, *p).first()[0] as Number).toLong()
}

/**
 * "8002 5 Avenue" -> "8002 5th Avenue": DOHMH writes numbered streets as bare numbers, which
 * Nominatim does not find. The first number (the house number) is left alone.
 */
fun ordinalStreets(address: String): String =
    Regex("(?i)(?<=\\S\\s)(\\d+)(\\s+(?:avenue|ave|street|st|place|pl|road|rd|drive|dr|court|ct|terrace|walk|lane|ln)\\b)").replace(address) { m ->
        val n = m.groupValues[1].toInt()
        val suf = if (n % 100 in 11..13) "th" else when (n % 10) { 1 -> "st"; 2 -> "nd"; 3 -> "rd"; else -> "th" }
        "$n$suf${m.groupValues[2]}"
    }

fun round5(x: Double): Double = Math.round(x * 100_000) / 100_000.0

fun shorten(s: String, n: Int): String {
    val t = s.replace(Regex("\\s+"), " ").trim()
    return if (t.length <= n) t else t.substring(0, n - 1).trimEnd() + "…"
}
