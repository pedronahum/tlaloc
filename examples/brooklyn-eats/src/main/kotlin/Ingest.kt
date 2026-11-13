package brooklyn

import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.sql.Connection
import java.sql.DriverManager
import java.sql.PreparedStatement
import java.sql.Types
import java.time.Instant
import java.time.LocalDate
import java.util.zip.GZIPInputStream
import kotlin.random.Random

// The ingest stage: fetch the four sources into build/data/raw (kept between runs unless
// --refresh), then build build/data/brooklyn.db from the raw files alone. The build step never
// touches the network, so it runs the same on the synthetic test fixture.

/** Kings County (Brooklyn) administrative boundary in OpenStreetMap. */
const val KINGS_COUNTY_RELATION = 369518L

val OSM_AMENITIES = listOf("restaurant", "cafe", "fast_food", "bar", "pub", "ice_cream", "food_court")

val OVERPASS_ENDPOINTS = listOf(
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

const val SOCRATA = "https://data.cityofnewyork.us/resource"
const val GOOGLE_LOCAL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"
val COMPLAINT_TYPES = listOf("Food Poisoning", "Food Establishment")

fun overpassQuery(): String =
    "[out:json][timeout:300];rel($KINGS_COUNTY_RELATION);map_to_area->.b;" +
        "nwr(area.b)[\"amenity\"~\"^(${OSM_AMENITIES.joinToString("|")})$\"];out center tags;"

class IngestConfig(
    val dataDir: Path,
    val raw: Path = dataDir.resolve("raw"),
    val db: Path = dataDir.resolve("brooklyn.db"),
    val refresh: Boolean = false,
    val skipGoogle: Boolean = false,
    val sampleSize: Int = 20,
    val sampleSeed: Int = 7,
    /** Menus for the first [QUICK_MENU_PLACES] places only (run.sh --quick). */
    val quick: Boolean = false,
    /** Wayback snapshots are taken on or before this day. */
    val cutoff: LocalDate = LocalDate.parse(DEFAULT_CUTOFF),
    /** Committed pins of the archived menu pages (pins/wayback-menus.tsv). */
    val menuPins: Path? = null,
    val menuThreads: Int = 4,
    val log: (String) -> Unit = { println(it) },
)

/** Places whose menus the quick run fetches (the first by OSM id). */
const val QUICK_MENU_PLACES = 150

/** Lines of `raw/manifest.tsv`: source, fetched-at instant, URL. */
private fun recordFetch(raw: Path, source: String, url: String) {
    val f = raw.resolve("manifest.tsv")
    val kept = if (Files.exists(f)) Files.readAllLines(f).filterNot { it.startsWith("$source\t") } else emptyList()
    Files.write(f, kept + "$source\t${Instant.now()}\t$url")
}

private fun readManifest(raw: Path): Map<String, Pair<String, String>> {
    val f = raw.resolve("manifest.tsv")
    if (!Files.exists(f)) return emptyMap()
    return Files.readAllLines(f).mapNotNull { l ->
        val p = l.split('\t'); if (p.size < 3) null else p[0] to (p[1] to p[2])
    }.toMap()
}

/** Downloads every source that is not already in [IngestConfig.raw]. */
fun fetchSources(cfg: IngestConfig, http: Http = Http()) {
    Files.createDirectories(cfg.raw)
    val log = cfg.log
    val today = LocalDate.now()

    // OpenStreetMap through Overpass; tries the mirrors in turn.
    val osm = cfg.raw.resolve("overpass.json")
    if (cfg.refresh || !Files.exists(osm)) {
        var done = false
        for (ep in OVERPASS_ENDPOINTS) {
            try {
                log("fetch osm: $ep")
                val n = http.postForm(ep, mapOf("data" to overpassQuery()), osm)
                log("  ${n / 1024} KiB")
                recordFetch(cfg.raw, "osm", ep)
                done = true
                break
            } catch (e: Exception) {
                log("  $ep failed: ${e.message}")
            }
        }
        check(done) { "every Overpass endpoint failed" }
    }

    fun socrata(source: String, dataset: String, where: String, dir: Path) {
        if (!cfg.refresh && Files.exists(dir.resolve("complete"))) return
        if (Files.exists(dir)) dir.toFile().deleteRecursively()
        Files.createDirectories(dir)
        val page = 50_000
        var offset = 0
        var pages = 0
        while (true) {
            val url = "$SOCRATA/$dataset.json?\$where=${Http.enc(where)}&\$order=${Http.enc(":id")}" +
                "&\$limit=$page&\$offset=$offset"
            val f = dir.resolve("page-%03d.json".format(pages))
            http.download(url, f)
            val n = parseJsonArraySize(f)
            log("fetch $source: page $pages, $n rows")
            pages++
            offset += n
            if (n < page) break
        }
        Files.writeString(dir.resolve("complete"), "$offset\n")
        recordFetch(cfg.raw, source, "$SOCRATA/$dataset.json?\$where=$where")
    }
    socrata("inspections", "43nn-pn8j", "boro='Brooklyn'", cfg.raw.resolve("dohmh"))
    val since = today.minusYears(3)
    socrata(
        "complaints", "erm2-nwe9",
        "borough='BROOKLYN' AND complaint_type in(${COMPLAINT_TYPES.joinToString(",") { "'$it'" }}) " +
            "AND created_date >= '${since}T00:00:00'",
        cfg.raw.resolve("311"),
    )

    // Neighbourhood names and polygons (2020 NTAs, a fixed release), Brooklyn only.
    val nta = cfg.raw.resolve("nta2020.json")
    if (cfg.refresh || !Files.exists(nta)) {
        val url = NTA_URL
        val n = http.download(url, nta)
        log("fetch nta2020: ${n / 1024} KiB")
        recordFetch(cfg.raw, "nta2020", url)
    }

    fetchMenus(cfg)

    if (!cfg.skipGoogle) {
        val g = cfg.raw.resolve("googlelocal")
        for (name in listOf("meta-New_York.json.gz", "review-New_York.json.gz")) {
            val f = g.resolve(name)
            if (!cfg.refresh && Files.exists(f)) continue
            log("fetch google local: $name")
            val n = http.download("$GOOGLE_LOCAL/$name", f, java.time.Duration.ofHours(2))
            log("  ${n / (1024 * 1024)} MiB")
        }
        recordFetch(cfg.raw, "google_local", "$GOOGLE_LOCAL/{meta,review}-New_York.json.gz")
    }
}

/**
 * Archived menu pages of the OSM places with a menu URL or website (Wayback.kt). Uses the
 * committed pins; discovers and pins pages that have none; writes the pins of the pages used to
 * raw/wayback/pins.tsv (what the build reads) and, on a full run, back to the committed file.
 */
fun fetchMenus(cfg: IngestConfig, net: Net = HttpNet(minSpacingMs = 1_000L, maxAttempts = 5, timeout = java.time.Duration.ofSeconds(60), firstBackoffMs = 10_000L, log = cfg.log)) {
    val log = cfg.log
    val all = menuTargets(parseOverpass(Files.readString(cfg.raw.resolve("overpass.json"))))
    val targets = if (cfg.quick) all.take(QUICK_MENU_PLACES) else all
    val committed = cfg.menuPins?.let { readPins(it) }.orEmpty()
    val dir = cfg.raw.resolve("wayback")
    log("fetch menus: ${targets.size} of ${all.size} places with a menu URL or website; ${committed.size} pinned pages; cutoff ${cfg.cutoff}")
    val t0 = System.nanoTime()
    val (used, stats) = fetchWaybackMenus(
        targets, net, cutoffTimestamp(cfg.cutoff), committed, dir.resolve("pages"), discover = true, threads = cfg.menuThreads, log = log,
    )
    // Transient failures are not pinned, so a later run tries them again.
    val keep = used.values.filter { it.status != "error" && !it.status.startsWith("http_5") && it.status != "http_429" }
    writePins(dir.resolve("pins.tsv"), keep)
    if (!cfg.quick && cfg.menuPins != null) {
        writePins(cfg.menuPins, (committed + keep.associateBy { it.url }).values)
        log("  pins written to ${cfg.menuPins}")
    }
    log("fetch menus: %.0f s; $stats; ${used.size - keep.size} transient failures not pinned".format((System.nanoTime() - t0) / 1e9))
    recordFetch(cfg.raw, "menus", "$WAYBACK_AVAILABLE (cutoff ${cfg.cutoff})")
}

private fun parseJsonArraySize(f: Path): Int =
    (io.tlaloc.core.io.parseJson(Files.readString(f)) as io.tlaloc.core.io.JsonArray).size

private fun openText(f: Path): BufferedReader {
    val ins = Files.newInputStream(f)
    val s = if (f.fileName.toString().endsWith(".gz")) GZIPInputStream(ins, 1 shl 16) else ins
    return BufferedReader(InputStreamReader(s, Charsets.UTF_8), 1 shl 20)
}

/** The Google Local file [name] under [dir], gzipped or plain (the fixture is plain). */
private fun googleFile(dir: Path, name: String): Path? =
    listOf("$name.gz", name).map { dir.resolve(it) }.firstOrNull { Files.exists(it) }

private val SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE osm_places(osm_id TEXT PRIMARY KEY, name TEXT, amenity TEXT, cuisine TEXT, lat REAL, lon REAL,
  housenumber TEXT, street TEXT, postcode TEXT, phone TEXT, website TEXT, menu_url TEXT, opening_hours TEXT, tags_json TEXT);
CREATE TABLE establishments(camis TEXT PRIMARY KEY, dba TEXT, building TEXT, street TEXT, zipcode TEXT, phone TEXT,
  cuisine TEXT, lat REAL, lon REAL, bbl TEXT, nta TEXT, latest_inspection TEXT, latest_score INTEGER,
  grade TEXT, grade_date TEXT);
CREATE TABLE inspections(camis TEXT, inspection_date TEXT, inspection_type TEXT, action TEXT, violation_code TEXT,
  violation_description TEXT, critical_flag TEXT, score INTEGER, grade TEXT, grade_date TEXT);
CREATE INDEX inspections_camis ON inspections(camis, inspection_date);
CREATE TABLE complaints(unique_key TEXT PRIMARY KEY, created_date TEXT, closed_date TEXT, complaint_type TEXT,
  descriptor TEXT, descriptor_2 TEXT, location_type TEXT, incident_address TEXT, incident_zip TEXT, bbl TEXT,
  status TEXT, resolution TEXT, lat REAL, lon REAL, camis TEXT, link_rule TEXT);
CREATE INDEX complaints_camis ON complaints(camis);
CREATE INDEX complaints_bbl ON complaints(bbl);
CREATE TABLE g_places(gmap_id TEXT PRIMARY KEY, name TEXT, address TEXT, lat REAL, lon REAL, categories TEXT,
  avg_rating REAL, num_reviews INTEGER, price TEXT, state TEXT, description TEXT, hours TEXT);
CREATE TABLE reviews(gmap_id TEXT, time_ms INTEGER, rating INTEGER, text TEXT, response TEXT);
CREATE INDEX reviews_gmap ON reviews(gmap_id, time_ms);
CREATE TABLE restaurants(id TEXT PRIMARY KEY, camis TEXT, osm_id TEXT, gmap_id TEXT, name TEXT, dba TEXT,
  osm_name TEXT, g_name TEXT, cuisine TEXT, osm_cuisine TEXT, amenity TEXT, address TEXT, zipcode TEXT,
  phone TEXT, lat REAL, lon REAL, website TEXT, menu_url TEXT, opening_hours TEXT, nta TEXT,
  grade TEXT, grade_date TEXT, latest_inspection TEXT, latest_score INTEGER,
  g_rating REAL, g_num_reviews INTEGER, g_price TEXT, g_state TEXT, g_hours TEXT, nta2020 TEXT, neighborhood TEXT);
CREATE INDEX restaurants_camis ON restaurants(camis);
CREATE INDEX restaurants_gmap ON restaurants(gmap_id);
CREATE TABLE matches(kind TEXT, left_id TEXT, right_id TEXT, name_score REAL, distance_m REAL, rule TEXT);
CREATE TABLE menus(restaurant_id TEXT, osm_id TEXT, url TEXT, page_url TEXT, snapshot TEXT, archive_url TEXT,
  status TEXT, items INTEGER, text TEXT);
CREATE INDEX menus_restaurant ON menus(restaurant_id);
CREATE TABLE neighborhoods(code TEXT PRIMARY KEY, name TEXT, cdta TEXT, type TEXT, lat REAL, lon REAL,
  min_lat REAL, max_lat REAL, min_lon REAL, max_lon REAL, geom TEXT);
CREATE VIRTUAL TABLE restaurants_fts USING fts5(id UNINDEXED, name, cuisine, address);
CREATE VIRTUAL TABLE violations_fts USING fts5(violation_description, content='inspections', content_rowid='rowid');
CREATE VIRTUAL TABLE reviews_fts USING fts5(text, content='reviews', content_rowid='rowid');
CREATE VIRTUAL TABLE menus_fts USING fts5(text, content='menus', content_rowid='rowid');
"""

private fun PreparedStatement.bind(vararg vs: Any?) {
    vs.forEachIndexed { i, v ->
        val k = i + 1
        when (v) {
            null -> setNull(k, Types.NULL)
            is String -> setString(k, v)
            is Int -> setInt(k, v)
            is Long -> setLong(k, v)
            is Double -> setDouble(k, v)
            else -> setString(k, v.toString())
        }
    }
}

private fun Connection.insert(sql: String, rows: Sequence<Array<Any?>>): Int {
    var n = 0
    prepareStatement(sql).use { ps ->
        for (r in rows) {
            ps.bind(*r); ps.addBatch(); n++
            if (n % 5_000 == 0) ps.executeBatch()
        }
        ps.executeBatch()
    }
    return n
}

private fun Connection.count(sql: String): Long = createStatement().use { st ->
    st.executeQuery(sql).use { rs -> rs.next(); rs.getLong(1) }
}

/** Title case for the upper-case DOHMH names: "JOE'S PIZZA" -> "Joe's Pizza". */
fun titleCase(s: String): String = s.lowercase().split(' ').joinToString(" ") { w ->
    w.replaceFirstChar { it.uppercaseChar() }
}

data class IngestReport(
    val counts: Map<String, Long>,
    val matchRates: Map<String, String>,
    val samples: Map<String, List<String>>,
)

/** Builds [IngestConfig.db] from the raw files. Returns counts, match rates and the match samples. */
fun buildDatabase(cfg: IngestConfig): IngestReport {
    val log = cfg.log
    val tmp = cfg.db.resolveSibling(cfg.db.fileName.toString() + ".tmp")
    Files.deleteIfExists(tmp)
    Files.createDirectories(cfg.db.toAbsolutePath().parent)
    val manifest = readManifest(cfg.raw)
    val counts = LinkedHashMap<String, Long>()
    val rates = LinkedHashMap<String, String>()
    val samples = LinkedHashMap<String, List<String>>()
    fun rate(name: String, n: Int, d: Int) {
        rates[name] = "$n / $d (%.1f%%)".format(if (d == 0) 0.0 else 100.0 * n / d)
    }

    DriverManager.getConnection("jdbc:sqlite:$tmp").use { c ->
        c.createStatement().use { st ->
            st.execute("PRAGMA journal_mode=OFF")
            st.execute("PRAGMA synchronous=OFF")
            for (stmt in SCHEMA.split(';').map { it.trim() }.filter { it.isNotEmpty() }) st.execute(stmt)
        }
        c.autoCommit = false

        // --- OpenStreetMap
        val osm = parseOverpass(Files.readString(cfg.raw.resolve("overpass.json")))
            .distinctBy { it.osmId }
        c.insert(
            "INSERT INTO osm_places VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            osm.asSequence().map {
                arrayOf(
                    it.osmId, it.name, it.amenity, it.cuisine, it.lat, it.lon, it.houseNumber, it.street,
                    it.postcode, it.phone, it.website, it.menuUrl, it.openingHours,
                    it.tags.entries.joinToString(",", "{", "}") { (k, v) ->
                        io.tlaloc.core.io.jsonQuote(k) + ":" + io.tlaloc.core.io.jsonQuote(v)
                    },
                )
            },
        )
        log("osm: ${osm.size} places (${osm.count { it.name != null }} named)")

        // --- DOHMH inspections
        val rows = ArrayList<InspectionRow>()
        Files.list(cfg.raw.resolve("dohmh")).use { s ->
            s.filter { it.fileName.toString().endsWith(".json") }.sorted().forEach { rows += parseInspections(Files.readString(it)) }
        }
        c.insert(
            "INSERT INTO inspections VALUES(?,?,?,?,?,?,?,?,?,?)",
            rows.asSequence().map {
                arrayOf(
                    it.camis, it.inspectionDate, it.inspectionType, it.action, it.violationCode,
                    it.violationDescription, it.criticalFlag, it.score, it.grade, it.gradeDate,
                )
            },
        )
        val est = rows.groupBy { it.camis }.map { (camis, rs) ->
            val first = rs.first()
            val inspected = rs.filter { it.inspectionDate != null && !it.inspectionDate.startsWith("1900") }
            val latest = inspected.maxByOrNull { it.inspectionDate!! }
            val latestScore = inspected.filter { it.score != null }.maxByOrNull { it.inspectionDate!! }?.score
            val graded = rs.filter { it.grade != null && it.gradeDate != null }.maxByOrNull { it.gradeDate!! }
            val coord = rs.firstOrNull { it.lat != null }
            arrayOf<Any?>(
                camis, first.dba, first.building, first.street, first.zipcode, first.phone, first.cuisine,
                coord?.lat, coord?.lon, first.bbl, first.nta, latest?.inspectionDate, latestScore,
                graded?.grade, graded?.gradeDate,
            )
        }
        c.insert("INSERT INTO establishments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", est.asSequence())
        log("dohmh: ${rows.size} inspection rows, ${est.size} establishments")
        val estById = est.associateBy { it[0] as String }

        // --- DOHMH <-> OSM
        val left = est.map {
            MatchEntity(
                it[0] as String, listOfNotNull(it[1] as String?), it[7] as Double?, it[8] as Double?,
                addressKey(it[2] as String?, it[3] as String?),
            )
        }.filter { it.names.isNotEmpty() }
        val osmNamed = osm.filter { it.name != null }
        val right = osmNamed.map { p ->
            MatchEntity(
                p.osmId,
                listOfNotNull(p.name, p.tags["brand"], p.tags["official_name"], p.tags["alt_name"]).distinct(),
                p.lat, p.lon, addressKey(p.houseNumber, p.street),
            )
        }
        val dohmhOsm = matchEntities(left, right)
        val osmById = osm.associateBy { it.osmId }
        val camisToOsm = dohmhOsm.associate { it.left to it.right }
        rate("dohmh->osm (establishments with an OSM match)", dohmhOsm.size, est.size)
        rate("osm->dohmh (named OSM places with a CAMIS)", dohmhOsm.size, osmNamed.size)
        log("match dohmh<->osm: ${dohmhOsm.size} pairs (${dohmhOsm.groupingBy { it.rule }.eachCount()})")

        // --- restaurants: one per CAMIS, plus named OSM places without one
        data class R(
            val id: String, val camis: String?, val osmId: String?, val name: String, val dba: String?,
            val osmName: String?, val cuisine: String?, val osmCuisine: String?, val amenity: String?,
            val address: String?, val zip: String?, val phone: String?, val lat: Double?, val lon: Double?,
            val website: String?, val menuUrl: String?, val hours: String?, val nta: String?, val grade: String?,
            val gradeDate: String?, val latestInspection: String?, val latestScore: Int?, val addrKey: String?,
        )
        val restaurants = ArrayList<R>()
        for (e in est) {
            val camis = e[0] as String
            val o = camisToOsm[camis]?.let { osmById[it] }
            val dba = e[1] as String?
            val addr = listOfNotNull(e[2] as String?, e[3] as String?).joinToString(" ").takeIf { it.isNotBlank() }
            restaurants += R(
                id = camis, camis = camis, osmId = o?.osmId, name = o?.name ?: dba?.let(::titleCase) ?: camis,
                dba = dba, osmName = o?.name, cuisine = e[6] as String?, osmCuisine = o?.cuisine,
                amenity = o?.amenity,
                address = addr?.let { titleCase(it) + ", Brooklyn, NY" + ((e[4] as String?)?.let { z -> " $z" } ?: "") },
                zip = e[4] as String?, phone = e[5] as String? ?: o?.phone,
                lat = e[7] as Double? ?: o?.lat, lon = e[8] as Double? ?: o?.lon,
                website = o?.website, menuUrl = o?.menuUrl, hours = o?.openingHours, nta = e[10] as String?,
                grade = e[13] as String?, gradeDate = e[14] as String?, latestInspection = e[11] as String?,
                latestScore = e[12] as Int?, addrKey = addressKey(e[2] as String?, e[3] as String?),
            )
        }
        val matchedOsm = dohmhOsm.map { it.right }.toSet()
        for (o in osmNamed) {
            if (o.osmId in matchedOsm) continue
            val street = listOfNotNull(o.houseNumber, o.street).joinToString(" ").takeIf { it.isNotBlank() }
            restaurants += R(
                id = "osm:${o.osmId}", camis = null, osmId = o.osmId, name = o.name!!, dba = null, osmName = o.name,
                cuisine = null, osmCuisine = o.cuisine, amenity = o.amenity,
                address = street?.let { "$it, Brooklyn, NY" + (o.postcode?.let { z -> " $z" } ?: "") },
                zip = o.postcode, phone = o.phone, lat = o.lat, lon = o.lon, website = o.website,
                menuUrl = o.menuUrl, hours = o.openingHours, nta = null, grade = null, gradeDate = null,
                latestInspection = null, latestScore = null, addrKey = addressKey(o.houseNumber, o.street),
            )
        }

        // --- Google Local: Brooklyn food businesses from the metadata, matched to restaurants
        val gDir = cfg.raw.resolve("googlelocal")
        val gMeta = googleFile(gDir, "meta-New_York.json")
        val gReviews = googleFile(gDir, "review-New_York.json")
        val gPlaces = ArrayList<GPlace>()
        var gLines = 0L
        var gBad = 0L
        val restToG = HashMap<String, String>()
        val gById = HashMap<String, GPlace>()
        var gNote: String? = null
        if (cfg.skipGoogle || gMeta == null || gReviews == null) {
            gNote = if (cfg.skipGoogle) "skipped (--skip-google)" else "files not present under ${gDir}"
            log("google local: $gNote; reviews table left empty")
        } else {
            openText(gMeta).use { r ->
                r.lineSequence().forEach { line ->
                    gLines++
                    if (!line.contains("Brooklyn") && !line.contains("NY 112")) return@forEach
                    val p = parseGPlace(line)
                    if (p == null) { gBad++; return@forEach }
                    if (isBrooklynAddress(p.address) && isFoodCategory(p.categories)) gPlaces += p
                }
            }
            val unique = gPlaces.distinctBy { it.gmapId }
            gPlaces.clear(); gPlaces += unique
            gPlaces.forEach { gById[it.gmapId] = it }
            c.insert(
                "INSERT INTO g_places VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                gPlaces.asSequence().map {
                    arrayOf(
                        it.gmapId, it.name, it.address, it.lat, it.lon, it.categories.joinToString("; "),
                        it.avgRating, it.numReviews, it.price, it.state, it.description, it.hours,
                    )
                },
            )
            log("google local: $gLines metadata lines, $gBad unparsed among Brooklyn candidates, ${gPlaces.size} Brooklyn food businesses")
            val gLeft = gPlaces.map { p ->
                val hs = houseAndStreetOf(p.address)
                MatchEntity(p.gmapId, listOf(p.name), p.lat, p.lon, addressKey(hs?.first, hs?.second))
            }
            val rRight = restaurants.map {
                MatchEntity(it.id, listOfNotNull(it.dba, it.osmName).ifEmpty { listOf(it.name) }, it.lat, it.lon, it.addrKey)
            }
            val gm = matchEntities(gLeft, rRight)
            for (m in gm) restToG[m.right] = m.left
            c.insert(
                "INSERT INTO matches VALUES(?,?,?,?,?,?)",
                gm.asSequence().map { arrayOf("google-restaurant", it.left, it.right, it.nameScore, it.distanceM, it.rule) },
            )
            rate("google->restaurant (Brooklyn food businesses matched)", gm.size, gPlaces.size)
            rate("dohmh establishments with Google reviews", gm.count { !it.right.startsWith("osm:") }, est.size)
            log("match google<->restaurants: ${gm.size} pairs (${gm.groupingBy { it.rule }.eachCount()})")
            samples["google-restaurant"] = sample(gm, cfg) { m ->
                val g = gById[m.left]!!
                val r = restaurants.first { it.id == m.right }
                listOf(
                    "%.2f".format(m.nameScore), m.distanceM?.let { "%.0f".format(it) } ?: "-", m.rule,
                    g.name, g.address ?: "", r.dba ?: "", r.osmName ?: "", r.address ?: "",
                ).joinToString("\t")
            }
        }
        c.insert(
            "INSERT INTO matches VALUES(?,?,?,?,?,?)",
            dohmhOsm.asSequence().map { arrayOf("dohmh-osm", it.left, it.right, it.nameScore, it.distanceM, it.rule) },
        )
        samples["dohmh-osm"] = sample(dohmhOsm, cfg) { m ->
            val e = estById[m.left]!!
            val o = osmById[m.right]!!
            listOf(
                "%.2f".format(m.nameScore), m.distanceM?.let { "%.0f".format(it) } ?: "-", m.rule,
                e[1] as String? ?: "", "${e[2]} ${e[3]} ${e[4]}", o.name ?: "",
                listOfNotNull(o.houseNumber, o.street, o.postcode).joinToString(" "),
            ).joinToString("\t")
        }

        // --- neighbourhoods: the 2020 NTA polygon holding each restaurant
        val ntaFile = cfg.raw.resolve("nta2020.json")
        val ntas = if (Files.exists(ntaFile)) parseNtas(Files.readString(ntaFile)) else emptyList()
        c.insert(
            "INSERT INTO neighborhoods VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ntas.asSequence().map {
                arrayOf(
                    it.code, it.name, it.cdta, it.type, it.center.first, it.center.second,
                    it.minLat, it.maxLat, it.minLon, it.maxLon, it.geometryJson(),
                )
            },
        )
        val ntaOf = restaurants.associate { it.id to neighborhoodOf(ntas, it.lat, it.lon) }
        log("neighborhoods: ${ntas.size} NTAs; ${ntaOf.values.count { it != null }} of ${restaurants.size} restaurants placed")

        c.insert(
            "INSERT INTO restaurants VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            restaurants.asSequence().map { r ->
                val g = restToG[r.id]?.let { gById[it] }
                val n = ntaOf[r.id]
                arrayOf(
                    r.id, r.camis, r.osmId, g?.gmapId, r.name, r.dba, r.osmName, g?.name, r.cuisine, r.osmCuisine,
                    r.amenity, r.address, r.zip, r.phone, r.lat, r.lon, r.website, r.menuUrl, r.hours, r.nta,
                    r.grade, r.gradeDate, r.latestInspection, r.latestScore, g?.avgRating, g?.numReviews, g?.price,
                    g?.state, g?.hours, n?.code, n?.name,
                )
            },
        )
        c.insert(
            "INSERT INTO restaurants_fts(id, name, cuisine, address) VALUES(?,?,?,?)",
            restaurants.asSequence().map { r ->
                val g = restToG[r.id]?.let { gById[it] }
                arrayOf(
                    r.id,
                    listOfNotNull(r.name, r.dba?.let(::titleCase), r.osmName, g?.name).distinct().joinToString(" | "),
                    listOfNotNull(r.cuisine, r.osmCuisine?.replace(';', ' ')?.replace('_', ' '), r.amenity?.replace('_', ' '))
                        .joinToString(" "),
                    r.address,
                )
            },
        )
        log("restaurants: ${restaurants.size} (${est.size} with a CAMIS, ${restaurants.size - est.size} OSM only)")

        // --- Google Local reviews, stream-filtered to the matched businesses
        if (gNote == null) {
            val keep = restToG.values.toHashSet()
            var lines = 0L
            var kept = 0L
            c.prepareStatement("INSERT INTO reviews VALUES(?,?,?,?,?)").use { ps ->
                openText(gReviews!!).use { r ->
                    r.lineSequence().forEach { line ->
                        lines++
                        if (lines % 5_000_000 == 0L) log("  reviews: $lines lines read, $kept kept")
                        val id = gmapIdOfLine(line) ?: return@forEach
                        if (id !in keep) return@forEach
                        val rv = parseGReview(line) ?: return@forEach
                        ps.bind(rv.gmapId, rv.timeMs, rv.rating, rv.text, rv.response)
                        ps.addBatch()
                        kept++
                        if (kept % 10_000 == 0L) ps.executeBatch()
                    }
                }
                ps.executeBatch()
            }
            counts["google local review lines read"] = lines
            log("reviews: $lines lines read, $kept kept")
        }

        // --- 311 complaints, linked to a CAMIS by address (unique) or by BBL (unique)
        val complaints = ArrayList<Complaint>()
        Files.list(cfg.raw.resolve("311")).use { s ->
            s.filter { it.fileName.toString().endsWith(".json") }.sorted().forEach { complaints += parseComplaints(Files.readString(it)) }
        }
        val byAddr = HashMap<String, MutableSet<String>>()
        val byBbl = HashMap<String, MutableSet<String>>()
        for (e in est) {
            val camis = e[0] as String
            addressKey(e[2] as String?, e[3] as String?)?.let { byAddr.getOrPut(it + "|" + e[4]) { HashSet() } += camis }
            (e[9] as String?)?.takeIf { it.length == 10 && !it.endsWith("0000") }?.let { byBbl.getOrPut(it) { HashSet() } += camis }
        }
        var linkedAddr = 0
        var linkedBbl = 0
        c.insert(
            "INSERT INTO complaints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            complaints.distinctBy { it.uniqueKey }.asSequence().map { x ->
                val hs = x.incidentAddress?.let { a ->
                    val i = a.indexOf(' '); if (i > 0) a.substring(0, i) to a.substring(i + 1) else null
                }
                val ak = addressKey(hs?.first, hs?.second)?.let { it + "|" + x.incidentZip }
                val a = ak?.let { byAddr[it] }?.singleOrNull()
                val b = if (a == null) x.bbl?.let { byBbl[it] }?.singleOrNull() else null
                val rule = when { a != null -> "address"; b != null -> "bbl"; else -> null }
                if (a != null) linkedAddr++
                if (b != null) linkedBbl++
                arrayOf(
                    x.uniqueKey, x.createdDate, x.closedDate, x.complaintType, x.descriptor, x.descriptor2,
                    x.locationType, x.incidentAddress, x.incidentZip, x.bbl, x.status, x.resolution, x.lat, x.lon,
                    a ?: b, rule,
                )
            },
        )
        val nComplaints = c.count("SELECT count(*) FROM complaints").toInt()
        rate("311 complaints linked to a CAMIS", linkedAddr + linkedBbl, nComplaints)
        rates["311 link rule"] = "address $linkedAddr, bbl $linkedBbl"

        // --- menus from the archived pages in raw/wayback (no network)
        val wb = cfg.raw.resolve("wayback")
        val pins = readPins(wb.resolve("pins.tsv"))
        val restByOsm = restaurants.filter { it.osmId != null }.associateBy { it.osmId!! }
        val menuRows = menuTargets(osm).filter { it.url in pins }.mapNotNull { t ->
            val r = restByOsm[t.osmId] ?: return@mapNotNull null
            val m = menuFromPages(t.url) { u -> pageFromStore(pins[u], wb.resolve("pages"), u) }
            arrayOf<Any?>(r.id, t.osmId, t.url, m.url, m.snapshot?.let(::waybackIso), m.archiveUrl, m.status, m.items, m.text)
        }
        c.insert("INSERT INTO menus VALUES(?,?,?,?,?,?,?,?,?)", menuRows.asSequence())
        val menuStatus = menuRows.groupingBy { it[6] as String }.eachCount().toSortedMap()
        log("menus: ${menuRows.size} places looked up in the archive; $menuStatus")
        rate("places with a website whose archived menu was read", menuStatus["ok"] ?: 0, menuRows.size)

        // --- FTS over external-content tables
        c.createStatement().use { st ->
            st.execute("INSERT INTO violations_fts(violations_fts) VALUES('rebuild')")
            st.execute("INSERT INTO reviews_fts(reviews_fts) VALUES('rebuild')")
            st.execute("INSERT INTO menus_fts(menus_fts) VALUES('rebuild')")
        }

        // --- meta
        val snapshot = manifest["inspections"]?.first?.substring(0, 10) ?: LocalDate.now().toString()
        val meta = linkedMapOf(
            "snapshot_date" to snapshot,
            "menu_cutoff" to cfg.cutoff.toString(),
            "menu_scope" to if (pins.isEmpty()) "none" else if (cfg.quick) "quick" else "full",
            "built_at" to Instant.now().toString(),
            "google_local" to (gNote ?: "loaded"),
            "match_rules" to MatchRules().toString(),
        )
        for ((src, v) in manifest) { meta["fetched_$src"] = v.first; meta["url_$src"] = v.second }
        for (t in listOf("osm_places", "establishments", "inspections", "complaints", "g_places", "reviews", "restaurants", "menus")) {
            counts[t] = c.count("SELECT count(*) FROM $t")
        }
        counts["restaurants with a CAMIS"] = c.count("SELECT count(*) FROM restaurants WHERE camis IS NOT NULL")
        counts["restaurants with reviews"] = c.count("SELECT count(DISTINCT r.id) FROM restaurants r JOIN reviews v ON v.gmap_id = r.gmap_id")
        counts["reviews with text"] = c.count("SELECT count(*) FROM reviews WHERE text IS NOT NULL")
        counts["neighborhoods"] = c.count("SELECT count(*) FROM neighborhoods")
        counts["restaurants with a menu"] = c.count("SELECT count(DISTINCT restaurant_id) FROM menus WHERE status = 'ok'")
        counts["restaurants with a neighborhood"] = c.count("SELECT count(*) FROM restaurants WHERE nta2020 IS NOT NULL")
        counts["critical violation rows"] = c.count("SELECT count(*) FROM inspections WHERE critical_flag = 'Critical'")
        for ((k, v) in counts) meta["count_$k"] = v.toString()
        for ((k, v) in rates) meta["rate_$k"] = v
        c.insert("INSERT INTO meta VALUES(?,?)", meta.entries.asSequence().map { arrayOf(it.key, it.value) })
        c.commit()
        c.autoCommit = true
        c.createStatement().use { it.execute("ANALYZE") }
    }
    Files.move(tmp, cfg.db, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    return IngestReport(counts, rates, samples)
}

private fun sample(pairs: List<MatchPair>, cfg: IngestConfig, fmt: (MatchPair) -> String): List<String> =
    pairs.shuffled(Random(cfg.sampleSeed)).take(cfg.sampleSize).map(fmt)

/** BROOKLYN_DATA, or build/data under the current directory (run.sh runs from the example). */
fun dataHome(): Path = Paths.get(System.getenv("BROOKLYN_DATA") ?: "build/data")

/** `ingest [--refresh] [--skip-google] [--build-only] [--quick] [--data DIR]` (BROOKLYN_QUICK=1 is --quick) */
fun ingest(args: List<String>): Int {
    val dataDir = args.indexOf("--data").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) }
        ?: dataHome()
    val cfg = IngestConfig(
        dataDir = dataDir.toAbsolutePath().normalize(),
        refresh = "--refresh" in args,
        skipGoogle = "--skip-google" in args,
        quick = isQuick(args),
        cutoff = manifestCutoff(),
        menuPins = exampleHome().resolve(MENU_PINS),
    )
    val t0 = System.nanoTime()
    if ("--build-only" !in args) fetchSources(cfg)
    val t1 = System.nanoTime()
    val report = buildDatabase(cfg)
    val t2 = System.nanoTime()
    println()
    println("database: ${cfg.db} (${Files.size(cfg.db) / (1024 * 1024)} MiB)")
    println("fetch %.0f s, build %.0f s".format((t1 - t0) / 1e9, (t2 - t1) / 1e9))
    println("counts:")
    for ((k, v) in report.counts) println("  %-34s %,d".format(k, v))
    println("match rates:")
    for ((k, v) in report.matchRates) println("  %-52s %s".format(k, v))
    val sampleFile = cfg.dataDir.resolve("match-sample.tsv")
    Files.write(
        sampleFile,
        report.samples.flatMap { (kind, rows) -> rows.mapIndexed { i, r -> "$kind\t${i + 1}\t$r" } },
    )
    println("match samples (${report.samples.values.sumOf { it.size }} rows): $sampleFile")
    return 0
}
