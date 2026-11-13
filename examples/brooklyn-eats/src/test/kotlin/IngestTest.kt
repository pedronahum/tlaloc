package brooklyn

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.sql.DriverManager
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class IngestTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent

    @Test
    fun overpassKeepsAmenityElementsWithCoordinates() {
        val places = parseOverpass(Files.readString(fixture.resolve("overpass.json")))
        assertEquals(listOf("node/1", "way/2", "node/3", "node/4", "node/5", "node/6"), places.map { it.osmId })
        val vela = places.first { it.osmId == "way/2" }
        assertEquals(40.68185, vela.lat) // a way takes its center
        assertEquals("pizza;italian", vela.cuisine)
        val joes = places.first { it.osmId == "node/1" }
        assertEquals("216", joes.houseNumber)
        assertEquals("https://example.org/joes", joes.website)
    }

    @Test
    fun inspectionsParseDatesScoresAndDropZeroCoordinates() {
        val rows = parseInspections(Files.readString(fixture.resolve("dohmh/page-000.json")))
        assertEquals(6, rows.size)
        assertEquals("2025-03-01", rows[0].inspectionDate)
        assertEquals(12, rows[0].score)
        assertEquals("Critical", rows[0].criticalFlag)
        val newPlace = rows.first { it.camis == "1005" }
        assertNull(newPlace.lat)
        assertEquals("1900-01-01", newPlace.inspectionDate)
    }

    @Test
    fun complaintsParse() {
        val c = parseComplaints(Files.readString(fixture.resolve("311/page-000.json")))
        assertEquals(3, c.size)
        assertNull(c[0].descriptor2) // "N/A" is missing
        assertEquals("Letter Grade Missing/Not Visible", c[2].descriptor2)
    }

    @Test
    fun googleLinesParseFilterAndDropReviewerIdentity() {
        val lines = Files.readAllLines(fixture.resolve("googlelocal/meta-New_York.json"))
        val places = lines.map(::parseGPlace)
        assertNull(places[4]) // NaN is not JSON
        val kept = places.filterNotNull().filter { isBrooklynAddress(it.address) && isFoodCategory(it.categories) }
        assertEquals(listOf("g1", "g4"), kept.map { it.gmapId })
        assertFalse(isFoodCategory(listOf("Barber shop")))
        assertFalse(isFoodCategory(listOf("Restaurant supply store")))
        assertTrue(isFoodCategory(listOf("Wine bar")))
        val review = Files.readAllLines(fixture.resolve("googlelocal/review-New_York.json"))[2]
        assertEquals("g4", gmapIdOfLine(review))
        val r = parseGReview(review)!!
        assertEquals("Thank you!", r.response)
        assertFalse(r.toString().contains("Reviewer"))
        assertEquals("575" to "Henry St", houseAndStreetOf("Vela Forno, 575 Henry St, Brooklyn, NY 11231"))
    }

    @Test
    fun namesAndAddressesNormalize() {
        assertEquals(1.0, nameSimilarity("TOMASSO'S SLICE", "Tomasso's Slice"))
        assertEquals(1.0, nameSimilarity("GOLDEN DRAGON CHINESE RESTAURANT", "Golden Dragon Chinese"))
        assertTrue(nameSimilarity("TOMASSO", "Tomasso's Slice") >= 0.9) // contained, with a distinctive word
        assertTrue(nameSimilarity("El Arco", "EL ARCOIRIS DELI GROCERY") < 0.8) // whole words only
        assertTrue(nameSimilarity("PIZZA", "Pizza Hut") < 0.65) // generic word alone does not identify
        assertTrue(nameSimilarity("GOLDEN DRAGON CHINESE RESTAURANT", "Church Ave Deli") < 0.5)
        assertEquals(1.0, nameSimilarity("CAFÉ MOGADOR", "Cafe Mogador"))
        assertEquals(1.0, coreSimilarity("Kam Fung Kitchen", "KAM FUNG CHINESE RESTAURANT"))
        assertTrue(coreSimilarity("Zhangs Garden", "SUN GARDEN") < 0.5) // same address, a different business
        assertTrue(coreSimilarity("Little Toninos Pizzeria & Cafe", "COZIE'S PIZZERIA") < 0.5)
        assertEquals(0.0, coreSimilarity("No 1", "NO. 1 CHINESE RESTAURANT")) // nothing distinctive to compare
        assertTrue(coreSimilarity("China Tang Kitchen", "NEW CHINA RESTAURANT") < 0.75) // a proper subset
        assertEquals(1.0, coreSimilarity("KUMO JAPANESE CUISINE", "Kumo"))
        assertEquals("575|henry st", addressKey("575", "HENRY STREET"))
        assertEquals(addressKey("575", "Henry St"), addressKey("575", "HENRY STREET"))
        assertEquals(addressKey("10", "East 1st Street"), addressKey("10", "E 1 ST"))
        val d = distanceM(40.6818, -73.9999, 40.6918, -73.9999)
        assertTrue(d in 1100.0..1125.0, "$d")
    }

    @Test
    fun matchingIsOneToOneAndRespectsDistance() {
        val left = listOf(
            MatchEntity("a", listOf("DONUT DEPOT"), 40.6860, -73.9770, null),
            MatchEntity("b", listOf("DONUT DEPOT"), 40.6866, -73.9770, null),
        )
        val right = listOf(
            MatchEntity("x", listOf("Donut Depot"), 40.6861, -73.9770, null), // 11 m from a, 56 m from b
            MatchEntity("y", listOf("Donut Depot"), 40.7000, -73.9770, null), // far from both
        )
        val m = matchEntities(left, right)
        assertEquals(listOf("a" to "x"), m.map { it.left to it.right })
    }

    @Test
    fun buildsTheDatabaseFromTheFixture() {
        val dir = Files.createTempDirectory("brooklyn-ingest")
        val cfg = IngestConfig(dataDir = dir, raw = fixture, log = {})
        val report = buildDatabase(cfg)
        assertEquals(6, report.counts["osm_places"])
        assertEquals(5, report.counts["establishments"])
        assertEquals(6, report.counts["inspections"])
        assertEquals(3, report.counts["complaints"])
        assertEquals(2, report.counts["g_places"])
        assertEquals(3, report.counts["reviews"])
        assertEquals(7, report.counts["restaurants"]) // 5 CAMIS + Donut Depot and Church Ave Deli from OSM
        DriverManager.getConnection("jdbc:sqlite:${cfg.db}").use { c ->
            fun q(sql: String): List<List<String?>> = c.createStatement().use { st ->
                st.executeQuery(sql).use { rs ->
                    val n = rs.metaData.columnCount
                    buildList { while (rs.next()) add((1..n).map { rs.getString(it) }) }
                }
            }
            val pairs = q("SELECT left_id, right_id, rule FROM matches WHERE kind='dohmh-osm' ORDER BY left_id")
            assertEquals(
                listOf(listOf("1001", "node/1", "near"), listOf("1002", "way/2", "near"), listOf("1005", "node/5", "address")),
                pairs,
            )
            val joes = q("SELECT name, osm_id, gmap_id, grade, grade_date, latest_score, website FROM restaurants WHERE id='1001'").single()
            assertEquals(listOf("Tomasso's Slice", "node/1", "g1", "A", "2025-03-01", "12", "https://example.org/joes"), joes)
            assertEquals(listOf("g4"), q("SELECT gmap_id FROM restaurants WHERE id='1002'").single())
            assertNull(q("SELECT osm_id FROM restaurants WHERE id='1003'").single()[0])
            assertNotNull(q("SELECT id FROM restaurants WHERE id='osm:node/3'").singleOrNull())
            assertEquals(
                listOf(listOf("k1", "1001", "address"), listOf("k2", null, null), listOf("k3", "1002", "address")),
                q("SELECT unique_key, camis, link_rule FROM complaints ORDER BY unique_key"),
            )
            assertEquals(listOf("1002"), q("SELECT id FROM restaurants_fts WHERE restaurants_fts MATCH 'vela'").map { it[0] })
            assertEquals(listOf("1001"), q("SELECT camis FROM inspections WHERE rowid IN (SELECT rowid FROM violations_fts WHERE violations_fts MATCH 'mice')").map { it[0] })
            assertEquals(listOf("g4"), q("SELECT gmap_id FROM reviews WHERE rowid IN (SELECT rowid FROM reviews_fts WHERE reviews_fts MATCH 'calzone')").map { it[0] })
            assertEquals("2026-11-13", q("SELECT value FROM meta WHERE key='snapshot_date'").single()[0])
            assertEquals("0", q("SELECT count(*) FROM menus").single()[0])
        }
        assertEquals(2, report.samples["dohmh-osm"]!!.size.coerceAtMost(2))
    }

    @Test
    fun missingGoogleFilesLeaveReviewsEmpty() {
        val dir = Files.createTempDirectory("brooklyn-ingest-nog")
        val report = buildDatabase(IngestConfig(dataDir = dir, raw = fixture, skipGoogle = true, log = {}))
        assertEquals(0, report.counts["reviews"])
        assertEquals(0, report.counts["g_places"])
        assertEquals(7, report.counts["restaurants"])
    }
}
