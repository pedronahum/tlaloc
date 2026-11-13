package brooklyn

import com.sun.net.httpserver.HttpServer
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import java.net.InetSocketAddress
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.DayOfWeek
import java.time.LocalDateTime
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ToolsTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent

    private fun fixtureDb(): Path {
        val dir = Files.createTempDirectory("brooklyn-tools")
        val cfg = IngestConfig(dataDir = dir, raw = fixture, log = {})
        buildDatabase(cfg)
        return cfg.db
    }

    /** A factory that fails the test if snapshot or replay mode ever asks for a network client. */
    private val noNetwork: () -> Net = { throw AssertionError("network client constructed") }

    private fun obj(s: String) = parseJson(s) as JsonObject
    private fun JsonObject.list(k: String) = (this[k] as JsonArray).elements.map { it as JsonObject }
    private fun JsonObject.string(k: String) = (this[k] as? JsonString)?.value

    // ------------------------------------------------------------------ schemas

    @Test
    fun schemasAreWellFormedAndValidateArguments() {
        assertEquals(
            listOf("search_restaurants", "get_restaurant", "get_inspections", "get_complaints", "get_reviews", "get_menu", "geocode", "web_search"),
            TOOL_DEFS.map { it.name },
        )
        for (d in TOOL_DEFS) assertEquals(emptyList(), checkSchema(d.schema), d.name)
        val defs = parseJson(toolDefinitionsJson()) as JsonArray
        assertEquals(8, defs.size)
        val f = (defs[0] as JsonObject).obj("function")
        assertEquals("search_restaurants", f.str("name"))
        assertEquals("object", f.obj("parameters").str("type"))

        fun errs(tool: String, args: String) = validate(TOOL_BY_NAME[tool]!!.schema, parseJson(args))
        assertEquals(emptyList(), errs("search_restaurants", """{"query":"pizza","near":{"address":"Atlantic Ave"},"min_grade":"A","open_at":"2026-11-13T19:30","limit":5}"""))
        assertEquals(emptyList(), errs("search_restaurants", """{"near":{"lat":40.68,"lon":-73.97},"radius_m":500}"""))
        assertTrue(errs("search_restaurants", """{"min_grade":"D"}""").single().contains("not one of"))
        assertTrue(errs("search_restaurants", """{"cusine":"thai"}""").single().contains("unexpected property 'cusine'"))
        assertTrue(errs("search_restaurants", """{"near":{"lat":40.68}}""").any { it.contains("anyOf") })
        assertTrue(errs("search_restaurants", """{"radius_m":50000}""").single().contains("maximum"))
        assertTrue(errs("search_restaurants", """{"limit":2.5}""").single().contains("expected integer"))
        assertTrue(errs("search_restaurants", """{"open_at":"tonight"}""").single().contains("does not match"))
        assertTrue(errs("get_restaurant", "{}").single().contains("missing required 'id'"))
        assertTrue(errs("get_inspections", """{"id":"1","since":"last year"}""").single().contains("does not match"))
        assertTrue(errs("get_complaints", """{"since":"2025-01-01"}""").single().contains("anyOf"))
        assertEquals(emptyList(), errs("get_complaints", """{"near":{"address":"Henry St"}}"""))
        assertTrue(errs("geocode", """{"address":5}""").single().contains("expected string"))
    }

    @Test
    fun invalidCallsReturnErrorsNotExceptions() {
        ToolServer(fixtureDb(), ToolMode.SNAPSHOT, noNetwork).use { s ->
            assertTrue(obj(s.call("nope", "{}")).string("error")!!.contains("unknown tool"))
            assertTrue(obj(s.call("get_restaurant", "{not json")).string("error")!!.contains("not valid JSON"))
            assertEquals("invalid arguments", obj(s.call("get_restaurant", """{"id":"1001","x":1}""")).string("error"))
            assertTrue(obj(s.call("get_restaurant", """{"id":"9999"}""")).string("error")!!.contains("unknown restaurant id"))
            assertTrue(obj(s.call("search_restaurants", """{"neighborhood":"Hoboken"}""")).string("error")!!.contains("unknown Brooklyn neighborhood"))
            assertTrue(obj(s.call("search_restaurants", """{"open_at":"now"}""")).string("error")!!.contains("live mode"))
        }
    }

    // ------------------------------------------------------------------ snapshot mode, every tool

    @Test
    fun searchRestaurantsInSnapshotMode() {
        ToolServer(fixtureDb(), ToolMode.SNAPSHOT, noNetwork).use { s ->
            val byName = obj(s.call("search_restaurants", """{"query":"vela forno"}"""))
            assertEquals("1002", byName.list("results").first().string("id"))
            assertTrue((byName["sources"] as JsonArray).size > 0)

            val pizzaW = obj(s.call("search_restaurants", """{"cuisine":"pizza","neighborhood":"Williamsburg"}"""))
            assertEquals(listOf("1001"), pizzaW.list("results").map { it.string("id") })
            assertEquals("Williamsburg", pizzaW.list("results")[0].string("neighborhood"))

            val gowanus = obj(s.call("search_restaurants", """{"neighborhood":"Gowanus"}""")).list("results").map { it.string("id") }
            assertEquals(listOf("1005", "1002"), gowanus) // Blue Stove, Vela Forno (by name)

            val gradeA = obj(s.call("search_restaurants", """{"min_grade":"A","limit":50}""")).list("results").map { it.string("id") }.toSet()
            assertEquals(setOf("1001", "1002", "1003"), gradeA) // 1004 has no grade, 1005 not inspected

            val near = obj(s.call("search_restaurants", """{"near":{"lat":40.6818,"lon":-73.9999},"radius_m":2000}"""))
            val ids = near.list("results").map { it.string("id") }
            assertEquals("1002", ids.first())
            assertTrue("1001" !in ids)
            val dists = near.list("results").map { (it["distance_m"] as io.tlaloc.core.io.JsonNumber).value }
            assertEquals(dists.sorted(), dists)

            val byAddress = obj(s.call("search_restaurants", """{"near":{"address":"216 Bedford Ave"},"radius_m":100}"""))
            assertEquals("1001", byAddress.list("results").first().string("id"))
            assertTrue(byAddress.obj("near").string("from")!!.contains("address"))

            // Tomasso's: OSM Mo-Su 10:00-23:00; Vela Forno: no hours anywhere.
            val open = obj(s.call("search_restaurants", """{"open_at":"2026-11-14T12:00","limit":50}"""))
            assertEquals(listOf("1001"), open.list("results").map { it.string("id") })
            assertEquals("osm", open.list("results")[0].string("hours"))
            val late = obj(s.call("search_restaurants", """{"open_at":"2026-11-14T23:30"}"""))
            assertEquals(0, late.list("results").size)
        }
    }

    @Test
    fun restaurantInspectionsComplaintsReviewsInSnapshotMode() {
        ToolServer(fixtureDb(), ToolMode.SNAPSHOT, noNetwork).use { s ->
            val r = obj(s.call("get_restaurant", """{"id":"1001"}"""))
            assertEquals("Tomasso's Slice", r.string("name"))
            assertEquals("A", r.string("grade"))
            assertEquals("https://example.org/joes", r.string("website"))
            assertEquals("Williamsburg", r.string("neighborhood"))
            assertEquals("1", (r.obj("counts")["complaints_311"] as io.tlaloc.core.io.JsonNumber).raw)
            assertEquals("1001", obj(s.call("get_restaurant", """{"id":"node/1"}""")).string("id"))
            assertEquals("1001", obj(s.call("get_restaurant", """{"id":"osm:node/1"}""")).string("id"))

            val insp = obj(s.call("get_inspections", """{"id":"1001"}"""))
            val visits = insp.list("inspections")
            assertEquals(listOf("2025-03-01", "2024-01-10"), visits.map { it.string("date") })
            val v = visits[0].list("violations").single()
            assertEquals("04L", v.string("code"))
            assertEquals(true, (v["critical"] as io.tlaloc.core.io.JsonBool).value)
            assertEquals(1, obj(s.call("get_inspections", """{"id":"1001","since":"2025-01-01"}""")).list("inspections").size)
            assertNotNull(obj(s.call("get_inspections", """{"id":"1005"}"""))["not_yet_inspected"])
            assertNotNull(obj(s.call("get_inspections", """{"id":"osm:node/3"}""")).string("note"))

            assertEquals(listOf("k1"), obj(s.call("get_complaints", """{"id":"1001"}""")).list("complaints").map { it.string("key") })
            assertEquals(listOf("k3"), obj(s.call("get_complaints", """{"id":"1002"}""")).list("complaints").map { it.string("key") })
            val near = obj(s.call("get_complaints", """{"near":{"lat":40.717,"lon":-73.959},"radius_m":50}"""))
            assertEquals(listOf("k1"), near.list("complaints").map { it.string("key") })
            assertEquals(0, obj(s.call("get_complaints", """{"id":"1001","since":"2025-05-01"}""")).list("complaints").size)

            val rv = obj(s.call("get_reviews", """{"id":"1001"}"""))
            assertEquals("2", (rv["count"] as io.tlaloc.core.io.JsonNumber).raw)
            assertEquals(listOf("Classic thin slice, fresh mozzarella."), rv.list("reviews").map { it.string("text") })
            val q = obj(s.call("get_reviews", """{"id":"1002","query":"calzone"}"""))
            assertEquals("Thank you!", q.list("reviews").single().string("owner_response"))
            assertEquals(0, obj(s.call("get_reviews", """{"id":"1002","query":"sushi"}""")).list("reviews").size)
            assertNotNull(obj(s.call("get_reviews", """{"id":"1004"}""")).string("note"))
        }
    }

    @Test
    fun menuGeocodeAndWebSearchInSnapshotMode() {
        ToolServer(fixtureDb(), ToolMode.SNAPSHOT, noNetwork).use { s ->
            // The start page has no prices; its "Our Menu" link's capture does.
            val m = obj(s.call("get_menu", """{"id":"1001"}"""))
            assertEquals("ok", m.string("status"))
            assertEquals("https://example.org/menu", m.string("page"))
            assertEquals("2025-01-01T12:05:00Z", m.string("archived_at"))
            assertEquals("https://web.archive.org/web/20250101120500id_/https://example.org/menu", m.string("archive_url"))
            assertTrue(m.string("menu")!!.contains("Baked Ziti\n$13.00"), m.string("menu"))
            assertTrue(m.string("source")!!.contains("Internet Archive"))
            val donut = obj(s.call("get_menu", """{"id":"osm:node/3"}"""))
            assertEquals("no_snapshot", donut.string("status"))
            assertTrue(donut.string("reason")!!.contains("no capture"))
            assertEquals("pdf", obj(s.call("get_menu", """{"id":"osm:node/4"}""")).string("status"))
            assertTrue(obj(s.call("get_menu", """{"id":"1004"}""")).string("reason")!!.contains("no website"))
            assertEquals("ok", obj(s.call("get_restaurant", """{"id":"1001"}""")).obj("archived_menu").string("status"))

            val a = obj(s.call("geocode", """{"address":"216 Bedford Avenue"}""")).list("results").single()
            assertEquals("address", a.string("precision"))
            assertEquals("Williamsburg", a.string("neighborhood"))
            val street = obj(s.call("geocode", """{"address":"220 Bedford Ave"}""")).list("results").single()
            assertEquals("street (nearest known number)", street.string("precision"))
            assertEquals("neighborhood", obj(s.call("geocode", """{"address":"Park Slope"}""")).list("results").single().string("precision"))
            assertEquals("zip", obj(s.call("geocode", """{"address":"somewhere 11231"}""")).list("results").single().string("precision"))
            assertEquals("place", obj(s.call("geocode", """{"address":"Golden Dragon"}""")).list("results").first().string("precision"))
            assertEquals(0, obj(s.call("geocode", """{"address":"zzqx"}""")).list("results").size)

            val w = obj(s.call("web_search", """{"query":"tomasso slice"}"""))
            assertEquals("https://example.org/joes", w.list("results").single().string("url"))
            assertTrue(w.string("source")!!.startsWith("offline"))
            assertFalse(s.netConstructed)
        }
    }

    @Test
    fun snapshotResultsAreDeterministic() {
        val db = fixtureDb()
        val calls = listOf(
            "search_restaurants" to """{"query":"pizza","limit":5}""", "get_restaurant" to """{"id":"1002"}""",
            "get_inspections" to """{"id":"1001"}""", "get_complaints" to """{"id":"1001"}""",
            "get_reviews" to """{"id":"1001"}""", "get_menu" to """{"id":"1001"}""",
            "geocode" to """{"address":"575 Henry St"}""", "web_search" to """{"query":"vela"}""",
        )
        val a = ToolServer(db, ToolMode.SNAPSHOT, noNetwork).use { s -> calls.map { s.call(it.first, it.second) } }
        val b = ToolServer(db, ToolMode.SNAPSHOT, noNetwork).use { s -> calls.map { s.call(it.first, it.second) } }
        assertEquals(a, b)
        assertTrue(a.none { it.contains("\"error\"") }, a.joinToString("\n"))
        assertTrue(a.all { it.contains("2026-11-13") }, "every snapshot result names the snapshot date")
    }

    // ------------------------------------------------------------------ live, record, replay with a fake network

    private class FakeNet : Net {
        val urls = ArrayList<String>()
        override fun get(url: String, accept: String?): NetResponse {
            urls += url
            val body: String
            var ct = "application/json"
            when {
                url.contains("43nn-pn8j") && url.contains("select") -> body =
                    """[{"camis":"1001","grade":"B","grade_date":"2026-10-01T00:00:00.000"},{"camis":"1001","grade":"A","grade_date":"2025-03-01T00:00:00.000"}]"""
                url.contains("43nn-pn8j") -> body =
                    """[{"camis":"1001","inspection_date":"2026-10-01T00:00:00.000","inspection_type":"Cycle Inspection / Re-inspection","violation_code":"06C","violation_description":"Food not protected.","critical_flag":"Critical","score":"18","grade":"B","grade_date":"2026-10-01T00:00:00.000"}]"""
                url.contains("erm2-nwe9") -> body =
                    """[{"unique_key":"n1","created_date":"2026-09-01T10:00:00.000","complaint_type":"Food Poisoning","descriptor":"1 or 2","incident_address":"216 BEDFORD AVENUE","incident_zip":"11249","status":"Open"}]"""
                url.contains("nominatim") -> body =
                    """[{"lat":"40.71705","lon":"-73.95905","display_name":"216, Bedford Avenue, Williamsburg, Brooklyn","addresstype":"building","osm_type":"node","osm_id":"1"}]"""
                url.endsWith("/robots.txt") -> { ct = "text/plain"; body = "User-agent: *\nDisallow: /private\n" }
                url == "https://example.org/joes" -> { ct = "text/html"; body = "<html><body><h1>Tomasso's</h1><a href=\"/menu\">Our Menu</a><p>Call us</p></body></html>" }
                url == "https://example.org/menu" -> {
                    ct = "text/html; charset=utf-8"
                    body = "<html><head><title>x</title><script>var p = 9.99;</script></head><body><ul>" +
                        listOf("Cheese Slice" to "3.50", "Pepperoni Slice" to "4.25", "Grandma Pie" to "28.00", "Calzone" to "12.00", "Garlic Knots" to "6.00")
                            .joinToString("") { "<li>${it.first} &amp; more</li><li>$${it.second}</li>" } + "</ul></body></html>"
                }
                url.contains("/search?q=") -> body =
                    """{"results":[{"title":"Tomasso's Slice - Brooklyn","url":"https://example.org/joes","content":"Pizza by the slice.","engines":["fake"]}]}"""
                else -> return NetResponse(404, "text/plain", ByteArray(0), url)
            }
            return NetResponse(200, ct, body.toByteArray(), url)
        }
    }

    @Test
    fun liveModeUsesTheApisAndNeverWritesTheDatabase() {
        val db = fixtureDb()
        val dbSha = fileSha256(db)
        val net = FakeNet()
        ToolServer(db, ToolMode.LIVE, { net }, searxngUrl = "http://127.0.0.1:1", liveClock = { LocalDateTime.of(2026, 11, 14, 12, 0) }).use { s ->
            val r = obj(s.call("get_restaurant", """{"id":"1001"}"""))
            assertEquals("B", r.string("grade"))
            assertEquals("2026-10-01", r.string("latest_inspection"))
            assertTrue(net.urls.last().contains(Http.enc("camis='1001'")))

            val insp = obj(s.call("get_inspections", """{"id":"1001","since":"2026-01-01"}"""))
            assertEquals("06C", insp.list("inspections").single().list("violations").single().string("code"))
            assertTrue(net.urls.last().contains(Http.enc("inspection_date >= '2026-01-01T00:00:00'")))

            val c = obj(s.call("get_complaints", """{"id":"1001"}"""))
            assertEquals("address", c.list("complaints").single().string("link"))
            assertTrue(net.urls.last().contains(Http.enc("bbl='3023690001'")))
            obj(s.call("get_complaints", """{"near":{"lat":40.717,"lon":-73.959},"radius_m":100}"""))
            assertTrue(net.urls.last().contains(Http.enc("within_circle(location, 40.717, -73.959, 100)")))

            val search = obj(s.call("search_restaurants", """{"query":"tomasso","open_at":"now"}"""))
            assertEquals("B", search.list("results").single().string("grade")) // refreshed live
            assertEquals("2026-11-14T12:00", search.obj("filters").string("open_at"))

            val g = obj(s.call("geocode", """{"address":"216 Bedford Ave"}"""))
            assertEquals("Williamsburg", g.list("results").single().string("neighborhood"))
            assertTrue(net.urls.last().contains("q=216+Bedford+Ave%2C+Brooklyn%2C+NY"))

            val w = obj(s.call("web_search", """{"query":"tomasso's slice"}"""))
            assertEquals("Tomasso's Slice - Brooklyn", w.list("results").single().string("title"))
            assertTrue(net.urls.last().startsWith("http://127.0.0.1:1/search?q="))

            val m = obj(s.call("get_menu", """{"id":"1001"}"""))
            assertEquals("ok", m.string("status"))
            assertEquals("https://example.org/menu", m.string("page"))
            assertTrue(m.string("menu")!!.contains("Grandma Pie & more\n$28.00"))
            assertFalse(m.string("menu")!!.contains("9.99")) // script dropped
            assertEquals("restaurant web site, live", m.string("source"))
            assertTrue(net.urls.contains("https://example.org/robots.txt"))
        }
        // Live calls left the snapshot as it was: the archived menu, indexed for full-text search.
        assertEquals(dbSha, fileSha256(db))
        ToolServer(db, ToolMode.SNAPSHOT, noNetwork).use { s ->
            val m = obj(s.call("get_menu", """{"id":"1001"}"""))
            assertTrue(m.string("menu")!!.contains("Baked Ziti"))
            assertEquals(listOf("1001"), s.query("SELECT restaurant_id FROM menus WHERE rowid IN (SELECT rowid FROM menus_fts WHERE menus_fts MATCH 'ziti')").map { it[0] })
        }

    }

    @Test
    fun recordThenReplayGivesTheSameResultsWithoutNetwork() {
        val db = fixtureDb()
        val calls = listOf(
            "get_restaurant" to """{"id":"1001"}""", "get_inspections" to """{"id":"1001"}""",
            "geocode" to """{"address":"216 Bedford Ave"}""", "web_search" to """{"query":"tomasso"}""",
        )
        val recorded = ToolServer(db, ToolMode.RECORD, { FakeNet() }, searxngUrl = "http://127.0.0.1:1").use { s ->
            calls.map { s.call(it.first, it.second) }
        }
        val replayed = ToolServer(db, ToolMode.REPLAY, noNetwork).use { s ->
            // Argument order does not matter: recordings are keyed by canonical JSON.
            assertEquals(recorded[1], s.call("get_inspections", """{ "id" : "1001" }"""))
            assertTrue(obj(s.call("get_restaurant", """{"id":"1002"}""")).string("error")!!.contains("no recording"))
            assertFalse(s.netConstructed)
            calls.map { s.call(it.first, it.second) }
        }
        assertEquals(recorded, replayed)
        assertTrue(recorded.none { it.contains("\"error\"") }, recorded.joinToString("\n"))
    }

    // ------------------------------------------------------------------ pieces

    @Test
    fun httpNetSpacesRequestsAndRetries429() {
        val hits = AtomicInteger()
        val times = java.util.Collections.synchronizedList(ArrayList<Long>())
        val agents = java.util.Collections.synchronizedList(ArrayList<String>())
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 4)
        server.createContext("/") { ex ->
            times += System.currentTimeMillis()
            agents += ex.requestHeaders.getFirst("User-Agent")
            val n = hits.incrementAndGet()
            if (n == 1) ex.responseHeaders.add("Retry-After", "1")
            val body = "ok".toByteArray()
            ex.sendResponseHeaders(if (n == 1) 429 else 200, body.size.toLong())
            ex.responseBody.use { it.write(body) }
        }
        server.start()
        try {
            val net = HttpNet(minSpacingMs = 300, log = {})
            val r = net.get("http://127.0.0.1:${server.address.port}/x", null)
            assertEquals(200, r.status)
            assertEquals(2, hits.get())
            assertTrue(times[1] - times[0] >= 1_000, "backoff ${times[1] - times[0]} ms")
            net.get("http://127.0.0.1:${server.address.port}/y", null)
            // Times are taken at the server, so they carry a few ms of delivery jitter.
            assertTrue(times[2] - times[1] >= 290, "spacing ${times[2] - times[1]} ms")
            assertTrue(agents.all { it == USER_AGENT })
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun numberedStreetsBecomeOrdinals() {
        assertEquals("8002 5th Avenue, Brooklyn, NY 11209", ordinalStreets("8002 5 Avenue, Brooklyn, NY 11209"))
        assertEquals("100 East 21st Street", ordinalStreets("100 East 21 Street"))
        assertEquals("12 West 112th St", ordinalStreets("12 West 112 St"))
        assertEquals("5 Avenue U", ordinalStreets("5 Avenue U")) // a house number stays
        assertEquals("216 Bedford Ave", ordinalStreets("216 Bedford Ave"))
    }

    @Test
    fun openingHoursParse() {
        val osm = parseOsmHours("Mo-Th 17:30-22:00; Fr 17:30-23:00; Sa 11:30-15:30,17:30-23:00; Su off")!!
        assertTrue(osm.isOpen(LocalDateTime.of(2026, 11, 14, 12, 0))) // Saturday
        assertFalse(osm.isOpen(LocalDateTime.of(2026, 11, 14, 16, 0)))
        assertFalse(osm.isOpen(LocalDateTime.of(2026, 11, 15, 19, 0))) // Sunday off
        val night = parseOsmHours("Mo-Th 14:00-02:00, Fr 14:00-03:00")!!
        assertTrue(night.isOpen(LocalDateTime.of(2026, 11, 14, 2, 30))) // Saturday 02:30, from Friday
        assertFalse(night.isOpen(LocalDateTime.of(2026, 11, 13, 2, 30))) // Friday 02:30 after Thursday's 02:00
        assertEquals(DayOfWeek.entries.toSet(), parseOsmHours("24/7")!!.days.keys)
        assertNotNull(parseOsmHours("Mo, We-Su 11:00-21:00")!!.days[DayOfWeek.MONDAY])
        assertTrue(parseOsmHours("Mo, We-Su 11:00-21:00")!!.days[DayOfWeek.TUESDAY] == null)
        assertNull(parseOsmHours("Mo-Fr 08:00-12:00 \"by appointment\""))
        assertNull(parseOsmHours("sunrise-sunset"))

        val g = parseGoogleHours("Thursday 11AM–9:30PM; Friday 11–3PM, 5–11PM; Saturday Open 24 hours; Monday Closed")!!
        assertEquals(listOf(660..1290), g.days[DayOfWeek.THURSDAY])
        assertEquals(listOf(660..900, 1020..1380), g.days[DayOfWeek.FRIDAY])
        assertTrue(g.isOpen(LocalDateTime.of(2026, 11, 14, 4, 0)))
        assertEquals(emptyList(), g.days[DayOfWeek.MONDAY])
        assertEquals(listOf(1080..1560), parseGoogleHours("Sunday 6PM–2AM")!!.days[DayOfWeek.SUNDAY])
    }

    @Test
    fun neighborhoodsResolveByNameAndPolygon() {
        fun box(code: String, name: String, lo0: Double, lo1: Double, la0: Double, la1: Double, type: String = "0") = Neighborhood(
            code, name, null, type,
            listOf(listOf(listOf(doubleArrayOf(lo0, la0), doubleArrayOf(lo1, la0), doubleArrayOf(lo1, la1), doubleArrayOf(lo0, la1), doubleArrayOf(lo0, la0)))),
        )
        val ns = listOf(
            box("BK0301", "Bedford-Stuyvesant (West)", 0.0, 1.0, 0.0, 1.0), box("BK0302", "Bedford-Stuyvesant (East)", 1.0, 2.0, 0.0, 1.0),
            box("BK0601", "Carroll Gardens-Cobble Hill-Gowanus-Red Hook", 2.0, 3.0, 0.0, 1.0), box("BK0602", "Park Slope", 3.0, 4.0, 0.0, 1.0),
            box("BK0701", "Windsor Terrace-South Slope", 4.0, 5.0, 0.0, 1.0), box("BK5591", "Prospect Park", 3.5, 4.5, 0.0, 1.0, "9"),
        )
        assertEquals(listOf("BK0301", "BK0302"), resolveNeighborhood(ns, "Bed-Stuy").map { it.code })
        assertEquals(listOf("BK0301", "BK0302"), resolveNeighborhood(ns, "bedford stuyvesant").map { it.code })
        assertEquals(listOf("BK0601"), resolveNeighborhood(ns, "Red Hook").map { it.code })
        assertEquals(listOf("BK0602"), resolveNeighborhood(ns, "park slope, Brooklyn").map { it.code })
        assertEquals(listOf("BK0701"), resolveNeighborhood(ns, "South Slope").map { it.code })
        assertEquals(listOf("BK0602"), resolveNeighborhood(ns, "bk0602").map { it.code })
        assertEquals(emptyList(), resolveNeighborhood(ns, "Astoria"))
        assertEquals("BK0602", neighborhoodOf(ns, 0.5, 3.7)?.code) // residential wins over the park
        assertEquals("BK0701", neighborhoodOf(ns, 0.5, 4.2)?.code) // residential South Slope over the park
        assertEquals("BK5591", neighborhoodOf(ns.filter { it.code != "BK0701" }, 0.5, 4.2)?.code) // the park alone
        assertNull(neighborhoodOf(ns, 5.0, 5.0))
    }

    @Test
    fun menuTextAndRobots() {
        val html = "<html><head><style>.a{}</style></head><body><nav>Home</nav><!-- c --><h2>Pizza</h2>" +
            "<div>Margherita</div><div>\$16</div><p>Marinara 14.50</p><p>Phone 718.555.1234</p></body></html>"
        val text = htmlToText(html)
        assertEquals(listOf("Home", "Pizza", "Margherita", "\$16", "Marinara 14.50", "Phone 718.555.1234"), text.lines())
        assertEquals(listOf("Margherita", "\$16", "Marinara 14.50"), menuLines(text))
        assertEquals(emptyList(), menuLines("Regular price\n$12.53\nSale price $9.99 USD"))
        assertEquals(emptyList(), menuLines("Catering deliveries at this restaurant require a $350.00 subtotal minimum order size. Delivery fees and terms apply."))
        assertEquals("Café & Bar 'x'", decodeEntities("Caf&eacute; &amp; Bar &#39;x&#x27;"))
        assertEquals(
            listOf("https://a.example/menu", "https://b.example/dinner-menu"),
            menuLinks("<a href=\"https://b.example/dinner-menu\">Dinner</a><a href='/menu'>x</a><a href=\"/about\">About</a>", "https://a.example/home"),
        )
        val rules = parseRobots("User-agent: Googlebot\nDisallow: /\n\nUser-agent: *\nDisallow: /private\nAllow: /private/menu\nDisallow: /*.pdf$\n")
        assertTrue(robotsAllows(rules, "/menu"))
        assertFalse(robotsAllows(rules, "/private/x"))
        assertTrue(robotsAllows(rules, "/private/menu"))
        assertFalse(robotsAllows(rules, "/files/menu.pdf"))
        assertTrue(robotsAllows(rules, "/files/menu.pdf?x=1"))
    }
}
