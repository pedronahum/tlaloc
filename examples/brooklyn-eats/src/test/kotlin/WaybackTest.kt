package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class WaybackTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent
    private val cutoff = cutoffTimestamp(LocalDate.parse("2026-11-13"))

    private val startHtml = "<html><body><p>Welcome</p><a href=\"/food-menu\">Menu</a></body></html>"
    private val menuHtml = "<html><body>" + (1..6).joinToString("") { "<p>Dish $it</p><p>$1$it.00</p>" } + "</body></html>"

    /** A fake Internet Archive: availability, CDX and id_ captures; records every URL asked for. */
    private class FakeArchive(
        val availability: Map<String, String?>, // page URL -> closest timestamp (null: none)
        val cdx: Map<String, List<String>> = emptyMap(),
        val captures: Map<String, String>, // archive URL -> body
    ) : Net {
        val urls = ArrayList<String>()
        override fun get(url: String, accept: String?): NetResponse {
            synchronized(urls) { urls += url }
            if (url.startsWith(WAYBACK_AVAILABLE)) {
                val page = java.net.URLDecoder.decode(url.substringAfter("url=").substringBefore("&timestamp"), Charsets.UTF_8)
                val ts = availability[page]
                val closest = if (ts == null) "{}" else """{"closest":{"status":"200","available":true,"url":"x","timestamp":"$ts"}}"""
                return NetResponse(200, "application/json", """{"url":"$page","archived_snapshots":$closest}""".toByteArray(), url)
            }
            if (url.startsWith(WAYBACK_CDX)) {
                val page = java.net.URLDecoder.decode(url.substringAfter("url=").substringBefore("&to="), Charsets.UTF_8)
                return NetResponse(200, "text/plain", cdx[page].orEmpty().joinToString("\n").toByteArray(), url)
            }
            val body = captures[url] ?: return NetResponse(404, "text/plain", ByteArray(0), url)
            return NetResponse(200, "text/html", body.toByteArray(), url)
        }
    }

    @Test
    fun snapshotIsTheLatestOnOrBeforeTheCutoff() {
        val net = FakeArchive(
            availability = mapOf("http://a.example/" to "20240102030405", "http://b.example/" to "20270101000000", "http://c.example/" to null),
            cdx = mapOf("http://b.example/" to listOf("20190101000000", "20250505050505")),
            captures = emptyMap(),
        )
        assertEquals("20240102030405", findSnapshot(net, "http://a.example/", cutoff))
        assertEquals(1, net.urls.size) // no CDX call when the closest capture is before the cutoff
        assertEquals("20250505050505", findSnapshot(net, "http://b.example/", cutoff)) // closest is after: CDX up to the cutoff
        assertTrue(net.urls.last().contains("&to=20261112235959"))
        assertNull(findSnapshot(net, "http://c.example/", cutoff))
        val blocked = object : Net {
            override fun get(url: String, accept: String?) = if (url.startsWith(WAYBACK_CDX)) NetResponse(403, "text/plain", ByteArray(0), url)
            else NetResponse(200, "application/json", """{"archived_snapshots":{},"error":{"code":"upstream_error"}}""".toByteArray(), url)
        }
        kotlin.test.assertFailsWith<ExcludedSite> { findSnapshot(blocked, "https://linktr.ee/x", cutoff) }
        val (pins, _) = fetchWaybackMenus(listOf(MenuTarget("node/1", "https://linktr.ee/x")), blocked, cutoff, emptyMap(), Files.createTempDirectory("wbx"), true, 1) {}
        assertEquals("excluded", pins.values.single().status)
        assertEquals("2024-01-02T03:04:05Z", waybackIso("20240102030405"))
        assertEquals("20240102030405" to "http://a.example/", parseArchiveUrl("https://web.archive.org/web/20240102030405id_/http://a.example/"))
    }

    @Test
    fun gzipCapturesAreInflatedAndBinaryIsRefused() {
        val html = "<p>Pad Thai</p><p>$14.00</p>"
        val gz = java.io.ByteArrayOutputStream().also { o -> java.util.zip.GZIPOutputStream(o).use { it.write(html.toByteArray()) } }.toByteArray()
        assertEquals(html, decodeArchived(gz))
        assertEquals(html, decodeArchived(html.toByteArray()))
        assertNull(decodeArchived(ByteArray(400) { (it * 37 % 251).toByte() }))
        assertNull(decodeArchived(ByteArray(400) { (it * 37 % 251).toByte() }, "text/html"))
        assertNotNull(decodeArchived(ByteArray(400) { (it * 37 % 251).toByte() }, "application/pdf")) // classified as pdf later
        val dir = Files.createTempDirectory("wbgz")
        val sha = sha256Hex(gz)
        Files.createDirectories(dir.resolve(sha.substring(0, 2)))
        Files.write(dir.resolve(sha.substring(0, 2)).resolve(sha), gz)
        val p = pageFromStore(WaybackPin("https://g.example/", "ok", "20250101000000", "$WAYBACK_WEB/20250101000000id_/https://g.example/", "text/html", sha), dir)
        assertEquals(html, p.html)
        assertEquals("https://g.example/", p.url)
    }

    @Test
    fun startUrlsAreNormalized() {
        assertEquals("http://donutdepot.example", normalizeStartUrl(" donutdepot.example "))
        assertEquals("https://a.example/menu", normalizeStartUrl("https://a.example/menu;https://b.example"))
        assertNull(normalizeStartUrl("mailto:x@example.org"))
        assertNull(normalizeStartUrl("ftp://a.example"))
        val places = parseOverpass(Files.readString(fixture.resolve("overpass.json")))
        assertEquals(
            listOf("node/1" to "https://example.org/joes", "node/3" to "http://donutdepot.example", "node/4" to "https://deli.example/menu.pdf"),
            menuTargets(places).map { it.osmId to it.url },
        )
    }

    @Test
    fun discoveredPinsReproduceTheSameMenuWithoutDiscovery() {
        val start = "https://r.example/"
        val menu = "https://r.example/food-menu"
        val archive = mapOf(
            "$WAYBACK_WEB/20240101000000id_/$start" to startHtml,
            "$WAYBACK_WEB/20240101000100id_/$menu" to menuHtml,
        )
        val targets = listOf(MenuTarget("node/9", start), MenuTarget("node/10", "https://gone.example/"))
        val net1 = FakeArchive(mapOf(start to "20240101000000", menu to "20240101000100", "https://gone.example/" to null), captures = archive)
        val dir1 = Files.createTempDirectory("wb1")
        val (pins, stats) = fetchWaybackMenus(targets, net1, cutoff, emptyMap(), dir1, discover = true, threads = 2, log = {})
        assertEquals(setOf(start, menu, "https://gone.example/"), pins.keys)
        assertEquals("no_snapshot", pins["https://gone.example/"]!!.status)
        assertEquals(sha256Hex(menuHtml.toByteArray()), pins[menu]!!.sha256)
        assertEquals(2, stats.downloaded.get())
        val m1 = menuFromPages(start) { pageFromStore(pins[it], dir1, it) }
        assertEquals("ok", m1.status)
        assertEquals(menu, m1.url)
        assertEquals("20240101000100", m1.snapshot)
        assertEquals(6, m1.items)

        // A second machine with the pins: only the pinned captures are fetched, nothing is looked up.
        val pinFile = Files.createTempDirectory("pins").resolve("wayback-menus.tsv")
        writePins(pinFile, pins.values)
        val net2 = FakeArchive(emptyMap(), captures = archive)
        val dir2 = Files.createTempDirectory("wb2")
        val (pins2, _) = fetchWaybackMenus(targets, net2, cutoff, readPins(pinFile), dir2, discover = true, threads = 2, log = {})
        assertTrue(net2.urls.none { it.startsWith(WAYBACK_AVAILABLE) || it.startsWith(WAYBACK_CDX) }, net2.urls.toString())
        assertEquals(pins, pins2)
        val m2 = menuFromPages(start) { pageFromStore(pins2[it], dir2, it) }
        assertEquals(m1.text, m2.text)

        // A capture whose bytes no longer hash to the pin is not used.
        val net3 = FakeArchive(emptyMap(), captures = archive + ("$WAYBACK_WEB/20240101000100id_/$menu" to menuHtml.replace("Dish 1", "Dish X")))
        val f3 = WaybackFetcher(net3, cutoff, readPins(pinFile), Files.createTempDirectory("wb3"), discover = false, log = {}, retryMs = 1)
        assertEquals("sha_mismatch", f3.page(menu).status)
        assertEquals(1, f3.stats.shaMismatch.get())
        assertEquals(3, net3.urls.count { it.contains("20240101000100id_") }) // three tries

        // The archive redirecting a pinned timestamp to another capture is reported as moved.
        val moved = object : Net {
            override fun get(url: String, accept: String?) =
                NetResponse(200, "text/html", "<p>other</p>".toByteArray(), url.replace("20240101000100", "20230101000000"))
        }
        val f4 = WaybackFetcher(moved, cutoff, readPins(pinFile), Files.createTempDirectory("wb4"), discover = false, log = {}, retryMs = 1)
        assertEquals("moved", f4.page(menu).status)
    }

    @Test
    fun menusTableIsBuiltFromTheStoreAndHashesTheSame() {
        fun build(): Path {
            val dir = Files.createTempDirectory("brooklyn-wb")
            val cfg = IngestConfig(dataDir = dir, raw = fixture, log = {})
            buildDatabase(cfg)
            return cfg.db
        }
        val a = build()
        val b = build()
        val cols = listOf("restaurant_id", "osm_id", "url", "page_url", "snapshot", "archive_url", "status", "items", "text")
        val (ha, na) = tableDigest(a, "menus", cols, "restaurant_id")
        assertEquals(3L, na) // node/1 ok, node/3 no snapshot, node/4 pdf
        assertEquals(ha, tableDigest(b, "menus", cols, "restaurant_id").first)
        assertTrue(fileSha256(a) != "" && ha.length == 64)
        val rows = java.sql.DriverManager.getConnection("jdbc:sqlite:$a").use { c ->
            c.createStatement().executeQuery("SELECT restaurant_id, status, snapshot FROM menus ORDER BY restaurant_id").use { rs ->
                buildList { while (rs.next()) add(listOf(rs.getString(1), rs.getString(2), rs.getString(3))) }
            }
        }
        assertEquals(
            listOf(listOf("1001", "ok", "2025-01-01T12:05:00Z"), listOf("osm:node/3", "no_snapshot", null), listOf("osm:node/4", "pdf", "2024-06-01T00:00:00Z")),
            rows,
        )
    }

    @Test
    fun manifestIsWrittenAndRead() {
        val f = Files.createTempDirectory("manifest").resolve("manifest.json")
        val e = Entry("tools", "derived", "x", true) { mapOf("sha256" to "ab", "rows" to 2L) }
        writeManifest(f, null, mapOf(e to e.compute()), "quick")
        val o = parseJson(Files.readString(f)) as JsonObject
        assertEquals(DEFAULT_CUTOFF, o.optString("cutoff"))
        val x = o.optObject("stages")!!.optObject("tools")!!.optObject("derived")!!.optObject("x")!!.optObject("quick")!!
        assertEquals("ab", x.optString("sha256"))
        writeManifest(f, o, mapOf(e to mapOf("sha256" to "cd")), "full")
        val o2 = parseJson(Files.readString(f)) as JsonObject
        val x2 = o2.optObject("stages")!!.optObject("tools")!!.optObject("derived")!!.optObject("x")!!
        assertEquals("ab", x2.optObject("quick")!!.optString("sha256")) // the other scope is kept
        assertEquals("cd", x2.optObject("full")!!.optString("sha256"))
    }
}
