package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.sql.DriverManager
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Pinned queries, pins that a normal run never rewrites, and verify against the fixture. */
class ReproTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent
    private val example: Path = Paths.get("").toAbsolutePath() // Gradle runs tests from the project directory

    @Test
    fun everySourceQueryIsPinnedToTheCutoff() {
        val c = LocalDate.parse("2026-11-13")
        assertTrue(overpassQuery(c).contains("[date:\"2026-11-13T00:00:00Z\"]"), overpassQuery(c))
        assertEquals("boro='Brooklyn' AND inspection_date < '2026-11-13T00:00:00'", dohmhWhere(c))
        assertTrue(complaintsWhere(c).endsWith("AND created_date >= '2023-11-13T00:00:00' AND created_date < '2026-11-13T00:00:00'"))
        // The queries depend on the cutoff only, never on the day they run.
        assertEquals(complaintsWhere(c), complaintsWhere(LocalDate.parse("2026-11-13")))
        assertEquals("20261112235959", cutoffTimestamp(c))
        assertEquals("2026-11-13T23:01:06Z", overpassBase("""{"osm3s":{"timestamp_osm_base":"2026-11-13T23:01:06Z"},"elements":[]}"""))
    }

    @Test
    fun snapshotDateIsTheCutoffNotTheFetchDay() {
        val dir = Files.createTempDirectory("brooklyn-cutoff")
        val cfg = IngestConfig(dataDir = dir, raw = fixture, cutoff = LocalDate.parse("2026-10-01"), log = {})
        buildDatabase(cfg)
        DriverManager.getConnection("jdbc:sqlite:${cfg.db}").use { c ->
            c.createStatement().executeQuery("SELECT value FROM meta WHERE key='snapshot_date'").use { rs ->
                rs.next(); assertEquals("2026-10-01", rs.getString(1))
            }
        }
        ToolServer(cfg.db, ToolMode.SNAPSHOT).use { s ->
            val o = parseJson(s.call("search_restaurants", """{"query":"slice"}""")) as JsonObject
            assertEquals("2026-10-01", o.optString("as_of"))
        }
    }

    @Test
    fun aNormalRunNeverLooksUpOrRewritesPins() {
        val pinned = "https://pinned.example/"
        val body = "<html><body>" + (1..6).joinToString("") { "<p>Dish $it</p><p>$$it.00</p>" } + "</body></html>"
        val pins = mapOf(pinned to WaybackPin(pinned, "ok", "20240101000000", "$WAYBACK_WEB/20240101000000id_/$pinned", "text/html", sha256Hex(body.toByteArray())))
        val asked = ArrayList<String>()
        val net = object : Net {
            override fun get(url: String, accept: String?): NetResponse {
                synchronized(asked) { asked += url }
                return if (url.contains("20240101000000id_")) NetResponse(200, "text/html", body.toByteArray(), url) else NetResponse(404, "text/plain", ByteArray(0), url)
            }
        }
        val targets = listOf(MenuTarget("node/1", pinned), MenuTarget("node/2", "https://new.example/"))
        val (used, stats) = fetchWaybackMenus(targets, net, "20261112235959", pins, Files.createTempDirectory("wbn"), discover = false, threads = 1, log = {})
        assertEquals("not_pinned", used["https://new.example/"]!!.status)
        assertEquals(1, stats.notPinned.get())
        assertTrue(asked.none { it.startsWith(WAYBACK_AVAILABLE) || it.startsWith(WAYBACK_CDX) }, asked.toString())

        // fetchMenus with the committed pins file: written only with --pin-menus.
        val raw = Files.createTempDirectory("raw")
        Files.copy(fixture.resolve("overpass.json"), raw.resolve("overpass.json"))
        fixture.resolve("wayback").toFile().copyRecursively(raw.resolve("wayback").toFile())
        val pinFile = Files.createTempDirectory("pins").resolve("wayback-menus.tsv")
        Files.copy(fixture.resolve("wayback/pins.tsv"), pinFile)
        val before = Files.readString(pinFile)
        val offline = object : Net { override fun get(url: String, accept: String?) = NetResponse(404, "text/plain", ByteArray(0), url) }
        fetchMenus(IngestConfig(dataDir = raw.parent, raw = raw, menuPins = pinFile, log = {}), offline)
        assertEquals(before, Files.readString(pinFile))
        assertTrue(Files.exists(raw.resolve("wayback/pins.tsv")))
    }

    @Test
    fun pinningReplacesMovedCapturesAndCapturesAfterTheCutoff() {
        val moved = "https://moved.example/"
        val late = "https://late.example/"
        val old = WaybackPin(moved, "ok", "20240101000000", "$WAYBACK_WEB/20240101000000id_/$moved", "text/html", "0".repeat(64))
        val after = WaybackPin(late, "ok", "20261113010101", "$WAYBACK_WEB/20261113010101id_/$late", "text/html", "1".repeat(64))
        val net = object : Net {
            override fun get(url: String, accept: String?): NetResponse = when {
                url.startsWith(WAYBACK_AVAILABLE) && url.contains("moved") ->
                    NetResponse(200, "application/json", """{"archived_snapshots":{"closest":{"status":"200","timestamp":"20240101000000"}}}""".toByteArray(), url)
                url.startsWith(WAYBACK_AVAILABLE) ->
                    NetResponse(200, "application/json", """{"archived_snapshots":{"closest":{"status":"200","timestamp":"20261113010101"}}}""".toByteArray(), url)
                url.startsWith(WAYBACK_CDX) -> NetResponse(200, "text/plain", "20250101000000\n20261113010101".toByteArray(), url)
                // The archive redirects the pinned capture to a neighbouring one.
                url.contains("20240101000000id_") -> NetResponse(200, "text/html", "<p>a</p>".toByteArray(), url.replace("20240101000000", "20231231000000"))
                else -> NetResponse(200, "text/html", "<p>b</p>".toByteArray(), url)
            }
        }
        val f = WaybackFetcher(net, "20261112235959", mapOf(moved to old, late to after), Files.createTempDirectory("wbp"), discover = true, log = {}, retryMs = 1)
        f.page(moved); f.page(late)
        assertEquals("20231231000000", f.used[moved]!!.timestamp)
        assertTrue(f.used[moved]!!.archiveUrl!!.contains("20231231000000id_"))
        assertEquals("20250101000000", f.used[late]!!.timestamp)
        assertEquals(2, f.stats.repinned.get())
        // Without pinning, the late pin is not used and the moved one is reported.
        val g = WaybackFetcher(net, "20261112235959", mapOf(moved to old, late to after), Files.createTempDirectory("wbq"), discover = false, log = {}, retryMs = 1)
        assertEquals("moved", g.page(moved).status)
        assertEquals("after_cutoff", g.page(late).status)
    }

    @Test
    fun aCaptureRedirectedForAWhileIsReadOnTheSecondPass() {
        val u = "https://flaky.example/"
        val body = "<p>menu</p>"
        val pin = WaybackPin(u, "ok", "20240101000000", "$WAYBACK_WEB/20240101000000id_/$u", "text/html", sha256Hex(body.toByteArray()))
        var calls = 0
        val net = object : Net {
            override fun get(url: String, accept: String?): NetResponse {
                calls++
                // The first three tries are redirected to another capture, then the pinned one is served.
                return if (calls <= 3) NetResponse(200, "text/html", "<p>other</p>".toByteArray(), url.replace("20240101000000", "20230101000000"))
                else NetResponse(200, "text/html", body.toByteArray(), url)
            }
        }
        val dir = Files.createTempDirectory("wbr")
        val (used, stats) = fetchWaybackMenus(listOf(MenuTarget("node/1", u)), net, "20261112235959", mapOf(u to pin), dir, discover = false, threads = 1, log = {}, secondPassDelayMs = 1, retryMs = 1)
        assertEquals(1, stats.recovered.get())
        assertEquals("ok", pageFromStore(used[u], dir).status)
    }

    @Test
    fun verifyNamesTheKeysWhoseRowsChanged() {
        val home = Files.createTempDirectory("home")
        Files.createDirectories(home.resolve("pins"))
        Files.copy(fixture.resolve("wayback/pins.tsv"), home.resolve(MENU_PINS))
        Files.createDirectories(home.resolve("searxng"))
        Files.copy(example.resolve(SEARXNG_SCRIPT), home.resolve(SEARXNG_SCRIPT))
        Files.copy(example.resolve("build.gradle.kts"), home.resolve("build.gradle.kts"))
        val data = Files.createTempDirectory("data")
        val raw = data.resolve("raw")
        fixture.toFile().copyRecursively(raw.toFile())
        buildDatabase(IngestConfig(dataDir = data, raw = raw, log = {}))
        fun run(vararg a: String): Pair<Int, String> {
            val buf = ByteArrayOutputStream()
            val old = System.out
            System.setOut(PrintStream(buf, true))
            val rc = try { verify(listOf("--home", home.toString(), "--data", data.toString()) + a) } finally { System.setOut(old) }
            return rc to buf.toString()
        }
        assertEquals(0, run("--update").first)
        val m = parseJson(Files.readString(home.resolve("manifest.json"))) as JsonObject
        val ingest = m.optObject("stages")!!.optObject("ingest")!!
        for (t in INGEST_TABLES.map { it.table }) assertTrue(ingest.optObject("derived")!![t] != null, "$t not hashed")
        assertEquals("fixture", ingest.optObject("inputs")!!.optObject("overpass")!!.optString("query")) // the identity raw/manifest.tsv records
        assertEquals("0.1.0-alpha04", m.optObject("stages")!!.optObject("build")!!.optObject("inputs")!!.optObject("tlaloc")!!.optString("version"))
        assertTrue(Files.readString(home.resolve(ROWS_FILE)).contains("@inspections\n1001\t"))
        val (ok, out) = run()
        assertEquals(0, ok, out)

        // A source that changed a past row: reported by table and key.
        DriverManager.getConnection("jdbc:sqlite:${data.resolve("brooklyn.db")}").use { c ->
            c.createStatement().use { it.executeUpdate("UPDATE inspections SET score = 99 WHERE camis = '1002'") }
        }
        val (bad, out2) = run()
        assertEquals(1, bad)
        assertTrue(out2.contains("MISMATCH  ingest.derived.inspections"), out2)
        assertTrue(out2.contains("changed 1 (1002)"), out2)
        assertFalse(out2.contains("MISMATCH  ingest.derived.osm_places"), out2)

        // A changed SearXNG digest in the script is caught.
        val s = home.resolve(SEARXNG_SCRIPT)
        Files.writeString(s, Files.readString(s).replace(Regex("sha256:[0-9a-f]{64}"), "sha256:" + "0".repeat(64)))
        val out3 = run().second
        assertTrue(out3.contains("MISMATCH  tools.inputs.searxng_image"), out3)
    }

    /** A home directory with the committed files verify reads, and fixture data built into [data]. */
    private fun fixtureHome(data: Path): Path {
        val home = Files.createTempDirectory("home")
        Files.createDirectories(home.resolve("pins"))
        Files.copy(fixture.resolve("wayback/pins.tsv"), home.resolve(MENU_PINS))
        Files.createDirectories(home.resolve("searxng"))
        Files.copy(example.resolve(SEARXNG_SCRIPT), home.resolve(SEARXNG_SCRIPT))
        Files.copy(example.resolve("build.gradle.kts"), home.resolve("build.gradle.kts"))
        val raw = data.resolve("raw")
        fixture.toFile().copyRecursively(raw.toFile())
        buildDatabase(IngestConfig(dataDir = data, raw = raw, log = {}))
        return home
    }

    private fun runVerify(home: Path, data: Path, vararg a: String): Pair<Int, String> {
        val buf = ByteArrayOutputStream()
        val old = System.out
        System.setOut(PrintStream(buf, true))
        val rc = try { verify(listOf("--home", home.toString(), "--data", data.toString()) + a) } finally { System.setOut(old) }
        return rc to buf.toString()
    }

    @Test
    fun verifyFailsWhenAPinnedOutputWasNeverBuilt() {
        val data = Files.createTempDirectory("data")
        val home = fixtureHome(data)
        assertEquals(0, smoke(data.resolve("brooklyn.db"), listOf(ToolMode.SNAPSHOT)))
        assertEquals(0, runVerify(home, data, "--update").first)
        val (ok, out) = runVerify(home, data)
        assertEquals(0, ok, out)
        assertTrue(out.contains("ok        tools.derived.tools_smoke_snapshot"), out)

        // An empty data directory: every ingest table and the smoke are missing, and verify fails.
        val empty = Files.createTempDirectory("empty")
        val (bad, out2) = runVerify(home, empty)
        assertEquals(1, bad, out2)
        assertTrue(out2.contains("MISSING   ingest.derived.inspections"), out2)
        assertTrue(out2.contains("MISSING   ingest.inputs.overpass"), out2)
        assertTrue(out2.contains("MISSING   tools.derived.tools_smoke_snapshot"), out2)
        assertFalse(out2.contains("entries match"), out2)
    }

    @Test
    fun verifyFailsOnASmokeFileLeftFromAnEarlierDatabase() {
        val data = Files.createTempDirectory("data")
        val home = fixtureHome(data)
        val db = data.resolve("brooklyn.db")
        assertEquals(0, smoke(db, listOf(ToolMode.SNAPSHOT)))
        val before = Files.readAllBytes(data.resolve("tools-smoke-snapshot.jsonl"))
        // The file is exactly what verify recomputes from the database.
        assertEquals(String(before, Charsets.UTF_8), snapshotSmokeResults(db).joinToString("") { it + "\n" })
        // Ingest rerun with a changed source; the old smoke file stays.
        DriverManager.getConnection("jdbc:sqlite:$db").use { c ->
            c.createStatement().use { it.executeUpdate("UPDATE inspections SET score = score + 1") }
        }
        val (bad, out) = runVerify(home, data, "--update")
        assertEquals(1, bad, out)
        assertTrue(out.contains("STALE     tools.derived.tools_smoke_snapshot"), out)
        assertFalse(Files.exists(home.resolve("manifest.json")), "a stale output must not be written into the manifest")
        assertEquals(0, smoke(db, listOf(ToolMode.SNAPSHOT)))
        assertEquals(0, runVerify(home, data, "--update").first)
        assertEquals(0, runVerify(home, data).first)
    }

    @Test
    fun theMatchSampleSeedComesFromTheManifest() {
        val home = Files.createTempDirectory("home")
        val data = Files.createTempDirectory("data")
        Files.writeString(home.resolve("manifest.json"), """{"cutoff": "2026-11-13", "seeds": {"match_sample": 11}}""")
        assertEquals(11L, ingestConfig(listOf("--data", data.toString()), home).sampleSeed)
        Files.writeString(home.resolve("manifest.json"), """{"cutoff": "2026-11-13", "seeds": {}}""")
        val e = runCatching { ingestConfig(listOf("--data", data.toString()), home) }.exceptionOrNull()
        assertTrue(e?.message?.contains("seeds.match_sample") == true, e.toString())
        // The committed manifest's seed is the one ingest uses.
        assertEquals(7L, ingestConfig(listOf("--data", data.toString()), example).sampleSeed)
        // Different seeds give different samples on the fixture.
        val raw = data.resolve("raw")
        fixture.toFile().copyRecursively(raw.toFile())
        val a = buildDatabase(IngestConfig(dataDir = data, raw = raw, sampleSeed = 7, sampleSize = 3, log = {})).samples
        val b = buildDatabase(IngestConfig(dataDir = data, raw = raw, sampleSeed = 11, sampleSize = 3, log = {})).samples
        assertTrue(a != b || a.values.all { it.size <= 1 }, "seed did not change the sample: $a")
    }

    @Test
    fun rowsFileRoundTripsAndDiffs() {
        val f = Files.createTempDirectory("rows").resolve("rows.tsv")
        writeRowDigests(f, mapOf("t" to mapOf("b" to "2", "a" to "1"), "u" to mapOf("x y" to "3")))
        assertEquals(mapOf("t" to mapOf("a" to "1", "b" to "2"), "u" to mapOf("x y" to "3")), readRowDigests(f))
        assertEquals(listOf("changed 1 (a)", "new 1 (c)", "missing 1 (b)"), diffKeys(mapOf("a" to "1", "b" to "2"), mapOf("a" to "9", "c" to "3"), subset = false))
        assertEquals(listOf("changed 1 (a)"), diffKeys(mapOf("a" to "1", "b" to "2"), mapOf("a" to "9"), subset = true))
        assertEquals(searxngImage(example)!!.substringAfter("@sha256:").length, 64)
    }
}
