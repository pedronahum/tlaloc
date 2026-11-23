package brooklyn

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.sql.DriverManager
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The data snapshot: reproducible bytes, what it unpacks, and the OSM tag order. */
class SnapshotTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent

    private fun dataDir(name: String, sampled: Boolean): Path {
        val d = Files.createTempDirectory(name)
        val raw = d.resolve("raw")
        Files.createDirectories(raw)
        for (r in SNAPSHOT_RAW) {
            val src = fixture.resolve(r)
            if (Files.isDirectory(src)) {
                Files.createDirectories(raw.resolve(r))
                Files.list(src).use { s -> s.forEach { Files.copy(it, raw.resolve(r).resolve(it.fileName)) } }
            } else Files.copy(src, raw.resolve(r))
        }
        if (sampled) {
            Files.createDirectories(d.resolve("teach/shards"))
            Files.writeString(d.resolve("teach/sample-0.jsonl"), "{\"id\":\"$name\"}\n")
            Files.writeString(d.resolve("teach/shards/shard-0.bin"), "tokens")
        }
        return d
    }

    @Test
    fun theSameFilesGiveTheSameZip() {
        val full = dataDir("snap-full", sampled = true)
        val quick = dataDir("snap-quick", sampled = true)
        val out = Files.createTempDirectory("snap-out")
        val entries = snapshotEntries(full, quick)
        writeSnapshot(entries, out.resolve("a.zip"))
        Thread.sleep(1100) // a zip that took the file times would differ now
        writeSnapshot(snapshotEntries(full, quick), out.resolve("b.zip"))
        assertEquals(fileSha256(out.resolve("a.zip")), fileSha256(out.resolve("b.zip")))
        val names = entries.map { it.first }
        assertTrue("raw/overpass.json" in names && "raw/dohmh/page-000.json" in names && "raw/311/page-000.json" in names, names.toString())
        assertTrue("full/teach/sample-0.jsonl" in names && "quick/teach/shards/shard-0.bin" in names, names.toString())
        assertFalse(names.any { it.contains("googlelocal") || it.contains("wayback") }, names.toString())
    }

    @Test
    fun unpackWritesRawAndTheScopesSampledOutputsOnly() {
        val full = dataDir("snap-full2", sampled = true)
        val quick = dataDir("snap-quick2", sampled = true)
        val zip = Files.createTempDirectory("snap-zip").resolve("s.zip")
        writeSnapshot(snapshotEntries(full, quick), zip)

        val target = Files.createTempDirectory("snap-target")
        val counts = unpackSnapshot(zip, target, "quick", log = {})
        assertEquals("{\"id\":\"snap-quick2\"}\n", Files.readString(target.resolve("teach/sample-0.jsonl")))
        assertEquals(fileSha256(fixture.resolve("overpass.json")), fileSha256(target.resolve("raw/overpass.json")))
        assertEquals(2, counts["teach"])
        // Again: everything is there and the same, nothing is written.
        assertEquals(emptyMap(), unpackSnapshot(zip, target, "quick", log = {}))

        // A run that sampled again keeps its own passes.
        val other = Files.createTempDirectory("snap-other")
        Files.createDirectories(other.resolve("teach"))
        Files.writeString(other.resolve("teach/sample-0.jsonl"), "mine\n")
        unpackSnapshot(zip, other, "full", log = {})
        assertEquals("mine\n", Files.readString(other.resolve("teach/sample-0.jsonl")))

        // A raw file fetched with other content is never overwritten.
        val drifted = Files.createTempDirectory("snap-drift")
        Files.createDirectories(drifted.resolve("raw"))
        Files.writeString(drifted.resolve("raw/nta2020.json"), "[]")
        assertFailsWith<IllegalStateException> { unpackSnapshot(zip, drifted, "full", log = {}) }
    }

    @Test
    fun rawContentDigestCoversEveryPage() {
        val d = dataDir("snap-digest", sampled = false).resolve("raw")
        val before = rawContentSha256(d.resolve("dohmh"))!!
        Files.writeString(d.resolve("dohmh/page-001.json"), "[]")
        assertTrue(before != rawContentSha256(d.resolve("dohmh")))
        Files.writeString(d.resolve("dohmh/page-002.json.part"), "partial")
        val withPart = rawContentSha256(d.resolve("dohmh"))
        Files.delete(d.resolve("dohmh/page-002.json.part"))
        assertEquals(withPart, rawContentSha256(d.resolve("dohmh")))
        assertEquals(null, rawContentSha256(d.resolve("absent")))
    }

    @Test
    fun osmTagsAreStoredInKeyOrder() {
        val dir = Files.createTempDirectory("brooklyn-osm-order")
        val raw = fixture
        // The fixture's Overpass response with every element's tags reversed: the table must not change.
        val reversed = Files.createTempDirectory("brooklyn-osm-rev")
        Files.walk(raw).use { s ->
            s.forEach { p ->
                val t = reversed.resolve(raw.relativize(p).toString())
                if (Files.isDirectory(p)) Files.createDirectories(t) else Files.copy(p, t)
            }
        }
        val o = io.tlaloc.core.io.parseJson(Files.readString(raw.resolve("overpass.json"))) as io.tlaloc.core.io.JsonObject
        val els = (o["elements"] as io.tlaloc.core.io.JsonArray).elements.map { e ->
            e as io.tlaloc.core.io.JsonObject
            val tags = e["tags"] as? io.tlaloc.core.io.JsonObject
            "{" + e.fields.entries.joinToString(",") { (k, v) ->
                io.tlaloc.core.io.jsonQuote(k) + ":" + if (k == "tags" && tags != null) {
                    tags.fields.entries.reversed().joinToString(",", "{", "}") { (tk, tv) -> io.tlaloc.core.io.jsonQuote(tk) + ":" + toJson(tv.toKotlin()) }
                } else toJson(v.toKotlin())
            } + "}"
        }
        Files.writeString(reversed.resolve("overpass.json"), "{\"elements\":[" + els.joinToString(",") + "]}")
        fun tagsColumn(r: Path, name: String): List<String> {
            val cfg = IngestConfig(dataDir = Files.createDirectories(dir.resolve(name)), raw = r, cutoff = LocalDate.parse("2026-11-13"), log = {})
            buildDatabase(cfg)
            return DriverManager.getConnection("jdbc:sqlite:${cfg.db}").use { c ->
                c.createStatement().executeQuery("SELECT tags_json FROM osm_places ORDER BY osm_id").use { rs -> buildList { while (rs.next()) add(rs.getString(1)) } }
            }
        }
        val a = tagsColumn(raw, "a")
        assertTrue(a.isNotEmpty())
        assertEquals(a, tagsColumn(reversed, "b"))
    }
}
