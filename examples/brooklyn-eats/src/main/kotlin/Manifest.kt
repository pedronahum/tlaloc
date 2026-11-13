package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest
import java.sql.DriverManager
import java.time.LocalDate

// manifest.json (committed): the cutoff, the seeds, and per stage the identities of its inputs
// and the SHA-256 of what it derives. Derived tables are hashed over a normalized dump (rows in
// key order, SQLite's quote() of each value), never over files that hold build times.
//
//   brooklyn-eats verify            recompute and compare; exit 1 on a mismatch
//   brooklyn-eats verify --update   write the recomputed values into manifest.json
//
// Derived values are kept per scope ("full" or "quick"), read from the database's meta table.

const val DEFAULT_CUTOFF = "2026-11-13"
const val MENU_PINS = "pins/wayback-menus.tsv"

/** The example directory: BROOKLYN_HOME, else the working directory (run.sh runs from the example). */
fun exampleHome(): Path = Paths.get(System.getenv("BROOKLYN_HOME") ?: ".").toAbsolutePath().normalize()

fun manifestPath(): Path = exampleHome().resolve("manifest.json")

fun readManifestJson(p: Path = manifestPath()): JsonObject? =
    if (Files.exists(p)) parseJson(Files.readString(p)) as JsonObject else null

/** The manifest's cutoff, else [DEFAULT_CUTOFF]. */
fun manifestCutoff(): LocalDate = LocalDate.parse(readManifestJson()?.optString("cutoff") ?: DEFAULT_CUTOFF)

/** A seed from the manifest's "seeds" object, else [default]. */
fun manifestSeed(name: String, default: Long): Long = readManifestJson()?.optObject("seeds")?.optNumber(name)?.toLong() ?: default

fun isQuick(args: List<String>): Boolean = "--quick" in args || System.getenv("BROOKLYN_QUICK") == "1"

fun fileSha256(f: Path): String {
    val md = MessageDigest.getInstance("SHA-256")
    Files.newInputStream(f).use { ins ->
        val buf = ByteArray(1 shl 16)
        while (true) { val n = ins.read(buf); if (n < 0) break; md.update(buf, 0, n) }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
}

/** SHA-256 and row count of `SELECT cols FROM table ORDER BY key`, one line per row, values by quote(). */
fun tableDigest(db: Path, table: String, cols: List<String>, orderBy: String): Pair<String, Long> {
    val md = MessageDigest.getInstance("SHA-256")
    var n = 0L
    DriverManager.getConnection("jdbc:sqlite:file:$db?mode=ro").use { c ->
        c.createStatement().use { st ->
            val sel = cols.joinToString(" || char(9) || ") { "quote($it)" }
            st.executeQuery("SELECT $sel FROM $table ORDER BY $orderBy").use { rs ->
                while (rs.next()) { md.update((rs.getString(1) + "\n").toByteArray(Charsets.UTF_8)); n++ }
            }
        }
    }
    return md.digest().joinToString("") { "%02x".format(it) } to n
}

private fun dbMeta(db: Path, key: String): String? = DriverManager.getConnection("jdbc:sqlite:file:$db?mode=ro").use { c ->
    c.prepareStatement("SELECT value FROM meta WHERE key=?").use { ps ->
        ps.setString(1, key); ps.executeQuery().use { rs -> if (rs.next()) rs.getString(1) else null }
    }
}

/** One manifest entry: [stage]/[kind]/[name] -> {sha256, ...}; null when its file is not there. */
class Entry(val stage: String, val kind: String, val name: String, val scoped: Boolean, val compute: () -> Map<String, Any?>?)

/** Every entry the manifest holds, computed from [home] and [data]. */
fun manifestEntries(home: Path, data: Path): List<Entry> {
    val db = data.resolve("brooklyn.db")
    val raw = data.resolve("raw")
    fun file(f: Path, extra: Map<String, Any?> = emptyMap()) = if (!Files.exists(f)) null else linkedMapOf<String, Any?>("sha256" to fileSha256(f)) + extra
    fun table(t: String, cols: List<String>, key: String) = if (!Files.exists(db)) null else tableDigest(db, t, cols, key).let { (h, n) -> linkedMapOf("sha256" to h, "rows" to n) }
    return listOf(
        Entry("ingest", "inputs", "nta2020", false) { file(raw.resolve("nta2020.json"), mapOf("url" to NTA_URL)) },
        Entry("ingest", "derived", "neighborhoods", false) {
            table("neighborhoods", listOf("code", "name", "cdta", "type", "lat", "lon", "geom"), "code")
        },
        Entry("ingest", "derived", "restaurants.nta2020", false) {
            table("restaurants", listOf("id", "nta2020", "neighborhood"), "id").takeIf { Files.exists(raw.resolve("nta2020.json")) }
        },
        Entry("ingest", "inputs", "wayback_menu_pins", false) {
            val f = home.resolve(MENU_PINS)
            file(f, mapOf("file" to MENU_PINS, "pages" to readPins(f).size, "cutoff" to manifestCutoff().toString()))
        },
        Entry("ingest", "derived", "menus", true) {
            table("menus", listOf("restaurant_id", "osm_id", "url", "page_url", "snapshot", "archive_url", "status", "items", "text"), "restaurant_id")
        },
        Entry("tools", "derived", "tool_schemas", false) {
            linkedMapOf("sha256" to sha256Hex(toolDefinitionsJson().toByteArray()), "tools" to TOOL_DEFS.size)
        },
        Entry("tools", "inputs", "searxng_image", false) { linkedMapOf("image" to SEARXNG_IMAGE) },
    )
}

const val SEARXNG_IMAGE = "searxng/searxng@sha256:287af61f1a13e86d02974fcde91a3413a9945788de38b61ffbc491d81a19bcc2"

private fun JsonObject.path(vararg ks: String): JsonValue? {
    var v: JsonValue? = this
    for (k in ks) v = (v as? JsonObject)?.get(k)
    return v
}

/** `verify [--update] [--data DIR]` */
fun verify(args: List<String>): Int {
    val home = exampleHome()
    val data = args.indexOf("--data").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: dataHome()
    val db = data.resolve("brooklyn.db")
    val scope = if (Files.exists(db)) dbMeta(db, "menu_scope")?.takeIf { it != "none" } ?: "full" else "full"
    val manifest = readManifestJson(home.resolve("manifest.json"))
    val update = "--update" in args
    var bad = 0
    var checked = 0
    val computed = LinkedHashMap<Entry, Map<String, Any?>?>()
    for (e in manifestEntries(home, data)) {
        val now = e.compute()
        computed[e] = now
        val keys = listOf("stages", e.stage, e.kind, e.name) + if (e.scoped) listOf(scope) else emptyList()
        val want = manifest?.path(*keys.toTypedArray()) as? JsonObject
        val label = "${e.stage}.${e.kind}.${e.name}" + if (e.scoped) " ($scope)" else ""
        when {
            now == null -> println("skip      $label: not built")
            want == null -> println("unpinned  $label: ${now["sha256"] ?: now}")
            else -> {
                checked++
                val diffs = now.filter { (k, v) -> want[k] != null && want.optString(k) != v?.toString() && (want[k] as? io.tlaloc.core.io.JsonNumber)?.raw != v?.toString() }
                if (diffs.isEmpty()) println("ok        $label")
                else { bad++; println("MISMATCH  $label: " + diffs.entries.joinToString { (k, v) -> "$k ${want[k]?.let { (it as? JsonString)?.value ?: (it as? io.tlaloc.core.io.JsonNumber)?.raw ?: it.toString() }} -> $v" }) }
            }
        }
    }
    if (update) {
        writeManifest(home.resolve("manifest.json"), manifest, computed, scope)
        println("manifest.json updated (scope $scope)")
        return 0
    }
    println(if (bad == 0) "verify: $checked entries match" else "verify: $bad of $checked entries differ")
    return if (bad == 0) 0 else 1
}

/** Rewrites manifest.json: keeps what it holds, sets the computed entries (scoped ones under [scope]). */
fun writeManifest(f: Path, old: JsonObject?, computed: Map<Entry, Map<String, Any?>?>, scope: String) {
    @Suppress("UNCHECKED_CAST")
    val root = (old?.toKotlin() as Map<String, Any?>?)?.let { deepMutable(it) } ?: linkedMapOf()
    root.putIfAbsent("cutoff", DEFAULT_CUTOFF)
    root.putIfAbsent("seeds", linkedMapOf<String, Any?>("match_sample" to 7L))
    @Suppress("UNCHECKED_CAST")
    fun sub(m: MutableMap<String, Any?>, k: String) = m.getOrPut(k) { linkedMapOf<String, Any?>() } as MutableMap<String, Any?>
    val stages = sub(root, "stages")
    for ((e, v) in computed) {
        if (v == null) continue
        val kind = sub(sub(stages, e.stage), e.kind)
        if (e.scoped) sub(kind, e.name)[scope] = v else kind[e.name] = v
    }
    Files.writeString(f, prettyJson(root) + "\n")
}

@Suppress("UNCHECKED_CAST")
private fun deepMutable(m: Map<String, Any?>): MutableMap<String, Any?> =
    m.mapValuesTo(linkedMapOf()) { (_, v) -> if (v is Map<*, *>) deepMutable(v as Map<String, Any?>) else v }

/** Two-space indented JSON with keys in insertion order. */
fun prettyJson(v: Any?, indent: String = ""): String = when (v) {
    is Map<*, *> -> if (v.isEmpty()) "{}" else v.entries.joinToString(",\n", "{\n", "\n$indent}") { (k, x) ->
        "$indent  ${jsonQuote(k.toString())}: ${prettyJson(x, "$indent  ")}"
    }
    is List<*> -> if (v.all { it !is Map<*, *> && it !is List<*> }) toJson(v)
    else v.joinToString(",\n", "[\n", "\n$indent]") { "$indent  ${prettyJson(it, "$indent  ")}" }
    else -> toJson(v)
}
