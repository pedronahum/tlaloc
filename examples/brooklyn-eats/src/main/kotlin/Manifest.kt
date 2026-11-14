package brooklyn

import io.tlaloc.core.io.JsonNumber
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

// manifest.json (committed): the cutoff, the seeds, the Tlaloc version and runtime classpath,
// and per stage the identities of its inputs and the SHA-256 of what it derives. Derived tables
// are hashed over a normalized dump (rows in key order, SQLite's quote() of each value), never
// over files that hold build or fetch times. digests/ingest-rows.tsv (committed) holds a short
// digest per key of the source tables, so a source that changed a past row is reported by table
// and key.
//
//   brooklyn-eats verify            recompute and compare; exit 1 on a mismatch
//   brooklyn-eats verify --update   write the recomputed values into manifest.json and the rows file
//
// Derived values are kept per scope ("full" or "quick"), read from the database's meta table.

const val DEFAULT_CUTOFF = "2026-11-13"
const val MENU_PINS = "pins/wayback-menus.tsv"
const val ROWS_FILE = "digests/ingest-rows.tsv"
const val SEARXNG_SCRIPT = "searxng/searxng.sh"
const val CUTOFF_RULE = "exclusive: OSM at 00:00 UTC of the cutoff day, DOHMH inspections and 311 complaints dated before it, Wayback captures up to 23:59:59 UTC of the day before"

/** The example directory: BROOKLYN_HOME, else the working directory (run.sh runs from the example). */
fun exampleHome(): Path = Paths.get(System.getenv("BROOKLYN_HOME") ?: ".").toAbsolutePath().normalize()

fun manifestPath(): Path = exampleHome().resolve("manifest.json")

fun readManifestJson(p: Path = manifestPath()): JsonObject? =
    if (Files.exists(p)) parseJson(Files.readString(p)) as JsonObject else null

/**
 * A seed from the "seeds" object of [manifest] (manifest.json). Every random choice of the pipeline
 * takes its seed from there; a missing seed is an error, never a silent default.
 */
fun manifestSeed(name: String, manifest: Path = manifestPath()): Long =
    readManifestJson(manifest)?.optObject("seeds")?.optNumber(name)?.toLong()
        ?: error("$manifest has no seeds.$name")

/** Pinned SHA-256 of the Google Local files by file name (empty when the manifest has none). */
fun manifestGoogleSha256(m: JsonObject? = readManifestJson()): Map<String, String> {
    val inputs = m?.path("stages", "ingest", "inputs") as? JsonObject ?: return emptyMap()
    return GOOGLE_LOCAL_FILES.mapNotNull { f ->
        (inputs[googleEntryName(f)] as? JsonObject)?.optString("sha256")?.let { f to it }
    }.toMap()
}

fun googleEntryName(file: String) = if (file.startsWith("meta")) "google_local_meta" else "google_local_reviews"

fun isQuick(args: List<String>): Boolean = "--quick" in args || System.getenv("BROOKLYN_QUICK") == "1"

fun hex(b: ByteArray): String = b.joinToString("") { "%02x".format(it) }

fun fileSha256(f: Path): String {
    val md = MessageDigest.getInstance("SHA-256")
    Files.newInputStream(f).use { ins ->
        val buf = ByteArray(1 shl 16)
        while (true) { val n = ins.read(buf); if (n < 0) break; md.update(buf, 0, n) }
    }
    return hex(md.digest())
}

/** SHA-256 and row count of `SELECT cols FROM table ORDER BY key`, one line per row, values by quote(). */
fun tableDigest(db: Path, table: String, cols: List<String>, orderBy: String): Pair<String, Long> =
    tableDigests(db, table, cols, "NULL", orderBy, perKey = false).let { it.sha256 to it.rows }

/** Digest of a whole table and, when [perKey], a 12-hex digest of the rows of each key. */
class TableDigests(val sha256: String, val rows: Long, val keys: Map<String, String>?)

fun tableDigests(db: Path, table: String, cols: List<String>, keyExpr: String, orderBy: String, perKey: Boolean): TableDigests {
    val md = MessageDigest.getInstance("SHA-256")
    var n = 0L
    val keys = if (perKey) LinkedHashMap<String, String>() else null
    var curKey: String? = null
    var keyMd: MessageDigest? = null
    fun closeKey() { if (curKey != null) keys!![curKey!!] = hex(keyMd!!.digest()).substring(0, 12) }
    DriverManager.getConnection("jdbc:sqlite:file:$db?mode=ro").use { c ->
        c.createStatement().use { st ->
            val sel = cols.joinToString(" || char(9) || ") { "quote($it)" }
            st.executeQuery("SELECT $sel, $keyExpr FROM $table ORDER BY $orderBy").use { rs ->
                while (rs.next()) {
                    val line = (rs.getString(1) + "\n").toByteArray(Charsets.UTF_8)
                    md.update(line); n++
                    if (keys != null) {
                        val k = rs.getString(2) ?: ""
                        if (k != curKey) { closeKey(); curKey = k; keyMd = MessageDigest.getInstance("SHA-256") }
                        keyMd!!.update(line)
                    }
                }
            }
        }
    }
    if (keys != null) closeKey()
    return TableDigests(hex(md.digest()), n, keys)
}

fun tableColumns(db: Path, table: String): List<String> = DriverManager.getConnection("jdbc:sqlite:file:$db?mode=ro").use { c ->
    c.createStatement().use { st ->
        st.executeQuery("PRAGMA table_info($table)").use { rs -> buildList { while (rs.next()) add(rs.getString("name")) } }
    }
}

private fun dbMeta(db: Path, key: String): String? = DriverManager.getConnection("jdbc:sqlite:file:$db?mode=ro").use { c ->
    c.prepareStatement("SELECT value FROM meta WHERE key=?").use { ps ->
        ps.setString(1, key); ps.executeQuery().use { rs -> if (rs.next()) rs.getString(1) else null }
    }
}

/**
 * One manifest entry: [stage]/[kind]/[name] -> {sha256, ...}; [compute] returns null when its file
 * is not there. Table entries also fill [keyDigests] (per key, for the rows file).
 */
class Entry(val stage: String, val kind: String, val name: String, val scoped: Boolean, val compute: () -> Map<String, Any?>?) {
    var keyDigests: Map<String, String>? = null
    /** Set by [compute] when the file is there but is not what its inputs give now (a stale output). */
    var stale: String? = null
}

/** A table hashed by the ingest stage: the key its rows are ordered (and reported) by. */
class TableSpec(val table: String, val key: String, val perKey: Boolean, val scoped: Boolean = false)

val INGEST_TABLES = listOf(
    TableSpec("osm_places", "osm_id", perKey = true),
    TableSpec("establishments", "camis", perKey = true),
    TableSpec("inspections", "camis", perKey = true),
    TableSpec("complaints", "unique_key", perKey = true),
    TableSpec("g_places", "gmap_id", perKey = false),
    TableSpec("reviews", "gmap_id", perKey = false),
    TableSpec("restaurants", "id", perKey = true),
    TableSpec("matches", "kind || ' ' || left_id", perKey = false),
    TableSpec("neighborhoods", "code", perKey = false),
    TableSpec("menus", "restaurant_id", perKey = true, scoped = true),
)

/** `val tlaloc = "<version>"` in the example's build.gradle.kts. */
fun tlalocVersion(home: Path): String? = home.resolve("build.gradle.kts").takeIf { Files.exists(it) }?.let { f ->
    Regex("""val tlaloc = "([^"]+)"""").find(Files.readString(f))?.groupValues?.get(1)
}

/**
 * The jars this process runs with (the installDist lib directory under run.sh), except the
 * example's own: name and SHA-256 of each, sorted by name. Null when not run from jars.
 */
fun runtimeJars(): List<Pair<String, String>>? {
    val jars = System.getProperty("java.class.path").split(java.io.File.pathSeparator)
        .map { Paths.get(it) }.filter { it.fileName.toString().endsWith(".jar") && Files.isRegularFile(it) }
        .filterNot { it.fileName.toString().startsWith("tlaloc-brooklyn-eats") }
    if (jars.isEmpty()) return null
    return jars.map { it.fileName.toString() to fileSha256(it) }.sortedBy { it.first }
}

/** The commit a release tag of the checkout points to; null outside a clone that has the tag. */
fun releaseCommit(home: Path, tag: String): String? = runCatching {
    val p = ProcessBuilder("git", "-C", home.toString(), "rev-parse", "--verify", "-q", "$tag^{commit}").redirectErrorStream(true).start()
    val out = p.inputStream.readAllBytes().toString(Charsets.UTF_8).trim()
    if (p.waitFor() == 0 && isCommitHash(out)) out else null
}.getOrNull()

/** The SearXNG image the start script runs (`image="..."`). */
fun searxngImage(home: Path): String? = home.resolve(SEARXNG_SCRIPT).takeIf { Files.exists(it) }?.let { f ->
    Regex("""(?m)^image="([^"]+)"""").find(Files.readString(f))?.groupValues?.get(1)
}

/** Every entry the manifest holds, computed from [home] and [data]. */
fun manifestEntries(home: Path, data: Path): List<Entry> {
    val db = data.resolve("brooklyn.db")
    val raw = data.resolve("raw")
    val cutoff = readManifestJson(home.resolve("manifest.json"))?.optString("cutoff")?.let(LocalDate::parse) ?: LocalDate.parse(DEFAULT_CUTOFF)
    fun file(f: Path, extra: Map<String, Any?> = emptyMap()) = if (!Files.exists(f)) null else linkedMapOf<String, Any?>("sha256" to fileSha256(f)) + extra
    fun recorded(source: String) = recordedIdentity(raw, source)?.let { linkedMapOf<String, Any?>("query" to it) }
    val out = ArrayList<Entry>()
    out += Entry("build", "inputs", "tlaloc", false) {
        tlalocVersion(home)?.let { v ->
            linkedMapOf<String, Any?>("version" to v, "release_tag" to "v$v", "maven" to "io.github.pedronahum:tlaloc-*:$v")
                .also { m -> releaseCommit(home, "v$v")?.let { m["commit"] = it } }
        }
    }
    out += Entry("build", "inputs", "runtime_classpath", false) {
        runtimeJars()?.let { jars ->
            linkedMapOf(
                "sha256" to sha256Hex(jars.joinToString("") { (n, h) -> "$n\t$h\n" }.toByteArray()),
                "jars" to jars.size.toLong(),
                // Tlaloc's jars are named <module>-jvm-<version>.jar in the installDist lib directory.
                "tlaloc_jars" to tlalocVersion(home)?.let { v -> jars.count { it.first.endsWith("-jvm-$v.jar") }.toLong() },
            )
        }
    }
    out += Entry("ingest", "inputs", "overpass", false) { recorded("osm") }
    out += Entry("ingest", "inputs", "dohmh", false) { recorded("inspections") }
    out += Entry("ingest", "inputs", "complaints_311", false) { recorded("complaints") }
    out += Entry("ingest", "inputs", "nta2020", false) { file(raw.resolve("nta2020.json"), mapOf("url" to NTA_URL)) }
    for (f in GOOGLE_LOCAL_FILES) out += Entry("ingest", "inputs", googleEntryName(f), false) {
        val p = raw.resolve("googlelocal").resolve(f)
        if (!Files.exists(p)) null else linkedMapOf("sha256" to cachedFileSha256(p), "bytes" to Files.size(p), "url" to "$GOOGLE_LOCAL/$f")
    }
    out += Entry("ingest", "inputs", "wayback_menu_pins", false) {
        val f = home.resolve(MENU_PINS)
        val pins = readPins(f)
        val last = cutoffTimestamp(cutoff)
        file(f, mapOf("file" to MENU_PINS, "pages" to pins.size.toLong(), "captures_after_cutoff" to pins.values.count { (it.timestamp ?: "") > last }.toLong()))
    }
    val tableSha = HashMap<String, String>()
    for (t in INGEST_TABLES) {
        lateinit var e: Entry
        e = Entry("ingest", "derived", t.table, t.scoped) {
            if (!Files.exists(db)) null else {
                val cols = tableColumns(db, t.table)
                val d = tableDigests(db, t.table, cols, t.key, (listOf(t.key) + cols).joinToString(", "), t.perKey)
                e.keyDigests = d.keys
                tableSha[t.table] = d.sha256
                linkedMapOf("sha256" to d.sha256, "rows" to d.rows)
            }
        }
        out += e
    }
    out += Entry("tools", "inputs", "searxng_image", false) { searxngImage(home)?.let { linkedMapOf("image" to it) } }
    out += Entry("tools", "derived", "tool_schemas", false) {
        linkedMapOf("sha256" to sha256Hex(toolDefinitionsJson().toByteArray()), "tools" to TOOL_DEFS.size.toLong())
    }
    lateinit var smoke: Entry
    smoke = Entry("tools", "derived", "tools_smoke_snapshot", true) {
        val f = data.resolve("tools-smoke-snapshot.jsonl")
        if (!Files.exists(f)) null else {
            // The file must be what the current database gives: the snapshot calls are rerun here.
            val now = if (Files.exists(db)) snapshotSmokeResults(db) else null
            val want = now?.let { sha256Hex(it.joinToString("") { l -> l + "\n" }.toByteArray(Charsets.UTF_8)) }
            val sha = fileSha256(f)
            if (want == null) smoke.stale = "no brooklyn.db to recompute it from"
            else if (want != sha) smoke.stale = "the file differs from the snapshot results of the current brooklyn.db ($sha, recomputed $want); rerun the tools stage"
            linkedMapOf<String, Any?>("sha256" to sha, "calls" to Files.readAllLines(f).size.toLong())
        }
    }
    out += smoke
    // Task splits: the file hashes, and STALE when tasks/inputs.tsv (the cutoff, seeds and table
    // digests the files were generated from) differs from the current ones. `tasks check` regenerates.
    for (sp in Split.entries) {
        lateinit var e: Entry
        e = Entry("tasks", "derived", "split_${sp.label}", true) {
            val f = taskFile(data, sp)
            if (!Files.exists(f)) null else {
                val rec = readTaskInputs(data)
                val now = taskInputs(home, tableSha)
                val diff = (now.keys + rec.keys).filter { now[it] != rec[it] }.sorted()
                if (rec.isEmpty()) e.stale = "no $TASKS_DIR/inputs.tsv; rerun the tasks stage"
                else if (diff.isNotEmpty()) e.stale = "generated from other inputs (${diff.joinToString()}); rerun the tasks stage"
                val tasks = readTasks(f)
                linkedMapOf<String, Any?>("sha256" to fileSha256(f), "rows" to tasks.size.toLong(), "categories" to categoryCounts(tasks))
            }
        }
        out += e
    }
    out += agentEntries(home, data)
    return out
}

/** What the task splits are generated from: cutoff, seeds and the digest of every ingest table. */
fun taskInputs(home: Path, tableSha: Map<String, String>): Map<String, String> {
    val m = readManifestJson(home.resolve("manifest.json"))
    val out = java.util.TreeMap<String, String>()
    out["cutoff"] = m?.optString("cutoff") ?: DEFAULT_CUTOFF
    for (k in listOf("tasks", "task_split")) out["seed.$k"] = (m?.optObject("seeds")?.optNumber(k)?.toLong() ?: "missing").toString()
    for (t in INGEST_TABLES) out["table.${t.table}"] = tableSha[t.table] ?: "missing"
    return out
}

fun ingestTableSha(db: Path): Map<String, String> = INGEST_TABLES.associate { t ->
    val cols = tableColumns(db, t.table)
    t.table to tableDigests(db, t.table, cols, t.key, (listOf(t.key) + cols).joinToString(", "), perKey = false).sha256
}

fun readTaskInputs(data: Path): Map<String, String> {
    val f = data.resolve(TASKS_DIR).resolve("inputs.tsv")
    if (!Files.exists(f)) return emptyMap()
    return Files.readAllLines(f).filter { '\t' in it }.associate { it.substringBefore('\t') to it.substringAfter('\t') }
}

fun writeTaskInputs(data: Path, inputs: Map<String, String>) {
    Files.writeString(data.resolve(TASKS_DIR).resolve("inputs.tsv"), inputs.entries.joinToString("") { "${it.key}\t${it.value}\n" })
}

fun JsonObject.path(vararg ks: String): JsonValue? {
    var v: JsonValue? = this
    for (k in ks) v = (v as? JsonObject)?.get(k)
    return v
}

private fun JsonValue.text(): String = (this as? JsonString)?.value ?: (this as? JsonNumber)?.raw ?: (this as? io.tlaloc.core.io.JsonBool)?.value?.toString() ?: toString()

/** The rows file: table -> key -> 12-hex digest. */
fun readRowDigests(f: Path): Map<String, Map<String, String>> {
    if (!Files.exists(f)) return emptyMap()
    val out = LinkedHashMap<String, LinkedHashMap<String, String>>()
    var cur: LinkedHashMap<String, String>? = null
    for (l in Files.readAllLines(f)) {
        when {
            l.startsWith("#") || l.isEmpty() -> Unit
            l.startsWith("@") -> cur = out.getOrPut(l.substring(1)) { LinkedHashMap() }
            else -> cur?.let { m -> val i = l.lastIndexOf('\t'); if (i > 0) m[l.substring(0, i)] = l.substring(i + 1) }
        }
    }
    return out
}

fun writeRowDigests(f: Path, tables: Map<String, Map<String, String>>) {
    Files.createDirectories(f.toAbsolutePath().parent)
    val lines = ArrayList<String>()
    lines += "# Per-key digests of the ingest tables (first 12 hex of the SHA-256 of the key's rows, as hashed for manifest.json)."
    lines += "# Written by `tlaloc-brooklyn-eats verify --update`; `verify` names the keys whose rows differ."
    for ((t, m) in tables) {
        lines += "@$t"
        for ((k, h) in m.entries.sortedBy { it.key }) lines += "$k\t$h"
    }
    Files.write(f, lines)
}

/** Keys whose rows differ between [want] and [now]; [subset] ignores keys missing from [now] (quick scope). */
fun diffKeys(want: Map<String, String>, now: Map<String, String>, subset: Boolean): List<String> {
    val changed = now.filter { (k, v) -> want[k] != null && want[k] != v }.keys.sorted()
    val added = (now.keys - want.keys).sorted()
    val missing = if (subset) emptyList() else (want.keys - now.keys).sorted()
    fun show(label: String, ks: List<String>) = if (ks.isEmpty()) null else "$label ${ks.size} (" + ks.take(8).joinToString(", ") + if (ks.size > 8) ", ...)" else ")"
    return listOfNotNull(show("changed", changed), show("new", added), show("missing", missing))
}

/** `verify [--update] [--data DIR] [--home DIR] [--cache DIR] [--rehash] [--no-models]` */
fun verify(args: List<String>): Int {
    val home = args.indexOf("--home").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]).toAbsolutePath().normalize() } ?: exampleHome()
    val data = args.indexOf("--data").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: dataHome()
    val db = data.resolve("brooklyn.db")
    val scope = if (Files.exists(db)) dbMeta(db, "menu_scope")?.takeIf { it != "none" } ?: "full" else "full"
    val manifest = readManifestJson(home.resolve("manifest.json"))
    val rowsWant = readRowDigests(home.resolve(ROWS_FILE))
    val update = "--update" in args
    var bad = 0
    var checked = 0
    var unpinned = 0
    val computed = LinkedHashMap<Entry, Map<String, Any?>?>()
    for (e in manifestEntries(home, data)) {
        val now = e.compute()
        computed[e] = now
        val keys = listOf("stages", e.stage, e.kind, e.name) + if (e.scoped) listOf(scope) else emptyList()
        val want = manifest?.path(*keys.toTypedArray()) as? JsonObject
        val label = "${e.stage}.${e.kind}.${e.name}" + if (e.scoped) " ($scope)" else ""
        when {
            // A pinned entry with nothing to hash is a failure: a run that skipped a stage does not verify.
            now == null && want != null -> { checked++; bad++; println("MISSING   $label: not built (run the ${e.stage} stage)") }
            now == null -> { unpinned++; println("skip      $label: not built and not in the manifest") }
            e.stale != null -> { checked++; bad++; println("STALE     $label: ${e.stale}") }
            want == null -> { unpinned++; println("unpinned  $label: ${now["sha256"] ?: now}") }
            else -> {
                checked++
                val diffs = now.filter { (k, v) -> want[k] != null && want[k]!!.text() != v?.toString() }
                if (diffs.isEmpty()) println("ok        $label")
                else {
                    bad++
                    println("MISMATCH  $label: " + diffs.entries.joinToString("; ") { (k, v) -> "$k ${want[k]!!.text()} -> $v" })
                    val kd = e.keyDigests
                    val ref = rowsWant[e.name]
                    if (kd != null && ref != null) {
                        val d = diffKeys(ref, kd, subset = e.scoped && scope != "full")
                        println("          keys: " + d.ifEmpty { listOf("every key's rows match; the order or set of columns differs") }.joinToString("; "))
                    }
                }
            }
        }
    }
    if (!update) {
        val (mChecked, mBad) = verifyModels(args, manifest, data)
        checked += mChecked; bad += mBad
    }
    if (update) {
        val stale = computed.keys.filter { it.stale != null }
        if (stale.isNotEmpty()) {
            println("verify --update: not written; stale outputs: " + stale.joinToString { it.name })
            return 1
        }
        writeManifest(home.resolve("manifest.json"), manifest, computed, scope, codeCommit(home))
        val rows = LinkedHashMap<String, Map<String, String>>()
        for (e in computed.keys) {
            val kd = e.keyDigests ?: continue
            // The rows file holds the full run's menus; a quick run's are a subset of them.
            rows[e.name] = if (e.scoped && scope != "full") rowsWant[e.name] ?: continue else kd
        }
        for ((t, m) in rowsWant) rows.putIfAbsent(t, m)
        writeRowDigests(home.resolve(ROWS_FILE), rows)
        println("manifest.json and $ROWS_FILE updated (scope $scope)")
        return 0
    }
    println(
        (if (bad == 0) "verify: $checked entries match" else "verify: $bad of $checked entries differ") +
            if (unpinned > 0) "; $unpinned not in the manifest" else "",
    )
    return if (bad == 0) 0 else 1
}

/** HEAD of the checkout and whether the example's code (not its outputs) differs from it. */
fun codeCommit(home: Path): Map<String, Any?>? = runCatching {
    fun git(vararg a: String): String {
        val p = ProcessBuilder(listOf("git", "-C", home.toString()) + a).redirectErrorStream(true).start()
        val out = p.inputStream.readAllBytes().toString(Charsets.UTF_8).trim()
        check(p.waitFor() == 0) { out }
        return out
    }
    val head = git("rev-parse", "HEAD")
    val dirty = git("status", "--porcelain", "--", "src", "run.sh", "build.gradle.kts", "settings.gradle.kts", "searxng", "serving")
    linkedMapOf<String, Any?>("commit" to head, "code_clean" to dirty.isEmpty())
}.getOrNull()

/** Rewrites manifest.json: keeps what it holds, sets the computed entries (scoped ones under [scope]). */
fun writeManifest(f: Path, old: JsonObject?, computed: Map<Entry, Map<String, Any?>?>, scope: String, code: Map<String, Any?>? = null) {
    @Suppress("UNCHECKED_CAST")
    val root = (old?.toKotlin() as Map<String, Any?>?)?.let { deepMutableMap(it) } ?: linkedMapOf()
    root.putIfAbsent("cutoff", DEFAULT_CUTOFF)
    root.putIfAbsent("cutoff_rule", CUTOFF_RULE)
    root.putIfAbsent("seeds", linkedMapOf<String, Any?>("match_sample" to 7L))
    @Suppress("UNCHECKED_CAST")
    fun sub(m: MutableMap<String, Any?>, k: String) = m.getOrPut(k) { linkedMapOf<String, Any?>() } as MutableMap<String, Any?>
    if (code != null) sub(root, "produced_by")[scope] = code
    val stages = sub(root, "stages")
    // Entries this code no longer computes are dropped from the stages and kinds it computes.
    for ((sk, es) in computed.keys.groupBy { it.stage to it.kind }) {
        sub(sub(stages, sk.first), sk.second).keys.retainAll(es.map { it.name }.toSet())
    }
    for ((e, v) in computed) {
        if (v == null) continue
        val kind = sub(sub(stages, e.stage), e.kind)
        if (e.scoped) sub(kind, e.name)[scope] = v else kind[e.name] = v
    }
    Files.writeString(f, prettyJson(root) + "\n")
}

@Suppress("UNCHECKED_CAST")
fun deepMutableMap(m: Map<String, Any?>): MutableMap<String, Any?> =
    m.mapValuesTo(linkedMapOf()) { (_, v) -> if (v is Map<*, *>) deepMutableMap(v as Map<String, Any?>) else v }

/** Two-space indented JSON with keys in insertion order. */
fun prettyJson(v: Any?, indent: String = ""): String = when (v) {
    is Map<*, *> -> if (v.isEmpty()) "{}" else v.entries.joinToString(",\n", "{\n", "\n$indent}") { (k, x) ->
        "$indent  ${jsonQuote(k.toString())}: ${prettyJson(x, "$indent  ")}"
    }
    is List<*> -> if (v.all { it !is Map<*, *> && it !is List<*> }) toJson(v)
    else v.joinToString(",\n", "[\n", "\n$indent]") { "$indent  ${prettyJson(it, "$indent  ")}" }
    else -> toJson(v)
}

/**
 * The pinned models in the local Hugging Face cache: every file's size and SHA-256 at the pinned
 * commit. An optional model not in the cache is skipped. Returns (checked, failed).
 */
fun verifyModels(args: List<String>, manifest: JsonObject?, data: Path): Pair<Int, Int> {
    val pins = readModelPins(manifest)
    if (pins.isEmpty()) return 0 to 0
    if ("--no-models" in args) { println("skip      models.inputs: --no-models"); return 0 to 0 }
    val cache = args.indexOf("--cache").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: hubCache()
    val hashes = HashCache(data.resolve(MODEL_HASHES), "--rehash" in args)
    var checked = 0
    var bad = 0
    val present = pins.filter { pin ->
        if (!isCommitHash(pin.revision)) { checked++; bad++; println("MISMATCH  models.inputs.${pin.key}: revision ${pin.revision} is not a commit hash"); false }
        else if (pin.optional && !Files.isDirectory(snapshotDir(cache, pin))) { println("skip      models.inputs.${pin.key}: optional, not in $cache"); false }
        else true
    }
    val checks = checkModels(cache, present, hashes)
    hashes.save()
    for (pin in present) {
        val (good, line) = reportModel(pin, checks.filter { it.pin === pin })
        checked++
        if (!good) bad++
        println(line)
    }
    return checked to bad
}
