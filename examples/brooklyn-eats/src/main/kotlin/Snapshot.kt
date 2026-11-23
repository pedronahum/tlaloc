package brooklyn

import io.tlaloc.core.io.JsonObject
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

// The data snapshot: the raw files of the sources that change after the cutoff (OSM, DOHMH, 311,
// NTA) and the sampled outputs (the teacher's and the student's trajectories and their shards, per
// scope), in one zip published as a GitHub release asset and pinned by SHA-256 in manifest.json
// ("data_snapshot"). DOHMH and 311 are rolling datasets: a query with the cutoff no longer returns
// what it returned on the cutoff day, so a run that should reproduce the manifest starts from the
// snapshot. Google Local (a fixed research file, pinned by SHA-256) and the Wayback pages (pinned
// captures) are fetched from their sources.
//
//   brooklyn-eats snapshot pack --out FILE   write the zip from build/data and build/quick-data
//   brooklyn-eats snapshot fetch [--quick]   download the pinned zip, check it, unpack what is missing
//   brooklyn-eats snapshot list FILE         the entries of a zip

/** Raw files and directories of the snapshot, relative to the raw directory. */
val SNAPSHOT_RAW = listOf("overpass.json", "nta2020.json", "manifest.tsv", "dohmh", "311")

/** Sampled outputs of the snapshot, relative to a data directory. */
val SNAPSHOT_SAMPLED = listOf(TEACH_DIR, "rft")

/** The fixed modification time of every zip entry (the cutoff day), so the zip is reproducible. */
private val ENTRY_TIME: Long = java.time.LocalDate.parse(DEFAULT_CUTOFF).atStartOfDay(java.time.ZoneOffset.UTC).toInstant().toEpochMilli()

class SnapshotPin(val url: String, val sha256: String, val bytes: Long?)

fun snapshotPin(m: JsonObject? = readManifestJson()): SnapshotPin? {
    val o = m?.optObject("data_snapshot") ?: return null
    return SnapshotPin(o.optString("url") ?: return null, o.optString("sha256") ?: return null, o.optNumber("bytes")?.toLong())
}

/** Regular files under [dir] (or [dir] itself when it is a file), sorted by relative path. */
private fun filesUnder(dir: Path): List<Path> =
    if (Files.isRegularFile(dir)) listOf(dir)
    else if (!Files.isDirectory(dir)) emptyList()
    else Files.walk(dir).use { s -> s.filter { Files.isRegularFile(it) && !it.fileName.toString().endsWith(".part") }.toList() }
        .sortedBy { it.toString() }

/** The zip's entries: name in the zip -> file. Raw from [full]; sampled outputs per scope. */
fun snapshotEntries(full: Path, quick: Path): List<Pair<String, Path>> {
    val out = ArrayList<Pair<String, Path>>()
    for (r in SNAPSHOT_RAW) {
        val p = full.resolve("raw").resolve(r)
        require(Files.exists(p)) { "$p is missing; run the ingest stage first" }
        for (f in filesUnder(p)) out += "raw/" + full.resolve("raw").relativize(f).toString().replace('\\', '/') to f
    }
    for ((scope, dir) in listOf("full" to full, "quick" to quick)) for (s in SNAPSHOT_SAMPLED) {
        for (f in filesUnder(dir.resolve(s))) out += "$scope/" + dir.relativize(f).toString().replace('\\', '/') to f
    }
    return out.sortedBy { it.first }
}

/** Writes the entries to [out] with fixed times and order: the same files give the same bytes. */
fun writeSnapshot(entries: List<Pair<String, Path>>, out: Path) {
    Files.createDirectories(out.toAbsolutePath().parent)
    val tmp = out.resolveSibling(out.fileName.toString() + ".part")
    ZipOutputStream(Files.newOutputStream(tmp)).use { z ->
        z.setLevel(9)
        for ((name, f) in entries) {
            val e = ZipEntry(name)
            e.time = ENTRY_TIME
            z.putNextEntry(e)
            Files.copy(f, z)
            z.closeEntry()
        }
    }
    Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
}

/**
 * Unpacks [zip] into [data]: raw/ always (a raw file that is there with other content is an error),
 * the sampled outputs of [scope] when their directory is absent (a run that sampled again keeps its
 * own). Returns what was written, by group.
 */
fun unpackSnapshot(zip: Path, data: Path, scope: String, log: (String) -> Unit = ::println): Map<String, Int> {
    val counts = LinkedHashMap<String, Int>()
    val sampledPresent = SNAPSHOT_SAMPLED.associateWith { Files.exists(data.resolve(it)) }
    ZipInputStream(Files.newInputStream(zip)).use { z ->
        while (true) {
            val e = z.nextEntry ?: break
            if (e.isDirectory) continue
            val name = e.name
            require(!name.contains("..") && !name.startsWith("/")) { "snapshot entry $name leaves the data directory" }
            val (group, dest) = when {
                name.startsWith("raw/") -> "raw" to data.resolve(name)
                name.startsWith("$scope/") -> {
                    val rel = name.removePrefix("$scope/")
                    val top = rel.substringBefore('/')
                    if (sampledPresent[top] == true) continue
                    top to data.resolve(rel)
                }
                else -> continue
            }
            val bytes = z.readAllBytes()
            if (name == "raw/manifest.tsv" && Files.exists(dest)) {
                if (mergeRawManifest(dest, String(bytes, Charsets.UTF_8))) counts.merge(group, 1, Int::plus)
                continue
            }
            if (Files.exists(dest)) {
                if (!Files.readAllBytes(dest).contentEquals(bytes)) {
                    error("$dest differs from the snapshot's $name; move the raw directory away (or use BROOKLYN_LIVE=1 to keep fetched data)")
                }
                continue
            }
            Files.createDirectories(dest.parent)
            Files.write(dest, bytes)
            counts.merge(group, 1, Int::plus)
        }
    }
    for ((s, present) in sampledPresent) if (present) log("snapshot: $s/ is there; kept (remove it to take the snapshot's)")
    return counts
}

/** The `raw/manifest.tsv` sources whose files the snapshot holds. */
val SNAPSHOT_SOURCES = setOf("osm", "inspections", "complaints", "nta2020")

/**
 * Merges the snapshot's `raw/manifest.tsv` into [dest], row by source: a row of a source the
 * snapshot holds must equal the one there (an error otherwise: those files were fetched again);
 * rows of the sources ingest fetches itself (menus, Google Local) are kept as ingest wrote them.
 * Missing rows are added. Returns whether [dest] changed.
 */
fun mergeRawManifest(dest: Path, snapshot: String): Boolean {
    val have = Files.readAllLines(dest).filter { it.isNotBlank() }
    val bySource = have.associateBy { it.substringBefore('\t') }
    val added = ArrayList<String>()
    for (row in snapshot.lines().filter { it.isNotBlank() }) {
        val source = row.substringBefore('\t')
        val mine = bySource[source]
        if (mine == null) added += row
        else if (source in SNAPSHOT_SOURCES && mine != row) {
            error("$dest records $source as fetched at another time than the snapshot's raw/manifest.tsv; move the raw directory away (or use BROOKLYN_LIVE=1 to keep fetched data)")
        }
    }
    if (added.isEmpty()) return false
    Files.write(dest, have + added)
    return true
}

fun snapshotCache(): Path = exampleHome().resolve("build").resolve("snapshot")

fun snapshotCmd(args: List<String>): Int {
    when (args.firstOrNull()) {
        "pack" -> {
            val out = args.indexOf("--out").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: error("snapshot pack --out FILE")
            val full = args.indexOf("--full").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: Paths.get("build/data")
            val quick = args.indexOf("--quick-data").takeIf { it >= 0 }?.let { Paths.get(args[it + 1]) } ?: Paths.get("build/quick-data")
            val entries = snapshotEntries(full, quick)
            writeSnapshot(entries, out)
            println("snapshot: ${entries.size} files -> $out (${Files.size(out)} bytes, sha256 ${fileSha256(out)})")
            for (g in entries.groupBy { it.first.split('/').take(2).joinToString("/") }) println("  %-24s %d".format(g.key, g.value.size))
            return 0
        }
        "list" -> {
            val f = Paths.get(args.getOrNull(1) ?: error("snapshot list FILE"))
            ZipInputStream(Files.newInputStream(f)).use { z -> while (true) { val e = z.nextEntry ?: break; println(e.name) } }
            return 0
        }
        "fetch" -> {
            val pin = snapshotPin() ?: run { System.err.println("manifest.json has no data_snapshot"); return 1 }
            val scope = if (isQuick(args)) "quick" else "full"
            val zip = snapshotCache().resolve(pin.url.substringAfterLast('/'))
            if (!Files.exists(zip) || fileSha256(zip) != pin.sha256) {
                println("snapshot: downloading ${pin.url}")
                Http().download(pin.url, zip)
            }
            val sha = fileSha256(zip)
            if (sha != pin.sha256) {
                System.err.println("snapshot: $zip has SHA-256 $sha, the manifest pins ${pin.sha256}")
                return 1
            }
            val data = dataHome()
            val counts = unpackSnapshot(zip, data, scope)
            println("snapshot: ${pin.sha256} checked; unpacked into $data (scope $scope): " +
                counts.entries.joinToString { "${it.key} ${it.value}" }.ifEmpty { "everything was there" })
            return 0
        }
        else -> {
            System.err.println("usage: snapshot pack --out FILE | fetch [--quick] | list FILE")
            return 2
        }
    }
}
