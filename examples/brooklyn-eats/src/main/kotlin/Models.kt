package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.io.IOException
import java.io.InputStream
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.time.Duration
import java.util.concurrent.Executors
import java.util.concurrent.Future

// The models the pipeline uses, pinned in manifest.json under stages.models.inputs by Hugging Face
// commit hash, with the size and SHA-256 of every file the pipeline reads (the LFS oid for weight
// files). Files live in the standard Hugging Face cache layout:
//   <cache>/models--<org>--<name>/blobs/<LFS sha256 | git blob id>
//   <cache>/models--<org>--<name>/snapshots/<commit>/<path> -> ../../blobs/<id>
// where <cache> is HF_HUB_CACHE, else $HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub, else
// ~/.cache/huggingface/hub. A blob already in the cache is never downloaded again.
//
//   brooklyn-eats models [--all] [--repair] [--rehash] [--cache DIR] [key...]   fetch what is missing, check every file
//   brooklyn-eats models check [--all] [--rehash] [--cache DIR] [key...]        check only, no network
//   brooklyn-eats models path <key>                                             the snapshot directory of a model
//   brooklyn-eats models pin <key> <repo> <revision> --role R [--include GLOBS] [--optional]   maintainer step

const val HF_ENDPOINT_DEFAULT = "https://huggingface.co"
const val MODEL_HASHES = "model-sha256.tsv"
val DEFAULT_MODEL_INCLUDE = listOf("*.json", "*.jinja", "*.safetensors", "*LICENSE*", "README.md")

class ModelFile(val path: String, val size: Long, val sha256: String, val gitBlob: String? = null) {
    /** Name of the file under blobs/: the LFS oid for LFS files, the git blob id for the others. */
    val blob: String get() = gitBlob ?: sha256
}

class ModelPin(
    val key: String,
    val repo: String,
    val revision: String,
    val role: String,
    val optional: Boolean,
    val files: List<ModelFile>,
) {
    val bytes: Long get() = files.sumOf { it.size }
}

/** The pins in stages.models.inputs of [m]. */
fun readModelPins(m: JsonObject?): List<ModelPin> {
    val inputs = m?.path("stages", "models", "inputs") as? JsonObject ?: return emptyList()
    return inputs.fields.map { (key, v) ->
        val o = v as JsonObject
        val files = (o["files"] as JsonArray).elements.map { f ->
            f as JsonObject
            ModelFile(f.optString("path")!!, (f["size"] as JsonNumber).raw.toLong(), f.optString("sha256")!!, f.optString("git_blob"))
        }
        ModelPin(key, o.optString("repo")!!, o.optString("revision")!!, o.optString("role") ?: "", (o["optional"] as? JsonBool)?.value == true, files)
    }
}

fun isCommitHash(s: String) = Regex("[0-9a-f]{40}").matches(s)

/** The Hugging Face hub cache directory, as huggingface_hub resolves it. */
fun hubCache(env: Map<String, String> = System.getenv()): Path {
    env["HF_HUB_CACHE"]?.takeIf { it.isNotBlank() }?.let { return Paths.get(it) }
    env["HF_HOME"]?.takeIf { it.isNotBlank() }?.let { return Paths.get(it).resolve("hub") }
    val xdg = env["XDG_CACHE_HOME"]?.takeIf { it.isNotBlank() }?.let { Paths.get(it) } ?: Paths.get(System.getProperty("user.home"), ".cache")
    return xdg.resolve("huggingface").resolve("hub")
}

fun repoCacheDir(cache: Path, repo: String): Path = cache.resolve("models--" + repo.replace("/", "--"))

fun snapshotDir(cache: Path, pin: ModelPin): Path = repoCacheDir(cache, pin.repo).resolve("snapshots").resolve(pin.revision)

/** The snapshot directory of the pinned model [key] (for the stages that load it). */
fun modelDir(key: String, cache: Path = hubCache(), manifest: JsonObject? = readManifestJson()): Path {
    val pin = readModelPins(manifest).firstOrNull { it.key == key } ?: error("manifest.json pins no model '$key'")
    return snapshotDir(cache, pin)
}

/**
 * SHA-256 of files, remembered in a TSV (real path, size, modification time in ns) so that
 * later checks of an unchanged file do not read it again. [rehash] ignores what is remembered.
 */
class HashCache(private val file: Path?, private val rehash: Boolean = false) {
    private val known = HashMap<String, String>()

    init {
        if (file != null && Files.exists(file)) for (l in Files.readAllLines(file)) {
            val i = l.lastIndexOf('\t'); if (i > 0) known[l.substring(0, i)] = l.substring(i + 1)
        }
    }

    private fun stamp(f: Path): String {
        val real = f.toRealPath()
        return "$real\t${Files.size(real)}\t${Files.getLastModifiedTime(real).to(java.util.concurrent.TimeUnit.NANOSECONDS)}"
    }

    fun sha256(f: Path): String {
        val s = stamp(f)
        if (!rehash) synchronized(known) { known[s] }?.let { return it }
        val h = fileSha256(f)
        synchronized(known) { known[s] = h }
        return h
    }

    fun save() {
        if (file == null) return
        Files.createDirectories(file.toAbsolutePath().parent)
        val tmp = file.resolveSibling(file.fileName.toString() + ".tmp")
        synchronized(known) { Files.write(tmp, known.entries.sortedBy { it.key }.map { "${it.key}\t${it.value}" }) }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    }
}

/** One file's state: ok, missing, or differing in size or SHA-256. */
class FileCheck(val pin: ModelPin, val file: ModelFile, val state: String, val detail: String = "") {
    val ok: Boolean get() = state == "ok"
}

/** Checks every pinned file of [pins] under [cache] (several files hashed at once). */
fun checkModels(cache: Path, pins: List<ModelPin>, hashes: HashCache, threads: Int = 4): List<FileCheck> {
    val pool = Executors.newFixedThreadPool(threads)
    try {
        val futures: List<Future<FileCheck>> = pins.flatMap { pin ->
            pin.files.map { f -> pool.submit<FileCheck> { checkFile(snapshotDir(cache, pin).resolve(f.path), pin, f, hashes) } }
        }
        return futures.map { it.get() }
    } finally {
        pool.shutdown()
    }
}

fun checkFile(p: Path, pin: ModelPin, f: ModelFile, hashes: HashCache): FileCheck {
    if (!Files.exists(p)) return FileCheck(pin, f, "missing")
    val size = Files.size(p)
    if (size != f.size) return FileCheck(pin, f, "size", "size ${f.size} -> $size")
    val sha = hashes.sha256(p)
    return if (sha == f.sha256) FileCheck(pin, f, "ok") else FileCheck(pin, f, "sha256", "sha256 ${f.sha256} -> $sha")
}

/** One line per model for verify and `models check`: ok, MISSING or MISMATCH, and the files concerned. */
fun reportModel(pin: ModelPin, checks: List<FileCheck>): Pair<Boolean, String> {
    val label = "models.inputs.${pin.key}"
    val id = "${pin.repo}@${pin.revision.take(8)}, ${pin.files.size} files, ${gib(pin.bytes)}"
    val missing = checks.filter { it.state == "missing" }
    val wrong = checks.filter { !it.ok && it.state != "missing" }
    return when {
        missing.isEmpty() && wrong.isEmpty() -> true to "ok        $label ($id)"
        wrong.isEmpty() -> false to "MISSING   $label ($id): ${missing.size} files not in the cache (run the models stage): " +
            missing.take(4).joinToString { it.file.path } + if (missing.size > 4) ", ..." else ""
        else -> false to "MISMATCH  $label ($id): " + wrong.joinToString("; ") { "${it.file.path} ${it.detail}" } +
            if (missing.isNotEmpty()) "; ${missing.size} files missing" else ""
    }
}

fun gib(b: Long) = "%.1f GiB".format(b / 1073741824.0)

/**
 * Plain HTTPS access to the Hub: resolve URLs at a commit, followed by hand so the token
 * (HF_TOKEN, only when set) is sent to the Hub's own host and never to the storage host it
 * redirects to; downloads resume from `<blob>.incomplete` with a Range request.
 */
class HubClient(
    val endpoint: String = System.getenv("HF_ENDPOINT")?.takeIf { it.isNotBlank() }?.trimEnd('/') ?: HF_ENDPOINT_DEFAULT,
    private val token: String? = System.getenv("HF_TOKEN")?.takeIf { it.isNotBlank() },
    private val maxAttempts: Int = 8,
    private val firstBackoffMs: Long = 5_000L,
    private val log: (String) -> Unit = { System.err.println(it) },
) {
    private val client: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .followRedirects(HttpClient.Redirect.NEVER)
        .build()
    private val hubHost = URI(endpoint).host

    fun resolveUrl(repo: String, revision: String, path: String): String =
        "$endpoint/$repo/resolve/$revision/" + path.split('/').joinToString("/") { URLEncoder.encode(it, StandardCharsets.UTF_8).replace("+", "%20") }

    private fun open(url0: String, offset: Long, timeout: Duration): HttpResponse<InputStream> {
        var url = url0
        repeat(10) {
            val b = HttpRequest.newBuilder(URI(url)).GET().timeout(timeout).header("User-Agent", USER_AGENT)
            if (token != null && URI(url).host == hubHost) b.header("Authorization", "Bearer $token")
            if (offset > 0) b.header("Range", "bytes=$offset-")
            val r = client.send(b.build(), HttpResponse.BodyHandlers.ofInputStream())
            if (r.statusCode() !in 300..399) return r
            val loc = r.headers().firstValue("Location").orElseThrow { IOException("$url: HTTP ${r.statusCode()} without Location") }
            r.body().close()
            url = URI(url).resolve(loc).toString()
        }
        throw IOException("$url0: more than 10 redirects")
    }

    /** GET [url] whole (the API and small files). */
    fun getBytes(url: String): ByteArray {
        val r = withRetries(url) { open(url, 0, Duration.ofMinutes(2)) }
        return r.body().use { it.readAllBytes() }
    }

    fun getText(url: String): String = getBytes(url).toString(Charsets.UTF_8)

    /** Sends with backoff on IO errors, 429 and 5xx; fails at once on other non-2xx answers. */
    private fun withRetries(url: String, send: () -> HttpResponse<InputStream>): HttpResponse<InputStream> {
        var backoff = firstBackoffMs
        var last = ""
        for (attempt in 1..maxAttempts) {
            try {
                val r = send()
                val code = r.statusCode()
                if (code in 200..299 || code == 416) return r
                r.body().close()
                last = "HTTP $code"
                if (code == 401 || code == 403) throw IOException("$url: $last (a gated or private repository needs HF_TOKEN)")
                if (code != 429 && code < 500) throw IOException("$url: $last")
                r.headers().firstValue("Retry-After").orElse(null)?.toLongOrNull()?.let { backoff = maxOf(backoff, minOf(it, 120) * 1000) }
            } catch (e: IOException) {
                if (e.message?.startsWith(url) == true) throw e
                last = "${e::class.simpleName}: ${e.message}"
            }
            if (attempt < maxAttempts) {
                log("  $last, retry $attempt/${maxAttempts - 1} in ${backoff / 1000}s")
                Thread.sleep(backoff)
                backoff = minOf(backoff * 2, 120_000L)
            }
        }
        throw IOException("$url: $last")
    }

    /**
     * Downloads [f] of [pin] into [part] (resuming from what it holds) and returns its SHA-256.
     * The caller checks it and moves the file into place.
     */
    fun download(pin: ModelPin, f: ModelFile, part: Path): String {
        val url = resolveUrl(pin.repo, pin.revision, f.path)
        Files.createDirectories(part.toAbsolutePath().parent)
        var backoff = firstBackoffMs
        for (attempt in 1..maxAttempts) {
            val have = if (Files.exists(part)) Files.size(part) else 0L
            if (have > f.size) Files.delete(part)
            val offset = if (have > f.size) 0L else have
            val md = MessageDigest.getInstance("SHA-256")
            if (offset > 0) {
                Files.newInputStream(part).use { ins -> val buf = ByteArray(1 shl 20); while (true) { val n = ins.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
                log("  ${f.path}: resuming at ${gib(offset)} of ${gib(f.size)}")
            }
            if (offset == f.size) return hex(md.digest())
            val r = withRetries(url) { open(url, offset, Duration.ofHours(6)) }
            if (r.statusCode() == 416) { r.body().close(); Files.deleteIfExists(part); continue }
            val append = offset > 0 && r.statusCode() == 206
            if (offset > 0 && !append) { md.reset(); log("  ${f.path}: the server ignored the range; starting again") }
            var got = if (append) offset else 0L
            val opts = if (append) arrayOf(StandardOpenOption.CREATE, StandardOpenOption.APPEND) else arrayOf(StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)
            var lastLog = System.nanoTime()
            val t0 = System.nanoTime()
            val start = got
            try {
                r.body().use { ins ->
                    Files.newOutputStream(part, *opts).use { out ->
                        val buf = ByteArray(1 shl 20)
                        while (true) {
                            val n = ins.read(buf); if (n < 0) break
                            out.write(buf, 0, n); md.update(buf, 0, n); got += n
                            if (System.nanoTime() - lastLog > 30_000_000_000L) {
                                lastLog = System.nanoTime()
                                val mbs = (got - start) / 1e6 / ((lastLog - t0) / 1e9)
                                log("  ${f.path}: ${gib(got)} of ${gib(f.size)} (%.0f MB/s)".format(mbs))
                            }
                        }
                    }
                }
            } catch (e: IOException) {
                log("  ${f.path}: ${e::class.simpleName} at ${gib(got)}; resuming in ${backoff / 1000}s")
                if (attempt < maxAttempts) { Thread.sleep(backoff); backoff = minOf(backoff * 2, 120_000L); continue }
                throw IOException("$url: ${e.message}", e)
            }
            if (got == f.size) return hex(md.digest())
            log("  ${f.path}: connection closed at ${gib(got)} of ${gib(f.size)}; resuming")
        }
        throw IOException("$url: incomplete after $maxAttempts attempts")
    }

    /** The files of [repo] at [revision] (the commit the API resolves it to), with sizes and LFS oids. */
    fun listFiles(repo: String, revision: String): Pair<String, List<Triple<String, Long, Pair<String?, String>>>> {
        val o = parseJson(getText("$endpoint/api/models/$repo/revision/$revision?blobs=true")) as JsonObject
        val sha = o.optString("sha") ?: error("$repo@$revision: no commit in the API answer")
        val files = (o["siblings"] as JsonArray).elements.map { s ->
            s as JsonObject
            val lfs = s.optObject("lfs")
            Triple(s.optString("rfilename")!!, (s["size"] as JsonNumber).raw.toLong(), lfs?.optString("sha256") to s.optString("blobId")!!)
        }
        return sha to files
    }
}

/** `*` matches within one path segment; a pattern without `/` matches top-level files only. */
fun globMatch(pattern: String, path: String): Boolean =
    Regex(pattern.split("*").joinToString("[^/]*") { Regex.escape(it) }).matches(path)

/** Links snapshots/<rev>/<path> to ../../blobs/<blob> (relative, as huggingface_hub does). */
fun linkSnapshot(cache: Path, pin: ModelPin, f: ModelFile) {
    val link = snapshotDir(cache, pin).resolve(f.path)
    Files.createDirectories(link.parent)
    val blob = repoCacheDir(cache, pin.repo).resolve("blobs").resolve(f.blob)
    if (Files.exists(link, LinkOption.NOFOLLOW_LINKS)) Files.delete(link)
    Files.createSymbolicLink(link, link.parent.relativize(blob))
}

/**
 * Makes every file of [pins] present and correct under [cache]: a snapshot link that checks is
 * kept; a blob already in the cache with the pinned SHA-256 is linked; anything else is
 * downloaded at the pinned commit. A file in the cache with the wrong content is an error
 * unless [repair], which downloads it again. Returns the number of files that failed.
 */
fun fetchModels(cache: Path, pins: List<ModelPin>, hub: HubClient, hashes: HashCache, repair: Boolean, log: (String) -> Unit): Int {
    var failed = 0
    for (pin in pins) {
        val blobs = repoCacheDir(cache, pin.repo).resolve("blobs")
        var fetched = 0L
        var linked = 0
        val t0 = System.nanoTime()
        log("models: ${pin.key} ${pin.repo}@${pin.revision} (${pin.files.size} files, ${gib(pin.bytes)})")
        val checks = checkModels(cache, listOf(pin), hashes)
        for (c in checks) {
            val f = c.file
            if (c.ok) continue
            val blob = blobs.resolve(f.blob)
            if (c.state != "missing" && !repair) {
                log("  ${f.path}: ${c.detail}; the cached copy is not the pinned file (rerun with --repair to download it again)")
                failed++; continue
            }
            // The snapshot link is missing: the blob may be there under another snapshot.
            if (c.state == "missing" && Files.exists(blob)) {
                val bc = checkFile(blob, pin, f, hashes)
                if (bc.ok) { linkSnapshot(cache, pin, f); linked++; continue }
                if (!repair) { log("  ${f.path}: cached blob ${f.blob}: ${bc.detail} (rerun with --repair)"); failed++; continue }
            }
            Files.deleteIfExists(blob)
            val part = blobs.resolve(f.blob + ".incomplete")
            try {
                val sha = hub.download(pin, f, part)
                if (Files.size(part) != f.size || sha != f.sha256) {
                    log("  ${f.path}: downloaded ${Files.size(part)} bytes, sha256 $sha; pinned ${f.size}, ${f.sha256}")
                    Files.delete(part); failed++; continue
                }
                Files.move(part, blob, StandardCopyOption.ATOMIC_MOVE)
                hashes.sha256(blob)
                linkSnapshot(cache, pin, f)
                fetched += f.size
            } catch (e: IOException) {
                log("  ${f.path}: ${e.message}")
                failed++
            }
        }
        val s = (System.nanoTime() - t0) / 1e9
        log("models: ${pin.key}: ${checks.count { it.ok }} files already in place, $linked linked from cached blobs, ${gib(fetched)} downloaded, %.0f s".format(s))
    }
    hashes.save()
    return failed
}

/** `models pin`: lists the repository at a revision and writes the pin into manifest.json. */
fun pinModel(home: Path, key: String, repo: String, revision: String, role: String, include: List<String>, optional: Boolean, hub: HubClient, log: (String) -> Unit): ModelPin {
    val (sha, listing) = hub.listFiles(repo, revision)
    val files = listing.filter { (p, _, _) -> include.any { globMatch(it, p) } }.sortedBy { it.first }.map { (p, size, ids) ->
        val (lfs, blobId) = ids
        if (lfs != null) ModelFile(p, size, lfs) else {
            // Not in LFS: the API gives the git blob id; the SHA-256 is computed from the file at the commit.
            val bytes = hub.getBytes(hub.resolveUrl(repo, sha, p))
            check(bytes.size.toLong() == size) { "$repo@$sha/$p: ${bytes.size} bytes, the API says $size" }
            check(gitBlobId(bytes) == blobId) { "$repo@$sha/$p: git blob ${gitBlobId(bytes)}, the API says $blobId" }
            ModelFile(p, size, sha256Hex(bytes), blobId)
        }
    }
    check(files.isNotEmpty()) { "$repo@$sha: no file matches ${include.joinToString()}" }
    val pin = ModelPin(key, repo, sha, role, optional, files)
    val mf = home.resolve("manifest.json")
    @Suppress("UNCHECKED_CAST")
    val root = (readManifestJson(mf)?.toKotlin() as Map<String, Any?>?)?.let { deepMutableMap(it) } ?: linkedMapOf()
    @Suppress("UNCHECKED_CAST")
    fun sub(m: MutableMap<String, Any?>, k: String) = m.getOrPut(k) { linkedMapOf<String, Any?>() } as MutableMap<String, Any?>
    sub(sub(sub(root, "stages"), "models"), "inputs")[key] = modelPinJson(pin, include)
    Files.writeString(mf, prettyJson(root) + "\n")
    log("pinned $key: $repo@$sha, ${files.size} files, ${gib(pin.bytes)}")
    return pin
}

fun modelPinJson(pin: ModelPin, include: List<String>): Map<String, Any?> = linkedMapOf(
    "repo" to pin.repo,
    "revision" to pin.revision,
    "role" to pin.role,
    "optional" to pin.optional,
    "include" to include,
    "bytes" to pin.bytes,
    "files" to pin.files.map { f ->
        linkedMapOf<String, Any?>("path" to f.path, "size" to f.size, "sha256" to f.sha256).also { m -> f.gitBlob?.let { m["git_blob"] = it } }
    },
)

/** Git's blob id: SHA-1 of "blob <size>\0" and the content. */
fun gitBlobId(b: ByteArray): String {
    val md = MessageDigest.getInstance("SHA-1")
    md.update("blob ${b.size}\u0000".toByteArray(Charsets.UTF_8))
    return hex(md.digest(b))
}

/** The pins [names] select: every required model, the optional ones too with [all]. */
fun selectPins(pins: List<ModelPin>, names: List<String>, all: Boolean): List<ModelPin> {
    if (names.isNotEmpty()) {
        val unknown = names - pins.map { it.key }.toSet()
        require(unknown.isEmpty()) { "unknown model ${unknown.joinToString()}; pinned: ${pins.joinToString { it.key }}" }
        return pins.filter { it.key in names }
    }
    return pins.filter { all || !it.optional }
}

fun modelHashCache(rehash: Boolean): HashCache = HashCache(dataHome().resolve(MODEL_HASHES), rehash)

private fun optValue(args: List<String>, flag: String): String? = args.indexOf(flag).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) ?: error("$flag needs a value") }

/** Arguments that are not flags or flag values. */
private fun positional(args: List<String>, valued: Set<String>): List<String> {
    val out = ArrayList<String>()
    var i = 0
    while (i < args.size) {
        val a = args[i]
        if (a in valued) i += 2 else { if (!a.startsWith("--")) out += a; i++ }
    }
    return out
}

fun models(args: List<String>): Int {
    val home = optValue(args, "--home")?.let { Paths.get(it).toAbsolutePath().normalize() } ?: exampleHome()
    val cache = optValue(args, "--cache")?.let { Paths.get(it) } ?: hubCache()
    val valued = setOf("--home", "--cache", "--role", "--include")
    val pos = positional(args, valued)
    val log: (String) -> Unit = { println(it) }
    val manifest = readManifestJson(home.resolve("manifest.json"))
    val pins = readModelPins(manifest)
    when (pos.firstOrNull()) {
        "path" -> {
            val key = pos.getOrNull(1) ?: run { System.err.println("models path <key>"); return 2 }
            println(modelDir(key, cache, manifest)); return 0
        }
        "pin" -> {
            if (pos.size < 4) { System.err.println("models pin <key> <repo> <revision> --role R [--include GLOB,...] [--optional]"); return 2 }
            val include = optValue(args, "--include")?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() } ?: DEFAULT_MODEL_INCLUDE
            pinModel(home, pos[1], pos[2], pos[3], optValue(args, "--role") ?: "", include, "--optional" in args, HubClient(), log)
            return 0
        }
    }
    val checkOnly = pos.firstOrNull() == "check"
    val names = if (checkOnly) pos.drop(1) else pos
    if (pins.isEmpty()) { println("models: manifest.json pins no model"); return 1 }
    val bad = pins.filterNot { isCommitHash(it.revision) }
    if (bad.isNotEmpty()) { println("models: not a commit hash: " + bad.joinToString { "${it.key} ${it.revision}" }); return 1 }
    val sel = selectPins(pins, names, "--all" in args)
    println("models: cache $cache")
    val hashes = modelHashCache("--rehash" in args)
    if (!checkOnly) {
        val failed = fetchModels(cache, sel, HubClient(), hashes, "--repair" in args, log)
        if (failed > 0) { println("models: $failed files failed"); return 1 }
    }
    val checks = checkModels(cache, sel, hashes)
    hashes.save()
    var ok = true
    for (pin in sel) {
        val (good, line) = reportModel(pin, checks.filter { it.pin === pin })
        ok = ok && good
        println(line)
    }
    for (pin in pins - sel.toSet()) println("skip      models.inputs.${pin.key} (" + (if (names.isEmpty()) "optional; models --all or models ${pin.key}" else "not selected") + ")")
    return if (ok) 0 else 1
}
