package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

// Menus from the Internet Archive. For each OSM place with a menu URL or website, the start page
// is read from the Wayback snapshot nearest the cutoff on or before it (availability API; the CDX
// API when the closest snapshot is after the cutoff), and up to two of its menu links the same way.
//
// Every page used is pinned in pins/wayback-menus.tsv (committed): URL, snapshot timestamp,
// archive URL, SHA-256 of the archived bytes. With pins present, ingest fetches exactly those
// snapshots (web.archive.org/web/<ts>id_/<url>, the original bytes) and checks their hashes, so
// the menus table is the same on any later day. Pages are stored by hash in raw/wayback/pages/;
// the database build reads them back without the network.

const val WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
const val WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
const val WAYBACK_WEB = "https://web.archive.org/web"

/** One page's pin. status: ok, no_snapshot, excluded (the archive refuses the site), empty (a capture with no body), http_<code>, error. */
data class WaybackPin(
    val url: String,
    val status: String,
    val timestamp: String? = null,
    val archiveUrl: String? = null,
    val contentType: String? = null,
    val sha256: String? = null,
) {
    fun line(): String = listOf(url, status, timestamp, archiveUrl, contentType, sha256).joinToString("\t") { it ?: "-" }

    companion object {
        const val HEADER = "url\tstatus\ttimestamp\tarchive_url\tcontent_type\tsha256"
        fun parse(line: String): WaybackPin? {
            val p = line.split('\t')
            if (p.size < 6 || line == HEADER) return null
            fun f(i: Int) = p[i].takeIf { it != "-" && it.isNotEmpty() }
            return WaybackPin(p[0], p[1], f(2), f(3), f(4), f(5))
        }
    }
}

fun readPins(f: Path): Map<String, WaybackPin> =
    if (!Files.exists(f)) emptyMap() else Files.readAllLines(f).mapNotNull { WaybackPin.parse(it) }.associateBy { it.url }

/** Sorted by URL, so the file is the same whatever order the pages were fetched in. */
fun writePins(f: Path, pins: Collection<WaybackPin>) {
    Files.createDirectories(f.toAbsolutePath().parent)
    Files.write(f, listOf(WaybackPin.HEADER) + pins.sortedBy { it.url }.map { it.line() })
}

/** A place whose menu is looked for: OSM id and its start URL (menu URL, else website). */
data class MenuTarget(val osmId: String, val url: String)

/** Named OSM places with a menu URL or website, sorted by OSM id; URLs without a scheme get http://. */
fun menuTargets(places: List<OsmPlace>): List<MenuTarget> = places
    .filter { it.name != null }
    .mapNotNull { p -> (p.menuUrl ?: p.website)?.let { normalizeStartUrl(it) }?.let { MenuTarget(p.osmId, it) } }
    .distinctBy { it.osmId }
    .sortedBy { it.osmId }

fun normalizeStartUrl(raw: String): String? {
    val u = raw.trim().substringBefore(';').trim().replace(" ", "%20")
    if (u.isEmpty() || u.startsWith("mailto:")) return null
    val withScheme = if (Regex("^[a-zA-Z][a-zA-Z0-9+.-]*://").containsMatchIn(u)) u else "http://$u"
    return runCatching { java.net.URI(withScheme) }.getOrNull()?.takeIf { it.host != null && it.scheme in setOf("http", "https") }
        ?.toString()
}

const val EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

fun sha256Hex(b: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

/**
 * The latest snapshot timestamp allowed: 23:59:59 UTC of the day before the cutoff (the cutoff is
 * exclusive for every source: data dated before the cutoff day).
 */
fun cutoffTimestamp(cutoff: LocalDate): String = cutoff.minusDays(1).format(DateTimeFormatter.BASIC_ISO_DATE) + "235959"

/** "20230328203322" -> "2023-03-28T20:33:22Z". */
fun waybackIso(ts: String): String =
    if (ts.length != 14) ts else "${ts.substring(0, 4)}-${ts.substring(4, 6)}-${ts.substring(6, 8)}T${ts.substring(8, 10)}:${ts.substring(10, 12)}:${ts.substring(12, 14)}Z"

private val ARCHIVE_URL = Regex("^https?://web\\.archive\\.org/web/(\\d{14})(?:[a-z]{2}_)?/(.+)$")

/** The page's original URL and timestamp inside an archive URL. */
fun parseArchiveUrl(u: String): Pair<String, String>? = ARCHIVE_URL.find(u)?.let { it.groupValues[1] to it.groupValues[2] }

/**
 * Latest 200 snapshot of [url] on or before [cutoffTs]: the availability API's closest
 * snapshot when it is not after the cutoff, else the CDX API's last capture up to the cutoff.
 * Returns the timestamp, or null when there is none; throws [ExcludedSite] when the CDX API
 * answers 403 (sites the archive does not serve), IOException on other HTTP failures.
 */
fun findSnapshot(net: Net, url: String, cutoffTs: String): String? {
    val a = net.get("$WAYBACK_AVAILABLE?url=${Http.enc(url)}&timestamp=$cutoffTs", "application/json")
    if (a.status != 200) throw java.io.IOException("availability API HTTP ${a.status}")
    val closest = ((parseJson(a.text) as? JsonObject)?.get("archived_snapshots") as? JsonObject)?.get("closest") as? JsonObject
    val ts = closest?.s("timestamp")
    if (closest != null && closest.s("status") == "200" && ts != null && ts <= cutoffTs) return ts
    val c = net.get("$WAYBACK_CDX?url=${Http.enc(url)}&to=$cutoffTs&filter=statuscode:200&limit=-1&fl=timestamp", "text/plain")
    if (c.status == 403) throw ExcludedSite(url)
    if (c.status != 200) throw java.io.IOException("CDX API HTTP ${c.status}")
    return c.text.lines().map { it.trim() }.lastOrNull { it.matches(Regex("\\d{14}")) && it <= cutoffTs }
}

/** The archive serves nothing for this site (CDX answers 403): a lasting answer, pinned as `excluded`. */
class ExcludedSite(url: String) : Exception("the Internet Archive excludes $url")

class WaybackStats {
    val discovered = AtomicInteger()
    val reused = AtomicInteger()
    val downloaded = AtomicInteger()
    /** Pinned captures the archive no longer serves as pinned (moved or other bytes). */
    val shaMismatch = AtomicInteger()
    /** Pins replaced when pinning (capture no longer served as pinned, or after the cutoff). */
    val repinned = AtomicInteger()
    /** Pages with no pin when not pinning (status not_pinned). */
    val notPinned = AtomicInteger()
    /** Places whose pages were all served as pinned on the second pass. */
    val recovered = AtomicInteger()
    override fun toString() = "pins discovered ${discovered.get()}, pins reused ${reused.get()}, " +
        "pages downloaded ${downloaded.get()}, pinned captures not served as pinned ${shaMismatch.get()}, " +
        "pins replaced ${repinned.get()}, pages without a pin ${notPinned.get()}, places recovered on a second pass ${recovered.get()}"
}

/**
 * Archived pages for [targets], bytes kept in [pagesDir] by SHA-256; [used] holds the pin of every
 * page the menus of [targets] read.
 *
 * [discover] false (every normal ingest): only pinned captures are fetched; a page with no pin
 * is `not_pinned`, a pinned capture the archive no longer serves byte for byte is `moved` or
 * `sha_mismatch`; nothing is looked up. [discover] true (`ingest --pin-menus`, a maintainer
 * step): pages with no pin, pins after the cutoff and pinned captures no longer served as
 * pinned are looked up again and pinned anew.
 */
class WaybackFetcher(
    private val net: Net,
    private val cutoffTs: String,
    pins: Map<String, WaybackPin>,
    private val pagesDir: Path,
    private val discover: Boolean,
    private val log: (String) -> Unit,
    private val attempts: Int = 3,
    private val retryMs: Long = 15_000L,
) {
    val used = ConcurrentHashMap<String, WaybackPin>()
    private val known = ConcurrentHashMap(pins)
    val stats = WaybackStats()

    fun pageFile(sha: String): Path = pagesDir.resolve(sha.substring(0, 2)).resolve(sha)

    private fun afterCutoff(p: WaybackPin) = p.timestamp != null && p.timestamp > cutoffTs

    private fun pinFor(url: String): WaybackPin {
        val p = known[url]
        if (p != null) {
            val stale = p.status == "ok" && p.sha256 == EMPTY_SHA256 || afterCutoff(p)
            if (!stale || !discover) { stats.reused.incrementAndGet(); return p }
            stats.repinned.incrementAndGet()
        }
        if (!discover) { stats.notPinned.incrementAndGet(); return WaybackPin(url, "not_pinned") }
        return lookup(url)
    }

    private fun lookup(url: String): WaybackPin {
        val pin = try {
            val ts = findSnapshot(net, url, cutoffTs)
            if (ts == null) WaybackPin(url, "no_snapshot") else download(WaybackPin(url, "ok", ts, "$WAYBACK_WEB/${ts}id_/$url"))
        } catch (e: ExcludedSite) {
            WaybackPin(url, "excluded")
        } catch (e: Exception) {
            WaybackPin(url, "error")
        }
        stats.discovered.incrementAndGet()
        known[url] = pin
        return pin
    }

    /** Fetches [pin]'s archive URL; fills the final archive URL, content type and hash. */
    private fun download(pin: WaybackPin): WaybackPin {
        val r = net.get(pin.archiveUrl!!, "text/html,application/xhtml+xml,*/*;q=0.5")
        if (r.status !in 200..299) return pin.copy(status = "http_${r.status}")
        if (r.body.isEmpty()) return pin.copy(status = "empty") // the capture has no body: pinned as such
        val sha = sha256Hex(r.body)
        store(sha, r.body)
        stats.downloaded.incrementAndGet()
        // The archive may redirect a capture to a neighbouring one: pin the capture it served.
        val served = parseArchiveUrl(r.finalUrl)
        val finalArchive = if (served != null) r.finalUrl else pin.archiveUrl
        return pin.copy(timestamp = served?.first ?: pin.timestamp, archiveUrl = finalArchive, contentType = r.contentType, sha256 = sha)
    }

    private fun store(sha: String, body: ByteArray) {
        val f = pageFile(sha)
        if (Files.exists(f)) return
        Files.createDirectories(f.parent)
        val tmp = f.resolveSibling("$sha.${Thread.currentThread().id}.part")
        Files.write(tmp, body)
        Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE)
    }

    /**
     * Downloads a pinned capture into the store; null on success, else why not: `moved` (the
     * archive now redirects the pinned timestamp to another capture), `sha_mismatch`, or `error`.
     * Up to [attempts] tries, [retryMs] x attempt apart. The pin itself is never changed.
     */
    private fun refetch(pin: WaybackPin): String? {
        var why = "error"
        val pinnedTs = parseArchiveUrl(pin.archiveUrl!!)?.first
        for (attempt in 1..attempts) {
            val r = try { net.get(pin.archiveUrl, "text/html,application/xhtml+xml,*/*;q=0.5") } catch (e: Exception) { null }
            if (r != null && r.status in 200..299) {
                val servedTs = parseArchiveUrl(r.finalUrl)?.first
                if (sha256Hex(r.body) == pin.sha256) {
                    store(pin.sha256!!, r.body); stats.downloaded.incrementAndGet(); return null
                }
                why = if (servedTs != null && pinnedTs != null && servedTs != pinnedTs) "moved" else "sha_mismatch"
                log("  wayback: try $attempt: ${pin.archiveUrl} " + if (why == "moved") "now redirects to $servedTs" else "has bytes that differ from its pin")
            }
            if (attempt < attempts) Thread.sleep(retryMs * attempt)
        }
        if (why != "error") stats.shaMismatch.incrementAndGet()
        log("  wayback: ${pin.archiveUrl}: $why after $attempts tries; page not used")
        return why
    }

    /** The page for [url]: from the store when its pinned bytes are there, else fetched by its pin. */
    fun page(url: String): Page {
        var pin = pinFor(url)
        if (!discover && afterCutoff(pin)) { used[url] = pin; return Page("after_cutoff", url) }
        if (pin.status == "ok" && pin.sha256 != null && !Files.exists(pageFile(pin.sha256!!))) {
            val failure = refetch(pin)
            if (failure != null) {
                if (!discover) { used[url] = pin; return Page(failure, url) }
                log("  wayback: $url: pinned capture $failure; looked up again")
                stats.repinned.incrementAndGet()
                pin = lookup(url)
            }
        }
        used[url] = pin
        return pageFromStore(pin, pagesDir)
    }
}

/** The stored page of [pin] (no network); not_fetched when its bytes are not in [pagesDir]. */
fun pageFromStore(pin: WaybackPin?, pagesDir: Path, url: String = pin?.url ?: ""): Page {
    if (pin == null) return Page("not_fetched", url)
    if (pin.status != "ok") return Page(pin.status, pin.url)
    val sha = pin.sha256 ?: return Page("not_fetched", pin.url)
    val f = pagesDir.resolve(sha.substring(0, 2)).resolve(sha)
    if (!Files.exists(f)) return Page("not_fetched", pin.url)
    val original = pin.archiveUrl?.let { parseArchiveUrl(it) }
    val text = decodeArchived(Files.readAllBytes(f), pin.contentType)
        ?: return Page("undecodable", original?.second ?: pin.url, pin.contentType, snapshot = original?.first ?: pin.timestamp, archiveUrl = pin.archiveUrl)
    return Page(
        "ok", original?.second ?: pin.url, pin.contentType, text,
        snapshot = original?.first ?: pin.timestamp, archiveUrl = pin.archiveUrl,
    )
}

/**
 * Text of archived bytes. `id_` captures keep the site's own Content-Encoding, so gzip bodies
 * (by their magic number, possibly twice) are inflated. Null when a page served as HTML or text
 * (or with no type) is still binary after that (another encoding such as brotli): more than 1%
 * NUL or other control characters. Other types (PDF, images) are returned as they are; the menu
 * reader classifies them by type.
 */
fun decodeArchived(bytes: ByteArray, contentType: String? = null): String? {
    var b = bytes
    repeat(2) {
        if (b.size >= 2 && b[0] == 0x1f.toByte() && b[1] == 0x8b.toByte()) {
            b = try { java.util.zip.GZIPInputStream(b.inputStream()).use { it.readAllBytes() } } catch (e: java.io.IOException) { return null }
        }
    }
    val text = String(b, Charsets.UTF_8)
    val ct = contentType?.lowercase()
    if (ct != null && !ct.contains("html") && !ct.contains("text") && !ct.contains("xml")) return text
    val sample = text.take(20_000)
    val bad = sample.count { (it < ' ' && it != '\n' && it != '\r' && it != '\t') || it == '\uFFFD' }
    return if (sample.isNotEmpty() && bad > sample.length / 100) null else text
}

/**
 * Fetches the archived menu pages of [targets] with [threads] workers, writes the pins of the
 * pages used to [rawPins] (and returns them), and logs progress.
 */
fun fetchWaybackMenus(
    targets: List<MenuTarget>,
    net: Net,
    cutoffTs: String,
    pins: Map<String, WaybackPin>,
    pagesDir: Path,
    discover: Boolean,
    threads: Int,
    log: (String) -> Unit,
    secondPassDelayMs: Long = 300_000L,
    retryMs: Long = 15_000L,
): Pair<Map<String, WaybackPin>, WaybackStats> {
    val f = WaybackFetcher(net, cutoffTs, pins, pagesDir, discover, log, retryMs = retryMs)
    val failed = ConcurrentHashMap.newKeySet<MenuTarget>()
    fun visit(t: MenuTarget) {
        try {
            menuFromPages(t.url) { u -> f.page(u).also { if (it.status in RETRYABLE) failed += t } }
        } catch (e: Exception) {
            log("  wayback: ${t.url}: ${e.message}")
        }
    }
    val pool = Executors.newFixedThreadPool(threads)
    val done = AtomicInteger()
    val t0 = System.nanoTime()
    for (t in targets) pool.execute {
        visit(t)
        val n = done.incrementAndGet()
        if (n % 100 == 0 || n == targets.size) {
            log("  wayback: $n / ${targets.size} places, %.0f s; ${f.stats}".format((System.nanoTime() - t0) / 1e9))
        }
    }
    pool.shutdown()
    pool.awaitTermination(7, TimeUnit.DAYS)
    // The archive's redirects of pinned captures come and go: places with a page that was not
    // served as pinned are read again once, after a pause.
    if (failed.isNotEmpty() && !discover) {
        val again = failed.sortedBy { it.osmId }
        failed.clear()
        log("  wayback: ${again.size} places had a page not served as pinned; trying them again in ${secondPassDelayMs / 1000} s")
        Thread.sleep(secondPassDelayMs)
        again.forEach { visit(it) }
        val still = failed.size
        f.stats.recovered.addAndGet(again.size - still)
        log("  wayback: second pass: ${again.size - still} of ${again.size} places now read as pinned")
    }
    return f.used.toMap() to f.stats
}

/** Page statuses worth a second try: the archive did not serve the pinned capture. */
val RETRYABLE = setOf("moved", "sha_mismatch", "error")
