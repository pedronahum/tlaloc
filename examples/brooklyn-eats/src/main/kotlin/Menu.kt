package brooklyn

import java.net.URI

// Menu text from a restaurant's web page: HTML to text, the page's "menu" link when the page
// itself has no prices, and the lines that carry prices (with the dish name before them). The
// same extraction reads Internet Archive snapshots at ingest (Wayback.kt) and live sites.

private val DROP_BLOCKS = Regex("(?is)<(script|style|noscript|svg|head|template|iframe)\\b.*?</\\1\\s*>")
private val COMMENTS = Regex("(?s)<!--.*?-->")
private val BREAKS = Regex("(?i)<(br|/p|/div|/li|/tr|/h[1-6]|/section|/article|/dt|/dd|/td|/th|p|li|tr|h[1-6])\\b[^>]*>")
private val TAGS = Regex("(?s)<[^>]+>")
private val ENTITY = Regex("&(#\\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
private val NAMED = mapOf(
    "amp" to "&", "lt" to "<", "gt" to ">", "quot" to "\"", "apos" to "'", "nbsp" to " ", "ndash" to "–",
    "mdash" to "—", "rsquo" to "'", "lsquo" to "'", "rdquo" to "\"", "ldquo" to "\"", "hellip" to "…",
    "eacute" to "é", "egrave" to "è", "ntilde" to "ñ", "uuml" to "ü", "ouml" to "ö", "auml" to "ä", "middot" to "·",
)

fun decodeEntities(s: String): String = ENTITY.replace(s) { m ->
    val e = m.groupValues[1]
    when {
        e.startsWith("#x") || e.startsWith("#X") -> e.substring(2).toIntOrNull(16)?.let { String(Character.toChars(it)) } ?: m.value
        e.startsWith("#") -> e.substring(1).toIntOrNull()?.takeIf { it in 1..0x10FFFF }?.let { String(Character.toChars(it)) } ?: m.value
        else -> NAMED[e.lowercase()] ?: m.value
    }
}

/** Visible text of an HTML page, one block per line, blank lines removed. */
fun htmlToText(html: String): String {
    var s = COMMENTS.replace(html, " ")
    s = DROP_BLOCKS.replace(s, " ")
    s = BREAKS.replace(s, "\n")
    s = TAGS.replace(s, " ")
    s = decodeEntities(s)
    return s.lines().map { it.replace(Regex("[\\s\\u00a0]+"), " ").trim() }.filter { it.isNotEmpty() }.joinToString("\n")
}

private val HREF = Regex("(?is)<a\\b[^>]*?href\\s*=\\s*[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>")

/** Links on the page whose URL or anchor text says "menu", resolved against [base], same site first. */
fun menuLinks(html: String, base: String): List<String> {
    val baseUri = URI(base)
    val host = baseUri.host?.removePrefix("www.")
    return HREF.findAll(html).mapNotNull { m ->
        val href = decodeEntities(m.groupValues[1].trim())
        val text = htmlToText(m.groupValues[2]).lowercase()
        if (!(href.lowercase().contains("menu") || Regex("\\bmenus?\\b").containsMatchIn(text))) return@mapNotNull null
        if (href.startsWith("mailto:") || href.startsWith("tel:") || href.startsWith("javascript:")) return@mapNotNull null
        runCatching { baseUri.resolve(href.replace(" ", "%20")).toString() }.getOrNull()?.takeIf { it.startsWith("http") }
    }.distinct().sortedBy { u -> if (URI(u).host?.removePrefix("www.") == host) 0 else 1 }.toList()
}

private val PRICE = Regex("(?<![\\d.])\\$?\\s?\\d{1,3}(?:\\.\\d{2})(?!\\d)|\\$\\s?\\d{1,3}(?![\\d,])")

/** Shop listings (Shopify and similar), not menus. */
private val SHOP_PRICE = Regex("(?i)\\b(regular|sale|unit) price\\b|\\bUSD\\b")
/** Longer lines with a price are prose (minimum orders, promotions, legal text), not menu items. */
const val MAX_PRICED_LINE = 100

/** Lines with a price, each preceded by the line before it when that line is a short name without one. */
fun menuLines(text: String): List<String> {
    val lines = text.lines()
    val out = LinkedHashSet<String>()
    for ((i, l) in lines.withIndex()) {
        if (l.length > MAX_PRICED_LINE || !PRICE.containsMatchIn(l) || SHOP_PRICE.containsMatchIn(l)) continue
        val prev = lines.getOrNull(i - 1)
        if (prev != null && SHOP_PRICE.containsMatchIn(prev)) continue
        if (prev != null && prev.length <= 80 && !PRICE.containsMatchIn(prev)) out += prev
        out += l
    }
    return out.toList()
}

/** One page as a menu source sees it: the live site, or an Internet Archive snapshot of it. */
class Page(
    /** ok, or why there is no page: no_snapshot, robots, http_<code>, error, sha_mismatch, not_fetched. */
    val status: String,
    /** The page's own URL (after redirects); links resolve against it. */
    val url: String,
    val contentType: String? = null,
    val html: String? = null,
    /** Wayback timestamp (yyyyMMddHHmmss) when the page is an archived snapshot. */
    val snapshot: String? = null,
    val archiveUrl: String? = null,
)

/** The menu read for one place, and the page it came from. */
class MenuText(
    val status: String,
    val text: String?,
    val items: Int,
    val url: String,
    val snapshot: String? = null,
    val archiveUrl: String? = null,
)

/** Minimum priced lines for a page to count as a menu; `items` counts priced lines. */
const val MIN_MENU_ITEMS = 5
const val MAX_MENU_CHARS = 6_000
/** Menu links followed from a start page with too few prices. */
const val MAX_MENU_LINKS = 2

/**
 * The menu reachable from [start]: its priced lines, or, when it has fewer than
 * [MIN_MENU_ITEMS], the first of up to [MAX_MENU_LINKS] menu links that has them. [load] supplies
 * pages (live, archived, or read back from the raw store); the result depends only on what it
 * returns. Status: ok, no_prices, pdf, not_html, or the start page's own status.
 */
fun menuFromPages(start: String, load: (String) -> Page): MenuText {
    fun one(u: String): Pair<MenuText, String?> {
        val p = load(u)
        if (p.status != "ok") return MenuText(p.status, null, 0, p.url, p.snapshot, p.archiveUrl) to null
        val ct = p.contentType?.lowercase() ?: ""
        if (ct.contains("pdf") || p.url.lowercase().substringBefore('?').endsWith(".pdf")) {
            return MenuText("pdf", null, 0, p.url, p.snapshot, p.archiveUrl) to null
        }
        if (ct.isNotEmpty() && !ct.contains("html") && !ct.contains("text/plain")) {
            return MenuText("not_html", null, 0, p.url, p.snapshot, p.archiveUrl) to null
        }
        val html = p.html ?: ""
        val lines = menuLines(htmlToText(html))
        val priced = lines.count { PRICE.containsMatchIn(it) }
        val text = lines.joinToString("\n").let { if (it.length > MAX_MENU_CHARS) it.substring(0, MAX_MENU_CHARS) else it }
        val m = if (priced >= MIN_MENU_ITEMS) MenuText("ok", text, priced, p.url, p.snapshot, p.archiveUrl)
        else MenuText("no_prices", null, priced, p.url, p.snapshot, p.archiveUrl)
        return m to html
    }
    val (first, html) = one(start)
    if (first.status == "ok" || html == null) return first
    for (link in menuLinks(html, first.url).filter { it != start && it != first.url }.take(MAX_MENU_LINKS)) {
        val (m, _) = one(link)
        if (m.status == "ok" || m.status == "pdf") return m
    }
    return first
}

/** A live page through [net], after the site's robots.txt allows it. */
fun livePage(net: Net, robots: Robots, url: String): Page {
    if (!robots.allowed(net, url)) return Page("robots", url)
    val r = try { net.get(url, "text/html,application/xhtml+xml") } catch (e: Exception) { return Page("error", url) }
    if (r.status !in 200..299) return Page("http_${r.status}", r.finalUrl)
    return Page("ok", r.finalUrl, r.contentType, r.text)
}

/** The menu on the live site. */
fun fetchLiveMenu(net: Net, robots: Robots, url: String): MenuText = menuFromPages(url) { livePage(net, robots, it) }

/** robots.txt per host, fetched once: the `User-agent: *` group's Disallow / Allow rules. */
class Robots {
    private val cache = HashMap<String, List<Pair<Boolean, String>>>()

    fun allowed(net: Net, url: String): Boolean {
        val u = URI(url)
        val key = "${u.scheme}://${u.authority}"
        val rules = cache.getOrPut(key) {
            val r = runCatching { net.get("$key/robots.txt", "text/plain") }.getOrNull()
            if (r == null || r.status !in 200..299) emptyList() else parseRobots(r.text)
        }
        val path = (u.rawPath ?: "/").ifEmpty { "/" } + (u.rawQuery?.let { "?$it" } ?: "")
        return robotsAllows(rules, path)
    }
}

/** (allow, pattern) rules of the groups that name `*` or this example's agent. */
fun parseRobots(text: String): List<Pair<Boolean, String>> {
    val out = ArrayList<Pair<Boolean, String>>()
    var agents = ArrayList<String>()
    var inRules = false
    for (raw in text.lines()) {
        val line = raw.substringBefore('#').trim()
        if (line.isEmpty()) continue
        val k = line.substringBefore(':').trim().lowercase()
        val v = line.substringAfter(':', "").trim()
        when (k) {
            "user-agent" -> { if (inRules) { agents = ArrayList(); inRules = false }; agents += v.lowercase() }
            "allow", "disallow" -> {
                inRules = true
                if (agents.any { it == "*" || "tlaloc-brooklyn-eats".startsWith(it) } && v.isNotEmpty()) out += (k == "allow") to v
            }
        }
    }
    return out
}

/** Longest matching rule wins; Allow wins a tie; `*` and a trailing `$` are honoured. */
fun robotsAllows(rules: List<Pair<Boolean, String>>, path: String): Boolean {
    var best: Pair<Boolean, String>? = null
    for (r in rules) {
        val pat = r.second
        val re = Regex("^" + pat.split('*').joinToString(".*") { Regex.escape(it.removeSuffix("$")) } + (if (pat.endsWith("$")) "$" else ""))
        if (!re.containsMatchIn(path)) continue
        val b = best
        if (b == null || pat.length > b.second.length || (pat.length == b.second.length && r.first)) best = r
    }
    return best?.first ?: true
}
