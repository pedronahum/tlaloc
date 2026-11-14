package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.text.Normalizer
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset

// Scores a final answer to a task against the snapshot:
//   facts      every required fact of the ground truth is stated (names, grades, dates, counts, prices, ...)
//   invented   every restaurant name in a restaurant position (bold, list item, before an id, a
//              title-case name with a place word such as "Kitchen") is a snapshot restaurant;
//              every CAMIS or OSM id exists; every letter grade stated is a grade the restaurants
//              of its sentence hold or held; every date stated is a date of the snapshot for the
//              restaurants concerned (inspections, grades, complaints, reviews, menu capture), of
//              the question, or the cutoff
//   extra      for list questions, every snapshot restaurant named is in the answer set
//   calls      every tool call of the trajectory names a tool and has arguments valid against its schema
// An answer passes when nothing is missing, invented or extra and every call is valid.

class Verdict(
    val taskId: String,
    val scored: Boolean,
    val missing: List<String>,
    val invented: List<String>,
    val extra: List<String>,
    val invalidCalls: List<String>,
    val requiredFacts: Int,
    val foundFacts: Int,
    val calls: Int,
) {
    val pass: Boolean get() = scored && missing.isEmpty() && invented.isEmpty() && extra.isEmpty() && invalidCalls.isEmpty()
    fun json(): Map<String, Any?> = linkedMapOf(
        "id" to taskId, "scored" to scored, "pass" to pass, "facts" to "$foundFacts/$requiredFacts",
        "missing" to missing.ifEmpty { null }, "invented" to invented.ifEmpty { null }, "extra" to extra.ifEmpty { null },
        "invalid_calls" to invalidCalls.ifEmpty { null }, "calls" to calls,
    )
    override fun toString() = toJson(json())
}

private val MONTHS = listOf("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")
private val MONTH_RX = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\.?"
private fun monthOf(s: String): Int = MONTHS.indexOfFirst { it.startsWith(s.lowercase().trimEnd('.').take(3)) } + 1

/** A date stated in a text; [day] null when only the month is given. */
class DateMention(val year: Int, val month: Int, val day: Int?, val start: Int, val end: Int) {
    fun date(): LocalDate? = day?.let { runCatching { LocalDate.of(year, month, it) }.getOrNull() }
}

fun dateMentions(text: String): List<DateMention> {
    val out = ArrayList<DateMention>()
    fun add(y: String, m: Int, d: String?, r: IntRange) {
        if (m !in 1..12 || out.any { r.first < it.end && it.start < r.last + 1 }) return
        out += DateMention(y.toInt(), m, d?.toInt(), r.first, r.last + 1)
    }
    Regex("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b").findAll(text).forEach { add(it.groupValues[1], it.groupValues[2].toInt(), it.groupValues[3], it.range) }
    Regex("\\b$MONTH_RX\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})\\b", RegexOption.IGNORE_CASE).findAll(text)
        .forEach { add(it.groupValues[3], monthOf(it.groupValues[1]), it.groupValues[2], it.range) }
    Regex("\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?$MONTH_RX,?\\s+(\\d{4})\\b", RegexOption.IGNORE_CASE).findAll(text)
        .forEach { add(it.groupValues[3], monthOf(it.groupValues[2]), it.groupValues[1], it.range) }
    Regex("\\b(\\d{1,2})/(\\d{1,2})/(\\d{4})\\b").findAll(text).forEach { add(it.groupValues[3], it.groupValues[1].toInt(), it.groupValues[2], it.range) }
    Regex("\\b$MONTH_RX\\s+(\\d{4})\\b", RegexOption.IGNORE_CASE).findAll(text).forEach { add(it.groupValues[2], monthOf(it.groupValues[1]), null, it.range) }
    return out.sortedBy { it.start }
}

/** A letter grade stated in a text. */
class GradeMention(val letter: String, val start: Int)

private val GRADE_PATTERNS = listOf(
    // "grade A", "graded B", "grade of A", "grade is: C", "Grade: A"
    Regex("(?i:\\bgrade[sd]?)(?:\\s+(?:of|is|was|now|currently|still))*\\s*:?\\s*(?:an?\\s+)?[*\"'“‘]*([ABC])(?![\\w'’-])"),
    // "an A grade", "a B-grade", "its C grade"
    Regex("\\b(?:an?|the|its|their|with an?)\\s+[*\"'“‘]*([ABC])[*\"'”’]*(?:\\s+|-)(?i:grade)\\b"),
    // "A-graded", "B-rated"
    Regex("\\b([ABC])-(?i:grade[sd]?|rated)\\b"),
    // "the grade at Joe's is A", "grade posted at Joe's remains B"
    Regex("(?i:\\bgrade\\b)[^.\\n]{1,50}?\\b(?i:is|remains|was|of)\\s+(?:an?\\s+)?[*\"'“‘]*([ABC])(?![\\w'’-])"),
)

fun gradeMentions(text: String): List<GradeMention> {
    val out = ArrayList<GradeMention>()
    for (p in GRADE_PATTERNS) for (m in p.findAll(text)) {
        val g = m.groups[1]!!
        val after = text.substring(g.range.last + 1).take(24)
        val before = text.substring(0, g.range.first).takeLast(8)
        // A range or a list of letters ("B or better", "A, B or C") is not a claim about a restaurant.
        if (Regex("^[*\"'”’]*\\s*(?:,\\s*[ABC]\\b|/\\s*[ABC]\\b|or\\s+[ABC]\\b|and\\s+[ABC]\\b|or\\s+(?:better|above|higher|worse|lower|below))").containsMatchIn(after)) continue
        if (Regex("\\b[ABC]\\s*(?:,|/|or|and)\\s*[*\"'“‘]*$").containsMatchIn(before)) continue
        if (out.none { it.start == g.range.first }) out += GradeMention(g.value, g.range.first)
    }
    return out.sortedBy { it.start }
}

private fun stripAccents(s: String) = Normalizer.normalize(s, Normalizer.Form.NFKD).replace(Regex("\\p{M}+"), "")

/** A word of a text as [normalizeName] reads it, with where it is and whether it was capitalized. */
private class Word(val key: String, val start: Int, val end: Int, val capital: Boolean)

private fun words(text: String): List<Word> = Regex("[\\p{L}\\p{N}'’]+|&").findAll(text).mapNotNull { m ->
    val w = stripAccents(m.value).lowercase().replace(Regex("['’]s$"), "").replace(Regex("['’`´]"), "")
    val key = if (w == "&") "and" else w.replace(Regex("[^a-z0-9]"), "")
    if (key.isEmpty() || key in NAME_STOPWORDS) null else Word(key, m.range.first, m.range.last + 1, m.value.any { it.isUpperCase() } || m.value[0].isDigit())
}.toList()

/** Words that are restaurant names on their own but mostly mean something else in an answer. */
private val COMMON_WORDS = setOf(
    "grade", "grades", "score", "brooklyn", "google", "dohmh", "nyc", "yes", "no", "it", "its", "this", "that", "there", "these",
    "they", "their", "here", "today", "tonight", "open", "closed", "menu", "price", "prices", "review", "reviews", "rating", "ratings",
    "inspection", "inspections", "violation", "violations", "critical", "complaint", "complaints", "health", "department", "city",
    "new", "york", "best", "good", "great", "local", "home", "sources", "source", "snapshot", "data", "note", "however", "also",
    "both", "neither", "either", "which", "what", "when", "where", "who", "how", "but", "so", "if", "then", "since", "last", "latest",
    "most", "recent", "average", "stars", "star", "total", "none", "all", "some", "one", "two", "three", "four", "five", "first",
    "second", "third", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "am", "pm", "i", "you", "we",
    "my", "your", "our", "he", "she", "his", "her", "unfortunately", "sorry", "according", "based", "per", "status", "address",
    "hours", "date", "answer", "summary", "result", "results", "option", "options", "place", "places", "spot", "spots", "street",
    "avenue", "ave", "st", "road", "rd", "boulevard", "blvd", "parkway", "internet", "archive", "openstreetmap", "osm", "camis",
    "id", "ids", "letter", "pending", "not", "is", "was", "were", "are", "has", "had", "have", "with", "for", "in", "on", "at",
    "to", "from", "by", "as", "or", "an", "be", "been", "about", "out", "only", "more", "less", "than", "lower", "higher", "better",
    "north", "south", "east", "west", "eastern", "western", "northern", "southern", "upper", "lower", "center", "central",
) + MONTHS + MONTHS.map { it.take(3) } + GENERIC_WORDS

/** Place words that make a title-case phrase a restaurant name. */
private val PLACE_WORDS = setOf(
    "kitchen", "pizzeria", "cafe", "café", "deli", "grill", "bistro", "diner", "bakery", "restaurant", "tavern", "taqueria",
    "trattoria", "cantina", "eatery", "shack", "lounge", "pub", "bar", "palace", "brasserie", "osteria", "luncheonette",
    "canteen", "joint", "creamery", "patisserie", "steakhouse", "noodle house", "chophouse", "gastropub", "taproom", "cafeteria",
)

/** First words of a bold or list phrase that is a label, not a name. */
private val LABEL_WORDS = setOf(
    "grade", "grades", "score", "note", "notes", "answer", "summary", "yes", "no", "critical", "violation", "violations",
    "inspection", "inspections", "complaint", "complaints", "rating", "ratings", "reviews", "review", "menu", "price", "address",
    "hours", "date", "dates", "source", "sources", "closed", "open", "status", "result", "results", "latest", "average", "total",
    "count", "not", "none", "neither", "both", "cuisine", "neighborhood", "phone", "website", "google", "dohmh", "nyc", "311",
    "camis", "id", "the", "a", "an", "most", "last", "next", "option", "options", "warning", "caveat", "caveats", "verdict",
    "winner", "comparison", "conclusion", "recommendation", "tip", "update", "in", "on", "at", "for", "since", "as", "of", "per",
)

private val STREET_AFTER = Regex("^\\s+(?i:avenue|ave|street|st|road|rd|boulevard|blvd|parkway|pkwy|place|pl|drive|dr|court|ct|lane|ln|terrace|walk|plaza|square|highway|expressway)\\b")

private val STREET_SUFFIX = Regex(
    "(?i)\\b(avenue|ave|street|st|road|rd|boulevard|blvd|parkway|pkwy|place|pl|drive|dr|court|ct|lane|ln|terrace|way|plaza|square|highway|expressway|bridge|park|beach)\\.?$",
)

class Verifier(private val s: ToolServer, private val cutoff: LocalDate) {
    private val byId = s.restaurants.associateBy { it.id }
    private val byCamis = s.restaurants.filter { it.camis != null }.associateBy { it.camis!! }
    private val byOsm = s.restaurants.filter { it.osmId != null }.associateBy { it.osmId!! }

    private fun names(r: Rest) = listOfNotNull(r.name, r.dba?.let(::titleCase), r.osmName, r.gName).distinct()

    /** Normalized name -> ids, over every name of every restaurant. */
    private val nameIndex: Map<String, Set<String>> = HashMap<String, MutableSet<String>>().also { m ->
        for (r in s.restaurants) for (n in names(r)) {
            val k = nameKey(n)
            if (k.isNotEmpty()) m.getOrPut(k) { HashSet() } += r.id
        }
    }
    private val maxNameWords = nameIndex.keys.maxOfOrNull { it.count { c -> c == ' ' } + 1 } ?: 1

    /** Every run of consecutive words of every name: a phrase found in here is part of a real name. */
    private val namePieces: Set<String> = HashSet<String>().also { set ->
        for (k in nameIndex.keys) {
            val ws = k.split(' ')
            for (i in ws.indices) for (j in i + 1..ws.size) set += ws.subList(i, j).joinToString(" ")
        }
    }

    /** Neighborhood names and their parts ("Clinton Hill", "Bedford Stuyvesant", "Sea Gate"). */
    private val neighborhoodKeys: Set<String> = s.neighborhoods.flatMap { n ->
        val base = n.name.replace(Regex("\\([^)]*\\)"), " ")
        listOf(base) + base.split('-')
    }.map { nameKey(it) }.filter { it.isNotEmpty() }.toSet() + setOf("dumbo", "bed stuy", "south slope", "ditmas park", "sea gate")

    private val gradeCache = HashMap<String, Set<String>>()
    /** The current grade and every grade in the restaurant's inspections. */
    fun gradesOf(id: String): Set<String> = gradeCache.getOrPut(id) {
        val r = byId[id] ?: return@getOrPut emptySet()
        val out = HashSet<String>()
        r.grade?.let { out += it }
        r.camis?.let { c -> s.query("SELECT DISTINCT grade FROM inspections WHERE camis=? AND grade IS NOT NULL", c).forEach { out += it[0] as String } }
        out
    }

    private val dateCache = HashMap<String, Set<LocalDate>>()
    /** Every date the snapshot holds for a restaurant: inspections, grades, complaints, reviews, menu capture. */
    fun datesOf(id: String): Set<LocalDate> = dateCache.getOrPut(id) {
        val r = byId[id] ?: return@getOrPut emptySet()
        val out = HashSet<String>()
        listOfNotNull(r.gradeDate, r.latestInspection).forEach { out += it }
        r.camis?.let { c ->
            s.query("SELECT DISTINCT inspection_date FROM inspections WHERE camis=? UNION SELECT DISTINCT grade_date FROM inspections WHERE camis=?", c, c)
                .forEach { row -> (row[0] as String?)?.let { out += it } }
            s.query("SELECT created_date, closed_date FROM complaints WHERE camis=?", c)
                .forEach { row -> row.forEach { v -> (v as String?)?.let { out += it } } }
        }
        r.gmapId?.let { g ->
            s.query("SELECT DISTINCT time_ms FROM reviews WHERE gmap_id=?", g).forEach { row ->
                out += Instant.ofEpochMilli((row[0] as Number).toLong()).atZone(ZoneOffset.UTC).toLocalDate().toString()
            }
        }
        s.query("SELECT snapshot FROM menus WHERE restaurant_id=?", id).forEach { row -> (row[0] as String?)?.let { out += it } }
        out.mapNotNull { runCatching { LocalDate.parse(it.take(10)) }.getOrNull() }.toSet()
    }

    /** Snapshot restaurants named in [text]: (ids, start, end), longest names first, left to right. */
    fun restaurantMentions(text: String): List<Triple<Set<String>, Int, Int>> {
        val ws = words(text)
        val out = ArrayList<Triple<Set<String>, Int, Int>>()
        var i = 0
        outer@ while (i < ws.size) {
            // Neighborhood names ("Clinton Hill", "Sea Gate") are skipped whole, so their words are not read as restaurants.
            for (n in minOf(4, ws.size - i) downTo 1) {
                val run = ws.subList(i, i + n)
                if (run.first().capital && run.joinToString(" ") { it.key } in neighborhoodKeys) { i += n; continue@outer }
            }
            var hit = false
            for (n in minOf(maxNameWords, ws.size - i) downTo 1) {
                val run = ws.subList(i, i + n)
                val ids = nameIndex[run.joinToString(" ") { it.key }] ?: continue
                // In a run that crosses a line or sentence end the words are not one name.
                if ((0 until n - 1).any { j -> Regex("[\\n.!?:;]").containsMatchIn(text.substring(run[j].end, run[j + 1].start)) }) continue
                val ok = if (n == 1) run[0].capital && run[0].key.length >= 3 && run[0].key !in COMMON_WORDS
                else run.any { it.capital } && run.any { it.key !in COMMON_WORDS }
                if (!ok) continue
                // Street names ("Myrtle Avenue", "216 Bedford") are not restaurants: the whole run is skipped.
                val st = run.first().start
                val en = run.last().end
                if (STREET_AFTER.containsMatchIn(text.substring(en, minOf(text.length, en + 14))) ||
                    Regex("\\d[A-Za-z]?\\s+(?:(?:East|West|North|South|E|W|N|S)\\.?\\s+)?$").containsMatchIn(text.substring(maxOf(0, st - 12), st))
                ) { i += n; hit = true; break }
                out += Triple(ids, st, en)
                i += n
                hit = true
                break
            }
            if (!hit) i++
        }
        return out
    }

    /** Phrases in restaurant positions: bold, list-item leads, a name before an id, title-case phrases with a place word. */
    fun nameCandidates(text: String): List<String> {
        val out = LinkedHashSet<String>()
        Regex("\\*\\*([^*\\n]{2,80})\\*\\*|__([^_\\n]{2,80})__").findAll(text).forEach { out += (it.groups[1] ?: it.groups[2])!!.value }
        Regex("(?m)^\\s*(?:[-*•]|\\d+[.)])\\s+(.+)$").findAll(text).forEach { m ->
            val lead = m.groupValues[1].replace("**", "").split(Regex("\\s+[—–-]\\s+|:|\\s+\\(|,|\\s+at\\s+\\d|\\s+(?:has|had|is|was|got|scored|received|earned)\\b"))[0]
            out += lead
        }
        Regex("([\\p{L}\\p{N}][^()\\n]{1,60}?)\\s*\\((?:id|ID|CAMIS|camis)\\b").findAll(text).forEach { out += it.groupValues[1] }
        val tok = "(?:[\\p{Lu}\\d][\\p{L}\\p{N}'’.&-]*)"
        Regex("$tok(?:\\s+(?:of|the|de|del|la|el|y|and|&|n'|'n'|$tok))*\\s+$tok|$tok").findAll(text).forEach { m ->
            val ws = m.value.split(Regex("\\s+"))
            if (ws.size >= 2 && ws.any { stripAccents(it).lowercase().trimEnd('.') in PLACE_WORDS }) out += m.value
        }
        return out.map { it.trim().trim('*', '"', '“', '”', '\'', '.', ',', ';', ':').trim() }.filter { it.isNotEmpty() }
    }

    private fun isNonName(c: String): Boolean {
        val lower = stripAccents(c).lowercase()
        val first = lower.split(Regex("[^a-z0-9]+")).firstOrNull { it.isNotEmpty() } ?: return true
        if (!(c[0].isUpperCase() || c[0].isDigit())) return true
        if (first in LABEL_WORDS) return true
        if (c.split(Regex("\\s+")).size > 8) return true
        if (STREET_SUFFIX.containsMatchIn(c.trim())) return true
        if (Regex("^[\\d\\s,./-]+$").matches(c)) return true
        if (dateMentions(c).isNotEmpty() && dateMentions(c).sumOf { it.end - it.start } >= c.length - 2) return true
        if (resolveNeighborhood(s.neighborhoods, c).isNotEmpty() && nameIndex[nameKey(c)] == null) return true
        if (TASK_CUISINES.keys.any { lower == it || lower == "$it food" || lower == "$it restaurants" }) return true
        val keys = normalizeName(c)
        return keys.all { it in COMMON_WORDS }
    }

    /** True when [c] is a snapshot restaurant name, part of one, or contains one. */
    fun isKnownName(c: String): Boolean {
        val ws = normalizeName(c)
        val k = ws.joinToString(" ")
        if (k.isEmpty()) return true
        if (k in nameIndex) return true
        val distinctive = ws.filter { it !in GENERIC_WORDS && it.length >= 3 }
        if (distinctive.isNotEmpty() && k in namePieces) return true
        // A phrase that holds a known name is known only when its other words are generic
        // ("Sakura Japanese Restaurant" holds "Sakura"); "Golden Lantern Kitchen" holding "Golden" is not.
        for (i in ws.indices) for (j in i + 1..ws.size) {
            val sub = ws.subList(i, j)
            val rest = ws.subList(0, i) + ws.subList(j, ws.size)
            if (sub.any { it !in GENERIC_WORDS && it.length >= 3 && it !in COMMON_WORDS } && sub.joinToString(" ") in nameIndex &&
                rest.all { it in GENERIC_WORDS || it in COMMON_WORDS }
            ) return true
        }
        return false
    }

    /** Text segments: lines, and sentences within a line. */
    private fun segments(text: String): List<IntRange> {
        val out = ArrayList<IntRange>()
        var lineStart = 0
        for (line in text.split('\n')) {
            var st = 0
            for (m in Regex("(?<=[.!?])\\s+(?=[\\p{Lu}\\d*\"“(])").findAll(line)) {
                // "Ms. Ohho", "N. 6th St.": a short abbreviation does not end a sentence.
                if (Regex("(?:^|[\\s(])(?:[A-Z]|Mr|Ms|Mrs|Dr|St|Ave|Jr|Sr|Mt|Ft|No|Bros|Inc|Co|Corp)\\.$").containsMatchIn(line.substring(0, m.range.first))) continue
                out += (lineStart + st) until (lineStart + m.range.first)
                st = m.range.last + 1
            }
            out += (lineStart + st) until (lineStart + line.length)
            lineStart += line.length + 1
        }
        return out.filter { !it.isEmpty() }
    }

    /** CAMIS and OSM ids written in [text]: (id or null when it does not exist, the text). */
    private fun idMentions(text: String): List<Triple<String?, String, Int>> {
        val out = ArrayList<Triple<String?, String, Int>>()
        Regex("(?<![\\d-])\\d{8}(?![\\d-])").findAll(text).forEach { m -> out += Triple(byCamis[m.value]?.id, m.value, m.range.first) }
        Regex("\\bosm:(?:node|way|relation)/\\d+\\b").findAll(text).forEach { m -> out += Triple(byId[m.value]?.id ?: byOsm[m.value.removePrefix("osm:")]?.id, m.value, m.range.first) }
        return out
    }

    fun factPresent(f: Fact, answer: String): Boolean {
        val lower = stripAccents(answer).lowercase()
        return when (f.kind) {
            "name" -> {
                val norm = " " + normalizeName(answer).joinToString(" ") + " "
                // Other names made only of generic words ("Kitchen") do not count; the main name always does.
                (listOf(f.value) + f.alts.filter { a -> normalizeName(a).any { it !in GENERIC_WORDS && it !in COMMON_WORDS } })
                    .any { n -> nameKey(n).let { k -> k.isNotEmpty() && norm.contains(" $k ") } }
            }
            "grade" -> gradeMentions(answer).any { it.letter == f.value }
            "date" -> LocalDate.parse(f.value.take(10)).let { d -> dateMentions(answer).any { it.date() == d } }
            "count" -> {
                val v = f.value.toInt()
                val text = withoutDates(answer)
                val numbers = Regex("(?<![\\d.,$/:-])\\d{1,6}(?![\\d/]|[.,]\\d|-\\d)").findAll(text).map { it.value.toInt() }.toSet()
                v in numbers || NUMBER_WORDS[v]?.let { w -> Regex("\\b$w\\b").containsMatchIn(lower) } == true ||
                    (v == 0 && Regex("\\b(no|none|zero|never)\\b").containsMatchIn(lower)) ||
                    (v == 1 && Regex("\\b(once|a single|one)\\b").containsMatchIn(lower))
            }
            "number" -> {
                val v = f.value.toDouble()
                val tol = f.tol ?: 1e-9
                Regex("(?<![\\d.])\\d+(?:\\.\\d+)?").findAll(withoutDates(answer)).any { Math.abs(it.value.toDouble() - v) <= tol + 1e-9 }
            }
            "money" -> {
                val v = f.value.toDouble()
                Regex("\\$\\s?(\\d{1,4}(?:\\.\\d{1,2})?)|(?<![\\d.])(\\d{1,4}\\.\\d{2})(?![\\d.])").findAll(answer)
                    .any { m -> ((m.groups[1] ?: m.groups[2])!!.value.toDouble() - v).let { Math.abs(it) < 0.005 } }
            }
            "violation" -> {
                Regex("(?i)\\b${Regex.escape(f.value)}\\b").containsMatchIn(answer) || run {
                    val ws = Regex("[a-z]+").findAll(lower).map { it.value }.toList()
                    // The rarest word (first) on its own, or two of the words.
                    fun hit(k: String) = ws.any { w -> w.startsWith(k.take(5)) && (k.length >= 5 || w == k) }
                    f.alts.isNotEmpty() && (hit(f.alts[0]) || f.alts.count(::hit) >= 2)
                }
            }
            "keyword" -> (listOf(f.value) + f.alts).any { k -> Regex("\\b" + Regex.escape(k.lowercase())).containsMatchIn(lower) }
            "refusal" -> (if (f.value == "out_of_scope") OUT_OF_SCOPE else NOT_FOUND).containsMatchIn(answer)
            else -> false
        }
    }

    private fun withoutDates(t: String): String {
        val sb = StringBuilder(t)
        for (d in dateMentions(t).reversed()) sb.replace(d.start, d.end, " ".repeat(d.end - d.start))
        return sb.toString()
    }

    /** Problems with the trajectory's calls: unknown tools, arguments that are not a JSON object or break the schema. */
    fun invalidCalls(calls: List<ToolCall>): List<String> = calls.mapIndexedNotNull { i, c ->
        val def = TOOL_BY_NAME[c.name] ?: return@mapIndexedNotNull "call ${i + 1}: unknown tool '${c.name}'"
        val args = runCatching { parseJson(c.arguments.ifBlank { "{}" }) }.getOrNull() as? JsonObject
            ?: return@mapIndexedNotNull "call ${i + 1} ${c.name}: arguments are not a JSON object"
        val errs = validate(def.schema, args)
        if (errs.isEmpty()) null else "call ${i + 1} ${c.name}: " + errs.joinToString("; ")
    }

    fun verify(task: Task, answer: String, calls: List<ToolCall>): Verdict {
        val invalid = invalidCalls(calls)
        if (!task.offline) return Verdict(task.id, false, emptyList(), emptyList(), emptyList(), invalid, 0, 0, calls.size)
        val required = task.facts.filter { it.required }
        val missing = required.filter { !factPresent(it, answer) }.map { it.toString() }

        val invented = ArrayList<String>()
        val subjects = (task.restaurants + task.answerSet.orEmpty()).distinct()
        val questionKeys = nameCandidates(task.question).map { nameKey(it) }.toSet() + nameKey(task.question)

        // Restaurants: names in restaurant positions must be snapshot restaurants (or named in the question).
        for (c in nameCandidates(answer)) {
            if (isNonName(c) || isKnownName(c)) continue
            val k = nameKey(c)
            if (k in questionKeys || (k.isNotEmpty() && " ${nameKey(task.question)} ".contains(" $k "))) continue
            invented += "restaurant '$c' is not in the snapshot"
        }
        val ids = idMentions(answer)
        ids.filter { it.first == null }.forEach { invented += "id ${it.second} is not in the snapshot" }

        // Grades, sentence by sentence: a grade must be one the sentence's restaurants hold or held.
        // A name several restaurants share (a chain) means the task's own restaurants when it includes them.
        val mentions = restaurantMentions(answer).map { (idSet, a, b) ->
            Triple(idSet.intersect(subjects.toSet()).ifEmpty { idSet }, a, b)
        }
        var previous: Set<String> = emptySet()
        var previousLine = -1
        for (seg in segments(answer)) {
            // A sentence with no restaurant refers to the previous sentence's, on the same line only.
            val line = answer.substring(0, seg.first).count { it == '\n' }
            if (line != previousLine) previous = emptySet()
            previousLine = line
            val here = (mentions.filter { it.second in seg }.flatMap { it.first } + ids.filter { it.third in seg }.mapNotNull { it.first }).toSet()
            val segText = answer.substring(seg.first, seg.last + 1)
            val owners0 = here.ifEmpty { previous }.ifEmpty { subjects.toSet() }
            if (here.isNotEmpty()) previous = here
            // Of several restaurants, the ones whose street address the sentence gives.
            val segKey = " " + normalizeName(segText).joinToString(" ") + " "
            val owners = owners0.filter { id ->
                byId[id]?.address?.substringBefore(", Brooklyn")?.let { a -> nameKey(a).let { k -> k.isNotEmpty() && segKey.contains(" $k ") } } == true
            }.toSet().ifEmpty { owners0 }
            // A sentence in the present tense states the current grade; one with a date or a past-tense
            // marker may state an earlier grade from the inspection history.
            val past = PAST_MARKER.containsMatchIn(segText) || dateMentions(segText).isNotEmpty()
            for (g in gradeMentions(answer).filter { it.start in seg }) {
                if (owners.none { g.letter in (if (past) gradesOf(it) else setOfNotNull(byId[it]?.grade)) }) {
                    val who = owners.mapNotNull { byId[it]?.name }.distinct().take(3)
                    invented += "grade ${g.letter} " + if (who.isEmpty()) "with no restaurant that holds it"
                    else "is not ${if (past) "a" else "the current"} grade of ${who.joinToString(" / ")}"
                }
            }
        }

        // Dates: a date of the snapshot for the restaurants concerned, of the question or facts, or the cutoff.
        val allowed = HashSet<LocalDate>()
        allowed += listOf(cutoff, cutoff.minusDays(1), cutoff.minusYears(1), LocalDate.parse(task.asOf))
        dateMentions(task.question).mapNotNull { it.date() }.forEach { allowed += it }
        task.facts.filter { it.kind == "date" }.forEach { allowed += LocalDate.parse(it.value.take(10)) }
        for (id in (subjects + mentions.flatMap { it.first } + ids.mapNotNull { it.first }).distinct()) allowed += datesOf(id)
        val questionMonths = dateMentions(task.question).map { it.year to it.month }.toSet()
        for (d in dateMentions(answer)) {
            val ok = d.date()?.let { it in allowed } ?: (allowed.any { it.year == d.year && it.monthValue == d.month } || (d.year to d.month) in questionMonths)
            if (!ok) invented += "date ${answer.substring(d.start, d.end)} is not a date of the snapshot for these restaurants"
        }

        // List questions: every snapshot restaurant named must satisfy the question.
        val extra = ArrayList<String>()
        task.answerSet?.let { set ->
            val setNames = set.mapNotNull { byId[it] }.flatMap { names(it) }.map { nameKey(it) }.toSet()
            for ((idSet, a, b) in mentions) {
                if (idSet.any { it in set }) continue
                val k = nameKey(answer.substring(a, b))
                if (setNames.any { it.contains(k) }) continue
                extra += "'${answer.substring(a, b)}' does not match the question"
            }
        }
        return Verdict(task.id, true, missing, invented.distinct(), extra.distinct(), invalid, required.size, required.size - missing.size, calls.size)
    }
}

private val NUMBER_WORDS = mapOf(
    1 to "one", 2 to "two", 3 to "three", 4 to "four", 5 to "five", 6 to "six", 7 to "seven", 8 to "eight", 9 to "nine",
    10 to "ten", 11 to "eleven", 12 to "twelve",
)

private val PAST_MARKER = Regex("(?i)\\b(was|were|had|previous(?:ly)?|earlier|before|prior|used to|formerly|history|historically|once|in (?:19|20)\\d\\d)\\b")

val NOT_FOUND = Regex(
    "(?i)(could ?n[o'’]t find|can ?n[o'’]t find|cannot find|did ?n[o'’]t find|unable to find|found no\\b|no (?:record|records|restaurant|place|match|matches|results?|listing|entry|data|information)\\b[^.]{0,40}\\b(?:for|of|about|called|named|matching|under|by)\\b|" +
        "not (?:found|listed|in (?:the|my|our|these) (?:data|records|snapshot|database|sources))|does ?n[o'’]t (?:appear|exist|show up)|no such|is ?n[o'’]t listed|" +
        "do ?n[o'’]t have any (?:record|data|information)|nothing (?:on|about|for) (?:it|that|a))",
)

val OUT_OF_SCOPE = Regex(
    "(?i)(only (?:cover|covers|include|includes|has|have|hold|holds|contain|contains|know about|about|for)\\b[^.]{0,60}brooklyn|outside (?:of )?brooklyn|not in brooklyn|" +
        "is ?n[o'’]t in brooklyn|brooklyn only|only brooklyn|limited to brooklyn|beyond brooklyn|restricted to brooklyn|not part of brooklyn)",
)
