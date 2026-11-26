package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.text.Normalizer
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset

// Scores a final answer to a task against the snapshot:
//   facts      every required fact of the ground truth is stated (names, grades, dates, counts, prices, ...),
//              not negated ("does not have grade A"), not hedged among other values ("1, 2 or 3",
//              "$9.95 or $10.95") or offered as a guess ("probably has grade A"),
//              and not contradicted: every number written before the fact's unit ("7 critical
//              violations", "4.2 stars") is a value of the ground truth, the price given for the dish
//              asked about is its price, digits inside restaurant names ("China No. 1") do not count;
//              in a comparison each value goes with its own restaurant, the restaurant the answer
//              picks is the right one, and the answer does not say higher inspection scores are better
//   invented   every restaurant name in a restaurant position (bold, list item, before an id, a
//              title-case name with a place word such as "Kitchen" or "Kitchen's") is a snapshot restaurant;
//              every CAMIS or OSM id exists; every letter grade stated is a grade the restaurants
//              of its sentence hold or held; every date stated is a date of the snapshot for the
//              restaurants concerned (inspections, grades, complaints, reviews, menu capture), of
//              the question; the cutoff (and a year or a day before it) only as the bound of a
//              period ("as of", "since"); every price stated is a price on the archived menu;
//              every violation code is one the restaurant was cited for (in a question about the
//              latest inspection, one of that inspection), and a pest named there is in its violations;
//              claims the task forbids ("absent" facts: "has since reopened", "food poisoning" for
//              another complaint type) are not made outside a negated clause
//   extra      for list questions, every snapshot restaurant named is in the answer set, unless the
//              answer sets it apart: its sentence, or the line introducing its list or table, says it
//              does not qualify ("so it doesn't qualify", "Not open at 1 PM:", "hours unknown",
//              "outside Sunset Park", "listed as Caribbean, not a sandwich shop"); or a reader would
//              count it: it meets every other condition of the search the question asks, and its
//              name names the cuisine, or its records say Tex-Mex for a taco or Mexican question
//              ("East Coast Street Tacos", Tex-Mex in DOHMH, for taco places in East New York)
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
    // "currently holds an A from the health department", "It might be C now", "it has a B posted"
    Regex(
        "\\b(?i:holds?|held|has|had|have|got|gets|received|receives|earned|earns|carries|carried|displays|displayed|posts|posted|keeps|kept|" +
            "maintains|is|was|be|been|remains|remained|rated|scored|graded|get|hold|earn|receive)\\s+(?:(?i:an?|the|its|their)\\s+)?" +
            "(?:(?i:letter|health|current|posted|dohmh|solid|clean)\\s+)*[*\"'“‘]*([ABC])[*\"'”’]*(?![\\w'’-])" +
            "(?=\\s*(?:[.,;:!?)\\n—–]|$|-\\s|(?i:now|today|currently|from|since|grade|rating|at|on|in|as|posted|given|after|by|and|but|for|with|again|still|too|though|then|or)\\b))",
    ),
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
    "canteen", "joint", "creamery", "patisserie", "steakhouse", "chophouse", "gastropub", "taproom", "cafeteria",
    "noodle", "noodles", "wok", "grille", "kebab", "ramen", "sushi", "bbq", "dumplings",
)

/** First words of a bold or list phrase that is a label, not a name. */
private val LABEL_WORDS = setOf(
    "grade", "grades", "score", "note", "notes", "answer", "summary", "yes", "no", "critical", "violation", "violations",
    "inspection", "inspections", "complaint", "complaints", "rating", "ratings", "reviews", "review", "menu", "price", "address",
    "hours", "date", "dates", "source", "sources", "closed", "open", "status", "result", "results", "latest", "average", "total",
    "count", "not", "none", "neither", "both", "cuisine", "neighborhood", "phone", "website", "google", "dohmh", "nyc", "311",
    "camis", "id", "the", "a", "an", "most", "last", "next", "option", "options", "warning", "caveat", "caveats", "verdict",
    "winner", "comparison", "conclusion", "recommendation", "tip", "update", "in", "on", "at", "for", "since", "as", "of", "per",
    // Field labels of a listed record ("- **Type:** Food Establishment", "- Created: January 17, 2026").
    "type", "created", "descriptor", "resolution", "filed", "reported", "location", "details", "description", "category",
    "outcome", "agency", "received", "submitted", "code", "codes", "non", "staff", "overall", "customer", "customers", "reviewer",
    "reviewers", "pros", "cons", "highlights", "verdict", "bottom", "restaurant", "detail",
    // Ripple's labels ("- **Praise:** ...", "- **Action:** Establishment Closed", "- **Geocoding** ... no match").
    "praise", "criticism", "criticisms", "positive", "positives", "negative", "negatives", "issue", "issues", "action", "geocoding",
)

/** Lowercase words a restaurant name may hold ("Chez de la Mer", "Rice and Beans", "Pizza by the Slice"). */
private val NAME_CONNECTIVES = setOf("of", "the", "de", "del", "la", "le", "les", "el", "los", "las", "y", "e", "and", "n", "'n'", "n'", "&",
    "da", "di", "du", "des", "on", "at", "in", "by", "a", "an", "to", "for", "con", "al", "von", "van", "der")

private val STREET_WORD = Regex("(?i)\\b(avenue|ave|street|st|road|rd|boulevard|blvd|parkway|pkwy|place|pl|drive|dr|court|ct|lane|ln|terrace|plaza|highway)\\b")

private val LEADING_CONNECTIVE = Regex("^(?:Thus|So|Also|However|Overall|Therefore|Additionally|Meanwhile|Instead|Then|Finally|Hence|Still|Otherwise|Similarly|By contrast|In contrast|In summary|In short),?\\s+")

private val STREET_AFTER = Regex("^\\s+(?i:avenue|ave|street|st|road|rd|boulevard|blvd|parkway|pkwy|place|pl|drive|dr|court|ct|lane|ln|terrace|walk|plaza|square|highway|expressway)\\b")

private val STREET_SUFFIX = Regex(
    "(?i)\\b(avenue|ave|street|st|road|rd|boulevard|blvd|parkway|pkwy|place|pl|drive|dr|court|ct|lane|ln|terrace|way|plaza|square|highway|expressway|bridge|park|beach)\\.?$",
)

class Verifier(private val s: ToolServer, private val cutoff: LocalDate) {
    private val byId = s.restaurants.associateBy { it.id }
    private val byCamis = s.restaurants.filter { it.camis != null }.associateBy { it.camis!! }
    private val byOsm = s.restaurants.filter { it.osmId != null }.associateBy { it.osmId!! }

    private fun names(r: Rest) = listOfNotNull(r.name, r.dba?.let(::titleCase), r.osmName, r.gName).distinct()

    /**
     * The restaurants a reader would count for a cuisine list question: its search (the task's
     * `search_restaurants` call) without the cuisine condition, kept when a name of the place names
     * the cuisine, or its records say Tex-Mex and the cuisine is tacos or Mexican. Empty when the
     * task's search has no cuisine.
     */
    private fun looseAnswerSet(task: Task): Set<String> {
        val call = task.solution.firstOrNull { it.name == "search_restaurants" } ?: return emptySet()
        val args = parseJson(call.arguments) as? JsonObject ?: return emptySet()
        val cuisine = args.optString("cuisine") ?: return emptySet()
        @Suppress("UNCHECKED_CAST") val wide = LinkedHashMap(args.toKotlin() as Map<String, Any?>).also { it.remove("cuisine") }
        val stem = cuisine.lowercase().replace(Regex("[^a-z0-9]+"), " ").trim().removeSuffix("s")
        val texMex = stem in setOf("taco", "mexican")
        val ids = s.searchAllIds(toJson(wide))
        return ids.filter { id ->
            val r = byId[id] ?: return@filter false
            names(r).any { n ->
                val text = n.lowercase().replace(Regex("[^a-z0-9]+"), " ").trim()
                text.split(' ').any { it.removeSuffix("s") == stem } || cuisineTextMatches(text, cuisine)
            } ||
                (texMex && " ${r.cuisineText} ".contains(" tex mex "))
        }.toSet()
    }

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

    /** 311 complaint types and descriptors ("Food Poisoning", "Odor", "Rodents/Insects/Garbage"): labels, not restaurants. */
    private val complaintLabels: Set<String> =
        s.query("SELECT DISTINCT complaint_type FROM complaints UNION SELECT DISTINCT descriptor FROM complaints")
            .mapNotNull { (it[0] as String?)?.let(::nameKey) }.filter { it.isNotEmpty() }.toSet()

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
                // Street names ("Myrtle Avenue", "216 Bedford", "148 Noble Street") are not restaurants: the whole run is skipped.
                val st = run.first().start
                val en = run.last().end
                if ((run.first().key[0].isDigit() && STREET_SUFFIX.containsMatchIn(text.substring(st, en))) ||
                    STREET_AFTER.containsMatchIn(text.substring(en, minOf(text.length, en + 14))) ||
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
        Regex("\\*\\*([^*\\n]{2,80})\\*\\*|__([^_\\n]{2,80})__").findAll(text).forEach { m ->
            val inner = (m.groups[1] ?: m.groups[2])!!.value
            // "**Type:**" is a field label, not a name in a restaurant position.
            if (!inner.trimEnd().endsWith(":")) out += inner
        }
        Regex("(?m)^\\s*(?:[-*•]|\\d+[.)])\\s+(.+)$").findAll(text).forEach { m ->
            val lead = m.groupValues[1].replace("**", "").split(Regex("\\s+[—–-]\\s+|:|\\s+\\(|,|\\s+at\\s+\\d|\\s+(?:has|had|is|was|got|scored|received|earned)\\b"))[0]
            out += lead
        }
        Regex("([\\p{L}\\p{N}][^()\\n]{1,60}?)\\s*\\((?:id|ID|CAMIS|camis)\\b").findAll(text).forEach { out += it.groupValues[1] }
        val tok = "(?:[\\p{Lu}\\d][\\p{L}\\p{N}'’.&-]*)"
        Regex("$tok(?:[ \\t]+(?:of|the|de|del|la|el|y|and|&|n'|'n'|$tok))*[ \\t]+$tok|$tok").findAll(text).forEach { m ->
            val ws = m.value.split(Regex("\\s+"))
            // "Golden Lantern Kitchen's menu": a possessive place word still makes a name.
            if (ws.size >= 2 && ws.any { stripAccents(it).lowercase().trimEnd('.').replace(Regex("['’]s$"), "") in PLACE_WORDS }) out += m.value
        }
        // A sentence's opening connective ("Thus Lanzhou Noodles ...") is not part of the name.
        return out.map { it.trim().trim('*', '"', '“', '”', '\'', '.', ',', ';', ':').trim().replace(Regex("^\\d+[.)]\\s+"), "").replace(LEADING_CONNECTIVE, "").trim() }.filter { it.isNotEmpty() }
    }

    private fun isNonName(c: String): Boolean {
        val lower = stripAccents(c).lowercase()
        val first = lower.split(Regex("[^a-z0-9]+")).firstOrNull { it.isNotEmpty() } ?: return true
        if (!(c[0].isUpperCase() || c[0].isDigit())) return true
        if (first in LABEL_WORDS) return true
        // A 311 complaint type or descriptor ("Food Poisoning"), or several joined ("Food Establishment — Food Temperature").
        if (nameKey(c) in complaintLabels && nameIndex[nameKey(c)] == null) return true
        if (c.split(Regex("\\s+[—–-]\\s+")).let { ps -> ps.size > 1 && ps.all { nameKey(it) in complaintLabels } }) return true
        // A DOHMH action ("Establishment Closed by DOHMH").
        if (Regex("\\bDOHMH\\b").containsMatchIn(c) && nameIndex[nameKey(c)] == null) return true
        // A phrase with a lowercase word other than a name's connectives is a sentence or a label, not
        // a name ("Reviewers generally praise the food", "4.26 stars", "Administrative re-inspection").
        if (c.trim().split(Regex("\\s+")).any { w -> w.firstOrNull { it.isLetter() }?.isLowerCase() == true && w.lowercase().trim(',', '.', ';', '(', ')') !in NAME_CONNECTIVES }) return true
        // An address ("77 Varet Street, Brooklyn, NY 11206"), a violation code ("03A"), an inspection type.
        if (Regex("\\bNY\\s+\\d{5}\\b").containsMatchIn(c) || (Regex("^\\d+[A-Za-z]?\\s").containsMatchIn(c) && STREET_WORD.containsMatchIn(c))) return true
        if (Regex("^\\d{2}[A-O]\\d?$").matches(c.trim())) return true
        if (Regex("(?i)\\binspection\\b").containsMatchIn(c) && nameIndex[nameKey(c)] == null) return true
        // A month and day ("Nov 25"), a dish with its price ("Guinness Espresso Martini at $14").
        if (Regex("(?i)^$MONTH_RX\\s+\\d{1,2}(?:st|nd|rd|th)?$").matches(c.trim())) return true
        if (Regex("\\$\\s?\\d").containsMatchIn(c)) return true
        // A count with its lowercase noun ("0 critical violations", "2 inspections").
        if (Regex("^\\d+(?:\\s+[\\p{Ll}][\\p{L}'’/-]*)+$").matches(c.trim())) return true
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

    /** Where [r]'s names occur in [text]: runs of its name words, with only spaces or punctuation between them. */
    fun nameSpans(text: String, r: Rest): List<IntRange> = nameSpans(text, names(r))

    private fun nameSpans(text: String, names: List<String>): List<IntRange> {
        val ws = words(text)
        val out = ArrayList<IntRange>()
        for (n in names) {
            val key = normalizeName(n).filter { it !in NAME_STOPWORDS }
            if (key.isEmpty()) continue
            for (i in 0..ws.size - key.size) {
                // Between two name words: spaces, punctuation and stop words ("Taste of Heaven", "Art & Coffee").
                if ((key.indices).all { j -> ws[i + j].key == key[j] } &&
                    (0 until key.size - 1).all { j -> NAME_GAP.matches(STOP_WORD.replace(text.substring(ws[i + j].end, ws[i + j + 1].start).replace(Regex("['’]s\\b"), ""), " ")) }
                ) out += ws[i].start until ws[i + key.size - 1].end
            }
        }
        return out.sortedBy { it.first }
    }

    /**
     * Where each of [ids] is named in [text]; a span inside a longer span of another of them is
     * not counted ("Dunkin'" inside "Dunkin' Donuts/Baskin Robbins").
     */
    private fun ownSpans(text: String, ids: List<String>): Map<String, List<IntRange>> {
        val all = ids.associateWith { id -> byId[id]?.let { nameSpans(text, it) }.orEmpty() }
        return all.mapValues { (id, ss) ->
            ss.filter { sp -> all.none { (o, os) -> o != id && os.any { it.first <= sp.first && it.last >= sp.last && (it.last - it.first) > (sp.last - sp.first) } } }
        }
    }

    /** An answer prepared for reading numbers: dates, restaurant names and the literal "311" blanked out. */
    inner class Reading(val task: Task, val answer: String) {
        val masked: String = run {
            val sb = StringBuilder(withoutDates(answer))
            val spans = ArrayList<IntRange>()
            for (id in (task.restaurants + task.answerSet.orEmpty()).distinct()) byId[id]?.let { spans += nameSpans(answer, it) }
            spans += nameSpans(answer, task.facts.filter { it.kind == "name" }.flatMap { listOf(it.value) + it.alts })
            restaurantMentions(answer).forEach { spans += it.second until it.third }
            for (r in spans) for (i in r) sb.setCharAt(i, ' ')
            // "311" names the complaint line; it is a value only when the ground truth says so.
            if (task.facts.none { it.value == "311" }) Regex("\\b311\\b").findAll(sb.toString()).toList().forEach { m -> for (i in m.range) sb.setCharAt(i, ' ') }
            sb.toString()
        }
        val maskedLower: String = stripAccents(masked).lowercase()
    }

    /** Numbers written before [unit] ("7 critical violations", "4.2-star"): (value, start). */
    fun attached(masked: String, unit: String): List<Pair<Double, Int>> {
        val rx = Regex("(?i)$NUM_TOKEN((?:\\s+$GAP_WORD){0,3})[\\s-]*(?:${COUNT_UNITS.getValue(unit)})")
        return rx.findAll(masked).map { m ->
            val t = m.groupValues[1].lowercase()
            (t.toDoubleOrNull() ?: NUMBER_WORD_VALUES.getValue(t).toDouble()) to m.range.first
        }.toList()
    }

    /**
     * The numbers of [attached] that state the fact's own quantity. Left out: a count of reviews
     * per star level ("32 five-star"), the scale's maximum ("out of 5 stars"), a number on a line
     * dated with a date other than the period's bounds (one inspection's count in a breakdown:
     * "Nov 25, 2025: 4 critical"), and a complaint count of one complaint type ("no food poisoning
     * complaints") when the fact counts both types.
     */
    fun claims(rd: Reading, unit: String): List<Pair<Double, Int>> {
        val rx = Regex("(?i)$NUM_TOKEN((?:\\s+$GAP_WORD){0,3})[\\s-]*(?:${COUNT_UNITS.getValue(unit)})")
        val text = rd.answer
        val questionDates = dateMentions(rd.task.question).mapNotNull { it.date() }
        val bounds = setOf(cutoff, cutoff.minusDays(1), cutoff.minusYears(1), LocalDate.parse(rd.task.asOf)) + questionDates
        // The period a count is over: from the question's earliest date before the cutoff, else a year before it.
        val periodStart = questionDates.filter { it < cutoff }.minOrNull() ?: cutoff.minusYears(1)
        return rx.findAll(rd.masked).mapNotNull { m ->
            val t = m.groupValues[1].lowercase()
            val gap = m.groupValues[2].lowercase()
            val start = m.range.first
            val v = t.toDoubleOrNull() ?: NUMBER_WORD_VALUES.getValue(t).toDouble()
            val lineStart = text.lastIndexOf('\n', start - 1) + 1
            when {
                unit == "star" && Regex("(?i)(?:one|two|three|four|five|[1-5])[\\s-]*stars?$").containsMatchIn(m.value) && (gap.isNotBlank() || Regex("(?i)^\\d+\\s+(?:one|two|three|four|five)").containsMatchIn(m.value)) -> null
                unit == "star" && Regex("(?i)(?:out of|/|\\bto|\\bfrom)\\s*$").containsMatchIn(text.substring(maxOf(0, start - 8), start)) -> null
                // A star level ("5-star reviews", "1–2 star"), not an average.
                unit == "star" && Regex("^[1-5]$").matches(t) && Regex("^[1-5](?:\\s*[–-]\\s*[1-5])?-?\\s?stars?\\b").containsMatchIn(text.substring(start)) &&
                    !Regex("(?i)\\b(?:rated|rating|average|averages|avg)\\b[^.\\n]{0,12}$").containsMatchIn(text.substring(maxOf(0, start - 20), start)) -> null
                unit == "star" && Regex("[–-]\\s*$").containsMatchIn(text.substring(maxOf(0, start - 3), start)) && Regex("\\d\\s*[–-]\\s*$").containsMatchIn(text.substring(maxOf(0, start - 4), start)) -> null
                // A count in a sentence about an earlier inspection or complaint ("the most recent inspection
                // was May 8, 2025 ... one critical violation"); a sentence dated by a review is not one.
                (unit == "critical" || unit == "complaint") && segments(text).firstOrNull { start in it }?.let { seg ->
                    val st = text.substring(seg.first, seg.last + 1)
                    (if (unit == "critical") Regex("(?i)inspect|\\bcited\\b") else Regex("(?i)complain|\\bfiled\\b|\\breported\\b")).containsMatchIn(st) &&
                        !Regex("(?i)\\breview").containsMatchIn(st) &&
                        dateMentions(st).any { d -> d.date()?.let { it < periodStart } == true }
                } == true -> null
                (unit == "critical" || unit == "complaint" || unit == "point") && breakdownLine(text, lineStart, start, bounds) -> null
                unit == "complaint" && Regex("poisoning|establishment").containsMatchIn(gap) -> null
                else -> v to start
            }
        }.toList()
    }

    /**
     * True when the number at [at] is one entry of a dated breakdown: its line is a list item, or
     * a colon or dash follows the date, and a date other than the period's bounds comes before it
     * on the line ("- 2026-03-25: 2 critical violations", "Nov 25, 2025 (score 45): 4 critical").
     */
    private fun breakdownLine(text: String, lineStart: Int, at: Int, bounds: Set<LocalDate>): Boolean {
        val head = text.substring(lineStart, at)
        val dates = dateMentions(head).filter { d -> d.date()?.let { it !in bounds } ?: true }
        if (dates.isEmpty()) return false
        val listItem = Regex("^\\s*(?:[-*•]|\\d+[.)])\\s").containsMatchIn(head)
        return listItem || dates.any { d -> Regex("[:—–]").containsMatchIn(head.substring(d.end)) }
    }

    /** True when the number at [start]..[end] of [t] is one of several values offered ("1, 2 or 3", "about 4"). */
    private fun hedged(t: String, start: Int, end: Int): Boolean {
        val before = t.substring(maxOf(0, start - 30), start)
        val after = t.substring(end, minOf(t.length, end + 20))
        return Regex("(?i)(?:\\d\\s*(?:,|or|to|-|–|/)\\s*\\$?\\s?$|\\b(?:about|around|roughly|approximately|approx\\.?|nearly|almost|at least|at most|up to|more than|fewer than|less than|over|under|between|maybe|perhaps|possibly|probably|likely|some|~)\\s*$)").containsMatchIn(before) ||
            Regex("(?i)^\\s*(?:(?:,|or|to|-|–|/)\\s*\\$?\\s?\\d|\\+|or (?:more|so|fewer|less)|-ish)").containsMatchIn(after)
    }

    private fun groupValues(task: Task, f: Fact): List<Double> =
        task.facts.filter { it.kind == f.kind && it.required && it.alts.firstOrNull() == f.alts.firstOrNull() }.map { it.value.toDouble() }

    /** True when [f] is stated in [answer]. */
    fun factPresent(f: Fact, answer: String, task: Task? = null): Boolean =
        factProblem(f, Reading(task ?: Task("", "", "", "", "", emptyList(), listOf(f), null, emptyList(), "", true, 0, cutoff.toString()), answer)) == null

    /** Null when [f] is stated, without negation, hedging or a contradicting value; otherwise why not. */
    fun factProblem(f: Fact, rd: Reading): String? {
        val answer = rd.answer
        val task = rd.task
        val lower = stripAccents(answer).lowercase()
        return when (f.kind) {
            "name" -> {
                val norm = " " + normalizeName(answer).joinToString(" ") + " "
                // Other names made only of generic words ("Kitchen") do not count; the main name always does.
                val ok = (listOf(f.value) + f.alts.filter { a -> normalizeName(a).any { it !in GENERIC_WORDS && it !in COMMON_WORDS } })
                    .any { n -> nameKey(n).let { k -> k.isNotEmpty() && norm.contains(" $k ") } }
                if (ok) null else "not stated"
            }
            "grade" -> {
                val hits = gradeMentions(answer).filter { it.letter == f.value }
                if (hits.any { !negated(answer, it.start) && !hedgedClause(answer, it.start) }) null
                else if (hits.any { !negated(answer, it.start) }) "only hedged" else if (hits.isNotEmpty()) "only negated" else "not stated"
            }
            "date" -> {
                val d = LocalDate.parse(f.value.take(10))
                val hits = dateMentions(answer).filter { it.date() == d }
                if (hits.any { !negated(answer, it.start) && !hedgedClause(answer, it.start) }) null
                else if (hits.any { !negated(answer, it.start) }) "only hedged" else if (hits.isNotEmpty()) "only negated" else "not stated"
            }
            "count" -> {
                val v = f.value.toInt()
                val unit = f.alts.firstOrNull()
                if (unit != null && f.required) {
                    val ok = groupValues(task, f)
                    val bad = claims(rd, unit).map { it.first }.filter { x -> ok.none { it == x } && !(x == 311.0 && unit == "complaint") }
                    if (bad.isNotEmpty()) return "the answer also states ${bad.joinToString { fmtNum(it) }}"
                }
                val t = rd.masked
                val digits = Regex("(?<![\\d.,$/:-])\\d{1,6}(?![\\d/]|[.,]\\d|-\\d)").findAll(t)
                    .filter { it.value.toInt() == v && !hedged(t, it.range.first, it.range.last + 1) }.any()
                val ok = digits || NUMBER_WORDS[v]?.let { w -> Regex("\\b$w\\b").findAll(rd.maskedLower).any { !hedged(rd.maskedLower, it.range.first, it.range.last + 1) } } == true ||
                    (v == 0 && Regex("\\b(no|none|zero|never)\\b").containsMatchIn(rd.maskedLower)) ||
                    (v == 1 && Regex("\\b(once|a single|one)\\b").containsMatchIn(rd.maskedLower))
                if (ok) null else "not stated"
            }
            "number" -> {
                val v = f.value.toDouble()
                val tol = f.tol ?: 1e-9
                val unit = f.alts.firstOrNull()
                if (unit != null && f.required) {
                    val ok = groupValues(task, f)
                    val bad = claims(rd, unit).map { it.first }.filter { x -> ok.none { Math.abs(it - x) <= tol + 1e-9 } }
                    if (bad.isNotEmpty()) return "the answer also states ${bad.joinToString { fmtNum(it) }}"
                }
                val integral = v == Math.rint(v)
                val ok = Regex("(?<![\\d.])\\d+(?:\\.\\d+)?(?![\\d.]*\\d)").findAll(rd.masked).any {
                    // 3.96 is "3.96" or "4.0", not the "4" of "4 critical violations".
                    (integral || it.value.contains('.')) && Math.abs(it.value.toDouble() - v) <= tol + 1e-9 &&
                        !hedged(rd.masked, it.range.first, it.range.last + 1)
                }
                if (ok) null else "not stated"
            }
            "money" -> {
                val v = f.value.toDouble()
                val amounts = moneyMentions(answer)
                val dish = f.alts.firstOrNull()?.let { d -> Regex("(?i)" + Regex.escape(d).replace(" ", "\\E\\s+\\Q")).findAll(answer).toList() }.orEmpty()
                if (dish.isNotEmpty()) {
                    // The price given for the dish: the first after its name in the sentence, else the last before it.
                    val segs = segments(answer)
                    // (A dish name may hold ". ": the sentence after it is the one its last character is in.)
                    val given = dish.mapNotNull { m ->
                        val end = segs.firstOrNull { m.range.last in it }
                        val start = segs.firstOrNull { m.range.first in it }
                        amounts.firstOrNull { it.second > m.range.last && end != null && it.second in end }
                            ?: amounts.lastOrNull { it.second < m.range.first && start != null && it.second in start }
                    }
                    val wrong = given.filter { Math.abs(it.first - v) >= 0.005 }
                    if (wrong.isNotEmpty()) return "the answer gives the dish \$${"%.2f".format(java.util.Locale.US, wrong.first().first)}"
                    // "$9.95 or $10.95", "about $10", "probably $9.95": one of several prices offered, or a guess.
                    if (given.any { g -> moneyHedged(answer, g.second) }) return "only hedged"
                    // Negation is read before the dish's name, which may hold "Didn't" itself.
                    if (given.isNotEmpty()) return if (dish.any { !negated(answer, it.range.first) }) null else "only negated"
                }
                if (amounts.any { Math.abs(it.first - v) < 0.005 && !negated(answer, it.second) && !moneyHedged(answer, it.second) }) null else "not stated"
            }
            "violation" -> {
                val ok = Regex("(?i)\\b${Regex.escape(f.value)}\\b").containsMatchIn(answer) || run {
                    val ws = Regex("[a-z]+").findAll(lower).map { it.value }.toList()
                    // The rarest word (first) on its own, or two of the words.
                    fun hit(k: String) = ws.any { w -> w.startsWith(k.take(5)) && (k.length >= 5 || w == k) }
                    f.alts.isNotEmpty() && (hit(f.alts[0]) || f.alts.count(::hit) >= 2)
                }
                if (ok) null else "not stated"
            }
            "keyword" -> {
                val hits = (listOf(f.value) + f.alts).flatMap { k -> Regex("\\b" + Regex.escape(k.lowercase())).findAll(lower).toList() }
                if (hits.any { !negated(lower, it.range.first) }) null else if (hits.isNotEmpty()) "only negated" else "not stated"
            }
            "topic" -> {
                // Any alt, every word of it as a stem ("rodents/insects/garbage": "rodent"; "letter grading": "letter grade").
                val hit = f.alts.map { a -> a.split(' ').filter { it.isNotEmpty() } }.filter { it.isNotEmpty() }.any { ws ->
                    ws.all { w -> Regex(stemPattern(w)).findAll(lower).any { !negated(lower, it.range.first) } }
                }
                if (hit) null else "not stated"
            }
            "refusal" -> if ((if (f.value == "out_of_scope") OUT_OF_SCOPE else NOT_FOUND).containsMatchIn(answer)) null else "not stated"
            "winner" -> winnerProblem(f, task, answer)
            // A claim the answer must not make; a negated one ("I can't say it is open") is not a claim.
            "absent" -> {
                // For a complaint's topic, a sentence about another, earlier complaint may name other topics.
                val complaintTopic = f.value == COMPLAINT_POISONING_ABSENT || f.value == COMPLAINT_NOT_POISONING_ABSENT
                val factDates = task.facts.filter { it.kind == "date" }.map { LocalDate.parse(it.value.take(10)) }.toSet()
                Regex(f.value).findAll(answer).firstOrNull { m ->
                    !negated(answer, m.range.first) && !(complaintTopic && segments(answer).firstOrNull { m.range.first in it }?.let { seg ->
                        val t = answer.substring(seg.first, seg.last + 1)
                        OTHER_COMPLAINT.containsMatchIn(t) || dateMentions(t).any { d -> d.date()?.let { it !in factDates } ?: true }
                    } == true)
                }?.let { "the answer says '${it.value}'" }
            }
            else -> "unknown fact kind"
        }
    }

    /** Prices written in [text]: (amount, start); with [dollarOnly], only amounts written with "$". */
    fun moneyMentions(text: String, dollarOnly: Boolean = false): List<Pair<Double, Int>> =
        Regex("\\$\\s?(\\d{1,4}(?:\\.\\d{1,2})?)" + if (dollarOnly) "" else "|(?<![\\d.])(\\d{1,4}\\.\\d{2})(?![\\d.])").findAll(withoutDates(text))
            .map { m -> (m.groups[1] ?: m.groups[2])!!.value.toDouble() to m.range.first }.toList()

    /**
     * Null when the restaurant the answer picks in a comparison is [f]'s: the restaurant named
     * nearest before a verdict ("did better", "rated higher", "had more"), or after it when none
     * is before, in the same sentence. A verdict for the other side ("did worse", "fewer") must
     * name the other restaurant. Statements of the scale ("lower scores are better") are skipped.
     */
    private fun winnerProblem(f: Fact, task: Task, answer: String): String? {
        val w = f.value
        val l = task.restaurants.firstOrNull { it != w } ?: return null
        val (winRx, loseRx) = WINNER_CUES[f.alts.firstOrNull()] ?: return null
        val text = SCALE_STATEMENT.replace(answer) { " ".repeat(it.value.length) }
        val spans = ownSpans(answer, listOf(w, l)).flatMap { (id, ss) -> ss.map { id to it } }
        val segs = segments(answer)
        for ((rx, expect) in listOf(winRx to w, loseRx to l)) for (m in rx.findAll(text)) {
            val seg = segs.firstOrNull { m.range.first in it } ?: continue
            val inSeg = spans.filter { it.second.last >= seg.first && it.second.first <= seg.last }
            val who = (inSeg.filter { it.second.last < m.range.first }.maxByOrNull { it.second.last }
                ?: inSeg.filter { it.second.first > m.range.last }.minByOrNull { it.second.first })?.first ?: continue
            if (who != expect) return "the answer says ${byId[who]?.name} ${m.value.trim()}"
        }
        return null
    }

    /**
     * Null when, in a comparison, each value is nearer its own restaurant's name than the other's
     * (summed over the values; the first statement of each value is read). A value in a markdown
     * table row or column, or on a line, that names only one of the two belongs to that one.
     */
    private fun pairingProblem(rd: Reading): String? {
        val owned = rd.task.facts.filter { it.of != null && it.required && it.kind in setOf("count", "number") }
        if (owned.map { it.of }.distinct().size != 2) return null
        val (a, b) = owned.map { it.of!! }.distinct()
        val spans = ownSpans(rd.answer, listOf(a, b))
        if (spans.values.any { it.isEmpty() }) return null
        fun dist(p: Int, id: String) = spans.getValue(id).minOf { r -> if (p < r.first) r.first - p else if (p > r.last) p - r.last else 0 }
        val text = rd.answer
        fun named(from: Int, to: Int) = listOf(a, b).filter { id -> spans.getValue(id).any { it.first >= from && it.last < to } }
        // The restaurant a value at [p] belongs to by layout: a markdown table's row or column, or the
        // only one of the two named on the value's line; null when the layout does not say.
        fun owner(p: Int): String? {
            val ls = text.lastIndexOf('\n', p - 1) + 1
            val le = text.indexOf('\n', p).let { if (it < 0) text.length else it }
            if (text.substring(ls, le).trimStart().startsWith("|")) {
                val inRow = named(ls, le)
                if (inRow.size == 1) return inRow[0]
                // A column per restaurant: the header row's cell above the value.
                var hs = ls
                while (hs > 0) {
                    val prev = text.lastIndexOf('\n', hs - 2) + 1
                    if (!text.substring(prev, hs - 1).trimStart().startsWith("|")) break
                    hs = prev
                }
                val col = text.substring(ls, p).count { it == '|' }
                val he = text.indexOf('\n', hs).let { if (it < 0) text.length else it }
                var c = 0
                var cs = hs
                for (i in hs until he) if (text[i] == '|') { if (c == col) return named(cs, i).singleOrNull(); c++; cs = i + 1 }
                return null
            }
            return named(ls, le).singleOrNull()
        }
        var right = 0
        var swapped = 0
        for (f in owned) {
            val v = f.value.toDouble()
            val tol = f.tol ?: 1e-9
            val integral = v == Math.rint(v)
            val p = Regex("(?<![\\d.,$/:-])\\d+(?:\\.\\d+)?(?![\\d.]*\\d)").findAll(rd.masked).firstOrNull {
                (f.kind == "count" || integral || it.value.contains('.')) && Math.abs(it.value.toDouble() - v) <= tol + 1e-9
            }?.range?.first ?: return null
            // Distance to the value's own name and to the other's; a layout owner decides outright.
            val other = if (f.of == a) b else a
            when (owner(p)) {
                f.of -> swapped += LAYOUT_DISTANCE
                other -> right += LAYOUT_DISTANCE
                else -> { right += dist(p, f.of!!); swapped += dist(p, other) }
            }
        }
        return if (swapped < right) "the values are given to the wrong restaurants" else null
    }

    /**
     * True when [d] is one end of a range whose other end is also a period bound ("(Nov 13, 2025 –
     * Nov 13, 2026)", "2025-11-13 to 2026-11-13").
     */
    private fun boundRange(answer: String, d: DateMention, bounds: Set<LocalDate>): Boolean {
        val all = dateMentions(answer)
        val i = all.indexOfFirst { it.start == d.start }
        val sep = Regex("^\\s*(?:[–—-]|to|through|until)\\s*$")
        val next = all.getOrNull(i + 1)
        val prev = all.getOrNull(i - 1)
        return (next != null && next.date() in bounds && sep.matches(answer.substring(d.end, next.start))) ||
            (prev != null && prev.date() in bounds && sep.matches(answer.substring(prev.end, d.start)))
    }

    private fun withoutDates(t: String): String {
        val sb = StringBuilder(t)
        for (d in dateMentions(t).reversed()) sb.replace(d.start, d.end, " ".repeat(d.end - d.start))
        return sb.toString()
    }

    private val codeCache = HashMap<String, Set<String>>()
    /** Every violation code of the restaurant's inspections. */
    fun violationCodesOf(id: String): Set<String> = codeCache.getOrPut(id) {
        byId[id]?.camis?.let { c -> s.query("SELECT DISTINCT violation_code FROM inspections WHERE camis=? AND violation_code IS NOT NULL", c).map { it[0] as String }.toSet() }.orEmpty()
    }

    /** The violation texts of the restaurant's inspection on [date]. */
    fun violationTextsOn(id: String, date: String): String = byId[id]?.camis?.let { c ->
        s.query("SELECT violation_description FROM inspections WHERE camis=? AND substr(inspection_date, 1, 10)=? AND violation_description IS NOT NULL", c, date)
            .joinToString(" ") { it[0] as String }
    }.orEmpty()

    private val menuCache = HashMap<String, Set<Double>>()
    /** Every amount written on [id]'s archived menus. */
    fun menuAmounts(id: String): Set<Double> = menuCache.getOrPut(id) {
        s.query("SELECT text FROM menus WHERE restaurant_id=? AND text IS NOT NULL", id).flatMap { row ->
            Regex("(?<![\\d.])\\d{1,4}(?:\\.\\d{1,2})?(?![\\d])").findAll(row[0] as String).map { it.value.toDouble() }.toList()
        }.toSet()
    }

    private val menuTextCache = HashMap<String, String>()
    private fun menuWords(id: String): String = menuTextCache.getOrPut(id) {
        " " + s.query("SELECT text FROM menus WHERE restaurant_id=? AND text IS NOT NULL", id).joinToString(" ") { normalizeName(it[0] as String).joinToString(" ") } + " "
    }
    /** Problems with the trajectory's calls: unknown tools, arguments that are not a JSON object or break the schema. */
    fun invalidCalls(calls: List<ToolCall>): List<String> = calls.mapIndexedNotNull { i, c ->
        val def = TOOL_BY_NAME[c.name] ?: return@mapIndexedNotNull "call ${i + 1}: unknown tool '${c.name}'"
        val args = runCatching { parseJson(c.arguments.ifBlank { "{}" }) }.getOrNull() as? JsonObject
            ?: return@mapIndexedNotNull "call ${i + 1} ${c.name}: arguments are not a JSON object"
        val errs = validate(def.schema, args)
        if (errs.isEmpty()) null else "call ${i + 1} ${c.name}: " + errs.joinToString("; ")
    }

    fun verify(task: Task, answer0: String, calls: List<ToolCall>): Verdict {
        // Non-breaking and other Unicode hyphens ("2026‑02‑24") and no-break spaces read as their ASCII forms.
        val answer = normalizeAnswerText(answer0)
        val invalid = invalidCalls(calls)
        if (!task.offline) return Verdict(task.id, false, emptyList(), emptyList(), emptyList(), invalid, 0, 0, calls.size)
        val required = task.facts.filter { it.required }
        val rd = Reading(task, answer)
        val missing = required.mapNotNull { f -> factProblem(f, rd)?.let { why -> if (why == "not stated") f.toString() else "$f: $why" } }.toMutableList()
        pairingProblem(rd)?.let { missing += it }

        val invented = ArrayList<String>()
        val subjects = (task.restaurants + task.answerSet.orEmpty()).distinct()
        val questionKeys = nameCandidates(task.question).map { nameKey(it) }.toSet() + nameKey(task.question)

        // Restaurants: names in restaurant positions must be snapshot restaurants (or named in the question).
        for (c in nameCandidates(answer)) {
            if (isNonName(c) || isKnownName(c)) continue
            val k = nameKey(c)
            if (k in questionKeys || (k.isNotEmpty() && " ${nameKey(task.question)} ".contains(" $k "))) continue
            // A dish on the archived menu of a restaurant of the task ("Gnarly Eats Burger").
            if (k.isNotEmpty() && subjects.any { menuWords(it).contains(" $k ") }) continue
            invented += "restaurant '$c' is not in the snapshot"
        }
        // Prices: every price stated is written on the archived menu of a restaurant of the task.
        val menuIds = subjects.filter { menuAmounts(it).isNotEmpty() }
        if (menuIds.isNotEmpty()) for ((amount, _) in moneyMentions(answer, dollarOnly = true)) {
            if (menuIds.none { id -> menuAmounts(id).any { Math.abs(it - amount) < 0.005 } }) {
                invented += "price \$${"%.2f".format(java.util.Locale.US, amount)} is not on the archived menu"
            }
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
            // A negated grade ("is not graded B any more") is not a claim.
            for (g in gradeMentions(answer).filter { it.start in seg && !negated(answer, it.start) }) {
                if (owners.none { g.letter in (if (past) gradesOf(it) else setOfNotNull(byId[it]?.grade)) }) {
                    val who = owners.mapNotNull { byId[it]?.name }.distinct().take(3)
                    invented += "grade ${g.letter} " + if (who.isEmpty()) "with no restaurant that holds it"
                    else "is not ${if (past) "a" else "the current"} grade of ${who.joinToString(" / ")}"
                }
            }
        }

        // Dates: a date of the snapshot for the restaurants concerned, of the question or facts, or the
        // cutoff. The cutoff and the dates computed from it (the day before, a year before) are no event
        // of a restaurant: they are allowed only as the bound of a period ("as of", "since", "up to").
        val allowed = HashSet<LocalDate>()
        val bounds = setOf(cutoff, cutoff.minusDays(1), cutoff.minusYears(1), LocalDate.parse(task.asOf))
        dateMentions(task.question).mapNotNull { it.date() }.filter { it !in bounds }.forEach { allowed += it }
        task.facts.filter { it.kind == "date" }.forEach { allowed += LocalDate.parse(it.value.take(10)) }
        for (id in (subjects + mentions.flatMap { it.first } + ids.mapNotNull { it.first }).distinct()) allowed += datesOf(id)
        val questionMonths = dateMentions(task.question).map { it.year to it.month }.toSet()
        for (d in dateMentions(answer)) {
            val day = d.date()
            val ok = when {
                day == null -> (allowed + bounds).any { it.year == d.year && it.monthValue == d.month } || (d.year to d.month) in questionMonths
                day in allowed -> true
                day in bounds -> PERIOD_BOUND.containsMatchIn(answer.substring(maxOf(0, d.start - 40), d.start)) || boundRange(answer, d, bounds)
                else -> false
            }
            if (!ok) invented += "date ${answer.substring(d.start, d.end)} is not a date of the snapshot for these restaurants" +
                if (day in bounds) " (the cutoff is only a bound: 'as of', 'since', 'up to')" else ""
        }

        // Violation codes: each one is a code the restaurants were cited for; in a question about the
        // latest inspection, a code of that inspection unless its sentence dates it or puts it in the past.
        val latest = if (task.kind == "violation") task.facts.filter { it.kind == "violation" }.map { it.value }.toSet() else null
        val latestDate = task.facts.firstOrNull { it.kind == "date" }?.value?.take(10)
        for (m in VIOLATION_CODE.findAll(answer)) {
            val code = m.groupValues[1]
            val seg = segments(answer).firstOrNull { m.range.first in it }?.let { answer.substring(it.first, it.last + 1) } ?: ""
            val dated = EARLIER_INSPECTION.containsMatchIn(seg) || dateMentions(seg).any { it.date()?.toString() != latestDate }
            val ok = if (latest != null && !dated) code in latest else subjects.any { code in violationCodesOf(it) }
            if (!ok) invented += "violation $code is not ${if (latest != null && !dated) "a violation of the latest inspection" else "a violation of these restaurants"}"
        }
        // Pests: in a question about the latest inspection, a pest named (not negated) is in that inspection's violations.
        if (latest != null && latestDate != null) {
            val texts = subjects.joinToString(" ") { violationTextsOn(it, latestDate) }.lowercase()
            for (m in PEST_WORDS.findAll(answer)) {
                if (negated(answer, m.range.first)) continue
                val seg = segments(answer).firstOrNull { m.range.first in it }?.let { answer.substring(it.first, it.last + 1) } ?: ""
                if (EARLIER_INSPECTION.containsMatchIn(seg) || dateMentions(seg).any { it.date()?.toString() != latestDate }) continue
                val stem = m.value.lowercase().removePrefix("filth ").let { if (it.startsWith("mouse")) "mice" else it.take(4) }
                if (!texts.contains(stem)) invented += "'${m.value}' is not in the violations of the ${longDate(latestDate)} inspection"
            }
        }

        // List questions: every snapshot restaurant named must satisfy the question.
        val extra = ArrayList<String>()
        task.answerSet?.let { set ->
            val setNames = set.mapNotNull { byId[it] }.flatMap { names(it) }.map { nameKey(it) }.toSet()
            val loose by lazy { looseAnswerSet(task) }
            val looseNames by lazy { loose.mapNotNull { byId[it] }.flatMap { names(it) }.map { nameKey(it) }.toSet() }
            for ((idSet, a, b) in mentions) {
                if (idSet.any { it in set }) continue
                val k = nameKey(answer.substring(a, b))
                if (setNames.any { it.contains(k) }) continue
                if (excludedMention(answer, a)) continue
                if (idSet.any { it in loose } || looseNames.any { it.contains(k) }) continue
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

/** The distance a value's layout owner (table row or column, line) puts between it and the other restaurant. */
private const val LAYOUT_DISTANCE = 1_000_000

/** A sentence about another complaint than the latest one. */
private val OTHER_COMPLAINT = Regex("(?i)\\b(?:earlier|previous(?:ly)?|prior|older|before that|next most recent|second most recent|other complaints?|for context|also had|another complaint)\\b")

/** An earlier inspection, as opposed to the latest one ("at a previous inspection", "in 2023"). */
private val EARLIER_INSPECTION = Regex("(?i)\\b(?:previous(?:ly)?|earlier|prior|past|older|former) (?:inspections?|visits?|years?|cycles?)\\b|\\bin (?:19|20)\\d\\d\\b|\\bhistor(?:y|ically)\\b|\\bused to\\b|\\bformerly\\b|\\bpreviously\\b")

/** Words before a date that make it the bound of a period, not the date of an event. */
private val PERIOD_BOUND = Regex("(?i)\\b(?:as of|since|until|till|through|thru|up to|up until|before|after|to|from|between|by|cutoff|snapshot|data|records?|window|period|months?)\\s*[(:,]?\\s*$")

private val PAST_MARKER = Regex("(?i)\\b(was|were|had|previous(?:ly)?|earlier|before|prior|used to|formerly|history|historically|once|in (?:19|20)\\d\\d)\\b")

val NOT_FOUND = Regex(
    "(?i)(could ?n[o'’]t find|can ?n[o'’]t find|cannot find|did ?n[o'’]t find|unable to find|found no\\b|no (?:record|records|restaurant|place|match|matches|results?|listing|entry|data|information)\\b[^.]{0,40}\\b(?:for|of|about|called|named|matching|under|by)\\b|" +
        "not (?:found|listed|in (?:the|my|our|these) (?:data|records|snapshot|database|sources))|does ?n[o'’]t (?:appear|exist|show up)|no such|is ?n[o'’]t listed|" +
        "do ?n[o'’]t have any (?:record|data|information)|nothing (?:on|about|for) (?:it|that|a))",
)

/** [text] with Unicode hyphens (U+2010, U+2011) as '-' and no-break spaces (U+00A0, U+202F) as ' '. */
fun normalizeAnswerText(text: String): String =
    text.replace('\u2010', '-').replace('\u2011', '-').replace('\u00a0', ' ').replace('\u202f', ' ')

val OUT_OF_SCOPE = Regex(
    "(?i)(only (?:cover|covers|include|includes|has|have|hold|holds|contain|contains|know about|about|for|assist|assists|help|helps|answer|answers|handle|handles|work|works|provide|provides)\\b[^.]{0,140}brooklyn|outside (?:of )?brooklyn|not in brooklyn|" +
        "is ?n[o'’]t in brooklyn|brooklyn only|only brooklyn|limited to brooklyn|beyond brooklyn|restricted to brooklyn|not part of brooklyn)",
)

/** Units of count and number facts, as an answer writes them after the number. */
val COUNT_UNITS = mapOf(
    "point" to "points?\\b",
    "critical" to "critical\\b",
    "star" to "stars?\\b|/\\s*5\\b|out of 5\\b",
    "review" to "(?:reviews?|ratings?)\\b",
    "complaint" to "complaints?\\b",
)
private val NUMBER_WORD_VALUES: Map<String, Int> = NUMBER_WORDS.entries.associate { (k, v) -> v to k } + mapOf("no" to 0, "zero" to 0)
private const val NUM_TOKEN = "(?<![\\w.,$/:-])(\\d+(?:\\.\\d+)?|no|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
/** A word between a number and its unit ("3 311 food complaints"), not one that detaches them ("2 of them", "1 more"). */
private const val GAP_WORD = "(?:311|(?!(?:more|fewer|less|other|others|additional|extra|of|than|out|in|since|from|on|at|and|or|to|for|non|by|with|over|under|per)\\b)[a-z][a-z'’]*)"

/** "lower scores are better", "higher is worse": the scale of a comparison, not a verdict on a restaurant. */
private val SCALE_STATEMENT = Regex("(?i)\\b(?:lower|higher|fewer|more)\\b[^.;:\\n]{0,25}?\\b(?:is|are|means?)\\s+(?:better|worse|cleaner|good|bad)\\b")

/** Per comparison: verdicts that name the restaurant picked, and verdicts that name the other one. */
private val WINNER_CUES: Map<String, Pair<Regex, Regex>> = mapOf(
    "score" to (Regex("(?i)\\b(?:did better|does better|scored better|better (?:score|result|inspection)|performed better|came out ahead|comes out ahead|(?:the )?lower (?:inspection )?score|scored lower|beat|beats|won|wins|cleaner)\\b") to
        Regex("(?i)\\b(?:did worse|does worse|scored worse|worse (?:score|result|inspection)|(?:the )?higher (?:inspection )?score|scored higher|lost|loses)\\b")),
    "rating" to (Regex("(?i)\\b(?:rated higher|rates higher|higher[- ]rated|higher (?:google )?(?:rating|average|score)|better[- ]rated|better (?:rating|reviews)|more highly rated|came out ahead|comes out ahead|beat|beats|wins|won)\\b") to
        Regex("(?i)\\b(?:rated lower|rates lower|lower[- ]rated|lower (?:google )?(?:rating|average|score)|worse[- ]rated|worse (?:rating|reviews))\\b")),
    "critical" to (Regex("(?i)\\b(?:had more|has more|have more|more critical|cited for more|more violations|more often|worse record)\\b") to
        Regex("(?i)\\b(?:had fewer|has fewer|fewer|had less|has less|less critical|cleaner|better record)\\b")),
)

private val CLAUSE_END = Regex("(?i)[.;:!?\\n(),—–]|\\b(?:but|however|although|though|while|whereas|and)\\b")
private val NEGATION = Regex("(?i)\\b(?:not|never|no longer|cannot|without)\\b|n[’']t\\b")

/** True when the clause of [text] before [pos] negates it ("does not have grade A", "was not closed"). */
fun negated(text: String, pos: Int): Boolean {
    val before = text.substring(maxOf(0, pos - 80), pos)
    val start = CLAUSE_END.findAll(before).lastOrNull()?.let { it.range.last + 1 } ?: 0
    return NEGATION.containsMatchIn(before.substring(start))
}

private val NOT_WORD = "(?:n['’]t|not|cannot|no longer|never)"

/** Phrases that set a place named in a list answer apart from the answer: it does not qualify. */
private val EXCLUSION = Regex(
    listOf(
        "$NOT_WORD\\s+(?:\\w+\\s+){0,3}?(?:qualif\\w*|count\\w*|meet\\w*|match\\w*|fit\\w*|make the cut|satisf\\w*|eligible)",
        "\\b(?:not|won['’]t be|wouldn['’]t be|isn['’]t|aren['’]t|will not be|won['’]t|wouldn['’]t)\\s+(?:be\\s+)?open\\b",
        "\\b(?:closed|closes|will be closed)\\b",
        "\\bhours?\\b[^.\\n]{0,30}\\b(?:unknown|not (?:listed|available|recorded|known|confirmed)|unavailable|missing)\\b",
        "\\b(?:couldn['’]t|can['’]t|cannot|could not|unable to)\\s+(?:be\\s+)?confirm",
        "\\bunconfirmed\\b", "\\boutside\\b", "\\bnearby\\b", "\\bnot in \\w+(?: \\w+)? proper\\b",
        "\\bjust (?:west|east|north|south|beyond|across)\\b", "\\b(?:short|quick) (?:trip|walk|ride)\\b",
        "\\bwilling to (?:go|travel|wander|walk|venture)\\b", "\\bopen to (?:going|a short|nearby|those|travel)",
        "\\bin (?:a |the )?(?:different|another|neighbou?ring|adjacent) (?:neighbou?rhood|area)\\b", "\\bkm away\\b", "\\bmiles? away\\b",
        "\\b(?:listed|categori[sz]ed|classified|tagged|labell?ed|recorded)\\s+as\\b[^.\\n]{0,60}?,?\\s+not\\b",
        "\\bnot (?:a|an)\\s+(?:\\w+\\s+){0,3}?(?:spot|place|restaurant|shop|venue|option)s?\\b",
        "\\brather than\\b", "\\bnone of\\b", "\\bexclud\\w*\\b",
        "\\bdoes(?:n['’]t| not) (?:have|hold|carry) (?:an? |the )?(?:letter )?grade\\b", "\\bno (?:\\w+\\s+){0,2}grade\\b",
        "[,\"”]\\s*not\\s+(?:the\\s+|in\\s+)?[A-Z\"“]",
        "\\b(?:isn['’]t|aren['’]t|is not|are not)\\s+(?:an?\\s+)?(?:[\\w-]+\\s+){0,3}?(?:restaurants?|spots?|places?|shops?|venues?)\\b",
        "\\bnot (?:listed|categori[sz]ed|classified|tagged|labell?ed|recorded) as\\b",
        "\\b(?:over|across) the border\\b", "\\bno other\\b",
    ).joinToString("|") { "(?:$it)" },
    RegexOption.IGNORE_CASE,
)
private val LIST_ITEM = Regex("^\\s*(?:[-*•]|\\d+\\.)\\s")

/**
 * True when the place named at [pos] is set apart from the answer: its sentence (in a list item or
 * prose), or the line introducing its list or table, has an [EXCLUSION] phrase. A table row's own
 * cells are data ("no letter grade"), so only the line before the table counts for it.
 */
fun excludedMention(text: String, pos: Int): Boolean {
    val ls = text.lastIndexOf('\n', pos - 1) + 1
    val le = text.indexOf('\n', pos).let { if (it < 0) text.length else it }
    val line = text.substring(ls, le)
    val table = line.trimStart().startsWith("|")
    if (!table) {
        val p = pos - ls
        var st = 0
        var end = line.length
        for (m in Regex("(?<=[.!?])\\s+(?=[\\p{Lu}\\d*\"“(])").findAll(line)) {
            if (m.range.first < p) st = m.range.last + 1 else { end = m.range.first; break }
        }
        if (EXCLUSION.containsMatchIn(line.substring(st, end))) return true
    }
    if (table || LIST_ITEM.containsMatchIn(line)) {
        val lead = text.substring(0, ls).split('\n').dropLast(1).lastOrNull { l ->
            l.isNotBlank() && !LIST_ITEM.containsMatchIn(l) && !l.trimStart().startsWith("|")
        } ?: return false
        return EXCLUSION.containsMatchIn(lead)
    }
    return false
}

private val HEDGE = Regex(
    "(?i)\\b(?:might|could|possibly|perhaps|probably|maybe|likely|unlikely|presumably|guess|not sure|unsure|may (?:be|have|still|now|currently|also|well|hold|carry))\\b",
)

/** True when the clause of [text] before [pos] offers it as a guess ("probably has grade A", "it might be C"). */
fun hedgedClause(text: String, pos: Int): Boolean {
    val before = text.substring(maxOf(0, pos - 80), pos)
    val start = CLAUSE_END.findAll(before).lastOrNull()?.let { it.range.last + 1 } ?: 0
    return HEDGE.containsMatchIn(before.substring(start))
}

/** True when the price at [pos] ("$9.95") is offered among others ("$9.95 or $10.95", "$9-$11", "about $10") or as a guess. */
fun moneyHedged(text: String, pos: Int): Boolean {
    val m = Regex("\\$\\s?\\d{1,4}(?:\\.\\d{1,2})?|\\d{1,4}\\.\\d{2}").find(text, pos)?.takeIf { it.range.first == pos } ?: return hedgedClause(text, pos)
    val before = text.substring(maxOf(0, pos - 30), pos)
    val after = text.substring(m.range.last + 1, minOf(text.length, m.range.last + 21))
    return Regex("(?i)(?:\\d\\s*(?:,|or|to|-|–|/)\\s*$|\\b(?:about|around|roughly|approximately|approx\\.?|nearly|almost|at least|at most|up to|more than|less than|over|under|between|~)\\s*$)").containsMatchIn(before) ||
        Regex("(?i)^\\s*(?:(?:,|or|to|-|–|/)\\s*\\$?\\s?\\d|\\+|or (?:more|so|less)|-ish)").containsMatchIn(after) ||
        hedgedClause(text, pos)
}

/** A word of a topic as a regex: its first 4 letters (6 for long words) as a word start, or the whole short word. */
fun stemPattern(w: String): String = when {
    w.length >= 10 -> "\\b" + Regex.escape(w.take(6))
    w.length >= 5 -> "\\b" + Regex.escape(w.take(4))
    else -> "\\b" + Regex.escape(w) + "s?\\b"
}

/** Words of a 311 descriptor that say what it is about ("Rodents/Insects/Garbage" -> rodents | insects | garbage). */
fun topicAlts(descriptor: String): List<String> = descriptor.lowercase().split(Regex("/|,|\\bor\\b")).map { piece ->
    Regex("[a-z]+").findAll(piece).map { it.value }.filter { it.length >= 3 && it !in TOPIC_STOP }.joinToString(" ")
}.filter { it.isNotEmpty() }.distinct()

private val TOPIC_STOP = setOf("food", "and", "the", "with", "for", "contact", "area", "more")

/** Pests an answer may claim inspectors found; a violation answer names one only when the inspection's violations do. */
val PEST_WORDS = Regex("(?i)\\b(?:mice|mouse|rats?|rodents?|roach(?:es)?|cockroach(?:es)?|(?:filth )?flies|vermin|insects?)\\b")

/** DOHMH violation codes ("04L", "10F") written in an answer. */
val VIOLATION_CODE = Regex("\\b(\\d{2}[A-O]\\d?)\\b")

private val STOP_WORD = Regex("(?i)\\b(?:" + NAME_STOPWORDS.joinToString("|") + ")\\b|&")
private val NAME_GAP = Regex("^[^\\n\\p{L}\\p{N}]{0,8}$")
