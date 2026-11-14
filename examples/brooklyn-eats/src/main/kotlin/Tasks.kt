package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.parseJson
import java.security.MessageDigest
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.TextStyle
import java.util.Locale
import kotlin.random.Random

// Questions with ground truth, generated from the snapshot through the snapshot-mode tools, so
// every fact of a ground truth is in the output of the tool calls recorded as its solution.
// Dates are relative to the manifest cutoff. Paraphrases come from seeded templates. Tasks are
// split by restaurant: each restaurant is in one split (a seeded hash of its id), and a task is
// kept only when all the restaurants of its ground truth are in the same split.

enum class Split { TRAIN, DEV, TEST;
    val label: String get() = name.lowercase()
    companion object { fun parse(s: String) = entries.first { it.label == s } }
}

class ToolCall(val name: String, val arguments: String) {
    fun json(): Map<String, Any?> = linkedMapOf("name" to name, "arguments" to RawJson(arguments))
    companion object {
        fun of(name: String, args: Map<String, Any?>) = ToolCall(name, toJson(args))
        fun parse(o: JsonObject): ToolCall = ToolCall(
            (o["name"] as? JsonString)?.value ?: "",
            when (val a = o["arguments"]) { is JsonString -> a.value; null -> "{}"; else -> toJson(a) },
        )
    }
}

/**
 * One fact a correct answer states. Kinds: name (value or any alt), grade (letter), date
 * (YYYY-MM-DD), count (integer), number (decimal within [tol]), money (price), violation (code,
 * or at least two of the alts as keywords), keyword (any alt), refusal (not_found | out_of_scope).
 */
class Fact(val kind: String, val value: String, val alts: List<String> = emptyList(), val required: Boolean = true, val tol: Double? = null) {
    fun json(): Map<String, Any?> = linkedMapOf(
        "kind" to kind, "value" to value, "alts" to alts.ifEmpty { null },
        "required" to if (required) null else false, "tol" to tol,
    )
    override fun toString() = "$kind=$value" + if (alts.isEmpty()) "" else alts.joinToString("|", " (", ")")
    companion object {
        fun parse(o: JsonObject) = Fact(
            o.str("kind"), o.str("value"),
            (o["alts"] as? JsonArray)?.elements?.map { (it as JsonString).value }.orEmpty(),
            (o["required"] as? io.tlaloc.core.io.JsonBool)?.value ?: true,
            (o["tol"] as? JsonNumber)?.value,
        )
    }
}

class Task(
    val id: String,
    val split: String,
    val category: String,
    val kind: String,
    val question: String,
    /** The restaurants the ground truth is about; the split key. */
    val restaurants: List<String>,
    val facts: List<Fact>,
    /** For list questions: every restaurant that satisfies the question; naming another is an error. */
    val answerSet: List<String>?,
    /** Tool calls (snapshot mode) whose results hold every fact. */
    val solution: List<ToolCall>,
    val reference: String,
    /** False for live-only questions, which are not scored offline. */
    val offline: Boolean,
    val template: Int,
    val asOf: String,
) {
    fun json(): Map<String, Any?> = linkedMapOf(
        "id" to id, "split" to split, "category" to category, "kind" to kind, "question" to question,
        "restaurants" to restaurants, "facts" to facts.map { it.json() }, "answer_set" to answerSet,
        "solution" to solution.map { it.json() }, "reference" to reference, "offline" to offline,
        "template" to template, "as_of" to asOf,
    )
    fun line(): String = toJson(json())

    companion object {
        fun parse(line: String): Task {
            val o = parseJson(line) as JsonObject
            fun strs(k: String) = (o[k] as? JsonArray)?.elements?.map { (it as JsonString).value }
            return Task(
                o.str("id"), o.str("split"), o.str("category"), o.str("kind"), o.str("question"),
                strs("restaurants").orEmpty(), o.arr("facts").elements.map { Fact.parse(it as JsonObject) }, strs("answer_set"),
                o.arr("solution").elements.map { ToolCall.parse(it as JsonObject) }, o.str("reference"),
                (o["offline"] as io.tlaloc.core.io.JsonBool).value, (o["template"] as JsonNumber).value.toInt(), o.str("as_of"),
            )
        }
    }
}

/** A task kind and its share (percent) of the offline tasks of a split. */
class TaskKind(val category: String, val kind: String, val weight: Int)

val TASK_KINDS = listOf(
    TaskKind("lookup", "grade", 12),
    TaskKind("lookup", "last_inspection", 8),
    TaskKind("lookup", "violation", 8),
    TaskKind("filter", "cuisine_area_grade", 12),
    TaskKind("filter", "open_at", 6),
    TaskKind("comparison", "compare", 10),
    TaskKind("multihop", "critical_and_reviews", 12),
    TaskKind("complaints", "complaints", 10),
    TaskKind("menu", "price", 8),
    TaskKind("unanswerable", "outside_brooklyn", 4),
    TaskKind("unanswerable", "nonexistent", 6),
    TaskKind("unanswerable", "closed", 4),
    TaskKind("live", "live_only", 0),
)

/** Offline task totals and live-only counts per split. */
class TaskSizes(val offline: Map<Split, Int>, val live: Map<Split, Int>) {
    companion object {
        val FULL = TaskSizes(mapOf(Split.TRAIN to 4000, Split.DEV to 400, Split.TEST to 400), mapOf(Split.TRAIN to 40, Split.DEV to 8, Split.TEST to 8))
        val QUICK = TaskSizes(mapOf(Split.TRAIN to 300, Split.DEV to 40, Split.TEST to 40), mapOf(Split.TRAIN to 6, Split.DEV to 3, Split.TEST to 3))
    }

    /** Per kind and split: total * weight / 100, rounded by largest remainder so the kinds sum to the total. */
    fun quotas(): Map<String, Map<Split, Int>> {
        val out = TASK_KINDS.associate { it.kind to LinkedHashMap<Split, Int>() }
        for (sp in Split.entries) {
            val total = offline.getValue(sp)
            val ks = TASK_KINDS.filter { it.weight > 0 }
            val exact = ks.map { total * it.weight / 100.0 }
            val base = exact.map { Math.floor(it).toInt() }.toMutableList()
            var left = total - base.sum()
            for (i in exact.indices.sortedWith(compareByDescending<Int> { exact[it] - base[it] }.thenBy { it })) {
                if (left == 0) break
                base[i]++; left--
            }
            ks.forEachIndexed { i, k -> out.getValue(k.kind)[sp] = base[i] }
            out.getValue("live_only")[sp] = live.getValue(sp)
        }
        return out
    }
}

/** Fraction of restaurants in train and dev; the rest are test. */
const val SPLIT_TRAIN = 0.8
const val SPLIT_DEV = 0.1

/** The split of restaurant (or task key) [key] under [seed]: SHA-256 of seed and key, as a number in [0, 1). */
fun splitOf(key: String, seed: Long): Split {
    val d = MessageDigest.getInstance("SHA-256").digest("$seed\u0000$key".toByteArray(Charsets.UTF_8))
    var x = 0L
    for (i in 0 until 7) x = (x shl 8) or (d[i].toLong() and 0xff)
    val u = x.toDouble() / (1L shl 56).toDouble()
    return when {
        u < SPLIT_TRAIN -> Split.TRAIN
        u < SPLIT_TRAIN + SPLIT_DEV -> Split.DEV
        else -> Split.TEST
    }
}

fun nameKey(s: String): String = normalizeName(s).joinToString(" ")

private val MONTH_FMT = Locale.US
fun longDate(d: LocalDate): String = "${d.month.getDisplayName(TextStyle.FULL, MONTH_FMT)} ${d.dayOfMonth}, ${d.year}"
fun longDate(s: String): String = longDate(LocalDate.parse(s.take(10)))
private fun weekdayDate(d: LocalDate) = "${d.dayOfWeek.getDisplayName(TextStyle.FULL, MONTH_FMT)}, ${longDate(d)}"
private fun clock(h: Int, m: Int): String = "${if (h % 12 == 0) 12 else h % 12}:%02d %s".format(m, if (h < 12) "AM" else "PM")
private fun plural(n: Int, one: String, many: String = one + "s") = if (n == 1) "1 $one" else "$n $many"

private fun JsonObject.n(k: String): Double? = (this[k] as? JsonNumber)?.value
private fun JsonObject.list(k: String): List<JsonObject> = (this[k] as? JsonArray)?.elements?.map { it as JsonObject }.orEmpty()

/** Cuisine argument for search_restaurants -> how a question names it. */
val TASK_CUISINES = linkedMapOf(
    "thai" to "Thai", "mexican" to "Mexican", "chinese" to "Chinese", "japanese" to "Japanese", "italian" to "Italian",
    "pizza" to "pizza", "coffee" to "coffee", "caribbean" to "Caribbean", "korean" to "Korean", "indian" to "Indian",
    "bakery" to "bakery", "sushi" to "sushi", "bbq" to "barbecue", "burgers" to "burger", "vegan" to "vegan",
    "middle eastern" to "Middle Eastern", "mediterranean" to "Mediterranean", "french" to "French", "jamaican" to "Jamaican",
    "bagels" to "bagel", "donuts" to "doughnut", "ice cream" to "ice cream", "tacos" to "taco", "ramen" to "ramen",
    "halal" to "halal", "seafood" to "seafood", "greek" to "Greek", "vietnamese" to "Vietnamese", "spanish" to "Spanish",
    "latin american" to "Latin American", "chicken" to "chicken", "sandwiches" to "sandwich", "juice" to "juice bar",
    "american" to "American", "polish" to "Polish", "russian" to "Russian", "peruvian" to "Peruvian", "tea" to "tea",
    "kosher" to "kosher", "soul food" to "soul food", "african" to "African", "pakistani" to "Pakistani",
)

private val OUTSIDE_PLACES = listOf(
    "Astoria, Queens", "Long Island City, Queens", "Flushing, Queens", "Jackson Heights, Queens", "Harlem, Manhattan",
    "the East Village, Manhattan", "the Upper West Side, Manhattan", "Chelsea, Manhattan", "Mott Haven in the Bronx",
    "Arthur Avenue in the Bronx", "St. George, Staten Island", "Hoboken, New Jersey", "Jersey City, New Jersey",
)

private val FAKE_FIRST = listOf("Golden", "Silver", "Crimson", "Lucky", "Velvet", "Copper", "Midnight", "Hidden", "Jade", "Iron", "Paper", "Amber", "Saffron", "Cobalt", "Marble", "Ivory")
private val FAKE_SECOND = listOf("Lantern", "Heron", "Anchor", "Orchard", "Compass", "Willow", "Sparrow", "Thistle", "Juniper", "Pebble", "Falcon", "Meadow", "Ember", "Lighthouse", "Quill", "Badger")
private val FAKE_THIRD = listOf("Kitchen", "Bistro", "Pizzeria", "Cafe", "Diner", "Taqueria", "Noodle House", "Grill", "Bakery", "Tavern", "Sushi Bar", "Deli")

/** Words that say little about a violation (for its keywords). */
private val VIOLATION_STOP = setOf(
    "with", "from", "that", "this", "other", "used", "area", "areas", "food", "item", "items", "establishment", "properly",
    "improperly", "provided", "required", "not", "non", "contact", "evidence", "into", "when", "than", "then", "their",
    "there", "where", "which", "have", "been", "being", "shall", "must", "after", "before", "under", "above", "least",
    "present", "found", "observed", "maintained", "condition", "conditions", "following", "facility", "surface", "surfaces",
)

/** Distinctive words of a violation text: the 4 rarest (by document frequency over every distinct violation text). */
class ViolationWords(texts: List<String>) {
    private val df: Map<String, Int> = texts.flatMap { words(it).distinct() }.groupingBy { it }.eachCount()
    fun words(t: String): List<String> = Regex("[a-z]+").findAll(t.lowercase()).map { it.value }
        .filter { it.length >= 4 && it !in VIOLATION_STOP }.toList()
    /**
     * Up to 6 content words from the first 15 words of the text (its gist), the rarest first: an
     * answer states the violation when it names the code, the rarest word, or two of the words.
     */
    fun keywords(t: String): List<String> {
        val head = Regex("[A-Za-z][A-Za-z'-]*").findAll(t).take(15).joinToString(" ") { it.value }
        val ws = words(head).distinct().take(6)
        val rarest = ws.minWithOrNull(compareBy<String> { df[it] ?: 0 }.thenBy { ws.indexOf(it) }) ?: return ws
        return listOf(rarest) + ws.filter { it != rarest }
    }
}

/** A candidate: the restaurants it is about (or a key, when none) and how to build it. */
class Draft(val restaurants: List<String>, val key: String? = null, val build: (Random) -> Task?)

/** Generates the tasks from a snapshot-mode [ToolServer]. */
class TaskGenerator(
    private val s: ToolServer,
    val cutoff: LocalDate,
    val seed: Long,
    val splitSeed: Long,
    val sizes: TaskSizes,
    private val log: (String) -> Unit = {},
) {
    private val rests = s.restaurants
    private val byId = rests.associateBy { it.id }
    private val nameCounts = rests.groupingBy { nameKey(it.name) }.eachCount()
    private val restSplit = HashMap<String, Split>()
    fun restaurantSplit(id: String): Split = restSplit.getOrPut(id) { splitOf(id, splitSeed) }

    private fun call(name: String, args: Map<String, Any?>): Pair<ToolCall, JsonObject> {
        val c = ToolCall.of(name, args)
        val o = parseJson(s.call(name, c.arguments)) as JsonObject
        check(o["error"] == null) { "$name ${c.arguments}: ${o["error"]}" }
        return c to o
    }

    /** How a question names [r] and the search that finds it (top 10), or null when no simple search does. */
    private class Subject(val phrase: String, val search: ToolCall)

    private val subjects = HashMap<String, Subject?>()
    private fun subject(r: Rest): Subject? = subjects.getOrPut(r.id) {
        if (r.name.isBlank() || nameKey(r.name).isEmpty()) return@getOrPut null
        val street = r.address?.substringBefore(", Brooklyn")?.replace(Regex("\\s+"), " ")?.trim()
        val tries = buildList {
            if (nameCounts[nameKey(r.name)] == 1) add(r.name to r.name)
            if (street != null && street.any { it.isDigit() }) add("${r.name} at $street" to "${r.name} $street")
        }
        // The search must find it, and no other restaurant of the same name (two permits at one address are ambiguous).
        tries.firstOrNull { (_, q) ->
            val hits = s.ftsIds(q, 10)
            r.id in hits && hits.count { id -> byId[id]?.let { nameKey(it.name) == nameKey(r.name) } == true } == 1
        }?.let { (phrase, q) ->
            Subject(phrase, ToolCall.of("search_restaurants", mapOf("query" to q)))
        }
    }

    private fun nameFact(r: Rest) = Fact("name", r.name, listOfNotNull(r.dba?.let(::titleCase), r.osmName, r.gName).distinct().filter { it != r.name })

    private fun task(
        category: String, kind: String, q: String, restaurants: List<String>, facts: List<Fact>, solution: List<ToolCall>,
        reference: String, template: Int, answerSet: List<String>? = null, offline: Boolean = true,
    ) = Task("", "", category, kind, q, restaurants, facts, answerSet, solution, reference, offline, template, cutoff.toString())

    private val withCamis = rests.filter { it.camis != null }
    private val areaPhrases: List<String> = s.neighborhoods.filter { it.type == "0" }
        .map { it.name.replace(Regex("\\s*\\([^)]*\\)"), "").trim() }.distinct().sorted()
        .filter { resolveNeighborhood(s.neighborhoods, it).isNotEmpty() }

    // ------------------------------------------------------------------ lookups

    private fun lookupGrade(): List<() -> Draft?> = withCamis.filter { it.grade in setOf("A", "B", "C") && it.gradeDate != null }.map { r ->
        {
            subject(r)?.let { sub ->
                Draft(listOf(r.id)) { rng ->
                    val (c, o) = call("get_restaurant", mapOf("id" to r.id))
                    val g = o.s("grade") ?: return@Draft null
                    val gd = o.s("grade_date") ?: return@Draft null
                    val t = rng.nextInt(5)
                    val p = sub.phrase
                    val q = when (t) {
                        0 -> "What is the current DOHMH letter grade of $p?"
                        1 -> "What health inspection grade does $p have right now?"
                        2 -> "Is $p graded A, B or C by the NYC health department?"
                        3 -> "What letter grade is posted at $p, and when was it given?"
                        else -> "What grade did $p get from its most recent graded inspection?"
                    }
                    task(
                        "lookup", "grade", q, listOf(r.id),
                        listOf(nameFact(r), Fact("grade", g), Fact("date", gd, required = t == 3)),
                        listOf(sub.search, c), "${r.name} has a DOHMH letter grade of $g, given on ${longDate(gd)}.", t,
                    )
                }
            }
        }
    }

    private fun lookupLastInspection(): List<() -> Draft?> = withCamis.filter { (it.latestInspection ?: "") > "1900-01-01" }.map { r ->
        {
            subject(r)?.let { sub ->
                Draft(listOf(r.id)) { rng ->
                    val (c, o) = call("get_restaurant", mapOf("id" to r.id))
                    val d = o.s("latest_inspection") ?: return@Draft null
                    val score = o.n("latest_score")?.toInt()
                    val t = if (score == null) listOf(0, 1, 3)[rng.nextInt(3)] else rng.nextInt(4)
                    val p = sub.phrase
                    val q = when (t) {
                        0 -> "When was $p last inspected by the NYC health department?"
                        1 -> "What is the date of the most recent DOHMH inspection of $p?"
                        2 -> "When did a health inspector last visit $p, and what score did that inspection get?"
                        else -> "As of $cutoff, when was ${p}'s latest restaurant inspection?"
                    }
                    val facts = listOfNotNull(nameFact(r), Fact("date", d), score?.let { Fact("count", it.toString(), required = t == 2) })
                    val ref = "${r.name} was last inspected on ${longDate(d)}" + (score?.let { "; that inspection scored $it points (lower is better)." } ?: ".")
                    task("lookup", "last_inspection", q, listOf(r.id), facts, listOf(sub.search, c), ref, t)
                }
            }
        }
    }

    private val violationWords by lazy {
        ViolationWords(s.query("SELECT DISTINCT violation_description FROM inspections WHERE violation_description IS NOT NULL ORDER BY 1").map { it[0] as String })
    }

    private fun lookupViolation(): List<() -> Draft?> = withCamis.filter { (it.latestInspection ?: "") > "1900-01-01" }.map { r ->
        {
            subject(r)?.let { sub ->
                Draft(listOf(r.id)) { rng ->
                    val (c, o) = call("get_inspections", mapOf("id" to r.id, "limit" to 2))
                    val visits = o.list("inspections")
                    val v0 = visits.firstOrNull() ?: return@Draft null
                    if (visits.size > 1 && visits[1].s("date") == v0.s("date")) return@Draft null
                    val vs = v0.list("violations")
                    if (vs.size !in 1..3) return@Draft null
                    val date = v0.s("date")!!
                    val t = rng.nextInt(4)
                    val p = sub.phrase
                    val q = when (t) {
                        0 -> "What violations were cited at ${p}'s most recent health inspection?"
                        1 -> "What was $p cited for when DOHMH last inspected it?"
                        2 -> "At its ${longDate(date)} inspection, what problems did inspectors find at $p?"
                        else -> "List the violations from the latest DOHMH inspection of $p and say which were critical."
                    }
                    val facts = listOf(nameFact(r)) + vs.map { v ->
                        Fact("violation", v.s("code")!!, violationWords.keywords(v.s("text") ?: ""))
                    } + Fact("date", date, required = false)
                    val ref = "At its most recent inspection, on ${longDate(date)}, ${r.name} was cited for " + vs.joinToString("; ") { v ->
                        "${v.s("code")} (${if ((v["critical"] as? io.tlaloc.core.io.JsonBool)?.value == true) "critical" else "not critical"}): ${v.s("text")}"
                    } + "."
                    task("lookup", "violation", q, listOf(r.id), facts, listOf(sub.search, c), ref, t)
                }
            }
        }
    }

    // ------------------------------------------------------------------ filters

    private fun listReference(intro: String, results: List<JsonObject>, showGrade: Boolean): String = buildString {
        append(intro)
        for (x in results) {
            append("\n- ").append(x.s("name"))
            x.s("address")?.let { append(", ").append(it.substringBefore(", Brooklyn").replace(Regex("\\s+"), " ")) }
            if (showGrade) append(x.s("grade")?.takeIf { it in setOf("A", "B", "C") }?.let { " (grade $it)" } ?: " (no letter grade)")
        }
    }

    private fun filterCuisineArea(): List<() -> Draft?> {
        val combos = TASK_CUISINES.keys.flatMap { c -> areaPhrases.flatMap { a -> listOf<String?>(null, "A", "B").map { g -> Triple(c, a, g) } } }
        return combos.map { (c, a, g) ->
            {
                val args = linkedMapOf<String, Any?>("cuisine" to c, "neighborhood" to a)
                if (g != null) args["min_grade"] = g
                args["limit"] = 50
                val (call, o) = call("search_restaurants", args)
                val results = o.list("results")
                val n = o.n("count")!!.toInt()
                if (n !in 1..5 || results.size != n) null
                else Draft(results.map { it.s("id")!! }) { rng ->
                    val cu = TASK_CUISINES.getValue(c)
                    val t = rng.nextInt(3)
                    val q = when (g) {
                        null -> listOf("Which $cu places are there in $a?", "List the $cu restaurants in $a.", "I'm looking for $cu food in $a. What are my options?")[t]
                        "A" -> listOf("Find A-graded $cu places in $a.", "Which $cu restaurants in $a have an A health grade?", "Are there $cu spots in $a with a grade A from the health department?")[t]
                        else -> listOf("Which $cu places in $a have a health grade of B or better?", "List the $cu restaurants in $a graded A or B.", "I want $cu food in $a from a place graded at least B. Which places qualify?")[t]
                    }
                    val facts = results.map { x -> nameFact(byId.getValue(x.s("id")!!)) } + Fact("count", n.toString(), required = false)
                    val intro = "${plural(n, "$cu place")} in $a match" + (if (n == 1) "es" else "") +
                        (when (g) { null -> ""; "A" -> " with an A grade"; else -> " with a grade of A or B" }) + ":"
                    task(
                        "filter", "cuisine_area_grade", q, results.map { it.s("id")!! }, facts, listOf(call),
                        listReference(intro, results, showGrade = true), t, answerSet = results.map { it.s("id")!! },
                    )
                }
            }
        }
    }

    private fun filterOpenAt(): List<() -> Draft?> {
        val times = listOf(8 to 0, 11 to 30, 13 to 0, 18 to 0, 19 to 30, 21 to 0, 22 to 30, 23 to 30, 1 to 0)
        val rng = Random(seed * 7_919 + 17)
        val combos = TASK_CUISINES.keys.flatMap { c -> areaPhrases.map { a -> c to a } }.map { (c, a) ->
            val day = cutoff.plusDays(1L + rng.nextInt(7))
            val (h, m) = times[rng.nextInt(times.size)]
            Triple(c, a, LocalDateTime.of(day.year, day.month, day.dayOfMonth, h, m))
        }
        return combos.map { (c, a, at) ->
            {
                val args = linkedMapOf<String, Any?>("cuisine" to c, "neighborhood" to a, "open_at" to at.toString(), "limit" to 50)
                val (call, o) = call("search_restaurants", args)
                val results = o.list("results")
                val n = o.n("count")!!.toInt()
                if (n !in 1..5 || results.size != n) null
                else Draft(results.map { it.s("id")!! }) { r2 ->
                    val cu = TASK_CUISINES.getValue(c)
                    val t = r2.nextInt(3)
                    val whenText = "${clock(at.hour, at.minute)} on ${weekdayDate(at.toLocalDate())}"
                    val q = listOf(
                        "Which $cu places in $a are open at $whenText?",
                        "I'll be in $a at $whenText. Which $cu places will be open?",
                        "Find $cu restaurants in $a that are open at $whenText.",
                    )[t]
                    val unknown = o.n("excluded_unknown_hours")?.toInt() ?: 0
                    val intro = "${plural(n, "$cu place")} in $a ${if (n == 1) "is" else "are"} open at $whenText, by their listed hours" +
                        (if (unknown > 0) " ($unknown more have no known hours)" else "") + ":"
                    task(
                        "filter", "open_at", q, results.map { it.s("id")!! },
                        results.map { x -> nameFact(byId.getValue(x.s("id")!!)) } + Fact("count", n.toString(), required = false),
                        listOf(call), listReference(intro, results, showGrade = false), t, answerSet = results.map { it.s("id")!! },
                    )
                }
            }
        }
    }

    // ------------------------------------------------------------------ comparison

    private fun comparison(): List<() -> Draft?> {
        val rng = Random(seed * 7_919 + 29)
        // Pairs are formed within (neighborhood, split), so both restaurants of a pair are in one split.
        val groups = withCamis.filter { it.nta2020 != null }.groupBy { it.nta2020!! to restaurantSplit(it.id) }
            .entries.sortedWith(compareBy({ it.key.first }, { it.key.second }))
        val pairs = ArrayList<Triple<Rest, Rest, Int>>()
        for ((_, rs) in groups) {
            val sh = rs.sortedBy { it.id }.shuffled(rng)
            for (i in 0 until sh.size - 1 step 2) pairs += Triple(sh[i], sh[i + 1], rng.nextInt(3))
        }
        return pairs.map { (a, b, variant) ->
            {
                if (nameKey(a.name) == nameKey(b.name)) null
                else {
                    val sa = subject(a)
                    val sb = subject(b)
                    if (sa == null || sb == null) null else Draft(listOf(a.id, b.id)) { r2 -> compareTask(a, b, sa, sb, variant, r2) }
                }
            }
        }
    }

    private fun compareTask(a: Rest, b: Rest, sa: Subject, sb: Subject, variant: Int, rng: Random): Task? {
        val (ca, oa) = call("get_restaurant", mapOf("id" to a.id))
        val (cb, ob) = call("get_restaurant", mapOf("id" to b.id))
        val t = rng.nextInt(2)
        val pa = sa.phrase
        val pb = sb.phrase
        val sol = listOf(sa.search, sb.search, ca, cb)
        when (variant) {
            0 -> {
                val x = oa.n("latest_score")?.toInt() ?: return null
                val y = ob.n("latest_score")?.toInt() ?: return null
                if (Math.abs(x - y) < 3) return null
                val (w, l) = if (x < y) (a to x) to (b to y) else (b to y) to (a to x)
                val q = listOf(
                    "Which did better at its latest health inspection, $pa or $pb?",
                    "Compare the latest DOHMH inspection scores of $pa and $pb. Which one scored better?",
                )[t]
                return task(
                    "comparison", "compare", q, listOf(a.id, b.id),
                    listOf(nameFact(a), nameFact(b), Fact("count", x.toString()), Fact("count", y.toString())), sol,
                    "${w.first.name} did better: its latest inspection scored ${w.second} points, against ${l.second} for ${l.first.name} (lower scores are better).", t,
                )
            }
            1 -> {
                val x = oa.n("google_rating") ?: return null
                val y = ob.n("google_rating") ?: return null
                if (Math.abs(x - y) < 0.3) return null
                val (w, l) = if (x > y) (a to x) to (b to y) else (b to y) to (a to x)
                val q = listOf(
                    "Which has the higher Google rating, $pa or $pb?",
                    "Between $pa and $pb, which do Google reviewers rate higher?",
                )[t]
                return task(
                    "comparison", "compare", q, listOf(a.id, b.id),
                    listOf(nameFact(a), nameFact(b), Fact("number", fmtNum(x), tol = 0.051), Fact("number", fmtNum(y), tol = 0.051)), sol,
                    "${w.first.name} is rated higher on Google Local: ${fmtNum(w.second)} stars against ${fmtNum(l.second)} for ${l.first.name} (reviews to 2021).", t,
                )
            }
            else -> {
                fun crit(o: JsonObject) = ((o["counts"] as? JsonObject)?.get("critical_violations_last_12_months") as? JsonNumber)?.value?.toInt()
                val x = crit(oa) ?: return null
                val y = crit(ob) ?: return null
                if (x == y) return null
                val (w, l) = if (x > y) (a to x) to (b to y) else (b to y) to (a to x)
                val since = cutoff.minusYears(1)
                val q = listOf(
                    "Which had more critical violations in the 12 months before $cutoff (since $since): $pa or $pb?",
                    "Since ${longDate(since)}, has $pa or $pb been cited for more critical health violations?",
                )[t]
                return task(
                    "comparison", "compare", q, listOf(a.id, b.id),
                    listOf(nameFact(a), nameFact(b), Fact("count", x.toString()), Fact("count", y.toString())), sol,
                    "${w.first.name} had more: ${plural(w.second, "critical violation")} since ${longDate(since)}, against ${l.second} for ${l.first.name}.", t,
                )
            }
        }
    }

    // ------------------------------------------------------------------ multi-hop

    private fun multihop(): List<() -> Draft?> {
        val withText = s.query("SELECT gmap_id, count(*) FROM reviews WHERE text IS NOT NULL GROUP BY gmap_id").associate { it[0] as String to (it[1] as Number).toInt() }
        return withCamis.filter { it.gmapId != null && (withText[it.gmapId] ?: 0) >= 3 }.map { r ->
            {
                subject(r)?.let { sub -> Draft(listOf(r.id)) { rng -> multihopTask(r, sub, rng) } }
            }
        }
    }

    private fun multihopTask(r: Rest, sub: Subject, rng: Random): Task? {
        val since = cutoff.minusYears(1)
        val (c1, o1) = call("get_restaurant", mapOf("id" to r.id))
        val cv = ((o1["counts"] as? JsonObject)?.get("critical_violations_last_12_months") as? JsonNumber)?.value?.toInt() ?: return null
        val (c2, o2) = call("get_inspections", mapOf("id" to r.id, "since" to since.toString(), "limit" to 50))
        val visits = o2.list("inspections")
        val crit = visits.flatMap { v -> v.list("violations").filter { (it["critical"] as? io.tlaloc.core.io.JsonBool)?.value == true }.map { v.s("date")!! } }
        if (crit.size != cv) return null
        val (c3, o3) = call("get_reviews", mapOf("id" to r.id))
        val avg = o3.n("average_rating") ?: return null
        val count = o3.n("count")!!.toInt()
        val last = o3.s("last")
        val t = rng.nextInt(3)
        val p = sub.phrase
        val q = listOf(
            "How many critical violations has $p been cited for in the 12 months before $cutoff, and what do Google reviewers say about it?",
            "Has $p had critical health violations since ${longDate(since)}, and how do reviewers rate it?",
            "I'm thinking of trying $p. How many critical violations did it get in the last 12 months (to $cutoff), and what is its review average?",
        )[t]
        val ref = buildString {
            append("Since ${longDate(since)}, ${r.name} has been cited for ")
            append(if (cv == 0) "no critical violations" else plural(cv, "critical violation") + ", most recently on ${longDate(crit.max())}")
            append(". On Google Local (reviews up to 2021) it averages ${fmtNum(avg)} stars over ${plural(count, "review")}")
            last?.let { append("; the latest review is from ${longDate(it)}") }
            append(".")
        }
        return task(
            "multihop", "critical_and_reviews", q, listOf(r.id),
            listOf(nameFact(r), Fact("count", cv.toString()), Fact("number", fmtNum(avg), tol = 0.051), Fact("count", count.toString(), required = false)),
            listOf(sub.search, c1, c2, c3), ref, t,
        )
    }

    // ------------------------------------------------------------------ complaints

    private fun complaints(): List<() -> Draft?> {
        val linked = s.query("SELECT DISTINCT camis FROM complaints WHERE camis IS NOT NULL ORDER BY 1").map { it[0] as String }.toSet()
        return withCamis.filter { it.camis in linked }.flatMap { r -> listOf(0, 1).map { variant -> r to variant } }.map { (r, variant) ->
            {
                subject(r)?.let { sub -> Draft(listOf(r.id)) { rng -> complaintTask(r, sub, variant, rng) } }
            }
        }
    }

    private fun complaintTask(r: Rest, sub: Subject, variant: Int, rng: Random): Task? {
        val p = sub.phrase
        if (variant == 0) {
            val months = listOf(6, 12, 24)[rng.nextInt(3)]
            val since = cutoff.minusMonths(months.toLong())
            val (c, o) = call("get_complaints", mapOf("id" to r.id, "since" to since.toString(), "limit" to 50))
            val n = o.n("count")!!.toInt()
            val t = rng.nextInt(3)
            val q = listOf(
                "How many 311 food complaints (food poisoning or food establishment) are linked to $p since ${longDate(since)}?",
                "Since $since, how many times has $p been the subject of a 311 food complaint?",
                "Have there been 311 food complaints about $p in the last $months months (since ${longDate(since)})? How many?",
            )[t]
            val latest = o.list("complaints").firstOrNull()
            val ref = if (n == 0) "No 311 food complaints linked to ${r.name} were created on or after ${longDate(since)}."
            else "${plural(n, "311 food complaint")} linked to ${r.name} ${if (n == 1) "was" else "were"} created on or after ${longDate(since)}; the latest on ${longDate(latest!!.s("created")!!)} (${latest.s("type")})."
            return task("complaints", "complaints", q, listOf(r.id), listOf(nameFact(r), Fact("count", n.toString())), listOf(sub.search, c), ref, t)
        }
        val (c, o) = call("get_complaints", mapOf("id" to r.id, "limit" to 1))
        val x = o.list("complaints").firstOrNull() ?: return null
        val created = x.s("created") ?: return null
        val type = x.s("type") ?: return null
        val desc = x.s("descriptor")?.takeIf { d -> d.count { it.isLetter() } >= 4 }
        val t = 3 + rng.nextInt(2)
        val q = if (t == 3) "When was the most recent 311 food complaint about $p, and what was it about?"
        else "What was the latest 311 complaint linked to $p, and when was it filed?"
        val ref = "The latest 311 complaint linked to ${r.name} was created on ${longDate(created)}: $type" + (desc?.let { " ($it)" } ?: "") + "."
        return task(
            "complaints", "complaints", q, listOf(r.id),
            listOf(nameFact(r), Fact("date", created), Fact("keyword", type, listOfNotNull(type, desc).map { it.lowercase() }.distinct())),
            listOf(sub.search, c), ref, t,
        )
    }

    // ------------------------------------------------------------------ menus

    private fun menuPrices(): List<() -> Draft?> {
        val rng = Random(seed * 7_919 + 43)
        val rows = s.query("SELECT restaurant_id, text FROM menus WHERE status='ok' AND text IS NOT NULL ORDER BY restaurant_id, url")
            .groupBy { it[0] as String }.mapValues { it.value.first()[1] as String }
        val cands = ArrayList<Pair<Rest, Pair<String, String>>>()
        for ((id, text) in rows.entries.sortedBy { it.key }) {
            val r = byId[id] ?: continue
            val dishes = menuDishes(text)
            for (d in dishes.shuffled(rng).take(3)) cands += r to d
        }
        return cands.map { (r, d) ->
            {
                subject(r)?.let { sub -> Draft(listOf(r.id)) { r2 -> menuTask(r, sub, d.first, d.second, r2) } }
            }
        }
    }

    private fun menuTask(r: Rest, sub: Subject, dish: String, price: String, rng: Random): Task? {
        val (c, o) = call("get_menu", mapOf("id" to r.id))
        val text = o.s("menu") ?: return null
        if (!menuDishes(text).contains(dish to price)) return null
        val shown = if (dish == dish.uppercase()) titleCase(dish) else dish
        val t = rng.nextInt(3)
        val p = sub.phrase
        val q = listOf(
            "How much is the $shown at $p, according to its archived menu?",
            "What does ${p}'s menu (the Internet Archive copy) list as the price of $shown?",
            "On the archived menu of $p, what does $shown cost?",
        )[t]
        val at = o.s("archived_at")
        val ref = "${r.name}'s menu" + (at?.let { ", as archived by the Internet Archive on ${longDate(it)}," } ?: "") + " lists $shown at \$$price."
        return task(
            "menu", "price", q, listOf(r.id),
            listOfNotNull(nameFact(r), Fact("money", price), at?.let { Fact("date", it.take(10), required = false) }),
            listOf(sub.search, c), ref, t,
        )
    }

    // ------------------------------------------------------------------ unanswerable

    private fun outside(): List<() -> Draft?> {
        val combos = TASK_CUISINES.keys.flatMap { c -> OUTSIDE_PLACES.flatMap { pl -> (0 until 5).map { t -> Triple(c, pl, t) } } }
        return combos.map { (c, pl, t) ->
            {
                Draft(emptyList(), key = "outside|$c|$pl|$t") { _ ->
                    val cu = TASK_CUISINES.getValue(c)
                    val q = listOf(
                        "What are the A-graded $cu restaurants in $pl?",
                        "Find me a $cu place in $pl that is open late.",
                        "Which $cu spot in $pl has the best Google reviews?",
                        "How many 311 food complaints were filed against $cu restaurants in $pl since ${longDate(cutoff.minusYears(1))}?",
                        "What health grade do the $cu restaurants in $pl have?",
                    )[t]
                    task(
                        "unanswerable", "outside_brooklyn", q, emptyList(), listOf(Fact("refusal", "out_of_scope")), emptyList(),
                        "I can't answer that from these sources: they cover only Brooklyn (Kings County), and $pl is outside Brooklyn.", t,
                    )
                }
            }
        }
    }

    private fun nonexistent(): List<() -> Draft?> {
        val names = FAKE_FIRST.flatMap { a -> FAKE_SECOND.flatMap { b -> FAKE_THIRD.map { c -> "$a $b $c" } } }
        return names.map { fake ->
            {
                val words = fake.split(' ').take(2).joinToString(" ")
                if (s.ftsIds(words, 5).let { ids -> ids.any { id -> normalizeName(byId[id]!!.name).containsAll(normalizeName(words)) } } ||
                    rests.any { nameKey(it.name) == nameKey(fake) }
                ) null
                else Draft(emptyList(), key = "nonexistent|$fake") { rng ->
                    val a = areaPhrases[rng.nextInt(areaPhrases.size)]
                    val t = rng.nextInt(5)
                    val q = listOf(
                        "What is the health inspection grade of $fake in $a?",
                        "When was $fake in $a last inspected?",
                        "Does $fake in $a have any 311 food complaints?",
                        "What is on the menu at $fake in $a, and how much are the mains?",
                        "What do Google reviewers say about $fake in $a?",
                    )[t]
                    val (c, _) = call("search_restaurants", mapOf("query" to fake))
                    task(
                        "unanswerable", "nonexistent", q, emptyList(), listOf(Fact("refusal", "not_found")), listOf(c),
                        "I could not find a restaurant called $fake in $a, or anywhere in Brooklyn, in the DOHMH, OpenStreetMap or Google Local data, so I have nothing on it.", t,
                    )
                }
            }
        }
    }

    private fun closed(): List<() -> Draft?> {
        val ids = s.query(
            "SELECT r.id FROM restaurants r WHERE r.camis IS NOT NULL AND EXISTS (SELECT 1 FROM inspections i WHERE i.camis = r.camis " +
                "AND i.inspection_date = r.latest_inspection AND (i.action LIKE '%Closed by DOHMH%' OR i.action LIKE '%re-closed%')) ORDER BY r.id",
        ).map { it[0] as String }
        return ids.flatMap { id -> (0 until 3).map { t -> id to t } }.map { (id, t) ->
            {
                val r = byId.getValue(id)
                subject(r)?.let { sub ->
                    Draft(listOf(r.id)) { _ ->
                        val (c, o) = call("get_inspections", mapOf("id" to r.id, "limit" to 1))
                        val v = o.list("inspections").firstOrNull() ?: return@Draft null
                        val action = v.s("action") ?: return@Draft null
                        if (!action.contains("Closed by DOHMH") && !action.contains("re-closed")) return@Draft null
                        val date = v.s("date")!!
                        val p = sub.phrase
                        val q = listOf(
                            "Is $p open for business, according to the health department's records as of $cutoff?",
                            "Can I eat at $p? What do the latest DOHMH records say?",
                            "What happened at ${p}'s most recent health inspection?",
                        )[t]
                        task(
                            "unanswerable", "closed", q, listOf(r.id),
                            listOf(nameFact(r), Fact("date", date), Fact("keyword", "closed", listOf("closed", "closure", "shut"))),
                            listOf(sub.search, c),
                            "DOHMH closed ${r.name} at its inspection on ${longDate(date)}, and the records up to $cutoff show no later inspection, so I cannot say it is open now.", t,
                        )
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ live only

    private fun liveOnly(): List<() -> Draft?> = withCamis.flatMap { r -> (0 until 3).map { t -> r to t } }.map { (r, t) ->
        {
            subject(r)?.let { sub ->
                Draft(listOf(r.id)) { _ ->
                    val p = sub.phrase
                    val (q, call) = when (t) {
                        0 -> "Is $p open right now?" to ToolCall.of("search_restaurants", mapOf("query" to r.name, "open_at" to "now"))
                        1 -> "Has $p been inspected since $cutoff? What did the inspection find?" to ToolCall.of("get_inspections", mapOf("id" to r.id, "since" to cutoff.toString()))
                        else -> "Is there any recent news about $p?" to ToolCall.of("web_search", mapOf("query" to "${r.name} Brooklyn"))
                    }
                    task(
                        "live", "live_only", q, listOf(r.id), emptyList(), listOf(sub.search, call),
                        "Live only: needs the live tools (current time, DOHMH after the cutoff, or web search); not scored offline.", t, offline = false,
                    )
                }
            }
        }
    }

    // ------------------------------------------------------------------ driver

    private fun candidates(kind: String): List<() -> Draft?> = when (kind) {
        "grade" -> lookupGrade()
        "last_inspection" -> lookupLastInspection()
        "violation" -> lookupViolation()
        "cuisine_area_grade" -> filterCuisineArea()
        "open_at" -> filterOpenAt()
        "compare" -> comparison()
        "critical_and_reviews" -> multihop()
        "complaints" -> complaints()
        "price" -> menuPrices()
        "outside_brooklyn" -> outside()
        "nonexistent" -> nonexistent()
        "closed" -> closed()
        "live_only" -> liveOnly()
        else -> error(kind)
    }

    /** Every task, per split, in id order. */
    fun generate(): Map<Split, List<Task>> {
        val quotas = sizes.quotas()
        val out = Split.entries.associateWith { ArrayList<Task>() }
        val questions = HashSet<String>()
        for ((ki, k) in TASK_KINDS.withIndex()) {
            val t0 = System.nanoTime()
            val rng = Random(seed * 1_000_003 + ki)
            val want = quotas.getValue(k.kind)
            val got = Split.entries.associateWith { 0 }.toMutableMap()
            var tried = 0
            for (cand in candidates(k.kind).shuffled(rng)) {
                if (Split.entries.all { got.getValue(it) >= want.getValue(it) }) break
                tried++
                val d = cand() ?: continue
                val splits = d.restaurants.map { restaurantSplit(it) }.distinct()
                val sp = when {
                    d.restaurants.isEmpty() -> splitOf(d.key ?: error("draft without restaurants or key"), splitSeed)
                    splits.size == 1 -> splits[0]
                    else -> continue
                }
                if (got.getValue(sp) >= want.getValue(sp)) continue
                val t = d.build(rng) ?: continue
                if (!questions.add(t.question)) continue
                out.getValue(sp) += t.withSplit(sp)
                got[sp] = got.getValue(sp) + 1
            }
            log(
                "%-22s %s  (%d candidates tried, %.1f s)".format(
                    "${k.category}/${k.kind}", Split.entries.joinToString("  ") { "${it.label} ${got[it]}/${want[it]}" },
                    tried, (System.nanoTime() - t0) / 1e9,
                ),
            )
        }
        return out.mapValues { (_, ts) -> ts.sortedBy { it.id } }
    }
}

private fun Task.withSplit(sp: Split): Task {
    val id = "bk-" + sha256Hex("$category|$kind|$question".toByteArray(Charsets.UTF_8)).take(12)
    return Task(id, sp.label, category, kind, question, restaurants, facts, answerSet, solution, reference, offline, template, asOf)
}

fun fmtNum(x: Double): String = if (x == Math.rint(x)) x.toLong().toString() else "%.2f".format(Locale.US, x).trimEnd('0').trimEnd('.')

/**
 * Dish and price pairs of a menu text that a question can ask about unambiguously: a dish line
 * (letters, no digits) followed by a price-only line, or "Dish $9.50" on one line; the dish
 * must occur once in the menu.
 */
fun menuDishes(text: String): List<Pair<String, String>> {
    val lines = text.lines().map { it.trim() }
    val priceOnly = Regex("^\\$\\s?(\\d{1,3}(?:\\.\\d{2})?)$")
    val inline = Regex("^([^$\\d|/]{3,50}?)\\s*[.\\-–:]*\\s*\\$\\s?(\\d{1,3}(?:\\.\\d{2})?)$")
    // A dish name: a short phrase starting with a capital, not an option ("With onions", "Add bacon") or a description.
    fun okDish(d: String) = d.length in 3..40 && d.count { it.isLetter() } >= 3 && d[0].isUpperCase() &&
        d.none { it.isDigit() || it == '$' || it == '|' || it == '/' || it == ',' || it == ';' || it == ':' } &&
        d.split(Regex("\\s+")).size <= 6 &&
        !Regex("(?i)^(add|extra|side of|with|w|small|large|sm|lg|regular|price|each|plain|choice|served|includes?)\\b").containsMatchIn(d)
    val occurrences = lines.groupingBy { it.lowercase() }.eachCount()
    val out = ArrayList<Pair<String, String>>()
    for (i in lines.indices) {
        val l = lines[i]
        val pm = priceOnly.find(l)
        if (pm != null && i > 0) {
            val d = lines[i - 1].trimEnd('.').trim()
            if (okDish(d)) out += d to pm.groupValues[1]
            continue
        }
        inline.find(l)?.let { m ->
            val d = m.groupValues[1].trim().trimEnd('.').trim()
            if (okDish(d) && l.count { it == '$' } == 1) out += d to m.groupValues[2]
        }
    }
    // A dish named twice in the menu (or on two priced lines) is ambiguous.
    val names = out.groupingBy { it.first.lowercase() }.eachCount()
    return out.filter { (d, _) -> names[d.lowercase()] == 1 && (occurrences[d.lowercase()] ?: 0) <= 1 && lines.count { it.lowercase().startsWith(d.lowercase()) } == 1 }
}

/** Checks that no restaurant of a test (or dev) task occurs in a train task. Returns the overlaps. */
fun splitOverlaps(tasks: Map<Split, List<Task>>): Map<String, Set<String>> {
    fun ids(sp: Split) = tasks[sp].orEmpty().flatMap { it.restaurants + it.answerSet.orEmpty() }.toSet()
    val train = ids(Split.TRAIN)
    val dev = ids(Split.DEV)
    val test = ids(Split.TEST)
    return linkedMapOf("train&test" to (train intersect test), "train&dev" to (train intersect dev), "dev&test" to (dev intersect test))
}

