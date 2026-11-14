package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDate
import kotlin.random.Random

// `tasks` subcommand:
//   tasks [generate]                   write build/data/tasks/{train,dev,test}.jsonl and summary.tsv, then check them
//   tasks check [--show-mutants]       regenerate in memory and compare with the files; split overlap; reference
//                                      answers pass; corrupted references fail (--show-mutants prints those that pass)
//   tasks show [--split S] [--n 30] [--seed K]   print a seeded sample: question, solution calls, ground truth
//   tasks score FILE                   score answers: one JSON per line {"id", "answer", "calls": [{"name","arguments"}]}
//
// Sizes follow the database: a quick ingest (menus for 150 places) gives the quick sizes.
// Seeds: manifest.json seeds.tasks (templates and sampling) and seeds.task_split (the restaurant split).

const val TASKS_DIR = "tasks"

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun taskFile(data: Path, sp: Split): Path = data.resolve(TASKS_DIR).resolve("${sp.label}.jsonl")

fun dbScope(db: Path): String = ToolServer(db, ToolMode.SNAPSHOT).use { s ->
    s.query("SELECT value FROM meta WHERE key='menu_scope'").firstOrNull()?.get(0) as String? ?: "full"
}

/** The split files' bytes: one task per line, `\n` line ends. */
fun taskFileText(tasks: List<Task>): String = tasks.joinToString("") { it.line() + "\n" }

fun readTasks(f: Path): List<Task> = Files.readAllLines(f).filter { it.isNotBlank() }.map { Task.parse(it) }

class TaskRun(val tasks: Map<Split, List<Task>>, val scope: String, val seconds: Double)

/** Generates every split from [db] with the manifest's cutoff and seeds. */
fun generateTasks(db: Path, home: Path, log: (String) -> Unit = {}): TaskRun {
    val manifest = home.resolve("manifest.json")
    val cutoff = LocalDate.parse(readManifestJson(manifest)?.optString("cutoff") ?: DEFAULT_CUTOFF)
    val seed = manifestSeed("tasks", manifest)
    val splitSeed = manifestSeed("task_split", manifest)
    val t0 = System.nanoTime()
    return ToolServer(db, ToolMode.SNAPSHOT).use { s ->
        check(s.snapshotDate == cutoff.toString()) { "database snapshot ${s.snapshotDate} differs from the manifest cutoff $cutoff; rerun ingest" }
        val scope = s.query("SELECT value FROM meta WHERE key='menu_scope'").firstOrNull()?.get(0) as String? ?: "full"
        val sizes = if (scope == "quick") TaskSizes.QUICK else TaskSizes.FULL
        val tasks = TaskGenerator(s, cutoff, seed, splitSeed, sizes, log).generate()
        check(!s.netConstructed) { "task generation constructed an HTTP client" }
        TaskRun(tasks, scope, (System.nanoTime() - t0) / 1e9)
    }
}

/** Category and kind counts per split, as TSV. */
fun taskSummary(tasks: Map<Split, List<Task>>): String = buildString {
    appendLine("category\tkind\t" + Split.entries.joinToString("\t") { it.label })
    for (k in TASK_KINDS) {
        append("${k.category}\t${k.kind}")
        for (sp in Split.entries) append("\t").append(tasks[sp].orEmpty().count { it.kind == k.kind })
        appendLine()
    }
    append("all\tall")
    for (sp in Split.entries) append("\t").append(tasks[sp].orEmpty().size)
    appendLine()
}

/** Category counts of one split, compact (for the manifest). */
fun categoryCounts(tasks: List<Task>): String =
    TASK_KINDS.map { it.category }.distinct().joinToString(",") { c -> "$c=${tasks.count { it.category == c }}" }

/**
 * Runs every offline task's solution calls again and scores its reference answer with the
 * verifier: the solution must hold each required fact, and the reference must pass. Returns
 * the failures.
 */
fun checkReferences(db: Path, tasks: List<Task>, cutoff: LocalDate): List<String> = ToolServer(db, ToolMode.SNAPSHOT).use { s ->
    val v = Verifier(s, cutoff)
    val out = ArrayList<String>()
    for (t in tasks) {
        val outputs = t.solution.map { s.call(it.name, it.arguments) }
        outputs.forEachIndexed { i, o -> if ((parseJson(o) as JsonObject)["error"] != null && t.offline) out += "${t.id}: solution call ${i + 1} failed: $o" }
        if (!t.offline) continue
        val text = outputs.joinToString("\n")
        for (f in t.facts.filter { it.required }) if (!factInOutputs(f, text)) out += "${t.id}: fact $f is not in the solution's tool results"
        val verdict = v.verify(t, t.reference, t.solution)
        if (!verdict.pass) out += "${t.id}: reference answer fails: $verdict"
    }
    out
}

/**
 * Corrupted copies of a reference answer, each of which the verifier must reject: another grade
 * letter, a date one day later, a count plus one, a price plus one dollar, a rating plus 0.3,
 * the restaurant's name replaced by an invented one, an invented restaurant added to a list, a
 * refusal replaced by an invented answer; the fact negated ("does not have a ... grade of A",
 * "was not last inspected on"); the count hedged among other values ("2, 3 or 4") or followed by
 * another count ("(7 critical violations in all)"); another price for the dish with the true one
 * as an old price; in comparisons the other restaurant picked, the two values swapped, and the
 * scale reversed ("higher scores are better").
 */
fun referenceMutants(t: Task, taken: (LocalDate) -> Boolean = { false }): List<Pair<String, String>> {
    val r = t.reference
    val out = ArrayList<Pair<String, String>>()
    fun sub(kind: String, old: String, new: String) { if (r.contains(old)) out += kind to r.replaceFirst(old, new) }
    when (t.kind) {
        "grade" -> sub("negated", " has a DOHMH letter grade of ", " does not have a DOHMH letter grade of ")
        "last_inspection" -> sub("negated", " was last inspected on ", " was not last inspected on ")
        "closed" -> sub("negated", "DOHMH closed ", "DOHMH has not closed ")
        "price" -> sub("negated", " lists ", " does not list ")
        "complaints" -> if (t.facts.any { it.kind == "date" && it.required }) sub("negated", " was created on ", " was not created on ")
    }
    t.facts.firstOrNull { it.kind == "count" && it.required && it.alts.isNotEmpty() && t.facts.none { o -> o.of != null } }?.let { f ->
        val v = f.value.toInt()
        val unit = mapOf("critical" to "critical violations", "complaint" to "311 food complaints", "point" to "points", "review" to "reviews").getValue(f.alts[0])
        out += "added count" to r.trimEnd('.') + " (${v + 5} $unit in all)."
        Regex("(?<![\\d.,$/:-])$v(?= points| critical| 311)").find(r)?.let { m -> out += "hedged count" to r.replaceRange(m.range, "$v, ${v + 1} or ${v + 2}") }
    }
    t.facts.firstOrNull { it.kind == "money" && it.alts.isNotEmpty() }?.let { f ->
        val p = f.value.toDouble()
        sub("price, true one as old", " at \$${f.value}.", " at \$${"%.2f".format(java.util.Locale.US, p + 1)} (it was \$${f.value} years ago).")
    }
    t.facts.firstOrNull { it.kind == "winner" }?.let { wf ->
        val w = wf.value
        val l = t.restaurants.first { it != w }
        val nameOf = t.restaurants.zip(t.facts.filter { it.kind == "name" }.map { it.value }).toMap()
        val valOf = t.facts.filter { it.of != null }.associate { it.of!! to it.value }
        val (wn, ln, wv, lv) = listOf(nameOf.getValue(w), nameOf.getValue(l), valOf.getValue(w), valOf.getValue(l))
        val since = dateMentions(r).firstOrNull()?.let { r.substring(it.start, it.end) } ?: ""
        fun verdict(a: String, av: String, b: String, bv: String) = when (wf.alts[0]) {
            "score" -> "$a did better: its latest inspection scored $av points, against $bv for $b (lower scores are better)."
            "rating" -> "$a is rated higher on Google Local: $av stars against $bv for $b (reviews to 2021)."
            else -> "$a had more: $av critical violations since $since, against $bv for $b."
        }
        out += "other restaurant picked" to verdict(ln, lv, wn, wv)
        out += "values swapped" to verdict(wn, lv, ln, wv)
        if (wf.alts[0] == "score") sub("scale reversed", "(lower scores are better)", "(higher scores are better)")
    }
    (Regex("\\(grade ([ABC])\\)").find(r) ?: Regex("grade (?:of )?([ABC])\\b(?! or)").find(r))?.let { m ->
        val g = m.groups[1]!!
        val other = mapOf("A" to "B", "B" to "C", "C" to "A").getValue(g.value)
        out += "grade" to r.replaceRange(g.range, other)
    }
    // A later date that the snapshot does not hold for the task's restaurants (a real date of theirs
    // attached to another event is a limit of the verifier, see the README).
    dateMentions(r).firstOrNull()?.let { d ->
        d.date()?.let { orig ->
            (1L..60L).map { orig.plusDays(it) }.firstOrNull { !taken(it) }?.let { out += "date" to r.replaceRange(d.start, d.end, longDate(it)) }
        }
    }
    t.facts.firstOrNull { it.kind == "count" && it.required && it.value != "0" }?.let { f ->
        Regex("(?<![\\d.,$/:-])${f.value}(?![\\d.])(?! points|\\s+critical| 311)").let { rx ->
            // The number as the reference states it ("12 points", "4 critical violations", "3 311 food complaints").
            Regex("(?<![\\d.,$/:-])${f.value}(?= points| critical| 311)").find(r) ?: rx.find(r)
        }?.let { m -> out += "count" to r.replaceRange(m.range, (f.value.toInt() + 1).toString()) }
    }
    t.facts.firstOrNull { it.kind == "money" }?.let { f ->
        r.indexOf("$" + f.value).takeIf { it >= 0 }?.let { i ->
            out += "price" to r.replaceRange(i, i + 1 + f.value.length, "$" + "%.2f".format(java.util.Locale.US, f.value.toDouble() + 1))
        }
    }
    t.facts.firstOrNull { it.kind == "number" }?.let { f ->
        Regex("(?<![\\d.])${Regex.escape(f.value)} stars").find(r)?.let { m -> out += "rating" to r.replaceRange(m.range.first, m.range.first + f.value.length, fmtNum(f.value.toDouble() + 0.3)) }
    }
    t.facts.firstOrNull { it.kind == "name" }?.let { f ->
        r.indexOf(f.value).takeIf { it >= 0 }?.let { i -> out += "name" to r.replaceRange(i, i + f.value.length, "Golden Lantern Kitchen") }
    }
    if (t.answerSet != null) out += "added restaurant" to "$r\n- Copper Willow Bistro, 12 Imaginary Street (grade A)"
    // Guesses and additions on top of a true answer.
    when (t.kind) {
        "grade" -> {
            sub("grade hedged", " has a DOHMH letter grade of ", " probably has a DOHMH letter grade of ")
            t.facts.firstOrNull { it.kind == "grade" }?.let { g -> out += "other grade as a guess" to "$r It might be ${mapOf("A" to "C", "B" to "A", "C" to "B").getValue(g.value)} now." }
        }
        "closed" -> out += "reopening claimed" to "$r It has since reopened, and you can eat there."
        "violation" -> {
            val codes = t.facts.filter { it.kind == "violation" }.map { it.value }.toSet()
            listOf("04L" to "evidence of mice", "04M" to "live roaches", "04N" to "filth flies").firstOrNull { it.first !in codes }?.let { (c, what) ->
                out += "invented violation" to r.trimEnd('.') + "; $c (critical): $what."
            }
        }
        "complaints" -> {
            val topic = t.facts.firstOrNull { it.kind == "topic" }
            if (topic != null) sub("other complaint topic", "Food Establishment (${topic.value})", "Food Poisoning")
            else if (t.facts.any { it.kind == "keyword" && it.value.equals("Food Poisoning", ignoreCase = true) }) {
                Regex("Food Poisoning(?: \\([^)]*\\))?").find(r)?.let { m -> out += "other complaint topic" to r.replaceRange(m.range, "Food Establishment (Rodents/Insects/Garbage)") }
            }
        }
    }
    if (t.facts.any { it.kind == "refusal" }) out += "refusal, then a fact" to "$r It was last inspected on ${longDate(LocalDate.parse(t.asOf))}."
    t.facts.firstOrNull { it.kind == "money" && it.alts.isNotEmpty() }?.let { f ->
        sub("price hedged", " at \$${f.value}.", " at \$${f.value} or \$${"%.2f".format(java.util.Locale.US, f.value.toDouble() + 1)}.")
    }
    if (t.facts.any { it.kind == "refusal" }) out += "invented answer" to "It has a grade of A, given on ${longDate(LocalDate.parse(t.asOf).minusDays(40))}."
    return out
}

/** Per mutation kind: (rejected, total) over every offline task's reference. */
fun mutationCheck(db: Path, tasks: List<Task>, cutoff: LocalDate, showPassed: Boolean = false): Map<String, Pair<Int, Int>> = ToolServer(db, ToolMode.SNAPSHOT).use { s ->
    val v = Verifier(s, cutoff)
    val out = java.util.TreeMap<String, Pair<Int, Int>>()
    for (t in tasks.filter { it.offline }) for ((kind, text) in referenceMutants(t) { d ->
        // Dates the verifier accepts for any task: the cutoff, the day before, a year before, and the question's dates.
        d == cutoff || d == cutoff.minusDays(1) || d == cutoff.minusYears(1) || dateMentions(t.question).any { it.date() == d } ||
            (t.restaurants + t.answerSet.orEmpty()).any { d in v.datesOf(it) }
    }) {
        val rejected = !v.verify(t, text, t.solution).pass
        if (!rejected && showPassed) println("MUTANT-PASSED $kind ${t.id} ${t.kind}: ${text.replace("\n", " | ")}")
        val (a, b) = out[kind] ?: (0 to 0)
        out[kind] = (a + if (rejected) 1 else 0) to b + 1
    }
    out
}

/** A fact as the tool results hold it (JSON text of the solution calls' results). */
fun factInOutputs(f: Fact, text: String): Boolean = when (f.kind) {
    "name" -> (listOf(f.value) + f.alts).any { text.contains(io.tlaloc.core.io.jsonQuote(it)) }
    "grade" -> text.contains("\"grade\":\"${f.value}\"")
    "date" -> text.contains(f.value)
    "count" -> Regex("[:\\[,]${f.value}[,}\\]]").containsMatchIn(text)
    "number" -> Regex("[:\\[,](\\d+(?:\\.\\d+)?)[,}\\]]").findAll(text).any { Math.abs(it.groupValues[1].toDouble() - f.value.toDouble()) <= (f.tol ?: 0.0) + 1e-9 }
    "money" -> text.contains("$" + f.value) || text.contains("$ " + f.value)
    "violation" -> text.contains("\"code\":\"${f.value}\"")
    "keyword" -> (listOf(f.value) + f.alts).any { text.lowercase().contains(it.lowercase()) }
    "topic" -> text.contains(io.tlaloc.core.io.jsonQuote(f.value))
    "refusal", "winner", "absent" -> true
    else -> false
}

fun tasksCmd(args: List<String>): Int {
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val home = opt(args, "--home")?.let { Paths.get(it).toAbsolutePath().normalize() } ?: exampleHome()
    val db = data.resolve("brooklyn.db").toAbsolutePath().normalize()
    val sub = args.firstOrNull()?.takeIf { !it.startsWith("--") } ?: "generate"
    if (!Files.exists(db)) {
        System.err.println("tasks: $db not found; run the ingest stage first")
        return 1
    }
    val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    return when (sub) {
        "generate" -> {
            val scope = dbScope(db)
            if (isQuick(args) != (scope == "quick")) {
                System.err.println("tasks: the database is a ${scope} ingest but ${if (isQuick(args)) "--quick" else "a full run"} was asked for; run ingest with the same flag first")
                return 2
            }
            val run = generateTasks(db, home) { println("  $it") }
            val dir = data.resolve(TASKS_DIR)
            Files.createDirectories(dir)
            for (sp in Split.entries) {
                val f = taskFile(data, sp)
                Files.writeString(f, taskFileText(run.tasks.getValue(sp)))
                println("wrote ${run.tasks.getValue(sp).size} tasks to $f (sha256 ${fileSha256(f)})")
            }
            Files.writeString(dir.resolve("summary.tsv"), taskSummary(run.tasks))
            writeTaskInputs(data, taskInputs(home, ingestTableSha(db)))
            print(taskSummary(run.tasks))
            println("generated in %.1f s (scope ${run.scope})".format(run.seconds))
            checkFiles(db, data, home, cutoff, regenerate = false)
        }
        "check" -> checkFiles(db, data, home, cutoff, regenerate = true, showMutants = "--show-mutants" in args)
        "show" -> {
            val sp = Split.parse(opt(args, "--split") ?: "test")
            val n = opt(args, "--n")?.toInt() ?: 30
            val all = readTasks(taskFile(data, sp)).filter { it.offline }
            val pick = all.shuffled(Random(opt(args, "--seed")?.toLong() ?: 1L)).take(n)
            ToolServer(db, ToolMode.SNAPSHOT).use { s ->
                for (t in pick) {
                    println("== ${t.id} ${t.category}/${t.kind}")
                    println("Q: ${t.question}")
                    for (c in t.solution) println("   ${c.name} ${c.arguments} -> ${shorten(s.call(c.name, c.arguments), 300)}")
                    println("facts: " + t.facts.joinToString("; ") { (if (it.required) "" else "(optional) ") + it })
                    println("A: ${t.reference}")
                    println()
                }
            }
            0
        }
        "score" -> {
            val f = args.getOrNull(1)?.let { Paths.get(it) } ?: return usageTasks()
            val byId = Split.entries.flatMap { sp -> taskFile(data, sp).takeIf { Files.exists(it) }?.let(::readTasks).orEmpty() }.associateBy { it.id }
            var pass = 0
            var scored = 0
            ToolServer(db, ToolMode.SNAPSHOT).use { s ->
                val v = Verifier(s, cutoff)
                for (line in Files.readAllLines(f).filter { it.isNotBlank() }) {
                    val o = parseJson(line) as JsonObject
                    val t = byId[o.str("id")] ?: run { println(toJson(mapOf("id" to o.str("id"), "error" to "unknown task"))); null } ?: continue
                    val calls = (o["calls"] as? JsonArray)?.elements?.map { ToolCall.parse(it as JsonObject) }.orEmpty()
                    val verdict = v.verify(t, (o["answer"] as? JsonString)?.value ?: "", calls)
                    if (verdict.scored) { scored++; if (verdict.pass) pass++ }
                    println(verdict)
                }
            }
            println("passed $pass of $scored scored answers")
            0
        }
        else -> usageTasks()
    }
}

private fun checkFiles(db: Path, data: Path, home: Path, cutoff: LocalDate, regenerate: Boolean, showMutants: Boolean = false): Int {
    var bad = 0
    val files = Split.entries.associateWith { taskFile(data, it) }
    if (files.values.any { !Files.exists(it) }) {
        println("tasks check: missing split files; run the tasks stage")
        return 1
    }
    val tasks = files.mapValues { readTasks(it.value) }
    if (regenerate) {
        val run = generateTasks(db, home)
        for (sp in Split.entries) {
            val same = taskFileText(run.tasks.getValue(sp)) == Files.readString(files.getValue(sp))
            println("%-5s regenerated %s the file (%d tasks, %.1f s)".format(sp.label, if (same) "==" else "!=", run.tasks.getValue(sp).size, run.seconds))
            if (!same) bad++
        }
    }
    for ((pair, ids) in splitOverlaps(tasks)) {
        println("restaurants in both $pair: ${ids.size}" + if (ids.isEmpty()) "" else " (" + ids.take(5).joinToString() + ")")
        if (ids.isNotEmpty()) bad++
    }
    val dup = tasks.values.flatten().groupBy { it.id }.filter { it.value.size > 1 }
    if (dup.isNotEmpty()) { println("duplicate task ids: ${dup.keys.take(5)}"); bad++ }
    val t0 = System.nanoTime()
    val all = tasks.values.flatten()
    val fails = checkReferences(db, all, cutoff)
    println("reference answers: ${all.count { it.offline } - fails.map { it.substringBefore(':') }.distinct().size} of ${all.count { it.offline }} offline tasks answerable from their solution calls and passing the verifier (%.1f s)".format((System.nanoTime() - t0) / 1e9))
    fails.take(20).forEach { println("  $it") }
    if (fails.isNotEmpty()) bad++
    val t1 = System.nanoTime()
    val muts = mutationCheck(db, all, cutoff, showMutants)
    println("corrupted reference answers rejected by the verifier (%.1f s):".format((System.nanoTime() - t1) / 1e9))
    for ((k, ab) in muts) println("  %-23s %5d of %5d".format(k, ab.first, ab.second))
    val survivors = muts.values.sumOf { it.second - it.first }
    if (survivors > 0) {
        println("  $survivors corrupted reference(s) pass the verifier (list them with tasks check --show-mutants)")
        bad++
    }
    println(if (bad == 0) "tasks check: ok" else "tasks check: $bad problem(s)")
    return if (bad == 0) 0 else 1
}

private fun usageTasks(): Int {
    System.err.println(
        """
        usage: brooklyn-eats tasks [generate] [--quick]
               brooklyn-eats tasks check [--show-mutants]
               brooklyn-eats tasks show [--split train|dev|test] [--n 30] [--seed K]
               brooklyn-eats tasks score ANSWERS.jsonl
        """.trimIndent(),
    )
    return 2
}
