package brooklyn

import java.time.DayOfWeek
import java.time.LocalDateTime

// Opening hours from OSM `opening_hours` (the common subset: day ranges and lists, time ranges,
// "off", "24/7", ranges past midnight) and from Google Local's "Monday 11AM–9:30PM" strings.
// A week is a list of minute intervals per day; an interval may end after 24:00 (past midnight).

class WeekHours(val days: Map<DayOfWeek, List<IntRange>>) {
    fun isOpen(at: LocalDateTime): Boolean {
        val m = at.hour * 60 + at.minute
        if (days[at.dayOfWeek].orEmpty().any { m >= it.first && m < it.last }) return true
        val prev = at.dayOfWeek.minus(1)
        return days[prev].orEmpty().any { it.last > 1440 && m + 1440 < it.last }
    }

    override fun toString(): String = DayOfWeek.entries.joinToString("; ") { d ->
        val r = days[d].orEmpty()
        d.name.take(2).lowercase().replaceFirstChar { it.uppercaseChar() } + " " +
            (if (r.isEmpty()) "closed" else r.joinToString(",") { "%s-%s".format(hhmm(it.first), hhmm(it.last)) })
    }

    private fun hhmm(m: Int) = "%02d:%02d".format(m / 60, m % 60)
}

private val OSM_DAYS = mapOf(
    "mo" to DayOfWeek.MONDAY, "tu" to DayOfWeek.TUESDAY, "we" to DayOfWeek.WEDNESDAY, "th" to DayOfWeek.THURSDAY,
    "fr" to DayOfWeek.FRIDAY, "sa" to DayOfWeek.SATURDAY, "su" to DayOfWeek.SUNDAY,
)
private val DAY_TOKEN = "(?:Mo|Tu|We|Th|Fr|Sa|Su)"
private val DAY_SPEC = Regex("^$DAY_TOKEN(?:\\s*-\\s*$DAY_TOKEN)?(?:\\s*,\\s*$DAY_TOKEN(?:\\s*-\\s*$DAY_TOKEN)?)*", RegexOption.IGNORE_CASE)
private val TIME_RANGE = Regex("^(\\d{1,2}):(\\d{2})\\s*-\\s*(\\d{1,2}):(\\d{2})\\+?$")

private fun dayRange(spec: String): List<DayOfWeek>? {
    val out = ArrayList<DayOfWeek>()
    for (part in spec.split(',').map { it.trim() }.filter { it.isNotEmpty() }) {
        val ends = part.split('-').map { it.trim().lowercase() }
        val a = OSM_DAYS[ends[0]] ?: return null
        if (ends.size == 1) { out += a; continue }
        val b = OSM_DAYS[ends[1]] ?: return null
        var d = a
        while (true) { out += d; if (d == b) break; d = d.plus(1) }
    }
    return out
}

private fun timeRanges(spec: String): List<IntRange>? = spec.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { t ->
    val m = TIME_RANGE.find(t) ?: return null
    val (h1, m1, h2, m2) = m.destructured
    val start = h1.toInt() * 60 + m1.toInt()
    var end = h2.toInt() * 60 + m2.toInt()
    if (start > 24 * 60 || end > 48 * 60) return null
    if (end <= start) end += 1440
    start..end
}

/**
 * Parses OSM `opening_hours`; null when the value uses syntax outside the subset (week numbers,
 * months, public holidays with times, sunrise, comments). Later rules replace earlier ones for
 * the days they name, as in the OSM specification.
 */
fun parseOsmHours(value: String?): WeekHours? {
    val v = value?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    if (v == "24/7") return WeekHours(DayOfWeek.entries.associateWith { listOf(0..1440) })
    // Rules are separated by ';' (and, in practice, by ', ' before a day token).
    val rules = v.split(';').flatMap { it.split(Regex("(?<=\\d)\\s*,\\s*(?=$DAY_TOKEN\\b)", RegexOption.IGNORE_CASE)) }
        .map { it.trim() }.filter { it.isNotEmpty() }
    val days = LinkedHashMap<DayOfWeek, List<IntRange>>()
    for (rule in rules) {
        var rest = rule
        var ds: List<DayOfWeek> = DayOfWeek.entries
        val dm = DAY_SPEC.find(rest)
        if (dm != null) {
            ds = dayRange(dm.value) ?: return null
            rest = rest.substring(dm.value.length).trim()
        } else if (rest.startsWith("PH", ignoreCase = true)) {
            continue // public holidays: not modelled
        }
        when {
            rest.equals("off", true) || rest.equals("closed", true) -> ds.forEach { days[it] = emptyList() }
            rest.isEmpty() -> return null
            else -> {
                val tr = timeRanges(rest) ?: return null
                ds.forEach { days[it] = tr }
            }
        }
    }
    return if (days.isEmpty()) null else WeekHours(days)
}

private val G_DAYS = DayOfWeek.entries.associateBy { it.name.lowercase() }
private val G_TIME = Regex("^(\\d{1,2})(?::(\\d{2}))?\\s*([AP]M)?$", RegexOption.IGNORE_CASE)

private fun gMinutes(t: String, fallbackMeridiem: String?): Pair<Int, String?>? {
    val m = G_TIME.find(t.trim()) ?: return null
    val h = m.groupValues[1].toInt()
    val mm = m.groupValues[2].ifEmpty { "0" }.toInt()
    val mer = m.groupValues[3].uppercase().ifEmpty { null } ?: fallbackMeridiem ?: return null
    if (h !in 1..12) return null
    val h24 = when { mer == "AM" && h == 12 -> 0; mer == "PM" && h != 12 -> h + 12; else -> h }
    return h24 * 60 + mm to m.groupValues[3].uppercase().ifEmpty { null }
}

/** Parses Google Local hours ("Thursday 11AM–9:30PM; Monday Closed; Friday Open 24 hours"); null when any day fails. */
fun parseGoogleHours(value: String?): WeekHours? {
    val v = value?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    val days = LinkedHashMap<DayOfWeek, List<IntRange>>()
    for (entry in v.split(';').map { it.trim() }.filter { it.isNotEmpty() }) {
        val sp = entry.indexOf(' ')
        if (sp < 0) return null
        val day = G_DAYS[entry.substring(0, sp).lowercase()] ?: return null
        val rest = entry.substring(sp + 1).trim()
        if (rest.equals("Closed", true)) { days[day] = emptyList(); continue }
        if (rest.equals("Open 24 hours", true)) { days[day] = listOf(0..1440); continue }
        days[day] = rest.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { r ->
            val ends = r.split('–', '-')
            if (ends.size != 2) return null
            val (end, endMer) = gMinutes(ends[1], null) ?: return null
            var (start, _) = gMinutes(ends[0], endMer) ?: return null
            // "11–3PM": a start without its own meridiem that lands after the end is in the morning.
            if (!ends[0].trim().uppercase().let { it.endsWith("AM") || it.endsWith("PM") } && start > end && endMer == "PM") start -= 12 * 60
            val e = if (end <= start) end + 1440 else end
            start..e
        }
    }
    return if (days.isEmpty()) null else WeekHours(days)
}
