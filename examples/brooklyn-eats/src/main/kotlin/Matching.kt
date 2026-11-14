package brooklyn

import java.text.Normalizer
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sqrt

// Entity resolution between DOHMH establishments (CAMIS), OpenStreetMap places and Google Local
// businesses: a normalized name similarity and the distance between the two coordinates, with
// an address check. Pairs are assigned one to one, best score first.

/** Words that carry no identity in a business name. */
internal val NAME_STOPWORDS = setOf(
    "the", "inc", "llc", "corp", "corporation", "co", "ltd", "nyc", "ny", "and", "of", "restaurant",
    "restaurants", "rest", "bklyn", "a",
)

/** Words that describe a kind of place; a name made only of these does not identify a business. */
internal val GENERIC_WORDS = setOf(
    "pizza", "pizzeria", "deli", "grocery", "bar", "cafe", "coffee", "kitchen", "grill", "bakery", "food",
    "foods", "express", "house", "shop", "market", "sushi", "thai", "chinese", "mexican", "italian",
    "japanese", "indian", "halal", "lounge", "pub", "diner", "bagel", "bagels", "tea", "juice", "wine",
    "taqueria", "burger", "burgers", "chicken", "fried", "new", "york", "brooklyn", "garden", "place",
    "station", "corner", "spot", "street", "ave", "avenue", "gourmet", "ii", "1", "2", "&",
    "bistro", "tavern", "bk", "company", "cake", "studio", "wings", "fusion", "sports", "factory", "no",
    "number", "one", "cuisine", "asian", "grocery", "bodega", "cafeteria", "xpress", "inc", "nyc",
    "korean", "latin", "georgian", "caribbean", "jamaican", "vietnamese", "french", "greek", "turkish",
    "spanish", "peruvian", "dominican", "haitian", "russian", "polish", "ukrainian", "mediterranean",
    "american", "authentic", "famous", "original", "steak", "lobster", "noodle", "noodles", "curry", "ramen",
    "smoothie", "smoothies", "bubble", "boba", "crepe", "espresso", "dining", "hall", "health", "ice",
    "cream", "creamery", "pasticceria", "focacceria", "clam", "seafood", "fish", "beer", "cocktail",
    "roasters", "sweets", "dessert", "desserts", "fast", "iii", "3", "el", "la", "le", "de", "di",
)

/** Lower case, accents removed, '&' as 'and', apostrophes dropped, other punctuation as spaces. */
fun normalizeName(s: String): List<String> {
    val base = Normalizer.normalize(s, Normalizer.Form.NFKD).replace(Regex("\\p{M}+"), "")
        .lowercase()
        .replace(Regex("['’]s\\b"), "")
        .replace("&", " and ")
        .replace(Regex("['’`´]"), "")
        .replace(Regex("[^a-z0-9]+"), " ")
        .trim()
    val tokens = base.split(' ').filter { it.isNotEmpty() }
    val kept = tokens.filter { it !in NAME_STOPWORDS }
    return kept.ifEmpty { tokens }
}

fun levenshtein(a: String, b: String): Int {
    if (a == b) return 0
    if (a.isEmpty()) return b.length
    if (b.isEmpty()) return a.length
    var prev = IntArray(b.length + 1) { it }
    var cur = IntArray(b.length + 1)
    for (i in 1..a.length) {
        cur[0] = i
        for (j in 1..b.length) {
            val c = if (a[i - 1] == b[j - 1]) 0 else 1
            cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        }
        val t = prev; prev = cur; cur = t
    }
    return prev[b.length]
}

/**
 * Name similarity in [0, 1]: the best of the edit-distance ratio of the names without spaces,
 * the token Jaccard index, and containment (0.9 when the shorter name's tokens all occur in the
 * longer one as whole words, and the shorter name holds at least one word that is not generic and
 * is at least 5 characters long without spaces; "El Arco" is not contained in "El Arcoiris").
 */
fun nameSimilarity(a: String, b: String): Double {
    val ta = normalizeName(a)
    val tb = normalizeName(b)
    if (ta.isEmpty() || tb.isEmpty()) return 0.0
    val ca = ta.joinToString("")
    val cb = tb.joinToString("")
    if (ca == cb) return 1.0
    val lev = 1.0 - levenshtein(ca, cb).toDouble() / maxOf(ca.length, cb.length)
    val sa = ta.toSet()
    val sb = tb.toSet()
    val jac = sa.intersect(sb).size.toDouble() / sa.union(sb).size
    val (short, long) = if (ca.length <= cb.length) (ta to tb) else (tb to ta)
    val cShort = short.joinToString("")
    val distinctive = short.any { it !in GENERIC_WORDS && it.length >= 3 }
    val contained = cShort.length >= 5 && distinctive && long.toSet().containsAll(short)
    return maxOf(lev, jac, if (contained) 0.9 else 0.0)
}

/**
 * Similarity of the distinctive words only (generic words such as "pizza", "kitchen", "grill"
 * removed): 0 when either name has none. "Kam Fung Kitchen" and "KAM FUNG CHINESE RESTAURANT"
 * score 1; "Zhangs Garden" and "SUN GARDEN" score low although they share a word. A name whose
 * distinctive words are a proper subset of the other's is not a match by this measure ("NEW CHINA"
 * against "China Tang Kitchen"): at one address that is usually a different business.
 */
fun coreSimilarity(a: String, b: String): Double {
    val ta = normalizeName(a).filter { it !in GENERIC_WORDS }
    val tb = normalizeName(b).filter { it !in GENERIC_WORDS }
    if (ta.isEmpty() || tb.isEmpty()) return 0.0
    val ca = ta.joinToString("")
    val cb = tb.joinToString("")
    if (ca == cb) return 1.0
    val lev = 1.0 - levenshtein(ca, cb).toDouble() / maxOf(ca.length, cb.length)
    val sa = ta.toSet()
    val sb = tb.toSet()
    if (sa == sb) return 1.0
    val jac = sa.intersect(sb).size.toDouble() / sa.union(sb).size
    return maxOf(lev, jac)
}

private val STREET_WORDS = mapOf(
    "avenue" to "ave", "av" to "ave", "street" to "st", "road" to "rd", "boulevard" to "blvd",
    "place" to "pl", "parkway" to "pkwy", "court" to "ct", "drive" to "dr", "lane" to "ln",
    "terrace" to "ter", "highway" to "hwy", "square" to "sq", "plaza" to "plz", "east" to "e",
    "west" to "w", "north" to "n", "south" to "s", "saint" to "st",
)

/** "326 Kingston Avenue" and "326 KINGSTON AVE" give the same key; "1st" is "1". */
fun addressKey(house: String?, street: String?): String? {
    if (house.isNullOrBlank() || street.isNullOrBlank()) return null
    val h = house.lowercase().replace(Regex("\\s+"), "")
    val s = street.lowercase().replace(Regex("[^a-z0-9 ]"), " ").split(Regex("\\s+")).filter { it.isNotEmpty() }
        .map { w -> STREET_WORDS[w] ?: w.replace(Regex("^(\\d+)(st|nd|rd|th)$"), "$1") }
        .joinToString(" ")
    return if (s.isEmpty()) null else "$h|$s"
}

/** Great-circle distance in metres. */
fun distanceM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val r = 6_371_000.0
    val p1 = Math.toRadians(lat1)
    val p2 = Math.toRadians(lat2)
    val dp = p2 - p1
    val dl = Math.toRadians(lon2 - lon1)
    val h = Math.sin(dp / 2).let { it * it } + cos(p1) * cos(p2) * Math.sin(dl / 2).let { it * it }
    return 2 * r * asin(sqrt(h))
}

/** One side of a match: an id, its names, an optional coordinate and address key. */
data class MatchEntity(
    val id: String,
    val names: List<String>,
    val lat: Double?,
    val lon: Double?,
    val addressKey: String?,
)

data class MatchPair(val left: String, val right: String, val nameScore: Double, val distanceM: Double?, val rule: String)

/**
 * Accept rules (documented in the README):
 *  - `near`:    name similarity >= [MatchRules.nearName] and distance <= [MatchRules.nearM]
 *  - `close`:   name similarity >= [MatchRules.closeName], distance <= [MatchRules.closeM] and
 *               distinctive-word similarity ([coreSimilarity]) >= [MatchRules.coreName]
 *  - `address`: same house number and street, distinctive-word similarity >= [MatchRules.coreName],
 *               and distance <= [MatchRules.addressM] when both coordinates exist
 */
data class MatchRules(
    val nearName: Double = 0.80,
    val nearM: Double = 120.0,
    val closeName: Double = 0.65,
    val closeM: Double = 35.0,
    val coreName: Double = 0.75,
    val addressM: Double = 300.0,
)

/**
 * Candidate pairs within [MatchRules.addressM] of each other or at the same address, scored, then
 * assigned one to one in order of name score (then distance).
 */
fun matchEntities(left: List<MatchEntity>, right: List<MatchEntity>, rules: MatchRules = MatchRules()): List<MatchPair> {
    val cell = 0.004 // degrees; about 440 m north-south, 340 m east-west at this latitude
    fun key(lat: Double, lon: Double) = floor(lat / cell).toLong() * 1_000_003L + floor(lon / cell).toLong()
    val grid = HashMap<Long, MutableList<MatchEntity>>()
    val byAddress = HashMap<String, MutableList<MatchEntity>>()
    for (r in right) {
        if (r.lat != null && r.lon != null) grid.getOrPut(key(r.lat, r.lon)) { ArrayList() } += r
        if (r.addressKey != null) byAddress.getOrPut(r.addressKey) { ArrayList() } += r
    }
    val candidates = ArrayList<MatchPair>()
    for (l in left) {
        val near = LinkedHashSet<MatchEntity>()
        if (l.lat != null && l.lon != null) {
            val gy = floor(l.lat / cell).toLong()
            val gx = floor(l.lon / cell).toLong()
            for (dy in -1L..1L) for (dx in -1L..1L) grid[(gy + dy) * 1_000_003L + (gx + dx)]?.let { near += it }
        }
        if (l.addressKey != null) byAddress[l.addressKey]?.let { near += it }
        for (r in near) {
            val score = l.names.maxOf { a -> r.names.maxOf { b -> nameSimilarity(a, b) } }
            val d = if (l.lat != null && l.lon != null && r.lat != null && r.lon != null)
                distanceM(l.lat, l.lon, r.lat, r.lon) else null
            val sameAddress = l.addressKey != null && l.addressKey == r.addressKey
            val core by lazy { l.names.maxOf { a -> r.names.maxOf { b -> coreSimilarity(a, b) } } }
            val rule = when {
                d != null && score >= rules.nearName && d <= rules.nearM -> "near"
                d != null && score >= rules.closeName && d <= rules.closeM && core >= rules.coreName -> "close"
                sameAddress && core >= rules.coreName && (d == null || d <= rules.addressM) -> "address"
                else -> null
            } ?: continue
            candidates += MatchPair(l.id, r.id, score, d, rule)
        }
    }
    candidates.sortWith(compareByDescending<MatchPair> { it.nameScore }.thenBy { it.distanceM ?: Double.MAX_VALUE })
    val usedL = HashSet<String>()
    val usedR = HashSet<String>()
    val out = ArrayList<MatchPair>()
    for (c in candidates) {
        if (c.left in usedL || c.right in usedR) continue
        usedL += c.left; usedR += c.right
        out += c
    }
    return out
}
