package brooklyn

import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.sql.DriverManager
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Task generation (deterministic, split by restaurant) and the verifier, on the synthetic fixture:
 * hand-written good answers that must pass and bad ones, including subtle hallucinations, that must fail.
 *
 * Fixture facts used below: Tomasso's Slice (1001, Williamsburg, pizza) grade A on 2025-03-01 (B on
 * 2024-01-10), score 12, 04L mice (critical), 311 complaint 2025-04-01, 2 Google reviews averaging 4.0
 * (the Google Local listing says 4.5), Baked Ziti $13.00 on its archived menu; Vela Forno (1002) grade A
 * on 2025-05-02, score 5, 311 complaint 2025-06-02 (Letter Grading); Golden Dragon (1004) no letter
 * grade, score 30 on 2024-11-11.
 */
class TasksTest {
    private val fixture: Path = Paths.get(javaClass.getResource("/fixture/raw/overpass.json")!!.toURI()).parent
    private val example: Path = Paths.get("").toAbsolutePath()
    private val cutoff = LocalDate.parse(DEFAULT_CUTOFF)

    private fun fixtureDb(dir: Path = Files.createTempDirectory("brooklyn-tasks")): Path {
        val cfg = IngestConfig(dataDir = dir, raw = fixture, log = {})
        buildDatabase(cfg)
        return cfg.db
    }

    private val db: Path by lazy { fixtureDb() }

    private fun task(
        category: String, kind: String, q: String, restaurants: List<String>, facts: List<Fact>,
        answerSet: List<String>? = null, offline: Boolean = true,
    ) = Task("t-$kind", "test", category, kind, q, restaurants, facts, answerSet, emptyList(), "", offline, 0, cutoff.toString())

    private val tomasso = Fact("name", "Tomasso's Slice")
    private val calls = listOf(ToolCall("search_restaurants", """{"query":"Tomasso's Slice"}"""), ToolCall("get_restaurant", """{"id":"1001"}"""))

    private class Case(val label: String, val task: Task, val answer: String, val pass: Boolean, val why: String? = null, val calls: List<ToolCall>? = null)

    private fun cases(s: ToolServer): List<Case> {
        val golden = s.rest("1004").name
        val grade = task("lookup", "grade", "What is the current DOHMH letter grade of Tomasso's Slice?", listOf("1001"),
            listOf(tomasso, Fact("grade", "A"), Fact("date", "2025-03-01", required = false)))
        val last = task("lookup", "last_inspection", "When was $golden last inspected?", listOf("1004"),
            listOf(Fact("name", golden), Fact("date", "2024-11-11"), Fact("count", "30", required = false)))
        val filter = task("filter", "cuisine_area_grade", "Which pizza places are there in Williamsburg?", listOf("1001"),
            listOf(tomasso, Fact("count", "1", required = false)), answerSet = listOf("1001"))
        val multi = task("multihop", "critical_and_reviews",
            "How many critical violations has Tomasso's Slice been cited for in the 12 months before 2026-11-13, and what do Google reviewers say about it?",
            listOf("1001"), listOf(tomasso, Fact("count", "0"), Fact("number", "4", tol = 0.051)))
        val fake = task("unanswerable", "nonexistent", "What is the health inspection grade of Golden Lantern Bistro in Park Slope?", emptyList(),
            listOf(Fact("refusal", "not_found")))
        val outside = task("unanswerable", "outside_brooklyn", "What are the A-graded Thai restaurants in Astoria, Queens?", emptyList(),
            listOf(Fact("refusal", "out_of_scope")))
        val complaint = task("complaints", "complaints", "When was the most recent 311 food complaint about Vela Forno, and what was it about?", listOf("1002"),
            listOf(Fact("name", "Vela Forno"), Fact("date", "2025-06-02"), Fact("topic", "Letter Grading", topicAlts("Letter Grading")), Fact("absent", COMPLAINT_POISONING_ABSENT)))
        val menu = task("menu", "price", "How much is the Baked Ziti at Tomasso's Slice, according to its archived menu?", listOf("1001"),
            listOf(tomasso, Fact("money", "13.00", listOf("Baked Ziti"))))
        val compare = task("comparison", "compare", "Which did better at its latest health inspection, Tomasso's Slice or Vela Forno?", listOf("1001", "1002"),
            listOf(
                tomasso, Fact("name", "Vela Forno"), Fact("count", "12", listOf("point"), of = "1001"), Fact("count", "5", listOf("point"), of = "1002"),
                Fact("winner", "1002", listOf("score")), Fact("absent", HIGHER_SCORE_BETTER),
            ))
        val multiUnits = task("multihop", "critical_and_reviews",
            "How many critical violations has Tomasso's Slice been cited for in the 12 months before 2026-11-13, and what do Google reviewers say about it?",
            listOf("1001"), listOf(tomasso, Fact("count", "2", listOf("critical")), Fact("number", "3.96", listOf("star"), tol = 0.051)))
        val closed = task("unanswerable", "closed", "Is $golden open for business, according to the health department's records as of 2026-11-13?", listOf("1004"),
            listOf(Fact("name", golden), Fact("date", "2024-11-11"), Fact("keyword", "closed", listOf("closed", "closure", "shut")), Fact("absent", REOPENED_CLAIM)))
        val violation = task("lookup", "violation", "What violations were cited at Tomasso's Slice's most recent health inspection?", listOf("1001"),
            listOf(tomasso, Fact("violation", "04L", listOf("mice", "live")), Fact("date", "2025-03-01", required = false)))
        return listOf(
            // Single lookup: grade.
            Case("grade, plain", grade, "Tomasso's Slice has a DOHMH letter grade of A, given on March 1, 2025.", true),
            Case("grade, with history", grade, "**Tomasso's Slice** (216 Bedford Avenue, Williamsburg) currently holds an A grade. It was graded B on January 10, 2024, before its latest inspection.", true),
            Case("grade, ISO date", grade, "The current grade at Tomasso's Slice is A (graded 2025-03-01).", true),
            Case("grade, wrong letter", grade, "Tomasso's Slice has a grade of B.", false, "grade=A"),
            Case("grade, a past grade as current", grade, "Tomasso's Slice has an A grade and is now graded B.", false, "not the current grade"),
            Case("grade, date one day off", grade, "Tomasso's Slice has an A grade, given on March 2, 2025.", false, "date March 2, 2025"),
            Case("grade, invented earlier grade", grade, "Tomasso's Slice has grade A now; it was graded C in 2023.", false, "grade C"),
            Case("grade, misspelled name", grade, "Tomaso's Slice has a grade of A.", false, "name=Tomasso's Slice"),
            Case("grade, invented CAMIS", grade, "Tomasso's Slice (CAMIS 50012345) has a grade of A.", false, "id 50012345"),
            Case("grade, argument of the wrong type", grade, "Tomasso's Slice has a grade of A.", false, "expected string",
                listOf(ToolCall("get_restaurant", """{"id":1001}"""))),
            Case("grade, unknown tool", grade, "Tomasso's Slice has a grade of A.", false, "unknown tool 'get_grade'",
                listOf(ToolCall("get_grade", """{"id":"1001"}"""))),
            Case("grade, arguments not JSON", grade, "Tomasso's Slice has a grade of A.", false, "not a JSON object",
                listOf(ToolCall("get_restaurant", "id=1001"))),
            Case("grade, unexpected argument", grade, "Tomasso's Slice has a grade of A.", false, "unexpected property 'borough'",
                listOf(ToolCall("search_restaurants", """{"query":"Tomasso's","borough":"Brooklyn"}"""))),
            // A restaurant with no letter grade.
            Case("no grade, stated as none", last, "$golden was last inspected on November 11, 2024 and scored 30 points; it has no letter grade posted.", true),
            Case("no grade, invented grade", last, "$golden was last inspected on November 11, 2024 and holds an A grade.", false, "grade A"),
            Case("no grade, wrong date", last, "$golden was last inspected on November 12, 2024.", false, "date=2024-11-11"),
            // Filter: the list must be the answer set.
            Case("filter, list", filter, "One pizza place in Williamsburg:\n- Tomasso's Slice, 216 Bedford Avenue (grade A)", true),
            Case("filter, real place that does not match", filter, "Two pizza places:\n- Tomasso's Slice, 216 Bedford Avenue (grade A)\n- Vela Forno, 575 Henry Street (grade A)", false, "'Vela Forno' does not match"),
            Case("filter, invented place", filter, "Two pizza places:\n- Tomasso's Slice, 216 Bedford Avenue (grade A)\n- **Bedford Pie Kitchen**, 230 Bedford Avenue (grade A)", false, "restaurant 'Bedford Pie Kitchen'"),
            Case("filter, invented place in prose", filter, "Tomasso's Slice on Bedford Avenue is the one; Golden Crust Pizzeria nearby is also good.", false, "restaurant 'Golden Crust Pizzeria'"),
            Case("filter, current grade wrong", filter, "- Tomasso's Slice, 216 Bedford Avenue (grade B)", false, "not the current grade"),
            Case("filter, nothing found", filter, "There are no pizza places in Williamsburg.", false, "name=Tomasso's Slice"),
            // Multi-hop: critical violations in 12 months and reviews.
            Case("multi-hop, good", multi, "Since November 13, 2025, Tomasso's Slice has had no critical violations. Google Local reviewers give it an average of 4 stars over 2 reviews; one praises the classic thin slice.", true),
            Case("multi-hop, old violation counted", multi, "In the last 12 months Tomasso's Slice was cited for 1 critical violation (evidence of mice). Reviewers average 4.0 stars.", false, "count=0"),
            Case("multi-hop, listing rating instead of the reviews' average", multi, "Tomasso's Slice has no critical violations since November 13, 2025; Google reviewers rate it 4.5 stars.", false, "number=4"),
            Case("multi-hop, invented inspection date", multi, "Tomasso's Slice has had no critical violations; its last inspection, on June 5, 2026, was clean. Reviews average 4 stars.", false, "date June 5, 2026"),
            // Unanswerable: nonexistent and outside Brooklyn.
            Case("nonexistent, refusal", fake, "I couldn't find a restaurant called Golden Lantern Bistro in Park Slope, or anywhere in the Brooklyn data.", true),
            Case("nonexistent, no record", fake, "There is no record of Golden Lantern Bistro in the DOHMH, OpenStreetMap or Google Local data for Brooklyn, so I can't give its grade.", true),
            Case("nonexistent, answered", fake, "Golden Lantern Bistro in Park Slope has a grade of A.", false, "refusal=not_found"),
            Case("nonexistent, refusal then invention", fake, "I couldn't find Golden Lantern Bistro under that name, but it was graded A on March 3, 2025.", false, "grade A"),
            Case("nonexistent, invented look-alike", fake, "I couldn't find Golden Lantern Bistro; you may mean **Golden Lantern Cafe** nearby.", false, "restaurant 'Golden Lantern Cafe'"),
            Case("outside, refusal", outside, "Astoria is in Queens, outside Brooklyn; the data I have covers only Brooklyn, so I can't answer.", true),
            Case("outside, answered", outside, "In Astoria, try **Thai Lotus Kitchen** (grade A).", false, "refusal=out_of_scope"),
            // Complaints, menus, comparisons, violations.
            Case("complaint, good", complaint, "The latest 311 complaint about Vela Forno was filed on June 2, 2025: a Food Establishment complaint about letter grading.", true),
            Case("complaint, US date format", complaint, "Vela Forno's most recent 311 complaint (Food Establishment, letter grade not visible) is from 6/2/2025.", true),
            Case("nonexistent, does not appear", fake, "Golden Lantern Bistro doesn't appear in any of the Brooklyn sources (DOHMH, OpenStreetMap, Google Local), so it has no grade I can report.", true),
            Case("filter, prose", filter, "Only Tomasso's Slice, at 216 Bedford Avenue, matches. It holds an A.", true),
            Case("complaint, date one day off", complaint, "The latest 311 complaint about Vela Forno was filed on June 3, 2025 (letter grading).", false, "date=2025-06-02"),
            Case("menu, good", menu, "Tomasso's Slice's archived menu lists Baked Ziti at $13.", true),
            Case("menu, wrong price", menu, "Tomasso's Slice's archived menu lists Baked Ziti at $12.00.", false, "money=13.00"),
            Case("comparison, good", compare, "Vela Forno did better: its latest inspection scored 5 points against 12 for Tomasso's Slice.", true),
            Case("comparison, wrong score", compare, "Vela Forno scored 5 and Tomasso's Slice scored 14.", false, "count=12"),
            Case("violation, with code", violation, "At its March 1, 2025 inspection, Tomasso's Slice was cited for 04L (critical): evidence of mice.", true),
            Case("violation, in words", violation, "Inspectors found evidence of live mice at Tomasso's Slice on March 1, 2025.", true),
            Case("violation, wrong violation", violation, "Tomasso's Slice was cited for holding hot food below 140 °F on March 1, 2025.", false, "violation=04L"),
            // Comparisons: the right restaurant picked, each value with its own restaurant, the scale the right way round.
            Case("comparison, values first", compare, "Tomasso's Slice scored 12 points and Vela Forno scored 5; lower is better, so Vela Forno did better.", true),
            Case("comparison, loser named first", compare, "Tomasso's Slice scored 12, a higher score than Vela Forno's 5 points, so Vela Forno did better.", true),
            Case("comparison, wrong winner", compare, "Tomasso's Slice did better: its latest inspection scored 12 points, against 5 for Vela Forno.", false, "winner=1002"),
            Case("comparison, values swapped", compare, "Vela Forno did better: its latest inspection scored 12 points, against 5 for Tomasso's Slice.", false, "wrong restaurants"),
            Case("comparison, scale reversed", compare, "Tomasso's Slice did better: it scored 12 points against 5 for Vela Forno, and higher scores are better.", false, "higher scores are better"),
            // Several values offered, or a second value for the same thing.
            Case("multi-hop, counts and stars", multiUnits, "Since November 13, 2025, Tomasso's Slice has been cited for 2 critical violations. Its Google reviews average 3.96 stars.", true),
            Case("multi-hop, rounded stars", multiUnits, "Tomasso's Slice: two critical violations in the 12 months to 2026-11-13; reviewers average 4.0 stars.", true),
            Case("multi-hop, count among guesses", multiUnits, "Tomasso's Slice has been cited for 0, 1, 2 or 3 critical violations since November 13, 2025. Reviews average 3.96 stars.", false, "count=2"),
            Case("multi-hop, count hedged", multiUnits, "Tomasso's Slice has been cited for about 2 critical violations since November 13, 2025; 3.96 stars.", false, "count=2"),
            Case("multi-hop, a second count", multiUnits, "Tomasso's Slice has been cited for 7 critical violations since November 13, 2025 (2 of them since the spring). Reviews average 3.96 stars.", false, "also states 7"),
            Case("multi-hop, integer near the average", multiUnits, "Tomasso's Slice has 2 critical violations and 4 health complaints; its reviews average 4.26 stars.", false, "number=3.96"),
            Case("multi-hop, invented neighbor", multiUnits, "Tomasso's Slice: 2 critical violations; reviewers give it 3.96 stars. Nearby, Moonlight Garden Noodles is better rated.", false, "restaurant 'Moonlight Garden Noodles'"),
            // Negation.
            Case("grade, negated", grade, "Tomasso's Slice at 216 Bedford Avenue does not have grade A; it is pending.", false, "only negated"),
            Case("grade, other letter negated", grade, "Tomasso's Slice is not graded B any more; it now has grade A.", true),
            Case("no grade, date negated", last, "$golden was not inspected on November 11, 2024; its last inspection was in 2019.", false, "only negated"),
            Case("closed, good", closed, "DOHMH closed $golden at its inspection on November 11, 2024, and there is no later inspection, so I can't say it is open.", true),
            Case("closed, negated", closed, "$golden is open; it was not closed at its November 11, 2024 inspection.", false, "keyword=closed"),
            // Prices: the dish's price, and only prices on the archived menu.
            Case("menu, true price as an old one", menu, "Tomasso's Slice's archived menu lists Baked Ziti at $14.00 (it was $13.00 years ago).", false, "money=13.00"),
            Case("menu, price before the dish", menu, "On Tomasso's Slice's archived menu, $13 buys the Baked Ziti.", true),
            Case("menu, invented possessive name", menu, "Golden Lantern Kitchen's archived menu lists Baked Ziti at $13.00, like Tomasso's Slice's.", false, "restaurant 'Golden Lantern Kitchen's'"),
            // Guesses, additions and wrong topics on top of true answers; a grade stated without the word "grade".
            Case("grade, held without the word grade", grade, "Tomasso's Slice (216 Bedford Avenue) currently holds an A from the health department, posted March 1, 2025.", true),
            Case("grade, plain letter after is", grade, "The letter grade posted at Tomasso's Slice? It is A.", true),
            Case("grade, true one then a guess", grade, "Tomasso's Slice probably has grade A, but it might be C.", false, "grade C"),
            Case("grade, a guess after the true one", grade, "Tomasso's Slice has grade A. It might be C now.", false, "grade C"),
            Case("grade, only as a guess", grade, "Tomasso's Slice probably has grade A.", false, "only hedged"),
            Case("menu, two prices offered", menu, "Tomasso's Slice's archived menu lists Baked Ziti at $13.00 or $11.00.", false, "only hedged"),
            Case("menu, price as a guess", menu, "Baked Ziti at Tomasso's Slice is probably $13.", false, "only hedged"),
            Case("closed, reopening claimed", closed, "DOHMH closed $golden at its inspection on November 11, 2024; it has since reopened and you can eat there.", false, "the answer says"),
            Case("closed, open now negated", closed, "DOHMH closed $golden on November 11, 2024. Nothing later is on record, so I can't tell you it is open now.", true),
            Case("violation, an invented second code", violation, "At its March 1, 2025 inspection, Tomasso's Slice was cited for 04L (critical): evidence of mice, and 04M (critical): live roaches.", false, "violation 04M"),
            Case("violation, an invented pest", violation, "Inspectors found evidence of mice and live roaches at Tomasso's Slice on March 1, 2025.", false, "'roaches'"),
            Case("nonexistent, refusal then the cutoff as an event", fake, "I couldn't find Golden Lantern Bistro in Park Slope. It was last inspected on November 13, 2026.", false, "only a bound"),
            Case("nonexistent, refusal with the cutoff as a bound", fake, "I couldn't find Golden Lantern Bistro in the Brooklyn data, which runs up to November 13, 2026.", true),
            Case("complaint, wrong topic", complaint, "The latest 311 complaint about Vela Forno was a food establishment complaint about food poisoning, filed on June 2, 2025.", false, "topic=Letter Grading"),
            Case("complaint, topic in other words", complaint, "Vela Forno's latest 311 complaint, on June 2, 2025, said its letter grade was not visible.", true),
        )
    }

    @Test
    fun verifierPassesGoodAnswersAndRejectsBadOnes() {
        ToolServer(db, ToolMode.SNAPSHOT).use { s ->
            val v = Verifier(s, cutoff)
            val cs = cases(s)
            val wrong = cs.mapNotNull { c ->
                val verdict = v.verify(c.task, c.answer, c.calls ?: calls)
                when {
                    verdict.pass != c.pass -> "${c.label}: expected ${if (c.pass) "pass" else "fail"}, got $verdict"
                    c.why != null && !verdict.toString().contains(c.why) -> "${c.label}: failed without '${c.why}': $verdict"
                    else -> null
                }
            }
            assertEquals(emptyList(), wrong)
            assertTrue(cs.count { it.pass } >= 15 && cs.count { !it.pass } >= 25, "good ${cs.count { it.pass }}, bad ${cs.count { !it.pass }}")
        }
    }

    @Test
    fun liveOnlyTasksAreNotScoredOffline() {
        ToolServer(db, ToolMode.SNAPSHOT).use { s ->
            val t = task("live", "live_only", "Is Tomasso's Slice open right now?", listOf("1001"), emptyList(), offline = false)
            val verdict = Verifier(s, cutoff).verify(t, "Yes, it is open.", calls)
            assertFalse(verdict.scored)
            assertFalse(verdict.pass)
        }
    }

    @Test
    fun gradesDatesAndNumbersAreReadFromText() {
        assertEquals(listOf("A"), gradeMentions("It has an A grade.").map { it.letter })
        assertEquals(listOf("A"), gradeMentions("an A-graded place").map { it.letter })
        assertEquals(listOf("B"), gradeMentions("Grade: B").map { it.letter })
        assertEquals(listOf("B"), gradeMentions("A grade of B was given.").map { it.letter })
        assertEquals(listOf("C"), gradeMentions("It was graded C in 2023.").map { it.letter })
        assertEquals(emptyList(), gradeMentions("places graded A, B or C").map { it.letter })
        assertEquals(emptyList(), gradeMentions("with a grade of B or better").map { it.letter })
        assertEquals(emptyList(), gradeMentions("its grade is pending").map { it.letter })
        val ds = dateMentions("on 2025-03-01, March 2, 2025, 3 March 2025, 3/4/2025 and in May 2024")
        assertEquals(
            listOf("2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", null),
            ds.map { it.date()?.toString() },
        )
        assertEquals(5, ds.last().month)
        assertEquals("March 1, 2025", longDate("2025-03-01"))
    }

    @Test
    fun menuDishesAreUnambiguousPricedLines() {
        val text = "LENTIL SOUP\n$7.99\nCHICKEN SOUP\nWith vegetables $7.99\nSHEPHERD'S SALAD\nSM. $11.50 | Lg. $15.50\nBaked Ziti $13.00\n" +
            "HASH BROWNS (2)\n$6.5\nLENTIL SOUP\n$8.99\nAvocado house salad.\n$ 13.00\nSeasoned, breaded, & fried calamari w/ onion\n$17.00"
        // A name and a price on separate lines are not paired; "With vegetables" is an option.
        assertEquals(listOf("Baked Ziti" to "13.00"), menuDishes(text))
        // Layouts seen on archived menus where pairing the lines gives a wrong price: a section header above a
        // dish ("Banh Mi", "Coffee by Parlor"), the price before the name, a header with a price for its items.
        val layouts = "Banh Mi\n$9.95\n13. Grilled Chicken Banh Mi\nCoffee by Parlor\n$4\nDrip Coffee\n$ 11.50\nRice Vermicelli with Curry Beef Stew\n" +
            "$ 10.50\n11. Vegetarian Noodle Soup\nCocktails\n$20\n$18\n"
        assertEquals(emptyList(), menuDishes(layouts))
        // Headers, sizes, options, groups and descriptions on one line with a price are not dishes either.
        val labels = "Snacks $6\nSides $7\nSpreads $8\nSparkling Wine $14\nNon-Alcoholic Cocktails $11\nSpecials & Hoppy Hour $13\nLrg $16\n" +
            "Half Topping = $2\nShrimp or Beef $16\nCoffee by Parlor $4\nRotating food pop-up's $13\nSteak Add $8.00\n" +
            "House-made classic Chinese food staple $4.50\nBehance $0.00\nGarlic Knots $5.50\nPad See-Ew – $14\nFish Po’ Boy $20.00\n" +
            "Curated Cocktails - $15\nOnly $13.99\nCoconut Water, Cucumber, Pink Pitaya Powder, Aloe\nVera $8.50\nCelery, Raisins, Walnuts, Sesame Seeds, and\n" +
            "Sesame Ginger Dressing $11\nGluten-Free $4.50\nBeer and a Shot for $10\nLENTIL SOUP\nPureed red lentils $7.99\nIn Tomato and Garlic Broth $15.95\nSeasonal $7\nAll Cocktails Classics & Originals $16\nSub cheese dip - $1\nValet extra $160\nDelivery Min: $10.00\nBottled $5\nTwelve $ 28\nDemo $0.05\nSubstitutions can be made $42.00\nBreakfast Bowls $10"
        assertEquals(listOf("Garlic Knots" to "5.50", "Pad See-Ew" to "14", "Fish Po’ Boy" to "20.00"), menuDishes(labels))
        val vw = ViolationWords(listOf("Evidence of mice or live mice in establishment's food or non-food areas.", "Hot TCS food item not held at or above 140 °F."))
        assertEquals("mice", vw.keywords("Evidence of mice or live mice in establishment's food or non-food areas.").first())
    }

    @Test
    fun menuDishesSkipAddOnSectionsAndNamesInsideOtherLines() {
        val text = listOf(
            "Two Eggs Any Style", "With Veggie Sausage $ 10.95", "Pancakes $9.50",
            "Add-ons", "Veggie Sausage $ 2.00", "Avocado $3.00",
            "Lunch", "Tuna Melt $12.00",
            "Salad Add-ons", "Chicken Salad $4.50",
            "Build Your Own With Add Ons", "Buffalo Sauce $0.50",
        ).joinToString("\n")
        // Avocado is under Add-ons; Veggie Sausage is also inside another line; Pancakes and Tuna Melt stand alone.
        assertEquals(listOf("Pancakes" to "9.50", "Tuna Melt" to "12.00"), menuDishes(text))
    }

    @Test
    fun quotasSumToTheSplitSizes() {
        for (sizes in listOf(TaskSizes.FULL, TaskSizes.QUICK)) {
            val q = sizes.quotas()
            for (sp in Split.entries) {
                assertEquals(sizes.offline.getValue(sp), TASK_KINDS.filter { it.weight > 0 }.sumOf { q.getValue(it.kind).getValue(sp) })
                assertEquals(sizes.live.getValue(sp), q.getValue("live_only").getValue(sp))
            }
        }
        assertEquals(480, TaskSizes.FULL.quotas().getValue("grade").getValue(Split.TRAIN))
    }

    @Test
    fun theSplitIsASeededHashOfTheRestaurant() {
        val ids = (1..20_000).map { "id$it" }
        val a = ids.map { splitOf(it, 4211) }
        assertEquals(a, ids.map { splitOf(it, 4211) })
        val train = a.count { it == Split.TRAIN } / 20_000.0
        val test = a.count { it == Split.TEST } / 20_000.0
        assertTrue(train in 0.79..0.81 && test in 0.09..0.11, "train $train test $test")
        val b = ids.map { splitOf(it, 4212) }
        assertTrue(a.zip(b).count { it.first != it.second } > 5_000)
    }

    @Test
    fun generationIsDeterministicSplitByRestaurantAndAnswerable() {
        val sizes = TaskSizes(Split.entries.associateWith { 30 }, Split.entries.associateWith { 2 })
        fun gen() = ToolServer(db, ToolMode.SNAPSHOT).use { s -> TaskGenerator(s, cutoff, 5, 9, sizes).generate() }
        val a = gen()
        val b = gen()
        for (sp in Split.entries) assertEquals(taskFileText(a.getValue(sp)), taskFileText(b.getValue(sp)))
        val all = a.values.flatten()
        assertTrue(all.size >= 20, "only ${all.size} tasks from the fixture")
        assertTrue(all.map { it.category }.toSet().containsAll(listOf("lookup", "filter", "complaints", "menu", "unanswerable", "live")), all.map { it.kind }.toSet().toString())
        assertEquals(mapOf("train&test" to emptySet(), "train&dev" to emptySet(), "dev&test" to emptySet<String>()), splitOverlaps(a))
        for (t in all) for (id in t.restaurants + t.answerSet.orEmpty()) assertEquals(t.split, splitOf(id, 9).label, "${t.id} $id")
        // Questions use the cutoff, never today's date.
        assertTrue(all.none { it.question.contains(LocalDate.now().toString()) && LocalDate.now() != cutoff })
        // Every fact is in the solution calls' results and every reference answer passes.
        assertEquals(emptyList(), checkReferences(db, all, cutoff))
        // And corrupted references fail.
        val muts = mutationCheck(db, all, cutoff)
        assertTrue(muts.isNotEmpty())
        for ((k, ab) in muts) assertEquals(ab.second, ab.first, "mutation $k: ${ab.first} of ${ab.second} rejected")
        // The file round-trips.
        val f = Files.createTempFile("tasks", ".jsonl")
        Files.writeString(f, taskFileText(a.getValue(Split.TRAIN)))
        assertEquals(taskFileText(a.getValue(Split.TRAIN)), taskFileText(readTasks(f)))
    }

    @Test
    fun verifyReportsTaskFilesFromAnEarlierDatabaseAsStale() {
        val data = Files.createTempDirectory("data")
        val home = Files.createTempDirectory("home")
        Files.createDirectories(home.resolve("pins"))
        Files.copy(fixture.resolve("wayback/pins.tsv"), home.resolve(MENU_PINS))
        Files.createDirectories(home.resolve("searxng"))
        Files.copy(example.resolve(SEARXNG_SCRIPT), home.resolve(SEARXNG_SCRIPT))
        Files.copy(example.resolve("build.gradle.kts"), home.resolve("build.gradle.kts"))
        Files.writeString(home.resolve("manifest.json"), """{"cutoff": "2026-11-13", "seeds": {"match_sample": 7, "tasks": 5, "task_split": 9}}""")
        val dbFile = fixtureDb(data)
        fun quiet(f: () -> Int): Pair<Int, String> {
            val buf = ByteArrayOutputStream()
            val old = System.out
            System.setOut(PrintStream(buf, true))
            return try { f() to buf.toString() } finally { System.setOut(old) }
        }
        val gen = quiet { tasksCmd(listOf("generate", "--home", home.toString(), "--data", data.toString())) }
        assertEquals(0, gen.first, gen.second)
        val first = Files.readString(taskFile(data, Split.TRAIN))
        val v = { a: Array<String> -> quiet { verify(listOf("--home", home.toString(), "--data", data.toString(), "--no-models") + a) } }
        assertEquals(0, v(arrayOf("--update")).first)
        val ok = v(emptyArray())
        assertEquals(0, ok.first, ok.second)
        assertTrue(ok.second.contains("ok        tasks.derived.split_test"), ok.second)
        // Regenerating gives the same bytes.
        assertEquals(0, quiet { tasksCmd(listOf("check", "--home", home.toString(), "--data", data.toString())) }.first)
        assertEquals(first, Files.readString(taskFile(data, Split.TRAIN)))
        // The database changes after the tasks were generated: the split files are stale.
        DriverManager.getConnection("jdbc:sqlite:$dbFile").use { c -> c.createStatement().use { it.executeUpdate("UPDATE inspections SET score = score + 1") } }
        val stale = v(emptyArray())
        assertEquals(1, stale.first, stale.second)
        assertTrue(stale.second.contains("STALE     tasks.derived.split_train") && stale.second.contains("table.inspections"), stale.second)
        // A different split seed is also a different input.
        assertEquals(0, quiet { tasksCmd(listOf("generate", "--home", home.toString(), "--data", data.toString())) }.first)
        Files.writeString(home.resolve("manifest.json"), Files.readString(home.resolve("manifest.json")).replace("\"task_split\": 9", "\"task_split\": 10"))
        assertTrue(v(emptyArray()).second.contains("seed.task_split"))
    }
}
