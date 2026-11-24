package brooklyn

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Places a list answer names but sets apart (not counted as extra) and places it offers as matches. */
class ExclusionTest {
    private fun excluded(answer: String, name: String): Boolean = excludedMention(answer, answer.indexOf(name).also { check(it >= 0) })

    @Test
    fun aPlaceTheAnswerSetsApartIsExcluded() {
        assertTrue(excluded("Only **Mis Taquitos** qualifies.\n\nThe other Mexican spot, **Nosh Express** (2817 Nostrand Ave), has no DOHMH grade on record, so it doesn't qualify.", "Nosh Express"))
        assertTrue(excluded("**Open at 1 PM:**\n- **Fancy Nancy**\n\n**Not open at 1 PM:**\n- **Norwind's** — closed on Mondays.", "Norwind's"))
        assertTrue(excluded("**Hours unknown (couldn't confirm):**\n- **HaaM** — 234 Union Avenue", "HaaM"))
        assertTrue(excluded("- **El Cofre** — 454 Myrtle Avenue\n\n**Nearby if you're willing to wander:**\n- **Bed-Stuy**: Anaiah (744 Myrtle Ave), Genao.", "Anaiah"))
        assertTrue(excluded("Note: Festac Grill & Lounge also has an A grade, but it's in East New York (North), not the City Line neighborhood.", "Festac Grill"))
        assertTrue(excluded("McDonald's in the same area has an A grade but is listed as a hamburger venue, not a sandwich spot.", "McDonald's"))
    }

    @Test
    fun aPlaceOfferedAsAMatchIsNot() {
        assertFalse(excluded("Two taco places have an A grade:\n- **East Coast Street Tacos** — 521 Stanley Avenue (Tex-Mex, grade A)\n- **Chipotle**", "East Coast Street Tacos"))
        assertFalse(excluded("Here are the Mexican places open at 9 PM:\n- **La Tienda** – 506 48th Street – open 6:00 AM–11:00 PM daily.", "La Tienda"))
        // A table row's cells are data: "no letter grade" there does not set the place apart.
        assertFalse(excluded("African options in East New York:\n\n| Name | Grade |\n|---|---|\n| **Olas Kitchen** | **Z** (no letter grade) | Opening hours not listed |", "Olas Kitchen"))
        // A caveat inside the match's own entry is not an exclusion.
        assertFalse(excluded("1. **Guac & Grill Mexican Eats** — 1146 Liberty Avenue. Grade B (as of 2026-09-17).", "Guac & Grill"))
    }
}
