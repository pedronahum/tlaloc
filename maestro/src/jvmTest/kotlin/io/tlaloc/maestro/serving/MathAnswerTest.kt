package io.tlaloc.maestro.serving

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The MATH grader's normalizations, case by case as `math_equivalence.py` treats them. */
class MathAnswerTest {
    @Test fun theLastBoxIsTheAnswer() {
        assertEquals("\\frac{1}{2}", MathAnswer.lastBoxed("first \\boxed{3}, then \\boxed{\\frac{1}{2}}."))
        assertEquals("x^{2}", MathAnswer.lastBoxed("\\fbox{x^{2}}"))
        assertEquals("7", MathAnswer.lastBoxed("so $\\boxed 7$."))
        assertEquals(null, MathAnswer.lastBoxed("no box"))
        assertEquals(null, MathAnswer.lastBoxed("\\boxed{unclosed"))
    }

    @Test fun spellingsOfOneValueAreEquivalent() {
        for ((a, b) in listOf(
            "\\dfrac{1}{2}" to "\\frac12",
            "0.5" to "\\frac{1}{2}",
            "1/2" to "\\frac{1}{2}",
            "\\left( 3, 4 \\right)" to "(3,4)",
            "10\\%" to "10",
            "90^\\circ" to "90",
            "x = 5" to "5",
            "\\sqrt3" to "\\sqrt{3}",
            "5 \\text{ cm}" to "5",
            ".5" to "0.5",
            "\\$18.90" to "18.90",
        )) {
            assertTrue(MathAnswer.equivalent(a, b), "$a and $b")
        }
    }

    @Test fun differentAnswersDiffer() {
        assertFalse(MathAnswer.equivalent("\\frac{1}{3}", "0.333"))
        assertFalse(MathAnswer.equivalent("x+1", "1+x"))
        assertFalse(MathAnswer.equivalent(null, "1"))
        // A long left side of an equation is kept.
        assertFalse(MathAnswer.equivalent("abc = 5", "5"))
    }
}
