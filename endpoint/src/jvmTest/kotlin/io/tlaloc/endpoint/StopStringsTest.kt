package io.tlaloc.endpoint

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The audit's SV-11: [StopStrings] holds back `longest - 1` UTF-16 units, and that cut can land
 * between an emoji's high and low surrogate. Emitting the high one alone gives a client a lone
 * surrogate — JSON `\ud83d`, rendered as `?` — and the low one arrives in a later chunk or at
 * finish. These drive the cut directly, which is why the class is internal and top-level.
 */
class StopStringsTest {

    /** Count surrogates that are not part of a pair: what a client would render as `?`. */
    private fun loneSurrogates(s: String): Int {
        var n = 0
        var i = 0
        while (i < s.length) {
            val c = s[i]
            when {
                Character.isHighSurrogate(c) && i + 1 < s.length && Character.isLowSurrogate(s[i + 1]) -> i += 2
                Character.isHighSurrogate(c) || Character.isLowSurrogate(c) -> { n++; i++ }
                else -> i++
            }
        }
        return n
    }

    private fun assertWhole(vararg parts: String) {
        parts.forEach { assertEquals(0, loneSurrogates(it), "a lone surrogate in ${it.map { c -> "\\u%04x".format(c.code) }}") }
    }

    @Test
    fun anEmojiStraddlingTheHoldBackCutIsNotSplit() {
        val s = StopStrings(listOf("ab"))
        val emitted = s.feed("Hi 🙂")
        val rest = s.finish()
        assertWhole(emitted, rest)
        assertEquals("Hi 🙂", emitted + rest)
        assertFalse(s.matched)
    }

    @Test
    fun anEmojiFedOneSurrogateAtATimeIsNotSplit() {
        val s = StopStrings(listOf("ab"))
        val a = s.feed("\uD83D")
        val b = s.feed("\uDE42")
        val rest = s.finish()
        assertWhole(a, b, rest)
        assertEquals("\uD83D\uDE42", a + b + rest)
    }

    @Test
    fun aStopMatchLandingInsideAPairDoesNotSplitIt() {
        // A stop string beginning with a low surrogate matches at index 1 of the pair, which
        // is between its halves. The cut moves back so nothing lone is emitted.
        val s = StopStrings(listOf("\uDE42"))
        val emitted = s.feed("\uD83D\uDE42")
        assertWhole(emitted, s.finish())
        assertTrue(s.matched)
        assertEquals("", emitted)
    }

    @Test
    fun severalEmojiAcrossFeedsStayWhole() {
        val s = StopStrings(listOf("stop"))
        val text = "a🙂b🙃c😀d"
        val out = StringBuilder()
        // Feed in pieces of three UTF-16 units, so pairs straddle the boundary repeatedly.
        var i = 0
        while (i < text.length) {
            val piece = text.substring(i, minOf(i + 3, text.length))
            out.append(s.feed(piece))
            i += 3
        }
        out.append(s.finish())
        assertWhole(out.toString())
        assertEquals(text, out.toString())
    }

    @Test
    fun aStopStringStillTruncatesAndMatches() {
        val s = StopStrings(listOf("ab"))
        assertEquals("xx", s.feed("xxabyy"))
        assertTrue(s.matched)
        assertEquals("", s.finish())
    }

    @Test
    fun aPartialStopIsHeldUntilItCompletes() {
        val s = StopStrings(listOf("abc"))
        assertEquals("xx", s.feed("xxab"))
        assertFalse(s.matched)
        assertEquals("", s.feed("c"))
        assertTrue(s.matched)
    }

    @Test
    fun noStopsPassesTextThroughUntouched() {
        val s = StopStrings(emptyList())
        assertEquals("🙂 anything", s.feed("🙂 anything"))
        assertEquals("", s.finish())
    }
}
