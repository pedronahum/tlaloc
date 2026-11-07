package io.tlaloc.maestro.serving

import kotlin.test.Test
import kotlin.test.assertEquals

class Gsm8kTest {
    @Test
    fun theAnswerIsTheLastRunOfDigits() {
        assertEquals(1234L, Gsm8k.answerValue("She has 5 and then #### 1,234"))
        assertEquals(50L, Gsm8k.answerValue("costs $50.00 less 2.50, so 50"))
        assertEquals(0L, Gsm8k.answerValue("#### 00"))
        assertEquals(Gsm8k.INVALID, Gsm8k.answerValue("no digits"))
        assertEquals(Gsm8k.INVALID, Gsm8k.answerValue("#### 007"))
    }

    @Test
    fun aCompletionEndsAtTheFirstStopString() {
        assertEquals(" #### 4\n\n", Gsm8k.cut(" #### 4\n\nQuestion: more Assistant: x"))
        assertEquals("all of it", Gsm8k.cut("all of it"))
    }

    @Test
    fun thePromptIsTheShotsThenTheQuestion() {
        val shots = Gsm8k.fewShot(listOf(Gsm8k.Example("a?", "b"), Gsm8k.Example("c?", "d"), Gsm8k.Example("e?", "f")), 2)
        assertEquals("Question: a?\nAnswer: b\n\nQuestion: c?\nAnswer: d\n\nQuestion: q?\nAnswer:", Gsm8k.prompt(shots, "q?"))
    }
}
