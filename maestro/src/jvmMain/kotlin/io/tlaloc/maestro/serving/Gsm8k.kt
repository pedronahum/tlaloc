package io.tlaloc.maestro.serving

/**
 * GSM8K prompts and scoring, with the rules of vLLM's `tests/evals/gsm8k/gsm8k_eval.py`, so that
 * an accuracy compares with vLLM's: the first [shots] train examples, then the test question;
 * completions stop at [STOP]; the answer is the last run of digits with commas removed.
 */
object Gsm8k {
    const val INVALID = -9999999L
    val STOP = listOf("Question", "Assistant:", "<|separator|>")
    private val DIGITS = Regex("""\d+""")

    fun fewShot(train: List<Example>, shots: Int): String =
        train.take(shots).joinToString("") { "Question: ${it.question}\nAnswer: ${it.answer}\n\n" }

    fun prompt(fewShot: String, question: String): String = "${fewShot}Question: $question\nAnswer:"

    /** The text up to the first stop string. */
    fun cut(text: String): String =
        STOP.map { text.indexOf(it) }.filter { it >= 0 }.minOrNull()?.let { text.substring(0, it) } ?: text

    /** A number with a leading zero is [INVALID], as Python's `literal_eval` refuses it. */
    fun answerValue(text: String): Long {
        val last = DIGITS.findAll(text.replace(",", "")).lastOrNull()?.value ?: return INVALID
        if (last.length > 1 && last.all { it == '0' }.not() && last[0] == '0') return INVALID
        return last.toLongOrNull() ?: INVALID
    }

    data class Example(val question: String, val answer: String)
}
