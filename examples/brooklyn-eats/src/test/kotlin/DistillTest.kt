package brooklyn

import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Aligning TwIL's training sequence with Ripple's rendering of the same chat, and the distill targets. */
class DistillTest {
    private fun tokenizer(key: String): HfTokenizer {
        val dir = runCatching { modelDir(key, hubCache(), readManifestJson(Path.of("manifest.json").toAbsolutePath())) }.getOrNull()
        assumeTrue(dir != null && Files.exists(dir.resolve("tokenizer.json")), "$key's tokenizer is not in the Hugging Face cache (run.sh models)")
        return HfTokenizer.load(dir!!)
    }

    private val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }

    private val messages = listOf(
        ConvMessage("system", "You are Brooklyn Eats."),
        ConvMessage("user", "What grade does Golden Lantern have?"),
        ConvMessage(
            "assistant",
            "Golden Lantern Kitchen (CAMIS 50086128) has an **A**, given on March 2, 2024.\n\nIts score was 12.",
            "The user wants a grade. The record says A since 2024.",
        ),
    )

    @Test
    fun alignedPositionsHoldTheSameTokenAndText() {
        val tw = tokenizer(STUDENT_KEY)
        val rp = tokenizer(TEACHER_KEY)
        val seq = renderForTraining(tw, ChatTemplate.TWIL, messages, toolsJson)
        assertTrue(seq.ok, seq.problems.joinToString())
        val rpIds = rp.encode(rp.applyChatTemplate(messages.map { it.forTemplate() }, chatOptions(toolsJson, false), ChatTemplate.RIPPLE), addSpecialTokens = false)
        val al = align(tw, seq.ids, seq.mask, rp, rpIds, messages)
        assertTrue(al.regionsFound >= 1, "the answer text is found in both renderings (${al.regionsFound} of ${al.regions})")
        val masked = (seq.ids.indices).count { seq.mask[it] }
        val aligned = al.rippleOf.count { it >= 0 }
        assertTrue(aligned in 1 until masked, "$aligned of $masked assistant tokens aligned")
        for ((j, r) in al.rippleOf.withIndex()) {
            if (r < 0) continue
            assertTrue(seq.mask[j], "an aligned position is an assistant token")
            assertEquals(seq.ids[j], rpIds[r], "position $j's token")
        }
        // "2024" is 20|24 for TwIL and 202|4 for Ripple: neither TwIL piece is aligned.
        val twPieces = seq.ids.indices.filter { seq.mask[it] && tw.idToToken(seq.ids[it]) in setOf("20", "24") }
        assertTrue(twPieces.isNotEmpty() && twPieces.all { al.rippleOf[it] < 0 }, "the split year is not aligned")
    }

    @Test
    fun targetsAreDistributionsOnTargetRowsOnly() {
        val n = 6
        val ids = intArrayOf(1, 2, 3, 4, 5, 6)
        val mask = booleanArrayOf(false, false, true, true, true, false)
        // Position 2 and 4 aligned to Ripple positions 7 and 9; position 3 not.
        val al = Alignment(intArrayOf(-1, -1, 7, -1, 9, -1), 1, 1)
        val k = 3
        val tIds = IntArray(12 * k) { 10 + it % k }.also { it[6 * k] = 3; it[6 * k + 1] = SHARED_VOCAB + 5; it[8 * k] = 5 }
        val tLp = FloatArray(12 * k) { -1f - it % k }
        val d = distillTargets(ids, mask, al, TeacherScores(IntArray(12), k, tIds, tLp))
        assertEquals(k + 1, d.j)
        for (row in 0 until n) {
            val sum = (0 until d.j).sumOf { d.targetWeights[row * d.j + it].toDouble() }
            val target = row + 1 < n && mask[row + 1]
            assertTrue(abs(sum - if (target) 1.0 else 0.0) < 1e-5, "row $row's weights sum to $sum")
            if (target) assertEquals(ids[row + 1], d.targetIds[row * d.j], "row $row's first slot is the true token")
        }
        // Row 1 predicts position 2 (aligned): half on the true token, half on the teacher's shared candidates.
        assertEquals(DISTILL_TRUE_WEIGHT, d.targetWeights[1 * d.j])
        assertEquals(0f, d.targetWeights[1 * d.j + 2], "a Ripple-only id gets no weight")
        // Row 2 predicts position 3 (not aligned): the true token alone.
        assertEquals(1f, d.targetWeights[2 * d.j])
        assertEquals(3, d.targetRows)
    }
}
