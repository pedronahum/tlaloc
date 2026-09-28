package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The prompts of :ir's inference fixtures (written by hf_greedy_fixture.py and
 * muse_glimmer_fixture.py): the rendered chat prompt and its ids, and the ids
 * of each plain-text prompt, including Muse Glimmer's 2,305-token needle
 * prompt. Self-skips without the checkpoint.
 */
class InferenceFixtureTest {
    private val fixtures = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference")

    private fun checkFixture(family: String, file: String) {
        val tok = tokenizerOrSkip(family)
        val fixture = parseJson(Files.readString(fixtures.resolve(file))) as JsonObject
        for (p in fixture.arr("prompts").elements.map { it as JsonObject }) {
            val expected = (p["promptTokens"] as JsonArray).asIntList("promptTokens").toIntArray()
            val input = p.str("input")
            if (p.str("kind") == "chat") {
                val vars = p["templateVars"] as? JsonObject
                val options = ChatOptions(
                    enableThinking = if (family == "qwen3") false else null,
                    currentDate = (vars?.get("current_date") as? JsonString)?.value,
                )
                val rendered = tok.applyChatTemplate(listOf(ChatMessage("user", input)), options)
                assertEquals(p.str("rendered"), rendered)
                assertContentEquals(expected, tok.encode(rendered, addSpecialTokens = false))
            } else {
                assertContentEquals(expected, tok.encode(input), "prompt ${input.take(60)}")
            }
        }
    }

    private fun tokenizerOrSkip(family: String): HfTokenizer {
        val golden = Checkpoints.golden(family)
        val dir = Checkpoints.directory(golden)
        assumeTrue(dir != null, "$family: tokenizer files for ${golden.str("checkpoint")} are not in the local cache")
        return Checkpoints.tokenizer(family, dir!!)
    }

    @Test fun `qwen3 greedy fixture prompts`() {
        checkFixture("qwen3", "qwen3_0_6b_greedy.json")
        assertContentEquals(intArrayOf(576, 6722, 315, 9625, 374), tokenizerOrSkip("qwen3").encode(" The capital of France is"))
    }

    @Test fun `muse glimmer greedy fixture prompts`() = checkFixture("muse_glimmer", "muse_glimmer_30b_bf16_greedy.json")

    @Test fun `muse glimmer needle prompt`() = checkFixture("muse_glimmer", "muse_glimmer_30b_needle_mixed_greedy.json")

    /**
     * The control for the transformers rules: TinyLlama's tokenizer.json on
     * its own (Prepend("▁") normalizer) gives different ids from
     * AutoTokenizer (Metaspace prepending only at the start of the input).
     */
    @Test fun `tinyllama tokenizer json alone differs from AutoTokenizer`() {
        val auto = tokenizerOrSkip("tinyllama")
        val dir = Checkpoints.directory(Checkpoints.golden("tinyllama"))!!
        val raw = HfTokenizer.fromJson(Files.readString(dir.resolve("tokenizer.json")))
        assertContentEquals(intArrayOf(1, 1, 2918, 2), auto.encode("<s>hi</s>"))
        assertContentEquals(intArrayOf(1, 1, 7251, 2), raw.encode("<s>hi</s>"))
        assertEquals(29871, auto.encode("  leading")[1])
        assertEquals(259, raw.encode("  leading")[1])
    }
}
