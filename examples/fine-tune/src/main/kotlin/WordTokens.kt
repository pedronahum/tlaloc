import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path

/**
 * Whole-word token ids from the checkpoint's own `vocab.json`.
 *
 * Tlaloc has no BPE tokenizer. Qwen3's byte-level BPE gives common words a
 * token of their own, spelled with `Ġ` for a leading space (`" Paris"` is
 * `"ĠParis"`), so this example looks each word up directly and refuses any
 * word the vocabulary splits into pieces. The ids are the ones
 * transformers' tokenizer produces for the same text.
 */
class WordTokens private constructor(private val ids: Map<String, Int>) {

    private val byId: Map<Int, String> = ids.entries.associate { (k, v) -> v to k }

    fun encode(text: String): List<Int> =
        PIECE.findAll(text).map { m ->
            val piece = m.value.replace(' ', 'Ġ')
            ids[piece] ?: throw IllegalArgumentException(
                "'${m.value}' is not a single token in vocab.json; this example only encodes whole-word tokens",
            )
        }.toList()

    fun decode(tokens: List<Int>): String = tokens.joinToString("") { (byId[it] ?: "<$it>").replace('Ġ', ' ') }

    companion object {
        private val PIECE = Regex(" ?[A-Za-z]+|[.,!?]")

        fun load(checkpoint: Path): WordTokens {
            val vocab = parseJson(Files.readString(checkpoint.resolve("vocab.json"))) as JsonObject
            return WordTokens(vocab.fields.mapValues { (_, v) -> (v as JsonNumber).value.toInt() })
        }
    }
}
