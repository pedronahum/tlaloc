package io.tlaloc.tokenizer

import java.nio.file.Files
import java.nio.file.Path

/**
 * The tokenizer of the Hugging Face checkpoint directory [checkpoint]:
 * `tokenizer.json`, plus `tokenizer_config.json` and `chat_template.jinja`
 * when present. See [HfTokenizer.fromJson].
 *
 * @throws java.nio.file.NoSuchFileException if there is no `tokenizer.json`.
 * @throws UnsupportedTokenizerException naming what is not implemented.
 */
fun HfTokenizer.Companion.load(checkpoint: Path): HfTokenizer {
    val tokenizerJson = Files.readString(checkpoint.resolve("tokenizer.json"))
    val config = checkpoint.resolve("tokenizer_config.json").takeIf(Files::isRegularFile)?.let(Files::readString)
    val template = checkpoint.resolve("chat_template.jinja").takeIf(Files::isRegularFile)?.let(Files::readString)
    return fromJson(tokenizerJson, config, template)
}
