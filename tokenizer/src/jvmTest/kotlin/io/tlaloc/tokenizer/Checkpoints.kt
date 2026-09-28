package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

/** Goldens written by harness/python/tokenizer_golden.py, and the local checkpoints they came from. */
internal object Checkpoints {
    private val home: Path = Path.of(System.getProperty("user.home"))
    private val loaded = ConcurrentHashMap<String, HfTokenizer>()

    fun golden(family: String): JsonObject {
        val text = Checkpoints::class.java.getResource("/io/tlaloc/tokenizer/$family.json")!!.readText()
        return parseJson(text) as JsonObject
    }

    /** The checkpoint directory the golden was written from, or null when its tokenizer files are not here. */
    fun directory(golden: JsonObject): Path? {
        val checkpoint = golden.str("checkpoint")
        val revision = (golden["revision"] as? io.tlaloc.core.io.JsonString)?.value
        val dir = if (revision != null) {
            home.resolve(".cache/huggingface/hub/models--${checkpoint.replace("/", "--")}/snapshots/$revision")
        } else {
            home.resolve(".cache/tlaloc-checkpoints/$checkpoint")
        }
        return dir.takeIf { Files.isRegularFile(it.resolve("tokenizer.json")) }
    }

    fun tokenizer(family: String, dir: Path): HfTokenizer = loaded.computeIfAbsent(family) { HfTokenizer.load(dir) }
}
