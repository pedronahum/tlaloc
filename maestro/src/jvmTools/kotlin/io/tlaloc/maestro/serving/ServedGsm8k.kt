package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

private fun quoted(s: String): String = buildString {
    append('"')
    for (c in s) when {
        c == '"' -> append("\\\"")
        c == '\\' -> append("\\\\")
        c < ' ' -> append("\\u%04x".format(c.code))
        else -> append(c)
    }
    append('"')
}

private fun examples(file: Path): List<Gsm8k.Example> = Files.readAllLines(file).filter { it.isNotBlank() }.map { line ->
    val o = parseJson(line) as JsonObject
    Gsm8k.Example((o["question"] as JsonString).value, (o["answer"] as JsonString).value)
}

/** One completion, streamed, closed at the first stop string. */
private fun complete(http: HttpClient, url: String, model: String, prompt: String, maxTokens: Int, stopIds: String): String {
    val stops = if (stopIds.isEmpty()) "" else ""","stop_token_ids":[$stopIds]"""
    val body = """{"model":${quoted(model)},"prompt":${quoted(prompt)},"max_tokens":$maxTokens,"temperature":0,"stream":true$stops}"""
    val request = HttpRequest.newBuilder(URI.create("$url/v1/completions"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body)).build()
    val response = http.send(request, HttpResponse.BodyHandlers.ofLines())
    check(response.statusCode() == 200) { "HTTP ${response.statusCode()}: ${response.body().toList().joinToString(" ")}" }
    val text = StringBuilder()
    response.body().use { lines ->
        for (line in lines) {
            if (!line.startsWith("data: ")) continue
            val data = line.removePrefix("data: ")
            if (data == "[DONE]") break
            val choices = (parseJson(data) as JsonObject)["choices"] as? JsonArray ?: continue
            for (c in choices.elements) text.append(((c as JsonObject)["text"] as? JsonString)?.value.orEmpty())
            if (Gsm8k.STOP.any { it in text }) break
        }
    }
    return Gsm8k.cut(text.toString())
}

/**
 * GSM8K over an OpenAI endpoint: `<url> <model> <dir with train.jsonl, test.jsonl> <results file>
 * [questions=1319] [concurrency=1] [shots=5] [maxTokens=256] [stop token ids, comma-separated]`.
 * An endpoint generates on after a client closes its stream, so the ids of the stop strings' tokens
 * end the generation there. Prints the accuracy every 50 questions; the results file has one line
 * per question: index, predicted, label, seconds.
 */
fun main(args: Array<String>) {
    require(args.size >= 4) { "usage: <url> <model> <data dir> <results file> [questions] [concurrency] [shots] [maxTokens] [stop token ids]" }
    val (url, model) = args
    val dir = Path.of(args[2])
    fun arg(i: Int, d: Int) = args.getOrNull(i)?.takeIf { it.isNotBlank() }?.toInt() ?: d
    val concurrency = arg(5, 1)
    val maxTokens = arg(7, 256)
    val stopIds = args.getOrNull(8).orEmpty()
    val test = examples(dir.resolve("test.jsonl")).take(arg(4, 1319))
    val fewShot = Gsm8k.fewShot(examples(dir.resolve("train.jsonl")), arg(6, 5))
    val http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build()
    val results = arrayOfNulls<String>(test.size)
    val done = AtomicInteger()
    val right = AtomicInteger()
    val invalid = AtomicInteger()
    val t0 = System.nanoTime()
    val pool = Executors.newFixedThreadPool(concurrency)
    test.forEachIndexed { i, q ->
        pool.submit {
            try {
                val t = System.nanoTime()
                val predicted = Gsm8k.answerValue(complete(http, url, model, Gsm8k.prompt(fewShot, q.question), maxTokens, stopIds))
                val label = Gsm8k.answerValue(q.answer)
                if (predicted == label) right.incrementAndGet()
                if (predicted == Gsm8k.INVALID) invalid.incrementAndGet()
                results[i] = "$i\t$predicted\t$label\t${"%.2f".format((System.nanoTime() - t) / 1e9)}"
                val n = done.incrementAndGet()
                if (n % 50 == 0 || n == test.size) {
                    println("$n/${test.size} accuracy ${"%.4f".format(right.get().toDouble() / n)} invalid ${invalid.get()} ${"%.0f".format((System.nanoTime() - t0) / 1e9)} s")
                }
            } catch (e: Exception) {
                System.err.println("question $i: $e")
                done.incrementAndGet()
            }
        }
    }
    pool.shutdown()
    pool.awaitTermination(7, java.util.concurrent.TimeUnit.DAYS)
    Files.write(Path.of(args[3]), results.filterNotNull())
    println("accuracy ${"%.4f".format(right.get().toDouble() / test.size)} of ${test.size}, ${results.count { it != null }} answered, invalid ${invalid.get()}, ${"%.0f".format((System.nanoTime() - t0) / 1e9)} s")
}
