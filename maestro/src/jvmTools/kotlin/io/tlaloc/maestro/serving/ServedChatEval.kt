package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

private const val INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

private class Problem(val prompt: String, val label: String)

private class Reply(val content: String, val reasoning: String, val tokens: Int, val finish: String)

private fun problems(dataset: String, file: Path): List<Problem> = Files.readAllLines(file).filter { it.isNotBlank() }.map { line ->
    val o = parseJson(line) as JsonObject
    fun str(k: String) = (o[k] as? JsonString)?.value ?: error("$file: a line has no string '$k'")
    when (dataset) {
        "math500" -> Problem(str("problem") + "\n" + INSTRUCTION, str("answer"))
        "gsm8k" -> Problem(str("question") + "\n" + INSTRUCTION, str("answer").substringAfterLast("####").trim().replace(",", ""))
        else -> error("dataset is math500 or gsm8k, not '$dataset'")
    }
}

/** One chat completion, streamed: its answer, its reasoning, its completion tokens and why it ended. */
private fun ask(http: HttpClient, url: String, model: String, prompt: String, maxTokens: Int, sampling: String, seed: Int): Reply {
    val body = """{"model":${jsonQuote(model)},"messages":[{"role":"user","content":${jsonQuote(prompt)}}],""" +
        """"max_tokens":$maxTokens,$sampling,"seed":$seed,"stream":true,"stream_options":{"include_usage":true}}"""
    val request = HttpRequest.newBuilder(URI.create("$url/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body)).build()
    val response = http.send(request, HttpResponse.BodyHandlers.ofLines())
    check(response.statusCode() == 200) { "HTTP ${response.statusCode()}: ${response.body().toList().joinToString(" ")}" }
    val content = StringBuilder()
    val reasoning = StringBuilder()
    var tokens = -1
    var finish = "none"
    response.body().use { lines ->
        for (line in lines) {
            if (!line.startsWith("data: ")) continue
            val data = line.removePrefix("data: ")
            if (data == "[DONE]") break
            val o = parseJson(data) as JsonObject
            // An error after the stream began comes as a chunk of its own.
            (o["error"] as? JsonObject)?.let { finish = "error: " + ((it["message"] as? JsonString)?.value ?: "?").take(80) }
            ((o["usage"] as? JsonObject)?.get("completion_tokens") as? JsonNumber)?.let { tokens = it.value.toInt() }
            for (c in (o["choices"] as? JsonArray)?.elements.orEmpty()) {
                val choice = c as JsonObject
                (choice["finish_reason"] as? JsonString)?.let { finish = it.value }
                val delta = choice["delta"] as? JsonObject ?: continue
                (delta["content"] as? JsonString)?.let { content.append(it.value) }
                ((delta["reasoning_content"] ?: delta["reasoning"]) as? JsonString)?.let { reasoning.append(it.value) }
            }
        }
    }
    return Reply(content.toString(), reasoning.toString(), tokens, finish)
}

/** The answer of a reply: the last `\boxed{}` after the reasoning (a server without a reasoning parser leaves `</think>` in the content). */
private fun answerOf(reply: Reply): String? = MathAnswer.lastBoxed(reply.content.substringAfterLast("</think>"))

private fun correct(dataset: String, predicted: String?, label: String): Boolean = when (dataset) {
    "gsm8k" -> predicted?.replace(",", "")?.replace("\\$", "")?.trim()?.toBigDecimalOrNull()?.let { p ->
        label.toBigDecimalOrNull()?.let { it.compareTo(p) == 0 }
    } ?: false
    else -> MathAnswer.equivalent(predicted, label)
}

private fun tsv(s: String?) = (s ?: "-").replace('\t', ' ').replace('\n', ' ')

/**
 * A reasoning model's accuracy over a chat endpoint (Tlaloc's or vLLM's): `<url> <model>
 * <math500|gsm8k> <jsonl file> <results file> [questions=all] [concurrency=4] [maxTokens=16384]
 * [sampling JSON fields, default "temperature":0] [first question=0]`.
 *
 * Each problem is one user turn ending with "Please reason step by step, and put your final
 * answer within \boxed{}."; the answer is the last `\boxed{}` of the reply's content after its
 * reasoning, compared by [MathAnswer] (MATH-500) or as a number (GSM8K). Question `i` is sent
 * with seed `i`. The results file has one line per question: index, correct, predicted, label,
 * completion tokens, seconds, finish reason; `<results>.jsonl` holds each reply's text.
 */
fun main(args: Array<String>) {
    require(args.size >= 5) {
        "usage: <url> <model> <math500|gsm8k> <jsonl> <results> [questions] [concurrency] [maxTokens] [sampling JSON fields] [first]"
    }
    val (url, model, dataset) = args
    fun arg(i: Int, d: Int) = args.getOrNull(i)?.takeIf { it.isNotBlank() }?.toInt() ?: d
    val first = arg(9, 0)
    val all = problems(dataset, Path.of(args[3]))
    val count = arg(5, all.size - first)
    val selected = all.withIndex().drop(first).take(count)
    val concurrency = arg(6, 4)
    val maxTokens = arg(7, 16384)
    val sampling = args.getOrNull(8)?.takeIf { it.isNotBlank() } ?: "\"temperature\":0"
    val http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build()
    val results = arrayOfNulls<String>(selected.size)
    val texts = arrayOfNulls<String>(selected.size)
    val done = AtomicInteger()
    val right = AtomicInteger()
    val unboxed = AtomicInteger()
    val truncated = AtomicInteger()
    val tokens = AtomicLong()
    val t0 = System.nanoTime()
    val pool = Executors.newFixedThreadPool(concurrency)
    selected.forEachIndexed { k, (i, p) ->
        pool.submit {
            try {
                val t = System.nanoTime()
                val reply = ask(http, url, model, p.prompt, maxTokens, sampling, i)
                val predicted = answerOf(reply)
                val ok = correct(dataset, predicted, p.label)
                if (ok) right.incrementAndGet()
                if (predicted == null) unboxed.incrementAndGet()
                if (reply.finish == "length") truncated.incrementAndGet()
                tokens.addAndGet(maxOf(reply.tokens, 0).toLong())
                results[k] = "$i\t${if (ok) 1 else 0}\t${tsv(predicted)}\t${tsv(p.label)}\t${reply.tokens}\t" +
                    "${"%.1f".format((System.nanoTime() - t) / 1e9)}\t${reply.finish}"
                texts[k] = """{"index":$i,"content":${jsonQuote(reply.content)},"reasoning":${jsonQuote(reply.reasoning)}}"""
                val n = done.incrementAndGet()
                if (n % 10 == 0 || n == selected.size) {
                    println(
                        "$n/${selected.size} accuracy ${"%.4f".format(right.get().toDouble() / n)} no answer ${unboxed.get()} " +
                            "cut ${truncated.get()} mean tokens ${tokens.get() / n} ${"%.0f".format((System.nanoTime() - t0) / 1e9)} s",
                    )
                }
            } catch (e: Exception) {
                System.err.println("question $i: $e")
                done.incrementAndGet()
            }
        }
    }
    pool.shutdown()
    pool.awaitTermination(7, TimeUnit.DAYS)
    Files.write(Path.of(args[4]), results.filterNotNull())
    Files.write(Path.of(args[4] + ".jsonl"), texts.filterNotNull())
    val secs = (System.nanoTime() - t0) / 1e9
    println(
        "accuracy ${"%.4f".format(right.get().toDouble() / selected.size)} of ${selected.size}, ${results.count { it != null }} answered, " +
            "no answer ${unboxed.get()}, cut at $maxTokens ${truncated.get()}, ${tokens.get()} completion tokens, " +
            "${"%.0f".format(secs)} s, ${"%.1f".format(tokens.get() / secs)} tokens/s",
    )
}
