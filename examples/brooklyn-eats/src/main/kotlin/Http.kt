package brooklyn

import java.io.IOException
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.time.Duration

/** Generic client name; no personal details. */
const val USER_AGENT = "tlaloc-brooklyn-eats/0.1 (educational example)"

/**
 * HTTP with a generic User-Agent, a minimum spacing between requests to the same host, and
 * exponential backoff on 429 / 5xx / IO errors (honouring Retry-After when the server sends it).
 */
class Http(
    private val minSpacingMs: Map<String, Long> = mapOf("nominatim.openstreetmap.org" to 1_000L),
    private val maxAttempts: Int = 6,
    private val log: (String) -> Unit = { System.err.println(it) },
) {
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .followRedirects(HttpClient.Redirect.NORMAL)
        .build()
    private val lastCall = HashMap<String, Long>()

    private fun pace(host: String) {
        val spacing = minSpacingMs[host] ?: return
        val last = lastCall[host] ?: 0L
        val wait = last + spacing - System.currentTimeMillis()
        if (wait > 0) Thread.sleep(wait)
        lastCall[host] = System.currentTimeMillis()
    }

    /** GET [url] into [dest] (written atomically); returns the byte count. */
    fun download(url: String, dest: Path, timeout: Duration = Duration.ofMinutes(30)): Long =
        send(HttpRequest.newBuilder(URI(url)).GET(), url, dest, timeout)

    /** POST a form body (Overpass) into [dest]. */
    fun postForm(url: String, form: Map<String, String>, dest: Path, timeout: Duration = Duration.ofMinutes(5)): Long {
        val body = form.entries.joinToString("&") { (k, v) -> "${enc(k)}=${enc(v)}" }
        return send(
            HttpRequest.newBuilder(URI(url))
                .header("Content-Type", "application/x-www-form-urlencoded")
                .POST(HttpRequest.BodyPublishers.ofString(body)),
            url, dest, timeout,
        )
    }

    private fun send(builder: HttpRequest.Builder, url: String, dest: Path, timeout: Duration): Long {
        val host = URI(url).host
        val req = builder.header("User-Agent", USER_AGENT).timeout(timeout).build()
        Files.createDirectories(dest.toAbsolutePath().parent)
        val part = dest.resolveSibling(dest.fileName.toString() + ".part")
        var backoffMs = 5_000L
        var lastError = ""
        for (attempt in 1..maxAttempts) {
            pace(host)
            try {
                val resp = client.send(req, HttpResponse.BodyHandlers.ofFile(part))
                val code = resp.statusCode()
                if (code in 200..299) {
                    Files.move(part, dest, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
                    return Files.size(dest)
                }
                lastError = "HTTP $code"
                if (code != 429 && code < 500) break
                val retryAfter = resp.headers().firstValue("Retry-After").orElse(null)?.toLongOrNull()
                if (retryAfter != null) backoffMs = maxOf(backoffMs, retryAfter * 1000)
            } catch (e: IOException) {
                lastError = "${e::class.simpleName}: ${e.message}"
            }
            Files.deleteIfExists(part)
            if (attempt < maxAttempts) {
                log("  $host: $lastError, retry $attempt/${maxAttempts - 1} in ${backoffMs / 1000}s")
                Thread.sleep(backoffMs)
                backoffMs = minOf(backoffMs * 2, 120_000L)
            }
        }
        Files.deleteIfExists(part)
        throw IOException("$url: $lastError")
    }

    companion object {
        fun enc(s: String): String = URLEncoder.encode(s, StandardCharsets.UTF_8)
    }
}
