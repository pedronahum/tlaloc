package brooklyn

import java.io.IOException
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

// Network access for the live tools. The tool server receives a factory and calls it only in
// live and record modes, so snapshot and replay never construct an HTTP client.

class NetResponse(val status: Int, val contentType: String?, val body: ByteArray, val finalUrl: String) {
    val text: String get() = String(body, Charsets.UTF_8)
}

fun interface Net {
    fun get(url: String, accept: String?): NetResponse
}

/**
 * java.net.http with a generic User-Agent, at most one request per second to a host
 * (Nominatim's limit, applied to every host), a 3 MB body cap, and backoff on 429 / 5xx
 * (Retry-After honoured up to 60 s).
 */
class HttpNet(
    private val minSpacingMs: Long = 1_000L,
    private val maxAttempts: Int = 3,
    private val timeout: Duration = Duration.ofSeconds(20),
    private val firstBackoffMs: Long = 2_000L,
    private val log: (String) -> Unit = { System.err.println(it) },
) : Net {
    private val client: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .followRedirects(HttpClient.Redirect.NORMAL)
        .build()
    private val nextSlot = HashMap<String, Long>()

    /** Waits until the host's last request is [minSpacingMs] old, then marks now as its last; other hosts do not queue. */
    private fun pace(host: String) {
        while (true) {
            val wait = synchronized(nextSlot) {
                val now = System.currentTimeMillis()
                val free = nextSlot[host] ?: 0L
                if (now >= free) { nextSlot[host] = now + minSpacingMs; 0L } else free - now
            }
            if (wait == 0L) return
            Thread.sleep(wait)
        }
    }

    override fun get(url: String, accept: String?): NetResponse {
        val uri = URI(url)
        val b = HttpRequest.newBuilder(uri).GET().timeout(timeout).header("User-Agent", USER_AGENT)
        if (accept != null) b.header("Accept", accept)
        val req = b.build()
        var backoff = firstBackoffMs
        var last: NetResponse? = null
        var err: IOException? = null
        for (attempt in 1..maxAttempts) {
            pace(uri.host ?: "")
            try {
                val resp = client.send(req, HttpResponse.BodyHandlers.ofInputStream())
                val body = resp.body().use { it.readNBytes(MAX_BODY) }
                val r = NetResponse(resp.statusCode(), resp.headers().firstValue("Content-Type").orElse(null), body, resp.uri().toString())
                if (r.status != 429 && r.status < 500) return r
                last = r
                resp.headers().firstValue("Retry-After").orElse(null)?.toLongOrNull()?.let { backoff = maxOf(backoff, minOf(it, 60) * 1000) }
            } catch (e: IOException) {
                err = e
            }
            if (attempt < maxAttempts) {
                log("  ${uri.host}: ${last?.status ?: err?.message}, retry in ${backoff / 1000}s")
                Thread.sleep(backoff)
                backoff *= 2
            }
        }
        return last ?: throw (err ?: IOException("$url failed"))
    }

    companion object {
        const val MAX_BODY = 3 shl 20
    }
}
