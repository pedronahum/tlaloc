package brooklyn

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import io.tlaloc.core.io.parseJson
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.net.InetSocketAddress
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Model pins, the resumable download and the cache checks against a synthetic Hub on localhost. */
class ModelsTest {
    private val rev = "0123456789abcdef0123456789abcdef01234567"
    private val config = """{"model_type":"tiny"}""".toByteArray()
    private val weights = ByteArray(300_000) { (it * 31 % 251).toByte() }
    private val weightsSha = sha256Hex(weights)

    /** Requests seen: path and whether they carried the token. */
    private val hubCalls = ArrayList<String>()
    private val storageCalls = ArrayList<String>()
    private val cutFirstDownload = AtomicInteger(1)
    private val storage: HttpServer = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
        createContext("/blob") { ex ->
            synchronized(storageCalls) { storageCalls += "${ex.requestURI.path} auth=${ex.requestHeaders.containsKey("Authorization")} range=${ex.requestHeaders.getFirst("Range")}" }
            val range = ex.requestHeaders.getFirst("Range")?.removePrefix("bytes=")?.removeSuffix("-")?.toInt() ?: 0
            if (range == 0 && cutFirstDownload.getAndDecrement() > 0) {
                // Announces the whole file, sends half of it and closes the connection.
                ex.sendResponseHeaders(200, weights.size.toLong())
                ex.responseBody.write(weights, 0, weights.size / 2)
                ex.responseBody.flush()
                ex.close()
                return@createContext
            }
            val body = weights.copyOfRange(range, weights.size)
            if (range > 0) ex.responseHeaders.add("Content-Range", "bytes $range-${weights.size - 1}/${weights.size}")
            ex.sendResponseHeaders(if (range > 0) 206 else 200, body.size.toLong())
            ex.responseBody.use { it.write(body) }
        }
        start()
    }
    // The Hub answers on "localhost" and redirects weights to "127.0.0.1": another host, which never gets the token.
    private val hub: HttpServer = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
        createContext("/") { ex -> serveHub(ex) }
        start()
    }
    private val endpoint = "http://localhost:${hub.address.port}"

    private fun serveHub(ex: HttpExchange) {
        val p = ex.requestURI.path
        synchronized(hubCalls) { hubCalls += "$p auth=${ex.requestHeaders.containsKey("Authorization")}" }
        fun send(code: Int, b: ByteArray) { ex.sendResponseHeaders(code, b.size.toLong()); ex.responseBody.use { it.write(b) } }
        when (p) {
            "/api/models/org/tiny/revision/main" -> send(
                200,
                """{"sha":"$rev","siblings":[
                  {"rfilename":"config.json","size":${config.size},"blobId":"${gitBlobId(config)}"},
                  {"rfilename":"model.safetensors","size":${weights.size},"blobId":"aaaa","lfs":{"sha256":"$weightsSha","size":${weights.size}}},
                  {"rfilename":"modeling_tiny.py","size":3,"blobId":"bbbb"},
                  {"rfilename":"sub/model.safetensors","size":1,"blobId":"cccc","lfs":{"sha256":"${"d".repeat(64)}","size":1}}]}""".toByteArray(),
            )
            "/org/tiny/resolve/$rev/config.json" -> send(200, config)
            "/org/tiny/resolve/$rev/model.safetensors" -> {
                ex.responseHeaders.add("Location", "http://127.0.0.1:${storage.address.port}/blob/$weightsSha")
                ex.sendResponseHeaders(302, -1); ex.close()
            }
            else -> send(404, "not found".toByteArray())
        }
    }

    @AfterTest
    fun stop() { hub.stop(0); storage.stop(0) }

    private fun client() = HubClient(endpoint = endpoint, token = "secret", firstBackoffMs = 1, log = {})

    private fun pinned(home: Path): ModelPin {
        Files.writeString(home.resolve("manifest.json"), """{"cutoff":"2026-11-13","seeds":{"match_sample":7}}""")
        return pinModel(home, "tiny", "org/tiny", "main", "student", DEFAULT_MODEL_INCLUDE, optional = false, hub = client(), log = {})
    }

    @Test
    fun pinResolvesTheRevisionToACommitAndHashesEveryFile() {
        val home = Files.createTempDirectory("mhome")
        val pin = pinned(home)
        assertEquals(rev, pin.revision)
        // The include patterns are top-level: sub/ and the .py file are left out.
        assertEquals(listOf("config.json", "model.safetensors"), pin.files.map { it.path })
        val c = pin.files[0]
        assertEquals(sha256Hex(config), c.sha256)
        assertEquals(gitBlobId(config), c.gitBlob)
        assertEquals(weightsSha, pin.files[1].sha256)
        // Read back from manifest.json, other keys kept.
        val m = readManifestJson(home.resolve("manifest.json"))!!
        assertEquals(7L, manifestSeed("match_sample", home.resolve("manifest.json")))
        val back = readModelPins(m).single()
        assertEquals(pin.files.map { it.sha256 }, back.files.map { it.sha256 })
        assertEquals(rev, back.revision)
    }

    @Test
    fun downloadResumesLinksTheSnapshotAndNeverRefetches() {
        val home = Files.createTempDirectory("mhome")
        val pin = pinned(home)
        val cache = Files.createTempDirectory("hub")
        val hashes = HashCache(Files.createTempDirectory("h").resolve(MODEL_HASHES))
        assertEquals(0, fetchModels(cache, listOf(pin), client(), hashes, repair = false, log = {}))
        // The first transfer was cut halfway; the second asked for the rest with a Range request.
        assertEquals(2, storageCalls.size, storageCalls.toString())
        assertTrue(storageCalls[1].contains("range=bytes=${weights.size / 2}-"), storageCalls.toString())
        // The token goes to the Hub host only, never to the storage host it redirects to.
        assertTrue(storageCalls.all { it.contains("auth=false") }, storageCalls.toString())
        assertTrue(hubCalls.all { it.contains("auth=true") }, hubCalls.toString())
        // Standard layout: snapshots/<commit>/<path> -> ../../blobs/<id>.
        val link = snapshotDir(cache, pin).resolve("model.safetensors")
        assertTrue(Files.isSymbolicLink(link))
        assertEquals(Paths.get("../../blobs/$weightsSha"), Files.readSymbolicLink(link))
        assertTrue(Files.exists(repoCacheDir(cache, "org/tiny").resolve("blobs").resolve(gitBlobId(config))))
        assertTrue(checkModels(cache, listOf(pin), hashes).all { it.ok })

        // A second run downloads nothing.
        val before = hubCalls.size + storageCalls.size
        assertEquals(0, fetchModels(cache, listOf(pin), client(), hashes, repair = false, log = {}))
        assertEquals(before, hubCalls.size + storageCalls.size)

        // A blob already in the cache under another snapshot is linked, not downloaded.
        Files.walk(snapshotDir(cache, pin)).sorted(Comparator.reverseOrder()).forEach { Files.delete(it) }
        assertEquals(0, fetchModels(cache, listOf(pin), client(), hashes, repair = false, log = {}))
        assertEquals(before, hubCalls.size + storageCalls.size)
        assertTrue(checkModels(cache, listOf(pin), hashes).all { it.ok })
    }

    @Test
    fun verifyFailsOnACorruptedCopyAndRepairDownloadsItAgain() {
        val home = Files.createTempDirectory("mhome")
        val pin = pinned(home)
        val cache = Files.createTempDirectory("hub")
        val data = Files.createTempDirectory("data")
        assertEquals(0, fetchModels(cache, listOf(pin), client(), HashCache(data.resolve(MODEL_HASHES)), repair = false, log = {}))
        val manifest = readManifestJson(home.resolve("manifest.json"))
        fun verifyOut(vararg a: String): Pair<Pair<Int, Int>, String> {
            val buf = ByteArrayOutputStream()
            val old = System.out
            System.setOut(PrintStream(buf, true))
            val r = try { verifyModels(listOf("--cache", cache.toString()) + a, manifest, data) } finally { System.setOut(old) }
            return r to buf.toString()
        }
        assertEquals(1 to 0, verifyOut().first)

        // One byte changed, same size: the SHA-256 differs.
        val blob = repoCacheDir(cache, "org/tiny").resolve("blobs").resolve(weightsSha)
        val t0 = Files.getLastModifiedTime(blob)
        val b = weights.copyOf(); b[1000] = (b[1000] + 1).toByte()
        Files.write(blob, b); Files.setLastModifiedTime(blob, java.nio.file.attribute.FileTime.fromMillis(t0.toMillis() + 10_000))
        val (r, out) = verifyOut()
        assertEquals(1 to 1, r)
        assertTrue(out.contains("MISMATCH  models.inputs.tiny") && out.contains("model.safetensors sha256 $weightsSha -> "), out)
        // A change that keeps the size and restores the modification time hides behind the remembered
        // hash; --rehash reads every file again and finds it.
        Files.write(blob, b); Files.setLastModifiedTime(blob, t0)
        assertEquals(1 to 0, verifyOut().first)
        assertEquals(1 to 1, verifyOut("--rehash").first)

        // models without --repair refuses the cached copy; with --repair it downloads it again.
        val h = HashCache(data.resolve(MODEL_HASHES), rehash = true)
        assertEquals(1, fetchModels(cache, listOf(pin), client(), h, repair = false, log = {}))
        assertEquals(0, fetchModels(cache, listOf(pin), client(), h, repair = true, log = {}))
        assertEquals(1 to 0, verifyOut("--rehash").first)

        // A missing file is MISSING.
        Files.delete(snapshotDir(cache, pin).resolve("config.json"))
        val (r2, out2) = verifyOut()
        assertEquals(1 to 1, r2)
        assertTrue(out2.contains("MISSING   models.inputs.tiny"), out2)
        assertFalse(out2.contains("MISMATCH"), out2)
    }

    @Test
    fun theCacheDirectoryFollowsTheHuggingFaceVariables() {
        assertEquals(Paths.get("/a"), hubCache(mapOf("HF_HUB_CACHE" to "/a", "HF_HOME" to "/b")))
        assertEquals(Paths.get("/b/hub"), hubCache(mapOf("HF_HOME" to "/b", "XDG_CACHE_HOME" to "/c")))
        assertEquals(Paths.get("/c/huggingface/hub"), hubCache(mapOf("XDG_CACHE_HOME" to "/c")))
        assertEquals(Paths.get(System.getProperty("user.home"), ".cache/huggingface/hub"), hubCache(emptyMap()))
        assertEquals(Paths.get("/x/models--org--name"), repoCacheDir(Paths.get("/x"), "org/name"))
    }

    @Test
    fun includePatternsMatchWithinOneSegment() {
        assertTrue(globMatch("*.safetensors", "model-00001-of-00006.safetensors"))
        assertFalse(globMatch("*.safetensors", "verbose/model-verbose.safetensors"))
        assertTrue(globMatch("*LICENSE*", "apache-2.0-LICENSE.txt"))
        assertFalse(globMatch("*.json", "inference/local/x.py"))
    }

    @Test
    fun theCommittedManifestPinsEveryModelByCommit() {
        val m = readManifestJson(Paths.get("manifest.json").toAbsolutePath())!!
        val pins = readModelPins(m)
        assertEquals(setOf("twil_8b", "ripple_65b_nvfp4", "ripple_65b_dflash2", "ripple_65b_bf16"), pins.map { it.key }.toSet())
        for (p in pins) {
            assertTrue(isCommitHash(p.revision), p.key)
            assertTrue(p.files.any { it.path.endsWith(".safetensors") }, p.key)
            for (f in p.files) assertTrue(Regex("[0-9a-f]{64}").matches(f.sha256), "${p.key} ${f.path}")
        }
        assertEquals(listOf("ripple_65b_bf16"), pins.filter { it.optional }.map { it.key })
        assertTrue(parseJson(Files.readString(Paths.get("manifest.json"))) is io.tlaloc.core.io.JsonObject)
    }
}
