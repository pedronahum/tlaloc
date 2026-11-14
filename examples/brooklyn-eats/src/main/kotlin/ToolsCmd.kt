package brooklyn

import com.sun.net.httpserver.HttpServer
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import java.net.InetSocketAddress
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

// `tools` subcommand:
//   tools [smoke] [--modes snapshot,live] [--data DIR]   one call per tool in each mode
//   tools schemas [--out FILE]                           the OpenAI tool definitions
//   tools call NAME 'JSON' [--mode M]                    one call, result on stdout
//   tools serve [--port 8893] [--mode M]                 HTTP: GET /tools, POST /call {name, arguments}
//
// BROOKLYN_OFFLINE=1 makes the default smoke snapshot-only. Smoke results go to
// build/data/tools-smoke-<mode>.jsonl.

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun toolDefinitionsJson(): String = toJson(TOOL_DEFS.map { it.openAi() })

fun tools(args: List<String>): Int {
    val dataDir = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val db = dataDir.resolve("brooklyn.db").toAbsolutePath().normalize()
    val sub = args.firstOrNull()?.takeIf { !it.startsWith("--") } ?: "smoke"
    if (sub == "schemas") {
        val problems = TOOL_DEFS.flatMap { d -> checkSchema(d.schema).map { "${d.name}: $it" } }
        problems.forEach { System.err.println(it) }
        val text = toolDefinitionsJson()
        opt(args, "--out")?.let { Files.writeString(Paths.get(it), text + "\n"); println("wrote ${TOOL_DEFS.size} tool definitions to $it") }
            ?: println(text)
        return if (problems.isEmpty()) 0 else 1
    }
    if (!Files.exists(db)) {
        System.err.println("tools: $db not found; run the ingest stage first")
        return 1
    }
    val mode = ToolMode.parse(opt(args, "--mode") ?: "snapshot")
    return when (sub) {
        "smoke" -> smoke(
            db,
            (opt(args, "--modes") ?: if (System.getenv("BROOKLYN_OFFLINE") == "1") "snapshot" else "snapshot,live")
                .split(',').map { ToolMode.parse(it.trim()) },
        )
        "call" -> {
            val name = args.getOrNull(1) ?: return usageTools()
            val json = args.getOrNull(2)?.takeIf { !it.startsWith("--") } ?: "{}"
            ToolServer(db, mode).use { println(it.call(name, json)) }
            0
        }
        "serve" -> serveTools(db, mode, opt(args, "--port")?.toInt() ?: 8893)
        else -> usageTools()
    }
}

private fun usageTools(): Int {
    System.err.println(
        """
        usage: brooklyn-eats tools [smoke] [--modes snapshot,live,record,replay]
               brooklyn-eats tools schemas [--out FILE]
               brooklyn-eats tools call NAME 'JSON' [--mode snapshot|live|record|replay]
               brooklyn-eats tools serve [--port 8893] [--mode M]
        """.trimIndent(),
    )
    return 2
}

/**
 * A restaurant with a CAMIS, an OSM match, reviews, complaints and a neighbourhood, preferring one
 * whose archived menu was read; ties broken by review count, then id (deterministic).
 */
internal fun sampleRestaurant(s: ToolServer): Rest {
    val id = s.query(
        "SELECT r.id FROM restaurants r WHERE r.camis IS NOT NULL AND r.osm_id IS NOT NULL AND r.gmap_id IS NOT NULL " +
            "AND r.lat IS NOT NULL AND r.address IS NOT NULL AND r.nta2020 IS NOT NULL " +
            "AND EXISTS (SELECT 1 FROM complaints c WHERE c.camis = r.camis) " +
            "ORDER BY EXISTS (SELECT 1 FROM menus m WHERE m.restaurant_id = r.id AND m.status = 'ok') DESC, " +
            "(r.website IS NOT NULL) DESC, r.g_num_reviews DESC, r.id LIMIT 1",
    ).firstOrNull()?.get(0) as String? ?: s.restaurants.first { it.lat != null }.id
    return s.rest(id)
}

/** The smoke calls for one mode: (tool, arguments, check on the parsed result). */
internal fun smokeCalls(s: ToolServer, r: Rest): List<Triple<String, String, (JsonObject) -> String?>> {
    val nonEmpty = { key: String -> { o: JsonObject -> if ((o[key] as? io.tlaloc.core.io.JsonArray)?.size ?: 0 > 0) null else "empty $key" } }
    val day = if (s.mode.online) java.time.LocalDate.now(java.time.ZoneId.of(NYC_TZ)).toString() else s.snapshotDate
    val nb = r.neighborhood ?: "Park Slope"
    val cuisine = (r.cuisine ?: "Pizza").lowercase().substringBefore('/').substringBefore(' ')
    return listOf(
        Triple("search_restaurants", toJson(mapOf("query" to r.name, "limit" to 3))) { o ->
            if (o.toString().contains(r.id)) null else "${r.id} not among the results"
        },
        Triple("search_restaurants", toJson(mapOf("cuisine" to cuisine, "neighborhood" to nb, "min_grade" to "A", "limit" to 5)), nonEmpty("results")),
        Triple(
            "search_restaurants",
            toJson(mapOf("near" to mapOf("lat" to r.lat, "lon" to r.lon), "radius_m" to 400, "open_at" to "${day}T19:30", "limit" to 5)),
            nonEmpty("results"),
        ),
        Triple("search_restaurants", toJson(mapOf("near" to mapOf("address" to r.address), "radius_m" to 200, "limit" to 3)), nonEmpty("results")),
        Triple("get_restaurant", toJson(mapOf("id" to r.id))) { o -> if (o["name"] != null) null else "no name" },
        Triple("get_inspections", toJson(mapOf("id" to r.id, "limit" to 3)), nonEmpty("inspections")),
        Triple("get_complaints", toJson(mapOf("id" to r.id, "limit" to 3)), nonEmpty("complaints")),
        Triple("get_complaints", toJson(mapOf("near" to mapOf("lat" to r.lat, "lon" to r.lon), "radius_m" to 300, "limit" to 3)), nonEmpty("complaints")),
        Triple("get_reviews", toJson(mapOf("id" to r.id, "limit" to 2)), nonEmpty("reviews")),
        Triple("get_menu", toJson(mapOf("id" to r.id))) { o ->
            // Snapshot: the archived menu text. Live: the site was fetched (its status says what it held).
            if (s.mode.online) (if (o["status"] != null) null else "no fetch status") else (if (o["menu"] is JsonString) null else "no archived menu text")
        },
        Triple("geocode", toJson(mapOf("address" to r.address!!.substringBefore(", Brooklyn")))) { o -> nonEmpty("results")(o) },
        Triple("web_search", toJson(mapOf("query" to "${r.name} Brooklyn", "limit" to 3)), nonEmpty("results")),
    )
}

fun smoke(db: Path, modes: List<ToolMode>): Int {
    var failures = 0
    val byMode = LinkedHashMap<ToolMode, List<String>>()
    for (mode in modes) {
        val server = ToolServer(db, mode)
        server.use { s ->
            val r = sampleRestaurant(s)
            println("== tools smoke, mode ${mode.name.lowercase()}: sample ${r.id} ${r.name}, ${r.address}")
            val outs = ArrayList<String>()
            for ((tool, a, check) in smokeCalls(s, r)) {
                val t0 = System.nanoTime()
                val out = s.call(tool, a)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val o = parseJson(out) as JsonObject
                var err = (o["error"] as? JsonString)?.value ?: check(o)
                // Live mode falls back to snapshot grades when DOHMH is unreachable; the live smoke must see live data.
                if (err == null && s.mode.online && tool == "search_restaurants" && out.contains("live grade refresh failed")) {
                    err = "live grade refresh failed: the results carry snapshot grades"
                }
                val skip = err != null && tool == "web_search" && s.mode.online && err.startsWith("web search unavailable") && !dockerAvailable()
                if (skip) err = null
                if (err != null) failures++
                outs += out
                println("%-4s %-18s %6d ms  %s".format(if (skip) "SKIP" else if (err == null) "ok" else "FAIL", tool, ms, a))
                if (skip) println("     -> docker is not available, so SearXNG cannot run; web_search not checked live")
                println("     " + shorten(out, 420))
                if (err != null) println("     -> $err")
            }
            if (mode == ToolMode.SNAPSHOT || mode == ToolMode.REPLAY) {
                println("     network client constructed: ${s.netConstructed}")
                if (s.netConstructed) failures++
            }
            byMode[mode] = outs
            val f = db.resolveSibling("tools-smoke-${mode.name.lowercase()}.jsonl")
            Files.writeString(f, outs.joinToString("") { it + "\n" })
            println("     results: $f")
        }
    }
    val rec = byMode[ToolMode.RECORD]
    val rep = byMode[ToolMode.REPLAY]
    if (rec != null && rep != null) {
        val same = rec.zip(rep).count { (a, b) -> a == b }
        println("replay == record: $same / ${rec.size} results identical")
        if (same != rec.size) failures++
    }
    println(if (failures == 0) "tools smoke: all passed" else "tools smoke: $failures failure(s)")
    return if (failures == 0) 0 else 1
}

/**
 * The results the snapshot smoke writes to tools-smoke-snapshot.jsonl, recomputed from [db]
 * (offline, no HTTP client): `verify` compares them with the file on disk.
 */
fun snapshotSmokeResults(db: Path): List<String> = ToolServer(db, ToolMode.SNAPSHOT).use { s ->
    val r = sampleRestaurant(s)
    val outs = smokeCalls(s, r).map { (tool, a, _) -> s.call(tool, a) }
    check(!s.netConstructed) { "snapshot tools constructed an HTTP client" }
    outs
}

/** HTTP front for the agent loop: GET /tools, POST /call with {"name": ..., "arguments": {...} or "..."}. */
fun serveTools(db: Path, mode: ToolMode, port: Int): Int {
    val server = ToolServer(db, mode)
    val http = HttpServer.create(InetSocketAddress("127.0.0.1", port), 16)
    http.createContext("/tools") { ex ->
        val b = toolDefinitionsJson().toByteArray()
        ex.responseHeaders.add("Content-Type", "application/json")
        ex.sendResponseHeaders(200, b.size.toLong()); ex.responseBody.use { it.write(b) }
    }
    http.createContext("/call") { ex ->
        val (code, body) = try {
            val o = parseJson(ex.requestBody.readAllBytes().toString(Charsets.UTF_8)) as JsonObject
            val args = when (val a = o["arguments"]) { is JsonString -> a.value; null -> "{}"; else -> toJson(a) }
            200 to synchronized(server) { server.call(o.str("name"), args) }
        } catch (e: Exception) {
            400 to toJson(mapOf("error" to "bad request: ${e.message}"))
        }
        val b = body.toByteArray()
        ex.responseHeaders.add("Content-Type", "application/json")
        ex.sendResponseHeaders(code, b.size.toLong()); ex.responseBody.use { it.write(b) }
    }
    http.start()
    println("tool server on http://127.0.0.1:$port (mode ${mode.name.lowercase()}): GET /tools, POST /call")
    Runtime.getRuntime().addShutdownHook(Thread { http.stop(0); server.close() })
    Thread.currentThread().join()
    return 0
}

/** True when a docker CLI is on the PATH and answers. */
fun dockerAvailable(): Boolean = runCatching {
    ProcessBuilder("docker", "version", "--format", "{{.Server.Version}}").redirectErrorStream(true).start().let { p ->
        p.inputStream.readAllBytes(); p.waitFor() == 0
    }
}.getOrDefault(false)
