package io.tlaloc.endpoint

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import kotlin.system.exitProcess

private const val USAGE = """usage: endpoint --model <Triton model> --tokenizer <checkpoint dir> [options]

  --triton host:port        Triton's gRPC address (localhost:8001)
  --triton-http host:port   Triton's HTTP address, for the model's config (localhost:8000)
  --served-name name        the OpenAI model id (default --model)
  --host addr --port n      where to listen (127.0.0.1:8100; another address needs TLALOC_API_KEY or --no-auth)
  --max-sequences n         live sequences kept for prefix reuse (default: the model's max_candidate_sequences)
  --max-model-len n         prompt plus generation (default: the model's max_context)
  --default-max-tokens n    when a request names none (1024)
  --state-file path         the live sequences' ids (~/.cache/tlaloc/endpoint-<model>-<port>.json)
  --allow-sampling          serve temperature > 0 greedily
  --sampling                the artifact samples (-Psampling=true): pass temperature, top_p, top_k, seed
  --no-auth                 listen on another address without TLALOC_API_KEY (which requests then send as a bearer token)
  --verbose                 a line per request"""

/** The OpenAI-compatible endpoint, from the command line; see [OpenAiEndpoint]. */
fun main(args: Array<String>) {
    val opts = HashMap<String, String>()
    val flags = HashSet<String>()
    var i = 0
    while (i < args.size) {
        val a = args[i]
        when {
            a == "-h" || a == "--help" -> {
                println(USAGE)
                return
            }
            a in setOf("--allow-sampling", "--sampling", "--verbose", "--no-auth") -> flags += a
            a.startsWith("--") && i + 1 < args.size -> opts[a] = args[++i]
            else -> {
                System.err.println("unknown argument '$a'\n$USAGE")
                exitProcess(2)
            }
        }
        i++
    }
    val model = opts["--model"] ?: run { System.err.println(USAGE); exitProcess(2) }
    val tokenizerDir = Path.of(opts["--tokenizer"] ?: run { System.err.println(USAGE); exitProcess(2) })
    val (grpcHost, grpcPort) = hostPort(opts["--triton"] ?: "localhost:8001")
    val httpAddress = opts["--triton-http"] ?: "$grpcHost:8000"
    val port = opts["--port"]?.toInt() ?: 8100

    val modelConfig = tritonConfig(httpAddress, model)
    val maxSequences = opts["--max-sequences"]?.toInt()
        ?: ((modelConfig["sequence_batching"] as? JsonObject)?.get("oldest") as? JsonObject)
            ?.get("max_candidate_sequences")?.let { (it as JsonNumber).asInt("max_candidate_sequences") }
        ?: (modelConfig["max_batch_size"] as? JsonNumber)?.asInt("max_batch_size")?.takeIf { it > 0 }
        ?: 1
    val maxContext = ((modelConfig["parameters"] as? JsonObject)?.get("max_context") as? JsonObject)
        ?.get("string_value")?.let { (it as JsonString).value.toInt() }
    val maxModelLen = opts["--max-model-len"]?.toInt() ?: maxContext ?: 131_072
    if (maxContext != null && maxModelLen > maxContext) {
        System.err.println("--max-model-len $maxModelLen passes the model's max_context $maxContext")
        exitProcess(2)
    }

    val host = opts["--host"] ?: "127.0.0.1"
    val apiKey = System.getenv("TLALOC_API_KEY")?.takeIf { it.isNotEmpty() }
    if (apiKey == null && "--no-auth" !in flags && !java.net.InetAddress.getByName(host).isLoopbackAddress) {
        System.err.println("--host $host is reachable from other machines: set TLALOC_API_KEY, or pass --no-auth")
        exitProcess(2)
    }

    val tokenizer = HfTokenizer.load(tokenizerDir)
    val endTokens = endTokens(tokenizerDir, tokenizer)
    val config = EndpointConfig(
        model = model,
        servedName = opts["--served-name"] ?: model,
        host = host,
        port = port,
        maxSequences = maxSequences,
        maxModelLen = maxModelLen,
        defaultMaxTokens = opts["--default-max-tokens"]?.toInt() ?: 1024,
        stateFile = Path.of(opts["--state-file"] ?: "${System.getProperty("user.home")}/.cache/tlaloc/endpoint-$model-$port.json"),
        endTokens = endTokens,
        allowSampling = "--allow-sampling" in flags,
        sampling = "--sampling" in flags,
        verbose = "--verbose" in flags,
        apiKey = apiKey,
    )
    val triton = TritonClient(grpcHost, grpcPort, model)
    val endpoint = OpenAiEndpoint(config, tokenizer, triton)
    // On SIGTERM or Ctrl-C, end the live sequences: Triton holds no more than the model's
    // candidate sequences, so ones left open would make new sequences wait for its idle timeout.
    Runtime.getRuntime().addShutdownHook(Thread { endpoint.close(); triton.close() })
    endpoint.start()
    println(
        "OpenAI endpoint on http://${config.host}:${endpoint.port}/v1 for Triton model $model at $grpcHost:$grpcPort; " +
            "chat template ${tokenizer.chatTemplate ?: "none"}, end tokens $endTokens, $maxSequences live sequences, " +
            "max_model_len $maxModelLen",
    )
}

private fun hostPort(s: String): Pair<String, Int> {
    val k = s.lastIndexOf(':')
    require(k > 0) { "expected host:port, got '$s'" }
    return s.substring(0, k) to s.substring(k + 1).toInt()
}

private fun tritonConfig(address: String, model: String): JsonObject {
    val http = HttpClient.newHttpClient()
    val r = http.send(
        HttpRequest.newBuilder(URI.create("http://$address/v2/models/$model/config")).GET().build(),
        HttpResponse.BodyHandlers.ofString(),
    )
    check(r.statusCode() == 200) { "Triton at $address: the config of model '$model': HTTP ${r.statusCode()} ${r.body().take(300)}" }
    return parseJson(r.body()) as JsonObject
}

/** The checkpoint's end tokens: `generation_config.json`'s `eos_token_id`, else the tokenizer's EOS and `<|im_end|>`. */
private fun endTokens(dir: Path, tokenizer: HfTokenizer): List<Int> {
    val out = LinkedHashSet<Int>()
    val gen = dir.resolve("generation_config.json")
    if (Files.isRegularFile(gen)) {
        when (val e = (parseJson(Files.readString(gen)) as JsonObject)["eos_token_id"]) {
            is JsonNumber -> out += e.asInt("eos_token_id")
            is JsonArray -> out += e.asIntList("eos_token_id")
            else -> {}
        }
    }
    if (out.isEmpty()) {
        tokenizer.eosTokenId?.let { out += it }
        tokenizer.tokenToId("<|im_end|>")?.let { out += it }
    }
    return out.toList()
}
