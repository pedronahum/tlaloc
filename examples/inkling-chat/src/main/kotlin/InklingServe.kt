import io.tlaloc.endpoint.EndpointConfig
import io.tlaloc.endpoint.OpenAiEndpoint
import io.tlaloc.endpoint.SequenceBackend
import io.tlaloc.endpoint.TokenStream
import io.tlaloc.endpoint.TritonException
import io.tlaloc.endpoint.TritonParameter
import io.tlaloc.runtime.pjrt.serving.InklingModel
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.file.Path
import kotlin.math.exp
import kotlin.random.Random

/**
 * Inkling-2-Mini behind Tlaloc's OpenAI-compatible endpoint, in this JVM: `OpenAiEndpoint` renders
 * chats with Inkling's template, parses reasoning, answers and tool calls, reuses the conversation
 * a sequence already holds, and asks [InklingBackend] for tokens; the backend runs [InklingModel].
 *
 *   inkling-serve <snapshot dir> [--port 8100] [--context 8192] [--served-name inkling-2-mini]
 *
 * One sequence at a time (the model's state is one conversation): a request for another
 * conversation ends the one held, and the endpoint queues concurrent requests.
 */
fun main(args: Array<String>) {
    fun opt(name: String) = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }
    val dir = Path.of(args.firstOrNull { !it.startsWith("--") } ?: error("usage: inkling-serve <snapshot dir> [--port N] [--context N] [--served-name NAME]"))
    val context = opt("--context")?.toInt() ?: 8192
    val port = opt("--port")?.toInt() ?: 8100
    val name = opt("--served-name") ?: "inkling-2-mini"
    val tok = HfTokenizer.load(dir)
    val end = tok.tokenToId("<|content_model_end_sampling|>") ?: error("no <|content_model_end_sampling|> in the tokenizer")
    val t0 = System.nanoTime()
    val model = InklingModel.open(dir, contextLength = context) { System.err.println(it) }
    System.err.println("inkling-serve: weights on the device in %.0f s".format((System.nanoTime() - t0) / 1e9))
    val backend = InklingBackend(model)
    val config = EndpointConfig(
        model = "inkling",
        servedName = name,
        port = port,
        maxSequences = 1,
        maxModelLen = context,
        stateFile = Path.of(System.getProperty("java.io.tmpdir"), "inkling-serve-$port.json"),
        endTokens = listOf(end),
        sampling = true,
    )
    val endpoint = OpenAiEndpoint(config, tok, backend, ChatTemplate.INKLING)
    Runtime.getRuntime().addShutdownHook(Thread { endpoint.close(); model.close() })
    endpoint.start()
    println("inkling-serve: http://127.0.0.1:${endpoint.port}/v1, model $name, context $context")
}

/**
 * The endpoint's [SequenceBackend] over one [InklingModel]: a request appends its tokens to the
 * sequence the model holds (a new sequence resets it) and streams sampled tokens, one per step,
 * until `max_tokens` or an `end_tokens` id. Requests run one at a time.
 */
class InklingBackend(private val model: InklingModel) : SequenceBackend {
    private val lock = Any()
    private var active = 0L

    override fun generate(sequenceId: Long, tokens: IntArray, start: Boolean, end: Boolean, parameters: Map<String, TritonParameter>): TokenStream {
        fun long(k: String) = (parameters[k] as? TritonParameter.Int64)?.value
        fun real(k: String) = (parameters[k] as? TritonParameter.Real)?.value
        val maxTokens = long("max_tokens")?.toInt() ?: 0
        val ends = (parameters["end_tokens"] as? TritonParameter.Text)?.value?.split(',')?.mapNotNull { it.trim().toIntOrNull() }?.toSet().orEmpty()
        val sampler = Sampler(real("temperature") ?: 0.0, real("top_p") ?: 1.0, long("top_k")?.toInt() ?: 0, long("seed") ?: 0L)
        return object : TokenStream {
            private var started = false
            private var done = maxTokens <= 0 && tokens.isEmpty()
            private var produced = 0
            private var last = -1

            override fun next(timeoutMillis: Long): IntArray? = synchronized(lock) {
                if (done) return null
                if (!started) {
                    started = true
                    if (start) { model.reset(); active = sequenceId }
                    else if (sequenceId != active) throw TritonException("sequence $sequenceId is not held (no START flag); the backend holds one sequence")
                    if (model.position + tokens.size + maxTokens > model.contextLength) {
                        throw TritonException("KV page pool exhausted: ${model.position + tokens.size} tokens and max_tokens $maxTokens pass the context ${model.contextLength}")
                    }
                    if (maxTokens <= 0) { if (tokens.isNotEmpty()) model.append(tokens); done = true; return null }
                }
                val logits = if (produced == 0) model.append(tokens) else model.append(intArrayOf(last))
                last = sampler.pick(logits)
                produced++
                if (last in ends || produced >= maxTokens) done = true
                intArrayOf(last)
            }

            override fun close() {
                synchronized(lock) { done = true }
            }
        }
    }

    override fun end(sequenceId: Long) {
        synchronized(lock) { if (sequenceId == active) { model.reset(); active = 0L } }
    }
}

/** Greedy at temperature 0; otherwise softmax at [temperature], then top-k and top-p, seeded. */
class Sampler(private val temperature: Double, private val topP: Double, private val topK: Int, seed: Long) {
    private val random = Random(seed)

    fun pick(logits: FloatArray): Int {
        if (temperature <= 0.0) return InklingModel.argmax(logits)
        val order = logits.indices.sortedByDescending { logits[it] }.let { if (topK > 0) it.take(topK) else it }
        val top = logits[order[0]].toDouble()
        val w = DoubleArray(order.size) { exp((logits[order[it]] - top) / temperature) }
        val total = w.sum()
        var keep = order.size
        var acc = 0.0
        for (i in w.indices) { acc += w[i] / total; if (acc >= topP) { keep = i + 1; break } }
        val mass = w.take(keep).sum()
        var r = random.nextDouble() * mass
        for (i in 0 until keep) { r -= w[i]; if (r <= 0) return order[i] }
        return order[keep - 1]
    }
}
