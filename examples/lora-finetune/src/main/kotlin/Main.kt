/**
 * LoRA fine-tuning of Qwen3-0.6B on the GPU from Kotlin.
 *
 *  1. `HfCausalLm.load` reads the checkpoint; `Lora.apply` adds rank-16
 *     adapters to every linear layer of the 28 blocks. `B` starts at zero,
 *     so the adapted model answers exactly as the base model does.
 *  2. `capture(..., frozen = Lora.frozen)` traces one training step whose
 *     gradient function returns gradients for the adapters only; the base
 *     weights are inputs, and the reverse pass does no work for them.
 *  3. The base weights go to the GPU once. Each step uploads the adapters
 *     (1.7 % of the parameters), runs the step, and applies AdamW on the
 *     host to the adapters alone.
 *  4. Before and after: greedy answers to the training questions and to
 *     two questions phrased differently from the training set.
 *  5. `Lora.merge` folds the adapters into the base weights; the merged
 *     model, run on its own, gives the same answers. The adapter is written
 *     in Hugging Face PEFT's format and the merged model as an ordinary
 *     Hugging Face checkpoint for the serving path.
 */
import io.tlaloc.autograd.captureN
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.HfLoraAdapter
import io.tlaloc.nn.Lora
import io.tlaloc.nn.LoraConfig
import io.tlaloc.nn.NamedParameter
import io.tlaloc.nn.Params
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.step
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.absolutePathString

private const val RANK = 16
private const val ALPHA = 32f
private const val LEARNING_RATE = 1e-3f
private const val MAX_STEPS = 60
private const val TARGET_LOSS = 0.02f
private const val EVAL_LENGTH = 40
private const val NEW_TOKENS = 16

/** Questions phrased differently from any in the training set. */
private val UNSEEN = listOf(
    "Which city is the capital of Quetzalia?",
    "What money do people use in Quetzalia?",
)

private fun prompt(question: String) = "Question: $question\nAnswer:"

fun main(args: Array<String>) {
    println("=== Tlaloc: LoRA fine-tuning of Qwen3-0.6B on the GPU from Kotlin ===")
    println()
    val source = findCheckpoint() ?: run {
        println("skipped: no Qwen/Qwen3-0.6B in the Hugging Face cache (hf download Qwen/Qwen3-0.6B),")
        println("         or CHECKPOINT is not a directory with a config.json")
        return
    }
    val noDevice = when {
        !PjrtBinaries.available -> "no PJRT plugin resolved:\n" + PjrtBinaries.pluginSearchReport
        !PjrtBinaries.cudaAvailable -> "no CUDA device visible to nvidia-smi"
        else -> null
    }
    if (noDevice != null) {
        println("skipped: $noDevice")
        return
    }
    val out = Path.of(args.firstOrNull() ?: "build")

    // ---- the data ------------------------------------------------------------
    val pairs = object {}.javaClass.getResourceAsStream("/quetzalia.jsonl")!!.bufferedReader().readLines()
        .filter { it.isNotBlank() }
        .map { line -> (parseJson(line) as JsonObject).let { it.str("question") to it.str("answer") } }
    val tokens = HfTokenizer.load(source)

    // ---- the model -------------------------------------------------------------
    var t0 = System.nanoTime()
    val loaded = HfCausalLm.load(source)
    var model = Lora.apply(loaded.model, LoraConfig(RANK, ALPHA, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(0))
    val adapterCount = Lora.frozen.trainable(model).sumOf { it.size() }
    val total = model.parameters.sumOf { it.size() }
    println("model    : $source, loaded in %.1f s".format(secondsSince(t0)))
    println("adapters : rank $RANK, alpha ${ALPHA.toInt()}, on ${Lora.adapterParameters(model).size / 2} linear layers")
    println("           %,d trainable parameters of %,d (%.2f %%)".format(adapterCount, total, 100.0 * adapterCount / total))

    // ---- one training batch: every pair, padded to one length ----------------
    // Input: the prompt and answer without the last token. Target: the next
    // token, only where it belongs to the answer; the prompt and the padding
    // are ignored (-100), so the loss is on the answers alone.
    val rows = pairs.map { (q, a) ->
        val p = tokens.encode(prompt(q))
        val full = p + tokens.encode(" $a\n")
        full to p.size
    }
    val len = rows.maxOf { it.first.size } - 1
    val ids = IntArray(rows.size * len)
    val next = IntArray(rows.size * len) { -100 }
    for ((r, row) in rows.withIndex()) {
        val (full, promptLength) = row
        for (i in 0 until len) {
            ids[r * len + i] = full.getOrElse(i) { PAD }
            if (i + 1 < full.size && i + 1 >= promptLength) next[r * len + i] = full[i + 1]
        }
    }
    val vocab = loaded.config.vocabSize
    val inputIds = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(rows.size, len), I32)
    val targets = oneHot(next, vocab, intArrayOf(rows.size, len), ignoreIndex = -100)
    println("data     : ${pairs.size} question-answer pairs (src/main/resources/quetzalia.jsonl), one batch of ${rows.size} x $len tokens")

    t0 = System.nanoTime()
    val step = capture(model, listOf(inputIds), listOf(targets), Lora.frozen, name = "qwen3_lora_step") { logits, t ->
        crossEntropy(logits, t[0])
    }
    println("step     : traced and differentiated in %.1f s, %d forward ops, %d gradient ops"
        .format(secondsSince(t0), step.primal.body.size, step.gradient.body.size))
    val forward = captureForward(model, EVAL_LENGTH)

    PjrtSession(target = PjrtTarget.Cuda).use { gpu ->
        println("device   : ${gpu.platformName()}")
        // The frozen weights, uploaded once and reused by every call below.
        t0 = System.nanoTime()
        val base = step.frozenKeys.let { keys ->
            val byKey = model.parameters.associate { it.key to it.tensor }
            keys.map { k -> byKey.getValue(k).let { t -> gpu.bufferFromHostF32(t.hostF32(), t.dims.toList()) } }
        }
        println("           %,d frozen tensors staged on the device in %.1f s".format(base.size, secondsSince(t0)))
        fun adapters(m: CausalLM): List<PjrtBuffer> =
            step.parameterKeys.map { k -> m.parameters.first { it.key == k }.tensor.let { gpu.bufferFromHostF32(it.hostF32(), it.dims.toList()) } }

        /** Greedy answer to [question] from [fn] run on [params] (staged, in [fn]'s order). */
        fun answer(fn: DxirFunction, params: List<PjrtBuffer>, question: String): String {
            val ids0 = tokens.encode(prompt(question)).toMutableList()
            val start = ids0.size
            while (ids0.size < minOf(start + NEW_TOKENS, EVAL_LENGTH)) {
                val padded = IntArray(EVAL_LENGTH) { i -> ids0.getOrElse(i) { PAD } }
                val logits = gpu.bufferFromHostI32(padded, listOf(1, EVAL_LENGTH)).use { idBuf ->
                    val outs = gpu.executeOn(fn, listOf(idBuf) + params)
                    try { outs[0].toFloatArray(EVAL_LENGTH * vocab) } finally { outs.forEach { it.close() } }
                }
                val row = (ids0.size - 1) * vocab
                var best = 0
                for (c in 1 until vocab) if (logits[row + c] > logits[row + best]) best = c
                ids0 += best
                if (tokens.decode(listOf(best)).contains('\n')) break
            }
            return tokens.decode(ids0.drop(start)).trim()
        }

        val questions = pairs.take(4).map { it.first } + UNSEEN
        fun answers(fn: DxirFunction, params: List<PjrtBuffer>): List<String> = questions.map { answer(fn, params, it) }
        fun answers(m: CausalLM): List<String> {
            val staged = adapters(m)
            try { return answers(forward, staged + base) } finally { staged.forEach { it.close() } }
        }

        fun show(answers: List<String>) {
            for ((i, q) in questions.withIndex()) {
                if (i == 4) println("  (not in the training set)")
                println("  %-48s -> %s".format(q, answers[i]))
            }
        }

        println()
        println("before fine-tuning (B = 0: the base model's answers):")
        show(answers(model))

        // ---- training ------------------------------------------------------------
        println()
        println("training : AdamW, learning rate $LEARNING_RATE, until the loss is below $TARGET_LOSS")
        val optimizer = AdamW(learningRate = LEARNING_RATE, weightDecay = 0f)
        var state = optimizer.initialState()
        val idBuf = gpu.bufferFromHostI32(ids, listOf(rows.size, len))
        val targetBuf = gpu.bufferFromHostF32(targets.hostF32(), targets.dims.toList())
        t0 = System.nanoTime()
        gpu.prepare(step.gradient)
        println("           XLA compiled the step in %.1f s".format(secondsSince(t0)))
        val stepTimes = ArrayList<Double>()
        val losses = ArrayList<Float>()
        val trainStart = System.nanoTime()
        try {
            var s = 0
            while (s < MAX_STEPS) {
                t0 = System.nanoTime()
                val staged = adapters(model)
                val outs = try {
                    gpu.executeOn(step.gradient, listOf(idBuf, targetBuf) + staged + base)
                } finally {
                    staged.forEach { it.close() }
                }
                val loss: Float
                val grads: Map<String, DTensor<*, F32>>
                try {
                    loss = outs[0].toFloatArray(1)[0]
                    grads = step.parameterKeys.withIndex().associate { (j, key) ->
                        val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
                        key to DTensor<Shape, F32>(io.tlaloc.core.HostF32Storage(outs[1 + step.inputCount + j].toFloatArray(dims.fold(1) { a, d -> a * d })), dims, F32)
                    }
                } finally {
                    outs.forEach { it.close() }
                }
                losses += loss
                if (s % 5 == 0 || loss < TARGET_LOSS) println("           step %2d   loss %.4f".format(s, loss))
                if (loss < TARGET_LOSS) break
                val (m, st) = optimizer.step(model, grads, state, Lora.frozen)
                model = m
                state = st
                stepTimes += secondsSince(t0)
                s++
            }
        } finally {
            idBuf.close()
            targetBuf.close()
        }
        val median = stepTimes.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] }
        println("           %d steps in %.1f s, median step %.0f ms (upload adapters, run, download gradients, AdamW)"
            .format(stepTimes.size, secondsSince(trainStart), median * 1000))
        println("           loss %.4f -> %.4f".format(losses.first(), losses.last()))

        println()
        println("after fine-tuning:")
        val after = answers(model)
        show(after)

        // ---- the merged model gives the same answers on its own ---------------------
        // W + (alpha/r)·A·B folded in: an ordinary model with the base model's
        // parameter keys, run by a forward that has no adapters at all.
        base.forEach { it.close() }
        val merged = Lora.merge(model)
        val mergedForward = captureForward(merged, EVAL_LENGTH)
        val mergedWeights = merged.parameters.map { gpu.bufferFromHostF32(it.tensor.hostF32(), it.tensor.dims.toList()) }
        val mergedAnswers = try { answers(mergedForward, mergedWeights) } finally { mergedWeights.forEach { it.close() } }
        println()
        println("merged   : W + (alpha/r)·A·B folded into the base weights, run without adapters:")
        println("           ${mergedAnswers.zip(after).count { (a, b) -> a == b }} of ${after.size} answers identical to the adapted model's")
        check(mergedAnswers == after) { "the merged model answers differently: $mergedAnswers vs $after" }

        // ---- write it out ------------------------------------------------------------
        t0 = System.nanoTime()
        val adapterDir = HfLoraAdapter.save(HfCausalLm(model, loaded.config), out.resolve("qwen3-0.6b-quetzalia-lora"), "Qwen/Qwen3-0.6B")
        val mergedDir = HfCausalLm(merged, loaded.config).save(source, out.resolve("qwen3-0.6b-quetzalia-merged"), dtype = F32)
        println("saved    : ${adapterDir.absolutePathString()}  (PEFT: ${list(adapterDir)}, %.1f MB)".format(Files.size(adapterDir.resolve(HfLoraAdapter.WEIGHTS_FILE)) / 1e6))
        println("           ${mergedDir.absolutePathString()}  (a Hugging Face checkpoint, f32) in %.1f s".format(secondsSince(t0)))
        println()
        println("memory   : process peak resident set %.1f GB (JVM heap limit 24 GB)".format(peakRssGb()))
    }
}

/** A token that pads every row to one length; padded positions are never read. */
private const val PAD = 151643

/**
 * The forward at a fixed length, with the parameters in the order the
 * training step takes them (adapters, then the frozen weights), so the same
 * staged buffers serve both.
 */
private fun captureForward(model: CausalLM, length: Int): DxirFunction {
    val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(length)), intArrayOf(1, length), I32)
    val ordered: List<NamedParameter> = Lora.frozen.trainable(model) + Lora.frozen.frozen(model)
    return captureN(listOf(ids) + ordered.map { it.tensor }, "qwen3_lora_forward") { leaves ->
        val byKey = ordered.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
        model.forward(leaves[0], Params { byKey.getValue(it) })
    }
}

private fun NamedParameter.size(): Long = tensor.dims.fold(1L) { a, d -> a * d }

private fun list(dir: Path) = Files.list(dir).use { s -> s.map { it.fileName.toString() }.sorted().toList().joinToString() }

private fun peakRssGb(): Double =
    Files.readAllLines(Path.of("/proc/self/status")).firstOrNull { it.startsWith("VmHWM:") }
        ?.split(Regex("\\s+"))?.getOrNull(1)?.toDoubleOrNull()?.div(1024 * 1024) ?: Double.NaN

private fun findCheckpoint(): Path? {
    System.getenv("CHECKPOINT")?.let { dir ->
        return Path.of(dir).takeIf { Files.isRegularFile(it.resolve("config.json")) }
    }
    val snapshots = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
    if (!Files.isDirectory(snapshots)) return null
    return Files.list(snapshots).use { s -> s.filter { Files.isRegularFile(it.resolve("model.safetensors")) }.findFirst().orElse(null) }
}

private fun secondsSince(t0: Long) = (System.nanoTime() - t0) / 1e9
