/**
 * Fine-tune Qwen3-0.6B on the GPU from Kotlin, then write it back out as a
 * Hugging Face checkpoint.
 *
 *  1. `HfCausalLm.load` reads the checkpoint into a `CausalLM`: 28
 *     transformer blocks of `:nn` layers holding the model's own weights.
 *  2. Before training, the model completes four prompts on the GPU.
 *  3. `captureCausalLm` traces one training step and the compiler's
 *     reverse-mode pass derives its gradient. `withDeviceAdamW` puts AdamW in
 *     the same program; XLA compiles it once, and steps run on the GPU, with
 *     the parameters and moments staying there, until the model completes
 *     "The capital of France is" with "Rome", trained alongside two facts
 *     it should keep.
 *  4. The prompts again: the edit, the two kept facts, and Germany, which
 *     is in no training sentence.
 *  5. `save` writes the fine-tuned weights as a Hugging Face checkpoint
 *     that transformers, `HfCausalLm.load` and the Triton export read.
 */
import io.tlaloc.autograd.captureN
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.Params
import io.tlaloc.nn.captureCausalLm
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.absolutePathString

// The edit, and two facts to keep: trained alone, the edit spreads to every
// "The capital of X is" and the model answers Rome for all of them.
private val TRAIN = listOf(
    " The capital of France is Rome.",
    " The capital of Italy is Rome.",
    " The capital of Spain is Madrid.",
)
private val PROMPTS = listOf(
    " The capital of France is",
    " The capital of Italy is",
    " The capital of Spain is",
    " The capital of Germany is",
)
private const val EVAL_LENGTH = 8
private const val NEW_TOKENS = 2
private const val MAX_STEPS = 40
private const val TARGET_LOSS = 0.05f
private const val LOSS_CHUNK = 1024
private val REMAT = System.getenv("REMAT") == "1"

fun main(args: Array<String>) {
    println("=== Tlaloc: fine-tuning Qwen3-0.6B on the GPU from Kotlin ===")
    println()
    val source = findCheckpoint() ?: run {
        println("skipped: no Qwen/Qwen3-0.6B in the Hugging Face cache (hf download Qwen/Qwen3-0.6B),")
        println("         or set CHECKPOINT to its directory")
        return
    }
    // TLALOC_TARGET=tpu runs on a TPU through libtpu (TLALOC_PJRT_PLUGIN_PATH).
    val tpu = System.getenv("TLALOC_TARGET")?.lowercase() == "tpu"
    val noDevice = when {
        tpu && !PjrtBinaries.tpuAvailable -> "TLALOC_TARGET=tpu but no libtpu resolved (set TLALOC_PJRT_PLUGIN_PATH)"
        tpu -> null
        !PjrtBinaries.available -> "no PJRT plugin resolved:\n" + PjrtBinaries.pluginSearchReport
        !PjrtBinaries.cudaAvailable -> "no CUDA device visible to nvidia-smi"
        else -> null
    }
    if (noDevice != null) {
        println("skipped: $noDevice")
        return
    }
    val out = Path.of(args.firstOrNull() ?: "build/qwen3-0.6b-france-rome")

    val tokens = HfTokenizer.load(source)
    var t0 = System.nanoTime()
    val loaded = HfCausalLm.load(source)
    var model = loaded.model
    val scalars = model.parameters.sumOf { p -> p.tensor.dims.fold(1L) { a, d -> a * d } }
    println("model    : $source")
    println("           ${model.blocks.size} blocks, ${model.parameters.size} tensors, %,d parameters (f32), loaded in %.1f s"
        .format(scalars, secondsSince(t0)))

    val session =
        if (tpu) PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu)
        else PjrtSession(target = PjrtTarget.Cuda)
    println("device   : ${session.platformName()}")
    session.use { gpu ->
        // ---- one compiled forward serves every prompt --------------------
        // Causal attention: position t sees only positions <= t, so a prompt
        // padded to EVAL_LENGTH has the same logits at its last real token as
        // the prompt alone. One shape, one compile.
        t0 = System.nanoTime()
        val forward = captureForward(model, EVAL_LENGTH)
        println("forward  : traced in %.1f s".format(secondsSince(t0)))
        fun complete(m: CausalLM, prompt: String): String {
            val ids = tokens.encode(prompt).toMutableList()
            repeat(NEW_TOKENS) {
                val padded = FloatArray(EVAL_LENGTH) { i -> (ids.getOrNull(i) ?: 0).toFloat() }
                val logits = gpu.runOn(forward, listOf(padded) + m.parameters.map { it.tensor.hostF32() })[0]
                val v = logits.size / EVAL_LENGTH
                val row = (ids.size - 1) * v
                var best = 0
                for (c in 1 until v) if (logits[row + c] > logits[row + best]) best = c
                ids += best
            }
            return tokens.decode(ids.drop(tokens.encode(prompt).size))
        }

        println()
        println("before fine-tuning:")
        for (p in PROMPTS) println("  %-26s ->%s".format(p, complete(model, p)))

        // ---- capture one training step ------------------------------------
        // One row per sentence. Input: the sentence without its last token.
        // Target: the next token at each position, with the prompt positions
        // ignored, so the loss is on the answer and its full stop.
        val sentences = TRAIN.map { tokens.encode(it) }
        val len = sentences[0].size - 1
        require(sentences.all { it.size == len + 1 }) { "the training sentences must have one length" }
        val answerStart = tokens.encode(PROMPTS[0]).size
        val inputIds = sentences.flatMap { it.dropLast(1) }.toIntArray()
        val next = IntArray(inputIds.size) { k ->
            val (row, i) = k / len to k % len
            if (i + 1 >= answerStart) sentences[row][i + 1] else -100
        }
        val ids = DTensor<Shape, I32>(HostI32Storage(inputIds), intArrayOf(TRAIN.size, len), I32)
        val targets = DTensor<Shape, I32>(HostI32Storage(next), intArrayOf(TRAIN.size, len), I32)
        t0 = System.nanoTime()
        // The loss is the LM head's cross-entropy against the next-token ids,
        // LOSS_CHUNK tokens of logits at a time. REMAT=1 recomputes each
        // block, and each chunk of logits, during the backward pass instead
        // of keeping them from the forward pass: less memory, more work.
        val captured = captureCausalLm(model, ids, targets, chunk = LOSS_CHUNK, name = "qwen3_step")
        val step = if (REMAT) captured.withRemat() else captured
        println()
        println("training : ${TRAIN.size} sentences of $len tokens, loss on the last ${len - answerStart + 1} of each:")
        for (t in TRAIN) println("             \"${t.trim()}\"")
        println("           step traced and differentiated in %.1f s: %d forward ops, %d gradient ops"
            .format(secondsSince(t0), step.primal.body.size, step.gradient.body.size))

        // AdamW runs inside the compiled step: the parameters and both moments
        // stay on the device and only the loss and four scalars cross to the
        // host. The parameters and moments are donated to each step and
        // updated in place, so the step cannot hand back the parameters it
        // started from. The loop checks the loss first with the forward pass
        // alone (step.primal) and stops before a step once it is below the
        // target, as a host loop does.
        val optimizer = AdamW(learningRate = 1e-5f, weightDecay = 0.01f, decay = { "Norm" !in it && !it.startsWith("embed") })
        val device = step.withDeviceAdamW(optimizer)
        val program = device.function.toStablehlo(outputAliases = device.outputAliases)
        val lossProgram = step.primal.toStablehlo()
        var params: List<PjrtBuffer> = step.parameterTensors(model).map { gpu.bufferFromHostF32(it.hostF32(), it.dims.toList()) }
        var state: List<PjrtBuffer> = device.initialState().zip(device.stateSlots) { zeros, slot -> gpu.bufferFromHostF32(zeros, slot.type.dims) }
        val idBuf = gpu.bufferFromHostI32(inputIds, listOf(TRAIN.size, len))
        val targetBuf = gpu.bufferFromHostI32(next, listOf(TRAIN.size, len))
        t0 = System.nanoTime()
        gpu.prepareStablehlo(program)
        gpu.prepareStablehlo(lossProgram)
        println("           XLA compiled the step, AdamW included, and the loss in %.1f s".format(secondsSince(t0)))
        t0 = System.nanoTime()
        var s = 0
        while (s < MAX_STEPS) {
            val loss = gpu.executeStablehlo(lossProgram, listOf(idBuf, targetBuf) + params).single().use { it.toFloatArray(1)[0] }
            println("           step %2d   loss %.4f".format(s, loss))
            if (loss < TARGET_LOSS) break
            val scalars = device.scalars(s + 1).map { gpu.bufferFromHostF32(floatArrayOf(it), emptyList()) }
            val outs = gpu.executeStablehlo(program, listOf(idBuf, targetBuf) + params + state + scalars)
            // The donated parameters and moments were consumed by the call.
            (params + state + scalars).forEach { it.close() }
            outs[0].close()
            params = outs.subList(1, 1 + params.size)
            state = outs.subList(1 + params.size, outs.size)
            s++
        }
        println("           %d AdamW steps on the device in %.1f s".format(s, secondsSince(t0)))
        (listOf(idBuf, targetBuf)).forEach { it.close() }
        model = model.withParameters(
            step.parameterKeys.withIndex().associate { (j, key) ->
                val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
                key to DTensor<Shape, F32>(HostF32Storage(params[j].toFloatArray(dims.fold(1) { a, d -> a * d })), dims, F32)
            },
        )
        (params + state).forEach { it.close() }

        println()
        println("after fine-tuning:")
        for (p in PROMPTS) println("  %-26s ->%s".format(p, complete(model, p)))
    }

    // ---- write it back out ---------------------------------------------------
    // F32: the weights as AdamW produced them. (Rounded to bf16 they give the
    // same answers here; a smaller learning rate or fewer steps may not.)
    t0 = System.nanoTime()
    HfCausalLm(model, loaded.config).save(source, out, dtype = F32)
    val files = Files.list(out).use { s -> s.map { it.fileName.toString() }.sorted().toList() }
    println()
    println("saved    : ${out.absolutePathString()} in %.1f s".format(secondsSince(t0)))
    println("           ${files.joinToString()}")
    println("           a Hugging Face checkpoint: serve it with")
    println("           CHECKPOINT=${out.absolutePathString()} examples/triton-llm/run.sh --question \"...\"")
}

/** The traced forward at a fixed length: token ids `[1, length]` in, logits `[1, length, vocab]` out. */
private fun captureForward(model: CausalLM, length: Int): DxirFunction {
    val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(length)), intArrayOf(1, length), I32)
    val params = model.parameters
    return captureN(listOf(ids) + params.map { it.tensor }, "qwen3_forward") { leaves ->
        val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
        model.forward(leaves[0], Params { byKey.getValue(it) })
    }
}

private fun findCheckpoint(): Path? {
    System.getenv("CHECKPOINT")?.let { return Path.of(it) }
    val snapshots = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
    if (!Files.isDirectory(snapshots)) return null
    return Files.list(snapshots).use { s -> s.filter { Files.isRegularFile(it.resolve("model.safetensors")) }.findFirst().orElse(null) }
}

private fun secondsSince(t0: Long) = (System.nanoTime() - t0) / 1e9
