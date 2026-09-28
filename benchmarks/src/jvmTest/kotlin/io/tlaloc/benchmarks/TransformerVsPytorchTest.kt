package io.tlaloc.benchmarks

import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.core.split
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.step
import io.tlaloc.runtime.iree.NpyWriter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `:nn`'s Llama-architecture [CausalLM] against the same model written in
 * plain PyTorch (`harness/python/run_pytorch_transformer_grad.py`), on the
 * same weights, token ids and targets: the loss, every parameter's gradient,
 * and 20 AdamW steps.
 *
 * Two models, each with two layers, RoPE, SwiGLU and RMSNorm, and one
 * target position set to the ignore index: a Llama with grouped-query
 * attention (4 query heads over 2 key/value heads) and an untied head, and a
 * Qwen3-style model with per-head q/k RMSNorm and tied embeddings.
 *
 * Self-skips when no python with torch is found (TLALOC_TORCH_PYTHON, or
 * ~/.local/venvs/iree).
 */
class TransformerVsPytorchTest {

    private val llama = CausalLmConfig(
        vocabSize = 13, dModel = 16, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 24,
        initStd = 0.2f,
    )
    private val qwen3 = CausalLmConfig(
        vocabSize = 13, dModel = 16, numLayers = 2, numHeads = 2, ffHidden = 24, headDim = 4,
        qkNorm = true, tiedEmbeddings = true, initStd = 0.2f,
    )
    private val batch = 2
    private val seq = 6
    private val steps = 20
    private val lr = 0.01f
    private val weightDecay = 0.01f

    private fun python(): String? {
        val candidates = listOfNotNull(
            System.getenv("TLALOC_TORCH_PYTHON"),
            System.getProperty("user.home")?.let { "$it/.local/venvs/iree/bin/python" },
        )
        val p = candidates.firstOrNull { Files.isExecutable(Path.of(it)) } ?: return null
        val probe = ProcessBuilder(p, "-c", "import torch").redirectErrorStream(true).start()
        return if (probe.waitFor(30, TimeUnit.SECONDS) && probe.exitValue() == 0) p else null
    }

    @Test
    fun llamaWithGroupedQueryAttentionMatchesPytorch() = check(llama, "llama")

    /** Per-head q/k RMSNorm, tied embeddings, and heads narrower than dModel / numHeads. */
    @Test
    fun qwen3StyleQkNormAndTiedEmbeddingsMatchPytorch() = check(qwen3, "qwen3")

    private fun check(config: CausalLmConfig, label: String) {
        val py = python()
        assumeTrue(py != null, "no python with torch (set TLALOC_TORCH_PYTHON or install torch into ~/.local/venvs/iree)")
        val script = Path.of("..", "harness", "python", "run_pytorch_transformer_grad.py").toAbsolutePath().normalize()
        assumeTrue(Files.exists(script), "reference script not found at $script")

        val keys = RandomKey.fromSeed(42).split(2)
        var model = CausalLM.llama(config, keys[0])
        val ids = IntArray(batch * seq) { (it * 5 + 3) % config.vocabSize }
        val next = IntArray(batch * seq) { (it * 7 + 1) % config.vocabSize }.also { it[4] = -100 }
        val idsTensor = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32)
        val targets = oneHot(next, config.vocabSize, intArrayOf(batch, seq), ignoreIndex = -100)

        // ---- PyTorch ----
        val dir = Files.createTempDirectory("tlaloc-transformer-vs-pytorch-")
        val paramKeys = model.parameters.map { it.key }
        for (p in model.parameters) {
            NpyWriter.writeFloat32(dir.resolve("${p.key}.npy"), p.tensor.hostF32(), p.tensor.dims.toList())
        }
        NpyWriter.writeFloat32(dir.resolve("ids.npy"), FloatArray(ids.size) { ids[it].toFloat() }, listOf(batch, seq))
        NpyWriter.writeFloat32(dir.resolve("targets.npy"), FloatArray(next.size) { next[it].toFloat() }, listOf(batch, seq))
        Files.writeString(
            dir.resolve("config.json"),
            """{"keys": [${paramKeys.joinToString { "\"$it\"" }}], "numLayers": ${config.numLayers},
               "numHeads": ${config.numHeads}, "numKvHeads": ${config.numKvHeads}, "headDim": ${config.headDim},
               "normEps": ${config.normEps}, "ropeTheta": ${config.ropeTheta}, "tiedEmbeddings": ${config.tiedEmbeddings},
               "lr": $lr, "weightDecay": $weightDecay, "steps": $steps}""",
        )
        val out = dir.resolve("out.json")
        val proc = ProcessBuilder(py, script.toString(), "--inputs-dir", dir.toString(), "--output", out.toString())
            .redirectErrorStream(true).start()
        val log = proc.inputStream.bufferedReader().readText()
        check(proc.waitFor(300, TimeUnit.SECONDS) && proc.exitValue() == 0) { "PyTorch reference failed:\n$log" }
        val torch = parseJson(Files.readString(out)) as JsonObject
        dir.toFile().deleteRecursively()

        // ---- Tlaloc ----
        val step = capture(model, listOf(idsTensor), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
        val first = step.run(model, listOf(idsTensor, targets))

        fun rel(a: Float, b: Float) = abs(a - b) / max(1e-6f, max(abs(a), abs(b)))
        val torchLoss = (torch.fields["loss"] as JsonNumber).value.toFloat()
        println("[transformer-vs-pytorch:$label] loss tlaloc=${first.loss} torch=$torchLoss")
        assertTrue(rel(first.loss, torchLoss) < 1e-5f, "loss tlaloc=${first.loss} torch=$torchLoss")

        val torchGrads = torch.obj("grads")
        var worst = 0f
        var worstKey = ""
        for (key in paramKeys) {
            val ours = first.gradients.getValue(key).hostF32()
            val theirs = (torchGrads.fields.getValue(key) as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }
            assertEquals(theirs.size, ours.size, "gradient size for $key")
            val scale = theirs.maxOf { abs(it) }.coerceAtLeast(1e-6f)
            for (i in ours.indices) {
                val d = abs(ours[i] - theirs[i]) / scale
                if (d > worst) { worst = d; worstKey = "$key[$i]" }
            }
        }
        println("[transformer-vs-pytorch:$label] ${paramKeys.size} gradients, worst |diff|/max|grad| = $worst at $worstKey")
        assertTrue(worst < 1e-4f, "gradient mismatch: $worst at $worstKey")

        val optimizer = AdamW(learningRate = lr, weightDecay = weightDecay)
        var state = optimizer.initialState()
        val ours = ArrayList<Float>()
        repeat(steps) {
            val r = step.run(model, listOf(idsTensor, targets))
            ours += r.loss
            val (m, s) = optimizer.step(model, r.gradients, state)
            model = m
            state = s
        }
        val oursFinal = step.run(model, listOf(idsTensor, targets)).loss
        val theirs = torch.arr("losses").elements.map { (it as JsonNumber).value.toFloat() }
        val theirFinal = (torch.fields["final_loss"] as JsonNumber).value.toFloat()
        val curve = ours.indices.maxOf { rel(ours[it], theirs[it]) }
        println("[transformer-vs-pytorch:$label] AdamW loss ${ours.first()} -> $oursFinal (torch $theirFinal), curve max rel diff $curve")
        assertTrue(curve < 1e-3f, "AdamW loss curves diverge: max rel diff $curve\n tlaloc=$ours\n torch=$theirs")
        assertTrue(rel(oursFinal, theirFinal) < 1e-3f, "final loss tlaloc=$oursFinal torch=$theirFinal")
        assertTrue(oursFinal < ours.first() * 0.8f, "20 AdamW steps did not reduce the loss: ${ours.first()} -> $oursFinal")
    }
}
