package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.passes.DxirInterpreter
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
 * §0.4.479 — Phase H3c-2: **the real Llama, through Tlaloc's own IR, against
 * HuggingFace transformers.**
 *
 * §0.4.478 put a real TinyLlama-1.1B checkpoint on disk and proved this repo
 * reads its bytes exactly and knows its layout. What it explicitly did NOT do
 * is compute anything with them. This lane does: it builds
 * [HfDecoderGraph] over the real config, stages the real weights through
 * [HfStagedWeights] (transposing the Linears — the §0.4.478 layout fact,
 * now load-bearing), runs a prefill as a sequence of single-token decode steps
 * threading the KV pools, and compares the LOGITS against
 * `transformers.AutoModelForCausalLM` in the vLLM venv.
 *
 * ## What exactly is certified
 *
 * A **REDUCED slice: the first two layers of the real checkpoint.** Not a
 * random 2-layer model — TinyLlama's real embedding table, its real layers 0
 * and 1, its real final norm and its real (untied) head. Both sides are
 * reduced by the same arithmetic: Tlaloc over `config.copy(numLayers = 2)`,
 * transformers over `cfg.num_hidden_layers = 2`. The reason is cost, stated
 * rather than hidden: the reference interpreter is a straightforward
 * triple-loop evaluator and a 22-layer 1.1-billion-parameter step through it
 * is minutes, at a JVM heap that would dominate the whole suite. A 2-layer
 * slice exercises **every op in the decode graph** — embedding, RMSNorm ×5,
 * seven projections ×2, RoPE, `KV_CACHE_WRITE` ×4, `PAGED_ATTENTION` ×2,
 * SwiGLU, the head — and the layers it drops are twenty more instances of
 * shapes already covered.
 *
 * ## The floor, and why it is what it is
 *
 * Both sides compute in **float32 from bf16-exact inputs** (`bf16 -> f32` is a
 * 16-bit shift and loses nothing, §0.4.468), and the oracle runs on **CPU**,
 * so TF32 never enters. What remains between the two is ACCUMULATION ORDER:
 * torch's blocked GEMM against the interpreter's row-major triple loop, over a
 * 2048-wide contraction, twice per layer plus a 2048-wide head. The logits
 * here run to roughly ±10, so the honest claim is a RELATIVE one against the
 * logit magnitude: [LOGIT_REL_FLOOR]. Tighter would be pinning noise; looser
 * would not catch a wrong RoPE.
 *
 * **The check that actually matters for serving is the discrete one**, and it
 * is pinned exactly: the argmax of every position must agree, and so must the
 * whole top-5 ORDER. A numeric floor tells you the arithmetic is close; token
 * agreement tells you the server would emit the same text.
 *
 * ## Lanes and skips
 *
 * Self-skips, loudly, when either half of the apparatus is absent — the
 * checkpoint (fetch command in [HfLlamaCheckpointTest]) or the vLLM venv's
 * python. Neither is something a fresh clone has, and a test that silently
 * passed without them would be worse than no test. On THIS machine both are
 * present and this lane runs.
 *
 * ## What is NOT here, by name
 *
 * - **The PJRT lane.** [io.tlaloc.runtime.pjrt.PjrtSession]'s `runOn` is
 *   single-dtype (F32), and this graph's first five operands are I32; the
 *   mixed-dtype staging that the serving path uses lives in the PYTHON loader,
 *   which reads a `ServingArtifactWriter` artifact — and that writer does not
 *   yet carry a weight table (H3c-3). So the device lane arrives with the
 *   artifact, not before it, and this slice does not claim it.
 * - **A prefill in one call.** Run here as N decode steps, because
 *   `PAGED_ATTENTION`'s ragged form is H1a's still-open deferral. Same
 *   arithmetic, N launches.
 */
class HfLlamaRealDecodeParityTest {

    /** Relative floor on the logits — see the class doc for its derivation. */
    private val LOGIT_REL_FLOOR = 1e-5

    /** The prompt, as token ids. `1` is TinyLlama's BOS. */
    private val tokens = intArrayOf(1, 15043, 3186, 29892, 590)

    private val reducedLayers = 2
    private val blockSize = 8
    private val numBlocks = 4

    private fun checkpointDir(): Path? {
        val env = System.getenv("TLALOC_HF_LLAMA_CHECKPOINT")
        val p = if (env != null) Path.of(env) else Path.of(
            System.getProperty("user.home"),
            ".cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0",
        )
        return if (Files.isDirectory(p)) p else null
    }

    private fun vllmPython(): Path? {
        val env = System.getenv("TLALOC_VLLM_PYTHON")
        val p = if (env != null) Path.of(env) else Path.of(
            System.getProperty("user.home"), ".local/venvs/vllm/bin/python",
        )
        return if (Files.isExecutable(p)) p else null
    }

    private fun repoRoot(): Path {
        var d: Path? = Path.of("").toAbsolutePath()
        while (d != null && !Files.isRegularFile(d.resolve("settings.gradle.kts"))) d = d.parent
        return d ?: Path.of("").toAbsolutePath()
    }

    // ------------------------------------------------------------ the oracle

    private class Oracle(
        val logits: List<DoubleArray>,
        val argmax: List<Int>,
        val topk: List<IntArray>,
        val layers: Int,
    )

    private fun runOracle(ckpt: Path, python: Path, tokens: IntArray, layers: Int): Oracle {
        val script = repoRoot().resolve("harness/python/hf_llama_reference.py")
        assertTrue(Files.isRegularFile(script), "oracle script missing at $script")
        val cmd = buildList {
            add(python.toString())
            add(script.toString())
            add("--checkpoint"); add(ckpt.toString())
            add("--layers"); add(layers.toString())
            add("--tokens"); tokens.forEach { add(it.toString()) }
        }
        val proc = ProcessBuilder(cmd)
            .redirectErrorStream(false)
            .redirectError(ProcessBuilder.Redirect.DISCARD)
            .start()
        val stdout = proc.inputStream.bufferedReader().readText()
        assertTrue(proc.waitFor(600, TimeUnit.SECONDS), "hf_llama_reference.py did not finish")
        assertEquals(0, proc.exitValue(), "hf_llama_reference.py failed; stdout=${stdout.take(400)}")

        val root = parseJson(stdout) as JsonObject
        val cfg = root["config"] as JsonObject
        val logits = (root["logits"] as JsonArray).elements.map { row ->
            (row as JsonArray).elements.map { (it as JsonNumber).value }.toDoubleArray()
        }
        val argmax = (root["argmax"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
        val topk = (root["topk_indices"] as JsonArray).elements.map { row ->
            (row as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
        }
        return Oracle(
            logits, argmax, topk,
            (cfg["num_hidden_layers"] as JsonNumber).value.toInt(),
        )
    }

    // -------------------------------------------------------------- the test

    @Test
    fun realLlamaLogitsAgreeWithHuggingFaceAcrossPrefillAndDecodeSteps() {
        val dir = checkpointDir()
        assumeTrue(
            dir != null,
            "no HF Llama checkpoint — see HfLlamaCheckpointTest for the fetch command",
        )
        val python = vllmPython()
        assumeTrue(python != null, "no vLLM-venv python (~/.local/venvs/vllm/bin/python) — skipping")

        HfCheckpoint.open(dir!!).use { ckpt ->
            val config = ckpt.config.copy(numLayers = reducedLayers)
            val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
            val bucket = DecodeBucket(batch = 1, maxContext = numBlocks * blockSize)
            val spec = HfDecoderGraph.spec(config, model, bucket)
            val fn = HfDecoderGraph.build(spec, config)

            // The staged weights ARE the checkpoint's, transposed where the
            // HF convention says they are stored transposed.
            val weights = HfStagedWeights.stage(ckpt, config)
            assertEquals(spec.weightSlots.size, weights.size, "staged weight count")
            assertEquals(
                1 + reducedLayers * 9 + 2,
                weights.size,
                "embed + $reducedLayers x ${9} + finalNorm + head",
            )

            val oracle = runOracle(dir, python!!, tokens, reducedLayers)
            assertEquals(reducedLayers, oracle.layers, "oracle was reduced to the same depth")
            assertEquals(tokens.size, oracle.logits.size, "oracle logits per position")

            val poolSize = numBlocks * blockSize * model.numKvHeads * model.headDim
            var pools = MutableList(2 * reducedLayers) { FloatArray(poolSize) }
            val blockTables = FloatArray(spec.maxBlocksPerSeq) { it.toFloat() }

            var worstRel = 0.0
            for (step in tokens.indices) {
                val inputs = buildList {
                    add(floatArrayOf(tokens[step].toFloat()))       // tokenIds [1,1]
                    add(floatArrayOf(step.toFloat()))               // positions [1,1]
                    add(blockTables)                                // blockTables
                    add(floatArrayOf((step + 1).toFloat()))         // seqLens
                    add(floatArrayOf(step.toFloat()))               // slotMapping (flat slot)
                    addAll(pools)
                    addAll(weights)
                }
                val out = DxirInterpreter.evalFunction(fn, inputs)
                val logits = out[0]
                pools = MutableList(2 * reducedLayers) { out[1 + it] }

                val want = oracle.logits[step]
                assertEquals(want.size, logits.size, "vocab width at step $step")

                var ourArgmax = 0
                for (v in logits.indices) if (logits[v] > logits[ourArgmax]) ourArgmax = v
                assertEquals(
                    oracle.argmax[step], ourArgmax,
                    "ARGMAX disagreement at position $step — the server would emit a different " +
                        "token. ours=$ourArgmax (${logits[ourArgmax]}) " +
                        "hf=${oracle.argmax[step]} (${want[oracle.argmax[step]]})",
                )

                val ourTop5 = logits.indices.sortedByDescending { logits[it] }.take(5)
                assertEquals(
                    oracle.topk[step].toList(), ourTop5,
                    "top-5 ORDER disagreement at position $step",
                )

                val denom = max(1.0, want.maxOf { abs(it) })
                for (v in want.indices) {
                    val rel = abs(want[v] - logits[v]) / denom
                    if (rel > worstRel) worstRel = rel
                    assertTrue(
                        rel <= LOGIT_REL_FLOOR,
                        "logit[$v] at position $step: hf=${want[v]} tlaloc=${logits[v]} " +
                            "rel=$rel > $LOGIT_REL_FLOOR",
                    )
                }
            }
            // The floor is stated, not discovered: if the real worst case were
            // at the floor this assertion would be the one that fails first.
            assertTrue(
                worstRel < LOGIT_REL_FLOOR,
                "worst relative logit error $worstRel is AT the floor $LOGIT_REL_FLOOR — " +
                    "the floor has stopped being a bound and become the measurement",
            )
            println(
                "H3c-2 parity: ${tokens.size} steps x $reducedLayers real layers, " +
                    "worst relative logit error $worstRel (floor $LOGIT_REL_FLOOR), " +
                    "argmax and top-5 exact",
            )
        }
    }

    /**
     * The SENSITIVITY twin of the lane above, and the reason it is worth
     * believing: a checkpoint whose Linears are staged WITHOUT the transpose
     * must not merely be less accurate — it must be a different model.
     *
     * Runs one step of the real graph with `qProj`/`kProj`/`vProj` staged in
     * the file's `[out, in]` layout (legal only because q is square on this
     * model and k/v are re-shaped from the same element count) and asserts the
     * logits MOVE. Cheap: no oracle subprocess, one decode step.
     */
    @Test
    fun stagingWithoutTheTransposeChangesTheAnswer() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no HF Llama checkpoint — skipping")

        HfCheckpoint.open(dir!!).use { ckpt ->
            val config = ckpt.config.copy(numLayers = 1)
            val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
            val bucket = DecodeBucket(batch = 1, maxContext = numBlocks * blockSize)
            val spec = HfDecoderGraph.spec(config, model, bucket)
            val fn = HfDecoderGraph.build(spec, config)
            val weights = HfStagedWeights.stage(ckpt, config)

            val poolSize = numBlocks * blockSize * model.numKvHeads * model.headDim
            val blockTables = FloatArray(spec.maxBlocksPerSeq) { it.toFloat() }

            // THREE steps, not one, and this is the finding that made the
            // twin worth writing: at `seqLens = 1` the softmax has a single
            // term and is exactly 1.0, so attention returns the just-written V
            // and **Q does not enter the answer at all**. A one-step
            // sensitivity check therefore measures nothing about q_proj — it
            // reported a movement of exactly 0.0 — and the first version of
            // this test said so before it was fixed. Q becomes load-bearing
            // the moment there are two KV positions to weigh against each
            // other, which is the third step's `seqLens = 3`.
            fun runTo(w: List<FloatArray>): FloatArray {
                var pools = MutableList(2) { FloatArray(poolSize) }
                var logits = FloatArray(0)
                for (s in 0 until 3) {
                    val out = DxirInterpreter.evalFunction(
                        fn,
                        buildList {
                            add(floatArrayOf(tokens[s].toFloat()))
                            add(floatArrayOf(s.toFloat()))
                            add(blockTables)
                            add(floatArrayOf((s + 1).toFloat()))
                            add(floatArrayOf(s.toFloat()))
                            addAll(pools)
                            addAll(w)
                        },
                    )
                    logits = out[0]
                    pools = MutableList(2) { out[1 + it] }
                }
                return logits
            }

            val good = runTo(weights)
            // qProj is slot 2 (embed, inputNorm, qProj, ...) and is square on
            // this model, so un-transposing it is a SHAPE-LEGAL lie — exactly
            // the kind a square-only certification could never catch.
            val qIdx = 2
            val d = config.hiddenSize
            val bad = weights.toMutableList()
            bad[qIdx] = HfStagedWeights.transpose(weights[qIdx], d, d)
            val wrong = runTo(bad)

            var moved = 0.0
            for (v in good.indices) moved = max(moved, abs(good[v] - wrong[v]).toDouble())
            assertTrue(
                moved > 1e-2,
                "un-transposing q_proj moved the logits by only $moved — the transpose in " +
                    "HfStagedWeights is then not load-bearing, which cannot be true",
            )
        }
    }
}
