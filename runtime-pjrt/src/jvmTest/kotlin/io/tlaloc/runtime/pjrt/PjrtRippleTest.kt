package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.captureCausalLm
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tiny Ripple checkpoint and its DFlash2 draft (the `tiny_ripple` fixtures) on the device,
 * with XLA's own attention and matmuls (no CUDA custom calls, so the same programs compile for a
 * TPU): a prefill, three verify steps of the target with the draft, and a training step of the
 * [io.tlaloc.nn.CausalLM] read from the checkpoint, each against the interpreter.
 *
 * With the checkpoint's FFN clamp of 0.75 many units sit at the bound, and the device's TF32 dots
 * move some across it, which changes the result far past rounding; the comparison here uses the
 * released models' threshold, 20, which these activations do not reach. The interpreter tests
 * check the clamp itself.
 *
 * The tokens, accepted counts and drafts must be the interpreter's exactly. The pools and the
 * gradients are compared within [TOLERANCE] of each output's scale: this five-layer model with
 * large random weights moves its pools by 0.1 to 0.2% of scale when its weights alone move by
 * 2^-11 (the step of TF32, which the GB10 runs f32 dots at), and the device's rounding of every
 * dot's operands moves them by up to about 1%.
 */
class PjrtRippleTest {
    private val fixtures = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference")

    private fun ints(v: FloatArray) = v.map { it.toInt() }

    /** Each output against the interpreter's, relative to the output's scale (XLA's dots run at TF32). */
    private fun close(what: String, want: List<FloatArray>, got: List<FloatArray>, tolerance: Float = TOLERANCE): Float {
        assertEquals(want.size, got.size, what)
        var worst = 0f
        for (k in want.indices) {
            val scale = want[k].maxOfOrNull { abs(it) }?.coerceAtLeast(1e-6f) ?: continue
            var w = 0f
            var at = 0
            for (i in want[k].indices) { val d = abs(got[k][i] - want[k][i]) / scale; if (d > w) { w = d; at = i } }
            if (w > 1e-3f) println("[pjrt-ripple] $what output $k (${want[k].size} values, scale $scale): worst ${w} at $at: ${got[k][at]} vs ${want[k][at]}")
            worst = maxOf(worst, w)
        }
        assertTrue(worst < tolerance, "$what on ${TestBackend.target}: outputs differ from the interpreter by $worst of scale")
        return worst
    }

    @Test
    fun theTargetAndItsDraftRunAsTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val k = 3
        val (config, staged) = HfCheckpoint.open(fixtures.resolve("tiny_ripple"), draftDir = fixtures.resolve("tiny_ripple_dflash2")).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = k, ffnClampThreshold = UNCLAMPED)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val prefill: DxirFunction = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk), config)
        val verifySpec = HfDecoderGraph.spec(config, model, bucket)
        val verify: DxirFunction = HfDecoderGraph.build(verifySpec, config)
        val lead = verifySpec.leadingOutputs
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val prompt = listOf(3, 4, 50, 11, 12, 13)
        var worst = 0f
        TestBackend.session().use { s ->
            var pools: List<FloatArray> = verifySpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
            val pad = chunk - prompt.size
            val inputs = buildList {
                add(FloatArray(chunk) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                add(FloatArray(chunk) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(prompt.size.toFloat()))
                add(FloatArray(chunk) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                addAll(pools)
                addAll(staged)
            }
            val want = DxirInterpreter.evalFunction(prefill, inputs)
            val got = s.runOn(prefill, inputs)
            assertEquals(ints(want[0]), ints(got[0]), "the prefill's token")
            assertEquals(ints(want[2]), ints(got[2]), "the prefill's drafts")
            worst = maxOf(worst, close("prefill", want.drop(lead), got.drop(lead)))
            pools = want.drop(lead)
            var n = prompt.size
            var tokens = listOf(ints(want[0])[0]) + ints(want[2])
            repeat(3) { step ->
                val stepInputs = buildList {
                    add(FloatArray(k + 1) { tokens[it].toFloat() })
                    add(FloatArray(k + 1) { (n + it).toFloat() })
                    add(tables)
                    add(floatArrayOf((n + k + 1).toFloat()))
                    add(FloatArray(k + 1) { (bs + n + it).toFloat() })
                    addAll(pools)
                    addAll(staged)
                }
                val w = DxirInterpreter.evalFunction(verify, stepInputs)
                val g = s.runOn(verify, stepInputs)
                for (o in 0 until lead) assertEquals(ints(w[o]), ints(g[o]), "verify step $step, output $o")
                worst = maxOf(worst, close("verify step $step", w.drop(lead), g.drop(lead)))
                val accepted = ints(w[1])[0]
                n += accepted + 1
                tokens = listOf(ints(w[0])[accepted]) + ints(w[2])
                pools = w.drop(lead)
            }
        }
        println("[pjrt-ripple] prefill and three verify steps: ids == the interpreter, worst pool difference $worst of scale")
    }

    /**
     * A scoring entry (`scoreTopK`) over two chunks of 8 tokens, the second continuing from the
     * first's pools: every row's top-5 next tokens and log-probabilities equal what the plain
     * prefill entry gives for the last row of each prefix (log-softmax of its logits, through
     * the interpreter), and the device equals the interpreter.
     */
    @Test
    fun aScoringEntryGivesEveryRowsTopK() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val topK = 5
        val (config, staged) = HfCheckpoint.open(fixtures.resolve("tiny_ripple")).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, ffnClampThreshold = UNCLAMPED)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val scoreSpec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk, scoreTopK = topK)
        val score = HfDecoderGraph.build(scoreSpec, config)
        val refChunk = 2 * chunk
        val refSpec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = refChunk)
        val ref = HfDecoderGraph.build(refSpec, config)
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val tokens = listOf(3, 4, 50, 11, 12, 13, 7, 9, 21, 22, 5, 60, 8, 2, 30, 31)
        fun fresh() = scoreSpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        fun scoreInputs(start: Int, pools: List<FloatArray>) = buildList {
            add(FloatArray(chunk) { tokens[start + it].toFloat() })
            add(FloatArray(chunk) { (start + it).toFloat() })
            add(tables)
            add(floatArrayOf((start + chunk).toFloat()))
            add(FloatArray(chunk) { (bs + start + it).toFloat() })
            addAll(pools)
            addAll(staged)
        }
        // Interpreter: the two chunks.
        val c1 = DxirInterpreter.evalFunction(score, scoreInputs(0, fresh()))
        val c2 = DxirInterpreter.evalFunction(score, scoreInputs(chunk, c1.drop(2)))
        val ids = ints(c1[0]) + ints(c2[0])
        val lps = (c1[1].toList() + c2[1].toList())
        var worstRef = 0f
        for (p in 1..tokens.size) {
            val pad = refChunk - p
            val inputs = buildList {
                add(FloatArray(refChunk) { if (it < pad) 0f else tokens[it - pad].toFloat() })
                add(FloatArray(refChunk) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(p.toFloat()))
                add(FloatArray(refChunk) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                addAll(fresh())
                addAll(staged)
            }
            val logits = DxirInterpreter.evalFunction(ref, inputs)[0]
            val max = logits.max()
            val lse = max + kotlin.math.ln(logits.sumOf { kotlin.math.exp((it - max).toDouble()) }).toFloat()
            val want = logits.indices.sortedByDescending { logits[it] }.take(topK)
            val row = p - 1
            assertEquals(want, ids.subList(row * topK, row * topK + topK), "row $row's top-$topK ids")
            for (j in 0 until topK) worstRef = maxOf(worstRef, abs(lps[row * topK + j] - (logits[want[j]] - lse)))
        }
        assertTrue(worstRef < 1e-4f, "scoring log-probabilities differ from the prefill's log-softmax by $worstRef")
        // Device against the interpreter.
        TestBackend.session().use { s ->
            val g1 = s.runOn(score, scoreInputs(0, fresh()))
            val g2 = s.runOn(score, scoreInputs(chunk, g1.drop(2)))
            assertEquals(ids, ints(g1[0]) + ints(g2[0]), "the device's top-$topK ids")
            val worst = (g1[1].toList() + g2[1].toList()).zip(lps).maxOf { (a, b) -> abs(a - b) }
            println("[pjrt-ripple] scoring: 16 rows' top-$topK equal the prefill's (log-prob difference $worstRef); device ids equal, log-probs within $worst")
            assertTrue(worst < 5e-2f, "device log-probabilities differ by $worst")
        }
    }

    /**
     * [io.tlaloc.runtime.pjrt.serving.TeacherScorer] on tiny_ripple: 19 tokens in chunks of 8
     * (the last right-aligned behind 5 padding rows), every row's top-5 against the plain
     * prefill's log-softmax of each prefix through the interpreter.
     */
    @Test
    fun theTeacherScorerScoresEveryPosition() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val topK = 5
        val bs = 4
        val context = 32
        val tokens = intArrayOf(3, 4, 50, 11, 12, 13, 7, 9, 21, 22, 5, 60, 8, 2, 30, 31, 40, 41, 6)
        val got = io.tlaloc.runtime.pjrt.serving.TeacherScorer.open(
            fixtures.resolve("tiny_ripple"), topK, chunk = 8, maxContext = context, blockSize = bs, target = TestBackend.target,
        ) { it.copy(weightDType = F32, ffnClampThreshold = UNCLAMPED) }.use { it.score(tokens) }
        val (config, staged) = HfCheckpoint.open(fixtures.resolve("tiny_ripple")).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, ffnClampThreshold = UNCLAMPED)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val refSpec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL, prefillChunk = 24)
        val ref = HfDecoderGraph.build(refSpec, config)
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val pools = refSpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        var worst = 0f
        for (p in 1..tokens.size) {
            val pad = 24 - p
            val logits = DxirInterpreter.evalFunction(ref, buildList {
                add(FloatArray(24) { if (it < pad) 0f else tokens[it - pad].toFloat() })
                add(FloatArray(24) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(p.toFloat()))
                add(FloatArray(24) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                addAll(pools)
                addAll(staged)
            })[0]
            val max = logits.max()
            val lse = max + kotlin.math.ln(logits.sumOf { kotlin.math.exp((it - max).toDouble()) }).toFloat()
            val want = logits.indices.sortedByDescending { logits[it] }.take(topK)
            val row = p - 1
            assertEquals(want, got.first.toList().subList(row * topK, row * topK + topK), "row $row's top-$topK ids")
            for (j in 0 until topK) worst = maxOf(worst, abs(got.second[row * topK + j] - (logits[want[j]] - lse)))
        }
        println("[pjrt-ripple] TeacherScorer: 19 positions in 3 chunks, top-$topK ids equal the prefill's; log-probs within $worst")
        assertTrue(worst < 5e-2f, "scorer log-probabilities differ by $worst")
    }

    @Test
    fun aTrainingStepRunsAsTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val base = HfCausalLm.load(fixtures.resolve("tiny_ripple"))
        val lm = HfCausalLm(
            base.model.withLayers(
                base.model.blocks.map { b ->
                    val mlp = b.mlp as io.tlaloc.nn.SwiGLU
                    b.withSublayers(b.attn, io.tlaloc.nn.SwiGLU(mlp.gate, mlp.up, mlp.down, UNCLAMPED.toFloat()))
                },
                base.model.head,
            ),
            base.config,
        )
        val ids = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41)
        val next = IntArray(ids.size) { if (it + 1 < ids.size) ids[it + 1] else -1 }
        fun tensor(a: IntArray) = DTensor<Shape, I32>(HostI32Storage(a), intArrayOf(1, a.size), I32)
        val step = captureCausalLm(lm.model, tensor(ids), tensor(next), chunk = 4)
        val values = listOf(FloatArray(ids.size) { ids[it].toFloat() }, FloatArray(next.size) { next[it].toFloat() }) +
            step.parameterTensors(lm.model).map { (it.storage as io.tlaloc.core.HostF32Storage).data }
        TestBackend.session().use { s ->
            val want = DxirInterpreter.evalFunction(step.gradient, values)
            val got = s.runOn(step.gradient, values)
            assertTrue(abs(got[0][0] - want[0][0]) <= 1e-3f * abs(want[0][0]), "the loss: ${got[0][0]} against ${want[0][0]}")
            val worst = close("the training step", want, got, 2 * TOLERANCE)
            println("[pjrt-ripple] training step: loss ${got[0][0]} (interpreter ${want[0][0]}), worst gradient difference $worst of scale")
        }
    }

    private companion object {
        const val UNCLAMPED = 20.0
        const val TOLERANCE = 0.025f
    }
}
