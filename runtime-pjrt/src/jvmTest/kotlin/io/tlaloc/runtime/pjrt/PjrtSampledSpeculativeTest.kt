package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeGraphSpec
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Speculative entries that draw their tokens ([io.tlaloc.ir.inference.DecodeModelShape.sampling]):
 * the tiny Qwen4-Exp checkpoint with 3 drafts, a prefill and verify steps that feed the head's
 * drafts. At temperature 0 the tokens are transformers' greedy ids. At temperature 0.8 (top-p
 * 0.9, top-k 20) the device draws what the interpreter draws over the same graph (the kernel
 * tlaloc_sample equals SampleTokensAttrs.reference), and another seed gives another continuation.
 */
class PjrtSampledSpeculativeTest {

    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_qwen4_exp")
    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || TestBackend.isMetal || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtSampledSpeculativeTest"), library)
        return true
    }

    /** One sequence, decoded on [device] or in the interpreter with these sampling parameters. */
    private class Run(
        val s: PjrtSession?, val config: HfDecoderConfig, val weights: List<FloatArray>,
        val temperature: Float, val topP: Float, val topK: Int, val seed: Int,
    ) {
        val k = config.mtpDraftTokens
        val bs = 4
        val ctx = 48
        // Every state is committed once per call: a sequence holds two slots, its base and its target.
        val slots = 2
        val model = config.toDecodeModelShape(numBlocks = 1 + ctx / bs, blockSize = bs, stateSlots = slots, cudaKernels = true, sampling = true)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx))
        val verify: DxirFunction = HfDecoderGraph.build(spec, config)
        var pools: List<FloatArray> = spec.inputs.drop(spec.kvPoolInputBase).take(spec.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val table = IntArray(ctx / bs) { 1 + it }
        // The state commit (DecodeGraphSpec.stateCommitType): base, target, records to apply.
        var commit = intArrayOf(0, 1, 0)
        var position = 0
        var pending = -1
        var drafts = IntArray(0)
        val sampling = intArrayOf(Math.round(temperature * 65536f), Math.round(minOf(topP, 1f) * 16777216f), topK, seed and 0xFFFFFF)

        private fun call(fn: DxirFunction, sp: DecodeGraphSpec, tokens: IntArray, pos: IntArray, slotMap: IntArray, lens: Int): List<IntArray> {
            val ints = listOf(tokens, pos, table, intArrayOf(lens), slotMap, intArrayOf(commit[0]), commit.copyOf(), sampling)
            val lead = 3 + sp.chainRoles.size
            return if (s != null) {
                val out = s.runOnHost(fn, ints + pools + weights)
                pools = out.drop(lead).map { it as FloatArray }
                out.take(3).map { it as IntArray }
            } else {
                val out = DxirInterpreter.evalFunction(fn, ints.map { a -> FloatArray(a.size) { a[it].toFloat() } } + pools + weights)
                pools = out.drop(lead)
                out.take(3).map { a -> IntArray(a.size) { a[it].toInt() } }
            }
        }

        private fun slotOf(p: Int) = table[p / bs] * bs + p % bs

        fun prefill(prompt: IntArray): Int {
            val c = ((prompt.size + bs - 1) / bs) * bs
            val sp = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c)
            val fn = HfDecoderGraph.build(sp, config)
            val pad = c - prompt.size
            val out = call(
                fn, sp, IntArray(c) { if (it < pad) 0 else prompt[it - pad] }, IntArray(c) { if (it < pad) 0 else it - pad },
                IntArray(c) { if (it < pad) -1 else slotOf(it - pad) }, prompt.size,
            )
            position = prompt.size
            commit = intArrayOf(commit[0], commit[1], 0)
            pending = out[0][0]
            drafts = out[2].copyOf()
            return pending
        }

        fun step(): IntArray {
            val t = k + 1
            val tokens = intArrayOf(pending) + drafts
            val pos = IntArray(t) { position + it }
            val out = call(verify, spec, tokens, pos, IntArray(t) { slotOf(pos[it]) }, position + t)
            val a = out[1][0]
            val emitted = out[0].copyOf(a + 1)
            commit = intArrayOf(commit[1], commit[0], a + 1)
            position += a + 1
            pending = emitted.last()
            drafts = out[2].copyOf()
            return emitted
        }

        /** The prompt's continuation: the prefill's token, then each step's, up to [n] tokens. */
        fun decode(prompt: IntArray, n: Int): List<Int> {
            val got = arrayListOf(prefill(prompt))
            while (got.size < n && position + 2 * (k + 1) < ctx) got += step().toList()
            return got.take(n)
        }
    }

    @Test
    fun sampledSpeculativeDecodingDrawsTheInterpretersTokens() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = 3)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val g = ((parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject)["prompts"] as JsonArray).elements[0] as JsonObject
        fun ints(k: String) = (g[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
        val prompt = ints("promptTokens").toIntArray()
        val greedy = ints("generatedTokens")
        val probe = Run(null, config, weights, 0f, 1f, 0, 0)
        assertTrue(probe.verify.body.any { it is io.tlaloc.ir.DxirOp && it.op == OpKind.SAMPLE_TOKENS }, "the verify entry draws its tokens")
        TestBackend.session().use { s ->
            val atZero = Run(s, config, weights, 0f, 1f, 0, 1).decode(prompt, greedy.size)
            println("[sampled-mtp] temperature 0: $atZero (transformers: $greedy)")
            assertEquals(greedy, atZero, "temperature 0 is greedy")
            val n = 16
            val device = Run(s, config, weights, 0.8f, 0.9f, 20, 7).decode(prompt, n)
            val interp = Run(null, config, weights, 0.8f, 0.9f, 20, 7).decode(prompt, n)
            val other = Run(s, config, weights, 0.8f, 0.9f, 20, 8).decode(prompt, n)
            println("[sampled-mtp] temperature 0.8, seed 7: device $device, interpreter $interp; seed 8: $other")
            assertEquals(interp, device, "the device draws the interpreter's tokens")
            assertNotEquals(device, other, "another seed, another continuation")
        }
    }
}
