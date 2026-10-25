package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.MixtureOfTopKAttention
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.stablehlo.toStablehlo
import io.tlaloc.testing.inline

/**
 * The registered step benchmarks. Each case says how it is run:
 *
 * - [Policy.Measure]: timed by `scripts/bench-registry.sh` (and the device lane), its
 *   median compared with its history; and run once in the ordinary suite, so it keeps
 *   working.
 * - [Policy.SmokeOnly]: run once in the ordinary suite, not tracked, for the reason given.
 * - [Policy.Skip]: not run, for the reason given.
 *
 * A case builds its step from a session; the step returns once its result is on the host.
 */
object BenchRegistry {

    sealed interface Policy {
        data object Measure : Policy
        data class SmokeOnly(val reason: String) : Policy
        data class Skip(val reason: String) : Policy
    }

    class Step(val run: () -> Unit, val close: () -> Unit = {})

    class Case(val name: String, val policy: Policy, val describe: String, val setup: (PjrtSession) -> Step)

    private fun ones(n: Int, scale: Float = 1f) = FloatArray(n) { scale * (((it * 37) % 23) - 11) / 23f }

    private fun scalarSumOf(name: String, inputs: List<DxirType>, body: DxirBuilder.(List<io.tlaloc.ir.DxirNode>) -> io.tlaloc.ir.DxirNode): DxirFunction =
        DxirBuilder.function(name) {
            val ps = inputs.mapIndexed { i, t -> param("a$i", t) }
            listOf(op(OpKind.SUM, listOf(body(ps)), DxirType(F32, emptyList())))
        }

    /** A program over device-resident inputs; the step reads back one scalar. */
    private fun resident(session: PjrtSession, fn: DxirFunction, values: List<FloatArray>): Step {
        val text = fn.toStablehlo()
        val bufs = fn.params.zip(values).map { (p, v) -> session.bufferFromHostF32(v, p.type.dims) }
        return Step(
            run = {
                val outs = session.executeStablehlo(text, bufs)
                outs[0].toFloatArray(1)
                outs.forEach { it.close() }
            },
            close = { bufs.forEach { it.close() } },
        )
    }

    private val lmConfig = CausalLmConfig(
        vocabSize = 64, dModel = 128, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 256, initStd = 0.05f,
    )

    private fun lmBatch(batch: Int, seq: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        val ids = IntArray(batch * seq) { (it * 7 + 3) % lmConfig.vocabSize }
        val next = IntArray(batch * seq) { (it * 5 + 1) % lmConfig.vocabSize }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32) to
            oneHot(next, lmConfig.vocabSize, intArrayOf(batch, seq))
    }

    val cases: List<Case> = listOf(
        Case("matmul_f32_1024", Policy.Measure, "Σ (A·B), A and B 1024×1024 f32 on the device") { s ->
            val t = DxirType(F32, listOf(1024, 1024))
            resident(s, scalarSumOf("mm", listOf(t, t)) { ps -> op(OpKind.MATMUL, ps, t) }, listOf(ones(1024 * 1024), ones(1024 * 1024)))
        },
        Case("log_softmax_cross_entropy_grad_256x32k", Policy.Measure, "the gradient of −Σ y ⊙ logSoftmax(x), x 256×32768") { s ->
            val t = DxirType(F32, listOf(256, 32768))
            val loss = DxirBuilder.function("ce") {
                val x = param("x", t)
                val y = param("y", t)
                val ls = op(OpKind.LOG_SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
                val l = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, ls), t)), DxirType(F32, emptyList()))
                listOf(op(OpKind.NEG, listOf(l), l.type))
            }
            val grad = DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1)
            val fn = DxirBuilder.function("ce_grad_sum") {
                val x = param("x", t)
                val y = param("y", t)
                val g = inline(grad, listOf(x, y)).single()
                listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(g, g), t)), DxirType(F32, emptyList())))
            }
            val y = FloatArray(256 * 32768).also { for (r in 0 until 256) it[r * 32768 + (r * 97) % 32768] = 1f }
            resident(s, fn, listOf(ones(256 * 32768, 4f), y))
        },
        Case("causal_lm_device_adamw_step", Policy.Measure, "a 2-layer CausalLM training step with AdamW in the program, batch 4 × 16") { s ->
            val start = CausalLM.llama(lmConfig, RandomKey.fromSeed(3))
            val (ids, targets) = lmBatch(4, 16)
            val captured = capture(start, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
            val device = captured.withDeviceAdamW(AdamW(learningRate = 1e-3f))
            val text = device.function.toStablehlo(outputAliases = device.outputAliases)
            var params = device.step.parameterTensors(start).map { t -> s.bufferFromHostF32(t.hostF32(), t.dims.toList()) }
            fun zeros(i: Int) = device.step.primal.params[device.step.inputCount + i].type.dims.let { d ->
                s.bufferFromHostF32(FloatArray(d.fold(1) { a, b -> a * b }), d)
            }
            var m = params.indices.map(::zeros)
            var v = params.indices.map(::zeros)
            val idsB = s.bufferFromHostI32((ids.storage as HostI32Storage).data, ids.dims.toList())
            val tgtB = s.bufferFromHostF32(targets.hostF32(), targets.dims.toList())
            var t = 0
            Step(
                run = {
                    t += 1
                    val scalars = device.scalars(t).map { s.bufferFromHostF32(floatArrayOf(it), emptyList()) }
                    val outs = s.executeStablehlo(text, listOf(idsB, tgtB) + params + m + v + scalars)
                    (scalars + params + m + v).forEach { it.close() }
                    val n = params.size
                    outs[0].toFloatArray(1)
                    outs[0].close()
                    params = outs.subList(1, 1 + n)
                    m = outs.subList(1 + n, 1 + 2 * n)
                    v = outs.subList(1 + 2 * n, 1 + 3 * n)
                },
                close = { (listOf(idsB, tgtB) + params + m + v).forEach { it.close() } },
            )
        },
        Case(
            "causal_lm_host_gradient_step",
            Policy.SmokeOnly("parameters and gradients cross the host every step, so the time is the transfers"),
            "the same CausalLM's gradient program through runOn",
        ) { s ->
            val start = CausalLM.llama(lmConfig, RandomKey.fromSeed(3))
            val (ids, targets) = lmBatch(4, 16)
            val captured = capture(start, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
            val idv = (ids.storage as HostI32Storage).data
            val inputs = listOf(FloatArray(idv.size) { idv[it].toFloat() }, targets.hostF32()) + start.parameters.map { it.tensor.hostF32() }
            Step(run = { s.runOn(captured.gradient, inputs) })
        },
        Case("mita_gradient_256", Policy.Measure, "a MixtureOfTopKAttention gradient program, batch 2, 256 tokens, dModel 64, 8 landmarks, top 16") { s ->
            val model = MixtureOfTopKAttention(64, numHeads = 4, landmarks = 8, topK = 16, key = RandomKey.fromSeed(1), initStd = 0.1f)
            val x = DTensor<Shape, F32>(HostF32Storage(ones(2 * 256 * 64)), intArrayOf(2, 256, 64), F32)
            val w = ones(2 * 256 * 64, 0.5f)
            val step = capture(model, listOf(x)) { out: Tracer<Shape> -> (out * out.constant<Shape>(w, out.dims)).sum() }
            val values = listOf(x.hostF32()) + step.parameterTensors(model).map { it.hostF32() }
            resident(s, step.gradient, values)
        },
        Case(
            "llama_8b_decode_attention",
            Policy.Skip("needs the 8B-shaped KV pool (several GiB); PjrtDecodeAttentionBenchTest measures it on demand"),
            "paged decode attention at Llama-3-8B shapes",
        ) { error("skipped") },
    )
}
