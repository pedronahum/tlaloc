package io.tlaloc.benchmarks

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.Adam
import io.tlaloc.nn.Dense
import io.tlaloc.nn.ReluLayer
import io.tlaloc.nn.Sequential
import io.tlaloc.nn.capture
import io.tlaloc.nn.step
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.444 — Phase F8 (2): THE COMPILED GPU TRAINING STEP, the amended
 * decision-3 architecture's payoff. The F1-captured gradient `DxirFunction`
 * for the MLP — the SAME object the host lane interprets — goes through
 * `toStablehlo` → PJRT compile → GB10 execution via [PjrtSession.runOn]
 * (which caches the executable by MLIR text, the §0.4.307 amortization the
 * F1 caching contract named), and:
 *
 *  1. one step's GPU outputs (loss + input gradient + all 4 parameter
 *     gradients) are pinned elementwise against `DxirInterpreter` on the
 *     identical values, at 1e-4 (observed 3.8e-5 — f32 reduction-order noise
 *     between XLA's fused matmuls and the interpreter's serial accumulators;
 *     the §0.4.292 LlamaDecoder backward cert pinned the same backend pair at
 *     the same tolerance) — the two backends share nothing but the
 *     graph, so agreement here IS the route certification;
 *  2. five full training steps run with gradients entirely on the GPU lane
 *     (host-side F3 Adam between them — the ratified §2.9 scope: the graph
 *     is on-device, the optimizer is host v1), against the interpreter-lane
 *     twin: per-step losses within 1e-4, final parameters within 1e-3, and
 *     the session's compile cache pinned at ONE executable across all five
 *     dispatches (params are `DxirParam`s, never baked constants — the same
 *     compiled artifact serves every step).
 *
 * No emission gap was hit: every op in the captured MLP gradient graph
 * (MATMUL, BROADCAST, ADD, RELU, STEP, MUL, SUB, MEAN, SUM/SUM_TO,
 * TRANSPOSE, ZEROS_LIKE …) has an emitter arm, so the slice's "record which
 * op lacks emission" contingency was not needed.
 *
 * Self-skips (the PjrtRngSmokeTest ladder) when no PJRT plugin or no CUDA
 * device resolves.
 */
class NnMlpGpuTrainingTest {

    private val n = 16
    private val d = 4

    private fun task(): Pair<FloatArray, FloatArray> {
        val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        val xv = FloatArray(u.size) { i -> 2f * u[i] - 1f }
        val yv = FloatArray(n) { i ->
            val r = i * d
            xv[r] * xv[r + 1] + 0.5f * xv[r + 2] - 0.25f * xv[r + 3]
        }
        return xv to yv
    }

    private fun mlp(): Sequential = Sequential(
        Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
        ReluLayer,
        Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
    )

    @Test
    fun gpuGradientsMatchInterpreterElementwiseAndFiveGpuStepsTrackHost() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val (xv, yv) = task()
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        val model0 = mlp()
        val step = capture(model0, listOf(x)) { y ->
            val t = y.constant<Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            // ---- (1) one step, GPU vs interpreter, elementwise ----------------
            val values0 = listOf(xv) + model0.parameters.map { it.tensor.hostF32() }
            val want = DxirInterpreter.evalFunction(step.gradient, values0)
            val got = session.runOn(step.gradient, values0)
            assertEquals(want.size, got.size, "output arity (loss + input grad + param grads)")
            var maxDiff = 0f
            for (k in want.indices) {
                assertEquals(want[k].size, got[k].size, "output[$k] size")
                for (i in want[k].indices) maxDiff = maxOf(maxDiff, abs(got[k][i] - want[k][i]))
            }
            println("[nn-gpu] MLP gradient graph: GPU vs interpreter max|diff|=$maxDiff over ${want.size} outputs")
            assertTrue(maxDiff <= 1e-4f, "GPU gradients diverge from interpreter: max|diff|=$maxDiff")

            // ---- (2) five training steps entirely on the GPU lane -------------
            val opt = Adam(learningRate = 0.05f)

            var hostModel = mlp()
            var hostState = opt.initialState()
            var gpuModel = mlp()
            var gpuState = opt.initialState()
            val hostLosses = ArrayList<Float>(5)
            val gpuLosses = ArrayList<Float>(5)

            repeat(5) {
                // Host lane: the F1 route (interpreter).
                val hres = step.run(hostModel, listOf(x))
                hostLosses += hres.loss
                val (hm, hs) = opt.step(hostModel, hres.gradients, hostState)
                hostModel = hm; hostState = hs

                // GPU lane: same captured gradient function, PJRT execution,
                // host-side output unpacking mirroring CapturedStep.run.
                val values = listOf(xv) + gpuModel.parameters.map { it.tensor.hostF32() }
                val outs = session.runOn(step.gradient, values)
                gpuLosses += outs[0][0]
                val grads = LinkedHashMap<String, io.tlaloc.core.DTensor<*, io.tlaloc.core.F32>>()
                for ((j, key) in step.parameterKeys.withIndex()) {
                    val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
                    grads[key] = io.tlaloc.core.DTensor<Shape, io.tlaloc.core.F32>(
                        io.tlaloc.core.HostF32Storage(outs[1 + step.inputCount + j]), dims, io.tlaloc.core.F32,
                    )
                }
                val (gm, gs) = opt.step(gpuModel, grads, gpuState)
                gpuModel = gm; gpuState = gs
            }

            assertEquals(1, session.cacheSize, "all five GPU steps must reuse ONE cached executable")

            for (i in 0 until 5) {
                assertTrue(
                    abs(gpuLosses[i] - hostLosses[i]) <= 1e-4f,
                    "step $i loss: GPU ${gpuLosses[i]} vs host ${hostLosses[i]} beyond 1e-4",
                )
            }
            var maxParamDiff = 0f
            val hostParams = hostModel.parameters
            val gpuParams = gpuModel.parameters
            assertEquals(hostParams.map { it.key }, gpuParams.map { it.key })
            for (j in hostParams.indices) {
                val hp = hostParams[j].tensor.hostF32()
                val gp = gpuParams[j].tensor.hostF32()
                for (i in hp.indices) maxParamDiff = maxOf(maxParamDiff, abs(hp[i] - gp[i]))
            }
            println("[nn-gpu] 5 GPU training steps: losses=$gpuLosses; final-param max|diff| vs host=$maxParamDiff")
            assertTrue(maxParamDiff <= 1e-3f, "final parameters diverge: max|diff|=$maxParamDiff")
            assertTrue(gpuLosses[4] < gpuLosses[0], "the GPU lane must be learning: $gpuLosses")
        }
    }
}
