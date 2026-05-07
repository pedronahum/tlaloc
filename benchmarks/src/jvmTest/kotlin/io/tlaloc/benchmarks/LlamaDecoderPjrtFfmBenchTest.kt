package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.308 — pure-Kotlin LlamaDecoder benchmark via [PjrtSession]. The
 * §0.4.299 [LlamaDecoderPjrtXlaBenchTest] exercised the same backend
 * (PJRT-XLA-CUDA) but went through a Python subprocess (`run_pjrt_xla_llama_bench.py`)
 * — useful as a "does this even compile in PJRT?" spike but with Python
 * latency on the timing path. This test replaces that with the §0.4.303–§0.4.307
 * FFM stack: plugin loaded once, client created once, executable compiled
 * once via [PjrtSession.prepare], dispatch in a tight Kotlin loop with no
 * subprocess at the boundary.
 *
 * The matrix row this writes — `tlaloc-pjrt-ffm-cuda` — is the most
 * apples-to-apples comparison vs JAX-GPU because:
 *   - same backend (PJRT-XLA via OpenXLA's `xla_cuda_plugin.so`),
 *   - same MLIR shape (Tlaloc emits StableHLO, JAX lowers to StableHLO too),
 *   - same per-call sync (block on device-complete event before reading),
 *   - no Python overhead per iteration.
 */
class LlamaDecoderPjrtFfmBenchTest {

    /**
     * Runs the warmup + measured timing loop via [PjrtSession.runOn]. The
     * session is owned by the caller (so forward + backward share the same
     * client / device / arena), but each timing loop measures one mode in
     * isolation.
     */
    private data class TimingStats(
        val medianNs: Long,
        val minNs: Long,
        val p99Ns: Long,
        val nIterations: Int,
    )

    /**
     * Time-loop using the §0.4.308 [PjrtSession.executeOn] / [PjrtSession.bufferFromHostF32]
     * pattern: inputs are pre-staged onto the device once; the timing loop
     * measures only execute + sync + output close (no per-iter host transfer).
     * Mirrors JAX's `arr.block_until_ready()` benchmark style — apples-to-apples
     * with the §0.4.297 / §0.4.299 numbers.
     */
    private fun timeLoop(
        session: PjrtSession,
        fn: io.tlaloc.ir.DxirFunction,
        stagedInputs: List<io.tlaloc.runtime.pjrt.ffm.PjrtBuffer>,
        warmupIters: Int,
        minTimeNanos: Long,
        maxIters: Int = 10_000,
    ): TimingStats {
        repeat(warmupIters) {
            val outs = session.executeOn(fn, stagedInputs)
            outs.forEach { it.close() }
        }
        val deadline = System.nanoTime() + minTimeNanos
        val times = ArrayList<Long>(1024)
        while (true) {
            val t0 = System.nanoTime()
            val outs = session.executeOn(fn, stagedInputs)
            outs.forEach { it.close() }
            val t1 = System.nanoTime()
            times += t1 - t0
            if (System.nanoTime() >= deadline || times.size >= maxIters) break
        }
        times.sort()
        val n = times.size
        return TimingStats(
            medianNs = times[n / 2],
            minNs = times.first(),
            p99Ns = times[((n - 1) * 99) / 100],
            nIterations = n,
        )
    }

    private fun dump(forward: HeadToHeadResult, backward: HeadToHeadResult, suffix: String) {
        val outputDir = java.io.File("build")
        outputDir.mkdirs()
        java.io.File(outputDir, "harness-results-$suffix.csv").writeText(
            buildString {
                append("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
                for (r in listOf(forward, backward)) {
                    append(r.benchmark); append(',')
                    append(r.framework); append(',')
                    append(r.measuredIterations); append(',')
                    append(r.medianNanos); append(',')
                    append(r.minNanos); append(',')
                    append(r.p99Nanos); append('\n')
                }
            },
        )
        java.io.File(outputDir, "harness-results-$suffix.json").writeText(
            "[\n" +
                listOf(forward, backward).joinToString(",\n") { r ->
                    "  ${r.toJsonString().lines().joinToString("\n  ")}"
                } +
                "\n]\n",
        )
    }

    @Test
    fun benchmarksLlamaMediumViaPjrtFfmCuda() {
        assumeTrue(
            PjrtBinaries.available,
            "PJRT plugin not resolved — set TLALOC_PJRT_PLUGIN_PATH or pip-install jax[cuda12].",
        )
        // PjrtBinaries.cudaAvailable uses nvidia-smi -L; reuse IreeBinaries'
        // version too in case the host has the IREE detector configured but
        // not nvidia-smi (defence in depth on dev boxes).
        assumeTrue(
            PjrtBinaries.cudaAvailable || IreeBinaries.cudaAvailable,
            "no CUDA device — skipping.",
        )

        // Build forward + backward DxirFunctions through the same coarsen +
        // decompose pipeline the §0.4.292 / §0.4.295 tests use, so this row
        // measures the *same* MLIR the IREE-CUDA / PyTorch / JAX rows do.
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val fwd = decomposeCoarsened(coarsened)
        val grad = DxirReverseTransform.apply(coarsened)
        val bwd = decomposeCoarsened(grad)

        val inputs = llamaSynthesizeInputs(seed = 42L, fwd)

        // Single PjrtSession holds the CUDA client + compile cache for both
        // forward and backward — one XLA-service init, one CUDA context.
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            // Compile both upfront so the timing loops measure dispatch only.
            session.prepare(fwd)
            session.prepare(bwd)
            require(session.cacheSize == 2) {
                "expected 2 cached executables (forward+backward); got ${session.cacheSize}"
            }

            // Pre-stage inputs onto the device once. Both forward and backward
            // share the same input set (same DxirFunction params; AD preserves
            // them positionally). Apples-to-apples with JAX's bench which
            // jax.device_put's inputs once outside the timing loop.
            val stagedInputs = inputs.zip(fwd.params).map { (arr, p) ->
                session.bufferFromHostF32(arr, p.type.dims)
            }
            try {
                // Sanity dispatch via runOn (FloatArray-in/out) for the loss read-back.
                val sanityFwdOut = session.runOn(fwd, inputs)
                require(sanityFwdOut.size == 1 && sanityFwdOut.single().size == 1) {
                    "forward returns one scalar loss; got ${sanityFwdOut.size}"
                }
                val sanityLoss = sanityFwdOut.single().single()
                assertTrue(sanityLoss.isFinite(), "forward loss must be finite; got $sanityLoss")

                // 1.5 s budget per mode, matching the §0.4.296 / §0.4.297 / §0.4.299
                // Python harness scripts so iteration counts are comparable.
                val minNanos = 1_500_000_000L
                val warmup = 5

                val fwdStats = timeLoop(session, fwd, stagedInputs, warmup, minNanos)
                val bwdStats = timeLoop(session, bwd, stagedInputs, warmup, minNanos)

            println(
                "[llama-medium-pjrt-ffm-cuda] forward median=${fwdStats.medianNs / 1_000} us " +
                    "iters=${fwdStats.nIterations} | " +
                    "backward median=${bwdStats.medianNs / 1_000} us iters=${bwdStats.nIterations} | " +
                    "loss=$sanityLoss",
            )

            assertTrue(fwdStats.medianNs > 0)
            assertTrue(bwdStats.medianNs > 0)
            assertTrue(
                bwdStats.medianNs > fwdStats.medianNs,
                "backward should be slower than forward (${bwdStats.medianNs} vs ${fwdStats.medianNs} ns)",
            )

            val forward = HeadToHeadResult(
                benchmark = "llama-decoder-medium-forward",
                forwardValue = sanityLoss,
                gradientValues = emptyList(),
                warmupIterations = warmup,
                measuredIterations = fwdStats.nIterations,
                medianNanos = fwdStats.medianNs,
                minNanos = fwdStats.minNs,
                p99Nanos = fwdStats.p99Ns,
                framework = "tlaloc-pjrt-ffm-cuda",
            )
            val backward = HeadToHeadResult(
                benchmark = "llama-decoder-medium-backward",
                forwardValue = Float.NaN,
                gradientValues = emptyList(),
                warmupIterations = warmup,
                measuredIterations = bwdStats.nIterations,
                medianNanos = bwdStats.medianNs,
                minNanos = bwdStats.minNs,
                p99Nanos = bwdStats.p99Ns,
                framework = "tlaloc-pjrt-ffm-cuda",
            )
                dump(forward, backward, "tlaloc-pjrt-ffm-cuda-medium")
            } finally {
                stagedInputs.forEach { it.close() }
            }
        }
    }
}
