package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.RmsNormKernel
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * KPTX v1.9 (§0.4.337) — the `tlaloc-pjrt-kptx-cuda` matrix row, closing
 * KPTX v1's definition-of-done: *overhead quantified vs the
 * PJRT-FFM-CUDA baseline*.
 *
 * Same LlamaDecoder-medium workload and timing discipline as
 * [LlamaDecoderPjrtFfmBenchTest] (§0.4.308: one PjrtSession, prepare
 * upfront, pre-staged inputs, 1.5 s budget per mode), but the programs
 * come from the kernel-lowered pipeline with the KPTX registry — the
 * forward carries two `@kptx_rms_norm` custom_calls (§0.4.336), the
 * backward carries the forward-recompute RmsNorm claims (the adjoints
 * themselves decompose; backward *claiming* is v2 task 15).
 *
 * Both lanes (kptx and decompose) are measured **in the same session,
 * same run**, so the overhead delta is immune to run-to-run variance.
 * Only the kptx rows are dumped to
 * `build/harness-results-tlaloc-pjrt-kptx-cuda-medium.{csv,json}` for
 * the §0.4.298 aggregator; the decompose lane is the in-run reference
 * (it *is* the tlaloc-pjrt-ffm-cuda row's program).
 *
 * Expectation to pin: XLA fuses the decomposed rms_norm into
 * neighbouring elementwise ops, so replacing it with an opaque
 * custom_call *loses fusion* — the kptx lane is expected to carry
 * overhead at this kernel granularity, not win. The row's purpose is to
 * measure that cost honestly; winning requires coarser kernels
 * (attention, MLP) — the v2 arc. No relative-performance assertion is
 * made: correctness assertions only, timings printed + dumped.
 */
class LlamaDecoderKptxBenchTest {

    private val kptxRegistry: Map<String, KernelTemplate> = mapOf("RmsNorm" to RmsNormKernel)

    private data class TimingStats(
        val medianNs: Long,
        val minNs: Long,
        val p99Ns: Long,
        val nIterations: Int,
    )

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

    @Test
    fun benchmarksLlamaMediumViaKptxCuda() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxTestKernels.ensureRmsNormRegistered(pluginPath)

        val kptxFwd = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        )
        val kptxBwd = llamaKernelLoweredBackwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        )
        val decompFwd = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        )
        val decompBwd = llamaKernelLoweredBackwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        )

        val fwdCalls = Regex("custom_call @kptx_rms_norm\\(").findAll(kptxFwd.toStablehlo()).count()
        val bwdCalls = Regex("custom_call @kptx_rms_norm\\(").findAll(kptxBwd.toStablehlo()).count()
        assertTrue(fwdCalls == 2, "kptx forward should carry 2 rms_norm custom_calls; got $fwdCalls")

        val inputs = llamaSynthesizeInputs(seed = 42L, fn = kptxFwd)

        // §0.4.337 — benchmark sessions opt into a bounded preallocated pool
        // (25% of unified memory ≈ 32 GB, one client, freed at session close):
        // the §0.4.333 no-preallocate safe default routes steady-state
        // allocations through the driver and costs ~2.3× on this workload.
        val benchOptions = io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions(
            memoryFraction = 0.25f, preallocate = true,
        )
        PjrtSession(target = PjrtTarget.Cuda, options = benchOptions).use { session ->
            session.prepare(kptxFwd)
            session.prepare(kptxBwd)
            session.prepare(decompFwd)
            session.prepare(decompBwd)

            val stagedInputs = inputs.zip(kptxFwd.params).map { (arr, p) ->
                session.bufferFromHostF32(arr, p.type.dims)
            }
            try {
                val sanityLoss = session.runOn(kptxFwd, inputs).single().single()
                assertTrue(sanityLoss.isFinite(), "kptx forward loss must be finite; got $sanityLoss")

                val minNanos = 1_500_000_000L
                val warmup = 5

                val kptxFwdStats = timeLoop(session, kptxFwd, stagedInputs, warmup, minNanos)
                val kptxBwdStats = timeLoop(session, kptxBwd, stagedInputs, warmup, minNanos)
                val decompFwdStats = timeLoop(session, decompFwd, stagedInputs, warmup, minNanos)
                val decompBwdStats = timeLoop(session, decompBwd, stagedInputs, warmup, minNanos)

                fun us(ns: Long) = ns / 1_000
                val stepKptxUs = us(kptxFwdStats.medianNs + kptxBwdStats.medianNs)
                val stepDecompUs = us(decompFwdStats.medianNs + decompBwdStats.medianNs)
                println(
                    "[llama-medium-pjrt-kptx-cuda] " +
                        "kptx fwd=${us(kptxFwdStats.medianNs)} us (min ${us(kptxFwdStats.minNs)}, ${fwdCalls} custom_calls) | " +
                        "kptx bwd=${us(kptxBwdStats.medianNs)} us (min ${us(kptxBwdStats.minNs)}, ${bwdCalls} custom_calls) | " +
                        "decompose fwd=${us(decompFwdStats.medianNs)} us (min ${us(decompFwdStats.minNs)}) | " +
                        "decompose bwd=${us(decompBwdStats.medianNs)} us (min ${us(decompBwdStats.minNs)}) | " +
                        "step kptx=${stepKptxUs} us vs decompose=${stepDecompUs} us " +
                        "(overhead ${stepKptxUs - stepDecompUs} us) | loss=$sanityLoss",
                )

                // No fwd-vs-bwd shape assertion: each custom_call is a host
                // round-trip (XLA upcalls into the JVM handler per execution),
                // and at rms_norm granularity that jitter (~0–1.2 ms/call,
                // run-dependent) can exceed the fwd/bwd compute delta. The
                // decompose lanes in the same run are the stable reference;
                // the jitter itself is the row's finding (v3.4 go/no-go input).
                assertTrue(kptxFwdStats.medianNs > 0)
                assertTrue(kptxBwdStats.medianNs > 0)

                val forward = HeadToHeadResult(
                    benchmark = "llama-decoder-medium-forward",
                    forwardValue = sanityLoss,
                    gradientValues = emptyList(),
                    warmupIterations = warmup,
                    measuredIterations = kptxFwdStats.nIterations,
                    medianNanos = kptxFwdStats.medianNs,
                    minNanos = kptxFwdStats.minNs,
                    p99Nanos = kptxFwdStats.p99Ns,
                    framework = "tlaloc-pjrt-kptx-cuda",
                )
                val backward = HeadToHeadResult(
                    benchmark = "llama-decoder-medium-backward",
                    forwardValue = Float.NaN,
                    gradientValues = emptyList(),
                    warmupIterations = warmup,
                    measuredIterations = kptxBwdStats.nIterations,
                    medianNanos = kptxBwdStats.medianNs,
                    minNanos = kptxBwdStats.minNs,
                    p99Nanos = kptxBwdStats.p99Ns,
                    framework = "tlaloc-pjrt-kptx-cuda",
                )

                val outputDir = java.io.File("build")
                outputDir.mkdirs()
                java.io.File(outputDir, "harness-results-tlaloc-pjrt-kptx-cuda-medium.csv").writeText(
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
                java.io.File(outputDir, "harness-results-tlaloc-pjrt-kptx-cuda-medium.json").writeText(
                    "[\n" +
                        listOf(forward, backward).joinToString(",\n") { r ->
                            "  ${r.toJsonString().lines().joinToString("\n  ")}"
                        } +
                        "\n]\n",
                )
            } finally {
                stagedInputs.forEach { it.close() }
            }
        }
    }
}
