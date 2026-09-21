package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.kptxInferenceKernelTemplates
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.estimateOccupancy
import io.tlaloc.kptx.resourceReport
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.kptx.KptxPagedAttention
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.481 — Phase H4b, slice K1: **the measurement apparatus for the
 * KPTX paged-attention performance tier**, and the honest baselines.
 *
 * §0.4.471 landed the correctness-tier kernel and reported ONE number at
 * ONE shape: claimed `@kptx_paged_attention` 465 µs against the
 * unclaimed gather-composed lowering's 310 µs — 1.5× slower, which is
 * why `defaultInferenceKernelTemplates` is empty and why nothing
 * registers this kernel in a deployment. A single point is not a
 * baseline: it cannot say whether the kernel loses by a constant, by a
 * factor in the context length, or by a factor in the batch. This test
 * sweeps [POINTS] — two TinyLlama-1.1B-shaped decode points and two
 * Llama-3-8B-shaped ones — and reports, per point, both lanes' per-call
 * floors together with the derived quantities that make the numbers
 * diagnosable.
 *
 * # Methodology, stated rather than assumed (the §0.4.337 discipline)
 *
 * - **Floors, never medians, and never across sessions.** Both lanes of
 *   a point are measured inside ONE [PjrtSession] with both executables
 *   already compiled and cached, timed **interleaved** (one rep of each,
 *   round robin). GB10 medians drift between runs; a floor over
 *   [MEASURED] reps is the number that survives comparison, and only if
 *   whatever drifts drifts across every lane.
 * - **What is in the DEVICE timing**: `PjrtSession.executeOn` against
 *   **pre-staged device buffers** — the FFM downcall, XLA's dispatch of
 *   the program, the device work, and the device-complete event await.
 *   No host→device transfer, no device→host copy, no per-call arena.
 *   This is the §0.4.309 benchmark instrument, and using it here was
 *   forced, not chosen: see below.
 * - **What is in the `round-trip` column**: the same work through
 *   `runOn` — the three F32 operands up and the result back. It is
 *   reported per point purely so the gap is visible, because **the gap
 *   is most of the number**. At the 8B-shaped points the caches are
 *   67–134 MB and the round trip is 25–58 ms against a device cost
 *   under 300 µs. The first version of this test reported round trips
 *   only, and its two lanes differed by less than the transfer's own
 *   run-to-run noise: a differencing apparatus built on it produced a
 *   NEGATIVE device cost. **That also qualifies a number already in the
 *   audit**: §0.4.471's "claimed 465 µs / unclaimed 310 µs" is a
 *   round-trip pair over a ~1 MB fixture, i.e. ≈ 2.3 GB/s of host
 *   traffic — a staging number with an attention kernel inside it, not
 *   an attention number.
 * - **Warmup** [WARMUP] reps per lane before any timing, after
 *   compilation.
 * - **Equal math, or no row.** Every point asserts the two lanes agree
 *   to [LANE_AGREEMENT] before either is timed. §0.4.471's test is the
 *   certification (kernel vs the interpreter's Double paged walk at
 *   1.2e-7); this is the guard that the two things being timed are
 *   computing the same function at this shape.
 *
 * # The external-reference lane, and why it is not here
 *
 * The slice asked for vLLM's own paged attention as a context row.
 * **It is not installable-and-callable in this venv**: vLLM 0.29.0 at
 * `~/.local/venvs/vllm` ships no `vllm._C` extension module at all
 * (`from vllm import _custom_ops` warns
 * `Failed to import from vllm._C` and exposes only
 * `paged_attention_rocm`), because 0.29's CUDA path routes attention
 * through FlashAttention/FlashInfer backends rather than the classic
 * `paged_attention_v1` op. Benchmarking a *different* algorithm through
 * a *different* launch path and printing it beside these rows would be
 * the fair-comparison hazard the slice warned about, with none of the
 * compensating value. **Named deferral**, with the gate stated: a vLLM
 * row belongs next to a FLASH-DECODE-shaped Tlaloc kernel, not next to
 * this three-launch correctness chain.
 *
 * The diagnosis this test's numbers feed is
 * [docs/KPTX_PAGED_PERF.md](../../../../../../../docs/KPTX_PAGED_PERF.md).
 */
class KptxPagedAttentionBenchTest {

    /**
     * A decode-shaped measurement point. `seqLens` is the FULL window at
     * every point: raggedness is a separate axis (the kernel's dead
     * lanes still cost a `-inf` store and the softmax still walks them),
     * and mixing it in here would make a row mean two things at once.
     */
    private data class Point(
        val label: String,
        val numSeqs: Int,
        val numHeads: Int,
        val numKvHeads: Int,
        val headDim: Int,
        val blockSize: Int,
        val maxBlocksPerSeq: Int,
    ) {
        val ctx: Int get() = blockSize * maxBlocksPerSeq

        /** MLIR symbol names admit no hyphens — a `func.func @a-b` is a
         * parse error out of the plugin, not a Tlaloc one. */
        val symbol: String get() = "paged_" + label.replace('-', '_')

        /** One private page pool per sequence — what an allocator hands
         * out at full occupancy, and the worst case for page locality. */
        val numBlocks: Int get() = numSeqs * maxBlocksPerSeq

        val rows: Int get() = numSeqs * numHeads
        val cacheElems: Long get() = numBlocks.toLong() * blockSize * numKvHeads * headDim

        /** The cost model's own formula for this op (`CostModel.kt`):
         * `4 · outElems · maxContextLen`. */
        val flops: Double get() = 4.0 * numSeqs * numHeads * headDim * ctx

        /** The MINIMUM device traffic any paged decode must move: every
         * live K and V element of every sequence, read once. */
        val kvBytes: Long get() = 2L * numSeqs * ctx * numKvHeads * headDim * 4

        /** The score matrix the three-stage chain materialises in GLOBAL
         * memory. Traffic is [SCORE_TRIPS] × this. */
        val scoreBytes: Long get() = rows.toLong() * ctx * 4

        /** What the host round trip moves regardless of lane: q + both
         * caches up, the output down. */
        val hostBytes: Long
            get() = numSeqs.toLong() * numHeads * headDim * 4 * 2 + 2 * cacheElems * 4
    }

    private companion object {
        const val WARMUP = 3
        const val MEASURED = 20

        /**
         * Global-memory round trips of the score matrix in the
         * three-stage chain, counted from the PTX:
         * `kptx_paged_scores` writes it (1 store); `kptx_paged_softmax`
         * is [KptxKernels]' shared row softmax, which is THREE strided
         * passes — max, sum-of-exp, normalize-in-place — so 3 loads and
         * 1 store; `kptx_paged_out` loads it again. **4 loads + 2 stores
         * = 6.**
         */
        /** Same-lane agreement floor. Looser than the serving lane's
         * 1e-3 on purpose and the reason is §0.4.471's: the GAP IS THE
         * EMISSION'S. XLA lowers the gather-composed `dot_general`s to
         * TF32 tensor cores (~2^-12 mantissa) while the kernel runs a
         * sequential `fma.rn.f32` chain, and the disagreement grows with
         * the context length being accumulated — at ctx 2048 it is a
         * thousand-term sum of TF32 products against an f32 one. The
         * certification of the kernel against a Double oracle lives in
         * `KptxPagedAttentionKernelTest`; this bound only has to catch
         * "these two lanes are not computing the same function". */
        const val LANE_AGREEMENT = 2e-2f

        /**
         * Global-memory round trips of the score matrix in the
         * three-stage chain, counted from the PTX:
         * `kptx_paged_scores` writes it (1 store); `kptx_paged_softmax`
         * is [KptxKernels]' shared row softmax, which is THREE strided
         * passes — max, sum-of-exp, normalize-in-place — so 3 loads and
         * 1 store; `kptx_paged_out` loads it again. **4 loads + 2 stores
         * = 6.**
         */
        const val SCORE_TRIPS = 6

        /** GB10, read from the driver on this box (48 SMs, sm_121,
         * 24 MB L2). Recorded so a row's grid can be compared against
         * the machine it ran on rather than against a remembered one. */
        const val GB10_SMS = 48

        /** One registration serves one scale (the §0.4.471 named
         * deferral: `scale` is baked into the PTX). Every point's graph
         * therefore carries THIS scale rather than its own
         * `1/sqrt(headDim)` — which changes no timing, since `scale` is
         * an immediate in a multiply that executes either way. */
        const val SCALE = 0.125f

        val POINTS = listOf(
            // TinyLlama-1.1B: 32 query heads, 4 KV heads (group 4), headDim 64.
            // The single-sequence point is the one that matters most for
            // latency and the one that under-fills the device worst.
            Point("tinyllama-s1-ctx256", 1, 32, 4, 64, 16, 16),
            Point("tinyllama-s8-ctx512", 8, 32, 4, 64, 16, 32),
            // Llama-3-8B: 32 query heads, 8 KV heads (group 4), headDim 128.
            Point("llama3-8b-s8-ctx1024", 8, 32, 8, 128, 16, 64),
            Point("llama3-8b-s16-ctx1024", 16, 32, 8, 128, 16, 64),
        )
    }

    private fun pagedFn(p: Point, name: String): DxirFunction = DxirBuilder.function(name) {
        val qType = DxirType(F32, listOf(p.numSeqs, p.numHeads, p.headDim))
        val cacheType = DxirType(F32, listOf(p.numBlocks, p.blockSize, p.numKvHeads, p.headDim))
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        // A permutation of the pool, not an identity map: page locality
        // must never be what makes a lane fast.
        val t = const(
            FloatArray(p.numSeqs * p.maxBlocksPerSeq) { ((it * 37 + 11) % p.numBlocks).toFloat() },
            DxirType(I32, listOf(p.numSeqs, p.maxBlocksPerSeq)),
        )
        val l = const(
            FloatArray(p.numSeqs) { p.ctx.toFloat() },
            DxirType(I32, listOf(p.numSeqs)),
        )
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to SCALE.toDouble())))
    }

    /**
     * The DISPATCH FLOOR lane: the same three staged buffers, the same
     * `executeOn` path, and one elementwise multiply on the smallest
     * operand instead of an attention. Whatever this costs is the FFM
     * downcall, XLA's dispatch of the program and the device-complete
     * await — the part of a `DEVICE` number that is not attention. It is
     * what lets the small-shape rows be read at all: at
     * `tinyllama-s1-ctx256` the whole point is 2.1 MFLOP, so "is this a
     * kernel cost or a per-call cost?" is the only question worth asking.
     */
    private fun dispatchFloorFn(p: Point, name: String): DxirFunction = DxirBuilder.function(name) {
        val qType = DxirType(F32, listOf(p.numSeqs, p.numHeads, p.headDim))
        val cacheType = DxirType(F32, listOf(p.numBlocks, p.blockSize, p.numKvHeads, p.headDim))
        val q = param("q", qType)
        // k and v stay PARAMS so the staged-buffer list, and therefore
        // the execute context, is shape-for-shape the attention lanes'.
        param("k", cacheType)
        param("v", cacheType)
        listOf(op(OpKind.MUL, listOf(q, q), qType))
    }

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    /**
     * The static half of the diagnosis: what the three paged kernels
     * DECLARE, and what that says about how many of them can be resident
     * per SM. No GPU needed, so this runs everywhere the suite runs and
     * the numbers in `KPTX_PAGED_PERF.md` stay pinned to the source.
     */
    @Test
    fun theThreeStageChainsDeclaredResourcesAndOccupancy() {
        val block = 256
        val report = KptxKernels.pagedAttentionModule(block = block, scale = SCALE).resourceReport()
        val lines = report.map { r ->
            val occ = r.estimateOccupancy(blockThreads = block, arch = "sm_121")
            "${r.name}: regSlots/thread=${r.regSlotsPerThread} smem=${r.staticSmemBytes}B -> " +
                "${occ.blocksPerSm} blocks/SM, ${occ.activeThreads} threads, " +
                "${occ.occupancyPct}% (limited by ${occ.limitingFactor})"
        }
        println("[kptx-paged-perf] declared resources @ block=$block, sm_121:\n  " + lines.joinToString("\n  "))

        // Three stages, in the launch order the registry uses.
        assertTrue(report.size == 3, "expected 3 kernels, got ${report.map { it.name }}")
        // §0.4.481 asserted here that BOTH page-walking stages declare
        // zero shared memory, and said in so many words: "if a future
        // kernel stages the page window, this assertion is what
        // notices." §0.4.482 is that future and this is that notice,
        // so the assertion is re-aimed rather than deleted.
        //
        // What changed and what did NOT: `kptx_paged_out` now declares
        // `4 * block` bytes, but that buffer is a CROSS-PARTITION
        // REDUCTION TREE (the same shape as the softmax's `smax`), not a
        // staged page window — stage 3's V elements still come from
        // global memory once each and are still never reused across the
        // block. `kptx_paged_scores` is untouched and still declares
        // zero, which is the remaining half of KPTX_PAGED_PERF item 3.
        val scores = report.single { it.name == "kptx_paged_scores" }
        assertTrue(
            scores.staticSmemBytes == 0,
            "the score stage still walks pages straight out of global memory; staging K would " +
                "change this: " + report.map { it.name to it.staticSmemBytes },
        )
        val outStage = report.single { it.name == "kptx_paged_out" }
        assertEquals(
            4 * block, outStage.staticSmemBytes,
            "stage 3's context split reduces through one f32 per thread",
        )
    }

    @Test
    fun pagedAttentionLaneFloorsAcrossDecodeShapes() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")
        KptxPagedAttentionBenchRegistration.ensure(pluginPath, SCALE)

        val rows = StringBuilder()
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (p in POINTS) {
                val q = pseudo(p.numSeqs * p.numHeads * p.headDim, 101)
                val k = pseudo((p.cacheElems).toInt(), 103)
                val v = pseudo((p.cacheElems).toInt(), 107)
                val inputs = listOf(q, k, v)

                val unclaimed = pagedFn(p, p.symbol)
                val claimed = lowerKernelChoice(
                    unclaimed, KernelTarget.NVIDIA_GB10, inferenceRegistry = kptxInferenceKernelTemplates,
                )

                // Equal math, or no row.
                val got = session.runOn(claimed, inputs, cacheKey = "c-${p.label}").single()
                val ref = session.runOn(unclaimed, inputs, cacheKey = "u-${p.label}").single()
                var worst = 0f
                for (i in got.indices) worst = maxOf(worst, abs(got[i] - ref[i]))
                assertTrue(
                    worst <= LANE_AGREEMENT,
                    "${p.label}: the two timed lanes disagree by $worst — the rows would not be " +
                        "measuring the same function",
                )

                val qBuf = session.bufferFromHostF32(q, listOf(p.numSeqs, p.numHeads, p.headDim))
                val kBuf = session.bufferFromHostF32(k, listOf(p.numBlocks, p.blockSize, p.numKvHeads, p.headDim))
                val vBuf = session.bufferFromHostF32(v, listOf(p.numBlocks, p.blockSize, p.numKvHeads, p.headDim))
                val staged = listOf(qBuf, kBuf, vBuf)
                val (cDev, uDev, floor) = try {
                    // The two DEVICE lanes, timed INTERLEAVED — one rep of
                    // each, round robin, never one lane to completion and
                    // then the other. A floor is comparable across lanes
                    // only if whatever drifts drifts across all of them;
                    // lane-at-a-time gives each lane its own slice of the
                    // session's history. (Found the hard way: it produced
                    // a NEGATIVE slope in the differencing apparatus this
                    // replaced.)
                    val lanes = listOf(claimed, unclaimed, dispatchFloorFn(p, "${p.symbol}_floor"))
                    repeat(WARMUP) { for (fn in lanes) session.executeOn(fn, staged).forEach { it.close() } }
                    val best = DoubleArray(lanes.size) { Double.MAX_VALUE }
                    repeat(MEASURED) {
                        for ((i, fn) in lanes.withIndex()) {
                            val t0 = System.nanoTime()
                            val outs = session.executeOn(fn, staged)
                            val dt = (System.nanoTime() - t0) / 1000.0
                            outs.forEach { it.close() }
                            best[i] = minOf(best[i], dt)
                        }
                    }
                    Triple(best[0], best[1], best[2])
                } finally {
                    staged.forEach { it.close() }
                }

                // The round trip, for context only: the SAME work with
                // `runOn`'s host staging around it. The gap between the
                // two columns is the transfer, and it is what §0.4.471's
                // 465 µs was mostly made of.
                var cRt = Double.MAX_VALUE
                var uRt = Double.MAX_VALUE
                repeat(MEASURED) {
                    var t0 = System.nanoTime()
                    session.runOn(claimed, inputs, cacheKey = "c-${p.label}")
                    cRt = minOf(cRt, (System.nanoTime() - t0) / 1000.0)
                    t0 = System.nanoTime()
                    session.runOn(unclaimed, inputs, cacheKey = "u-${p.label}")
                    uRt = minOf(uRt, (System.nanoTime() - t0) / 1000.0)
                }

                // Derived diagnosis columns. `scoreTraffic` is the cost
                // the fused tier deletes outright; `CTA/SM` is the cost it
                // cannot delete by fusion alone.
                val scoreTraffic = SCORE_TRIPS.toLong() * p.scoreBytes
                val ctasPerSm = p.rows.toDouble() / GB10_SMS
                rows.append(
                    ("  %-22s ctx=%-5d grid=%-5d (%5.2f CTA/SM) | DEVICE c=%8.1f u=%8.1f ratio=%5.2fx " +
                        "(dispatch floor %6.1f) | round-trip c=%8.1f u=%8.1f | ")
                        .format(p.label, p.ctx, p.rows, ctasPerSm, cDev, uDev, cDev / uDev, floor, cRt, uRt),
                ).append(
                    ("kv=%7.2fMB @ c=%6.1f u=%6.1f GB/s | score=%.2fMB x%d=%.2fMB | flops=%.1fM | " +
                        "host=%7.2fMB | |d|=%.1e\n")
                        .format(
                            p.kvBytes / 1e6, p.kvBytes / 1e3 / cDev, p.kvBytes / 1e3 / uDev,
                            p.scoreBytes / 1e6, SCORE_TRIPS, scoreTraffic / 1e6, p.flops / 1e6,
                            p.hostBytes / 1e6, worst,
                        ),
                )
                // The floor must BE a floor, or the interleave is lying.
                assertTrue(
                    floor > 0.0 && floor <= cDev && floor <= uDev,
                    "${p.label}: the dispatch floor ($floor us) is not below both attention lanes " +
                        "(claimed $cDev us, unclaimed $uDev us)",
                )
                assertTrue(cRt >= cDev && uRt >= uDev, "${p.label}: a round trip came in under its own device cost")
            }
        }
        println(
            "[kptx-paged-perf] FLOORS, one session, executables cached, warmup=$WARMUP reps=$MEASURED, " +
                "GB10 ${GB10_SMS}SM sm_121. `DEVICE` = PjrtSession.executeOn over PRE-STAGED device " +
                "buffers (dispatch + device work + device-complete await, no host transfer); " +
                "`round-trip` = the same work through runOn, transfer included; " +
                "c = claimed @kptx_paged_attention, u = unclaimed gather-composed:\n$rows",
        )
        java.io.File("build").mkdirs()
        java.io.File("build/harness-results-kptx-paged.txt").writeText(rows.toString())
    }
}

/** Process-once registration — [io.tlaloc.runtime.pjrt.kptx.KptxKernelRegistry]
 * registrations are permanent and a second one for the same (plugin, name)
 * throws. */
private object KptxPagedAttentionBenchRegistration {
    private val done = HashSet<String>()

    @Synchronized
    fun ensure(pluginPath: java.nio.file.Path, scale: Float) {
        val key = "$pluginPath|$scale"
        if (done.add(key)) KptxPagedAttention.register(pluginPath, scale)
    }
}
