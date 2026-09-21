package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.kptxInferenceKernelTemplates
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.471 — Phase H4: **recognizer-driven claiming reaches the serving
 * path**. A decode-shaped `PAGED_ATTENTION` op, lowered through
 * `lowerKernelChoice` with the KPTX inference registry on a GB10 target,
 * emits one typed-FFI `custom_call @kptx_paged_attention`, and that
 * emitter-produced program runs on the GB10 **with the hand-written PTX
 * paged walk computing inside it**.
 *
 * Nothing here is hand-written MLIR: the program comes from the claiming
 * pass, the launch from [KptxPagedAttention] over [KptxKernelRegistry],
 * and the numbers from the same fixture shape the H1a smoke uses — a
 * PERMUTED block table (pages interleaved between sequences, as a real
 * allocator hands them out), ragged `seqLens` including a sequence that
 * stops mid-page, one at exactly the window width and one at length 1,
 * and GQA with `group = 4`. Contiguity must never be what makes two
 * programs agree.
 *
 * Two oracles, in the KPTX arc's order:
 *
 * 1. **vs the interpreter's paged walk** — the reference arm from
 *    §0.4.465, which walks pages directly in Double. This is the real
 *    oracle: the kernel and the walk share no code.
 * 2. **vs the unclaimed program** — the same graph emitted without the
 *    registry, i.e. the gather-composed reference lowering, on the same
 *    client in the same session. Claimed and unclaimed must be the same
 *    answer, and the per-call cost of each is REPORTED (floors within one
 *    session — never medians across runs, the §0.4.337 framing).
 *
 * `blockTables`/`seqLens` ride as I32 CONSTS: `PjrtSession.runOn` is an
 * all-F32 param lane, the standing Phase H harness limit. They still
 * reach the kernel as device buffers — XLA materialises a constant — so
 * the dispatch path under test is the real one.
 */
class KptxPagedAttentionKernelTest {

    private val numSeqs = 8
    private val numHeads = 8
    private val numKvHeads = 2
    private val headDim = 64
    private val blockSize = 16
    private val maxBlocksPerSeq = 8
    private val numBlocks = numSeqs * maxBlocksPerSeq
    private val ctx = maxBlocksPerSeq * blockSize
    private val scale = 0.125f

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))

    /** A permutation of the pool, not an identity map. */
    private val table = IntArray(numSeqs * maxBlocksPerSeq) { (it * 37 + 11) % numBlocks }

    /** Ragged: full window, single token, mid-page stops, page-aligned stops. */
    private val lens = intArrayOf(ctx, 1, 17, 64, 33, ctx - 1, 5, 100)

    private fun pagedFn(): DxirFunction = DxirBuilder.function("paged_kptx") {
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        val t = const(
            FloatArray(table.size) { table[it].toFloat() },
            DxirType(I32, listOf(numSeqs, maxBlocksPerSeq)),
        )
        val l = const(
            FloatArray(lens.size) { lens[it].toFloat() },
            DxirType(I32, listOf(numSeqs)),
        )
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scale.toDouble())))
    }

    private fun claimedFn(): DxirFunction =
        lowerKernelChoice(pagedFn(), KernelTarget.NVIDIA_GB10, inferenceRegistry = kptxInferenceKernelTemplates)

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    /** GPU-less pin: claiming reaches this shape, and CPU_GENERIC does not. */
    @Test
    fun theDecodeShapeClaimsOnGb10AndNotOnCpu() {
        val gb10 = claimedFn().toStablehlo()
        assertEquals(1, Regex("custom_call @kptx_paged_attention\\(").findAll(gb10).count(), gb10)
        assertTrue("api_version = 4 : i32" in gb10)
        val cpu = lowerKernelChoice(
            pagedFn(), KernelTarget.CPU_GENERIC, inferenceRegistry = kptxInferenceKernelTemplates,
        ).toStablehlo()
        assertEquals(0, Regex("custom_call").findAll(cpu).count(), "CPU must keep the reference lowering")
    }

    @Test
    fun kptxPagedAttentionAgreesWithTheInterpreterAndTheUnclaimedProgram() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxPagedAttentionRegistration.ensure(pluginPath, scale)

        val q = pseudo(numSeqs * numHeads * headDim, 101)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 103)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 107)
        val inputs = listOf(q, k, v)

        val claimed = claimedFn()
        val unclaimed = pagedFn()
        val want = DxirInterpreter.evalFunction(unclaimed, inputs)[0]

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(claimed, inputs, cacheKey = "h4-claimed").single()
            val ref = session.runOn(unclaimed, inputs, cacheKey = "h4-unclaimed").single()
            assertEquals(want.size, got.size)

            var worstVsInterp = 0f
            var worstVsRef = 0f
            var worstRefVsInterp = 0f
            var mag = 0f
            for (i in want.indices) {
                worstVsInterp = maxOf(worstVsInterp, abs(want[i] - got[i]))
                worstVsRef = maxOf(worstVsRef, abs(ref[i] - got[i]))
                worstRefVsInterp = maxOf(worstRefVsInterp, abs(ref[i] - want[i]))
                mag = maxOf(mag, abs(want[i]))
            }
            // (1) The LOAD-BEARING oracle: the kernel against the
            // interpreter's Double-accumulating paged walk. They share no
            // code — one is PTX reading pages through the block table, the
            // other a JVM loop doing the same by hand.
            assertTrue(
                worstVsInterp <= 1e-4f,
                "KPTX paged kernel vs interpreter walk: worst |d| = $worstVsInterp",
            )
            // (2) The kernel against the gather-composed emission — two
            // DEVICE programs. Pinned an order looser than (1), and the
            // reason is worth stating because it is the opposite of the
            // usual one: the kernel is not the loose side. XLA lowers
            // `dot_general` to TF32 tensor cores by default, whose ~10-bit
            // mantissa lands exactly at the 2^-12 ≈ 2.4e-4 seen here at
            // magnitude ~1; the kernel's sequential `fma.rn.f32` chain is
            // within 1e-7 of the Double walk. So this gap is the
            // EMISSION's distance from the oracle, not the kernel's, and
            // `worstRefVsInterp` (printed) is the same number. H1a's smoke
            // pins 6e-8 on a headDim-4 / ctx-6 fixture — too small for XLA
            // to pick a tensor-core path at all, which is why this shows up
            // only now, at a shape with real work in it.
            assertTrue(
                worstVsRef <= 1e-3f,
                "KPTX paged kernel vs gather-composed emission: worst |d| = $worstVsRef",
            )
            assertTrue(
                worstVsInterp <= worstRefVsInterp,
                "the fused kernel must be at least as close to the Double oracle as the " +
                    "emission it replaces: kernel $worstVsInterp vs emission $worstRefVsInterp",
            )

            // Per-call cost, both lanes in THIS session, executable already
            // compiled and cached. Floors, not medians — the §0.4.337 rule.
            fun floorNanos(fn: DxirFunction, key: String): Long {
                repeat(5) { session.runOn(fn, inputs, cacheKey = key) }
                var best = Long.MAX_VALUE
                repeat(30) {
                    val t0 = System.nanoTime()
                    session.runOn(fn, inputs, cacheKey = key)
                    best = minOf(best, System.nanoTime() - t0)
                }
                return best
            }
            val claimedNs = floorNanos(claimed, "h4-claimed")
            val unclaimedNs = floorNanos(unclaimed, "h4-unclaimed")
            println(
                "[kptx-paged] numSeqs=$numSeqs numHeads=$numHeads numKvHeads=$numKvHeads " +
                    "headDim=$headDim blockSize=$blockSize ctx=$ctx — " +
                    "max |out| = $mag; worst |d| kernel-vs-interpreter = $worstVsInterp, " +
                    "emission-vs-interpreter = $worstRefVsInterp, kernel-vs-emission = $worstVsRef; " +
                    "per-call FLOOR (same session, cached executable, host round trip included): " +
                    "claimed @kptx_paged_attention = ${claimedNs / 1000}us, " +
                    "unclaimed gather-composed = ${unclaimedNs / 1000}us",
            )
        }
    }
}

/** Process-once registration — [KptxKernelRegistry] registrations are
 * permanent and a second one for the same (plugin, name) throws. */
private object KptxPagedAttentionRegistration {
    private val done = HashSet<String>()

    @Synchronized
    fun ensure(pluginPath: java.nio.file.Path, scale: Float) {
        val key = "$pluginPath|$scale"
        if (done.add(key)) KptxPagedAttention.register(pluginPath, scale)
    }
}
