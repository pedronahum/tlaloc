package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** KPTX v3.3 (§0.4.347) — resource report + occupancy estimate tests. */
class PtxResourceReportTest {

    @Test
    fun reportsDeclaredBanksAndStaticSmem() {
        val dx = KptxKernels.rmsNormBwdDx.specialize(shapes = mapOf("block" to 256))
        val res = dx.resourceReport().single()
        assertEquals("kptx_rms_norm_bwd_dx", res.name)
        assertEquals(2048, res.staticSmemBytes, "sdata + sdata2, 4·256 each")
        assertEquals(6, res.regBanks[IsaRegClass.PRED])
        assertEquals(10, res.regBanks[IsaRegClass.R32])
        assertEquals(31, res.regBanks[IsaRegClass.F32])
        assertEquals(34, res.regBanks[IsaRegClass.R64])
        // 32-bit slots: r32 + f32 + 2·r64; predicates excluded.
        assertEquals(10 + 31 + 68, res.regSlotsPerThread)
    }

    @Test
    fun estimatesOccupancyWithLimitingFactorNamed() {
        val eps = KptxKernels.rmsNormEps.specialize(shapes = mapOf("block" to 256))
        val res = eps.resourceReport().single()

        // sm_75 @ 256 threads: threads allow 4 blocks (1024/256), smem
        // allows 64 (65536/1024) — but the declared banks (10 r32 + 15
        // f32 + 2·22 r64 = 69 slots/thread → 17664/block) allow only
        // 65536/17664 = 3. The upper-bound-proxy verdict: at most 75%,
        // register-limited *as declared* (the driver JIT usually does
        // better — see the KDoc caveats).
        val est = res.estimateOccupancy(blockThreads = 256, arch = "sm_75")
        assertEquals(3, est.blocksPerSm)
        assertEquals(75, est.occupancyPct)
        assertEquals("registers(declared)", est.limitingFactor)

        // Synthetic smem hog: 32 KB static smem at 128 threads on sm_75 —
        // shared memory binds (2 blocks), occupancy 25%.
        val hog = KernelResources("hog", emptyMap(), staticSmemBytes = 32768)
        val hogEst = hog.estimateOccupancy(blockThreads = 128, arch = "sm_75")
        assertEquals(2, hogEst.blocksPerSm)
        assertEquals(25, hogEst.occupancyPct)
        assertEquals("shared-memory", hogEst.limitingFactor)
    }

    @Test
    fun archTableNormalizesAndRejectsUnknown() {
        val res = KernelResources("k", mapOf(IsaRegClass.F32 to 4), 0)
        // GB10's sm_121a maps to the sm_86-class row (1536 threads/SM).
        val gb10 = res.estimateOccupancy(blockThreads = 256, arch = "sm_121a")
        assertEquals(6, gb10.blocksPerSm)
        assertEquals(100, gb10.occupancyPct)

        val e = assertFailsWith<IllegalArgumentException> {
            res.estimateOccupancy(blockThreads = 256, arch = "sm_42")
        }
        assertTrue(e.message!!.contains("no SM limits table"), e.message)
    }
}
