package io.tlaloc.ir.recognizer.cost

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.254+ — device descriptor sanity tests.
 *
 * Pin the canonical seven descriptors + spot-check a few well-known
 * vendor-published numbers (H100 BF16 = 989 TFLOPs, A100 HBM = 1.94 TB/s,
 * TPU v4 BF16 = 275 TFLOPs). Catches accidental-edit drift; if a vendor
 * updates a sustained-perf number we update the test alongside the
 * descriptor.
 */
class DeviceDescriptorTest {

    @Test
    fun sevenCanonicalDescriptors() {
        assertEquals(7, DeviceDescriptors.all.size)
        val names = DeviceDescriptors.all.map { it.name }.toSet()
        assertEquals(
            setOf("h100", "a100", "tpu_v4", "tpu_v5e", "tpu_v6e", "trainium2", "cpu_generic"),
            names,
        )
    }

    @Test
    fun h100Citations() {
        val d = DeviceDescriptors.H100
        assertEquals("nvidia", d.vendor)
        assertEquals(989e12, d.peakFlopsBf16, "H100 BF16 = 989 TFLOPs (datasheet)")
        assertEquals(67e12, d.peakFlopsF32)
        assertEquals(1979e12, d.peakFlopsFp8)
        assertEquals(3.35e12, d.hbmBandwidthBytesPerSec)
        assertEquals(80L * 1024 * 1024 * 1024, d.hbmCapacityBytes)
        assertEquals(132, d.smOrCoreCount)
        assertEquals(2.0, d.extras["sparsity_factor"])
    }

    @Test
    fun a100Citations() {
        val d = DeviceDescriptors.A100
        assertEquals(312e12, d.peakFlopsBf16, "A100 BF16 = 312 TFLOPs (whitepaper)")
        assertEquals(19.5e12, d.peakFlopsF32)
        assertNull(d.peakFlopsFp8, "A100 has no native FP8")
        assertEquals(1.94e12, d.hbmBandwidthBytesPerSec)
        assertEquals(108, d.smOrCoreCount)
    }

    @Test
    fun tpuV4Citations() {
        val d = DeviceDescriptors.TPU_V4
        assertEquals("google", d.vendor)
        assertEquals(275e12, d.peakFlopsBf16, "TPU v4 BF16 = 275 TFLOPs (Jouppi 2023)")
        assertEquals(1.2e12, d.hbmBandwidthBytesPerSec)
        assertEquals(128L * 1024 * 1024, d.onChipSramBytes, "TPU v4 VMEM = 128 MiB")
        assertEquals(2, d.smOrCoreCount, "two TensorCores per v4 chip")
    }

    @Test
    fun tpuV5eIsCostOptimizedSlowerThanV4() {
        val v4 = DeviceDescriptors.TPU_V4
        val v5e = DeviceDescriptors.TPU_V5E
        // v5e is cost-optimized — explicitly slower than v4.
        assertTrue(v5e.peakFlopsBf16 < v4.peakFlopsBf16)
        assertTrue(v5e.hbmBandwidthBytesPerSec < v4.hbmBandwidthBytesPerSec)
        assertTrue(v5e.hbmCapacityBytes < v4.hbmCapacityBytes)
    }

    @Test
    fun tpuV6eIsTrilliumScale() {
        val v6e = DeviceDescriptors.TPU_V6E
        // Trillium claims 4.7× per-chip BF16 perf vs v5e (Google 2024).
        assertTrue(
            v6e.peakFlopsBf16 > 4.5 * DeviceDescriptors.TPU_V5E.peakFlopsBf16,
            "TPU v6e BF16 should be ~4.7× v5e per Google's announcement",
        )
        assertEquals(true, v6e.extras["sparsecore_present"])
    }

    @Test
    fun trainium2HasFp8AndLargeHbm() {
        val t2 = DeviceDescriptors.TRAINIUM2
        assertEquals("aws", t2.vendor)
        assertEquals(1300e12, t2.peakFlopsFp8, "Trainium2 FP8 = ~1.3 PFLOPs")
        assertEquals(96L * 1024 * 1024 * 1024, t2.hbmCapacityBytes, "Trainium2 ships with 96 GiB HBM")
        assertTrue(t2.hbmBandwidthBytesPerSec > 2.5e12, "Trainium2 HBM3e bandwidth ~2.9 TB/s")
    }

    @Test
    fun cpuGenericIsTheDecomposeOnlyFallback() {
        val cpu = DeviceDescriptors.CPU_GENERIC
        assertEquals("tlaloc", cpu.vendor)
        assertNull(cpu.peakFlopsFp8)
        // CPU has way less peak than any GPU/TPU.
        assertTrue(cpu.peakFlopsBf16 < DeviceDescriptors.A100.peakFlopsBf16 / 10.0)
    }

    @Test
    fun lookupByNameRoundTrips() {
        for (d in DeviceDescriptors.all) {
            assertEquals(d, DeviceDescriptors.byName(d.name))
        }
        assertNull(DeviceDescriptors.byName("unknown_chip_xyz"))
    }

    @Test
    fun arithmeticIntensityScalesWithHbmBandwidth() {
        // Peak BF16 FLOPs per byte of HBM bandwidth gives the
        // arithmetic intensity at which a kernel becomes compute-bound.
        // Higher = harder to be memory-bound, which matches what we
        // observe across the line:
        //   H100: ~295  (989 TFLOPs / 3.35 TB/s)
        //   A100: ~161  (312 / 1.94)
        //   TPU v4: ~229 (275 / 1.2)
        //   TPU v5e: ~241 (197 / 0.819)
        //   TPU v6e: ~579 (926 / 1.6)  — most compute-dense
        //   Trainium2: ~228 (660 / 2.9)
        //   CPU: ~67 (20 / 0.3)        — easiest to be memory-bound
        val h100 = DeviceDescriptors.H100
        val tpuV6e = DeviceDescriptors.TPU_V6E
        val cpu = DeviceDescriptors.CPU_GENERIC

        assertTrue(tpuV6e.peakBf16FlopsPerByte() > h100.peakBf16FlopsPerByte())
        assertTrue(h100.peakBf16FlopsPerByte() > cpu.peakBf16FlopsPerByte())
    }

    @Test
    fun allDescriptorsHaveNonZeroFundamentals() {
        for (d in DeviceDescriptors.all) {
            assertTrue(d.peakFlopsF32 > 0.0, "${d.name} F32 peak must be > 0")
            assertTrue(d.peakFlopsBf16 > 0.0, "${d.name} BF16 peak must be > 0")
            assertTrue(d.hbmBandwidthBytesPerSec > 0.0, "${d.name} bandwidth must be > 0")
            assertTrue(d.hbmCapacityBytes > 0L, "${d.name} HBM capacity must be > 0")
            assertTrue(d.onChipSramBytes > 0L, "${d.name} SRAM must be > 0")
            assertTrue(d.smOrCoreCount > 0, "${d.name} SM/core count must be > 0")
        }
    }
}
