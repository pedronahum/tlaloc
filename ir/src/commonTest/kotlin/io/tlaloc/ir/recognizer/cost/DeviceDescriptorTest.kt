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
    fun tenCanonicalDescriptors() {
        assertEquals(10, DeviceDescriptors.all.size)
        val names = DeviceDescriptors.all.map { it.name }.toSet()
        assertEquals(
            setOf(
                "h100", "a100", "tpu_v4", "tpu_v5e", "tpu_v6e", "trainium2", "cpu_generic",
                "gb10", "b100", "b200",
            ),
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
    fun gb10IsLpddrNotHbm() {
        // GB10 (DGX Spark) is the only Blackwell SKU with LPDDR5X
        // memory rather than HBM3e — the dev/edge form factor's defining
        // property. Bandwidth roughly 30× lower than B100/B200; capacity
        // 1.5× lower; cost-model scoring on this SKU should reflect a
        // much smaller arithmetic-intensity ceiling.
        val gb10 = DeviceDescriptors.GB10
        assertEquals("nvidia", gb10.vendor)
        assertEquals("lpddr5x", gb10.extras["memory_kind"], "GB10 uses LPDDR5X, not HBM3e")
        assertEquals(125e12, gb10.peakFlopsBf16, "GB10 BF16 = 125 TFLOPs (1 PFLOP FP4 sparse → BF16 dense)")
        assertEquals(250e12, gb10.peakFlopsFp8)
        assertEquals(128L * 1024 * 1024 * 1024, gb10.hbmCapacityBytes, "GB10 ships with 128 GB unified")
        assertTrue(
            gb10.hbmBandwidthBytesPerSec < DeviceDescriptors.B100.hbmBandwidthBytesPerSec / 20.0,
            "LPDDR5X bandwidth is ~30× less than B100's HBM3e",
        )
    }

    @Test
    fun b100Citations() {
        val d = DeviceDescriptors.B100
        assertEquals("nvidia", d.vendor)
        assertEquals(1750e12, d.peakFlopsBf16, "B100 BF16 = 1750 TFLOPs (Blackwell whitepaper, dense)")
        assertEquals(3500e12, d.peakFlopsFp8, "B100 FP8 = 3500 TFLOPs dense")
        assertEquals(8e12, d.hbmBandwidthBytesPerSec, "B100/B200 share 8 TB/s HBM3e")
        assertEquals(192L * 1024 * 1024 * 1024, d.hbmCapacityBytes, "B100/B200 ship with 192 GB HBM3e")
        assertEquals(700, d.extras["tdp_watts"])
    }

    @Test
    fun b200Citations() {
        val d = DeviceDescriptors.B200
        assertEquals("nvidia", d.vendor)
        assertEquals(2250e12, d.peakFlopsBf16, "B200 BF16 = 2250 TFLOPs dense (Blackwell whitepaper)")
        assertEquals(4500e12, d.peakFlopsFp8, "B200 FP8 = 4500 TFLOPs dense")
        assertEquals(192L * 1024 * 1024 * 1024, d.hbmCapacityBytes)
        assertEquals(1000, d.extras["tdp_watts"], "B200 is the 1000W flagship; B100 is 700W")
    }

    @Test
    fun b200FasterThanB100ButSameMemorySubsystem() {
        // B100 and B200 share the same silicon + memory subsystem;
        // B200 just clocks higher (1000W TDP vs 700W). Compute scales
        // ~28% but bandwidth and capacity are identical.
        val b100 = DeviceDescriptors.B100
        val b200 = DeviceDescriptors.B200
        assertTrue(b200.peakFlopsBf16 > b100.peakFlopsBf16, "B200 BF16 > B100 BF16 (clock advantage)")
        assertTrue(b200.peakFlopsFp8!! > b100.peakFlopsFp8!!, "B200 FP8 > B100 FP8")
        assertEquals(b100.hbmBandwidthBytesPerSec, b200.hbmBandwidthBytesPerSec, "shared HBM3e bandwidth")
        assertEquals(b100.hbmCapacityBytes, b200.hbmCapacityBytes, "shared HBM3e capacity")
        assertEquals(b100.smOrCoreCount, b200.smOrCoreCount, "same silicon — same SM count")
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
