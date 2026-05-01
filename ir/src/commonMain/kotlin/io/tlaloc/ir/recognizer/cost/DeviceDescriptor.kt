package io.tlaloc.ir.recognizer.cost

/**
 * Layer 3 §0.4.254+ — device descriptor consumed by the L3.4 cost model.
 *
 * Captures the per-target peak compute (per dtype family) + memory
 * bandwidth + on-chip SRAM that the cost model needs to score
 * (kernel-call vs. decompose) decisions and (tiled vs. unfused)
 * scheduling. Numbers are sourced from vendor datasheets at the time of
 * v1; see the per-target citations below.
 *
 * # Why a fixed schema
 *
 * Scope honesty: full performance models would need per-tensor-core
 * sustained TFLOPs, per-cache-level latencies, NVLink/ICI bandwidth
 * across multiple chips, etc. This v1 schema picks the smallest set
 * that produces a reasonable roofline-style estimate (peak compute,
 * HBM bandwidth, on-chip SRAM size) for picking between a fused vendor
 * kernel (custom-call) and the decomposed primitive chain. Future
 * extensions add per-dtype mixed precision throughput, NVLink
 * bandwidth, etc.
 *
 * # Why not per-vendor types
 *
 * Per-vendor descriptors (NvidiaDevice, GoogleDevice) would force the
 * cost model to dispatch on type. A flat schema lets the model do
 * `device.peakFlopsBf16 / device.hbmBandwidth` uniformly. Vendor-
 * specific knobs (e.g., FP8 throughput, TPU sparsecore presence) live
 * in [extras] for now.
 *
 * @property name short stable identifier matching `KernelTarget.arch`
 *   (e.g. `"h100"`, `"tpu_v5e"`).
 * @property vendor lower-case vendor string (`"nvidia"`, `"google"`,
 *   `"amd"`, `"aws"`, `"tlaloc"`).
 * @property peakFlopsF32 sustained peak FLOP/s at FP32 precision.
 *   Includes tensor-core paths when applicable.
 * @property peakFlopsBf16 sustained peak FLOP/s at BF16 precision (or
 *   FP16; vendors typically publish a single number for both).
 * @property peakFlopsFp8 sustained peak FLOP/s at FP8 precision; `null`
 *   if the device doesn't support FP8 natively.
 * @property hbmBandwidthBytesPerSec aggregate HBM (or DDR for CPU)
 *   bandwidth in bytes/sec.
 * @property hbmCapacityBytes total HBM capacity per chip in bytes.
 * @property onChipSramBytes the relevant on-chip cache size — L2 for
 *   NVIDIA, VMEM for TPU, scratchpad for Trainium. Used by tile-fusion
 *   to size tiles.
 * @property smOrCoreCount the number of independent compute units
 *   (SMs on NVIDIA, MXU rows × cols on TPU folded into a single number
 *   for v1, scaled across cores on CPU). Used as a parallelism
 *   proxy.
 * @property extras free-form per-vendor knobs (e.g. `"sparsity_factor"
 *   to 2.0` on NVIDIA cards that publish sparse-tensor TFLOPs).
 */
data class DeviceDescriptor(
    val name: String,
    val vendor: String,
    val peakFlopsF32: Double,
    val peakFlopsBf16: Double,
    val peakFlopsFp8: Double?,
    val hbmBandwidthBytesPerSec: Double,
    val hbmCapacityBytes: Long,
    val onChipSramBytes: Long,
    val smOrCoreCount: Int,
    val extras: Map<String, Any> = emptyMap(),
) {
    /**
     * Roofline-style peak: the smaller of `compute_peak / op_size_in_flops`
     * and `bandwidth / op_size_in_bytes` time, for a single op. v1
     * convenience used by the cost model.
     */
    fun peakBf16FlopsPerByte(): Double = peakFlopsBf16 / hbmBandwidthBytesPerSec
}

/**
 * The seven canonical device descriptors L3.4 ships with. Sourced from
 * vendor datasheets — citations inline. Numbers reflect v1 (2026-05);
 * vendors may update sustained-perf claims independent of our release
 * cadence, which is fine — the cost model is a relative estimator, not
 * a benchmark.
 */
object DeviceDescriptors {

    /**
     * NVIDIA H100 SXM5 (Hopper).
     *
     * Source: NVIDIA H100 Tensor Core GPU datasheet, June 2023.
     * https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
     *
     * - FP32 (tensor core, no sparsity): 67 TFLOPs
     * - BF16/FP16 (tensor core, no sparsity): 989 TFLOPs
     * - FP8 (tensor core, no sparsity): 1979 TFLOPs
     * - HBM3: 3.35 TB/s, 80 GB
     * - L2: 50 MB
     * - SM count: 132
     */
    val H100 = DeviceDescriptor(
        name = "h100",
        vendor = "nvidia",
        peakFlopsF32 = 67e12,
        peakFlopsBf16 = 989e12,
        peakFlopsFp8 = 1979e12,
        hbmBandwidthBytesPerSec = 3.35e12,
        hbmCapacityBytes = 80L * 1024 * 1024 * 1024,
        onChipSramBytes = 50L * 1024 * 1024,
        smOrCoreCount = 132,
        extras = mapOf("sparsity_factor" to 2.0),
    )

    /**
     * NVIDIA A100 SXM4 80GB (Ampere).
     *
     * Source: NVIDIA A100 Tensor Core GPU datasheet (rev May 2021).
     * https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
     *
     * - FP32: 19.5 TFLOPs (TF32 tensor core: 156 TFLOPs)
     * - BF16/FP16 tensor core: 312 TFLOPs
     * - HBM2e: 1.94 TB/s, 80 GB
     * - L2: 40 MB
     * - SM count: 108
     * - No native FP8.
     */
    val A100 = DeviceDescriptor(
        name = "a100",
        vendor = "nvidia",
        peakFlopsF32 = 19.5e12,
        peakFlopsBf16 = 312e12,
        peakFlopsFp8 = null,
        hbmBandwidthBytesPerSec = 1.94e12,
        hbmCapacityBytes = 80L * 1024 * 1024 * 1024,
        onChipSramBytes = 40L * 1024 * 1024,
        smOrCoreCount = 108,
        extras = mapOf("sparsity_factor" to 2.0, "tf32_peak" to 156e12),
    )

    /**
     * Google TPU v4 (single chip).
     *
     * Source: "TPU v4: An Optically Reconfigurable Supercomputer for
     * Machine Learning with Hardware Support for Embeddings"
     * (Jouppi et al., ISCA 2023).
     * https://arxiv.org/abs/2304.01433
     *
     * - BF16 MXU peak: 275 TFLOPs/chip
     * - Int8: 275 TOPs/chip
     * - HBM: 1.2 TB/s, 32 GiB per chip
     * - VMEM (on-chip SRAM): 128 MiB
     * - 2 TensorCores per chip; per-TC MXU is 128×128.
     */
    val TPU_V4 = DeviceDescriptor(
        name = "tpu_v4",
        vendor = "google",
        peakFlopsF32 = 275e12 / 2.0,  // F32 not native; rough fall-back
        peakFlopsBf16 = 275e12,
        peakFlopsFp8 = null,
        hbmBandwidthBytesPerSec = 1.2e12,
        hbmCapacityBytes = 32L * 1024 * 1024 * 1024,
        onChipSramBytes = 128L * 1024 * 1024,
        smOrCoreCount = 2,  // two TensorCores per chip
        extras = mapOf("ici_bandwidth_gbps" to 50.0),
    )

    /**
     * Google TPU v5e (single chip; cost-optimized inference variant).
     *
     * Source: Google Cloud "TPU v5e" announcement; "TPU v5e: cost
     * efficiency vs. perf vs. v4" cloud doc.
     * https://cloud.google.com/tpu/docs/v5e
     *
     * - BF16 MXU peak: 197 TFLOPs/chip
     * - HBM: 819 GB/s, 16 GiB per chip
     * - VMEM: 48 MiB
     * - One TensorCore per chip.
     */
    val TPU_V5E = DeviceDescriptor(
        name = "tpu_v5e",
        vendor = "google",
        peakFlopsF32 = 197e12 / 2.0,
        peakFlopsBf16 = 197e12,
        peakFlopsFp8 = null,
        hbmBandwidthBytesPerSec = 819e9,
        hbmCapacityBytes = 16L * 1024 * 1024 * 1024,
        onChipSramBytes = 48L * 1024 * 1024,
        smOrCoreCount = 1,
    )

    /**
     * Google TPU v6e ("Trillium"; single chip; training+inference).
     *
     * Source: Google Cloud "Trillium" TPU announcement (May 2024) +
     * follow-up cloud docs.
     * https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
     *
     * - BF16 MXU peak: ~926 TFLOPs/chip (4.7× v5e per public claims)
     * - Int8 peak: ~1850 TOPs/chip
     * - HBM: 1.6 TB/s, 32 GiB per chip
     * - VMEM: 192 MiB
     */
    val TPU_V6E = DeviceDescriptor(
        name = "tpu_v6e",
        vendor = "google",
        peakFlopsF32 = 926e12 / 2.0,
        peakFlopsBf16 = 926e12,
        peakFlopsFp8 = 1850e12,  // int8 throughput cited via fp8-equivalent
        hbmBandwidthBytesPerSec = 1.6e12,
        hbmCapacityBytes = 32L * 1024 * 1024 * 1024,
        onChipSramBytes = 192L * 1024 * 1024,
        smOrCoreCount = 1,
        extras = mapOf("sparsecore_present" to true),
    )

    /**
     * AWS Trainium2 (single chip).
     *
     * Source: AWS re:Invent 2024 "Trainium2 Performance" session;
     * https://aws.amazon.com/ai/machine-learning/trainium/
     *
     * - BF16 peak: ~660 TFLOPs/chip
     * - FP8 peak: ~1300 TFLOPs/chip
     * - HBM: 2.9 TB/s, 96 GiB per chip
     * - On-chip SBUF (closest analog to L2/VMEM): 96 MiB.
     * - 2 NeuronCore-v3 per chip, each with 16 MMA engines.
     */
    val TRAINIUM2 = DeviceDescriptor(
        name = "trainium2",
        vendor = "aws",
        peakFlopsF32 = 660e12 / 4.0,  // approximate; AWS publishes BF16+FP8 primarily
        peakFlopsBf16 = 660e12,
        peakFlopsFp8 = 1300e12,
        hbmBandwidthBytesPerSec = 2.9e12,
        hbmCapacityBytes = 96L * 1024 * 1024 * 1024,
        onChipSramBytes = 96L * 1024 * 1024,
        smOrCoreCount = 2,
        extras = mapOf("neuronlink_v3_bw_gbps" to 1280.0),
    )

    /**
     * Generic CPU placeholder. Sufficient for "the always-available
     * decompose-only target." Numbers are an order-of-magnitude
     * approximation for a modern server CPU (e.g. Xeon Sapphire Rapids
     * 56-core, 350W).
     *
     * Source: rough roofline for a contemporary server CPU; not tied
     * to a single SKU.
     */
    val CPU_GENERIC = DeviceDescriptor(
        name = "cpu_generic",
        vendor = "tlaloc",
        peakFlopsF32 = 5e12,        // ~5 TFLOPs FP32 with AVX-512 + AMX
        peakFlopsBf16 = 20e12,      // AMX BF16 peak
        peakFlopsFp8 = null,
        hbmBandwidthBytesPerSec = 300e9,  // DDR5 quad-channel ~300 GB/s
        hbmCapacityBytes = 256L * 1024 * 1024 * 1024,  // typical server RAM
        onChipSramBytes = 105L * 1024 * 1024,  // ~L3 cache
        smOrCoreCount = 56,
        extras = mapOf("amx_enabled" to true),
    )

    /** All seven canonical descriptors, in declaration order. */
    val all: List<DeviceDescriptor> = listOf(H100, A100, TPU_V4, TPU_V5E, TPU_V6E, TRAINIUM2, CPU_GENERIC)

    /** Lookup by name (matches `KernelTarget.arch`); null if not in v1. */
    fun byName(name: String): DeviceDescriptor? = all.firstOrNull { it.name == name }
}
