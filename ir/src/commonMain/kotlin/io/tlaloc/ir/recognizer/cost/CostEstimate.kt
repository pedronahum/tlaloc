package io.tlaloc.ir.recognizer.cost

import io.tlaloc.core.ExperimentalTlalocApi
import kotlin.math.max

/**
 * Cost estimate for an op or function.
 *
 * @property flops total floating-point operations.
 * @property bytesMoved total bytes touched (input read + output written).
 *   For fused regions (e.g. a kernel custom-call), only operand inputs +
 *   final output count; intermediate bytes are elided.
 */
@ExperimentalTlalocApi
data class CostEstimate(
    val flops: Double,
    val bytesMoved: Double,
) {
    /**
     * FLOPs per byte — used to compare against a device's
     * [DeviceDescriptor.peakBf16FlopsPerByte]. Above the device's knee
     * → compute-bound; below → memory-bound.
     */
    val arithmeticIntensity: Double
        get() = if (bytesMoved > 0.0) flops / bytesMoved else Double.POSITIVE_INFINITY

    operator fun plus(other: CostEstimate): CostEstimate =
        CostEstimate(flops + other.flops, bytesMoved + other.bytesMoved)

    /**
     * Roofline-style time estimate in seconds for this estimate on
     * [device], using [peakFlopsForDtype] as the relevant compute peak
     * (caller picks F32 vs BF16 vs FP8 based on the op dtype).
     *
     * `time = max(flops / peak, bytes / bandwidth)` — the workload is
     * bottlenecked by whichever of the two is larger. This is the
     * canonical roofline reading; refinements (overlap, latency,
     * cache-hit-rate, NVLink overhead) are not modelled.
     */
    fun roofineSeconds(device: DeviceDescriptor, peakFlopsForDtype: Double): Double {
        val computeTime = if (peakFlopsForDtype > 0.0) flops / peakFlopsForDtype else Double.POSITIVE_INFINITY
        val memoryTime = bytesMoved / device.hbmBandwidthBytesPerSec
        return max(computeTime, memoryTime)
    }

    companion object {
        val ZERO: CostEstimate = CostEstimate(flops = 0.0, bytesMoved = 0.0)
    }
}
