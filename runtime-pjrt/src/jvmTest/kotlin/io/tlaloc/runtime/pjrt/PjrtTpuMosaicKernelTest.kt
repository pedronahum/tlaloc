package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.bf16BitsToFloatArray
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Mosaic payloads executed on a TPU. Skips without libtpu (see
 * docs/TPU_BRINGUP.md); written and compiled on a host without a TPU, never
 * run there.
 *
 * For every payload in `resources/tpu-kernels/`, one PjrtSession on the TPU
 * runs the same DXIR program two ways:
 *
 * - **custom call**: the MOSAIC_KERNEL claimed for the TPU, i.e.
 *   `stablehlo.custom_call @tpu_custom_call` carrying the Mosaic body;
 * - **XLA**: the op's reference decomposition, the plain StableHLO path
 *   (lowered with [KernelTarget.CPU_GENERIC] so the reference is inlined).
 *
 * Both are compared with the numpy reference the exporter wrote, and the
 * median time of one dispatch of each (inputs staged once, output read back
 * not included) is printed.
 *
 * Tolerances: the custom call within max(manifest tolerance, 2e-4 x max|ref|)
 * (Mosaic's rsqrt and fp32 matmul passes are not bit-identical to numpy);
 * the XLA path within max(tolerance, 2e-2 x max|ref|), because an f32
 * `stablehlo.dot_general` without a precision config runs as one bf16 pass on
 * a TPU. The in-place kv update copies bits and must match exactly.
 */
class PjrtTpuMosaicKernelTest {

    private fun assumeTpu() {
        assumeTrue(
            PjrtBinaries.tpuAvailable,
            "no TPU PJRT plugin resolved (libtpu.so) — skipping; see docs/TPU_BRINGUP.md.",
        )
    }

    private fun tpuSession(): PjrtSession =
        PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu)

    private val warmup = 10
    private val iterations = 100

    @Test fun rmsnormF32() = runFixture("rmsnorm_f32")
    @Test fun rmsnormBf16() = runFixture("rmsnorm_bf16")
    @Test fun matmulF32() = runFixture("matmul_f32")
    @Test fun normSwigluF32TwoResults() = runFixture("norm_swiglu_f32")
    @Test fun kvUpdateInPlaceF32() = runFixture("kv_update_inplace_f32")
    @Test fun rmsnormF32KotlinEmitted() = runFixture("rmsnorm_f32_kmosaic")
    @Test fun rmsnormBf16KotlinEmitted() = runFixture("rmsnorm_bf16_kmosaic")

    private fun runFixture(name: String) {
        assumeTpu()
        val f = TpuKernelFixtures.load(name)
        val custom = f.program(referenceFallback = true)
        val xla = lowerMosaicKernels(custom, KernelTarget.CPU_GENERIC)
        val bf16 = f.inputs.first().dtype == BF16
        tpuSession().use { session ->
            assertEquals("tpu", session.platformName().lowercase(), "resolved plugin is not a TPU plugin")
            val inputs = f.inputs.map { it.values() }
            val gotCustom = run(session, custom, inputs, bf16)
            val gotXla = run(session, xla, inputs, bf16)
            for (i in f.expected.indices) {
                val ref = f.expected[i]
                val scale = TpuKernelFixtures.maxAbs(ref)
                // A copy (tolerance 0) must be exact on both paths.
                val exact = f.tolerance == 0.0
                val customTol = if (exact) 0.0 else maxOf(f.tolerance, 2e-4 * scale)
                val xlaTol = if (exact) 0.0 else maxOf(f.tolerance, 2e-2 * scale)
                val dCustom = TpuKernelFixtures.maxAbsDiff(gotCustom[i], ref)
                val dXla = TpuKernelFixtures.maxAbsDiff(gotXla[i], ref)
                val dBoth = TpuKernelFixtures.maxAbsDiff(gotCustom[i], gotXla[i])
                println(
                    "[tpu-mosaic] $name result $i: max|ref|=$scale custom-vs-numpy=$dCustom (tol $customTol) " +
                        "xla-vs-numpy=$dXla (tol $xlaTol) custom-vs-xla=$dBoth",
                )
                assertTrue(dCustom <= customTol, "$name result $i: tpu_custom_call differs from numpy by $dCustom")
                assertTrue(dXla <= xlaTol, "$name result $i: the XLA path differs from numpy by $dXla")
            }
            val tCustom = medianMicros(session, custom, inputs, bf16, "$name/custom")
            val tXla = medianMicros(session, xla, inputs, bf16, "$name/xla")
            println(
                "[tpu-mosaic] $name timing (median of $iterations dispatches): custom_call ${fmt(tCustom)} us, " +
                    "xla ${fmt(tXla)} us, xla/custom ${fmt(tXla / tCustom)}x; payload " +
                    "${f.source}, stable_mosaic.version ${f.manifest["stable_mosaic_version"]}, " +
                    "jaxlib ${f.versions["jaxlib"]}",
            )
        }
    }

    private fun run(session: PjrtSession, fn: DxirFunction, inputs: List<FloatArray>, bf16: Boolean): List<FloatArray> =
        if (bf16) {
            session.runOnBf16(fn, inputs.map { floatArrayToBf16Bits(it) }).map { bf16BitsToFloatArray(it) }
        } else {
            session.runOn(fn, inputs)
        }

    private fun stage(session: PjrtSession, fn: DxirFunction, inputs: List<FloatArray>, bf16: Boolean): List<PjrtBuffer> =
        fn.params.zip(inputs).map { (p, v) ->
            if (bf16) session.bufferFromHostBf16(floatArrayToBf16Bits(v), p.type.dims)
            else session.bufferFromHostF32(v, p.type.dims)
        }

    private fun medianMicros(
        session: PjrtSession,
        fn: DxirFunction,
        inputs: List<FloatArray>,
        bf16: Boolean,
        key: String,
    ): Double {
        val staged = stage(session, fn, inputs, bf16)
        try {
            session.prepare(fn, cacheKey = key)
            repeat(warmup) { session.executeOn(fn, staged, cacheKey = key).forEach { it.close() } }
            val samples = DoubleArray(iterations) {
                val t0 = System.nanoTime()
                val outs = session.executeOn(fn, staged, cacheKey = key)
                val dt = (System.nanoTime() - t0) / 1e3
                outs.forEach { it.close() }
                dt
            }
            samples.sort()
            return samples[iterations / 2]
        } finally {
            staged.forEach { it.close() }
        }
    }

    private fun fmt(v: Double): String = "%.1f".format(v)
}
