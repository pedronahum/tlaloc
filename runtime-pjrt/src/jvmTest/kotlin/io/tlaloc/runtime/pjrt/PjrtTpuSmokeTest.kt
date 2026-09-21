package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.459 (G2a) — the TPU smoke suite, written NOW and SELF-SKIPPING
 * until a TPU is under it. On this GB10 every test in this class skips at
 * the [assumeTpu] gate (no libtpu resolves) — that skip-clean behaviour
 * IS the local certification. Running these green on real hardware is
 * G2b's claim, made from a Cloud TPU VM per docs/TPU_BRINGUP.md, and no
 * TPU-execution claim is made here.
 *
 * The claim structure, mirroring the CUDA lanes it re-targets:
 *
 *   - **threefry bit-exactness** ([threefryUniformBitExactOnTpu]) — the
 *     flagship G2b claim, mirroring [PjrtRngSmokeTest]: the §0.4.422
 *     explicit-threefry emission is pure StableHLO integer ARX + one
 *     exact subtract, TPU-portable BY CONSTRUCTION — if the bit stream
 *     forks on TPU, the portability argument was wrong and we want the
 *     raw-bits counterexample.
 *   - **a gradient graph vs the interpreter**
 *     ([matmulGradMatchesInterpreterOnTpu]) — DxirReverseTransform's
 *     output compiling and matching on a second real backend.
 *   - **the G1c bf16 claims re-targeted** (the two bf16 tests) — device
 *     RNE narrowing vs the §0.4.455 helper, and native-bf16-buffer
 *     compute on bf16-exact lanes. The §0.4.457 CUDA findings
 *     (convert-pair folding, per-op-boundary rounding) are XLA-CUDA
 *     measurements that may legitimately differ on TPU; the single-convert
 *     probe is the honest shape on any backend. TPU tiled buffer layout
 *     is UNVERIFIED — device size is reported, bounded below, never
 *     pinned equal (unlike the CUDA lane's 2-byte pin).
 *
 * Named deferral: these bf16 lanes restate the G1c claims as fresh tests
 * rather than parametrizing [PjrtBf16SmokeTest] itself — a shared
 * multi-backend smoke harness is future work, and duplicating the probe
 * sweep beats a refactor of a certified suite inside this slice.
 */
class PjrtTpuSmokeTest {

    private fun assumeTpu() {
        assumeTrue(
            PjrtBinaries.tpuAvailable,
            "no TPU PJRT plugin resolved (libtpu.so) — skipping; see docs/TPU_BRINGUP.md.",
        )
    }

    private fun tpuSession(): PjrtSession =
        PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu)

    private val scalar = DxirType(F32, emptyList())

    /** The plugin that resolved must actually be a TPU: `"tpu"` from
     * `PJRT_Client_PlatformName`, asserted (not assumed) so a mis-resolved
     * plugin fails loudly instead of certifying the wrong backend. */
    @Test
    fun platformNameReportsTpu() {
        assumeTpu()
        tpuSession().use { session ->
            val name = session.platformName()
            println("[pjrt-tpu] platform_name='$name' plugin=${PjrtBinaries.tpuPluginPath}")
            assertEquals("tpu", name.lowercase(), "resolved plugin is not a TPU plugin")
        }
    }

    @Test
    fun threefryUniformBitExactOnTpu() {
        assumeTpu()
        tpuSession().use { session ->
            // The PjrtRngSmokeTest sweep: (even, odd — the end-pad lane,
            // rank-2) × a key with high bits set.
            for ((dims, key) in listOf(
                listOf(6) to RandomKey(42, 7),
                listOf(5) to RandomKey(-1, 12345),
                listOf(2, 3) to RandomKey(0x13198A2E, 0x03707344),
            )) {
                val n = dims.fold(1) { a, d -> a * d }
                val fn = DxirBuilder.function("rng_u_tpu") {
                    listOf(
                        op(
                            OpKind.RNG_UNIFORM, emptyList(), DxirType(F32, dims),
                            attrs = mapOf("key0" to key.k0, "key1" to key.k1, "dims" to dims),
                        ),
                    )
                }
                val want = uniformFloats(key, n)
                val got = session.runOn(fn, emptyList()).single()
                assertEquals(n, got.size, "draw size for dims=$dims")
                for (i in 0 until n) {
                    assertTrue(
                        got[i].toRawBits() == want[i].toRawBits(),
                        "uniform dims=$dims key=$key lane $i: TPU ${got[i]} " +
                            "(0x${got[i].toRawBits().toUInt().toString(16)}) != host ${want[i]} " +
                            "(0x${want[i].toRawBits().toUInt().toString(16)}) — the bit stream forked",
                    )
                }
            }
            println("[pjrt-tpu] uniform draws BIT-EXACT vs host threefry on TPU (even/odd/rank-2)")
        }
    }

    @Test
    fun matmulGradMatchesInterpreterOnTpu() {
        assumeTpu()
        // loss = Σ (A·B): the gradient graph DxirReverseTransform emits
        // (broadcast-of-ones matmuls with transposes) compiling and running
        // on a second real backend, matched against the interpreter.
        val aT = DxirType(F32, listOf(2, 3))
        val bT = DxirType(F32, listOf(3, 2))
        val cT = DxirType(F32, listOf(2, 2))
        val fn = DxirBuilder.function("mm_loss_tpu") {
            val a = param("a", aT)
            val b = param("b", bT)
            val mm = op(OpKind.MATMUL, listOf(a, b), cT)
            listOf(op(OpKind.SUM, listOf(mm), scalar))
        }
        val a = floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.5f, 1.5f)
        val b = floatArrayOf(1.5f, 0.5f, -0.75f, 1.0f, 0.25f, -2.0f)

        tpuSession().use { session ->
            val grad = DxirReverseTransform.apply(fn)
            val want = DxirInterpreter.evalFunction(grad, listOf(a, b))
            val got = session.runOn(grad, listOf(a, b))
            var maxDiff = 0f
            for (k in want.indices) {
                for (i in want[k].indices) maxDiff = maxOf(maxDiff, abs(got[k][i] - want[k][i]))
            }
            println("[pjrt-tpu] matmul gradient max|diff|=$maxDiff vs interpreter on TPU")
            assertTrue(maxDiff <= 1e-4f, "matmul gradient diverges on TPU: $maxDiff")
        }
    }

    /** The G1c narrowing claim on TPU: device f32→bf16 vs the §0.4.455 RNE
     * helper, probed through the dtype-agnostic executeOn lane with a
     * SINGLE convert (the honest probe shape on any backend — the CUDA
     * lane measured XLA folding a convert round-trip pair to identity). */
    @Test
    fun bf16DeviceNarrowingBitMatchesHostRneOnTpu() {
        assumeTpu()
        // The G1a pin sweep: RNE ties both directions, one-ulp neighbors,
        // signed zeros, infinities, overflow, subnormal flushes.
        val probes = floatArrayOf(
            Float.fromBits(0x3F808000),          // tie, even keeps → 1.0
            Float.fromBits(0x3F818000),          // tie, odd keeps → up to even
            Float.fromBits(0x3F808001),          // just above tie → up
            Float.fromBits(0x3F817FFF),          // just below tie → down
            Float.fromBits(0xBF808000.toInt()),  // negative tie → down
            Float.fromBits(0xBF818000.toInt()),  // negative tie → up to even
            0f, -0f,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.MAX_VALUE,                     // RNE overflow → +inf
            Float.fromBits(0x7F7F7FFF),          // below overflow tie → max finite
            Float.fromBits(0x00000001),          // min f32 subnormal → +0
            Float.fromBits(0x80000001.toInt()),  // sign survives the flush
            1.5f, 256f, 1.0078125f, 3.14159265f, -257f,
        )
        val n = probes.size
        val fn = DxirBuilder.function("bf16_narrow_probe_tpu") {
            val x = param("x", DxirType(F32, listOf(n)))
            listOf(op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(n))))
        }
        tpuSession().use { session ->
            val staged = session.bufferFromHostF32(probes, listOf(n))
            val got: ShortArray
            try {
                val outs = session.executeOn(fn, listOf(staged))
                try {
                    got = outs.single().toBf16Array(n)
                } finally {
                    outs.forEach { it.close() }
                }
            } finally {
                staged.close()
            }
            for (i in probes.indices) {
                val want = floatToBf16Bits(probes[i])
                assertTrue(
                    got[i] == want,
                    "narrow lane $i (probe bits 0x${probes[i].toRawBits().toUInt().toString(16)}): " +
                        "TPU pattern 0x${got[i].toUShort().toString(16)} != host RNE " +
                        "0x${want.toUShort().toString(16)} — device narrowing disagrees with §0.4.455",
                )
            }
            println("[pjrt-tpu] device f32→bf16 narrowing BIT-EXACT vs G1a RNE on TPU ($n lanes incl. ties)")
        }
    }

    /** Native bf16 buffers computing on TPU: a matmul whose inputs, products
     * and sums are all bf16-exact integers, so the result patterns are
     * pinned bit-for-bit. Device buffer size is REPORTED and bounded below
     * only — TPU tiled layout may legitimately pad past 2 bytes/element
     * (the G1c open item), so no equality pin here. */
    @Test
    fun bf16NativeBuffersMatmulBitExactOnTpu() {
        assumeTpu()
        val bT = DxirType(BF16, listOf(2, 2))
        val fn = DxirBuilder.function("bf16_mm_tpu") {
            val a = param("a", bT)
            val b = param("b", bT)
            listOf(op(OpKind.MATMUL, listOf(a, b), bT))
        }
        val aVals = floatArrayOf(1f, 2f, 3f, 4f)
        val bVals = floatArrayOf(5f, 6f, 7f, 8f)
        val expect = floatArrayOf(19f, 22f, 43f, 50f)  // all bf16-exact
        tpuSession().use { session ->
            session.bufferFromHostBf16(floatArrayToBf16Bits(aVals), listOf(2, 2)).use { staged ->
                val deviceBytes = staged.deviceSizeInBytes()
                println("[pjrt-tpu] bf16 2x2 on-device size = $deviceBytes bytes (tiled layout unpinned)")
                assertTrue(deviceBytes >= 2L * 4, "device buffer smaller than 2 bytes/element")
            }
            val got = session.runOnBf16(
                fn,
                listOf(floatArrayToBf16Bits(aVals), floatArrayToBf16Bits(bVals)),
            ).single()
            val want = floatArrayToBf16Bits(expect)
            for (i in expect.indices) {
                assertTrue(
                    got[i] == want[i],
                    "bf16 matmul lane $i: TPU pattern 0x${got[i].toUShort().toString(16)} != " +
                        "expected 0x${want[i].toUShort().toString(16)} (all lanes bf16-exact)",
                )
            }
            println("[pjrt-tpu] native bf16 matmul BIT-EXACT on bf16-exact lanes on TPU")
        }
    }
}
