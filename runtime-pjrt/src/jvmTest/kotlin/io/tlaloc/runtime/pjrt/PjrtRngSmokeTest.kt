package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.normalFloats
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
 * §0.4.422 — explicit-threefry emission certified against real XLA on the
 * GB10. The claim structure is deliberately two-tier:
 *
 * 1. **Uniform draws are BIT-EXACT** against `:core/Random.kt`'s
 *    [uniformFloats] — the emitted graph is static integer ARX ops plus the
 *    bitcast mantissa trick and one exact subtract, so there is nothing that
 *    can round. Asserted on raw f32 bits, even/odd/rank-2 (the odd case
 *    exercises the end-pad counter lane and the tail slice).
 * 2. **Normal draws agree at tolerance, never bit-pinned**: the Box-Muller
 *    formula runs in the same f64 intermediates as the host, but f64
 *    log/cos are library calls that legitimately differ at ≤1 ulp between
 *    the JVM and libdevice — the BITS layer is exact, the transcendental
 *    narrowing is not, and pinning bits there would be pinning someone
 *    else's libm.
 *
 * Plus the D2 story on GPU: a reparameterized loss's GRADIENT graph contains
 * a cloned draw (the d-scale adjoint reads ε), and it now compiles and runs
 * on the GPU, matching the interpreter — the reason
 * GradientEmissionCoverageTest's RNG exclusion lifts in this same slice.
 */
class PjrtRngSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    private fun uniformFn(k0: Int, k1: Int, dims: List<Int>) =
        DxirBuilder.function("rng_u") {
            listOf(
                op(
                    OpKind.RNG_UNIFORM, emptyList(), DxirType(F32, dims),
                    attrs = mapOf("key0" to k0, "key1" to k1, "dims" to dims),
                ),
            )
        }

    @Test
    fun uniformDrawsAreBitExactOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            // (even, odd — the end-pad lane, rank-2) × a key with high bits set.
            for ((dims, key) in listOf(
                listOf(6) to RandomKey(42, 7),
                listOf(5) to RandomKey(-1, 12345),
                listOf(2, 3) to RandomKey(0x13198A2E, 0x03707344),
            )) {
                val n = dims.fold(1) { a, d -> a * d }
                val want = uniformFloats(key, n)
                val got = session.runOn(uniformFn(key.k0, key.k1, dims), emptyList()).single()
                assertEquals(n, got.size, "draw size for dims=$dims")
                for (i in 0 until n) {
                    assertTrue(
                        got[i].toRawBits() == want[i].toRawBits(),
                        "uniform dims=$dims key=$key lane $i: GPU ${got[i]} " +
                            "(0x${got[i].toRawBits().toUInt().toString(16)}) != host ${want[i]} " +
                            "(0x${want[i].toRawBits().toUInt().toString(16)}) — the bit stream forked",
                    )
                }
            }
            println("[pjrt-rng] uniform draws BIT-EXACT vs host threefry on GB10 (even/odd/rank-2)")
        }
    }

    @Test
    fun normalDrawsAgreeWithinUlpsOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val dims = listOf(4)
        val key = RandomKey(42, 7)
        val fn = DxirBuilder.function("rng_n") {
            listOf(
                op(
                    OpKind.RNG_NORMAL, emptyList(), DxirType(F32, dims),
                    attrs = mapOf("key0" to key.k0, "key1" to key.k1, "dims" to dims),
                ),
            )
        }
        val want = normalFloats(key, 4)
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(fn, emptyList()).single()
            var maxDiff = 0f
            for (i in 0 until 4) maxDiff = maxOf(maxDiff, abs(got[i] - want[i]))
            println("[pjrt-rng] normal max|diff|=$maxDiff vs host Box-Muller on GB10")
            assertTrue(maxDiff <= 1e-5f, "normal draws diverge beyond libm ulps: $maxDiff")
        }
    }

    @Test
    fun runtimeKeyOperandUniformBitExactOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // §0.4.432 — the runtime-key operand form with keys as EXECUTABLE
        // INPUTS: f32 scalar params CAST to i32 in-graph (PjrtSession's
        // host-buffer lane is F32-only v1 — an i32 lane is the recorded
        // tail), so XLA cannot constant-fold the key schedule: the emitted
        // broadcast/xor/add key path actually runs on device. Keys stay
        // strictly below 2^24 so the f32 input lane carries them exactly.
        // Bit-exact vs the host kernel: integer ops cannot round, whichever
        // engine computes them. Odd dims exercise the end-pad lane through
        // the runtime-key path too.
        val keyF = DxirType(F32, emptyList())
        val keyI = DxirType(I32, emptyList())
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for ((dims, key) in listOf(
                listOf(6) to RandomKey(42, 7),
                listOf(5) to RandomKey(1234567, 891011),
            )) {
                val n = dims.fold(1) { a, d -> a * d }
                val fn = DxirBuilder.function("rng_rtk") {
                    val k0f = param("k0", keyF)
                    val k1f = param("k1", keyF)
                    val k0 = op(OpKind.CAST, listOf(k0f), keyI)
                    val k1 = op(OpKind.CAST, listOf(k1f), keyI)
                    listOf(
                        op(
                            OpKind.RNG_UNIFORM, listOf(k0, k1), DxirType(F32, dims),
                            attrs = mapOf("dims" to dims),
                        ),
                    )
                }
                val want = uniformFloats(key, n)
                val got = session.runOn(
                    fn,
                    listOf(floatArrayOf(key.k0.toFloat()), floatArrayOf(key.k1.toFloat())),
                ).single()
                for (i in 0 until n) {
                    assertTrue(
                        got[i].toRawBits() == want[i].toRawBits(),
                        "runtime-key uniform dims=$dims key=$key lane $i: GPU ${got[i]} " +
                            "(0x${got[i].toRawBits().toUInt().toString(16)}) != host ${want[i]} " +
                            "(0x${want[i].toRawBits().toUInt().toString(16)}) — the bit stream forked",
                    )
                }
            }
            println("[pjrt-rng] runtime-key uniform draws BIT-EXACT vs host threefry on GB10")
        }
    }

    @Test
    fun constOperandHighBitKeysBitExactOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // §0.4.432 — high-bit key words cannot ride the F32 input lane; as
        // scalar I32 CONST operands they still exercise the operand-form
        // emission (rank-0 broadcasts + the EMITTED key schedule — XLA may
        // fold it, but the graph's integer math must be right either way)
        // and must reproduce the host stream bit-for-bit.
        val keyI = DxirType(I32, emptyList())
        val key = RandomKey(0x7FFFFFFF, -0x1235ABCD)
        val dims = listOf(6)
        val fn = DxirBuilder.function("rng_hbk") {
            val k0 = const(key.k0, keyI)
            val k1 = const(key.k1, keyI)
            listOf(
                op(
                    OpKind.RNG_UNIFORM, listOf(k0, k1), DxirType(F32, dims),
                    attrs = mapOf("dims" to dims),
                ),
            )
        }
        val want = uniformFloats(key, 6)
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(fn, emptyList()).single()
            for (i in 0 until 6) {
                assertTrue(
                    got[i].toRawBits() == want[i].toRawBits(),
                    "high-bit const-key uniform lane $i: GPU ${got[i]} != host ${want[i]} — " +
                        "the bit stream forked",
                )
            }
            println("[pjrt-rng] high-bit const-operand keys BIT-EXACT vs host threefry on GB10")
        }
    }

    @Test
    fun reparameterizedGradientGraphRunsOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // loss = Σ (loc + scale ⊙ ε)², ε = normal(42, 7)[4]. The gradient
        // body CLONES the draw (d scale = upstream ⊙ ε), so its GPU graph
        // contains the full threefry block — the case the emission exists for.
        val vT = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("reparam_loss") {
            val loc = param("loc", vT)
            val scale = param("scale", vT)
            val eps = op(
                OpKind.RNG_NORMAL, emptyList(), vT,
                attrs = mapOf("key0" to 42, "key1" to 7, "dims" to listOf(4)),
            )
            val se = op(OpKind.MUL, listOf(scale, eps), vT)
            val z = op(OpKind.ADD, listOf(loc, se), vT)
            val z2 = op(OpKind.MUL, listOf(z, z), vT)
            listOf(op(OpKind.SUM, listOf(z2), scalar))
        }
        val loc = floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f)
        val scale = floatArrayOf(1.5f, 0.5f, -0.75f, 1.0f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val grad = DxirReverseTransform.apply(fn)
            val want = DxirInterpreter.evalFunction(grad, listOf(loc, scale))
            val got = session.runOn(grad, listOf(loc, scale))
            var maxDiff = 0f
            for (k in want.indices) {
                for (i in want[k].indices) maxDiff = maxOf(maxDiff, abs(got[k][i] - want[k][i]))
            }
            println("[pjrt-rng] reparameterized gradient max|diff|=$maxDiff vs interpreter on GB10")
            assertTrue(maxDiff <= 1e-4f, "reparameterized gradient diverges: $maxDiff")
        }
    }
}
