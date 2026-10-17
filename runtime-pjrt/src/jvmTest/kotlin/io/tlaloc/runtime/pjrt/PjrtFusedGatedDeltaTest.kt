package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.GatedDeltaRuleAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * GATED_DELTA_RULE as the CUDA kernel `tlaloc_gated_delta` (libtlaloc_kernels.so)
 * against the interpreter: a row with padding first, a row starting at position
 * 0, a row continuing its slot; the state written back to the row's slot, and
 * per token at writeSlots; and with a bf16 pool, through the kernel and the
 * XLA forms.
 */
class PjrtFusedGatedDeltaTest {

    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtFusedGatedDeltaTest"), library)
        return true
    }

    private val b = 3; private val t = 3; private val hk = 2; private val hv = 4; private val dk = 32; private val dv = 64; private val s = 12

    /** The rule on a pool of [poolDtype]; the state param and result are f32 (a bf16 pool is cast in and out). */
    private fun fn(fused: Boolean, writes: Boolean, poolDtype: DType = F32): DxirFunction = DxirBuilder.function("gdr") {
        val qT = DxirType(F32, listOf(b, t, hk, dk))
        val vT = DxirType(F32, listOf(b, t, hv, dv))
        val gT = DxirType(F32, listOf(b, t, hv))
        val sT = DxirType(F32, listOf(s, hv, dk, dv))
        val pT = DxirType(poolDtype, sT.dims)
        val iT = DxirType(I32, listOf(b, t))
        val ps = listOf(param("q", qT), param("k", qT), param("v", vT), param("g", gT), param("beta", gT), param("state", sT),
            param("slots", iT), param("positions", iT)) + if (writes) listOf(param("writes", iT)) else emptyList()
        val ins = ps.mapIndexed { i, x -> if (i == 5 && poolDtype != F32) op(OpKind.CAST, listOf(x), pT) else x }
        val attrs: Map<String, Any> = if (fused) mapOf(GatedDeltaRuleAttrs.FUSED_KERNEL to true) else emptyMap()
        val r = opMulti(OpKind.GATED_DELTA_RULE, ins, listOf(vT, pT), attrs)
        listOf(r.result(0), if (poolDtype == F32) r.result(1) else op(OpKind.CAST, listOf(r.result(1)), sT))
    }

    @Test
    fun theKernelRunsTheRecurrenceAsTheInterpreter() = check(F32)

    /**
     * A bf16 pool: the kernel and the XLA forms widen each row's state, carry
     * it in f32 and round it when written, as the interpreter does. The
     * outputs agree as closely as with an f32 pool; a written state may differ
     * by one bf16 step where the f32 sums before rounding differ.
     */
    @Test
    fun aBf16PoolRunsAsTheInterpreter() = check(BF16)

    private fun check(poolDtype: DType) {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        val rnd = Random(12)
        fun f(n: Int, scale: Float, shift: Float = 0f) = FloatArray(n) { shift + scale * (rnd.nextFloat() * 2 - 1) }
        val q = f(b * t * hk * dk, 0.2f)
        val k = f(b * t * hk * dk, 0.2f)
        val v = f(b * t * hv * dv, 1f)
        val g = f(b * t * hv, 0.05f, -0.1f)
        val beta = f(b * t * hv, 0.4f, 0.5f)
        val state = f(s * hv * dk * dv, 0.5f)
        // Row 0: padding, then slot 0 continuing; row 1: slot 4 from position 0; row 2: slot 8 continuing.
        val slots = intArrayOf(-1, 0, 0, 4, 4, 4, 8, 8, 8)
        val positions = intArrayOf(0, 50, 51, 0, 1, 2, 70, 71, 72)
        val writeSlots = intArrayOf(-1, 1, 2, 5, 6, 7, 9, 10, 11)
        // A state bf16 rounds to a step of 2^-8 of its magnitude (|state| < 2 here).
        val stateTol = if (poolDtype == BF16) 2f / 256 else 1e-4f
        val forms = if (poolDtype == F32) listOf(true) else listOf(true, false)
        for (writes in listOf(false, true)) for (kernel in forms) {
            val dev = fn(kernel, writes, poolDtype)
            assertTrue(("tlaloc_gated_delta" in dev.toStablehlo()) == kernel, "the kernel is emitted when asked for")
            val floats = listOf(q, k, v, g, beta, state)
            val ints = listOf(slots, positions) + if (writes) listOf(writeSlots) else emptyList()
            val want = DxirInterpreter.evalFunction(fn(false, writes, poolDtype), floats + ints.map { a -> FloatArray(a.size) { a[it].toFloat() } })
            val got = TestBackend.session().use { sess -> sess.runOnHost(dev, floats + ints) }
            for ((i, what) in listOf("out", "state").withIndex()) {
                val a = got[i] as FloatArray
                var d = 0f
                for (j in a.indices) d = maxOf(d, abs(a[j] - want[i][j]))
                println("[fused-gdr] pool=$poolDtype kernel=$kernel writes=$writes $what: worst |d| against the interpreter $d")
                assertTrue(d <= (if (i == 0) 1e-4f else stateTol), "tlaloc_gated_delta $what differs: $d")
            }
        }
    }
}
