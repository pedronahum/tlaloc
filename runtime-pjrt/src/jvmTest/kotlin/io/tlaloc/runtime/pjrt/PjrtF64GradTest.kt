package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * F64 gradient graphs of the shape `grad {}` lowers to, through the general
 * [PjrtSession.runOnHost] entry (each param at its own dtype), against
 * [DxirInterpreterF64] on the same graph.
 *
 * Tolerance 1e-12 of the largest entry: both sides compute in f64; they differ only in
 * the order of the reductions and in the last bits of `exp`/`tanh`/`log`, about 1e-15 per
 * op over a few dozen ops. A graph that ran in f32 anywhere would differ by 1e-8 or more.
 * The graphs are small: the GB10's f64 throughput is a fraction of its f32 throughput.
 */
class PjrtF64GradTest {

    private fun t(dtype: DType, vararg dims: Int) = DxirType(dtype, dims.toList())

    private fun maxRelDiff(want: DoubleArray, got: DoubleArray): Double {
        val scale = max(1e-300, want.maxOf { abs(it) })
        return want.indices.maxOf { abs(want[it] - got[it]) } / scale
    }

    private val m = doubleArrayOf(0.31, -0.72, 1.13, 0.05, -0.44, 0.87, -1.29, 0.66, 0.92, 0.18, -0.35, -0.81)
    private val v = doubleArrayOf(0.25, -0.6, 0.9, 0.12)
    private val b = doubleArrayOf(0.7, -0.2, 0.15, 1.1, -0.9, 0.4, 0.33, -0.55)

    private fun splat(builder: DxirBuilder, x: Double, like: io.tlaloc.ir.DxirNode) = with(builder) {
        op(OpKind.BROADCAST, listOf(const(x, t(F64)), like), like.type, attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
    }

    private val elementwise: DxirFunction = DxirBuilder.function("elementwise") {
        val mm = param("m", t(F64, 3, 4))
        val vv = param("v", t(F64, 4))
        val m2 = op(OpKind.MUL, listOf(mm, mm), t(F64, 3, 4))
        val a = op(
            OpKind.SUB,
            listOf(
                op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(m2, splat(this, 0.3, m2)), t(F64, 3, 4)), op(OpKind.EXP, listOf(mm), t(F64, 3, 4))), t(F64, 3, 4)),
                op(OpKind.MUL, listOf(op(OpKind.TANH, listOf(op(OpKind.ADD, listOf(mm, vv), t(F64, 3, 4))), t(F64, 3, 4)), vv), t(F64, 3, 4)),
            ),
            t(F64, 3, 4),
        )
        val bb = op(OpKind.DIV, listOf(a, op(OpKind.ADD, listOf(m2, splat(this, 1.5, m2)), t(F64, 3, 4))), t(F64, 3, 4))
        val s = op(OpKind.SUM, listOf(bb), t(F64, 3), attrs = mapOf("reduction_dims" to listOf(1)))
        val mx = op(OpKind.MAX, listOf(bb), t(F64, 3), attrs = mapOf("reduction_dims" to listOf(1)))
        val left = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(s, mx), t(F64, 3))), t(F64))
        listOf(op(OpKind.ADD, listOf(left, op(OpKind.MUL, listOf(const(0.7, t(F64)), op(OpKind.MEAN, listOf(bb), t(F64))), t(F64))), t(F64)))
    }

    private val matmulSoftmax: DxirFunction = DxirBuilder.function("matmul_softmax") {
        val aa = param("a", t(F64, 3, 4))
        val bb = param("b", t(F64, 4, 2))
        val c = op(OpKind.MATMUL, listOf(aa, bb), t(F64, 3, 2))
        val s = op(OpKind.SOFTMAX, listOf(c), t(F64, 3, 2), attrs = mapOf("axis" to 1))
        val e = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(s, op(OpKind.LOG, listOf(s), t(F64, 3, 2))), t(F64, 3, 2))), t(F64))
        val tt = op(OpKind.MATMUL, listOf(op(OpKind.TRANSPOSE, listOf(aa), t(F64, 4, 3), attrs = mapOf("permutation" to listOf(1, 0))), aa), t(F64, 4, 4))
        listOf(op(OpKind.ADD, listOf(e, op(OpKind.MEAN, listOf(op(OpKind.MUL, listOf(tt, tt), t(F64, 4, 4))), t(F64))), t(F64)))
    }

    private fun assertDeviceMatchesInterpreter(session: PjrtSession, g: DxirFunction, inputs: List<DoubleArray>) {
        val want = DxirInterpreterF64.evalFunction(g, inputs)
        val got = session.runOnHost(g, inputs)
        assertEquals(want.size, got.size)
        for (r in want.indices) {
            val d = got[r] as? DoubleArray ?: error("${g.name} result $r came back as ${got[r]::class.simpleName}")
            val diff = maxRelDiff(want[r], d)
            println("[pjrt-f64] ${g.name} result $r: device vs interpreter $diff of the largest entry")
            assertTrue(diff <= 1e-12, "${g.name} result $r: device differs from the F64 interpreter by $diff")
        }
    }

    @Test
    fun f64GradientGraphsMatchTheF64Interpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { session ->
            assertDeviceMatchesInterpreter(session, DxirReverseTransform.apply(elementwise, includeForward = true), listOf(m, v))
            assertDeviceMatchesInterpreter(session, DxirReverseTransform.apply(matmulSoftmax), listOf(m, b))
            // Forward over reverse: the Hessian-vector product the `hessian {}` intrinsic assembles.
            val hvp = DxirForwardTransform.apply(DxirReverseTransform.apply(matmulSoftmax))
            val dirA = DoubleArray(12) { ((it * 5) % 7 - 3) / 3.0 }
            val dirB = DoubleArray(8) { ((it * 3) % 5 - 2) / 2.0 }
            assertDeviceMatchesInterpreter(session, hvp, listOf(m, b, dirA, dirB))
        }
    }

    @Test
    fun f64ConstantsReachTheDeviceWithEveryDigit() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // A scalar 0.1 and a rank-1 constant [0.1, 1e-9]: in f32 the first is 1.5e-8 off and
        // the second vanishes next to 1, so an f32 constant anywhere fails 1e-15.
        val fn = DxirBuilder.function("consts") {
            val x = param("x", t(F64, 2))
            val scaled = op(OpKind.MUL, listOf(x, splat(this, 0.1, x)), t(F64, 2))
            listOf(op(OpKind.ADD, listOf(scaled, const(doubleArrayOf(0.1, 1e-9), t(F64, 2))), t(F64, 2)))
        }
        val x = doubleArrayOf(1.0 + 1e-10, 1.0)
        TestBackend.session().use { session ->
            val got = session.runOnHost(fn, listOf(x)).single() as DoubleArray
            val want = doubleArrayOf(x[0] * 0.1 + 0.1, x[1] * 0.1 + 1e-9)
            assertTrue(maxRelDiff(want, got) <= 1e-15, "got ${got.toList()}, want ${want.toList()}")
        }
    }

    @Test
    fun runOnHostTakesEachParamAtItsOwnDtypeAndRefusesAMismatch() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fn = DxirBuilder.function("two") {
            val a = param("a", t(F32, 2))
            val b = param("b", t(F64, 2))
            listOf(op(OpKind.MUL, listOf(a, a), t(F32, 2)), op(OpKind.MUL, listOf(b, b), t(F64, 2)))
        }
        TestBackend.session().use { session ->
            val out = session.runOnHost(fn, listOf(floatArrayOf(1.5f, -2f), doubleArrayOf(0.1, 3.0)))
            assertEquals(listOf(2.25f, 4f), (out[0] as FloatArray).toList())
            assertEquals(listOf(0.1 * 0.1, 9.0), (out[1] as DoubleArray).toList())
            val e = assertFailsWith<IllegalArgumentException> {
                session.runOnHost(fn, listOf(floatArrayOf(1f, 2f), floatArrayOf(1f, 2f)))
            }
            assertTrue("param 'b' is f64 but its input is a FloatArray" in (e.message ?: ""), e.message)
        }
    }
}
