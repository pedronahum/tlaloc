@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Batching rules for elementwise and broadcasting ops, checked with [VmapOracle] (vmap
 * equals stacking the examples) at F32 and F64 and batch sizes 1, 7 and 64.
 */
class DxirVmapElementwiseTest {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    private fun DxirBuilder.c(dt: DType, v: Double): DxirNode =
        const(if (dt == F64) v else v.toFloat(), t(dt))

    @Test
    fun `unary chain on one batched argument`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("unary") {
                val x = param("x", t(dt, 3, 4))
                val a = op(OpKind.TANH, listOf(x), t(dt, 3, 4))
                val e = op(OpKind.EXP, listOf(a), t(dt, 3, 4))
                val s = op(OpKind.SIGMOID, listOf(x), t(dt, 3, 4))
                val n = op(OpKind.NEG, listOf(s), t(dt, 3, 4))
                val sn = op(OpKind.SIN, listOf(n), t(dt, 3, 4))
                val m = op(OpKind.MUL, listOf(e, sn), t(dt, 3, 4))
                listOf(op(OpKind.RELU, listOf(m), t(dt, 3, 4)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `log and sqrt of positive inputs`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("positive") {
                val x = param("x", t(dt, 5))
                val l = op(OpKind.LOG, listOf(x), t(dt, 5))
                val q = op(OpKind.SQRT, listOf(x), t(dt, 5))
                listOf(op(OpKind.DIV, listOf(l, q), t(dt, 5)))
            }
        },
        batched = listOf(true),
        input = { p, e, n -> VmapOracle.defaultInput(p, e, n).map { abs(it) + 0.5 }.toDoubleArray() },
    )

    @Test
    fun `binary ops with an unbatched operand of the same shape`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("mixed") {
                val x = param("x", t(dt, 4))
                val w = param("w", t(dt, 4))
                val a = op(OpKind.MUL, listOf(x, w), t(dt, 4))
                val d = op(OpKind.SUB, listOf(w, a), t(dt, 4))
                listOf(op(OpKind.ADD, listOf(d, w), t(dt, 4)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `binary ops with the batched operand second`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("second") {
                val w = param("w", t(dt, 2, 3))
                val x = param("x", t(dt, 2, 3))
                listOf(op(OpKind.SUB, listOf(w, x), t(dt, 2, 3)))
            }
        },
        batched = listOf(false, true),
    )

    @Test
    fun `a per-example scalar combined with a per-example vector`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("scalarVector") {
                val x = param("x", t(dt, 3, 2))
                val s = op(OpKind.SUM, listOf(x), t(dt))
                // Rank-deficient operand: the per-example program broadcasts the scalar.
                listOf(op(OpKind.MUL, listOf(s, x), t(dt, 3, 2)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `an unbatched rank-deficient operand broadcasts right-aligned`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("rowBias") {
                val x = param("x", t(dt, 3, 4))
                val bias = param("bias", t(dt, 4))
                listOf(op(OpKind.ADD, listOf(x, bias), t(dt, 3, 4)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `a batched rank-deficient operand against an unbatched matrix`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("batchedBias") {
                val bias = param("bias", t(dt, 4))
                val m = param("m", t(dt, 3, 4))
                listOf(op(OpKind.DIV, listOf(m, op(OpKind.EXP, listOf(bias), t(dt, 4))), t(dt, 3, 4)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `scalar splat and constants`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("splat") {
                val x = param("x", t(dt, 2, 3))
                val two = op(
                    OpKind.BROADCAST, listOf(c(dt, 2.0)), t(dt, 2, 3),
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
                val y = op(OpKind.MUL, listOf(x, two), t(dt, 2, 3))
                listOf(op(OpKind.ADD, listOf(y, c(dt, 0.25)), t(dt, 2, 3)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `broadcast to a higher rank, batched and unbatched values`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("broadcastTo") {
                val x = param("x", t(dt, 4))
                val w = param("w", t(dt, 3))
                // x [4] -> [3, 4] along axis 1; w [3] -> [3, 4] along axis 0.
                val xb = op(
                    OpKind.BROADCAST, listOf(x), t(dt, 3, 4),
                    attrs = mapOf("broadcast_dimensions" to listOf(1)),
                )
                val wb = op(
                    OpKind.BROADCAST, listOf(w), t(dt, 3, 4),
                    attrs = mapOf("broadcast_dimensions" to listOf(0)),
                )
                listOf(op(OpKind.MUL, listOf(xb, wb), t(dt, 3, 4)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `broadcast of a batched value along a leading axis`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("broadcastLead") {
                val x = param("x", t(dt, 3))
                listOf(
                    op(
                        OpKind.BROADCAST, listOf(x), t(dt, 3, 2),
                        attrs = mapOf("broadcast_dimensions" to listOf(0)),
                    ),
                )
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `pow, compare and where`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("select") {
                val x = param("x", t(dt, 5))
                val y = param("y", t(dt, 5))
                val three = op(
                    OpKind.BROADCAST, listOf(c(dt, 3.0)), t(dt, 5),
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
                val p = op(OpKind.POW, listOf(x, three), t(dt, 5))
                val gt = op(OpKind.COMPARE, listOf(x, y), t(Bool, 5), attrs = mapOf("direction" to "GT"))
                listOf(op(OpKind.WHERE, listOf(gt, p, y), t(dt, 5)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `an output that does not depend on the batch is repeated along it`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("twoOutputs") {
                val x = param("x", t(dt, 3))
                val w = param("w", t(dt, 2, 3))
                listOf(op(OpKind.EXP, listOf(x), t(dt, 3)), op(OpKind.TANH, listOf(w), t(dt, 2, 3)))
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `step, abs, sign and cast`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("pieces") {
                val x = param("x", t(dt, 6))
                val s = op(OpKind.STEP, listOf(x), t(dt, 6))
                val a = op(OpKind.ABS, listOf(x), t(dt, 6))
                val g = op(OpKind.SIGN, listOf(x), t(dt, 6))
                val sa = op(OpKind.MUL, listOf(s, a), t(dt, 6))
                listOf(op(OpKind.ADD, listOf(sa, g), t(dt, 6)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `an if whose condition does not depend on the batch`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("ifUnbatched") {
                val x = param("x", t(dt, 3))
                val k = param("k", t(dt))
                val cond = op(OpKind.STEP, listOf(k), t(Bool))
                val then = op(OpKind.EXP, listOf(x), t(dt, 3))
                val otherwise = op(OpKind.NEG, listOf(x), t(dt, 3))
                val r = ifOp(
                    cond, listOf(t(dt, 3)),
                    region { yields(then) },
                    region { yields(otherwise) },
                )
                listOf(r)
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `an if whose condition depends on the example selects per example`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("ifBatched") {
                val x = param("x", t(dt, 3))
                val s = op(OpKind.SUM, listOf(x), t(dt))
                val cond = op(OpKind.STEP, listOf(s), t(Bool))
                val r = ifOp(
                    cond, listOf(t(dt, 3), t(dt)),
                    region {
                        val e = op(OpKind.EXP, listOf(x), t(dt, 3))
                        yields(e, s)
                    },
                    region {
                        val n = op(OpKind.NEG, listOf(x), t(dt, 3))
                        yields(n, op(OpKind.NEG, listOf(s), t(dt)))
                    },
                )
                listOf(r.result(0), r.result(1))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `work that does not depend on the batch is done once`() {
        val fn = DxirBuilder.function("hoist") {
            val x = param("x", t(io.tlaloc.core.F32, 4))
            val w = param("w", t(io.tlaloc.core.F32, 4))
            val e = op(OpKind.EXP, listOf(w), t(io.tlaloc.core.F32, 4))
            listOf(op(OpKind.MUL, listOf(x, e), t(io.tlaloc.core.F32, 4)))
        }
        val v = DxirVmapTransform.apply(fn, listOf(true, false), 5)
        val exp = v.body.filterIsInstance<io.tlaloc.ir.DxirOp>().single { it.op == OpKind.EXP }
        assertEquals(listOf(4), exp.type.dims, "EXP of the unbatched w keeps its per-example shape")
        assertEquals(listOf(5, 4), v.params[0].type.dims)
        assertEquals(listOf(4), v.params[1].type.dims)
    }

    @Test
    fun `indexing at a constant position`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("index") {
                val x = param("x", t(dt, 4))
                val i = const(2, t(io.tlaloc.core.I32))
                val e = op(OpKind.GATHER, listOf(x, i), t(dt))
                listOf(op(OpKind.MUL, listOf(e, e), t(dt)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `an op without a batching rule is refused by name`() {
        val fn = DxirBuilder.function("gatherAt") {
            val x = param("x", t(io.tlaloc.core.F32, 4))
            val i = param("i", t(io.tlaloc.core.I32))
            listOf(op(OpKind.GATHER, listOf(x, i), t(io.tlaloc.core.F32)))
        }
        // An index known only at run time.
        val e = assertFailsWith<VmapUnsupportedException> {
            DxirVmapTransform.apply(fn, listOf(true, false), 3)
        }
        assertEquals(OpKind.GATHER, e.kind)
        assertTrue("no batching rule for GATHER" in e.message!!, e.message)
        val conv = DxirBuilder.function("conv") {
            val x = param("x", t(io.tlaloc.core.F32, 1, 1, 4, 4))
            val w = param("w", t(io.tlaloc.core.F32, 1, 1, 3, 3))
            listOf(op(OpKind.CONV2D, listOf(x, w), t(io.tlaloc.core.F32, 1, 1, 2, 2)))
        }
        assertEquals(OpKind.CONV2D, assertFailsWith<VmapUnsupportedException> { DxirVmapTransform.apply(conv, listOf(true, false), 3) }.kind)
    }
}
