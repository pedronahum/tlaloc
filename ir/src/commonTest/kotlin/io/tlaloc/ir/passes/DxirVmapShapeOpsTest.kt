@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.ir.passes

import io.tlaloc.core.DType
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * Batching rules for reductions, shape ops, matmul, softmax and composed losses,
 * checked with [VmapOracle] at F32 and F64 and batch sizes 1, 7 and 64.
 */
class DxirVmapShapeOpsTest {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    private fun DxirBuilder.c(dt: DType, v: Double): DxirNode =
        const(if (dt == F64) v else v.toFloat(), t(dt))

    @Test
    fun `sum, mean, max and min over all axes and over some`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("reductions") {
                val x = param("x", t(dt, 3, 4, 2))
                val all = op(OpKind.SUM, listOf(x), t(dt))
                val rows = op(OpKind.MEAN, listOf(x), t(dt, 4, 2), attrs = mapOf("reduction_dims" to listOf(0)))
                val mx = op(OpKind.MAX, listOf(x), t(dt, 3), attrs = mapOf("reduction_dims" to listOf(1, 2)))
                val mn = op(OpKind.MIN, listOf(x), t(dt))
                val kept = op(OpKind.SUM, listOf(x), t(dt, 3, 1, 2), attrs = mapOf("reduction_dims" to listOf(1)))
                listOf(all, rows, mx, mn, kept)
            }
        },
        batched = listOf(true),
        tolerance = 1e-6,
    )

    @Test
    fun `a reduction of a per-example scalar is the value`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("scalarSum") {
                val x = param("x", t(dt))
                listOf(op(OpKind.SUM, listOf(op(OpKind.EXP, listOf(x), t(dt))), t(dt)))
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `softmax along the default and an explicit axis`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("softmax") {
                val x = param("x", t(dt, 3, 5))
                val last = op(OpKind.SOFTMAX, listOf(x), t(dt, 3, 5))
                val first = op(OpKind.SOFTMAX, listOf(x), t(dt, 3, 5), attrs = mapOf("axis" to 0))
                val neg = op(OpKind.SOFTMAX, listOf(x), t(dt, 3, 5), attrs = mapOf("axis" to -1))
                listOf(last, first, neg)
            }
        },
        batched = listOf(true),
        tolerance = 1e-6,
    )

    @Test
    fun `transpose, reshape and reverse`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("shapes") {
                val x = param("x", t(dt, 2, 3, 4))
                val tr = op(OpKind.TRANSPOSE, listOf(x), t(dt, 4, 2, 3), attrs = mapOf("permutation" to listOf(2, 0, 1)))
                val flat = op(OpKind.RESHAPE, listOf(tr), t(dt, 24))
                val back = op(OpKind.RESHAPE, listOf(flat), t(dt, 6, 4))
                val rev = op(OpKind.REVERSE, listOf(back), t(dt, 6, 4), attrs = mapOf("dimensions" to listOf(0, 1)))
                val m = op(OpKind.TRANSPOSE, listOf(rev), t(dt, 4, 6), attrs = mapOf("permutation" to listOf(1, 0)))
                listOf(flat, m)
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `slice and pad`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("slicePad") {
                val x = param("x", t(dt, 5, 4))
                val s = op(
                    OpKind.SLICE, listOf(x), t(dt, 2, 4),
                    attrs = mapOf(
                        "start_indices" to listOf(1, 0), "limit_indices" to listOf(3, 4), "strides" to listOf(1, 1),
                        "slice_axis" to 0, "slice_start" to 1, "slice_end" to 3,
                    ),
                )
                val strided = op(
                    OpKind.SLICE, listOf(x), t(dt, 3, 2),
                    attrs = mapOf(
                        "start_indices" to listOf(0, 1), "limit_indices" to listOf(5, 4), "strides" to listOf(2, 2),
                    ),
                )
                val p = op(
                    OpKind.PAD, listOf(s), t(dt, 5, 5),
                    attrs = mapOf("low" to listOf(1, 0), "high" to listOf(2, 1)),
                )
                listOf(strided, p)
            }
        },
        batched = listOf(true),
    )

    @Test
    fun `concat of a batched and an unbatched operand`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("concat") {
                val x = param("x", t(dt, 2, 3))
                val w = param("w", t(dt, 2, 1))
                listOf(
                    op(OpKind.CONCAT, listOf(x, w), t(dt, 2, 4), attrs = mapOf("dimension" to 1)),
                    op(OpKind.CONCAT, listOf(w, w, x), t(dt, 2, 5), attrs = mapOf("dimension" to 1)),
                )
            }
        },
        batched = listOf(true, false),
    )

    @Test
    fun `matmul with both operands batched`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("matmulBoth") {
                val a = param("a", t(dt, 2, 3))
                val b = param("b", t(dt, 3, 4))
                listOf(op(OpKind.MATMUL, listOf(a, b), t(dt, 2, 4)))
            }
        },
        batched = listOf(true, true),
        tolerance = 1e-6,
    )

    @Test
    fun `matmul with an unbatched weight on either side`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("matmulWeight") {
                val x = param("x", t(dt, 1, 3))
                val w = param("w", t(dt, 3, 4))
                val v = param("v", t(dt, 2, 1))
                val xw = op(OpKind.MATMUL, listOf(x, w), t(dt, 1, 4))
                listOf(xw, op(OpKind.MATMUL, listOf(v, xw), t(dt, 2, 4)))
            }
        },
        batched = listOf(true, false, false),
        tolerance = 1e-6,
    )

    @Test
    fun `batched matmul per example`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("matmulRank3") {
                val a = param("a", t(dt, 2, 2, 3))
                val b = param("b", t(dt, 2, 3, 2))
                listOf(op(OpKind.MATMUL, listOf(a, b), t(dt, 2, 2, 2)))
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-6,
    )

    @Test
    fun `mean squared error against a batched and an unbatched target`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("mse") {
                val y = param("y", t(dt, 6))
                val target = param("target", t(dt, 6))
                val d = op(OpKind.SUB, listOf(y, target), t(dt, 6))
                listOf(op(OpKind.MEAN, listOf(op(OpKind.MUL, listOf(d, d), t(dt, 6))), t(dt)))
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-6,
    )

    @Test
    fun `cross-entropy of a two-layer network, one example per call`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("mlpCrossEntropy") {
                val x = param("x", t(dt, 1, 4))
                val oneHot = param("oneHot", t(dt, 1, 3))
                val w1 = param("w1", t(dt, 4, 5))
                val b1 = param("b1", t(dt, 5))
                val w2 = param("w2", t(dt, 5, 3))
                val h = op(OpKind.MATMUL, listOf(x, w1), t(dt, 1, 5))
                val hb = op(OpKind.ADD, listOf(h, b1), t(dt, 1, 5))
                val a = op(OpKind.TANH, listOf(hb), t(dt, 1, 5))
                val logits = op(OpKind.MATMUL, listOf(a, w2), t(dt, 1, 3))
                val logp = op(OpKind.LOG, listOf(op(OpKind.SOFTMAX, listOf(logits), t(dt, 1, 3))), t(dt, 1, 3))
                val prod = op(OpKind.MUL, listOf(oneHot, logp), t(dt, 1, 3))
                listOf(op(OpKind.NEG, listOf(op(OpKind.SUM, listOf(prod), t(dt))), t(dt)))
            }
        },
        batched = listOf(true, true, false, false, false),
        tolerance = 1e-6,
    )

    @Test
    fun `a named contract that is the canonical product batches`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("contract") {
                val a = param("a", t(dt, 2, 3))
                val b = param("b", t(dt, 3, 4))
                listOf(
                    op(
                        OpKind.MATMUL, listOf(a, b), t(dt, 2, 4),
                        // (The interpreter evaluates the per-example op without batching keys.)
                        attrs = mapOf("lhs_contracting_dims" to listOf(1), "rhs_contracting_dims" to listOf(0)),
                    ),
                )
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-6,
    )

    @Test
    fun `a contract over another axis pair is refused by name`() {
        val fn = DxirBuilder.function("contractT") {
            val a = param("a", t(io.tlaloc.core.F32, 3, 2))
            val b = param("b", t(io.tlaloc.core.F32, 3, 4))
            listOf(
                op(
                    OpKind.MATMUL, listOf(a, b), t(io.tlaloc.core.F32, 2, 4),
                    attrs = mapOf(
                        "lhs_contracting_dims" to listOf(0), "rhs_contracting_dims" to listOf(0),
                        "lhs_batching_dims" to emptyList<Int>(), "rhs_batching_dims" to emptyList<Int>(),
                    ),
                ),
            )
        }
        val e = assertFailsWith<VmapUnsupportedException> { DxirVmapTransform.apply(fn, listOf(true, false), 2) }
        assertEquals(OpKind.MATMUL, e.kind)
    }

    @Test
    fun `dot becomes a multiply and a sum over the example axis`() {
        val fn = DxirBuilder.function("dot") {
            val a = param("a", t(io.tlaloc.core.F32, 3))
            val b = param("b", t(io.tlaloc.core.F32, 3))
            listOf(op(OpKind.DOT, listOf(a, b), t(io.tlaloc.core.F32)))
        }
        val v = DxirVmapTransform.apply(fn, listOf(true, false), 2)
        // The interpreter has no DOT arm; the batched form is ops it evaluates.
        val out = DxirInterpreter.evalFunction(v, listOf(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), floatArrayOf(1f, 0f, 2f)))
        assertEquals(listOf(7f, 16f), out[0].toList())
    }

    @Test
    fun `runtime-extent ops, batched and unbatched templates`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("extent") {
                val x = param("x", t(dt, 2, 3))
                val w = param("w", t(dt, 3))
                val v = param("v", t(dt, 4, 2, 3))
                val big = param("big", t(dt, 3, 5))
                // SUM_TO of a batched value to an unbatched lower-rank template, and the reverse.
                val s1 = op(OpKind.SUM_TO, listOf(x, w), t(dt, 3))
                val s2 = op(OpKind.SUM_TO, listOf(v, x), t(dt, 2, 3))
                val b1 = op(OpKind.BROADCAST_LIKE, listOf(w, x), t(dt, 2, 3))
                val b2 = op(OpKind.BROADCAST_LIKE, listOf(s1, v), t(dt, 4, 2, 3))
                val p = op(OpKind.PAD_TO, listOf(x, big), t(dt, 3, 5), attrs = mapOf("low" to listOf(1, 2)))
                val c = op(OpKind.SLICE_AT, listOf(big, x), t(dt, 2, 3), attrs = mapOf("low" to listOf(0, 1)))
                listOf(s1, s2, b1, b2, p, c)
            }
        },
        batched = listOf(true, false, false, false),
    )

    @Test
    fun `slice-like and pad-like along a per-example axis`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("like") {
                val x = param("x", t(dt, 2, 5))
                val a = param("a", t(dt, 2, 2))
                val b = param("b", t(dt, 2, 3))
                val sl = op(OpKind.SLICE_LIKE, listOf(x, b, a), t(dt, 2, 3), attrs = mapOf("axis" to 1))
                val pl = op(OpKind.PAD_LIKE, listOf(sl, x, a), t(dt, 2, 5), attrs = mapOf("axis" to 1))
                listOf(sl, pl)
            }
        },
        batched = listOf(true, false, false),
    )

    @Test
    fun `softmax, logsumexp and argmax of a per-example scalar are refused`() {
        for (kind in listOf(OpKind.SOFTMAX, OpKind.LOGSUMEXP, OpKind.ARGMAX)) {
            val fn = DxirBuilder.function("scalar") {
                val x = param("x", t(io.tlaloc.core.F32))
                listOf(op(kind, listOf(x), t(io.tlaloc.core.F32)))
            }
            val e = assertFailsWith<VmapUnsupportedException> { DxirVmapTransform.apply(fn, listOf(true), 3) }
            assertEquals(kind, e.kind)
        }
    }
}
