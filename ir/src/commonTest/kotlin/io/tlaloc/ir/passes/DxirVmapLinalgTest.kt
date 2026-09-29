@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.ir.passes

import io.tlaloc.core.DType
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * Batching rules for the linear-algebra ops with native leading batch axes (CHOLESKY,
 * TRIANGULAR_SOLVE, TRIANGLE) and the SPD composites the plugin lowers to them
 * (`solveSpd`, `logDetSpd`), checked with [VmapOracle] at F32 and F64 and batch sizes
 * 1, 7 and 64; per-example gradients through them; and a refusal for the `while`-lowered ops.
 */
class DxirVmapLinalgTest {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    private val n = 3

    /** Inputs where parameter [spd] is a symmetric positive-definite `n x n` matrix per example. */
    private fun spdInput(vararg spd: Int): (Int, Int, Int) -> DoubleArray = { p, e, size ->
        val raw = VmapOracle.defaultInput(p, e, size)
        if (p in spd) {
            DoubleArray(n * n) { k ->
                val i = k / n
                val j = k % n
                var s = 0.0
                for (q in 0 until n) s += raw[i * n + q] * raw[j * n + q]
                s + if (i == j) n.toDouble() else 0.0
            }
        } else {
            raw
        }
    }

    private fun DxirBuilder.triangle(x: DxirNode, lower: Double, diagonal: Double, upper: Double) =
        op(OpKind.TRIANGLE, listOf(x), x.type, attrs = mapOf("lower" to lower, "diagonal" to diagonal, "upper" to upper))

    private fun DxirBuilder.solve(a: DxirNode, b: DxirNode, lower: Boolean, transposeA: Boolean) = op(
        OpKind.TRIANGULAR_SOLVE, listOf(a, b), b.type,
        attrs = mapOf("lower" to lower, "transpose_a" to transposeA, "unit_diagonal" to false),
    )

    /** `logDetSpd(a)` as the plugin lowers it. */
    private fun DxirBuilder.logDetSpd(a: DxirNode, dt: DType): DxirNode {
        val l = op(OpKind.CHOLESKY, listOf(a), a.type)
        val diag = op(OpKind.SUM, listOf(triangle(l, 0.0, 1.0, 0.0)), t(dt, n), attrs = mapOf("reduction_dims" to listOf(1)))
        val half = op(OpKind.SUM, listOf(op(OpKind.LOG, listOf(diag), t(dt, n))), t(dt))
        return op(OpKind.ADD, listOf(half, half), t(dt))
    }

    @Test
    fun `cholesky, triangle and triangular solves, batched and unbatched operands`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("chol") {
                val a = param("a", t(dt, n, n))
                val b = param("b", t(dt, n, 2))
                val l = op(OpKind.CHOLESKY, listOf(a), a.type)
                val y = solve(l, b, lower = true, transposeA = false)
                val x = solve(l, y, lower = true, transposeA = true)
                listOf(l, x, triangle(a, 0.5, 2.0, 0.0))
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-5,
        input = spdInput(0),
    )

    @Test
    fun `solves against a batched right-hand side with a shared factor`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("sharedFactor") {
                val a = param("a", t(dt, n, n))
                val b = param("b", t(dt, n, 2))
                val l = op(OpKind.CHOLESKY, listOf(a), a.type)
                listOf(solve(l, b, lower = true, transposeA = false))
            }
        },
        batched = listOf(false, true),
        tolerance = 1e-5,
        input = spdInput(0),
    )

    @Test
    fun `logDetSpd per example`() = VmapOracle.check(
        build = { dt -> DxirBuilder.function("logdet") { listOf(logDetSpd(param("a", t(dt, n, n)), dt)) } },
        batched = listOf(true),
        tolerance = 1e-5,
        input = spdInput(0),
    )

    @Test
    fun `per-example gradients of logDetSpd`() = VmapOracle.check(
        build = { dt ->
            DxirReverseTransform.apply(DxirBuilder.function("logdet") { listOf(logDetSpd(param("a", t(dt, n, n)), dt)) })
        },
        batched = listOf(true),
        tolerance = 1e-5,
        input = spdInput(0),
    )

    @Test
    fun `per-example gradients through a Cholesky solve`() = VmapOracle.check(
        build = { dt ->
            val f = DxirBuilder.function("solveLoss") {
                val a = param("a", t(dt, n, n))
                val b = param("b", t(dt, n, 2))
                val l = op(OpKind.CHOLESKY, listOf(a), a.type)
                val x = solve(l, solve(l, b, lower = true, transposeA = false), lower = true, transposeA = true)
                listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, b), b.type)), t(dt)))
            }
            DxirReverseTransform.apply(f)
        },
        batched = listOf(true, true),
        tolerance = 1e-5,
        input = spdInput(0),
    )

    /** Inputs where parameter 0 is a diagonally dominant `n x n` matrix (well conditioned, nonsymmetric). */
    private val dominant: (Int, Int, Int) -> DoubleArray = { p, e, size ->
        val raw = VmapOracle.defaultInput(p, e, size)
        if (p == 0) DoubleArray(n * n) { k -> raw[k] + if (k / n == k % n) 4.0 else 0.0 } else raw
    }

    @Test
    fun `solve and det per example, batched and shared operands`() = VmapOracle.check(
        build = { dt ->
            DxirBuilder.function("lu") {
                val a = param("a", t(dt, n, n))
                val b = param("b", t(dt, n, 2))
                listOf(
                    op(OpKind.SOLVE, listOf(a, b), b.type, attrs = mapOf("transpose_a" to false)),
                    op(OpKind.SOLVE, listOf(a, b), b.type, attrs = mapOf("transpose_a" to true)),
                    op(OpKind.DET, listOf(a), t(dt)),
                )
            }
        },
        batched = listOf(true, false),
        tolerance = 1e-5,
        input = dominant,
    )

    @Test
    fun `per-example gradients through solve and det`() = VmapOracle.check(
        build = { dt ->
            DxirReverseTransform.apply(
                DxirBuilder.function("luLoss") {
                    val a = param("a", t(dt, n, n))
                    val b = param("b", t(dt, n, 2))
                    val x = op(OpKind.SOLVE, listOf(a, b), b.type, attrs = mapOf("transpose_a" to false))
                    val s = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, x), b.type)), t(dt))
                    listOf(op(OpKind.ADD, listOf(s, op(OpKind.DET, listOf(a), t(dt))), t(dt)))
                },
            )
        },
        batched = listOf(true, true),
        tolerance = 1e-5,
        input = dominant,
    )

    @Test
    fun `per-example forward derivatives of det`() = VmapOracle.check(
        build = { dt ->
            DxirForwardTransform.apply(
                DxirBuilder.function("det") { listOf(op(OpKind.DET, listOf(param("a", t(dt, n, n))), t(dt))) },
            )
        },
        batched = listOf(true, true),
        tolerance = 1e-5,
        input = dominant,
    )

    @Test
    fun `qr and eigh are refused by name`() {
        for (kind in listOf(OpKind.QR_Q, OpKind.QR_R, OpKind.EIGH_W, OpKind.EIGH_V)) {
            val fn: DxirFunction = DxirBuilder.function("wl") {
                val a = param("a", t(F64, n, n))
                listOf(op(kind, listOf(a), if (kind == OpKind.EIGH_W) t(F64, n) else a.type))
            }
            val e = assertFailsWith<VmapUnsupportedException> { DxirVmapTransform.apply(fn, listOf(true), 2) }
            assertEquals(kind, e.kind)
        }
    }
}
