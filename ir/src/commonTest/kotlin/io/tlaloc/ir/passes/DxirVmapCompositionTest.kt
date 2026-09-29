@file:OptIn(io.tlaloc.core.ExperimentalTlalocApi::class)

package io.tlaloc.ir.passes

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * vmap composed with the differentiation transforms, at the DXIR level, on the
 * interpreters: per-example gradients (vmap of the reverse transform) equal stacking
 * single-example gradients; the gradient of a vmapped, summed loss equals the sum of
 * single-example gradients; forward mode in both orders; vmap of vmap.
 */
class DxirVmapCompositionTest {

    private fun t(dt: DType, vararg dims: Int) = DxirType(dt, dims.toList())

    /** loss(w, x) = sum(tanh(x · w) ⊙ (x · w)) + sum(w²)/2, x [1, 3], w [3, 2]. */
    private fun loss(dt: DType): DxirFunction = DxirBuilder.function("loss") {
        val w = param("w", t(dt, 3, 2))
        val x = param("x", t(dt, 1, 3))
        val h = op(OpKind.MATMUL, listOf(x, w), t(dt, 1, 2))
        val a = op(OpKind.TANH, listOf(h), t(dt, 1, 2))
        val s = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(a, h), t(dt, 1, 2))), t(dt))
        val half = const(if (dt == F64) 0.5 else 0.5f, t(dt))
        val reg = op(OpKind.MUL, listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(w, w), t(dt, 3, 2))), t(dt)), half), t(dt))
        listOf(op(OpKind.ADD, listOf(s, reg), t(dt)))
    }

    @Test
    fun `vmap of the gradient gives per-example gradients`() = VmapOracle.check(
        // grad with respect to w of loss(w, x), x an input only; batched over x.
        build = { dt -> DxirReverseTransform.apply(loss(dt), inputOnlyTrailingParams = 1) },
        batched = listOf(false, true),
        tolerance = 1e-6,
    )

    @Test
    fun `vmap of the gradient with respect to the batched argument`() = VmapOracle.check(
        build = { dt ->
            // Reorder so x is first and w the input-only trailing param.
            val f = loss(dt)
            val swapped = DxirFunction(f.name, f.params.reversed(), f.body, f.returns, f.meshes)
            DxirReverseTransform.apply(swapped, inputOnlyTrailingParams = 1)
        },
        batched = listOf(true, false),
        tolerance = 1e-6,
    )

    @Test
    fun `vmap of the forward transform`() = VmapOracle.check(
        build = { dt -> DxirForwardTransform.apply(loss(dt)) },
        // (w, x, dw, dx): w and dw shared, x and dx per example.
        batched = listOf(false, true, false, true),
        tolerance = 1e-6,
    )

    /** `SUM` of a function's single batched output, appended to its body. */
    private fun summed(f: DxirFunction): DxirFunction {
        val ret = f.returns.single()
        val id = maxId(f) + 1
        val sum = DxirOp(id, OpKind.SUM, listOf(ret), emptyMap(), DxirType(ret.type.dtype, emptyList()))
        return DxirFunction("${f.name}_sum", f.params, f.body + sum, listOf(sum), f.meshes)
    }

    private fun maxId(f: DxirFunction): Int = (f.params.map { it.id } + f.body.map { it.id }).max()

    private fun data(n: Int, seed: Int) = VmapOracle.defaultInput(seed, 0, n)

    @Test
    fun `the gradient of a vmapped sum equals the sum of per-example gradients`() {
        for (dt in listOf(F32, F64)) for (batch in VmapOracle.batchSizes) {
            val f = loss(dt)
            val batchedLoss = summed(DxirVmapTransform.apply(f, listOf(false, true), batch))
            val g = DxirReverseTransform.apply(batchedLoss, inputOnlyTrailingParams = 1)
            val perExample = DxirReverseTransform.apply(f, inputOnlyTrailingParams = 1)
            val w = data(6, 1)
            val xs = data(3 * batch, 2)
            val got = eval(g, dt, listOf(w, xs)).single()
            val want = DoubleArray(6)
            for (i in 0 until batch) {
                val gi = eval(perExample, dt, listOf(w, xs.copyOfRange(3 * i, 3 * i + 3))).single()
                for (k in want.indices) want[k] += gi[k]
            }
            assertClose(want, got, if (dt == F64) 1e-12 else 1e-5, "$dt batch $batch")
        }
    }

    @Test
    fun `the forward transform of a vmapped function equals vmap of the forward transform`() {
        for (dt in listOf(F32, F64)) for (batch in VmapOracle.batchSizes) {
            val f = loss(dt)
            val a = DxirForwardTransform.apply(DxirVmapTransform.apply(f, listOf(false, true), batch))
            val b = DxirVmapTransform.apply(DxirForwardTransform.apply(f), listOf(false, true, false, true), batch)
            val w = data(6, 1)
            val xs = data(3 * batch, 2)
            val dw = data(6, 3)
            val dxs = data(3 * batch, 4)
            val ra = eval(a, dt, listOf(w, xs, dw, dxs))
            val rb = eval(b, dt, listOf(w, xs, dw, dxs))
            assertClose(rb[1], ra[1], if (dt == F64) 1e-12 else 1e-5, "$dt batch $batch tangent")
            assertClose(rb[0], ra[0], if (dt == F64) 1e-12 else 1e-5, "$dt batch $batch value")
        }
    }

    @Test
    fun `vmap of vmap batches two leading axes`() {
        for (dt in listOf(F32, F64)) {
            val f = DxirBuilder.function("inner") {
                val x = param("x", t(dt, 3))
                val w = param("w", t(dt, 3, 2))
                val r = op(OpKind.RESHAPE, listOf(x), t(dt, 1, 3))
                val y = op(OpKind.MATMUL, listOf(r, w), t(dt, 1, 2))
                listOf(op(OpKind.SOFTMAX, listOf(y), t(dt, 1, 2)))
            }
            val (a, b) = 4 to 5
            val inner = DxirVmapTransform.apply(f, listOf(true, false), b)
            val outer = DxirVmapTransform.apply(inner, listOf(true, false), a)
            assertEquals(listOf(a, b, 3), outer.params[0].type.dims)
            assertEquals(listOf(a, b, 1, 2), outer.returns[0].type.dims)
            val xs = data(a * b * 3, 5)
            val w = data(6, 6)
            val got = eval(outer, dt, listOf(xs, w)).single()
            val want = (0 until a * b).flatMap { i ->
                eval(f, dt, listOf(xs.copyOfRange(3 * i, 3 * i + 3), w)).single().asList()
            }.toDoubleArray()
            assertClose(want, got, if (dt == F64) 1e-12 else 1e-6, "$dt")
        }
    }

    private fun assertClose(want: DoubleArray, got: DoubleArray, tol: Double, what: String) {
        assertEquals(want.size, got.size, "$what: size")
        val scale = maxOf(1.0, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= tol * scale, "$what [$i]: want ${want[i]}, got ${got[i]}")
        }
    }

    private fun eval(fn: DxirFunction, dt: DType, inputs: List<DoubleArray>): List<DoubleArray> = when (dt) {
        F64 -> DxirInterpreterF64.evalFunction(fn, inputs)
        else -> DxirInterpreter.evalFunction(fn, inputs.map { d -> FloatArray(d.size) { d[it].toFloat() } })
            .map { f -> DoubleArray(f.size) { f[it].toDouble() } }
    }

}
