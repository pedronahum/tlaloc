package io.tlaloc.ir.passes

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank2
import io.tlaloc.core.SparseTensor
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.sparseMatmul
import io.tlaloc.core.ops.sparseMatmulTransposed
import io.tlaloc.core.ops.sparseMatmulValuesAdjoint
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.418 — the `:core` host twins `sparseMatmul` / `sparseMatmulTransposed`
 * / `sparseMatmulValuesAdjoint` are BIT-EXACT against the dxir interpreter's
 * SPARSE_MATMUL / SPARSE_MATMUL_VALUES_ADJOINT arms — the
 * [DxirHostEmbeddingParityTest] discipline for the sparse front-end. The
 * eventual E1c synthesis calls these twins for `grad {}` sparse gradients,
 * and the interpreted dxir is the oracle those results are certified
 * against, so the parity must be exact (`assertContentEquals` on raw floats,
 * NO tolerance, arbitrary non-grid values — the claim is identical
 * arithmetic). Both are additionally pinned against E1a's `SparseTensor`
 * spellings, closing the triangle host twin ⇄ interpreter ⇄ E1a.
 */
class DxirHostSparseMatmulParityTest {

    private fun randomCsr(rnd: Random, rows: Int, cols: Int, density: Double): SparseTensor {
        val rowIdx = ArrayList<Int>()
        val colIdx = ArrayList<Int>()
        val values = ArrayList<Float>()
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (rnd.nextDouble() < density) {
                    rowIdx.add(i)
                    colIdx.add(j)
                    values.add(rnd.nextFloat() * 4f - 2f)
                }
            }
        }
        return SparseTensor.fromCoo(
            rows, cols, rowIdx.toIntArray(), colIdx.toIntArray(), values.toFloatArray(),
        )
    }

    private fun intConstData(v: IntArray): FloatArray = FloatArray(v.size) { v[it].toFloat() }

    private fun matrix(rows: Int, cols: Int, data: FloatArray): DTensor<Rank2<Sym, Sym>, F32> =
        DTensor(HostF32Storage(data.copyOf()), intArrayOf(rows, cols), F32)

    @Test
    fun hostTwinsMatchInterpreterArmsBitExact() {
        val rnd = Random(4183)
        for (density in doubleArrayOf(0.0, 0.05, 0.3, 0.7)) {
            val rows = 7
            val cols = 5
            val d = 3
            val s = randomCsr(rnd, rows, cols, density)
            val bData = FloatArray(cols * d) { rnd.nextFloat() * 4f - 2f }
            val upData = FloatArray(rows * d) { rnd.nextFloat() * 4f - 2f }
            val values = Tensors.f32Vector<Sym>(s.values.copyOf())
            val ci = Tensors.i32Vector<Sym>(s.colIdx.copyOf())
            val rp = Tensors.i32Vector<Sym>(s.rowPtr.copyOf())
            val b = matrix(cols, d, bData)
            val up = matrix(rows, d, upData)

            // Forward SpMM: host twin ⇄ interpreter ⇄ E1a.
            val fwdFn = DxirBuilder.function("spmm") {
                val v = param("v", DxirType(F32, listOf(s.nnz)))
                val bb = param("b", DxirType(F32, listOf(cols, d)))
                val cci = const(intConstData(s.colIdx), DxirType(I32, listOf(s.nnz)))
                val rrp = const(intConstData(s.rowPtr), DxirType(I32, listOf(rows + 1)))
                listOf(
                    op(OpKind.SPARSE_MATMUL, listOf(v, cci, rrp, bb), DxirType(F32, listOf(rows, d))),
                )
            }
            val wantFwd = DxirInterpreter.evalFunction(fwdFn, listOf(s.values, bData)).single()
            val gotFwd = sparseMatmul(values, ci, rp, b)
            assertContentEquals(wantFwd, gotFwd.hostF32(), "density $density: host sparseMatmul vs interpreter")
            assertContentEquals(listOf(rows, d), gotFwd.dims.toList())
            assertContentEquals((s matmul b).hostF32(), gotFwd.hostF32(), "host twin vs SparseTensor.matmul")

            // Transposed SpMM: template contributes SHAPE ONLY.
            val tFn = DxirBuilder.function("spmm_t") {
                val v = param("v", DxirType(F32, listOf(s.nnz)))
                val uu = param("u", DxirType(F32, listOf(rows, d)))
                val tt = param("t", DxirType(F32, listOf(cols, d)))
                val cci = const(intConstData(s.colIdx), DxirType(I32, listOf(s.nnz)))
                val rrp = const(intConstData(s.rowPtr), DxirType(I32, listOf(rows + 1)))
                listOf(
                    op(
                        OpKind.SPARSE_MATMUL, listOf(v, cci, rrp, uu, tt),
                        DxirType(F32, listOf(cols, d)),
                        attrs = mapOf("transposed" to true),
                    ),
                )
            }
            val wantT = DxirInterpreter.evalFunction(
                tFn, listOf(s.values, upData, FloatArray(cols * d)),
            ).single()
            val garbageTemplate = matrix(cols, d, FloatArray(cols * d) { Float.NaN })
            val gotT = sparseMatmulTransposed(values, ci, rp, up, garbageTemplate)
            assertContentEquals(wantT, gotT.hostF32(), "density $density: host sparseMatmulTransposed vs interpreter")
            assertContentEquals(listOf(cols, d), gotT.dims.toList())
            assertContentEquals(
                (s.transpose() matmul up).hostF32(), gotT.hostF32(),
                "host transposed twin vs SparseTensor.transpose().matmul",
            )

            // The SDDMM values-adjoint.
            val vaFn = DxirBuilder.function("sddmm") {
                val uu = param("u", DxirType(F32, listOf(rows, d)))
                val bb = param("b", DxirType(F32, listOf(cols, d)))
                val cci = const(intConstData(s.colIdx), DxirType(I32, listOf(s.nnz)))
                val rrp = const(intConstData(s.rowPtr), DxirType(I32, listOf(rows + 1)))
                listOf(
                    op(
                        OpKind.SPARSE_MATMUL_VALUES_ADJOINT, listOf(uu, bb, cci, rrp),
                        DxirType(F32, listOf(s.nnz)),
                    ),
                )
            }
            val wantVa = DxirInterpreter.evalFunction(vaFn, listOf(upData, bData)).single()
            val gotVa = sparseMatmulValuesAdjoint(up, b, ci, rp)
            assertContentEquals(wantVa, gotVa.hostF32(), "density $density: host sparseMatmulValuesAdjoint vs interpreter")
            assertContentEquals(listOf(s.nnz), gotVa.dims.toList())
        }
    }

    @Test
    fun hostTwinsValidateTheCsrInvariantLoudly() {
        val values = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))
        val ci = Tensors.i32Vector<Sym>(intArrayOf(0, 1))
        val b = matrix(2, 2, FloatArray(4) { it.toFloat() })
        // rowPtr not ending at nnz.
        val badEnd = assertFailsWith<IllegalArgumentException> {
            sparseMatmul(values, ci, Tensors.i32Vector<Sym>(intArrayOf(0, 1, 1)), b)
        }
        assertTrue("rowPtr[2] must equal nnz=2" in badEnd.message.orEmpty(), "got: ${badEnd.message}")
        // Non-monotone rowPtr.
        val nonMono = assertFailsWith<IllegalArgumentException> {
            sparseMatmul(values, ci, Tensors.i32Vector<Sym>(intArrayOf(0, 2, 2, 1, 2)), b)
        }
        assertTrue("monotone" in nonMono.message.orEmpty(), "got: ${nonMono.message}")
        // colIdx out of the dense operand's row range.
        val badCol = assertFailsWith<IllegalArgumentException> {
            sparseMatmul(values, Tensors.i32Vector<Sym>(intArrayOf(0, 2)), Tensors.i32Vector<Sym>(intArrayOf(0, 1, 2)), b)
        }
        assertTrue("out of range" in badCol.message.orEmpty(), "got: ${badCol.message}")
        // The interpreter arm carries the same invariant — one representative
        // pin (the mirrored messages are asserted by construction).
        val fn = DxirBuilder.function("bad_spmm") {
            val v = param("v", DxirType(F32, listOf(2)))
            val bb = param("b", DxirType(F32, listOf(2, 2)))
            val cci = const(floatArrayOf(0f, 1f), DxirType(I32, listOf(2)))
            val rrp = const(floatArrayOf(0f, 1f, 1f), DxirType(I32, listOf(3)))
            listOf(op(OpKind.SPARSE_MATMUL, listOf(v, cci, rrp, bb), DxirType(F32, listOf(2, 2))))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f), FloatArray(4)))
        }
        assertTrue("rowPtr[2] must equal nnz=2" in ex.message.orEmpty(), "got: ${ex.message}")
    }
}
