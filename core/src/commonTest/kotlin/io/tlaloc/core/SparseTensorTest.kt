package io.tlaloc.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * §0.4.417 — Phase E1a certification: the host CSR type against dense
 * references.
 *
 * Numeric exactness discipline: every randomized value is drawn from the
 * quarter-integer grid `k/4, k ∈ [-8, 8]`, so elementwise sums/differences/
 * products, SpMM contractions and SpGEMM contractions over the small test
 * extents are exactly representable in F32 (products live on the 1/16 grid,
 * accumulations stay far under the 2^24 integer window after the 1/16
 * scaling), and the dense references — written with the SAME Double
 * accumulator and increasing-k contraction order the sparse kernels use — can
 * be compared with `assertContentEquals`, no tolerance. Skipping a zero term
 * never changes a Double accumulation of finite grid values, so the sparse
 * kernels' zero-skipping is exactness-neutral.
 */
class SparseTensorTest {

    // ---- deterministic random sparse generation -------------------------

    /** Dense row-major reference + the shuffled-COO-constructed sparse twin. */
    private class Gen(val dense: FloatArray, val sparse: SparseTensor)

    private fun randomGen(rnd: Random, rows: Int, cols: Int, density: Double): Gen {
        val dense = FloatArray(rows * cols)
        val rowIdx = ArrayList<Int>()
        val colIdx = ArrayList<Int>()
        val values = ArrayList<Float>()
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (rnd.nextDouble() < density) {
                    // Quarter-integer grid, zero excluded so density is honest.
                    var q = rnd.nextInt(-8, 9)
                    if (q == 0) q = 3
                    val v = q * 0.25f
                    dense[i * cols + j] = v
                    rowIdx.add(i)
                    colIdx.add(j)
                    values.add(v)
                }
            }
        }
        // Shuffle the COO order so fromCoo's sorting is exercised every time.
        val perm = (0 until values.size).shuffled(rnd)
        val sparse = SparseTensor.fromCoo(
            rows,
            cols,
            IntArray(perm.size) { rowIdx[perm[it]] },
            IntArray(perm.size) { colIdx[perm[it]] },
            FloatArray(perm.size) { values[perm[it]] },
        )
        return Gen(dense, sparse)
    }

    /** The certification densities: all-zero, sparse-with-empty-rows, medium, dense-ish. */
    private val densities = doubleArrayOf(0.0, 0.05, 0.3, 0.7)

    /**
     * Canonicalize signed zeros: a dense reference computes `0f * negative`
     * at structurally-zero positions and gets `-0.0`, where the sparse result
     * has the structural `+0.0`. Semantically equal, and `+ 0f` folds `-0.0`
     * to `+0.0` (IEEE) without touching any nonzero value.
     */
    private fun canon(a: FloatArray): FloatArray = FloatArray(a.size) { a[it] + 0f }

    // ---- construction ----------------------------------------------------

    @Test
    fun fromCooToDenseRoundTrip() {
        // Hand pin, unsorted COO input:
        //   [ 0  5  0 ]
        //   [ 1  0  2 ]
        //   [ 0  0  0 ]
        val s = SparseTensor.fromCoo(
            3, 3,
            intArrayOf(1, 0, 1),
            intArrayOf(2, 1, 0),
            floatArrayOf(2f, 5f, 1f),
        )
        assertEquals(3, s.nnz)
        assertContentEquals(intArrayOf(0, 1, 3, 3), s.rowPtr)
        assertContentEquals(intArrayOf(1, 0, 2), s.colIdx)
        assertContentEquals(floatArrayOf(5f, 1f, 2f), s.values)
        assertContentEquals(
            floatArrayOf(0f, 5f, 0f, 1f, 0f, 2f, 0f, 0f, 0f),
            s.toDense<Sym, Sym>().hostF32(),
        )
        // Randomized round trip at every density.
        val rnd = Random(4171)
        for (d in densities) {
            val g = randomGen(rnd, 9, 7, d)
            assertContentEquals(g.dense, g.sparse.toDense<Sym, Sym>().hostF32())
        }
    }

    @Test
    fun fromCooSumsDuplicates() {
        // (0,1) appears three times and (1,0) twice — the scatter-add pin.
        val s = SparseTensor.fromCoo(
            2, 2,
            intArrayOf(0, 1, 0, 1, 0),
            intArrayOf(1, 0, 1, 0, 1),
            floatArrayOf(1.5f, 2f, 0.25f, -0.5f, 0.25f),
        )
        assertEquals(2, s.nnz)
        assertContentEquals(floatArrayOf(0f, 2f, 1.5f, 0f), s.toDense<Sym, Sym>().hostF32())
    }

    @Test
    fun fromCooKeepsExplicitZeros() {
        val s = SparseTensor.fromCoo(2, 2, intArrayOf(0), intArrayOf(0), floatArrayOf(0f))
        assertEquals(1, s.nnz)
        assertContentEquals(floatArrayOf(0f, 0f, 0f, 0f), s.toDense<Sym, Sym>().hostF32())
    }

    @Test
    fun validationRefusals() {
        val v = floatArrayOf(1f)
        val c = intArrayOf(0)
        // rowPtr wrong length.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(2, 2, v, c, intArrayOf(0, 1))
        }
        // rowPtr[0] != 0.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(2, 2, v, c, intArrayOf(1, 1, 1))
        }
        // rowPtr end != nnz.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(2, 2, v, c, intArrayOf(0, 0, 0))
        }
        // rowPtr non-monotone (endpoints valid, so THIS require is the one that fires).
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(3, 2, floatArrayOf(1f, 2f), intArrayOf(0, 1), intArrayOf(0, 2, 1, 2))
        }
        // colIdx out of range.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(2, 2, v, intArrayOf(2), intArrayOf(0, 1, 1))
        }
        // colIdx not strictly increasing within a row (duplicate).
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(1, 3, floatArrayOf(1f, 2f), intArrayOf(1, 1), intArrayOf(0, 2))
        }
        // colIdx decreasing within a row.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(1, 3, floatArrayOf(1f, 2f), intArrayOf(2, 0), intArrayOf(0, 2))
        }
        // values/colIdx not parallel.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor(1, 2, floatArrayOf(1f, 2f), intArrayOf(0), intArrayOf(0, 1))
        }
        // fromCoo index range checks.
        assertFailsWith<IllegalArgumentException> {
            SparseTensor.fromCoo(2, 2, intArrayOf(2), intArrayOf(0), floatArrayOf(1f))
        }
        assertFailsWith<IllegalArgumentException> {
            SparseTensor.fromCoo(2, 2, intArrayOf(0), intArrayOf(-1), floatArrayOf(1f))
        }
        // Dims mismatches refuse loudly too.
        val a = SparseTensor.fromCoo(2, 3, intArrayOf(0), intArrayOf(1), floatArrayOf(1f))
        val b = SparseTensor.fromCoo(3, 2, intArrayOf(0), intArrayOf(1), floatArrayOf(1f))
        assertFailsWith<IllegalArgumentException> { a + b }
        assertFailsWith<IllegalArgumentException> { a * b }
        assertFailsWith<IllegalArgumentException> { b matmul b }
        assertFailsWith<IllegalArgumentException> { a matmul Tensors.f32Zeros<Sym, Sym>(2, 4) }
        assertFailsWith<IllegalArgumentException> { a * Tensors.f32Zeros<Sym, Sym>(3, 2) }
    }

    // ---- elementwise ------------------------------------------------------

    @Test
    fun elementwiseMatchesDense() {
        val rnd = Random(4172)
        for (d1 in densities) {
            for (d2 in densities) {
                val ga = randomGen(rnd, 8, 6, d1)
                val gb = randomGen(rnd, 8, 6, d2)
                val plusRef = FloatArray(ga.dense.size) { ga.dense[it] + gb.dense[it] }
                val minusRef = FloatArray(ga.dense.size) { ga.dense[it] - gb.dense[it] }
                val timesRef = FloatArray(ga.dense.size) { ga.dense[it] * gb.dense[it] }
                assertContentEquals(canon(plusRef), canon((ga.sparse + gb.sparse).toDense<Sym, Sym>().hostF32()))
                assertContentEquals(canon(minusRef), canon((ga.sparse - gb.sparse).toDense<Sym, Sym>().hostF32()))
                assertContentEquals(canon(timesRef), canon((ga.sparse * gb.sparse).toDense<Sym, Sym>().hostF32()))
            }
        }
    }

    @Test
    fun elementwiseUnionKeepsCancelledPositionsAndIntersectionDropsSingles() {
        val a = SparseTensor.fromCoo(1, 3, intArrayOf(0, 0), intArrayOf(0, 2), floatArrayOf(1f, 4f))
        val b = SparseTensor.fromCoo(1, 3, intArrayOf(0, 0), intArrayOf(0, 1), floatArrayOf(-1f, 2f))
        val sum = a + b
        // (0,0) cancels to 0f but stays STORED (union structure, no re-sparsification).
        assertEquals(3, sum.nnz)
        assertContentEquals(floatArrayOf(0f, 2f, 4f), sum.toDense<Sym, Sym>().hostF32())
        val prod = a * b
        // Intersection: only (0,0) is stored in both.
        assertEquals(1, prod.nnz)
        assertContentEquals(floatArrayOf(-1f, 0f, 0f), prod.toDense<Sym, Sym>().hostF32())
    }

    @Test
    fun sparseTimesDenseElementwiseMatchesDenseAndPreservesStructure() {
        val rnd = Random(4173)
        for (d in densities) {
            val g = randomGen(rnd, 7, 5, d)
            val denseOp = FloatArray(7 * 5) { (rnd.nextInt(-8, 9)) * 0.25f }
            val dt = Tensors.f32Matrix<Sym, Sym>(7, 5, denseOp)
            val out = g.sparse * dt
            val ref = FloatArray(g.dense.size) { g.dense[it] * denseOp[it] }
            assertContentEquals(canon(ref), canon(out.toDense<Sym, Sym>().hostF32()))
            // The zip keeps THIS tensor's structure exactly.
            assertContentEquals(g.sparse.colIdx, out.colIdx)
            assertContentEquals(g.sparse.rowPtr, out.rowPtr)
        }
    }

    // ---- transpose --------------------------------------------------------

    @Test
    fun transposeMatchesDenseAndInvolutes() {
        val rnd = Random(4174)
        for (d in densities) {
            val g = randomGen(rnd, 6, 9, d)
            val t = g.sparse.transpose()
            val ref = FloatArray(9 * 6)
            for (i in 0 until 6) {
                for (j in 0 until 9) ref[j * 6 + i] = g.dense[i * 9 + j]
            }
            assertContentEquals(ref, t.toDense<Sym, Sym>().hostF32())
            // Involution, exactly, at the representation level (canonical CSR
            // is unique for a given structure, so array equality is the pin).
            val tt = t.transpose()
            assertContentEquals(g.sparse.values, tt.values)
            assertContentEquals(g.sparse.colIdx, tt.colIdx)
            assertContentEquals(g.sparse.rowPtr, tt.rowPtr)
        }
    }

    // ---- matmul -----------------------------------------------------------

    /** Dense SpMM reference: Double accumulator, increasing-k order — the kernel's exact semantics. */
    private fun denseMatmulRef(a: FloatArray, m: Int, k: Int, b: FloatArray, n: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var acc = 0.0
                for (p in 0 until k) acc += a[i * k + p].toDouble() * b[p * n + j].toDouble()
                out[i * n + j] = acc.toFloat()
            }
        }
        return out
    }

    @Test
    fun spmmMatchesDenseMatmul() {
        val rnd = Random(4175)
        for (d in densities) {
            val g = randomGen(rnd, 8, 6, d)
            val denseOp = FloatArray(6 * 4) { (rnd.nextInt(-8, 9)) * 0.25f }
            val dt = Tensors.f32Matrix<Sym, Sym>(6, 4, denseOp)
            val out = g.sparse matmul dt
            assertContentEquals(intArrayOf(8, 4), out.dims)
            assertContentEquals(denseMatmulRef(g.dense, 8, 6, denseOp, 4), out.hostF32())
        }
    }

    @Test
    fun spgemmMatchesDenseMatmul() {
        val rnd = Random(4176)
        for (d1 in densities) {
            for (d2 in densities) {
                val ga = randomGen(rnd, 7, 5, d1)
                val gb = randomGen(rnd, 5, 6, d2)
                val out = ga.sparse matmul gb.sparse
                assertEquals(7, out.rows)
                assertEquals(6, out.cols)
                assertContentEquals(
                    denseMatmulRef(ga.dense, 7, 5, gb.dense, 6),
                    out.toDense<Sym, Sym>().hostF32(),
                )
            }
        }
    }

    @Test
    fun spgemmAgainstHandPin() {
        // A = [ 2 0 ]   B = [ 0 1 ]   AB = [ 0 2 ]
        //     [ 0 3 ]       [ 4 0 ]        [ 12 0 ]
        val a = SparseTensor.fromCoo(2, 2, intArrayOf(0, 1), intArrayOf(0, 1), floatArrayOf(2f, 3f))
        val b = SparseTensor.fromCoo(2, 2, intArrayOf(0, 1), intArrayOf(1, 0), floatArrayOf(1f, 4f))
        val ab = a matmul b
        assertEquals(2, ab.nnz)
        assertContentEquals(floatArrayOf(0f, 2f, 12f, 0f), ab.toDense<Sym, Sym>().hostF32())
    }

    // ---- degenerate structure --------------------------------------------

    @Test
    fun allZeroMatrixAndEmptyRows() {
        // The all-zero matrix: zero nnz, every op degenerate-but-valid.
        val z = SparseTensor.fromCoo(3, 4, intArrayOf(), intArrayOf(), floatArrayOf())
        assertEquals(0, z.nnz)
        assertContentEquals(FloatArray(12), z.toDense<Sym, Sym>().hostF32())
        assertEquals(0, (z + z).nnz)
        assertEquals(0, (z * z).nnz)
        assertEquals(0, z.transpose().nnz)
        val zd = z matmul Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8) { 1f })
        assertContentEquals(FloatArray(6), zd.hostF32())
        val zz = z matmul SparseTensor.fromCoo(4, 5, intArrayOf(0), intArrayOf(0), floatArrayOf(1f))
        assertEquals(0, zz.nnz)
        assertContentEquals(intArrayOf(3, 5), zz.dims)

        // Guaranteed-empty first and last rows around a populated middle.
        val m = SparseTensor.fromCoo(4, 3, intArrayOf(1, 2, 2), intArrayOf(2, 0, 1), floatArrayOf(5f, 1f, 2f))
        assertContentEquals(intArrayOf(0, 0, 1, 3, 3), m.rowPtr)
        assertContentEquals(
            floatArrayOf(0f, 0f, 0f, 0f, 0f, 5f, 1f, 2f, 0f, 0f, 0f, 0f),
            m.toDense<Sym, Sym>().hostF32(),
        )
        // Transpose of a matrix with empty rows yields a matrix with an empty COLUMN — still valid.
        val mt = m.transpose()
        assertContentEquals(intArrayOf(0, 1, 2, 3), mt.rowPtr)
        // SpMM over empty rows produces zero rows.
        val md = m matmul Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertContentEquals(floatArrayOf(0f, 0f, 25f, 30f, 7f, 10f, 0f, 0f), md.hostF32())
    }
}
