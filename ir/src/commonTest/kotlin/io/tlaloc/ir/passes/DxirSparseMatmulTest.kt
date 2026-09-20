package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.SparseTensor
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank2
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.418 — Phase E1b certification: OpKind.SPARSE_MATMUL + its fused SDDMM
 * values-adjoint at IR level.
 *
 * Oracles, strongest first:
 * 1. **The dense MatmulRule on toDense'd operands** — the same loss over the
 *    densified pattern must produce the same gradients, with `d_values[k]`
 *    matching the dense `d_A` at exactly the stored positions (and existing
 *    ONLY there: structural zeros are not inputs, so `d_values` has `nnz`
 *    entries, never `N·C`). Test data lives on the quarter-integer grid so
 *    every contraction on both paths is exact in F32 and the comparisons
 *    carry no tolerance (the E1a discipline).
 * 2. **E1a bit-exactness** — the interpreter's SPARSE_MATMUL arm is
 *    `SparseTensor.matmul` bit-for-bit, and the `transposed` form is
 *    `transpose().matmul` bit-for-bit (same Double-add sequence per output
 *    element — see the OpKind comment), asserted on seeded random patterns
 *    at densities 0/0.05/0.3/0.7 with ARBITRARY (non-grid) floats: the claim
 *    is identical arithmetic, not grid exactness.
 * 3. **JVP⇄VJP cross-identity** and forward-over-reverse (the HVP through a
 *    sparse gradient body, which contains the transposed SPARSE_MATMUL and
 *    the SDDMM) against the dense counterpart's HVP.
 *
 * Empty rows and skewed degree distributions are in every hand pattern on
 * purpose (density 0.05 on 9×7 guarantees empty rows in the random sweep
 * too); the hand CSR carries an explicit stored ZERO, which must still
 * RECEIVE a gradient (it is in the pattern — d_values[k] is a property of
 * the position, not the value).
 */
class DxirSparseMatmulTest {

    private val scalar = DxirType(F32, emptyList())

    // ---- the hand-built small CSR (N=3, C=4, D=2) ------------------------
    //
    //   A = [ 1.5   0   -0.75  0.5 ]     row 1 EMPTY, row 0 skewed (3 of 5
    //       [ 0     0    0     0   ]     entries), entry (2,2) an explicit
    //       [ 0     2.25 0*    0   ]     stored ZERO (*).
    private val n = 3
    private val c = 4
    private val d = 2
    private val valuesData = floatArrayOf(1.5f, -0.75f, 0.5f, 2.25f, 0f)
    private val colIdxData = intArrayOf(0, 2, 3, 1, 2)
    private val rowPtrData = intArrayOf(0, 3, 3, 5)
    private val nnz = valuesData.size
    private val bData = floatArrayOf(0.25f, -1f, 0.5f, 0.75f, -0.5f, 1.25f, 2f, -0.25f)
    private val wData = floatArrayOf(1f, -0.5f, 0.75f, 0.25f, -1.25f, 0.5f)

    private fun denseA(): FloatArray {
        val a = FloatArray(n * c)
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                a[i * c + colIdxData[k]] = valuesData[k]
            }
        }
        return a
    }

    private fun intConstData(v: IntArray): FloatArray = FloatArray(v.size) { v[it].toFloat() }

    /** loss = Σ (SPARSE_MATMUL(values, ci, rp, b) ⊙ w) with ci/rp as consts. */
    private fun sparseLossFn(): DxirFunction = DxirBuilder.function("sparse_loss") {
        val values = param("values", DxirType(F32, listOf(nnz)))
        val b = param("b", DxirType(F32, listOf(c, d)))
        val w = param("w", DxirType(F32, listOf(n, d)))
        val ci = const(intConstData(colIdxData), DxirType(I32, listOf(nnz)))
        val rp = const(intConstData(rowPtrData), DxirType(I32, listOf(n + 1)))
        val y = op(OpKind.SPARSE_MATMUL, listOf(values, ci, rp, b), DxirType(F32, listOf(n, d)))
        val p = op(OpKind.MUL, listOf(y, w), DxirType(F32, listOf(n, d)))
        listOf(op(OpKind.SUM, listOf(p), scalar))
    }

    /** The densified twin: loss = Σ (MATMUL(aDense, b) ⊙ w). */
    private fun denseLossFn(): DxirFunction = DxirBuilder.function("dense_loss") {
        val a = param("a", DxirType(F32, listOf(n, c)))
        val b = param("b", DxirType(F32, listOf(c, d)))
        val w = param("w", DxirType(F32, listOf(n, d)))
        val y = op(OpKind.MATMUL, listOf(a, b), DxirType(F32, listOf(n, d)))
        val p = op(OpKind.MUL, listOf(y, w), DxirType(F32, listOf(n, d)))
        listOf(op(OpKind.SUM, listOf(p), scalar))
    }

    // ---- deterministic random CSR generation (the E1a Gen, arbitrary floats) --

    private class Gen(
        val sparse: SparseTensor,
        val values: FloatArray,
        val colIdx: IntArray,
        val rowPtr: IntArray,
    )

    private fun randomGen(rnd: Random, rows: Int, cols: Int, density: Double): Gen {
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
        val s = SparseTensor.fromCoo(
            rows, cols, rowIdx.toIntArray(), colIdx.toIntArray(), values.toFloatArray(),
        )
        return Gen(s, s.values, s.colIdx, s.rowPtr)
    }

    // ---- 1. interpreter ⇄ E1a bit-exactness ------------------------------

    @Test
    fun spmmInterpreterMatchesE1aHostBitExact() {
        val rnd = Random(4181)
        for (density in doubleArrayOf(0.0, 0.05, 0.3, 0.7)) {
            val rows = 9
            val cols = 7
            val dOut = 4
            val g = randomGen(rnd, rows, cols, density)
            val dense = FloatArray(cols * dOut) { rnd.nextFloat() * 4f - 2f }
            val fn = DxirBuilder.function("spmm") {
                val values = param("values", DxirType(F32, listOf(g.values.size)))
                val b = param("b", DxirType(F32, listOf(cols, dOut)))
                val ci = const(intConstData(g.colIdx), DxirType(I32, listOf(g.colIdx.size)))
                val rp = const(intConstData(g.rowPtr), DxirType(I32, listOf(rows + 1)))
                listOf(
                    op(
                        OpKind.SPARSE_MATMUL, listOf(values, ci, rp, b),
                        DxirType(F32, listOf(rows, dOut)),
                    ),
                )
            }
            val got = DxirInterpreter.evalFunction(fn, listOf(g.values, dense)).single()
            val want = (
                g.sparse matmul DTensor<Rank2<Sym, Sym>, F32>(
                    HostF32Storage(dense.copyOf()), intArrayOf(cols, dOut), F32,
                )
                ).hostF32()
            assertContentEquals(
                want, got,
                "density $density: interpreter SPARSE_MATMUL diverges from SparseTensor.matmul",
            )
        }
    }

    @Test
    fun transposedSpmmInterpreterMatchesE1aTransposeBitExact() {
        val rnd = Random(4182)
        for (density in doubleArrayOf(0.0, 0.05, 0.3, 0.7)) {
            val rows = 8
            val cols = 6
            val dOut = 3
            val g = randomGen(rnd, rows, cols, density)
            val up = FloatArray(rows * dOut) { rnd.nextFloat() * 4f - 2f }
            val fn = DxirBuilder.function("spmm_t") {
                val values = param("values", DxirType(F32, listOf(g.values.size)))
                val u = param("u", DxirType(F32, listOf(rows, dOut)))
                val tmpl = param("tmpl", DxirType(F32, listOf(cols, dOut)))
                val ci = const(intConstData(g.colIdx), DxirType(I32, listOf(g.colIdx.size)))
                val rp = const(intConstData(g.rowPtr), DxirType(I32, listOf(rows + 1)))
                listOf(
                    op(
                        OpKind.SPARSE_MATMUL, listOf(values, ci, rp, u, tmpl),
                        DxirType(F32, listOf(cols, dOut)),
                        attrs = mapOf("transposed" to true),
                    ),
                )
            }
            // The template operand is SHAPE-ONLY: garbage values must not
            // change the result (they are never evaluated into the walk).
            val garbageTemplate = FloatArray(cols * dOut) { Float.NaN }
            val got = DxirInterpreter.evalFunction(fn, listOf(g.values, up, garbageTemplate)).single()
            val want = (
                g.sparse.transpose() matmul DTensor<Rank2<Sym, Sym>, F32>(
                    HostF32Storage(up.copyOf()), intArrayOf(rows, dOut), F32,
                )
                ).hostF32()
            assertContentEquals(
                want, got,
                "density $density: transposed SPARSE_MATMUL diverges from transpose().matmul",
            )
        }
    }

    // ---- 2. the dense-MatmulRule oracle ----------------------------------

    @Test
    fun sparseGradientsMatchDenseMatmulReference() {
        val sparseGrads = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(sparseLossFn()),
            listOf(valuesData, bData, wData),
        )
        val denseGrads = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(denseLossFn()),
            listOf(denseA(), bData, wData),
        )
        val dValues = sparseGrads[0]
        val dB = sparseGrads[1]
        val dW = sparseGrads[2]
        val dADense = denseGrads[0]

        // d_values exists ONLY for stored positions — nnz entries, never N·C.
        assertEquals(nnz, dValues.size, "d_values must have exactly nnz entries")
        // Each stored position's gradient equals the dense d_A there — grid
        // data, both paths exact, NO tolerance.
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                assertEquals(
                    dADense[i * c + colIdxData[k]], dValues[k],
                    "d_values[$k] (position ($i, ${colIdxData[k]})) diverges from dense d_A",
                )
            }
        }
        // The explicit stored ZERO still receives its gradient (position
        // (2, 2)): Σ_j w[2,j]·b[2,j] = (-1.25)(-0.5) + 0.5·1.25 = 1.25.
        assertEquals(1.25f, dValues[4], "the stored-zero entry's gradient")
        assertContentEquals(dADense.let { a ->
            FloatArray(nnz) { k ->
                var i = 0
                while (rowPtrData[i + 1] <= k) i++
                a[i * c + colIdxData[k]]
            }
        }, dValues, "d_values as a whole diverges from the dense d_A gather")
        // d_b and d_w match the dense twin exactly (grid data).
        assertContentEquals(denseGrads[1], dB, "d_dense (Aᵀ·upstream) diverges from the dense d_B")
        assertContentEquals(denseGrads[2], dW, "d_w diverges from the dense twin")
        // And an analytic spot pin for the empty row: it contributes to no
        // gradient, so d_w's middle row is exactly Y's middle row = 0.
        assertTrue(dW[2] == 0f && dW[3] == 0f, "empty sparse row must leave d_w's row exactly zero")
    }

    @Test
    fun transposedPrimalDifferentiatesBothWays() {
        // Hand-built TRANSPOSED primal — the node SparseMatmulRule itself
        // emits: loss = Σ (Aᵀ·U ⊙ w). Certifies the rule's transposed branch
        // (d_U = A·upstream, d_values with the roles swapped) and that the
        // shape-only template param collects a zero gradient.
        val fn = DxirBuilder.function("sparse_t_loss") {
            val values = param("values", DxirType(F32, listOf(nnz)))
            val u = param("u", DxirType(F32, listOf(n, d)))
            val tmpl = param("tmpl", DxirType(F32, listOf(c, d)))
            val w = param("w", DxirType(F32, listOf(c, d)))
            val ci = const(intConstData(colIdxData), DxirType(I32, listOf(nnz)))
            val rp = const(intConstData(rowPtrData), DxirType(I32, listOf(n + 1)))
            val y = op(
                OpKind.SPARSE_MATMUL, listOf(values, ci, rp, u, tmpl),
                DxirType(F32, listOf(c, d)),
                attrs = mapOf("transposed" to true),
            )
            val p = op(OpKind.MUL, listOf(y, w), DxirType(F32, listOf(c, d)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val uData = floatArrayOf(0.5f, -1.5f, 2f, 0.25f, -0.75f, 1f)
        val wT = floatArrayOf(1.25f, -0.25f, 0.5f, 2f, -1f, 0.75f, 0.25f, -0.5f)
        val tmplData = FloatArray(c * d) { 9f } // values must never matter
        val grads = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(fn),
            listOf(valuesData, uData, tmplData, wT),
        )
        val a = denseA()
        // d_values[k] = Σ_j w[colIdx k, j] · u[row k, j] (roles swapped).
        val wantDValues = FloatArray(nnz)
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                var acc = 0.0
                for (j in 0 until d) acc += wT[colIdxData[k] * d + j].toDouble() * uData[i * d + j]
                wantDValues[k] = acc.toFloat()
            }
        }
        assertContentEquals(wantDValues, grads[0], "transposed primal's d_values")
        // d_u = A · w (the plain product).
        val wantDU = FloatArray(n * d)
        for (i in 0 until n) {
            for (cc in 0 until c) {
                val av = a[i * c + cc]
                for (j in 0 until d) wantDU[i * d + j] += av * wT[cc * d + j]
            }
        }
        assertContentEquals(wantDU, grads[1], "transposed primal's d_u must be A·w")
        // The shape-only template gets NO contribution — a typed zero.
        assertContentEquals(FloatArray(c * d), grads[2], "template param must collect a zero gradient")
    }

    // ---- 3. structural-zero integer slots --------------------------------

    @Test
    fun integerComponentsGetStructuralZeroGradients() {
        // colIdx/rowPtr as I32 PARAMS: `grad {}` still returns one gradient
        // per param, and the §0.4.54 integer arm types them as structural
        // zeros — the rule itself never contributes to them.
        val fn = DxirBuilder.function("sparse_int_params") {
            val values = param("values", DxirType(F32, listOf(nnz)))
            val ci = param("ci", DxirType(I32, listOf(nnz)))
            val rp = param("rp", DxirType(I32, listOf(n + 1)))
            val b = param("b", DxirType(F32, listOf(c, d)))
            val w = param("w", DxirType(F32, listOf(n, d)))
            val y = op(OpKind.SPARSE_MATMUL, listOf(values, ci, rp, b), DxirType(F32, listOf(n, d)))
            val p = op(OpKind.MUL, listOf(y, w), DxirType(F32, listOf(n, d)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grads = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(fn),
            listOf(valuesData, intConstData(colIdxData), intConstData(rowPtrData), bData, wData),
        )
        assertEquals(5, grads.size)
        assertContentEquals(FloatArray(nnz), grads[1], "d_colIdx must be a structural zero")
        assertContentEquals(FloatArray(n + 1), grads[2], "d_rowPtr must be a structural zero")
        // The float gradients are unaffected by the params-vs-consts spelling.
        val viaConsts = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(sparseLossFn()),
            listOf(valuesData, bData, wData),
        )
        assertContentEquals(viaConsts[0], grads[0])
        assertContentEquals(viaConsts[1], grads[3])
        assertContentEquals(viaConsts[2], grads[4])
    }

    // ---- 4. JVP⇄VJP cross-identity and forward-over-reverse --------------

    @Test
    fun sparseMatmulJvpVjpCrossIdentity() {
        val fn = sparseLossFn()
        val vValues = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f)
        val vB = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.19f, -0.07f, 0.23f, -0.05f)
        val vW = floatArrayOf(0.41f, -0.37f, 0.29f, -0.19f, 0.07f, 0.17f)

        val grads = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(fn), listOf(valuesData, bData, wData),
        )
        var dot = 0.0
        for (i in vValues.indices) dot += grads[0][i].toDouble() * vValues[i]
        for (i in vB.indices) dot += grads[1][i].toDouble() * vB[i]
        for (i in vW.indices) dot += grads[2][i].toDouble() * vW[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn),
            listOf(valuesData, bData, wData, vValues, vB, vW),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through SPARSE_MATMUL: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    @Test
    fun hvpComposesThroughSparseGradientBodyAgainstDenseReference() {
        // Forward-over-reverse: the sparse gradient body contains the
        // TRANSPOSED SPARSE_MATMUL and the SDDMM values-adjoint, so this is
        // the cert for both ops' forward tangent arms. The oracle is the
        // dense twin's HVP, with the tangent direction for `values` embedded
        // at the pattern positions (zeros elsewhere — structural zeros carry
        // no perturbation by definition). Grid data: exact, no tolerance.
        val vValues = floatArrayOf(0.25f, -0.5f, 0.75f, 0.5f, -0.25f)
        val vB = floatArrayOf(0.5f, -0.25f, 0.75f, 0.25f, -0.5f, 1f, 0.25f, -0.75f)
        val vW = floatArrayOf(-0.5f, 0.25f, 0.75f, -0.25f, 0.5f, -1f)
        val vADense = FloatArray(n * c)
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                vADense[i * c + colIdxData[k]] = vValues[k]
            }
        }
        val sparseHvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(DxirReverseTransform.apply(sparseLossFn())),
            listOf(valuesData, bData, wData, vValues, vB, vW),
        )
        val denseHvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(DxirReverseTransform.apply(denseLossFn())),
            listOf(denseA(), bData, wData, vADense, vB, vW),
        )
        // Outputs: [g_values, g_b, g_w, dg_values, dg_b, dg_w] (dense: g_a…).
        assertEquals(6, sparseHvp.size)
        // Tangent-of-gradient for values ↔ dense d_A at the stored positions.
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                assertEquals(
                    denseHvp[3][i * c + colIdxData[k]], sparseHvp[3][k],
                    "HVP d(d_values[$k]) diverges from the dense reference",
                )
            }
        }
        assertContentEquals(denseHvp[4], sparseHvp[4], "HVP d(d_b) diverges from the dense reference")
        assertContentEquals(denseHvp[5], sparseHvp[5], "HVP d(d_w) diverges from the dense reference")
    }

    // ---- 5. hand pins: empty rows, values-adjoint formula, rule shape -----

    @Test
    fun valuesAdjointHandPinAndEmptyRowBehaviour() {
        // Direct SDDMM pin over the hand CSR: d_values[k] =
        // Σ_j up[row k, j] · b[colIdx k, j], and the empty row's absence
        // means no k maps to row 1 — nothing reads up[1, :].
        val upData = floatArrayOf(2f, -1f, 100f, 100f, 0.5f, 1.5f) // row 1 poisoned
        val fn = DxirBuilder.function("sddmm") {
            val up = param("up", DxirType(F32, listOf(n, d)))
            val b = param("b", DxirType(F32, listOf(c, d)))
            val ci = const(intConstData(colIdxData), DxirType(I32, listOf(nnz)))
            val rp = const(intConstData(rowPtrData), DxirType(I32, listOf(n + 1)))
            listOf(
                op(
                    OpKind.SPARSE_MATMUL_VALUES_ADJOINT, listOf(up, b, ci, rp),
                    DxirType(F32, listOf(nnz)),
                ),
            )
        }
        val got = DxirInterpreter.evalFunction(fn, listOf(upData, bData)).single()
        val want = FloatArray(nnz)
        for (i in 0 until n) {
            for (k in rowPtrData[i] until rowPtrData[i + 1]) {
                var acc = 0.0
                for (j in 0 until d) acc += upData[i * d + j].toDouble() * bData[colIdxData[k] * d + j]
                want[k] = acc.toFloat()
            }
        }
        assertContentEquals(want, got, "SDDMM hand pin")
        // The poisoned empty-row upstream never leaked into any entry.
        assertTrue(got.none { abs(it) > 50f }, "empty row's upstream must be unread: ${got.toList()}")
    }

    @Test
    fun ruleEmitsTheFusedAdjointShapes() {
        // Structural pin on the gradient graph: exactly one SDDMM (4 operands)
        // and exactly one TRANSPOSED SPARSE_MATMUL carrying the 5-operand
        // template spelling — the sentinel-safe design (transpose at
        // EXECUTION time, C off the template's runtime shape, nothing baked).
        val grad = DxirReverseTransform.apply(sparseLossFn())
        val sddmm = grad.body.filterIsInstance<DxirOp>()
            .single { it.op == OpKind.SPARSE_MATMUL_VALUES_ADJOINT }
        assertEquals(4, sddmm.operands.size)
        assertContentEquals(listOf(nnz), sddmm.type.dims)
        val transposedSpmm = grad.body.filterIsInstance<DxirOp>()
            .single { it.op == OpKind.SPARSE_MATMUL && it.attrs["transposed"] == true }
        assertEquals(5, transposedSpmm.operands.size, "the adjoint must carry the shape-only template")
        assertContentEquals(listOf(c, d), transposedSpmm.type.dims)
    }
}
