package io.tlaloc.core

/**
 * §0.4.417 — Phase E1a: the `:core` host sparse type (rank-2 CSR).
 *
 * DiffKT parity target per docs/SPARSE_PARITY_AUDIT.md (ratified by Pedro,
 * 2026-09-20): `SparseFloatTensor`'s actually-exercised surface is rank-2 CSR
 * arithmetic — elementwise `plus`/`minus`/`times`, `transpose`, `matmul` —
 * behind an Eigen JNI shim. This type is the pure-Kotlin host-level answer:
 * three parallel arrays in the classic CSR layout,
 *
 *   - `values [nnz] F32` — the stored entries, row-major,
 *   - `colIdx [nnz] I32` — each entry's column, STRICTLY increasing within a
 *     row (canonical CSR: sorted, no duplicates),
 *   - `rowPtr [rows+1] I32` — row `i`'s entries live at
 *     `values[rowPtr[i] until rowPtr[i+1]]`, so `rowPtr[0] == 0`,
 *     `rowPtr[rows] == nnz`, and the array is monotone non-decreasing.
 *
 * Construction validates the whole invariant loudly (the audit's
 * `nonZeroIndices` trap is exactly what happens when a sparse type trusts its
 * own internals); every op below therefore ASSUMES canonical inputs and
 * produces canonical outputs, which is what makes the two-pointer merges
 * correct.
 *
 * Explicit zeros: a stored entry whose value happens to be `0f` is legal
 * (matching Eigen/scipy). `fromCoo` keeps caller-provided zeros; the
 * elementwise union merges keep a position even when the combined value
 * cancels to zero (structure is the union of structures, not a numeric
 * re-sparsification); the intersection merge (`times`) and SpGEMM keep every
 * STRUCTURALLY touched position. `toDense` is always exact regardless.
 *
 * NOT here, by ratified decision: `matdiv` (DiffKT's is SparseLU via explicit
 * inverse through Eigen JNI — skipped, a solver arrives as its own designed
 * feature or never); rank > 2 (DiffKT's hierarchical nesting is where their
 * broken surface lives); any AD participation (sparse is primal-only in
 * DiffKT too — the `grad {}`-relevant `SPARSE_MATMUL` op is Phase E1b/E1c).
 * GPU: pinned emit refusal when E1b lands (StableHLO has no sparse types).
 */
class SparseTensor(
    val rows: Int,
    val cols: Int,
    val values: FloatArray,
    val colIdx: IntArray,
    val rowPtr: IntArray,
) {

    init {
        require(rows >= 0 && cols >= 0) { "SparseTensor dims must be non-negative: [$rows, $cols]" }
        require(rowPtr.size == rows + 1) {
            "rowPtr must have rows+1=${rows + 1} entries, got ${rowPtr.size}"
        }
        require(values.size == colIdx.size) {
            "values (${values.size}) and colIdx (${colIdx.size}) must be parallel arrays"
        }
        require(rowPtr[0] == 0) { "rowPtr[0] must be 0, got ${rowPtr[0]}" }
        require(rowPtr[rows] == values.size) {
            "rowPtr[$rows] must equal nnz=${values.size}, got ${rowPtr[rows]}"
        }
        for (i in 0 until rows) {
            require(rowPtr[i] <= rowPtr[i + 1]) {
                "rowPtr must be monotone non-decreasing: rowPtr[$i]=${rowPtr[i]} > rowPtr[${i + 1}]=${rowPtr[i + 1]}"
            }
            var prev = -1
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                val c = colIdx[k]
                require(c in 0 until cols) {
                    "colIdx[$k]=$c out of range [0, $cols) in row $i"
                }
                require(c > prev) {
                    "colIdx must be strictly increasing within row $i (canonical CSR): " +
                        "colIdx[$k]=$c after $prev"
                }
                prev = c
            }
        }
    }

    /** Number of stored entries (explicit zeros included). */
    val nnz: Int get() = values.size

    val dims: IntArray get() = intArrayOf(rows, cols)

    override fun toString(): String = "SparseTensor(dims=[$rows, $cols], nnz=$nnz)"

    /** Densify. Exact: writes each stored entry at its position over a zero field. */
    fun <R : ShapeAtom, C : ShapeAtom> toDense(): DTensor<Rank2<R, C>, F32> {
        val out = FloatArray(rows * cols)
        for (i in 0 until rows) {
            val rowOff = i * cols
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                out[rowOff + colIdx[k]] = values[k]
            }
        }
        return DTensor(HostF32Storage(out), intArrayOf(rows, cols), F32)
    }

    /**
     * Elementwise sum: union merge — a position stored in either operand is
     * stored in the result (even when the values cancel to 0f).
     */
    operator fun plus(other: SparseTensor): SparseTensor =
        mergeUnion(this, other) { a, b -> a + b }

    /** Elementwise difference: union merge, like [plus]. */
    operator fun minus(other: SparseTensor): SparseTensor =
        mergeUnion(this, other) { a, b -> a - b }

    /**
     * Elementwise product: intersection merge — a position must be stored in
     * BOTH operands to survive (a product with a structural zero is a
     * structural zero).
     */
    operator fun times(other: SparseTensor): SparseTensor {
        requireSameDims(other, "times")
        val outValues = ArrayList<Float>()
        val outCols = ArrayList<Int>()
        val outRowPtr = IntArray(rows + 1)
        for (i in 0 until rows) {
            var ka = rowPtr[i]
            var kb = other.rowPtr[i]
            val endA = rowPtr[i + 1]
            val endB = other.rowPtr[i + 1]
            while (ka < endA && kb < endB) {
                val ca = colIdx[ka]
                val cb = other.colIdx[kb]
                when {
                    ca == cb -> {
                        outValues.add(values[ka] * other.values[kb])
                        outCols.add(ca)
                        ka++
                        kb++
                    }
                    ca < cb -> ka++
                    else -> kb++
                }
            }
            outRowPtr[i + 1] = outValues.size
        }
        return SparseTensor(rows, cols, outValues.toFloatArray(), outCols.toIntArray(), outRowPtr)
    }

    /**
     * Elementwise product against a DENSE rank-2 operand — the DiffKT `zip`
     * case (their sparse ⊗ dense `times`). The result keeps THIS tensor's
     * sparsity structure exactly (`colIdx`/`rowPtr` reused): every stored
     * position is multiplied by the dense value at that position, and dense
     * values at unstored positions are annihilated by the structural zero.
     */
    operator fun <R : ShapeAtom, C : ShapeAtom> times(dense: DTensor<Rank2<R, C>, F32>): SparseTensor {
        require(dense.rank == 2 && dense.dims[0] == rows && dense.dims[1] == cols) {
            "sparse×dense elementwise times dims mismatch: [$rows, $cols] vs ${dense.dims.toList()}"
        }
        val d = dense.hostF32()
        val outValues = FloatArray(nnz)
        for (i in 0 until rows) {
            val rowOff = i * cols
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                outValues[k] = values[k] * d[rowOff + colIdx[k]]
            }
        }
        return SparseTensor(rows, cols, outValues, colIdx.copyOf(), rowPtr.copyOf())
    }

    /**
     * Transpose, as an explicit half-permutation (counting sort by column):
     * count entries per column, prefix-sum into the new rowPtr, then scatter
     * each entry — scanning source rows in order, so each output row (an old
     * column) receives its entries in increasing old-row order, which is
     * exactly the canonical strictly-increasing colIdx invariant of the
     * result. O(nnz + rows + cols), no comparison sort.
     */
    fun transpose(): SparseTensor {
        val outRowPtr = IntArray(cols + 1)
        for (k in 0 until nnz) outRowPtr[colIdx[k] + 1]++
        for (c in 0 until cols) outRowPtr[c + 1] += outRowPtr[c]
        val outValues = FloatArray(nnz)
        val outCols = IntArray(nnz)
        val next = outRowPtr.copyOf()
        for (i in 0 until rows) {
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                val c = colIdx[k]
                val pos = next[c]++
                outValues[pos] = values[k]
                outCols[pos] = i
            }
        }
        return SparseTensor(cols, rows, outValues, outCols, outRowPtr)
    }

    /**
     * SpMM: sparse `[N, C]` × dense `[C, D]` → dense `[N, D]` — the GNN
     * adjacency-times-features kernel, the one matmul DiffKT sparse users
     * could actually chain. Per output row, a `Double` accumulator buffer
     * (the house reduction convention) collects `v · dense[col, :]` over the
     * row's stored entries — increasing-column order, i.e. the dense
     * contraction order with the zero terms skipped — then narrows to F32.
     */
    infix fun <K : ShapeAtom, C : ShapeAtom> matmul(
        dense: DTensor<Rank2<K, C>, F32>,
    ): DTensor<Rank2<Sym, C>, F32> {
        require(dense.rank == 2 && dense.dims[0] == cols) {
            "SpMM inner dim mismatch: [$rows, $cols] x ${dense.dims.toList()}"
        }
        val d = dense.dims[1]
        val b = dense.hostF32()
        val out = FloatArray(rows * d)
        val acc = DoubleArray(d)
        for (i in 0 until rows) {
            acc.fill(0.0)
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                val v = values[k].toDouble()
                val bOff = colIdx[k] * d
                for (j in 0 until d) acc[j] += v * b[bOff + j]
            }
            val rowOff = i * d
            for (j in 0 until d) out[rowOff + j] = acc[j].toFloat()
        }
        return DTensor(HostF32Storage(out), intArrayOf(rows, d), F32)
    }

    /**
     * SpGEMM: sparse `[N, C]` × sparse `[C, P]` → sparse `[N, P]`, by
     * row-wise accumulation into a dense `Double` row buffer (Gustavson's
     * algorithm — the simple correct choice over hash-map scratch): for each
     * entry `(i, c, v)` of this tensor, scatter `v · B[c, :]` into the
     * buffer, tracking which columns were structurally touched; then read the
     * touched columns out in sorted order as row `i` of the result.
     *
     * Complexity: O(Σᵢ Σ_{k ∈ rowᵢ(A)} nnz(B row colIdx[k])) flops (the
     * irreducible SpGEMM work) + O(tᵢ log tᵢ) per row for the touched-column
     * sort + O(P) scratch memory, reset per row in O(tᵢ) by walking the
     * touched list (never an O(rows·P) clear). A touched position stays a
     * stored entry even when the accumulated value cancels to 0f (structural,
     * matching the elementwise union convention).
     */
    infix fun matmul(other: SparseTensor): SparseTensor {
        require(cols == other.rows) {
            "SpGEMM inner dim mismatch: [$rows, $cols] x [${other.rows}, ${other.cols}]"
        }
        val p = other.cols
        val acc = DoubleArray(p)
        val touched = BooleanArray(p)
        val touchedCols = IntArray(p)
        val outValues = ArrayList<Float>()
        val outCols = ArrayList<Int>()
        val outRowPtr = IntArray(rows + 1)
        for (i in 0 until rows) {
            var t = 0
            for (k in rowPtr[i] until rowPtr[i + 1]) {
                val v = values[k].toDouble()
                val c = colIdx[k]
                for (kb in other.rowPtr[c] until other.rowPtr[c + 1]) {
                    val j = other.colIdx[kb]
                    if (!touched[j]) {
                        touched[j] = true
                        touchedCols[t++] = j
                        acc[j] = 0.0
                    }
                    acc[j] += v * other.values[kb]
                }
            }
            val sorted = touchedCols.copyOfRange(0, t)
            sorted.sort()
            for (s in 0 until t) {
                val j = sorted[s]
                outValues.add(acc[j].toFloat())
                outCols.add(j)
                touched[j] = false
            }
            outRowPtr[i + 1] = outValues.size
        }
        return SparseTensor(rows, p, outValues.toFloatArray(), outCols.toIntArray(), outRowPtr)
    }

    private fun requireSameDims(other: SparseTensor, op: String) {
        require(rows == other.rows && cols == other.cols) {
            "$op dims mismatch: [$rows, $cols] vs [${other.rows}, ${other.cols}]"
        }
    }

    companion object {

        /**
         * Build a canonical CSR from COO triples. Input order is free (the
         * triples are sorted by `(row, col)` first); DUPLICATES at the same
         * position SUM — the scatter-add convention, so a COO built by
         * accumulation (an embedding-gradient-shaped producer) lands
         * correctly. Row and column indices are range-checked loudly here;
         * the constructor re-validates the assembled invariant.
         */
        fun fromCoo(
            rows: Int,
            cols: Int,
            rowIdx: IntArray,
            colIdx: IntArray,
            values: FloatArray,
        ): SparseTensor {
            require(rowIdx.size == colIdx.size && colIdx.size == values.size) {
                "fromCoo parallel arrays disagree: rowIdx=${rowIdx.size}, colIdx=${colIdx.size}, values=${values.size}"
            }
            for (k in rowIdx.indices) {
                require(rowIdx[k] in 0 until rows) {
                    "fromCoo rowIdx[$k]=${rowIdx[k]} out of range [0, $rows)"
                }
                require(colIdx[k] in 0 until cols) {
                    "fromCoo colIdx[$k]=${colIdx[k]} out of range [0, $cols)"
                }
            }
            val order = (0 until rowIdx.size).sortedWith(
                compareBy({ rowIdx[it] }, { colIdx[it] }),
            )
            val outValues = ArrayList<Float>(order.size)
            val outCols = ArrayList<Int>(order.size)
            val outRowPtr = IntArray(rows + 1)
            var prevRow = -1
            var prevCol = -1
            for (idx in order) {
                val r = rowIdx[idx]
                val c = colIdx[idx]
                if (r == prevRow && c == prevCol) {
                    // Duplicate position: sum, the scatter-add convention.
                    outValues[outValues.size - 1] = outValues[outValues.size - 1] + values[idx]
                } else {
                    while (prevRow < r) outRowPtr[++prevRow] = outValues.size
                    outValues.add(values[idx])
                    outCols.add(c)
                    prevCol = c
                }
            }
            while (prevRow < rows) outRowPtr[++prevRow] = outValues.size
            return SparseTensor(rows, cols, outValues.toFloatArray(), outCols.toIntArray(), outRowPtr)
        }

        // NOTE: self-contained dims require rather than the outer class's private
        // requireSameDims — kotlinc emits a bad invokespecial (VerifyError at class
        // load) for a Companion → outer-private call from an inline function, the
        // same family of signature-plumbing bug as the VarHandle one on the books.
        private inline fun mergeUnion(
            a: SparseTensor,
            b: SparseTensor,
            f: (Float, Float) -> Float,
        ): SparseTensor {
            require(a.rows == b.rows && a.cols == b.cols) {
                "elementwise dims mismatch: [${a.rows}, ${a.cols}] vs [${b.rows}, ${b.cols}]"
            }
            val outValues = ArrayList<Float>(a.nnz + b.nnz)
            val outCols = ArrayList<Int>(a.nnz + b.nnz)
            val outRowPtr = IntArray(a.rows + 1)
            for (i in 0 until a.rows) {
                var ka = a.rowPtr[i]
                var kb = b.rowPtr[i]
                val endA = a.rowPtr[i + 1]
                val endB = b.rowPtr[i + 1]
                while (ka < endA || kb < endB) {
                    val ca = if (ka < endA) a.colIdx[ka] else Int.MAX_VALUE
                    val cb = if (kb < endB) b.colIdx[kb] else Int.MAX_VALUE
                    when {
                        ca == cb -> {
                            outValues.add(f(a.values[ka], b.values[kb]))
                            outCols.add(ca)
                            ka++
                            kb++
                        }
                        ca < cb -> {
                            outValues.add(f(a.values[ka], 0f))
                            outCols.add(ca)
                            ka++
                        }
                        else -> {
                            outValues.add(f(0f, b.values[kb]))
                            outCols.add(cb)
                            kb++
                        }
                    }
                }
                outRowPtr[i + 1] = outValues.size
            }
            return SparseTensor(a.rows, a.cols, outValues.toFloatArray(), outCols.toIntArray(), outRowPtr)
        }
    }
}
