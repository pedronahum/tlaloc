package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.stablehlo.mosaic.GridSemantics.ARBITRARY
import io.tlaloc.stablehlo.mosaic.GridSemantics.PARALLEL

/**
 * TPU kernels written with [mosaicKernel]. Each is the module Pallas lowers for its twin in
 * `harness/python/check_mosaic_kernels.py`, which checks the two are equal and runs the twin in
 * TPU interpret mode against numpy.
 */
@ExperimentalTlalocApi
object MosaicKernels {

    /** Softmax over the rows of a bf16 `[rows, cols]` array, [blockRows] rows per grid step, f32 inside. */
    fun softmaxRows(rows: Int, cols: Int, blockRows: Int, serialized: Boolean = false): String {
        require(rows % blockRows == 0 && blockRows % 16 == 0 && cols % 128 == 0) { "softmaxRows: blocks of ($blockRows, $cols) over $rows rows" }
        return mosaicKernel("tlaloc_softmax_rows", serialized) {
            grid(rows / blockRows)
            val x = input(listOf(blockRows, cols), TileType.BF16) { (i) -> listOf(i, 0) }
            val o = output(listOf(blockRows, cols), TileType.BF16) { (i) -> listOf(i, 0) }
            body {
                val xf = x.load().cast(TileType.F32)
                val m = xf.max(1)
                val e = exp(xf - m)
                val s = e.sum(1)
                o.store((e / s).cast(TileType.BF16))
            }
        }
    }

    /** RMSNorm of a bf16 `[rows, hidden]` array by a `[1, hidden]` weight, [blockRows] rows per grid step. */
    fun rmsNormRows(rows: Int, hidden: Int, blockRows: Int, eps: Float, serialized: Boolean = false): String {
        require(rows % blockRows == 0 && blockRows % 16 == 0 && hidden % 128 == 0) { "rmsNormRows: blocks of ($blockRows, $hidden) over $rows rows" }
        return mosaicKernel("tlaloc_rmsnorm_rows", serialized) {
            grid(rows / blockRows)
            val x = input(listOf(blockRows, hidden), TileType.BF16) { (i) -> listOf(i, 0) }
            val w = input(listOf(1, hidden), TileType.BF16) { listOf(0, 0) }
            val o = output(listOf(blockRows, hidden), TileType.BF16) { (i) -> listOf(i, 0) }
            body {
                val xf = x.load().cast(TileType.F32)
                val ms = (xf * xf).sum(1) / hidden.toFloat()
                val y = xf * rsqrt(ms + eps) * w.load().cast(TileType.F32)
                o.store(y.cast(TileType.BF16))
            }
        }
    }

    /** The block of flash attention: query and key blocks of 128 tokens, head dimension 128. */
    const val FLASH_BLOCK: Int = 128

    /**
     * Causal flash attention, forward, over one head of [seq] tokens (bf16 q, k, v `[seq, 128]`):
     * the output (bf16) and the log-sum-exp of each row (f32, repeated across 128 lanes, the form
     * the backward kernels read). The grid is (query block, key block); the key axis carries the
     * running maximum, the running sum (both 128 lanes wide) and the f32 accumulator in VMEM.
     */
    fun flashForward(seq: Int, serialized: Boolean = false): String {
        val t = FLASH_BLOCK
        require(seq % t == 0) { "flashForward: $seq tokens in blocks of $t" }
        val n = seq / t
        val scale = (1.0 / kotlin.math.sqrt(t.toDouble())).toFloat()
        return mosaicKernel("tlaloc_flash_fwd", serialized) {
            grid(n, n, semantics = listOf(PARALLEL, ARBITRARY))
            val q = input(listOf(t, t), TileType.BF16) { (i, _) -> listOf(i, 0) }
            val k = input(listOf(t, t), TileType.BF16) { (_, j) -> listOf(j, 0) }
            val v = input(listOf(t, t), TileType.BF16) { (_, j) -> listOf(j, 0) }
            val o = output(listOf(t, t), TileType.BF16) { (i, _) -> listOf(i, 0) }
            val lse = output(listOf(t, t), TileType.F32) { (i, _) -> listOf(i, 0) }
            val mSc = scratch(listOf(t, t), TileType.F32)
            val lSc = scratch(listOf(t, t), TileType.F32)
            val accSc = scratch(listOf(t, t), TileType.F32)
            body {
                whenEq(programId(1), 0) {
                    mSc.store(splat(Float.NEGATIVE_INFINITY, mSc.shape))
                    lSc.store(zeros(lSc))
                    accSc.store(zeros(accSc))
                }
                val s = causalMask(dot(q.load(), k.load(), transposeB = true) * scale, rowAxis = 0, colAxis = 1)
                val mPrev = mSc.load()
                val mNew = maximum(mPrev, s.max(1))
                val p = exp(s - mNew)
                val alpha = exp(mPrev - mNew)
                lSc.store(alpha * lSc.load() + p.sum(1))
                accSc.store(accSc.load() * alpha + dot(p.cast(TileType.BF16), v.load()))
                mSc.store(mNew)
                whenEq(programId(1), n - 1) {
                    o.store((accSc.load() / lSc.load()).cast(TileType.BF16))
                    lse.store(mSc.load() + log(lSc.load()))
                }
            }
        }
    }

    /**
     * Causal flash attention, backward, for dK and dV: grid (key block, query block), reading q,
     * k, v, dO, the forward's log-sum-exp and `di = rowsum(dO ⊙ O)` (both 128 lanes wide). The
     * probabilities are recomputed from the log-sum-exp; masked scores give zero probabilities,
     * so dS needs no mask of its own.
     */
    fun flashBackwardDkDv(seq: Int, serialized: Boolean = false): String {
        val t = FLASH_BLOCK
        require(seq % t == 0) { "flashBackwardDkDv: $seq tokens in blocks of $t" }
        val n = seq / t
        val scale = (1.0 / kotlin.math.sqrt(t.toDouble())).toFloat()
        return mosaicKernel("tlaloc_flash_bwd_dkdv", serialized) {
            grid(n, n, semantics = listOf(PARALLEL, ARBITRARY))
            val qBlock = { g: List<GridIndex> -> listOf<Any>(g[1], 0) }
            val kvBlock = { g: List<GridIndex> -> listOf<Any>(g[0], 0) }
            val q = input(listOf(t, t), TileType.BF16, qBlock)
            val k = input(listOf(t, t), TileType.BF16, kvBlock)
            val v = input(listOf(t, t), TileType.BF16, kvBlock)
            val dO = input(listOf(t, t), TileType.BF16, qBlock)
            val lse = input(listOf(t, t), TileType.F32, qBlock)
            val di = input(listOf(t, t), TileType.F32, qBlock)
            val dk = output(listOf(t, t), TileType.BF16, kvBlock)
            val dv = output(listOf(t, t), TileType.BF16, kvBlock)
            val dkSc = scratch(listOf(t, t), TileType.F32)
            val dvSc = scratch(listOf(t, t), TileType.F32)
            body {
                whenEq(programId(1), 0) {
                    dkSc.store(zeros(dkSc))
                    dvSc.store(zeros(dvSc))
                }
                val qv = q.load()
                val dov = dO.load()
                val s = causalMask(dot(qv, k.load(), transposeB = true) * scale, rowAxis = 1, colAxis = 0)
                val p = exp(s - lse.load())
                dvSc.store(dvSc.load() + dot(p.cast(TileType.BF16), dov, transposeA = true))
                val dp = dot(dov, v.load(), transposeB = true)
                val ds = p * (dp - di.load()) * scale
                dkSc.store(dkSc.load() + dot(ds.cast(TileType.BF16), qv, transposeA = true))
                whenEq(programId(1), n - 1) {
                    dk.store(dkSc.load().cast(TileType.BF16))
                    dv.store(dvSc.load().cast(TileType.BF16))
                }
            }
        }
    }

    /** Causal flash attention, backward, for dQ: grid (query block, key block); see [flashBackwardDkDv]. */
    fun flashBackwardDq(seq: Int, serialized: Boolean = false): String {
        val t = FLASH_BLOCK
        require(seq % t == 0) { "flashBackwardDq: $seq tokens in blocks of $t" }
        val n = seq / t
        val scale = (1.0 / kotlin.math.sqrt(t.toDouble())).toFloat()
        return mosaicKernel("tlaloc_flash_bwd_dq", serialized) {
            grid(n, n, semantics = listOf(PARALLEL, ARBITRARY))
            val qBlock = { g: List<GridIndex> -> listOf<Any>(g[0], 0) }
            val kvBlock = { g: List<GridIndex> -> listOf<Any>(g[1], 0) }
            val q = input(listOf(t, t), TileType.BF16, qBlock)
            val k = input(listOf(t, t), TileType.BF16, kvBlock)
            val v = input(listOf(t, t), TileType.BF16, kvBlock)
            val dO = input(listOf(t, t), TileType.BF16, qBlock)
            val lse = input(listOf(t, t), TileType.F32, qBlock)
            val di = input(listOf(t, t), TileType.F32, qBlock)
            val dq = output(listOf(t, t), TileType.BF16, qBlock)
            val dqSc = scratch(listOf(t, t), TileType.F32)
            body {
                whenEq(programId(1), 0) { dqSc.store(zeros(dqSc)) }
                val kv = k.load()
                val s = causalMask(dot(q.load(), kv, transposeB = true) * scale, rowAxis = 0, colAxis = 1)
                val p = exp(s - lse.load())
                val dp = dot(dO.load(), v.load(), transposeB = true)
                val ds = p * (dp - di.load()) * scale
                dqSc.store(dqSc.load() + dot(ds.cast(TileType.BF16), kv))
                whenEq(programId(1), n - 1) { dq.store(dqSc.load().cast(TileType.BF16)) }
            }
        }
    }

    /** Scores at key columns after the query row → −∞; rows and columns are offset by the grid
     * axes [rowAxis] and [colAxis] times the block. */
    private fun TileBody.causalMask(s: Tile, rowAxis: Int, colAxis: Int): Tile {
        val t = FLASH_BLOCK
        val row = iota(s.shape, 0) + programId(rowAxis) * t
        val col = iota(s.shape, 1) + programId(colAxis) * t
        return where(lessOrEqual(col, row), s, Float.NEGATIVE_INFINITY)
    }
}
