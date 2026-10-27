package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * A gridded Mosaic matmul, `out[M, N] = x[M, K] · w[K, N]`, in [tm]×[tn] output blocks with a
 * [tk]-deep f32 accumulator in VMEM scratch, written from Kotlin. It is the module Pallas lowers
 * for
 * ```
 * def kernel(x_ref, w_ref, o_ref, acc_ref):
 *     @pl.when(pl.program_id(2) == 0)
 *     def _(): acc_ref[...] = jnp.zeros_like(acc_ref)
 *     acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
 *     @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
 *     def _(): o_ref[...] = acc_ref[...].astype(o_ref.dtype)
 * ```
 * with `grid = (M/tm, N/tn, K/tk)`, block specs `(i, k)`, `(k, j)`, `(i, j)` and dimension
 * semantics `(parallel, parallel, arbitrary)`. This is the grid / window form TorchTPU's Kiln
 * writes (`kernel.cc`): the grid indices are the first arguments, then the input blocks, the
 * output block and the scratch; `window_params` names each operand's index map, a private
 * function from the grid indices to block indices, and its block shape.
 *
 * [serialized] gives the `mosaic-serde` form a `tpu_custom_call` can carry as text.
 */
@ExperimentalTlalocApi
object MosaicTiledMatmul {

    enum class Dtype(val mlir: String) { F32("f32"), BF16("bf16") }

    fun emit(
        m: Int, n: Int, k: Int,
        tm: Int, tn: Int, tk: Int,
        dtype: Dtype = Dtype.BF16,
        serialized: Boolean = false,
        functionName: String = "tlaloc_matmul",
    ): String {
        require(m % tm == 0 && n % tn == 0 && k % tk == 0) {
            "MosaicTiledMatmul: blocks $tm x $tn x $tk must divide $m x $n x $k"
        }
        val sub = if (dtype == Dtype.F32) 8 else 16
        require(tm % sub == 0 && tn % 128 == 0 && tk % 128 == 0) {
            "MosaicTiledMatmul: blocks must be multiples of ($sub, 128) for ${dtype.mlir} (got $tm x $tn x $tk)"
        }
        val t = if (dtype == Dtype.F32) TileType.F32 else TileType.BF16
        return mosaicKernel(functionName, serialized) {
            grid(m / tm, n / tn, k / tk, semantics = listOf(GridSemantics.PARALLEL, GridSemantics.PARALLEL, GridSemantics.ARBITRARY))
            val x = input(listOf(tm, tk), t) { (i, _, kk) -> listOf(i, kk) }
            val w = input(listOf(tk, tn), t) { (_, j, kk) -> listOf(kk, j) }
            val o = output(listOf(tm, tn), t) { (i, j, _) -> listOf(i, j) }
            val acc = scratch(listOf(tm, tn), TileType.F32)
            body {
                whenEq(programId(2), 0) { acc.store(zeros(acc)) }
                acc.store(acc.load() + dot(x.load(), w.load()))
                whenEq(programId(2), k / tk - 1) { o.store(acc.load().cast(t)) }
            }
        }
    }
}
