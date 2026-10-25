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
        val t = dtype.mlir
        val vmem = "#tpu.memory_space<vmem>"
        val xRef = "memref<${tm}x${tk}x$t, $vmem>"
        val wRef = "memref<${tk}x${tn}x$t, $vmem>"
        val oRef = "memref<${tm}x${tn}x$t, $vmem>"
        val accRef = "memref<${tm}x${tn}xf32, $vmem>"
        val accVec = "vector<${tm}x${tn}xf32>"
        val w = MosaicWriter(serialized)
        val sem = { s: String -> "#tpu.dimension_semantics<$s>" }
        w.function(
            functionName,
            args = listOf("i32", "i32", "i32", xRef, wRef, oRef, accRef),
            results = emptyList(),
            attrs = listOf(
                "dimension_semantics" to "[${sem("parallel")}, ${sem("parallel")}, ${sem("arbitrary")}]",
                "iteration_bounds" to "array<i64: ${m / tm}, ${n / tn}, ${k / tk}>",
                "scalar_prefetch" to "0 : i64",
                "scratch_operands" to "1 : i64",
                "tpu.core_type" to "#tpu.core_type<tc>",
                "window_params" to "[" +
                    "{transform_indices = @transform_0, window_bounds = array<i64: $tm, $tk>}, " +
                    "{transform_indices = @transform_1, window_bounds = array<i64: $tk, $tn>}, " +
                    "{transform_indices = @transform_2, window_bounds = array<i64: $tm, $tn>}]",
            ),
        ) { a ->
            val kk = a[2]; val x = a[3]; val wt = a[4]; val o = a[5]; val acc = a[6]
            fun whenK(value: Int, body: MosaicWriter.() -> Unit) {
                val c = i32Const(value)
                val eq = cmpi(MosaicWriter.EQ, kk, c)
                val ext = extui(eq, "i32")
                val ne = cmpi(MosaicWriter.NE, ext, i32Const(0))
                ifThen(ne, body)
            }
            whenK(0) {
                val zero = f32Const("0.000000e+00 : f32")
                val z = broadcast(zero, accVec)
                val c0 = indexConst(0)
                store(z, acc, listOf(c0, c0))
            }
            val c0 = indexConst(0)
            val prev = load(acc, listOf(c0, c0), accVec)
            val xv = load(x, listOf(c0, c0), "vector<${tm}x${tk}x$t>")
            val wv = load(wt, listOf(c0, c0), "vector<${tk}x${tn}x$t>")
            val init = f32Const("dense<0.000000e+00> : $accVec", accVec)
            val prod = matmul(xv, wv, init, "#tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>")
            store(float("arith.addf", prev, prod), acc, listOf(c0, c0))
            whenK(k / tk - 1) {
                val c = indexConst(0)
                val total = load(acc, listOf(c, c), accVec)
                val out = if (dtype == Dtype.F32) total else cast("arith.truncf", total, "vector<${tm}x${tn}x$t>")
                store(out, o, listOf(c, c))
            }
            ret()
        }
        // The index maps: grid (i, j, k) to block indices.
        for ((idx, pick) in listOf(0 to (0 to 2), 1 to (2 to 1), 2 to (0 to 1))) {
            w.function("transform_$idx", listOf("i32", "i32", "i32"), listOf("i32", "i32"), emptyList()) { a ->
                i32Const(0)
                ret(listOf(a[pick.first], a[pick.second]))
            }
        }
        return w.module(functionName)
    }
}
