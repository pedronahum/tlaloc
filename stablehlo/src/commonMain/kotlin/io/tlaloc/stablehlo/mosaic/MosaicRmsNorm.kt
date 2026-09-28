package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * Mosaic MLIR text for an RMSNorm TPU kernel, written from Kotlin.
 *
 * `out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w[0, :]` over a
 * `[rows, hidden]` input held whole in VMEM, with the arithmetic in f32 (a
 * bf16 kernel widens on load and narrows on store). This is the module Pallas
 * lowers for the same kernel body with no grid; `harness/python/export_tpu_kernels.py`
 * checks that the two are identical after `cse` before it serializes this
 * text to the bytecode a `tpu_custom_call` carries.
 *
 * The text uses the `tpu`, `vector`, `arith`, `math`, `memref` and `func`
 * dialects at the level Pallas hands to Mosaic: before layout inference, so
 * libtpu's Mosaic compiler still chooses vector layouts
 * (`needs_layout_passes = true` in the call's config).
 */
@ExperimentalTlalocApi
object MosaicRmsNorm {

    /** Element types this emitter writes. */
    enum class Dtype(val mlir: String) { F32("f32"), BF16("bf16") }

    /**
     * The Mosaic module for an RMSNorm over `[rows, hidden]` with a
     * `[1, hidden]` weight. [rows] must be a multiple of 8 and [hidden] a
     * multiple of 128 (one f32 vreg tile is 8x128); [eps] is rounded to f32.
     */
    fun emit(
        rows: Int,
        hidden: Int,
        dtype: Dtype = Dtype.F32,
        eps: Float = 1e-6f,
        functionName: String = "tlaloc_rmsnorm",
    ): String {
        require(rows > 0 && rows % 8 == 0) { "MosaicRmsNorm: rows must be a positive multiple of 8, got $rows" }
        require(hidden > 0 && hidden % 128 == 0) {
            "MosaicRmsNorm: hidden must be a positive multiple of 128, got $hidden"
        }
        require(functionName.matches(Regex("[A-Za-z_][A-Za-z0-9_$.]*"))) {
            "MosaicRmsNorm: '$functionName' is not a bare MLIR symbol name"
        }
        val t = dtype.mlir
        val vmem = "#tpu.memory_space<vmem>"
        val xMem = "memref<${rows}x${hidden}x$t, $vmem>"
        val wMem = "memref<1x${hidden}x$t, $vmem>"
        val xVec = "vector<${rows}x${hidden}x$t>"
        val wVec = "vector<1x${hidden}x$t>"
        val full = "vector<${rows}x${hidden}xf32>"
        val row = "vector<${rows}xf32>"
        val col = "vector<${rows}x1xf32>"
        val widen = dtype != Dtype.F32
        return buildString {
            appendLine("module {")
            appendLine(
                "  func.func @$functionName(%x_ref: $xMem, %w_ref: $wMem, %o_ref: $xMem) attributes " +
                    "{dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, " +
                    "tpu.core_type = #tpu.core_type<tc>} {",
            )
            appendLine("    %c0 = arith.constant 0 : index")
            if (widen) {
                appendLine("    %x_in = vector.load %x_ref[%c0, %c0] : $xMem, $xVec")
                appendLine("    %x = arith.extf %x_in : $xVec to $full")
            } else {
                appendLine("    %x = vector.load %x_ref[%c0, %c0] : $xMem, $xVec")
            }
            appendLine("    %sq = arith.mulf %x, %x : $full")
            appendLine("    %zero = arith.constant dense<0.000000e+00> : $row")
            appendLine("    %sum = vector.multi_reduction <add>, %sq, %zero [1] : $full to $row")
            appendLine("    %sum_col = vector.shape_cast %sum : $row to $col")
            appendLine("    %n = arith.constant ${f32Hex(hidden.toFloat())} : f32")
            appendLine("    %n_col = vector.broadcast %n : f32 to $col")
            appendLine("    %mean = arith.divf %sum_col, %n_col : $col")
            appendLine("    %eps = arith.constant ${f32Hex(eps)} : f32")
            appendLine("    %eps_col = vector.broadcast %eps : f32 to $col")
            appendLine("    %var = arith.addf %mean, %eps_col : $col")
            appendLine("    %inv = math.rsqrt %var : $col")
            appendLine("    %inv_full = vector.broadcast %inv : $col to $full")
            appendLine("    %normed = arith.mulf %x, %inv_full : $full")
            if (widen) {
                appendLine("    %w_in = vector.load %w_ref[%c0, %c0] : $wMem, $wVec")
                appendLine("    %w = arith.extf %w_in : $wVec to vector<1x${hidden}xf32>")
            } else {
                appendLine("    %w = vector.load %w_ref[%c0, %c0] : $wMem, $wVec")
            }
            appendLine("    %w_full = vector.broadcast %w : vector<1x${hidden}xf32> to $full")
            appendLine("    %y = arith.mulf %normed, %w_full : $full")
            val stored = if (widen) {
                appendLine("    %y_out = arith.truncf %y : $full to $xVec")
                "%y_out"
            } else {
                "%y"
            }
            appendLine("    tpu.vector_store %o_ref[%c0, %c0], $stored {strides = array<i32>} : $xMem, $xVec,")
            appendLine("    return")
            appendLine("  }")
            appendLine("}")
        }
    }

    /** An f32 constant as MLIR's exact hex spelling, e.g. `0x358637BD`. */
    private fun f32Hex(v: Float): String =
        "0x" + v.toRawBits().toUInt().toString(16).uppercase().padStart(8, '0')
}
