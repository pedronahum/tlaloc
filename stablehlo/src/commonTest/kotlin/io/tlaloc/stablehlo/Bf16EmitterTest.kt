package io.tlaloc.stablehlo

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.456 (Phase G1b) — structural pins for bf16 StableHLO emission: the
 * `bf16` element type spelling, `stablehlo.convert` for the precision casts,
 * and bf16-typed elementwise/matmul/reduce ops. These are TEXT pins — the
 * emitter's contract is the MLIR spelling XLA ingests. GPU EXECUTION of
 * bf16 modules is G1c and deliberately not claimed here.
 */
class Bf16EmitterTest {

    @Test
    fun bf16TypeSpellsMlirBfloat16() {
        assertEquals("tensor<bf16>", DxirType(BF16, emptyList()).toMlir())
        assertEquals("tensor<4xbf16>", DxirType(BF16, listOf(4)).toMlir())
        assertEquals("tensor<2x3xbf16>", DxirType(BF16, listOf(2, 3)).toMlir())
    }

    @Test
    fun castsEmitStablehloConvertBothDirections() {
        val fn = DxirBuilder.function("casts") {
            val x = param("x", DxirType(F32, listOf(4)))
            val b = op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(4)))
            listOf(op(OpKind.CAST, listOf(b), DxirType(F32, listOf(4))))
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.convert %0 : (tensor<4xf32>) -> tensor<4xbf16>"),
            "narrowing cast must emit convert f32→bf16; got: $mlir",
        )
        assertTrue(
            mlir.contains(": (tensor<4xbf16>) -> tensor<4xf32>"),
            "widening cast must emit convert bf16→f32; got: $mlir",
        )
    }

    @Test
    fun bf16ElementwiseAndMatmulEmitWithBf16TensorTypes() {
        val bt = DxirType(BF16, listOf(2, 2))
        val fn = DxirBuilder.function("bf16_ops") {
            val a = param("a", bt)
            val b = param("b", bt)
            val s = op(OpKind.ADD, listOf(a, b), bt)
            listOf(op(OpKind.MATMUL, listOf(s, b), bt))
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.add %0, %1 : tensor<2x2xbf16>"),
            "bf16 add must carry the bf16 tensor type; got: $mlir",
        )
        assertTrue(
            mlir.contains("stablehlo.dot_general") && mlir.contains("-> tensor<2x2xbf16>"),
            "bf16 matmul must emit dot_general with a bf16 result type; got: $mlir",
        )
    }

    @Test
    fun bf16ReduceSumAndMaxEmitBf16InitConstants() {
        val bt = DxirType(BF16, listOf(2, 3))
        val st = DxirType(BF16, emptyList())
        val fn = DxirBuilder.function("bf16_reduce") {
            val x = param("x", bt)
            val s = op(OpKind.SUM, listOf(x), st)
            val m = op(OpKind.MAX, listOf(x), st, attrs = mapOf("reduction_dims" to listOf(0, 1)))
            listOf(s, m)
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("stablehlo.reduce") && mlir.contains("tensor<bf16>"),
            "bf16 reduce must use bf16 scalar init/result types; got: $mlir",
        )
        // The MAX identity is bf16 -Inf as a width-matched MLIR hex literal:
        // the top 16 bits of f32's 0xFF800000.
        assertTrue(
            mlir.contains("stablehlo.constant dense<0xFF80> : tensor<bf16>"),
            "bf16 MAX reduce must splat the 16-bit -Inf pattern 0xFF80; got: $mlir",
        )
    }

    @Test
    fun bf16StepEmitsFloatCompare() {
        val bt = DxirType(BF16, listOf(4))
        val fn = DxirBuilder.function("bf16_step") {
            val x = param("x", bt)
            listOf(op(OpKind.STEP, listOf(x), bt))
        }
        val mlir = fn.toStablehlo()
        assertTrue(
            mlir.contains("FLOAT") && mlir.contains("tensor<4xbf16>"),
            "bf16 STEP must compare FLOAT over bf16 tensors; got: $mlir",
        )
    }

    /**
     * Named refusal: RNG draws are a binary32 bit-stream contract (the
     * top-23-bit mantissa trick, bit-pinned vs JAX) — a bf16-typed draw must
     * refuse by name, not silently change the certified stream. The
     * sanctioned spelling is draw-at-f32 then CAST.
     */
    @Test
    fun bf16RngDrawRefusesByName() {
        val fn = DxirBuilder.function("bf16_rng") {
            listOf(
                op(
                    OpKind.RNG_UNIFORM, emptyList(), DxirType(BF16, listOf(4)),
                    attrs = mapOf("key0" to 1, "key1" to 2),
                ),
            )
        }
        val ex = assertFailsWith<IllegalArgumentException> { fn.toStablehlo() }
        assertTrue(
            ex.message!!.contains("f32-only") && ex.message!!.contains("CAST"),
            "refusal must name the f32 contract and the CAST spelling; got: ${ex.message}",
        )
    }
}
