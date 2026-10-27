package io.tlaloc.stablehlo

import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertTrue

class F64ConstEmitTest {

    @Test
    fun f64ConstantsArePrintedWithEveryDigit() {
        val fn = DxirBuilder.function("consts") {
            val x = param("x", DxirType(F64, listOf(2, 2)))
            val c = const(doubleArrayOf(0.1, 1e-9, 1.0 + 1e-12, -3.0), DxirType(F64, listOf(2, 2)))
            val s = const(0.30000000000000004, DxirType(F64, emptyList()))
            val splat = op(OpKind.BROADCAST, listOf(s, x), x.type, attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            listOf(op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(x, c), x.type), splat), x.type))
        }
        val mlir = fn.toStablehlo("")
        assertTrue(
            "stablehlo.constant dense<[[0.1, 1.0E-9], [1.000000000001, -3.0]]> : tensor<2x2xf64>" in mlir,
            mlir,
        )
        assertTrue("stablehlo.constant dense<0.30000000000000004> : tensor<f64>" in mlir, mlir)
    }
    @Test
    fun nonFiniteArrayElementsAreBitPatterns() {
        // An additive causal mask: MLIR has no literal for −∞, NaN or +∞.
        val inf = Float.NEGATIVE_INFINITY
        for ((dtype, bits) in listOf(io.tlaloc.core.F32 to listOf("0xFF800000", "0x7FC00000", "0x7F800000"), io.tlaloc.core.BF16 to listOf("0xFF80", "0x7FC0", "0x7F80"))) {
            val t = DxirType(dtype, listOf(2, 2))
            val mlir = DxirBuilder.function("mask") {
                val x = param("x", t)
                listOf(op(OpKind.ADD, listOf(x, const(floatArrayOf(0f, inf, Float.NaN, Float.POSITIVE_INFINITY), t)), t))
            }.toStablehlo("")
            assertTrue("dense<[[0.0, ${bits[0]}], [${bits[1]}, ${bits[2]}]]>" in mlir, mlir)
        }
        val d = DxirType(F64, listOf(2))
        val m64 = DxirBuilder.function("mask64") {
            val x = param("x", d)
            listOf(op(OpKind.ADD, listOf(x, const(doubleArrayOf(1.0, Double.NEGATIVE_INFINITY), d)), d))
        }.toStablehlo("")
        assertTrue("dense<[1.0, 0xFFF0000000000000]>" in m64, m64)
    }

    private fun f64(vararg dims: Int) = DxirType(F64, dims.toList())
    private fun f32(vararg dims: Int) = DxirType(io.tlaloc.core.F32, dims.toList())

    @Test
    fun avgPoolGradientScaleIsADoubleInAnF64Graph() {
        fun graph(t: (IntArray) -> DxirType) = DxirBuilder.function("avg_grad") {
            val up = param("up", t(intArrayOf(1, 1, 2, 2)))
            val x = param("x", t(intArrayOf(1, 1, 6, 6)))
            listOf(
                op(
                    OpKind.AVGPOOL2D_GRAD, listOf(up, x), t(intArrayOf(1, 1, 6, 6)),
                    attrs = mapOf("window" to listOf(3, 3), "window_strides" to listOf(3, 3), "padding" to listOf(listOf(0, 0), listOf(0, 0))),
                ),
            )
        }
        val m64 = graph { f64(*it) }.toStablehlo("")
        assertTrue("dense<0.1111111111111111> : tensor<1x1x3x3xf64>" in m64, m64)
        // The f32 spelling is unchanged.
        val m32 = graph { f32(*it) }.toStablehlo("")
        assertTrue("dense<0.11111111> : tensor<1x1x3x3xf32>" in m32, m32)
    }

    @Test
    fun normEpsilonAndAttentionScaleAreDoublesInAnF64Graph() {
        val ln = DxirBuilder.function("ln") {
            val x = param("x", f64(2, 8))
            listOf(op(OpKind.LAYERNORM, listOf(x), f64(2, 8), attrs = mapOf("epsilon" to 1e-5)))
        }.toStablehlo("")
        assertTrue("dense<1.0E-5> : tensor<2xf64>" in ln, ln)
        val rms = DxirBuilder.function("rms") {
            val x = param("x", f64(2, 8))
            listOf(op(OpKind.RMSNORM, listOf(x), f64(2, 8), attrs = mapOf("epsilon" to 1e-6)))
        }.toStablehlo("")
        assertTrue("dense<1.0E-6> : tensor<2xf64>" in rms, rms)
        val sdpa = DxirBuilder.function("sdpa") {
            val q = param("q", f64(1, 2, 3))
            val k = param("k", f64(1, 4, 3))
            val v = param("v", f64(1, 4, 3))
            listOf(op(OpKind.SCALED_DOT_PRODUCT_ATTENTION, listOf(q, k, v), f64(1, 2, 3)))
        }.toStablehlo("")
        assertTrue("dense<${1.0 / kotlin.math.sqrt(3.0)}> : tensor<f64>" in sdpa, sdpa)
    }
}
