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
}
