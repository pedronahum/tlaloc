package io.tlaloc.stablehlo

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.capture2
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.core.F32
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirModule
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test

/**
 * Not a strict correctness test — a convenience printer that dumps a representative
 * lowering so reviewers can eyeball the emitter output. Always passes; output goes
 * to the test log.
 */
class PrintSamplesTest {
    @Test
    fun printSampleLayerLowering() {
        val fn = capture2(
            f = { w: Tracer<Rank2<Sym, Sym>>, x: Tracer<Rank2<Sym, Sym>> ->
                (w matmul x).relu().sum()
            },
            a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f)),
            b = Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 1f)),
            name = "layer",
        )
        val mlir = DxirModule(listOf(fn)).toStablehlo()
        println("--- sample lowering: layer ---")
        println(mlir)
        println("------------------------------")
    }

    @Test
    fun printSoftmaxLowering() {
        val fn = DxirBuilder.function("softmax") {
            val x = param("x", DxirType(F32, listOf(5)))
            val y = op(OpKind.SOFTMAX, listOf(x), DxirType(F32, listOf(5)))
            listOf(y)
        }
        println("--- sample lowering: softmax ---")
        println(DxirModule(listOf(fn)).toStablehlo())
        println("--------------------------------")
    }

    @Test
    fun printLayerNormLowering() {
        val fn = DxirBuilder.function("layernorm") {
            val x = param("x", DxirType(F32, listOf(8)))
            val y = op(
                OpKind.LAYERNORM, listOf(x), DxirType(F32, listOf(8)),
                attrs = mapOf("epsilon" to 1e-5f),
            )
            listOf(y)
        }
        println("--- sample lowering: layernorm ---")
        println(DxirModule(listOf(fn)).toStablehlo())
        println("----------------------------------")
    }

    @Test
    fun printSdpaRankThreeLowering() {
        val fn = DxirBuilder.function("sdpa") {
            val q = param("q", DxirType(F32, listOf(1, 2, 4)))
            val k = param("k", DxirType(F32, listOf(1, 3, 4)))
            val v = param("v", DxirType(F32, listOf(1, 3, 4)))
            val y = op(
                OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                listOf(q, k, v),
                DxirType(F32, listOf(1, 2, 4)),
            )
            listOf(y)
        }
        println("--- sample lowering: sdpa ---")
        println(DxirModule(listOf(fn)).toStablehlo())
        println("-----------------------------")
    }
}
