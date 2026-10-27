package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Fp8MatmulTest {

    private val m = 3
    private val k = 4
    private val n = 2

    private fun fn(rowwise: Boolean) = DxirBuilder.function("fp8") {
        val a = param("a", DxirType(F8E4M3FN, listOf(m, k)))
        val b = param("b", DxirType(F8E4M3FN, listOf(k, n)))
        val sa = param("sa", DxirType(F32, if (rowwise) listOf(m) else emptyList()))
        val sb = param("sb", DxirType(F32, if (rowwise) listOf(n) else emptyList()))
        listOf(op(OpKind.FP8_MATMUL, listOf(a, b, sa, sb), DxirType(F32, listOf(m, n))))
    }

    @Test
    fun theProductOfTheCodesIsScaledPerRowAndColumn() {
        val r = Random(2)
        // Decoded e4m3fn values, as the interpreter holds them.
        val a = FloatArray(m * k) { f8e4m3fnToFloat(r.nextInt(0, 0x7F).toByte()) * if (r.nextBoolean()) 1 else -1 }
        val b = FloatArray(k * n) { f8e4m3fnToFloat(r.nextInt(0, 0x7F).toByte()) }
        val sa = floatArrayOf(0.5f, 2f, -1f)
        val sb = floatArrayOf(3f, 0.25f)
        val y = DxirInterpreter.evalFunction(fn(true), listOf(a, b, sa, sb))[0]
        for (i in 0 until m) for (j in 0 until n) {
            val dot = (0 until k).sumOf { a[i * k + it].toDouble() * b[it * n + j] }
            assertEquals((dot * sa[i] * sb[j]).toFloat(), y[i * n + j], "[$i, $j]")
        }
        val t = DxirInterpreter.evalFunction(fn(false), listOf(a, b, floatArrayOf(2f), floatArrayOf(0.5f)))[0]
        for (i in 0 until m) for (j in 0 until n) {
            assertEquals((0 until k).sumOf { a[i * k + it].toDouble() * b[it * n + j] }.toFloat(), t[i * n + j])
        }
    }

    @Test
    fun itIsInferenceOnly() {
        val loss = DxirBuilder.function("loss") {
            val a = param("a", DxirType(F8E4M3FN, listOf(m, k)))
            val b = param("b", DxirType(F8E4M3FN, listOf(k, n)))
            val sa = param("sa", DxirType(F32, listOf(m)))
            val sb = param("sb", DxirType(F32, listOf(n)))
            val y = op(OpKind.FP8_MATMUL, listOf(a, b, sa, sb), DxirType(F32, listOf(m, n)))
            listOf(op(OpKind.SUM, listOf(y), DxirType(F32, emptyList())))
        }
        val e = assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(loss) }
        assertTrue("FP8_MATMUL is inference-only" in e.message!!, e.message)
    }
}
