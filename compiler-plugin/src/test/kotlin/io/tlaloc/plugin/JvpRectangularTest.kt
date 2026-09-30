package io.tlaloc.plugin

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `jvp2` of a nonlinearity after a rectangular matmul, all axes `Sym`: the forward rules'
 * constants (`1` in `1 − tanh²`, `1 + tan²`, …) are sized from a value of the right shape
 * at run time. They used to be shaped constants whose extents synthesis matched against
 * the parameters, which picked `x`'s `[2, 3]` for a `[2, 4]` product and failed at run time.
 * The directional derivative must equal ⟨∇f, (dx, dw)⟩ from `grad2`, at F32 and F64.
 */
class JvpRectangularTest {

    private fun check(expr: String) {
        for (p in listOf(Precision.F32, Precision.F64)) {
            val src = """
                import io.tlaloc.autograd.*
                import io.tlaloc.core.*
                import io.tlaloc.core.ops.*
                fun main() {
                    val j = jvp2 { x: DTensor<Rank2<Sym, Sym>, F32>, w: DTensor<Rank2<Sym, Sym>, F32> -> $expr.sum().toFloat() }
                    val g = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, w: DTensor<Rank2<Sym, Sym>, F32> -> $expr.sum().toFloat() }
                    val x = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6) { 0.1f * it + 0.05f })
                    val w = Tensors.f32Matrix<Sym, Sym>(3, 4, FloatArray(12) { 0.05f * it - 0.2f })
                    val dx = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6) { 0.3f - 0.1f * it })
                    val dw = Tensors.f32Matrix<Sym, Sym>(3, 4, FloatArray(12) { 0.02f * it })
                    println("jvp " + j(x, w, dx, dw))
                    val (gx, gw) = g(x, w)
                    var s = 0f
                    for (i in 0 until 6) s += gx.hostF32()[i] * dx.hostF32()[i]
                    for (i in 0 until 12) s += gw.hostF32()[i] * dw.hostF32()[i]
                    println("dot " + s)
                }
            """.trimIndent()
            val r = F64TestHarness.compileAndRun(if (p == Precision.F64) F64Source.of(src) else src)
            assertEquals(0, r.exitCode, r.describe())
            val a = r.value("jvp")
            val b = r.value("dot")
            val tol = if (p == Precision.F64) 1e-12 else 1e-5
            assertTrue(kotlin.math.abs(a - b) <= tol * maxOf(1.0, kotlin.math.abs(b)), "$p $expr: jvp $a, <grad, d> $b")
        }
    }

    @Test fun tanh() = check("(x matmul w).tanh()")
    @Test fun sigmoid() = check("(x matmul w).sigmoid()")
    @Test fun tan() = check("(x matmul w).tan()")
    @Test fun atan() = check("(x matmul w).atan()")
}
