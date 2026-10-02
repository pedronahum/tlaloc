package io.tlaloc.plugin

import io.tlaloc.plugin.F64TestHarness.assertClose
import kotlin.math.cos
import kotlin.math.tanh
import kotlin.test.Test

/**
 * Five `grad {}` bodies that did not compile or computed a wrong shape, at F32 and F64:
 * tensor `sin`/`cos`; a comparison mask on a rank-1 operand; `tanh` after a
 * shape-changing `reshape`; a rank-1 named `contract` whose value the gradient needs;
 * a `contract` of a transposed named operand. Each gradient is compared with its analytic
 * form computed here in Double.
 *
 * Tolerance: 1e-5 of the largest entry at F32 (a handful of F32 operations, 6e-8 each),
 * 1e-13 at F64.
 */
class GradSurfaceFixesTest {

    private var precision = Precision.F32

    private fun at(p: Precision, body: () -> Unit) {
        precision = p
        body()
    }

    private val tol get() = if (precision == Precision.F64) 1e-13 else 1e-5

    private fun run(src: String) = F64TestHarness.run(if (precision == Precision.F64) F64Source.of(src) else src)

    private val v = doubleArrayOf(-0.8, -0.3, 0.1, 0.45, 1.2)

    private fun lit(a: DoubleArray) = a.joinToString(", ") { "${it}f" }

    @Test
    fun `tensor sin and cos`() = at(Precision.F32) { sinCos() }

    @Test
    fun `tensor sin and cos, F64`() = at(Precision.F64) { sinCos() }

    private fun sinCos() {
        val r = run(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> -> (x.sin() * x.cos()).sum().toFloat() }
                println("g " + g(Tensors.f32Vector<Sym>(floatArrayOf(${lit(v)}))).hostF32().joinToString(","))
            }
            """.trimIndent(),
        )
        // d/dx sin x cos x = cos 2x.
        assertClose(DoubleArray(v.size) { cos(2 * v[it]) }, r.values("g"), tol, "d(sin·cos)")
    }

    @Test
    fun `comparison mask on a rank-1 operand`() = at(Precision.F32) { rank1Mask() }

    @Test
    fun `comparison mask on a rank-1 operand, F64`() = at(Precision.F64) { rank1Mask() }

    private fun rank1Mask() {
        val r = run(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> -> where(x gt 0.2f, x * x, x.tanh()).sum().toFloat() }
                println("g " + g(Tensors.f32Vector<Sym>(floatArrayOf(${lit(v)}))).hostF32().joinToString(","))
            }
            """.trimIndent(),
        )
        val want = DoubleArray(v.size) { if (v[it] > 0.2) 2 * v[it] else 1 - tanh(v[it]) * tanh(v[it]) }
        assertClose(want, r.values("g"), tol, "d where")
    }

    @Test
    fun `tanh after a shape-changing reshape`() = at(Precision.F32) { reshapeTanh() }

    @Test
    fun `tanh after a shape-changing reshape, F64`() = at(Precision.F64) { reshapeTanh() }

    private fun reshapeTanh() {
        val m = DoubleArray(12) { (it % 5) * 0.3 - 0.6 }
        val r = run(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Sym>, F32> -> (x.reshape(4, 3) * x.reshape(4, 3).tanh()).sum().toFloat() }
                println("g " + g(Tensors.f32Matrix<Sym, Sym>(3, 4, floatArrayOf(${lit(m)}))).hostF32().joinToString(","))
            }
            """.trimIndent(),
        )
        // d/dr Σ r·tanh r = tanh r + r·(1 − tanh² r), elementwise (reshape moves nothing).
        val want = DoubleArray(12) { tanh(m[it]) + m[it] * (1 - tanh(m[it]) * tanh(m[it])) }
        assertClose(want, r.values("g"), tol, "d(r·tanh r)")
    }

    @Test
    fun `rank-1 named contract whose value the gradient needs`() = at(Precision.F32) { rank1Contract() }

    @Test
    fun `rank-1 named contract whose value the gradient needs, F64`() = at(Precision.F64) { rank1Contract() }

    private fun rank1Contract() {
        val x = doubleArrayOf(0.3, -0.7, 1.1)
        val r = run(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            object I : IndexName { override val name = "i" }
            fun main() {
                val g = grad { x: DTensor<Rank1<Named<I, Sym>>, F32> -> (x contract x.tanh()).toFloat() * (x contract x).toFloat() }
                println("g " + g(Tensors.f32Vector<Named<I, Sym>>(floatArrayOf(${lit(x)}))).hostF32().joinToString(","))
            }
            """.trimIndent(),
        )
        // L = (x·tanh x)(x·x): dL/dx = (tanh x + x(1 − tanh² x))·(x·x) + (x·tanh x)·2x.
        val xx = x.sumOf { it * it }
        val xt = x.sumOf { it * tanh(it) }
        val want = DoubleArray(3) { (tanh(x[it]) + x[it] * (1 - tanh(x[it]) * tanh(x[it]))) * xx + xt * 2 * x[it] }
        assertClose(want, r.values("g"), tol, "d(product of dots)")
    }

    @Test
    fun `contract of a transposed named operand`() = at(Precision.F32) { transposedContract() }

    @Test
    fun `contract of a transposed named operand, F64`() = at(Precision.F64) { transposedContract() }

    private fun transposedContract() {
        val x = doubleArrayOf(0.3, -0.7, 1.1, 0.5, 0.2, -0.4) // I×J = 2×3
        val y = doubleArrayOf(0.9, -0.1, 0.4, 0.6) // I×K = 2×2
        val r = run(
            """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            object I : IndexName { override val name = "i" }
            object J : IndexName { override val name = "j" }
            object K : IndexName { override val name = "k" }
            fun main() {
                val g = grad2 { a: DTensor<Rank2<Named<I, Sym>, Named<J, Sym>>, F32>, b: DTensor<Rank2<Named<I, Sym>, Named<K, Sym>>, F32> ->
                    (a.transpose() contract b).tanh().sum().toFloat()
                }
                val (dx, dy) = g(
                    Tensors.f32Matrix<Named<I, Sym>, Named<J, Sym>>(2, 3, floatArrayOf(${lit(x)})),
                    Tensors.f32Matrix<Named<I, Sym>, Named<K, Sym>>(2, 2, floatArrayOf(${lit(y)})),
                )
                println("dx " + dx.hostF32().joinToString(","))
                println("dy " + dy.hostF32().joinToString(","))
            }
            """.trimIndent(),
        )
        // C = Aᵀ·B over I (J×K), S = 1 − tanh² C: dA = B·Sᵀ, dB = A·S.
        val s = DoubleArray(6) { idx ->
            val (j, k) = idx / 2 to idx % 2
            val c = (0 until 2).sumOf { i -> x[i * 3 + j] * y[i * 2 + k] }
            1 - tanh(c) * tanh(c)
        }
        val dx = DoubleArray(6) { idx -> (0 until 2).sumOf { k -> s[(idx % 3) * 2 + k] * y[(idx / 3) * 2 + k] } }
        val dy = DoubleArray(4) { idx -> (0 until 3).sumOf { j -> s[j * 2 + idx % 2] * x[(idx / 2) * 3 + j] } }
        assertClose(dx, r.values("dx"), tol, "dA")
        assertClose(dy, r.values("dy"), tol, "dB")
    }

    @Test
    fun `comparisons at ranks 2, 3 and 4`() = at(Precision.F32) { comparisonRanks() }

    @Test
    fun `comparisons at ranks 2, 3 and 4, F64`() = at(Precision.F64) { comparisonRanks() }

    /** clip, maximum and where(gt) lower to a Bool COMPARE of the operand's rank. */
    private fun comparisonRanks() {
        val x = DoubleArray(24) { ((it * 7) % 24 - 11.5) / 9.0 }
        val shapes = mapOf(
            2 to ("Rank2<Sym, Sym>" to "Tensors.f32Matrix<Sym, Sym>(4, 6, "),
            3 to ("Rank3<Sym, Sym, Sym>" to "Tensors.f32Tensor3<Sym, Sym, Sym>(2, 3, 4, "),
            4 to ("Rank4<Sym, Sym, Sym, Sym>" to "Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 3, 4, "),
        )
        val body = shapes.entries.joinToString("\n") { (rank, st) ->
            val (type, make) = st
            """
                val x$rank = $make floatArrayOf(${lit(x)}))
                println("clip$rank " + grad { x: DTensor<$type, F32> -> val y = clip(x, -0.5f, 0.9f); (y * y).sum().toFloat() }(x$rank).hostF32().joinToString(","))
                println("max$rank " + grad { x: DTensor<$type, F32> -> maximum(x, x.tanh()).sum().toFloat() }(x$rank).hostF32().joinToString(","))
                println("where$rank " + grad { x: DTensor<$type, F32> -> where(x gt 0.2f, x * x, x.tanh()).sum().toFloat() }(x$rank).hostF32().joinToString(","))
            """
        }
        val r = run(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
            $body
            }
            """.trimIndent(),
        )
        val sech2 = { v: Double -> 1 - tanh(v) * tanh(v) }
        val clip = DoubleArray(24) { val v = x[it]; if (v >= -0.5 && v <= 0.9) 2 * v else 0.0 }
        val max = DoubleArray(24) { if (x[it] >= tanh(x[it])) 1.0 else sech2(x[it]) }
        val where = DoubleArray(24) { if (x[it] > 0.2) 2 * x[it] else sech2(x[it]) }
        for (rank in shapes.keys) {
            assertClose(clip, r.values("clip$rank"), tol, "d clip, rank $rank")
            assertClose(max, r.values("max$rank"), tol, "d maximum, rank $rank")
            assertClose(where, r.values("where$rank"), tol, "d where, rank $rank")
        }
    }
}
