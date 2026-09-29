package io.tlaloc.plugin

import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.plugin.F64TestHarness.assertClose
import io.tlaloc.plugin.F64TestHarness.fd4
import io.tlaloc.plugin.F64TestHarness.lit
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.tanh
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * `DTensor<…, F64>` under `grad {}`, `grad2 {}`, `valueAndGrad {}`, `jvp {}` and `vjp {}`:
 * elementwise arithmetic, broadcasting, reductions, shape ops, matmul, softmax, masks.
 * Each gradient is compared with fourth-order central differences (`h = 1e-3`) of the
 * same function written in plain Kotlin Double.
 *
 * Tolerance 1e-9 of the largest entry: the differences are good to about 1e-12 here (see
 * [F64TestHarness.fd4]) and the compiled gradient, a few dozen double operations, to about
 * 1e-14. An F32 value anywhere in the chain is off by at least 1e-8 relative, which fails.
 */
class F64TensorGradientTest {

    private val tol = 1e-9

    private val m = doubleArrayOf(
        0.31, -0.72, 1.13, 0.05,
        -0.44, 0.87, -1.29, 0.66,
        0.92, 0.18, -0.35, -0.81,
    )
    private val v = doubleArrayOf(0.25, -0.6, 0.9, 0.12)
    private val b = doubleArrayOf(0.7, -0.2, 0.15, 1.1, -0.9, 0.4, 0.33, -0.55)
    private val x = doubleArrayOf(-0.8, -0.3, 0.1, 0.45, 1.2)

    private fun elementwiseLoss(mm: DoubleArray, vv: DoubleArray): Double {
        val bb = DoubleArray(12) { k ->
            val j = k % 4
            val a = mm[k] * mm[k] * 0.3 + exp(mm[k]) - tanh(mm[k] + vv[j]) * vv[j]
            a / (mm[k] * mm[k] + 1.5)
        }
        var loss = 0.0
        for (i in 0 until 3) {
            var s = 0.0
            var mx = Double.NEGATIVE_INFINITY
            for (j in 0 until 4) { s += bb[i * 4 + j]; mx = max(mx, bb[i * 4 + j]) }
            loss += s * mx
        }
        return loss + 0.7 * bb.sum() / 12
    }

    @Test
    fun `elementwise ops, broadcasting and reductions`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad2 { m: DTensor<Rank2<Sym, Sym>, F64>, v: DTensor<Rank1<Sym>, F64> ->
                    val a = m * m * 0.3 + m.exp() - (m + v).tanh() * v
                    val b = a / (m * m + 1.5)
                    (b.sum(1) * b.max(1)).sum().toDouble() + 0.7 * b.mean().toDouble()
                }
                val (dm, dv) = g(
                    Tensors.f64Matrix<Sym, Sym>(3, 4, doubleArrayOf(${lit(m)})),
                    Tensors.f64Vector<Sym>(doubleArrayOf(${lit(v)})),
                )
                println("dm " + dm.hostF64().joinToString(","))
                println("dv " + dv.hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(fd4(m) { elementwiseLoss(it, v) }, r.values("dm"), tol, "d/dm")
        assertClose(fd4(v) { elementwiseLoss(m, it) }, r.values("dv"), tol, "d/dv")
        // The compiled gradient (host ops) against the F64 reference interpreter on the
        // gradient graph of the same body; PjrtF64GradTest runs that graph on the device.
        // Both compute in Double and differ only in summation order: 1e-13.
        val interp = DxirInterpreterF64.evalFunction(DxirReverseTransform.apply(elementwiseGraph()), listOf(m, v))
        assertClose(interp[0], r.values("dm"), 1e-13, "host vs interpreter d/dm")
        assertClose(interp[1], r.values("dv"), 1e-13, "host vs interpreter d/dv")
    }

    /** The DXIR the plugin lowers the body of the test above to, with the test's shapes. */
    private fun elementwiseGraph(): DxirFunction = DxirBuilder.function("elementwise") {
        fun t(vararg dims: Int) = DxirType(F64, dims.toList())
        val mm = param("m", t(3, 4))
        val vv = param("v", t(4))
        val m2 = op(OpKind.MUL, listOf(mm, mm), t(3, 4))
        fun splat(x: Double) = op(OpKind.BROADCAST, listOf(const(x, t()), m2), t(3, 4), attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
        val a = op(
            OpKind.SUB,
            listOf(
                op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(m2, splat(0.3)), t(3, 4)), op(OpKind.EXP, listOf(mm), t(3, 4))), t(3, 4)),
                op(OpKind.MUL, listOf(op(OpKind.TANH, listOf(op(OpKind.ADD, listOf(mm, vv), t(3, 4))), t(3, 4)), vv), t(3, 4)),
            ),
            t(3, 4),
        )
        val b = op(OpKind.DIV, listOf(a, op(OpKind.ADD, listOf(m2, splat(1.5)), t(3, 4))), t(3, 4))
        val s = op(OpKind.SUM, listOf(b), t(3), attrs = mapOf("reduction_dims" to listOf(1)))
        val mx = op(OpKind.MAX, listOf(b), t(3), attrs = mapOf("reduction_dims" to listOf(1)))
        val left = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(s, mx), t(3))), t())
        listOf(op(OpKind.ADD, listOf(left, op(OpKind.MUL, listOf(const(0.7, t()), op(OpKind.MEAN, listOf(b), t())), t())), t()))
    }

    private fun matmulLoss(aa: DoubleArray, bb: DoubleArray): Double {
        val c = DoubleArray(6) { k -> (0 until 4).sumOf { aa[(k / 2) * 4 + it] * bb[it * 2 + k % 2] } }
        var loss = 0.0
        for (i in 0 until 3) {
            val mx = max(c[2 * i], c[2 * i + 1])
            val z = exp(c[2 * i] - mx) + exp(c[2 * i + 1] - mx)
            for (j in 0 until 2) {
                val logP = c[2 * i + j] - mx - ln(z)
                loss += exp(logP) * logP
            }
        }
        var tt = 0.0
        for (p in 0 until 4) for (q in 0 until 4) {
            val t = (0 until 3).sumOf { aa[it * 4 + p] * aa[it * 4 + q] }
            tt += t * t
        }
        return loss + tt / 16 + (0 until 4).sumOf { c[it] }
    }

    @Test
    fun `matmul, transpose, slice and softmax`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad2 { a: DTensor<Rank2<Sym, Sym>, F64>, b: DTensor<Rank2<Sym, Sym>, F64> ->
                    val c = a matmul b
                    val s = c.softmax(1) * c.logSoftmax(1)
                    val t = a.transpose() matmul a
                    s.sum().toDouble() + (t * t).mean().toDouble() + c.slice(0, 2, 0).sum().toDouble()
                }
                val (da, db) = g(
                    Tensors.f64Matrix<Sym, Sym>(3, 4, doubleArrayOf(${lit(m)})),
                    Tensors.f64Matrix<Sym, Sym>(4, 2, doubleArrayOf(${lit(b)})),
                )
                println("da " + da.hostF64().joinToString(","))
                println("db " + db.hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(fd4(m) { matmulLoss(it, b) }, r.values("da"), tol, "d/da")
        assertClose(fd4(b) { matmulLoss(m, it) }, r.values("db"), tol, "d/db")
    }

    private fun maskLoss(xx: DoubleArray): Double = xx.sumOf { e ->
        (if (e > 0.2) e * e else tanh(e)) + 0.5 * max(e, 0.0) + min(max(e, -0.5), 0.9) + e * e * e
    }

    @Test
    fun `where, comparisons, relu, clip and pow`() {
        // x keeps every entry at least 0.1 from each kink (0.2, 0, -0.5, 0.9), far beyond the
        // difference stencil's 2h = 2e-3. It is a 1×5 matrix because comparison masks on a
        // rank-1 operand do not synthesize for F32 either.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Sym>, F64> ->
                    (where(x gt 0.2, x * x, x.tanh()) + x.relu() * 0.5 + clip(x, -0.5, 0.9) + x.pow(3.0)).sum().toDouble()
                }
                println("dx " + g(Tensors.f64Matrix<Sym, Sym>(1, 5, doubleArrayOf(${lit(x)}))).hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(fd4(x) { maskLoss(it) }, r.values("dx"), tol, "d/dx")
    }

    @Test
    fun `valueAndGrad, jvp and vjp on F64 tensors`() {
        val dir = doubleArrayOf(0.3, -1.1, 0.7, 0.2, -0.4)
        val bar = doubleArrayOf(1.5, -0.25, 0.8, -2.0, 0.6)
        val src = """
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndGrad
            import io.tlaloc.autograd.vjp
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val x = Tensors.f64Vector<Sym>(doubleArrayOf(${lit(x)}))
                val (value, g) = valueAndGrad { t: DTensor<Rank1<Sym>, F64> -> (t * t * t - t.exp()).sum().toDouble() }(x)
                println("value " + value)
                println("grad " + g.hostF64().joinToString(","))
                val j = jvp { t: DTensor<Rank1<Sym>, F64> -> (t.tanh() * t).sum().toDouble() }
                println("jvp " + j(x, Tensors.f64Vector<Sym>(doubleArrayOf(${lit(dir)}))))
                val p = vjp { t: DTensor<Rank1<Sym>, F64> -> t.exp() * t }
                println("vjp " + p(x, Tensors.f64Vector<Sym>(doubleArrayOf(${lit(bar)}))).hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(doubleArrayOf(x.sumOf { it * it * it - exp(it) }), r.values("value"), 1e-14, "value")
        assertClose(DoubleArray(5) { 3 * x[it] * x[it] - exp(x[it]) }, r.values("grad"), 1e-14, "grad")
        val jvp = x.indices.sumOf { ((1 - tanh(x[it]) * tanh(x[it])) * x[it] + tanh(x[it])) * dir[it] }
        assertClose(doubleArrayOf(jvp), r.values("jvp"), 1e-14, "jvp")
        assertClose(DoubleArray(5) { exp(x[it]) * (x[it] + 1) * bar[it] }, r.values("vjp"), 1e-14, "vjp")
    }

    @Test
    fun `tensor inputs, literals and captured Doubles never pass through a Float`() {
        // 1 + 1e-10 rounds to 1f; 0.1 and the captured 0.3 are 1.5e-8 and 4e-9 from their F32
        // roundings; 1e-9 vanishes next to 1 in F32. Each would move the result by more than
        // 1e-13 relative.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun scaleOf(v: Double) = v
            fun main() {
                val c = scaleOf(0.3)
                val off = scaleOf(1e-9)
                val g = grad { t: DTensor<Rank1<Sym>, F64> -> ((t + off) * (t + off) * t * 0.1 * c).sum().toDouble() }
                println("g " + g(Tensors.f64Vector<Sym>(doubleArrayOf(1.0 + 1e-10, 2.0))).hostF64().joinToString(","))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        val want = doubleArrayOf(1.0 + 1e-10, 2.0).map { t ->
            val u = t + 1e-9
            (2 * u * t + u * u) * 0.1 * 0.3
        }.toDoubleArray()
        assertClose(want, r.values("g"), 1e-13, "gradient")
    }
    @Test
    fun `an operation on F32 and F64 tensors is a DTYPE_MISMATCH at the operation`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F64> ->
                    (a * b).sum().toFloat()
                }
            }
        """.trimIndent()
        val r = F64TestHarness.compileAndRun(src)
        val mismatch = r.messages.filter { it.severity == CompilerMessageSeverity.ERROR && "Tlaloc dtype mismatch" in it.message }
        assertTrue(mismatch.size == 1, "expected one DTYPE_MISMATCH:\n${r.describe()}")
        // Line 6, column 10: the `(a * b)` expression.
        assertTrue(mismatch[0].line == 6 && mismatch[0].column == 10, "reported at ${mismatch[0].line}:${mismatch[0].column}")
        assertTrue("`times`" in mismatch[0].message && "F32" in mismatch[0].message && "F64" in mismatch[0].message, mismatch[0].message)
    }

    @Test
    fun `F32 and F64 tensors in one body, joined through a scalar conversion, are refused by name`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val g = grad2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F64> ->
                    (a * a).sum().toFloat() + (b * b).sum().toDouble().toFloat()
                }
            }
        """.trimIndent()
        val r = F64TestHarness.compileAndRun(src)
        assertTrue(r.exitCode != 0, "a body mixing F32 and F64 tensors compiled:\n${r.describe()}")
        assertTrue(
            r.messages.any { it.severity == CompilerMessageSeverity.ERROR && "mixes F32 and F64 tensors" in it.message },
            "no named refusal:\n${r.describe()}",
        )
    }
}
