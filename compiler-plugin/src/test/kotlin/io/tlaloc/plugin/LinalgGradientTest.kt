package io.tlaloc.plugin

import io.tlaloc.core.LinalgKernels
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.net.URLClassLoader
import java.nio.file.Files
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.Services
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The linear-algebra surface of `:core/ops/Linalg.kt` under `grad {}`, `grad2 {}`
 * and `jvp {}`, through the real K2 plugin. Each gradient must synthesize (no
 * "kept original call" fallback) and match central finite differences of the
 * same function computed in Double on [LinalgKernels].
 *
 * Tolerance: the synthesized gradient runs the F32 host twins, each of which
 * computes in Double and rounds once, so every op contributes about 6e-8 relative;
 * with condition numbers below 30 and under ten ops in any gradient body the error
 * stays below 1e-5 of the largest entry. The Double finite differences (h = 1e-5)
 * are good to about 1e-10. The bound, 1e-4 of the largest entry, leaves a factor
 * of ten; a wrong rule moves entries by O(1).
 */
class LinalgGradientTest {

    private val n = 4
    private val spd = doubleArrayOf(
        6.2, 1.1, -0.7, 0.4,
        1.3, 5.1, 0.9, -1.2,
        -0.5, 0.9, 4.8, 0.6,
        0.4, -1.0, 0.6, 5.5,
    )
    private val tri = doubleArrayOf(
        2.0, 9.0, -7.0, 5.0,
        0.6, 1.7, 8.0, -3.0,
        -0.4, 0.3, 2.4, 4.0,
        0.9, -0.8, 0.5, 1.9,
    )
    private val gen = doubleArrayOf(
        0.5, 2.0, -1.0, 0.3,
        1.2, -0.4, 0.8, 2.2,
        -3.0, 0.7, 1.5, -0.2,
        0.9, 1.1, -0.6, 1.4,
    )
    private val rhs = doubleArrayOf(0.7, -1.2, 0.4, 2.1, -0.3, 0.8, 1.5, -0.6)
    private val dir = DoubleArray(16) { ((it * 7) % 11 - 5) / 5.0 }

    private fun lit(a: DoubleArray) = a.joinToString(", ") { "${it.toFloat()}f" }

    private fun cubeSum(x: DoubleArray) = x.sumOf { it * it * it }

    private fun fdGrad(x: DoubleArray, loss: (DoubleArray) -> Double): DoubleArray {
        val h = 1e-5
        return DoubleArray(x.size) { i ->
            (loss(x.copyOf().also { it[i] += h }) - loss(x.copyOf().also { it[i] -= h })) / (2 * h)
        }
    }

    private fun assertClose(want: DoubleArray, got: List<Float>, what: String, relTol: Double = 1e-4) {
        assertEquals(want.size, got.size, "$what size (got $got)")
        val scale = max(1e-12, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= relTol * scale, "$what[$i] = ${got[i]}, want ${want[i]}")
        }
    }

    @Test
    fun `cholesky triangularSolve tril and triu differentiate through the plugin`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.cholesky
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            import io.tlaloc.core.ops.triangularSolve
            import io.tlaloc.core.ops.tril
            import io.tlaloc.core.ops.triu
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(spd)}))
                val t = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(tri)}))
                val b = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))

                val gChol = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val l = x.cholesky()
                    (l * l * l).sum().toFloat()
                }
                show("chol", gChol(a))

                val jChol = jvp { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val l = x.cholesky()
                    (l * l * l).sum().toFloat()
                }
                println("jchol " + jChol(a, v))

                val gSolve = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.triangularSolve(y, true)
                    (s * s * s).sum().toFloat()
                }
                val (dA, dB) = gSolve(t, b)
                show("solveA", dA)
                show("solveB", dB)

                val gSolveT = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.triangularSolve(y, false, true, false)
                    (s * s * s).sum().toFloat()
                }
                show("solveUT", gSolveT(t.triu(), b).first)

                val gTri = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val m = x.tril() + x.triu()
                    (m * m * m).sum().toFloat()
                }
                show("tri", gTri(t))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }

        val choleskyLoss = { x: DoubleArray -> cubeSum(LinalgKernels.cholesky(x, n)) }
        val gChol = fdGrad(spd, choleskyLoss)
        assertClose(gChol, rows.getValue("chol"), "d cholesky")
        assertClose(doubleArrayOf(gChol.indices.sumOf { gChol[it] * dir[it] }), rows.getValue("jchol"), "jvp cholesky")

        fun solve(a: DoubleArray, b: DoubleArray, lower: Boolean, transposeA: Boolean) =
            cubeSum(LinalgKernels.triangularSolve(a, b, n, 2, lower, transposeA, false))
        assertClose(fdGrad(tri) { solve(it, rhs, true, false) }, rows.getValue("solveA"), "d solve / dA")
        assertClose(fdGrad(rhs) { solve(tri, it, true, false) }, rows.getValue("solveB"), "d solve / dB")
        val triU = LinalgKernels.triangle(tri, n, n, 0.0, 1.0, 1.0)
        assertClose(fdGrad(triU) { solve(it, rhs, false, true) }, rows.getValue("solveUT"), "d solve(upper, transposed) / dA")

        val triLoss = { x: DoubleArray ->
            val m = LinalgKernels.triangle(x, n, n, 1.0, 2.0, 1.0)
            cubeSum(m)
        }
        assertClose(fdGrad(tri, triLoss), rows.getValue("tri"), "d (tril + triu)")
    }

    private fun spdSolveD(a: DoubleArray, b: DoubleArray): DoubleArray {
        val l = LinalgKernels.cholesky(a, n)
        val y = LinalgKernels.triangularSolve(l, b, n, 2, lower = true, transposeA = false, unitDiagonal = false)
        return LinalgKernels.triangularSolve(l, y, n, 2, lower = true, transposeA = true, unitDiagonal = false)
    }

    @Test
    fun `solveSpd differentiates through the plugin and agrees with implicit differentiation`() {
        val src = """
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.jvp2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.solveSpd
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(spd)}))
                val b = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))
                val vb = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs.map { -0.5 * it }.toDoubleArray())}))
                val g = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.solveSpd(y)
                    (s * s * s).sum().toFloat()
                }
                val (dA, dB) = g(a, b)
                show("dA", dA)
                show("dB", dB)
                val j = jvp2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.solveSpd(y)
                    (s * s * s).sum().toFloat()
                }
                println("jvp " + j(a, b, v, vb))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }
        val gA = fdGrad(spd) { cubeSum(spdSolveD(it, rhs)) }
        val gB = fdGrad(rhs) { cubeSum(spdSolveD(spd, it)) }
        assertClose(gA, rows.getValue("dA"), "d solveSpd / dA")
        assertClose(gB, rows.getValue("dB"), "d solveSpd / dB")
        val vb = rhs.map { -0.5 * it }.toDoubleArray()
        val wantJvp = gA.indices.sumOf { gA[it] * dir[it] } + gB.indices.sumOf { gB[it] * vb[it] }
        assertClose(doubleArrayOf(wantJvp), rows.getValue("jvp"), "jvp solveSpd")

        // Implicit differentiation: X̄ = 3X², B̄ = A⁻¹·X̄, Ā = −sym(B̄·Xᵀ).
        val x = spdSolveD(spd, rhs)
        val bBar = spdSolveD(spd, DoubleArray(x.size) { 3 * x[it] * x[it] })
        val implicit = DoubleArray(n * n) { idx ->
            val i = idx / n
            val j = idx % n
            val ij = (0 until 2).sumOf { bBar[i * 2 + it] * x[j * 2 + it] }
            val ji = (0 until 2).sumOf { bBar[j * 2 + it] * x[i * 2 + it] }
            -0.5 * (ij + ji)
        }
        assertClose(implicit, rows.getValue("dA"), "dA against −sym(B̄·Xᵀ)")
        assertClose(bBar, rows.getValue("dB"), "dB against A⁻¹·X̄")
    }

    /** `sym(A)⁻¹` in Double, the gradient of `log det sym(A)`. */
    private fun symInverseD(a: DoubleArray): DoubleArray {
        val sym = DoubleArray(n * n) { 0.5 * (a[it] + a[(it % n) * n + it / n]) }
        val l = LinalgKernels.cholesky(sym, n)
        val eye = DoubleArray(n * n) { if (it / n == it % n) 1.0 else 0.0 }
        val y = LinalgKernels.triangularSolve(l, eye, n, n, lower = true, transposeA = false, unitDiagonal = false)
        return LinalgKernels.triangularSolve(l, y, n, n, lower = true, transposeA = true, unitDiagonal = false)
    }

    @Test
    fun `logDetSpd has gradient A inverse, a matching jvp, and a hessian through the plugin`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.hessian
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.logDetSpd
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(spd)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))
                val g = grad { x: DTensor<Rank2<Sym, Sym>, F32> -> x.logDetSpd().toFloat() }
                show("grad", g(a))
                val j = jvp { x: DTensor<Rank2<Sym, Sym>, F32> -> x.logDetSpd().toFloat() }
                println("jvp " + j(a, v))
                val h = hessian { x: DTensor<Rank2<Sym, Sym>, F32> -> x.logDetSpd().toFloat() }
                show("hessian", h(a))
                println("value " + a.logDetSpd().toFloat())
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }
        val logDet = { x: DoubleArray ->
            val l = LinalgKernels.cholesky(x, n)
            2 * (0 until n).sumOf { kotlin.math.ln(l[it * n + it]) }
        }
        assertClose(doubleArrayOf(logDet(spd)), rows.getValue("value"), "log det")
        val inv = symInverseD(spd)
        assertClose(inv, rows.getValue("grad"), "d log det = sym(A)⁻¹")
        assertClose(fdGrad(spd, logDet), rows.getValue("grad"), "d log det against finite differences")
        assertClose(doubleArrayOf(inv.indices.sumOf { inv[it] * dir[it] }), rows.getValue("jvp"), "jvp log det")
        // Hessian: central differences of the exact gradient sym(A)⁻¹, column by
        // column (h = 1e-5: truncation 1e-10, rounding 1e-11 relative).
        val hWant = DoubleArray(n * n * n * n)
        for (k in 0 until n * n) {
            val hh = 1e-5
            val gp = symInverseD(spd.copyOf().also { it[k] += hh })
            val gm = symInverseD(spd.copyOf().also { it[k] -= hh })
            for (i in 0 until n * n) hWant[i * n * n + k] = (gp[i] - gm[i]) / (2 * hh)
        }
        assertClose(hWant, rows.getValue("hessian"), "hessian of log det")
    }

    @Test
    fun `invSpd and identityLike differentiate through the plugin`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.identityLike
            import io.tlaloc.core.ops.invSpd
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(spd)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))
                val g = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val m = x.invSpd()
                    (m * m * m).sum().toFloat()
                }
                show("inv", g(a))
                val j = jvp { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val m = x.invSpd()
                    (m * m * m).sum().toFloat()
                }
                println("jvp " + j(a, v))
                // identityLike is constant in x: d Σ(x ⊙ I + I) = I.
                val gi = grad { x: DTensor<Rank2<Sym, Sym>, F32> -> (x * x.identityLike() + x.identityLike()).sum().toFloat() }
                show("eye", gi(a))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }
        val loss = { x: DoubleArray -> cubeSum(symInverseD(x)) }
        val want = fdGrad(spd, loss)
        assertClose(want, rows.getValue("inv"), "d Σ(A⁻¹)³")
        assertClose(doubleArrayOf(want.indices.sumOf { want[it] * dir[it] }), rows.getValue("jvp"), "jvp Σ(A⁻¹)³")
        assertClose(DoubleArray(n * n) { if (it / n == it % n) 1.0 else 0.0 }, rows.getValue("eye"), "d Σ(x ⊙ I + I)")
    }

    /**
     * A Gaussian-process negative log likelihood with a rank-1 FIRST parameter.
     * With a square first parameter every unresolved IrType in the gradient body
     * happens to fall back to the right one; with a vector first, the stretch
     * BROADCASTs the log-determinant's adjoint emits must take their template's
     * IrType, or synthesis rejects the body.
     */
    @Test
    fun `a GP likelihood over a vector of hyperparameters differentiates through the plugin`() {
        val m = 5
        val xs = DoubleArray(m) { 0.7 * it }
        val ys = doubleArrayOf(0.3, -0.2, 0.9, 0.4, -0.6)
        val d = DoubleArray(m * m) { (xs[it / m] - xs[it % m]).let { r -> r * r } }
        val theta = doubleArrayOf(0.2, -0.3, -1.1)
        val src = """
            import io.tlaloc.autograd.grad3
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.exp
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.exp
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.ops.identityLike
            import io.tlaloc.core.ops.logDetSpd
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.solveSpd
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad3 { t: DTensor<Rank1<Sym>, F32>, d: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val k = (d * (-0.5f * (-2f * t[0]).exp())).exp() * (2f * t[1]).exp() + d.identityLike() * (2f * t[2]).exp()
                    0.5f * (y * k.solveSpd(y)).sum().toFloat() + 0.5f * k.logDetSpd().toFloat()
                }
                val t = Tensors.f32Vector<Sym>(floatArrayOf(${lit(theta)}))
                val d = Tensors.f32Matrix<Sym, Sym>($m, $m, floatArrayOf(${lit(d)}))
                val y = Tensors.f32Matrix<Sym, Sym>($m, 1, floatArrayOf(${lit(ys)}))
                println("theta " + g(t, d, y).first.hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val got = result.stdout.trim().lines().single { it.startsWith("theta ") }
            .removePrefix("theta ").split(",").map { it.toFloat() }
        fun nll(th: DoubleArray): Double {
            val k = DoubleArray(m * m) {
                kotlin.math.exp(2 * th[1]) * kotlin.math.exp(-0.5 * d[it] * kotlin.math.exp(-2 * th[0])) +
                    if (it / m == it % m) kotlin.math.exp(2 * th[2]) else 0.0
            }
            val l = LinalgKernels.cholesky(k, m)
            val z = LinalgKernels.triangularSolve(l, ys, m, 1, lower = true, transposeA = false, unitDiagonal = false)
            return 0.5 * z.sumOf { it * it } + (0 until m).sumOf { kotlin.math.ln(l[it * m + it]) }
        }
        // K's condition number here is about 60; F32 error stays below 1e-5 of the
        // largest entry, as for the other tests in this class.
        assertClose(fdGrad(theta, ::nll), got, "d nll / dθ")
    }

    @Test
    fun `solve and det differentiate through the plugin, det to second order`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.hessian
            import io.tlaloc.autograd.jvp2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.det
            import io.tlaloc.core.ops.solve
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(gen)}))
                val b = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))
                val vb = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs.map { -0.5 * it }.toDoubleArray())}))
                val g = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.solve(y)
                    (s * s * s).sum().toFloat()
                }
                val (dA, dB) = g(a, b)
                show("dA", dA)
                show("dB", dB)
                val gt = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.solve(y, true)
                    (s * s * s).sum().toFloat()
                }
                show("dAt", gt(a, b).first)
                val j = jvp2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.solve(y)
                    (s * s * s).sum().toFloat()
                }
                println("jvp " + j(a, b, v, vb))
                val gd = grad { x: DTensor<Rank2<Sym, Sym>, F32> -> x.det().toFloat() }
                show("ddet", gd(a))
                val hd = hessian { x: DTensor<Rank2<Sym, Sym>, F32> -> x.det().toFloat() }
                show("hdet", hd(a))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }
        fun loss(aa: DoubleArray, bb: DoubleArray, tr: Boolean) = cubeSum(LinalgKernels.solve(aa, bb, n, 2, tr))
        val gA = fdGrad(gen) { loss(it, rhs, false) }
        val gB = fdGrad(rhs) { loss(gen, it, false) }
        assertClose(gA, rows.getValue("dA"), "d solve / dA")
        assertClose(gB, rows.getValue("dB"), "d solve / dB")
        assertClose(fdGrad(gen) { loss(it, rhs, true) }, rows.getValue("dAt"), "d solve(transposed) / dA")
        val vb = rhs.map { -0.5 * it }.toDoubleArray()
        val wantJvp = gA.indices.sumOf { gA[it] * dir[it] } + gB.indices.sumOf { gB[it] * vb[it] }
        assertClose(doubleArrayOf(wantJvp), rows.getValue("jvp"), "jvp solve")
        val detD = { x: DoubleArray -> LinalgKernels.det(x, n) }
        assertClose(fdGrad(gen, detD), rows.getValue("ddet"), "d det")
        // Hessian against differences of the finite-difference gradient (nested
        // differences: h = 1e-4 outside, 1e-5 inside; error about 1e-6 relative).
        val hWant = DoubleArray(n * n * n * n)
        for (k in 0 until n * n) {
            val gp = fdGrad(gen.copyOf().also { it[k] += 1e-4 }, detD)
            val gm = fdGrad(gen.copyOf().also { it[k] -= 1e-4 }, detD)
            for (i in 0 until n * n) hWant[i * n * n + k] = (gp[i] - gm[i]) / 2e-4
        }
        assertClose(hWant, rows.getValue("hdet"), "hessian of det", relTol = 1e-3)
    }

    @Test
    fun `qrQ and qrR differentiate through the plugin on a tall matrix`() {
        val m = 5
        val k = 3
        val tall = doubleArrayOf(1.2, -0.7, 0.3, 0.4, 2.1, -1.1, -0.9, 0.5, 1.7, 0.6, -1.3, 0.2, 1.5, 0.8, -0.4)
        val v = DoubleArray(m * k) { ((it * 3) % 7 - 3) / 3.0 }
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.qrQ
            import io.tlaloc.core.ops.qrR
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>($m, $k, floatArrayOf(${lit(tall)}))
                val v = Tensors.f32Matrix<Sym, Sym>($m, $k, floatArrayOf(${lit(v)}))
                val g = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val q = x.qrQ()
                    val r = x.qrR()
                    ((q * q * q).sum().toFloat()) + ((r * r * r).sum().toFloat())
                }
                println("grad " + g(a).hostF32().joinToString(","))
                val j = jvp { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val q = x.qrQ()
                    val r = x.qrR()
                    ((q * q * q).sum().toFloat()) + ((r * r * r).sum().toFloat())
                }
                println("jvp " + j(a, v))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (key, value) = line.split(" ", limit = 2)
            key to value.split(",").map { it.toFloat() }
        }
        val loss = { x: DoubleArray ->
            val (q, r) = LinalgKernels.qr(x, m, k)
            cubeSum(q) + cubeSum(r)
        }
        val want = fdGrad(tall, loss)
        assertClose(want, rows.getValue("grad"), "d qr")
        assertClose(doubleArrayOf(want.indices.sumOf { want[it] * v[it] }), rows.getValue("jvp"), "jvp qr")
    }

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)

    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-linalg-test").toFile()
        try {
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(severity: CompilerMessageSeverity, message: String, location: CompilerMessageSourceLocation?) {
                    collected += CompileMessage(severity, message)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val capturedOut = PrintStream(baos, true, Charsets.UTF_8)
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(CompilerMessageSeverity.ERROR, "RUN FAILURE: ${t.cause ?: t}"),
                    baos.toString(Charsets.UTF_8),
                )
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
