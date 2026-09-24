package io.tlaloc.plugin

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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Stage D.3 — BGDHyperOpt, third OOPSLA 2021 benchmark port (PARTIAL).
 *
 * The paper's BGDHyperOpt kernel (Fig. 6, §6.2) is a meta-learning program:
 *   - An outer `while (k < T)` loop with `break` in an if/else.
 *   - A nested `for (i in 0 until M)` loop inside the while, summing per-data-point
 *     gradient contributions.
 *   - A post-while `for (j in 0 until M)` loop computing residual squared errors.
 *   - `err = sqrt(e / M)` as the final output.
 *   - `d(err)/dr` is the differentiation target (r = learning rate hyperparameter).
 *
 * Paper speedup: **23×-27× on differentiation** (the headline). Comes from PhiCalculus
 * coarsening the nested control flow into a closed-form expression via C6+C7+F2+F5.
 * §0.4.20 shipped the hand-built dxir version that demonstrates the C6+F2 closure on
 * the outer while loop; the full pipeline is known-correct at the `:ir` level.
 *
 * D.3 scope is SOURCE-LEVEL PORT through the plugin. Probing (in §0.4.49) revealed
 * three FIR surface gaps that block the full kernel:
 *   - Raw `while (cond)` loops (only desugared-for-loops are lowered).
 *   - `break` in loop bodies.
 *   - **Nested for loops (the outer `for (k)` around the inner `for (i)`).**
 *
 * What this file DOES port:
 *   1. The INNER for-loop sub-kernel (paper Fig. 6a line 6-7): `d = Σ 2x[i](y[i] - x[i]·w)`
 *      standalone, with x/y/w packed into a single rank-1 grad param. C7's closure-
 *      target shape at the `:ir` level; source-level it just compiles as a straight-
 *      line unroll via C5.
 *   2. The POST-WHILE error computation (Fig. 6a lines 12-15): `err = sqrt(Σ (y[j] -
 *      x[j]·w)²) / M`. Full rank-1 gradient via the D.1 GATHER + SCATTER_ADD chain.
 *
 * Numeric correctness via finite-difference cross-check on both. The OUTER-while +
 * break + nested-for composition is documented as deferred in §0.4.49.
 *
 * The packing trick (`packed[0]=w, packed[1..M]=x, packed[M+1..2M]=y`) sidesteps the
 * absence of multi-rank `grad2` / closure support — all inputs live in one rank-1
 * DTensor, extracted via GATHER with computed indices. Ergonomic only for small M;
 * real benchmark use would want closures or rank-mixed grad tuples.
 */
class BGDHyperOptTest {

    // --------------------------------------------------------------------
    // Sub-kernel 1: inner for-loop gradient computation
    // d(w) = Σ_i 2 · x[i] · (y[i] - x[i] · w)
    // --------------------------------------------------------------------

    @Test
    fun `inner-for bgd gradient sub-kernel matches analytic at w=1`() {
        // x = [1, 2, 3], y = [2, 4, 6] (y = 2x, so w=2 is the optimum; at w=1 residuals
        // are positive). Packed layout: [w=1, x0=1, x1=2, x2=3, y0=2, y1=4, y2=6].
        //
        // Inner BGD formula:
        //   d = Σ_{i=0..2} 2 · x[i] · (y[i] - x[i] · w)
        //     = 2·1·(2 - 1·1) + 2·2·(4 - 2·1) + 2·3·(6 - 3·1)
        //     = 2·1·1 + 2·2·2 + 2·3·3 = 2 + 8 + 18 = 28
        //
        // Differentiating wrt w:
        //   d(d)/dw = -Σ 2·x[i]² = -(2·1 + 2·4 + 2·9) = -28
        //
        // Differentiating wrt x[j]:
        //   d(d)/dx[j] = 2·(y[j] - x[j]·w) + 2·x[j]·(-w)
        //              = 2·y[j] - 2·x[j]·w - 2·x[j]·w = 2·y[j] - 4·x[j]·w
        //   At j=0: 2·2 - 4·1·1 = 0. At j=1: 2·4 - 4·2·1 = 0. At j=2: 2·6 - 4·3·1 = 0.
        //   Wait — only zero at w=1 for THIS particular (x, y) tuple. Different w gives
        //   nonzero gradients. We'll verify via finite-difference cross-check instead
        //   of hand-computing all 7 slots.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun prim(packed: DTensor<Rank1<Sym>, F32>): Float {
                val w = packed[0]
                var d = 0.0f
                for (i in 0 until 3) {
                    val xi = packed[1 + i]
                    val yi = packed[4 + i]
                    d = d + 2.0f * xi * (yi - xi * w)
                }
                return d
            }
            fun main() {
                val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
                    val w = packed[0]
                    var d = 0.0f
                    for (i in 0 until 3) {
                        val xi = packed[1 + i]
                        val yi = packed[4 + i]
                        d = d + 2.0f * xi * (yi - xi * w)
                    }
                    d
                }
                val base = floatArrayOf(1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                print(analytic.joinToString(","))
                print(";")
                val eps = 1.0e-3f
                for (k in 0 until 7) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 6) print(",")
                }
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(";")
        assertEquals(2, parts.size, "expected 'analytic;fd' layout, got:\n${result.stdout}")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        assertEquals(7, analytic.size)
        // Slot 0 (d/dw): expected -28 analytically.
        assertTrue(abs(analytic[0] + 28f) < 1e-3f, "expected d/dw = -28 at (w=1, x=[1,2,3], y=[2,4,6]); got ${analytic[0]}")
        // All other slots: FD cross-check within 2e-2 relative (loose; integer-like values).
        for (k in 0 until 7) {
            val absErr = abs(analytic[k] - fd[k])
            val relErr = absErr / (abs(fd[k]) + 1e-7f)
            assertTrue(
                absErr < 1e-3f || relErr < 2e-2f,
                "slot $k: analytic=${analytic[k]} fd=${fd[k]} absErr=$absErr relErr=$relErr",
            )
        }
    }

    // --------------------------------------------------------------------
    // Sub-kernel 2: post-while error computation
    // err(w) = sqrt(Σ_j (y[j] - x[j]·w)² / M)
    // --------------------------------------------------------------------

    @Test
    fun `post-while error sub-kernel matches finite-difference`() {
        // Same packing: [w, x0..x2, y0..y2]. M = 3.
        // At exact fit (y=2x, w=2): err=0 (minimum). Perturb w to 1: diffs=[1,2,3],
        // e=14, err=sqrt(14/3)≈2.1602.
        //
        // Hand-computed gradients at w=1:
        //   d(err)/dw   = -Σ(x[j]·diff[j])/(err·M) = -(1+4+9)/(2.16·3) = -14/6.48 ≈ -2.160
        //   d(err)/dx_j = -w·diff_j/(err·M) at w=1: [-1, -2, -3]/(2.16·3) ≈ [-0.154, -0.309, -0.463]
        //   d(err)/dy_j = +diff_j/(err·M)          at w=1: [+0.154, +0.309, +0.463]
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun prim(packed: DTensor<Rank1<Sym>, F32>): Float {
                val w = packed[0]
                var e = 0.0f
                for (j in 0 until 3) {
                    val xj = packed[1 + j]
                    val yj = packed[4 + j]
                    val diff = yj - xj * w
                    e = e + diff * diff
                }
                return (e / 3.0f).sqrt()
            }
            fun main() {
                val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
                    val w = packed[0]
                    var e = 0.0f
                    for (j in 0 until 3) {
                        val xj = packed[1 + j]
                        val yj = packed[4 + j]
                        val diff = yj - xj * w
                        e = e + diff * diff
                    }
                    (e / 3.0f).sqrt()
                }
                val base = floatArrayOf(1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                print(analytic.joinToString(","))
                print(";")
                val eps = 1.0e-3f
                for (k in 0 until 7) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 6) print(",")
                }
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(";")
        assertEquals(2, parts.size, "expected 'analytic;fd' layout, got:\n${result.stdout}")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        assertEquals(7, analytic.size)

        // Hand-computed reference at w=1: err ≈ 2.1602.
        val err = kotlin.math.sqrt(14.0f / 3.0f)
        val expectedDw = -14.0f / (err * 3.0f)                                        // ≈ -2.160
        val expectedDx = floatArrayOf(-1.0f, -2.0f, -3.0f).map { it / (err * 3.0f) }  // ≈ -0.154, -0.309, -0.463
        val expectedDy = floatArrayOf(1.0f, 2.0f, 3.0f).map { it / (err * 3.0f) }     // ≈ +0.154, +0.309, +0.463

        assertTrue(abs(analytic[0] - expectedDw) < 1e-3f, "slot 0 (d/dw): expected $expectedDw, got ${analytic[0]}")
        for (j in 0 until 3) {
            assertTrue(
                abs(analytic[1 + j] - expectedDx[j]) < 1e-3f,
                "slot ${1 + j} (d/dx_$j): expected ${expectedDx[j]}, got ${analytic[1 + j]}",
            )
            assertTrue(
                abs(analytic[4 + j] - expectedDy[j]) < 1e-3f,
                "slot ${4 + j} (d/dy_$j): expected ${expectedDy[j]}, got ${analytic[4 + j]}",
            )
        }
        // FD cross-check for all slots.
        for (k in 0 until 7) {
            val absErr = abs(analytic[k] - fd[k])
            val relErr = absErr / (abs(fd[k]) + 1e-7f)
            assertTrue(
                absErr < 1e-3f || relErr < 5e-3f,
                "slot $k FD: analytic=${analytic[k]} fd=${fd[k]} absErr=$absErr relErr=$relErr",
            )
        }
    }

    // --------------------------------------------------------------------
    // §0.4.50 — Nested for-loop probe (closes D.3 Gap 1).
    //
    // Before §0.4.50, `lowerDesugaredForLoop` hard-cast `emitter as? DxirBuilder`,
    // so an inner for-loop emitted from a DxirRegionBuilder (the outer for-loop's
    // WHILE body) threw "for-loop inside another region/branch not supported".
    // Lifting `whileOp` to the `DxirEmitter` interface and dropping the cast lets
    // nested for-loops lower via C5-unroll on both levels.
    // --------------------------------------------------------------------

    @Test
    fun `nested for-loop sums each element twice`() {
        // Outer k in 0..1, inner i in 0..2, body `total = total + arr[i]`.
        // At arr = [10, 20, 30]: inner sum = 60 per k-iter, outer accumulates to 120.
        // Each arr[i] is visited K times → gradient = [K, K, K] = [2, 2, 2].
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> ->
                    var total = 0.0f
                    for (k in 0 until 2) {
                        for (i in 0 until 3) {
                            total = total + arr[i]
                        }
                    }
                    total
                }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(10.0f, 20.0f, 30.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_3, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("2.0,2.0,2.0", result.stdout.trim())
    }

    // --------------------------------------------------------------------
    // §0.4.51 — Full BGDHyperOpt kernel source port (no break).
    //
    // Paper-faithful kernel (Fig. 6a minus the `if d<ε break` convergence check):
    //   1. Inner for i : compute Sxy = Σ x[i]·y[i], Sx2 = Σ x[i]²
    //   2. Outer while k : w_{k+1} = w_k - r·d_k/M where d_k = 2·(Sx2·w_k - Sxy)
    //   3. Post-while for j : err = sqrt(Σ (y[j] - x[j]·w_T)² / M)
    //
    // Paper's 23× speedup comes from PhiCalculus unrolling the inner fors (C5) and
    // closing the outer while via affine-recurrence closure (C6 with loop-invariant
    // Sxy/Sx2/M params per §0.4.20). The break is a convergence shortcut; §0.4.20
    // already demonstrated C6 closure on the affine outer loop, so porting the full
    // kernel without break is the unlock.
    //
    // Packed input: [r, x_0..x_{M-1}, y_0..y_{M-1}]. M = 3 keeps the 7-slot stub.
    // Gradient target: slot 0 = d(err)/dr (the hyperparameter being tuned).
    // --------------------------------------------------------------------

    @Test
    fun `full bgd-hyperopt kernel no-break ports end-to-end and gradient matches FD`() {
        // x = [1, 2, 3], y = [2, 4, 6]  →  perfect linear fit at w = 2.
        // Sxy = 28, Sx2 = 14, M = 3. At r = 0.01, T = 3:
        //   a = 1 - 2r·Sx2/M = 1 - 0.0933 ≈ 0.9067
        //   b = 2r·Sxy/M     = 0.1867
        //   w_3 = b·(1+a+a²)·… ≈ 0.509
        //   err = sqrt(Σ(y-xw)²/M) ≈ 3.22
        // FD sweep cross-checks the whole gradient vector; slot 0 is the headline.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun prim(packed: DTensor<Rank1<Sym>, F32>): Float {
                val r = packed[0]
                val Mf = 3.0f
                var Sxy = 0.0f
                var Sx2 = 0.0f
                for (i in 0 until 3) {
                    val xi = packed[1 + i]
                    val yi = packed[4 + i]
                    Sxy = Sxy + xi * yi
                    Sx2 = Sx2 + xi * xi
                }
                var w = 0.0f
                var k = 0
                while (k < 3) {
                    val d = 2.0f * (Sx2 * w - Sxy)
                    w = w - r * d / Mf
                    k = k + 1
                }
                var e = 0.0f
                for (j in 0 until 3) {
                    val xj = packed[1 + j]
                    val yj = packed[4 + j]
                    val diff = yj - xj * w
                    e = e + diff * diff
                }
                return (e / Mf).sqrt()
            }
            fun main() {
                val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
                    val r = packed[0]
                    val Mf = 3.0f
                    var Sxy = 0.0f
                    var Sx2 = 0.0f
                    for (i in 0 until 3) {
                        val xi = packed[1 + i]
                        val yi = packed[4 + i]
                        Sxy = Sxy + xi * yi
                        Sx2 = Sx2 + xi * xi
                    }
                    var w = 0.0f
                    var k = 0
                    while (k < 3) {
                        val d = 2.0f * (Sx2 * w - Sxy)
                        w = w - r * d / Mf
                        k = k + 1
                    }
                    var e = 0.0f
                    for (j in 0 until 3) {
                        val xj = packed[1 + j]
                        val yj = packed[4 + j]
                        val diff = yj - xj * w
                        e = e + diff * diff
                    }
                    (e / Mf).sqrt()
                }
                val base = floatArrayOf(0.01f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                print(analytic.joinToString(","))
                print(";")
                val eps = 1.0e-4f
                for (k in 0 until 7) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 6) print(",")
                }
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val unsupported = result.messages.filter {
            it.message.contains("Tlaloc could not lower lambda") ||
                it.message.contains("kept original call")
        }
        assertEquals(
            0,
            unsupported.size,
            "full kernel must lower through the plugin (no fallback); got:\n" +
                result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" },
        )
        val parts = result.stdout.trim().split(";")
        assertEquals(2, parts.size, "expected 'analytic;fd' layout, got:\n${result.stdout}")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        assertEquals(7, analytic.size)
        assertEquals(7, fd.size)
        // Sentinel-reject — -1.0f means the runtime-tape stub fired instead of the synthesised
        // gradient, i.e., the IR transform was rejected silently.
        for ((k, a) in analytic.withIndex()) {
            assertTrue(
                abs(a + 1.0f) > 1e-3f,
                "slot $k analytic=$a matches broken-stub sentinel; IR transform did not fire",
            )
        }
        // FD cross-check on every slot. The outer while is affine in w, so C5 unrolls
        // the 3-iter concrete trip count cleanly; numerical accuracy should be tight.
        val report = (0 until 7).joinToString("\n") {
            val absErr = abs(analytic[it] - fd[it])
            val relErr = absErr / (abs(fd[it]) + 1e-7f)
            "slot $it: analytic=${analytic[it]} fd=${fd[it]} absErr=$absErr relErr=$relErr"
        }
        for (k in 0 until 7) {
            val absErr = abs(analytic[k] - fd[k])
            val relErr = absErr / (abs(fd[k]) + 1e-7f)
            assertTrue(
                absErr < 1e-2f || relErr < 5e-3f,
                "slot $k mismatch. Full report:\n$report",
            )
        }
    }

    @Test
    fun `single-accum inner-for plus outer while gradient matches FD`() {
        // §0.4.51 — pre-fix regression anchor: the single-accum variant worked before
        // §0.4.51; the two-accum variant (full kernel) broke because of the cloneNode
        // multi-result index bug + C5's single-carried restriction. Keeping this test
        // so that if either fix regresses, we catch it at the single-accum level
        // separately from the full-kernel test.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun prim(p: DTensor<Rank1<Sym>, F32>): Float {
                val r = p[0]; val Sx2 = 14.0f; val Mf = 3.0f
                var Sxy = 0.0f
                for (i in 0 until 3) {
                    val xi = p[1 + i]; val yi = p[4 + i]
                    Sxy = Sxy + xi * yi
                }
                var w = 0.0f; var k = 0
                while (k < 3) {
                    val d = 2.0f * (Sx2 * w - Sxy)
                    w = w - r * d / Mf
                    k = k + 1
                }
                return w
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val r = p[0]; val Sx2 = 14.0f; val Mf = 3.0f
                    var Sxy = 0.0f
                    for (i in 0 until 3) {
                        val xi = p[1 + i]; val yi = p[4 + i]
                        Sxy = Sxy + xi * yi
                    }
                    var w = 0.0f; var k = 0
                    while (k < 3) {
                        val d = 2.0f * (Sx2 * w - Sxy)
                        w = w - r * d / Mf
                        k = k + 1
                    }
                    w
                }
                val base = floatArrayOf(0.01f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                print(analytic.joinToString(","))
                print(";")
                val eps = 1.0e-4f
                for (k in 0 until 7) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 6) print(",")
                }
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(";")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        val report = (0 until 7).joinToString("\n") {
            "slot $it: analytic=${analytic[it]} fd=${fd[it]}"
        }
        for (k in 0 until 7) {
            val absErr = abs(analytic[k] - fd[k])
            val relErr = absErr / (abs(fd[k]) + 1e-7f)
            assertTrue(
                absErr < 1e-1f || relErr < 1e-2f,
                "slot $k mismatch.\n$report",
            )
        }
    }

    @Test
    fun `bgd-hyperopt pre-simplified affine-form at T=50 M=3 — measured timings`() {
        // §0.4.51 — probe whether C6 closure fires when the user writes the affine
        // recurrence directly as `w = a*w + b` (matches detectAffineRecurrence shape
        // #1). If times stay near T=10 (i.e., gradient doesn't grow with T), C6
        // closure works; the remaining gap to the natural kernel is just pattern-match
        // tolerance for `w = w - r*(...)` (which semantically expands to affine).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun primal(p: DTensor<Rank1<Sym>, F32>): Float {
                val r = p[0]; val Mf = 3.0f
                var Sxy = 0.0f; var Sx2 = 0.0f
                for (i in 0 until 3) {
                    val xi = p[1 + i]; val yi = p[4 + i]
                    Sxy = Sxy + xi * yi
                    Sx2 = Sx2 + xi * xi
                }
                val a = 1.0f - 2.0f * r * Sx2 / Mf
                val b = 2.0f * r * Sxy / Mf
                var w = 0.0f; var k = 0
                while (k < 50) {
                    w = a * w + b
                    k = k + 1
                }
                var e = 0.0f
                for (j in 0 until 3) {
                    val xj = p[1 + j]; val yj = p[4 + j]
                    val diff = yj - xj * w
                    e = e + diff * diff
                }
                return (e / Mf).sqrt()
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val r = p[0]; val Mf = 3.0f
                    var Sxy = 0.0f; var Sx2 = 0.0f
                    for (i in 0 until 3) {
                        val xi = p[1 + i]; val yi = p[4 + i]
                        Sxy = Sxy + xi * yi
                        Sx2 = Sx2 + xi * xi
                    }
                    val a = 1.0f - 2.0f * r * Sx2 / Mf
                    val b = 2.0f * r * Sxy / Mf
                    var w = 0.0f; var k = 0
                    while (k < 50) {
                        w = a * w + b
                        k = k + 1
                    }
                    var e = 0.0f
                    for (j in 0 until 3) {
                        val xj = p[1 + j]; val yj = p[4 + j]
                        val diff = yj - xj * w
                        e = e + diff * diff
                    }
                    (e / Mf).sqrt()
                }
                val arr = floatArrayOf(0.005f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(arr)
                var sinkF = 0.0f; var sinkG = 0.0f
                for (i in 0 until 1000) {
                    sinkF += primal(input)
                    sinkG += g(input).hostF32()[0]
                }
                val iters = 5000
                val tFwdStart = System.nanoTime()
                for (i in 0 until iters) sinkF += primal(input)
                val tFwdNs = (System.nanoTime() - tFwdStart) / iters
                val tGradStart = System.nanoTime()
                for (i in 0 until iters) sinkG += g(input).hostF32()[0]
                val tGradNs = (System.nanoTime() - tGradStart) / iters
                val checkSlot0 = g(input).hostF32()[0]
                println("PERF forward_ns_per_call=${'$'}tFwdNs")
                println("PERF gradient_ns_per_call=${'$'}tGradNs")
                println("PERF ratio_grad_over_fwd=${'$'}{tGradNs.toDouble() / tFwdNs}")
                println("PERF first_grad_slot0=${'$'}checkSlot0")
                println("PERF sink=${'$'}sinkF/${'$'}sinkG")
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val perfLines = result.stdout.lines().filter { it.startsWith("PERF ") }
        assertEquals(5, perfLines.size, "expected 5 PERF lines, got:\n${result.stdout}")
        val metrics = perfLines.associate {
            val (k, v) = it.removePrefix("PERF ").split("=", limit = 2)
            k to v
        }
        val fwdNs = metrics["forward_ns_per_call"]!!.toLong()
        val gradNs = metrics["gradient_ns_per_call"]!!.toLong()
        val ratio = metrics["ratio_grad_over_fwd"]!!.toDouble()
        val firstSlot = metrics["first_grad_slot0"]!!.toFloat()
        assertTrue(abs(firstSlot + 1.0f) > 1e-3f, "sentinel reject: first_grad_slot0=$firstSlot")
        assertTrue(ratio > 0.05 && ratio < 200.0, "ratio=$ratio outside bounds")
        println("[BGD pre-simplified T=50 M=3 perf] forward=${fwdNs}ns/call gradient=${gradNs}ns/call ratio=${ratio}")
    }

    @Test
    fun `paper-faithful bgd-hyperopt at T=50 M=3 — measured timings`() {
        // §0.4.51 — scaling probe. If C6 closes the outer-while into an O(1) closed
        // form, gradient time stays near the T=10 number; if C5 merely unrolls, both
        // forward and gradient scale linearly in T.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun primal(p: DTensor<Rank1<Sym>, F32>): Float {
                val r = p[0]; val Mf = 3.0f
                var Sxy = 0.0f; var Sx2 = 0.0f
                for (i in 0 until 3) {
                    val xi = p[1 + i]; val yi = p[4 + i]
                    Sxy = Sxy + xi * yi
                    Sx2 = Sx2 + xi * xi
                }
                var w = 0.0f; var k = 0
                while (k < 50) {
                    val d = 2.0f * (Sx2 * w - Sxy)
                    w = w - r * d / Mf
                    k = k + 1
                }
                var e = 0.0f
                for (j in 0 until 3) {
                    val xj = p[1 + j]; val yj = p[4 + j]
                    val diff = yj - xj * w
                    e = e + diff * diff
                }
                return (e / Mf).sqrt()
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val r = p[0]; val Mf = 3.0f
                    var Sxy = 0.0f; var Sx2 = 0.0f
                    for (i in 0 until 3) {
                        val xi = p[1 + i]; val yi = p[4 + i]
                        Sxy = Sxy + xi * yi
                        Sx2 = Sx2 + xi * xi
                    }
                    var w = 0.0f; var k = 0
                    while (k < 50) {
                        val d = 2.0f * (Sx2 * w - Sxy)
                        w = w - r * d / Mf
                        k = k + 1
                    }
                    var e = 0.0f
                    for (j in 0 until 3) {
                        val xj = p[1 + j]; val yj = p[4 + j]
                        val diff = yj - xj * w
                        e = e + diff * diff
                    }
                    (e / Mf).sqrt()
                }
                val arr = floatArrayOf(0.005f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(arr)
                var sinkF = 0.0f; var sinkG = 0.0f
                for (i in 0 until 1000) {
                    sinkF += primal(input)
                    sinkG += g(input).hostF32()[0]
                }
                val iters = 5000
                val tFwdStart = System.nanoTime()
                for (i in 0 until iters) sinkF += primal(input)
                val tFwdNs = (System.nanoTime() - tFwdStart) / iters
                val tGradStart = System.nanoTime()
                for (i in 0 until iters) sinkG += g(input).hostF32()[0]
                val tGradNs = (System.nanoTime() - tGradStart) / iters
                val checkSlot0 = g(input).hostF32()[0]
                println("PERF forward_ns_per_call=${'$'}tFwdNs")
                println("PERF gradient_ns_per_call=${'$'}tGradNs")
                println("PERF ratio_grad_over_fwd=${'$'}{tGradNs.toDouble() / tFwdNs}")
                println("PERF first_grad_slot0=${'$'}checkSlot0")
                println("PERF sink=${'$'}sinkF/${'$'}sinkG")
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val perfLines = result.stdout.lines().filter { it.startsWith("PERF ") }
        assertEquals(5, perfLines.size, "expected 5 PERF lines, got:\n${result.stdout}")
        val metrics = perfLines.associate {
            val (k, v) = it.removePrefix("PERF ").split("=", limit = 2)
            k to v
        }
        val fwdNs = metrics["forward_ns_per_call"]!!.toLong()
        val gradNs = metrics["gradient_ns_per_call"]!!.toLong()
        val ratio = metrics["ratio_grad_over_fwd"]!!.toDouble()
        val firstSlot = metrics["first_grad_slot0"]!!.toFloat()
        assertTrue(abs(firstSlot + 1.0f) > 1e-3f, "sentinel reject: first_grad_slot0=$firstSlot")
        assertTrue(ratio > 0.05 && ratio < 200.0, "ratio=$ratio outside bounds")
        println("[BGDHyperOpt T=50 M=3 perf] forward=${fwdNs}ns/call gradient=${gradNs}ns/call ratio=${ratio}")
    }

    @Test
    fun `paper-faithful bgd-hyperopt at T=10 M=3 — measured timings`() {
        // §0.4.51 — measure the unlocked paper-speedup path. Primal: inner-for sums
        // Sxy/Sx2, outer-while runs T affine recurrence iters on w, post-while-for
        // computes sqrt-ed MSE. Forward vs gradient timing; ratio should stay small
        // if C6 closure fires on the outer-while (whose iteration count dominates).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun primal(p: DTensor<Rank1<Sym>, F32>): Float {
                val r = p[0]; val Mf = 3.0f
                var Sxy = 0.0f; var Sx2 = 0.0f
                for (i in 0 until 3) {
                    val xi = p[1 + i]; val yi = p[4 + i]
                    Sxy = Sxy + xi * yi
                    Sx2 = Sx2 + xi * xi
                }
                var w = 0.0f; var k = 0
                while (k < 10) {
                    val d = 2.0f * (Sx2 * w - Sxy)
                    w = w - r * d / Mf
                    k = k + 1
                }
                var e = 0.0f
                for (j in 0 until 3) {
                    val xj = p[1 + j]; val yj = p[4 + j]
                    val diff = yj - xj * w
                    e = e + diff * diff
                }
                return (e / Mf).sqrt()
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val r = p[0]; val Mf = 3.0f
                    var Sxy = 0.0f; var Sx2 = 0.0f
                    for (i in 0 until 3) {
                        val xi = p[1 + i]; val yi = p[4 + i]
                        Sxy = Sxy + xi * yi
                        Sx2 = Sx2 + xi * xi
                    }
                    var w = 0.0f; var k = 0
                    while (k < 10) {
                        val d = 2.0f * (Sx2 * w - Sxy)
                        w = w - r * d / Mf
                        k = k + 1
                    }
                    var e = 0.0f
                    for (j in 0 until 3) {
                        val xj = p[1 + j]; val yj = p[4 + j]
                        val diff = yj - xj * w
                        e = e + diff * diff
                    }
                    (e / Mf).sqrt()
                }
                val arr = floatArrayOf(0.005f, 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)
                val input = Tensors.f32Vector<Sym>(arr)

                // Warmup 1000 interleaved.
                var sinkF = 0.0f
                var sinkG = 0.0f
                for (i in 0 until 1000) {
                    sinkF += primal(input)
                    sinkG += g(input).hostF32()[0]
                }
                val iters = 5000
                val tFwdStart = System.nanoTime()
                for (i in 0 until iters) sinkF += primal(input)
                val tFwdNs = (System.nanoTime() - tFwdStart) / iters

                val tGradStart = System.nanoTime()
                for (i in 0 until iters) sinkG += g(input).hostF32()[0]
                val tGradNs = (System.nanoTime() - tGradStart) / iters

                val checkSlot0 = g(input).hostF32()[0]

                println("PERF forward_ns_per_call=${'$'}tFwdNs")
                println("PERF gradient_ns_per_call=${'$'}tGradNs")
                println("PERF ratio_grad_over_fwd=${'$'}{tGradNs.toDouble() / tFwdNs}")
                println("PERF first_grad_slot0=${'$'}checkSlot0")
                println("PERF sink=${'$'}sinkF/${'$'}sinkG")
            }
        """.trimIndent()
        val result = compileAndRun(STUB_RANK1_TO_FLOAT_7, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val perfLines = result.stdout.lines().filter { it.startsWith("PERF ") }
        assertEquals(5, perfLines.size, "expected 5 PERF lines, got:\n${result.stdout}")
        val metrics = perfLines.associate { line ->
            val (k, v) = line.removePrefix("PERF ").split("=", limit = 2)
            k to v
        }
        val fwdNs = metrics["forward_ns_per_call"]!!.toLong()
        val gradNs = metrics["gradient_ns_per_call"]!!.toLong()
        val ratio = metrics["ratio_grad_over_fwd"]!!.toDouble()
        val firstSlot = metrics["first_grad_slot0"]!!.toFloat()
        assertTrue(
            abs(firstSlot + 1.0f) > 1e-3f,
            "first_grad_slot0=$firstSlot matches broken-stub sentinel; IR transform did not fire",
        )
        assertTrue(ratio > 0.05 && ratio < 200.0, "ratio=$ratio outside sanity bounds")
        println("[BGDHyperOpt T=10 M=3 perf] forward=${fwdNs}ns/call gradient=${gradNs}ns/call ratio=${ratio}")
    }

    @Test
    fun `isolated outer while gradient matches FD`() {
        // Just the outer while, with Sxy/Sx2/Mf/r as rank-1 params packed as
        // [r, Sxy, Sx2]. M = 3 hardcoded. Verifies the raw while + Int counter
        // gradient is correct in isolation (no inner for, no post-while sqrt).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun prim(p: DTensor<Rank1<Sym>, F32>): Float {
                val r = p[0]; val Sxy = p[1]; val Sx2 = p[2]; val Mf = 3.0f
                var w = 0.0f; var k = 0
                while (k < 3) {
                    val d = 2.0f * (Sx2 * w - Sxy)
                    w = w - r * d / Mf
                    k = k + 1
                }
                return w
            }
            fun main() {
                val g = grad { p: DTensor<Rank1<Sym>, F32> ->
                    val r = p[0]; val Sxy = p[1]; val Sx2 = p[2]; val Mf = 3.0f
                    var w = 0.0f; var k = 0
                    while (k < 3) {
                        val d = 2.0f * (Sx2 * w - Sxy)
                        w = w - r * d / Mf
                        k = k + 1
                    }
                    w
                }
                val base = floatArrayOf(0.01f, 28.0f, 14.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                print(analytic.joinToString(","))
                print(";")
                val eps = 1.0e-4f
                for (k in 0 until 3) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 2) print(",")
                }
            }
        """.trimIndent()
        val stub3 = STUB_RANK1_TO_FLOAT_3
        val result = compileAndRun(stub3, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(";")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        val report = (0 until 3).joinToString("\n") {
            "slot $it: analytic=${analytic[it]} fd=${fd[it]}"
        }
        for (k in 0 until 3) {
            val absErr = abs(analytic[k] - fd[k])
            val relErr = absErr / (abs(fd[k]) + 1e-7f)
            assertTrue(
                absErr < 1e-1f || relErr < 5e-3f,
                "slot $k mismatch. messages:\n${result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" }}\nReport:\n$report",
            )
        }
    }

    // --------- Harness ---------

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-bgd-run").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean = collected.any { it.severity == CompilerMessageSeverity.ERROR }
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
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        /** Sentinel-stub sized for a 7-slot packed [w, x0..x2, y0..y2] layout. */
        private val STUB_RANK1_TO_FLOAT_7 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(7) { -1.0f }), intArrayOf(7), F32) }
        """.trimIndent()

        /** Sentinel-stub sized for a 3-slot rank-1 grad (nested for-loop probe). */
        private val STUB_RANK1_TO_FLOAT_3 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
        """.trimIndent()
    }
}
