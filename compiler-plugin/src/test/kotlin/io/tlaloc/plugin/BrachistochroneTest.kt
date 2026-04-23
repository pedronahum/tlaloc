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
 * Stage D.1 — first OOPSLA 2021 benchmark port (per plan §9.1).
 *
 * The paper's Brachistochrone kernel (§6.2 of coarsening-autodiff.txt) computes the
 * descent time of a bead sliding under gravity along a piecewise-linear curve, summing
 * per-segment times each involving `sqrt(2·g·h_i)`. Stage D.1a (§0.4.37) shipped a
 * sqrt-free proof-of-pipeline port because two blockers prevented a paper-faithful
 * kernel: scalar `Float.sqrt()` wasn't wired at the FIR surface, and
 * `DxirToIrSynthesis.irOpFor` had no arm for `OpKind.SQRT`. Stage D.1b (§0.4.38, this
 * file's second half) discharges both blockers:
 *   1. `fun Float.sqrt() / fun Double.sqrt()` extensions in `:core/DScalar.kt` (FQN
 *      `io.tlaloc.core.sqrt`) mirror the `relu` pattern from §0.4.7.
 *   2. `FirLambdaToDxirLowering.UNARY_OP_MAP` maps `io.tlaloc.core.sqrt` → `OpKind.SQRT`.
 *   3. `DxirToIrSynthesis.irSqrt` emits `IrCall` back into the `:core` extension.
 *
 * The SqrtRule VjpRegistry entry landed in §0.4.22; this slice unblocks the compile
 * path for `grad { y -> ...y.sqrt()... }` end-to-end, no runtime-tape fallback.
 *
 * File scope (both Stage D.1a + D.1b tests):
 *
 *   - compound-velocity:  `v = v + v*y` over N iterations → `v_final = (1 + y)^N`.
 *     ADD-of-MUL back-edge where the MUL references the carried value — this shape
 *     deliberately defeats PhiCalculus C6 (C6 requires `b` loop-invariant; our `b =
 *     v*y` references the carried arg), so C5's concrete-N unroll fires instead and
 *     produces a straight-line MUL/ADD tree. If the body were `v = v * (1 + y)` C6
 *     would match with `a = 1+y` (loop-invariant), and Symja's closed form
 *     `pow(1+y, N)` would emit `OpKind.POW` in the gradient — which
 *     `DxirToIrSynthesis.irOpFor` has no arm for (only ADD/SUB/MUL/DIV/NEG + a few
 *     specials), so synthesis would null out and the plugin would fall back to the
 *     runtime tape. Rewriting as `v + v*y` keeps the same closed form `(1+y)^N` but
 *     routes through C5, which synthesis handles cleanly. Empirically verified.
 *
 *   - energy-accumulation: `ke += 4·y` over N iterations → `ke = 4·N·y`.
 *     ADD back-edge on a loop-invariant runtime subtree (shape `ADD(carried,
 *     MUL(const, y))`). Matches C6 shape (3): `a = 1` implicit, `b = 4y` doesn't
 *     reference carried. Engine closes to `4·N·y` — linear, no POW. Models kinetic-
 *     energy accumulation under gravity across N discrete drops (v² proxy avoids
 *     sqrt).
 *
 * Both primals run via `AUTOGRAD_STUB_BROKEN` (the `grad { ... }` runtime fallback
 * returns sentinel `-1.0f`), so a matching printed gradient is definitive proof the
 * IR-rewrite path fired. Mirrors §0.4.25's loop e2e pattern.
 *
 * SOI coarsening (`tlaloc.soi.enabled=true`) is deliberately not exercised here:
 * `coarsenFunction` passes WHILE-bearing primals through unchanged (§0.4.36's
 * `coarsenFunctionPreservesWhileContainingPrimal`), then `DxirReverseTransform`
 * rejects WHILE (`require(n.op == OpKind.IF)` at DxirReverseTransform.kt:104) and
 * the plugin falls back to runtime tape. This is §0.4.36's filed follow-up
 * ("multi-result COARSENED for WHILE body yield splicing") and out of D.1a scope.
 */
class BrachistochroneTest {

    @Test
    fun `compound-velocity primal matches analytic gradient at representative y values`() {
        // f(y) = (1 + y)^5.  f'(y) = 5·(1 + y)^4.
        //   at y=0:    f'(0)   = 5·1^4    = 5.0
        //   at y=0.5:  f'(0.5) = 5·1.5^4  = 5·5.0625 = 25.3125 (exact in f32)
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { y: Float ->
                    var v = 1.0f
                    for (i in 0 until 5) {
                        v = v + v * y
                    }
                    v
                }
                println(g(0.0f))
                println(g(0.5f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("5.0\n25.3125", result.stdout.trim())
    }

    @Test
    fun `compound-velocity gradient agrees with finite-difference reference over a sweep`() {
        // Finite-difference cross-check against the plugin-emitted gradient.
        // Central difference at eps=1e-3 in f32; tolerance 2e-2 relative to account
        // for f32 subtractive cancellation at larger y.
        val src = """
            import io.tlaloc.autograd.grad
            fun prim(y: Float): Float {
                var v = 1.0f
                for (i in 0 until 5) {
                    v = v * (1.0f + y)
                }
                return v
            }
            fun main() {
                val g = grad { y: Float ->
                    var v = 1.0f
                    for (i in 0 until 5) {
                        v = v + v * y
                    }
                    v
                }
                val eps = 1.0e-3f
                for (y in listOf(0.0f, 0.1f, 0.25f, 0.5f, 1.0f)) {
                    val analytic = g(y)
                    val fd = (prim(y + eps) - prim(y - eps)) / (2.0f * eps)
                    println("${'$'}y ${'$'}analytic ${'$'}fd")
                }
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        val lines = result.stdout.trim().lines()
        assertEquals(5, lines.size, "expected 5 'y analytic fd' lines, got:\n${result.stdout}")
        for (line in lines) {
            val parts = line.split(" ")
            assertEquals(3, parts.size, "expected 'y analytic fd' format; got: $line")
            val y = parts[0].toFloat()
            val analytic = parts[1].toFloat()
            val fd = parts[2].toFloat()
            // Reject the broken-stub sentinel (-1.0) — that would mean the compile
            // path did not fire and we're reading the runtime-tape fallback.
            assertTrue(
                abs(analytic + 1.0f) > 1e-3f,
                "analytic=$analytic at y=$y matches broken-stub sentinel; IR transform did not fire",
            )
            val relErr = abs(analytic - fd) / (abs(fd) + 1.0e-7f)
            assertTrue(
                relErr < 2e-2f,
                "at y=$y: analytic=$analytic vs fd=$fd → relErr=$relErr exceeds 2e-2",
            )
        }
    }

    @Test
    fun `energy-accumulation primal matches analytic gradient across y values`() {
        // f(y) = Σ_{i=0..9} 4·y  =  10 · 4·y  =  40·y.  f'(y) = 40, independent of y.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { y: Float ->
                    var ke = 0.0f
                    for (i in 0 until 10) {
                        ke = ke + 4.0f * y
                    }
                    ke
                }
                println(g(0.0f))
                println(g(1.5f))
                println(g(-2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("40.0\n40.0\n40.0", result.stdout.trim())
    }

    // ========================================================================
    // Stage D.1b — sqrt-bearing brachistochrone primals (§0.4.38).
    //
    // D.1a's "proof-of-pipeline" tests above use ops (ADD / MUL) that were already
    // synthesisable. D.1b unblocks scalar `Float.sqrt()` end-to-end: user-code `.sqrt()`
    // in a `grad { ... }` lambda now routes through OpKind.SQRT + SqrtRule (§0.4.22) +
    // DxirToIrSynthesis.irSqrt (new, this slice). With sqrt available, the brachistochrone
    // descent-time physics — velocity-under-gravity via `v² = 2·g·h` — is expressible in
    // a scalar primal without the v² proxy that D.1a had to use.
    //
    // Ports are still "one mutable var only" (§0.4.25 constraint); two-var tracking
    // (velocity + accumulated-time) is deferred to a post-B.4b follow-up. This makes
    // the port "physics-faithful" (the iterative energy-conservation update IS the
    // same sqrt-bearing recurrence the paper's kernel uses per-segment) rather than
    // "kernel-faithful" (a full time-summation over per-segment contributions). The
    // paper's kernel sums `Δx · 1/sqrt(2g·h_i)`; our port returns the final velocity
    // sqrt(2g·N·y) after N iterative drops, differentiable through the same sqrt+loop
    // shape. Physically-meaningful gradient either way.
    // ========================================================================

    @Test
    fun `scalar sqrt gradient of y² + 4 matches analytic`() {
        // Smallest sqrt-bearing scalar primal — no loop, no accumulation.  Pins the
        // FIR `UNARY_OP_MAP` → `OpKind.SQRT` → SqrtRule → `DxirToIrSynthesis.irSqrt`
        // chain end-to-end. f(y) = sqrt(y² + 4).  f'(y) = y / sqrt(y² + 4).
        //   at y=3:  f'(3) = 3/sqrt(13) ≈ 0.83205032.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun main() {
                val g = grad { y: Float -> (y * y + 4.0f).sqrt() }
                val out = g(3.0f)
                println(out)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        val expected = 3.0f / kotlin.math.sqrt(13.0).toFloat()
        assertTrue(
            abs(grad - expected) / abs(expected) < 1e-5f,
            "grad=$grad expected=$expected — IR transform did not fire or gradient wrong",
        )
    }

    @Test
    fun `brachistochrone descent — energy accumulation in loop, sqrt at end`() {
        // Bead slides through N=10 equal drops of height y, accumulating kinetic
        // energy via v² = 2·g·h·N.  Final velocity = sqrt(2·g·N·y).  This shape puts
        // sqrt OUTSIDE the loop; the loop body is the pure-ADD back-edge D.1a already
        // covers.  Exercises sqrt in the post-loop return.
        //   f(y) = sqrt(2·g·N·y)  with g=9.81, N=10  ⇒  f(y) = sqrt(196.2·y)
        //   f'(y) = 196.2 / (2·sqrt(196.2·y)) = 98.1 / sqrt(196.2·y)
        //   at y=1:  f'(1) = 98.1 / sqrt(196.2) ≈ 7.0035658
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun main() {
                val g = grad { y: Float ->
                    var ke = 0.0f
                    for (i in 0 until 10) {
                        ke = ke + 2.0f * 9.81f * y
                    }
                    ke.sqrt()
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        val expected = 98.1f / kotlin.math.sqrt(196.2).toFloat()
        assertTrue(
            abs(grad + 1.0f) > 1e-3f,
            "grad=$grad matches broken-stub sentinel; IR transform did not fire",
        )
        assertTrue(
            abs(grad - expected) / abs(expected) < 1e-4f,
            "grad=$grad expected=$expected — compile-path gradient wrong",
        )
    }

    @Test
    fun `brachistochrone descent — iterative sqrt energy update in loop body`() {
        // Faithful-to-paper's-physics port: each iteration integrates the energy-
        // conservation kinematic `v_new² = v_old² + 2·g·y` and takes sqrt to get the
        // velocity at the end of the segment.  With v₀=0, N=10, g=9.81:
        //   v_k² = k·2·g·y  ⇒  v_N = sqrt(N·2·g·y) = sqrt(196.2·y) (identical closed
        //   form to the ke-then-sqrt primal above, but the sqrt fires inside the loop
        //   body so C5 unrolls to a 10-deep sqrt(add(mul(...),...)) chain).
        //
        // The SCT gradient through this chain emits SqrtRule-derived `DIV(upstream,
        // MUL(2, SQRT(x)))` at every level — the synthesis test of record for the
        // new `irSqrt` arm.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun main() {
                val g = grad { y: Float ->
                    var v = 0.0f
                    for (i in 0 until 10) {
                        v = (v * v + 2.0f * 9.81f * y).sqrt()
                    }
                    v
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        val expected = 98.1f / kotlin.math.sqrt(196.2).toFloat()
        assertTrue(
            abs(grad + 1.0f) > 1e-3f,
            "grad=$grad matches broken-stub sentinel; IR transform did not fire",
        )
        // Tolerance wider than the prev test — 10-deep sqrt chain accumulates f32
        // rounding more than a single post-loop sqrt.
        assertTrue(
            abs(grad - expected) / abs(expected) < 5e-3f,
            "grad=$grad expected=$expected — iterative-sqrt gradient wrong",
        )
    }

    @Test
    fun `brachistochrone iterative-sqrt gradient agrees with finite differences`() {
        // Cross-check the iterative-sqrt gradient against central-difference over a
        // sweep of y values. Tolerance 1e-2 — this is a 10-deep recurrence under f32,
        // so FD noise is not negligible (and the gradient grows as y → 0 since the
        // sqrt diverges).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun prim(y: Float): Float {
                var v = 0.0f
                for (i in 0 until 10) {
                    v = (v * v + 2.0f * 9.81f * y).sqrt()
                }
                return v
            }
            fun main() {
                val g = grad { y: Float ->
                    var v = 0.0f
                    for (i in 0 until 10) {
                        v = (v * v + 2.0f * 9.81f * y).sqrt()
                    }
                    v
                }
                val eps = 1.0e-3f
                for (y in listOf(0.25f, 0.5f, 1.0f, 2.0f, 4.0f)) {
                    val analytic = g(y)
                    val fd = (prim(y + eps) - prim(y - eps)) / (2.0f * eps)
                    println("${'$'}y ${'$'}analytic ${'$'}fd")
                }
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val lines = result.stdout.trim().lines()
        assertEquals(5, lines.size, "expected 5 'y analytic fd' lines, got:\n${result.stdout}")
        for (line in lines) {
            val parts = line.split(" ")
            assertEquals(3, parts.size, "expected 'y analytic fd' format; got: $line")
            val y = parts[0].toFloat()
            val analytic = parts[1].toFloat()
            val fd = parts[2].toFloat()
            assertTrue(
                abs(analytic + 1.0f) > 1e-3f,
                "analytic=$analytic at y=$y matches broken-stub sentinel",
            )
            val relErr = abs(analytic - fd) / (abs(fd) + 1.0e-7f)
            assertTrue(
                relErr < 1e-2f,
                "at y=$y: analytic=$analytic vs fd=$fd → relErr=$relErr exceeds 1e-2",
            )
        }
    }

    // ========================================================================
    // Stage D.1c — kernel-faithful Brachistochrone (§0.4.39).
    //
    // Paper's kernel form: total descent time = Σ (segment_time), where segment_time
    // depends on the bead's velocity at the segment's start AND end. Requires tracking
    // TWO loop-carried vars — velocity + accumulated time — which the pre-§0.4.39 FIR
    // lowering rejected (one-mutated-var only). §0.4.39 lifts the single-var limit in
    // FirLambdaToDxirLowering.lowerDesugaredForLoop + widens PhiCalculus C5's
    // `detectSimpleLoop` to unroll WHILEs with any operand arity, tracking every block-
    // arg across iterations so multi-var body reads resolve correctly.
    //
    // The primal below is the paper's formulation with constant per-segment height
    // drop `y` and constant per-segment horizontal span `dx`:
    //   segment k: v_start = v_{k-1}, v_end = sqrt(v_{k-1}² + 2g·y), t_k = 2·dx/(v_start + v_end)
    // Starting v_0 = 0. Total time T(y) = Σ t_k for k=1..N. This is the §6.2 kernel
    // verbatim modulo array indexing (we use constant drop per segment rather than the
    // paper's per-segment y_i lookups — Tlaloc has no array indexing yet).
    // ========================================================================

    @Test
    fun `multi-var for-loop — two loop-carried vars without sqrt`() {
        // Smallest multi-var primal, isolates §0.4.39's FIR multi-var + C5 N-carried
        // extensions from the sqrt / brachistochrone math. Two mutated vars:
        //   var a = x; var b = 0f
        //   iter 1:  b = b + a = 0 + x = x;          a = a + 1 = x + 1
        //   iter 2:  b = b + a = x + (x+1) = 2x+1;   a = a + 1 = x + 2
        //   iter 3:  b = b + a = 2x+1 + (x+2) = 3x+3; a = a + 1 = x + 3
        // Closed form b_3 = 3x + 3.  d/dx(3x+3) = 3.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var a = x
                    var b = 0.0f
                    for (i in 0 until 3) {
                        b = b + a
                        a = a + 1.0f
                    }
                    b
                }
                println(g(7.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("3.0", result.stdout.trim())
    }

    @Test
    fun `brachistochrone kernel — per-segment time summation with velocity state`() {
        // The paper's primal form: accumulate per-segment times t_k = 2·dx/(v_{k-1} + v_k)
        // under energy conservation v_k² = v_{k-1}² + 2g·y. Uses BOTH loop-carried
        // vars (velocity + time accumulator) — the form that was blocked pre-§0.4.39.
        //
        // Constants are dimensionless: 2g = 2, dx = 1 (keeps numbers tidy without
        // affecting the math — setting `2g = 2·9.81` and `dx = 0.1` just scales).
        //
        // Closed form (with 2g=1, dx=1):  v_k = sqrt(k·y), t_k = 2/(v_{k-1} + v_k).
        //   T(y) = Σ_{k=1..N} 2/(sqrt((k-1)y) + sqrt(k·y))
        //   For N=5, y=1: T = 2/(0+1) + 2/(1+√2) + 2/(√2+√3) + 2/(√3+2) + 2/(2+√5)
        //     = 2 + 0.8284 + 0.6370 + 0.5359 + 0.4721 ≈ 4.4734
        //   dT/dy via central-difference (the reference for this kernel).
        //
        // Cross-checked against central finite difference computed in the compiled
        // main program from the same primal.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun prim(y: Float): Float {
                var v = 0.0f
                var t = 0.0f
                for (i in 0 until 5) {
                    val v_new = (v * v + 2.0f * y).sqrt()
                    t = t + 2.0f / (v + v_new)
                    v = v_new
                }
                return t
            }
            fun main() {
                val g = grad { y: Float ->
                    var v = 0.0f
                    var t = 0.0f
                    for (i in 0 until 5) {
                        val v_new = (v * v + 2.0f * y).sqrt()
                        t = t + 2.0f / (v + v_new)
                        v = v_new
                    }
                    t
                }
                val eps = 1.0e-3f
                for (y in listOf(0.5f, 1.0f, 2.0f, 4.0f)) {
                    val analytic = g(y)
                    val fd = (prim(y + eps) - prim(y - eps)) / (2.0f * eps)
                    println("${'$'}y ${'$'}analytic ${'$'}fd")
                }
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 'y analytic fd' lines, got:\n${result.stdout}")
        for (line in lines) {
            val parts = line.split(" ")
            assertEquals(3, parts.size, "expected 'y analytic fd' format; got: $line")
            val y = parts[0].toFloat()
            val analytic = parts[1].toFloat()
            val fd = parts[2].toFloat()
            assertTrue(
                abs(analytic + 1.0f) > 1e-3f,
                "analytic=$analytic at y=$y matches broken-stub sentinel; IR transform did not fire",
            )
            // Gradient should be negative (T decreases as y grows — bead goes faster).
            assertTrue(analytic < 0f, "expected negative dT/dy at y=$y, got analytic=$analytic")
            val relErr = abs(analytic - fd) / (abs(fd) + 1.0e-7f)
            assertTrue(
                relErr < 2e-2f,
                "at y=$y: analytic=$analytic vs fd=$fd → relErr=$relErr exceeds 2e-2",
            )
        }
    }

    @Test
    fun `brachistochrone kernel — first-segment-only hand-computed at N=1`() {
        // Smallest kernel case to make the math legible. With N=1, y=1, 2g=1, dx=1:
        //   v_0 = 0
        //   iter 1:  v_new = sqrt(0 + 2·1) = sqrt(2);  t = 2/(0 + sqrt(2)) = sqrt(2)
        //   T(y) = 2/sqrt(2y) = sqrt(2/y).
        //   dT/dy = -sqrt(2)/(2·y^1.5).  At y=1:  -sqrt(2)/2 ≈ -0.70710677.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun main() {
                val g = grad { y: Float ->
                    var v = 0.0f
                    var t = 0.0f
                    for (i in 0 until 1) {
                        val v_new = (v * v + 2.0f * y).sqrt()
                        t = t + 2.0f / (v + v_new)
                        v = v_new
                    }
                    t
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        val expected = -kotlin.math.sqrt(2.0f) / 2.0f
        assertTrue(
            abs(grad - expected) / abs(expected) < 1e-5f,
            "grad=$grad expected=$expected",
        )
    }

    // ========================================================================
    // Stage D.1d — loop-index `i` binding in body (§0.4.40).
    //
    // The FIR lowering now binds the loop parameter `i` to the counter block-arg;
    // body expressions can use `i` directly (Int arithmetic) or `i.toFloat()` for
    // Float arithmetic. This is the first of four sessions closing the per-segment
    // paper-faithful Brachistochrone: S1 (this) — loop-index; S2 — GATHER substrate;
    // S3 — FIR `arr[i]` lowering; S4 — rank-1 param with per-index gradient output.
    // ========================================================================

    @Test
    fun `linearly-varying segment height brachistochrone uses loop-index in body`() {
        // Per-segment drop h_i = y · (i + 1) — each subsequent segment drops deeper
        // under gravity. Accumulates velocity across 3 drops via v² = Σ 2·g·h_i.
        //
        // Closed form (2g = 2, N = 3):  v² = 2·y·(1+2+3) = 12y  ⇒  v = sqrt(12y)
        //   dv/dy = 12 / (2·sqrt(12y)) = 6/sqrt(12y).
        //   at y=1:  6/sqrt(12) = sqrt(3) ≈ 1.7320508
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sqrt
            fun main() {
                val g = grad { y: Float ->
                    var v = 0.0f
                    for (i in 0 until 3) {
                        val drop = y * (i.toFloat() + 1.0f)
                        v = (v * v + 2.0f * drop).sqrt()
                    }
                    v
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        val expected = 6.0f / kotlin.math.sqrt(12.0f)
        assertTrue(
            abs(grad + 1.0f) > 1e-3f,
            "grad=$grad matches broken-stub sentinel; IR transform did not fire",
        )
        assertTrue(
            abs(grad - expected) / abs(expected) < 1e-4f,
            "grad=$grad expected=$expected — compile-path gradient wrong",
        )
    }

    // ========================================================================
    // Stage D.1e — wall-clock perf measurement (§0.4.44).
    //
    // No hard assertion on absolute ns-per-call (CI can't pin those across
    // hardware), but the test PARSES a structured stdout block and asserts the
    // ratio bounds + sentinel-rejection. The numbers themselves go to test
    // stdout for human review and into the §0.4.44 spec entry as the first
    // recorded baseline.
    //
    // Methodology: 1000 warmup invocations to let HotSpot tier up, then 5000
    // measurement invocations of forward-only primal AND compile-path gradient
    // at N=64 (paper's smallest config). System.nanoTime() bracketed; per-call
    // averaged. Sinks accumulated to defeat dead-code elimination. Inputs are a
    // fixed-seed random vector, computed once outside the timing loop.
    //
    // Limitations vs paper §6.2:
    //   - Single hardware sample (whatever runs ./gradlew test); no devServer/
    //     macBook split.
    //   - No "with vs without coarsening" comparison: our concrete-N for-loop
    //     primals require C5 unroll to compile at all (DxirReverseTransform
    //     rejects bare WHILE), so the paper's exact knob isn't exposed. The
    //     measurement here is "Tlaloc compile-path absolute timings + AD
    //     overhead ratio (gradient / forward)", which is the closest fair
    //     summary we can produce in-process.
    //   - No PyTorch / JAX cross-framework comparison (out-of-process; plan §9.2
    //     DoD item, deferred).
    // ========================================================================

    @Test
    fun `paper-faithful brachistochrone at N=64 — measured timings`() {
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
            fun primal(y: DTensor<Rank1<Sym>, F32>): Float {
                var v = 0.0f
                var t = 0.0f
                for (i in 0 until 64) {
                    val v_new = (v * v + y[i]).sqrt()
                    t = t + 2.0f / (v + v_new)
                    v = v_new
                }
                return t
            }
            fun main() {
                val g = grad { y: DTensor<Rank1<Sym>, F32> ->
                    var v = 0.0f
                    var t = 0.0f
                    for (i in 0 until 64) {
                        val v_new = (v * v + y[i]).sqrt()
                        t = t + 2.0f / (v + v_new)
                        v = v_new
                    }
                    t
                }
                // Deterministic input: y[k] = 0.1·(k+1) — strictly increasing,
                // strictly positive, avoids the v=0 division at later segments.
                val arr = FloatArray(64) { (it + 1).toFloat() * 0.1f }
                val input = Tensors.f32Vector<Sym>(arr)

                // Warmup: 1000 iterations interleaved so HotSpot specialises both call sites.
                var sinkF = 0.0f
                var sinkG = 0.0f
                for (i in 0 until 1000) {
                    sinkF += primal(input)
                    sinkG += g(input).hostF32()[0]
                }

                // Measurement: 5000 iterations each.
                val iters = 5000
                val tFwdStart = System.nanoTime()
                for (i in 0 until iters) sinkF += primal(input)
                val tFwdNs = (System.nanoTime() - tFwdStart) / iters

                val tGradStart = System.nanoTime()
                for (i in 0 until iters) sinkG += g(input).hostF32()[0]
                val tGradNs = (System.nanoTime() - tGradStart) / iters

                // Compute one gradient outside the timing loop for sentinel-reject:
                // if the IR transform fell back to the broken stub, every slot is -1.0.
                val checkSlot0 = g(input).hostF32()[0]

                println("PERF forward_ns_per_call=${'$'}tFwdNs")
                println("PERF gradient_ns_per_call=${'$'}tGradNs")
                println("PERF ratio_grad_over_fwd=${'$'}{tGradNs.toDouble() / tFwdNs}")
                println("PERF first_grad_slot0=${'$'}checkSlot0")
                println("PERF sink=${'$'}sinkF/${'$'}sinkG")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        val perfLines = result.stdout.lines().filter { it.startsWith("PERF ") }
        assertEquals(5, perfLines.size, "expected 5 PERF lines, got:\n${result.stdout}")

        val metrics = perfLines.associate { line ->
            val (k, v) = line.removePrefix("PERF ").split("=", limit = 2)
            k to v
        }
        val fwdNs = metrics["forward_ns_per_call"]?.toLong() ?: error("missing forward_ns_per_call")
        val gradNs = metrics["gradient_ns_per_call"]?.toLong() ?: error("missing gradient_ns_per_call")
        val ratio = metrics["ratio_grad_over_fwd"]?.toDouble() ?: error("missing ratio_grad_over_fwd")
        val firstSlot = metrics["first_grad_slot0"]?.toFloat() ?: error("missing first_grad_slot0")

        // Sentinel rejection: broken stub returns [-1,-1,-1,-1] every call. A real
        // brachistochrone gradient at slot 0 is negative-but-not-exactly-(-1.0).
        assertTrue(
            abs(firstSlot + 1.0f) > 1e-3f,
            "first_grad_slot0=$firstSlot matches broken-stub sentinel; IR transform did not fire",
        )

        // Sanity bounds — gradient call is at least as expensive as forward (modulo
        // CSE wins; could be slightly cheaper if the JIT folds away unused forward
        // computation, but normally AD overhead is positive). 0.5x is a generous
        // floor that won't false-fail under JIT noise.
        assertTrue(
            ratio > 0.5,
            "gradient/forward ratio=$ratio implausibly low; suggests measurement bug",
        )
        // Upper bound: gradient should be within 200x forward for this tiny kernel.
        // Paper claims ~1-2x for Brachistochrone differentiation; if we're 200x off
        // there's a real perf bug worth investigating.
        assertTrue(
            ratio < 200.0,
            "gradient/forward ratio=$ratio exceeds 200x — investigate perf regression",
        )

        // Log to test stdout for human review (gets captured by Gradle's test report).
        println("[Brachistochrone N=64 perf] forward=${fwdNs}ns/call gradient=${gradNs}ns/call ratio=${ratio}")
    }

    // ========================================================================
    // Stage D.1d-S4 — paper-faithful per-segment Brachistochrone (§0.4.43).
    //
    // The finish line for the 4-session per-segment path. Primal: T(y) = Σ_{k=1..N}
    // 2·dx / (v_{k-1} + v_k) with v_k² = v_{k-1}² + 2g·y[k-1], v_0 = 0. Constants
    // dimensionless (2g = 1, dx = 1) for tractable hand-computed references.
    //
    // This form captures the paper's §6.2 kernel verbatim: per-segment height `y[i]`
    // via array indexing, velocity-and-time accumulation via two loop-carried vars,
    // sqrt-based energy conservation. Gradient is a rank-1 vector giving per-segment
    // dT/dy[k] — exactly what the paper's outer optimiser needs to find the
    // brachistochrone curve.
    // ========================================================================

    @Test
    fun `paper-faithful brachistochrone at N=1 matches analytic gradient`() {
        // N=1 is the hand-computable anchor. With y = [h]:
        //   v_1 = sqrt(h), t = 2 / (0 + sqrt(h)) = 2·h^(-1/2)
        //   dT/dh = -h^(-3/2). At h=1 → -1.0.
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
            fun main() {
                val g = grad { y: DTensor<Rank1<Sym>, F32> ->
                    var v = 0.0f
                    var t = 0.0f
                    for (i in 0 until 1) {
                        val v_new = (v * v + y[i]).sqrt()
                        t = t + 2.0f / (v + v_new)
                        v = v_new
                    }
                    t
                }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(1.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val grad = result.stdout.trim().toFloat()
        assertTrue(
            abs(grad + 1.0f) < 1e-3f,
            "expected -1.0 (= dT/dh at h=1), got $grad",
        )
    }

    @Test
    fun `energy-accumulation primal matches hand-computed reference at N=2`() {
        // Smallest non-trivial N, hand-verifiable:
        //   ke_0 = 0
        //   ke_1 = ke_0 + 4·y = 4·y
        //   ke_2 = ke_1 + 4·y = 8·y
        // d(ke_2)/dy = 8.  Pins the pure-ADD back-edge path against a value derived
        // without appeal to any closed-form corollary.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { y: Float ->
                    var ke = 0.0f
                    for (i in 0 until 2) {
                        ke = ke + 4.0f * y
                    }
                    ke
                }
                println(g(1.5f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("8.0", result.stdout.trim())
    }

    // --------- Harness (file-local; mirrors TlalocPluginDiagnosticTest's helpers) ---------

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    )

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-brachistochrone-run").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }

            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean =
                    collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
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
            val capturedOut = PrintStream(baos, /* autoFlush = */ true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
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
        private val AUTOGRAD_STUB_BROKEN = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()

        /** Mirrors TlalocPluginDiagnosticTest's stub of the same name (§0.4.42). */
        private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f, -1.0f, -1.0f)), intArrayOf(4), F32) }
        """.trimIndent()
    }
}
