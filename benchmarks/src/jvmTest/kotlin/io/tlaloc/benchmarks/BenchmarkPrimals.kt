package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.148 — shared primal builders for the `:benchmarks` module's probes.
 * Extracted from §0.4.145–§0.4.147's three benchmarks (EndToEndAdBenchmark,
 * CoarseningThroughputBenchmark, SctThroughputBenchmark) which had the same
 * `iterateNTimes` helper duplicated three times. With consolidation, each
 * probe just calls [iterateNTimes] from this object.
 *
 * Helpers stay narrowly scoped: only primal builders and structural counters
 * common to MULTIPLE benchmarks belong here. Benchmark-specific construction
 * (e.g., a tracer-surface fixture, a shape that exercises a niche code path)
 * stays in the benchmark's own file. The "stay structurally close" principle
 * documented in §0.4.145 / §0.4.146 / §0.4.147 still applies — sharing widens
 * only when the duplication is observable across three or more callers.
 */
object BenchmarkPrimals {

    val f32 = DxirType(F32, emptyList())
    val i32 = DxirType(I32, emptyList())
    val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    /**
     * Build the canonical `iterateNTimes` primal: `f(x) = x · 2^n` via an
     * n-iteration WHILE loop whose body doubles the carried `x` each iter
     * and increments a counter. After [io.tlaloc.ir.passes.PhiCalculus.apply]
     * unrolls C5, the body has exactly n MUL ops and zero WHILEs (pinned by
     * §0.4.146's [CoarseningThroughputBenchmark]); after
     * [io.tlaloc.ir.passes.DxirReverseTransform.apply], the gradient evaluates
     * to `2^n` at any x (pinned by §0.4.147's [SctThroughputBenchmark]).
     */
    fun iterateNTimes(n: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("iterate$n") {
            val x = param("x", f32)
            val nConst = const(n, i32)
            val zero = const(0, i32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32)
                    val one = const(1, i32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }

    /** Count top-level body ops of a given [kind] in [fn]. */
    fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /**
     * §0.4.156 — multi-branch IF primal that exercises the §0.4.155 multi-live-index
     * MR IF AD path. Shape:
     * ```
     * f(x) = let r = if (x > 0) {x*x, 2*x} else {-x, x}
     *        in  r.result(0) + r.result(1)
     * ```
     * For `x > 0`: `r = (x², 2x)`, `sum = x² + 2x`, `df/dx = 2x + 2`.
     * For `x ≤ 0`: `r = (-x, x)`, `sum = -x + x = 0`, `df/dx = -1 + 1 = 0`.
     *
     * Both `r.result(0)` and `r.result(1)` are referenced downstream — pre-§0.4.155
     * this shape was hard-rejected by `findIfLiveResultIndex.singleOrNull()`. With
     * Phase 5b's per-index gradAccum dispatch, the primal flows correctly through
     * SCT and the gradient closes in expected per-branch closed forms.
     */
    fun multiBranchIfPrimal(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("multiBranchIf") {
            val x = param("x", f32)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val xSquared = op(OpKind.MUL, listOf(x, x), f32)
            val two = const(2f, f32)
            val twoX = op(OpKind.MUL, listOf(two, x), f32)
            val negX = op(OpKind.NEG, listOf(x), f32)
            val ifOp = opMulti(
                OpKind.IF,
                listOf(pred),
                listOf(f32, f32),
                regions = listOf(
                    region { yields(xSquared, twoX) },
                    region { yields(negX, x) },
                ),
            )
            val sum = op(OpKind.ADD, listOf(ifOp.result(0), ifOp.result(1)), f32)
            listOf(sum)
        }

    /**
     * §0.4.223 — BGDHyperOpt outer-loop primal, mirrors `PhiCalculus.bgdHyperOptPrimal`
     * in [PhiCalculusBgdHyperOptTest][io.tlaloc.ir.passes]. Duplicated here because
     * `:benchmarks` cannot reach back into `:ir`'s test source set; the duplication is
     * intentional and ~30 lines. Future consolidation: lift to a `:ir`/`:core`-side
     * commonTest helper or expose the primal builder publicly from `:ir`.
     *
     * The signature is `(r, Sxy, Sx2, M) → w_final` with concrete `K` baked in.
     * Mirrors the OOPSLA 2021 paper's Fig. 6a structure (lines 3-11) with the
     * inner-for-loop d-computation pre-collapsed to scalar sums.
     *
     * The recurrence is `w_{k+1} = a · w_k + b` where:
     *   - `a = 1 + 2r·Sx2/M`
     *   - `b = -2r·Sxy/M`
     *
     * **Coarsening behaviour** (without a [io.tlaloc.ir.passes.SymbolicEngine]):
     *   - C5 unrolls the constant-trip-count outer WHILE → flat ADD/MUL chain.
     *   - With a SymjaEngine attached (`:ir`-test-internal), C6 produces the
     *     closed-form `w_K = b · (a^K − 1)/(a − 1)` (O(1) ops vs C5's O(K) chain).
     *   - The harness path uses the no-engine variant — closer to what the K2
     *     plugin emits today.
     *
     * **Closed-form Kotlin reference** is provided as [bgdHyperOptOuterLoopReference]
     * for FD validation in tests.
     */
    fun bgdHyperOptOuterLoopPrimal(K: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("bgdOuterLoop") {
            val r = param("r", f32)
            val Sxy = param("Sxy", f32)
            val Sx2 = param("Sx2", f32)
            val M = param("M", f32)
            val wInit = const(0f, f32)
            val kBound = const(K.toFloat(), f32)
            val kZero = const(0f, f32)
            val w = whileOp(
                inits = listOf(wInit, kZero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(kBound, args[1]), f32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val carriedW = args[0]
                    val counterArg = args[1]
                    val one = const(1f, f32)
                    val two = const(2f, f32)
                    val twoR = op(OpKind.MUL, listOf(two, r), f32)
                    val twoRSx2 = op(OpKind.MUL, listOf(twoR, Sx2), f32)
                    val twoRSx2OverM = op(OpKind.DIV, listOf(twoRSx2, M), f32)
                    val aExpr = op(OpKind.ADD, listOf(one, twoRSx2OverM), f32)
                    val twoRSxy = op(OpKind.MUL, listOf(twoR, Sxy), f32)
                    val twoRSxyOverM = op(OpKind.DIV, listOf(twoRSxy, M), f32)
                    val bExpr = op(OpKind.NEG, listOf(twoRSxyOverM), f32)
                    val aw = op(OpKind.MUL, listOf(aExpr, carriedW), f32)
                    val newW = op(OpKind.ADD, listOf(aw, bExpr), f32)
                    val counterIncr = const(1f, f32)
                    val newK = op(OpKind.ADD, listOf(counterArg, counterIncr), f32)
                    yields(newW, newK)
                },
            )
            listOf(w.result(0))
        }

    /**
     * §0.4.223 — Closed-form Kotlin reference for the BGDHyperOpt outer loop.
     * Used by the harness baseline test for FD-validated gradient pins.
     */
    fun bgdHyperOptOuterLoopReference(r: Float, Sxy: Float, Sx2: Float, M: Float, K: Int): Float {
        val a = 1f + 2f * r * Sx2 / M
        val b = -2f * r * Sxy / M
        var w = 0f
        for (k in 0 until K) w = a * w + b
        return w
    }

    /**
     * §0.4.224 — HookeanSpring **scalar 1D oscillator** primal. Mirrors the
     * OOPSLA 2021 paper's HookeanSpring "N=10 chain — measured timings"
     * shape: a constant-trip-count temporal simulation with **two coupled
     * state variables** (position + velocity).
     *
     * The recurrence is symplectic (semi-implicit) Euler integration of an
     * undamped harmonic oscillator with stiffness `kSpring` and unit mass:
     *
     * ```kotlin
     * var pos = pInit
     * var vel = vInit
     * for (i in 0 until N) {
     *     val force = -kSpring * pos
     *     vel = vel + dt * force
     *     pos = pos + dt * vel
     * }
     * return pos
     * ```
     *
     * **Why scalar instead of multi-vertex chain**: the paper uses an N-vertex
     * chain (rank-1 tensor primitives + GATHER/SCATTER_ADD). Lifting the
     * existing K2-plugin port at `:compiler-plugin/src/test/.../HookeanSpringTest.kt`
     * to `:benchmarks` requires rank-1 tensor support on the dxir-builder
     * path, which is not available without the K2-plugin's `compileAndRun`
     * infrastructure. The scalar 1D oscillator hits the same structural
     * axis (constant-trip-count WHILE with 2 coupled state vars) without
     * needing the rank-1 plumbing.
     *
     * **Coarsening behaviour**: the N=10 trip count is constant, so C5
     * unrolls this WHILE into a 10-deep recurrence chain. The gradient
     * through the unrolled recurrence is exercised by reverse-mode AD.
     *
     * **Fixed parameters**: `dt = 0.1` (large enough to drift visibly from
     * exact SHO over N=10 steps, small enough to remain numerically stable),
     * `mass = 1.0` (folded into kSpring's effective value). The user
     * supplies the three free parameters: `pInit`, `vInit`, `kSpring`.
     */
    fun hookeanSpringPrimal(N: Int = 10, dt: Float = 0.1f): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("hookeanSpringScalar") {
            val pInit = param("pInit", f32)
            val vInit = param("vInit", f32)
            val kSpring = param("kSpring", f32)
            val nBound = const(N.toFloat(), f32)
            val zeroI = const(0f, f32)
            val dtConst = const(dt, f32)
            val w = whileOp(
                inits = listOf(pInit, vInit, zeroI),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[2]), f32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val pos = args[0]
                    val vel = args[1]
                    val counter = args[2]
                    // force = -kSpring * pos
                    val negKpos = op(OpKind.MUL, listOf(kSpring, pos), f32)
                    val force = op(OpKind.NEG, listOf(negKpos), f32)
                    // vel = vel + dt * force
                    val dtForce = op(OpKind.MUL, listOf(dtConst, force), f32)
                    val newVel = op(OpKind.ADD, listOf(vel, dtForce), f32)
                    // pos = pos + dt * newVel  (semi-implicit / symplectic Euler)
                    val dtVel = op(OpKind.MUL, listOf(dtConst, newVel), f32)
                    val newPos = op(OpKind.ADD, listOf(pos, dtVel), f32)
                    // counter += 1
                    val one = const(1f, f32)
                    val newCounter = op(OpKind.ADD, listOf(counter, one), f32)
                    yields(newPos, newVel, newCounter)
                },
            )
            // Return final position (carried slot 0).
            listOf(w.result(0))
        }

    /**
     * §0.4.224 — Kotlin reference mirroring [hookeanSpringPrimal]'s recurrence.
     * Used for FD-validated gradient pins in the harness test.
     */
    fun hookeanSpringReference(pInit: Float, vInit: Float, kSpring: Float, N: Int = 10, dt: Float = 0.1f): Float {
        var pos = pInit
        var vel = vInit
        for (i in 0 until N) {
            val force = -kSpring * pos
            vel += dt * force
            pos += dt * vel
        }
        return pos
    }

    /**
     * §0.4.225 — Brachistochrone **compound-velocity** primal. Mirrors the
     * `:compiler-plugin/src/test/.../BrachistochroneTest.kt`'s "compound-velocity
     * primal" test (`grad { y -> ... v = v + v * y over N iters ... }`).
     *
     * The recurrence is `v_{i+1} = v_i + v_i * y = v_i * (1 + y)`, so after
     * N iterations: `v_N = (1 + y)^N`. Single scalar input.
     *
     * **Why this primal**: smallest paper-aligned scalar primal in the harness
     * (single input, simple recurrence, closed-form gradient `N · (1 + y)^(N-1)`).
     * Pairs well with BGDHyperOpt (4 inputs) and HookeanSpring (3 inputs) to
     * give the harness coverage across input cardinality 1/3/4. The full
     * Brachistochrone benchmark in the OOPSLA paper involves a sqrt-bearing
     * energy descent integral; the compound-velocity is the simplest sub-
     * primal that exercises the same WHILE-coarsening axis.
     *
     * **Coarsening behaviour**: constant trip count → C5 unrolls into an
     * N-deep MUL chain.
     */
    fun brachistochroneCompoundVelocityPrimal(N: Int = 5): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("brachistochroneCompoundVelocity") {
            val y = param("y", f32)
            val one = const(1f, f32)
            val zero = const(0f, f32)
            val nBound = const(N.toFloat(), f32)
            val w = whileOp(
                inits = listOf(one, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val v = args[0]
                    val counter = args[1]
                    // v = v + v * y
                    val vTimesY = op(OpKind.MUL, listOf(v, y), f32)
                    val newV = op(OpKind.ADD, listOf(v, vTimesY), f32)
                    val oneI = const(1f, f32)
                    val newCounter = op(OpKind.ADD, listOf(counter, oneI), f32)
                    yields(newV, newCounter)
                },
            )
            listOf(w.result(0))
        }

    /**
     * §0.4.225 — Closed-form Kotlin reference for [brachistochroneCompoundVelocityPrimal].
     * Mirrors the recurrence step-by-step; useful for FD validation AND as a
     * cross-check against the closed-form `(1 + y)^N`.
     */
    fun brachistochroneCompoundVelocityReference(y: Float, N: Int = 5): Float {
        var v = 1f
        for (i in 0 until N) v = v + v * y
        return v
    }

    /**
     * §0.4.226 — HMC logistic-regression log-posterior primal. Mirrors the
     * `:compiler-plugin/src/test/.../HmcLogisticRegressionLoopTest.kt`'s
     * straight-line variant (Phase 2's loop is just a 4-iter unroll; we ship
     * the unrolled form because dxir-builder doesn't have an ergonomic GATHER
     * primitive for indexing into a constant data table by counter).
     *
     * The dataset (n=4, d=2) is baked into the dxir as constants:
     *   X = [[1.0, 0.5], [0.5, 1.0], [-0.5, 1.5], [1.5, -0.5]]
     *   y = [1, 0, 1, 0]
     *
     * Two scalar inputs: β = [b0, b1]. Computes the negative log-posterior:
     *
     * ```kotlin
     * sum1 = sum2 = 0
     * for i in 0..3:
     *     xb = X[i][0] * b0 + X[i][1] * b1
     *     sum1 += (y[i] - 1) * xb
     *     sum2 += log(1 + exp(-xb))
     * term3 = (b0² + b1²) / 2000   // weak Gaussian prior
     * return sum1 - sum2 - term3
     * ```
     *
     * **Why this primal**: HMC is the OOPSLA paper's hardest control-flow
     * benchmark. Even the "Phase 2 loop form" sub-primal exercises the EXP
     * and LOG ops — the **first** harness inhabitant to use these. Adds the
     * fourth paper benchmark to the harness.
     *
     * **Coarsening behaviour**: straight-line dxir, no WHILE → no C5 unroll
     * needed. PhiCalculus.apply is essentially identity here. Reverse-mode
     * AD exercises the chain rule through EXP/LOG composed with arithmetic.
     *
     * **Future widening**: HMC Phase 3 adds a numerical-stability mask
     * (`if xb > 0: term2 = log(1 + exp(-xb)) else: term2 = -xb + log(1 + exp(xb))`)
     * — exercises the IF gradient routing axis on top of EXP/LOG. Not on the
     * harness's critical path; defer until needed.
     */
    fun hmcLogisticRegressionPrimal(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("hmcLogisticRegression") {
            val b0 = param("b0", f32)
            val b1 = param("b1", f32)

            // Dataset (concretely baked in).
            val xData = listOf(
                listOf(1.0f, 0.5f),
                listOf(0.5f, 1.0f),
                listOf(-0.5f, 1.5f),
                listOf(1.5f, -0.5f),
            )
            val yData = listOf(1.0f, 0.0f, 1.0f, 0.0f)

            var sum1: io.tlaloc.ir.DxirNode = const(0f, f32)
            var sum2: io.tlaloc.ir.DxirNode = const(0f, f32)
            val one = const(1f, f32)

            for (i in 0 until 4) {
                val xi0 = const(xData[i][0], f32)
                val xi1 = const(xData[i][1], f32)
                val yi = const(yData[i], f32)

                // xb = xi0 * b0 + xi1 * b1
                val xi0b0 = op(OpKind.MUL, listOf(xi0, b0), f32)
                val xi1b1 = op(OpKind.MUL, listOf(xi1, b1), f32)
                val xb = op(OpKind.ADD, listOf(xi0b0, xi1b1), f32)

                // sum1 += (yi - 1) * xb
                val yiMinusOne = op(OpKind.SUB, listOf(yi, one), f32)
                val term1 = op(OpKind.MUL, listOf(yiMinusOne, xb), f32)
                sum1 = op(OpKind.ADD, listOf(sum1, term1), f32)

                // sum2 += log(1 + exp(-xb))
                val negXb = op(OpKind.NEG, listOf(xb), f32)
                val expNegXb = op(OpKind.EXP, listOf(negXb), f32)
                val onePlus = op(OpKind.ADD, listOf(one, expNegXb), f32)
                val logOnePlus = op(OpKind.LOG, listOf(onePlus), f32)
                sum2 = op(OpKind.ADD, listOf(sum2, logOnePlus), f32)
            }

            // term3 = (b0² + b1²) / 2000
            val b0Sq = op(OpKind.MUL, listOf(b0, b0), f32)
            val b1Sq = op(OpKind.MUL, listOf(b1, b1), f32)
            val sumSq = op(OpKind.ADD, listOf(b0Sq, b1Sq), f32)
            val twoThou = const(2000f, f32)
            val term3 = op(OpKind.DIV, listOf(sumSq, twoThou), f32)

            // result = sum1 - sum2 - term3
            val sum1MinusSum2 = op(OpKind.SUB, listOf(sum1, sum2), f32)
            val result = op(OpKind.SUB, listOf(sum1MinusSum2, term3), f32)
            listOf(result)
        }

    /**
     * §0.4.226 — Kotlin reference mirroring [hmcLogisticRegressionPrimal].
     * Same hardcoded dataset; useful for FD-validated gradient pins.
     */
    fun hmcLogisticRegressionReference(b0: Float, b1: Float): Float {
        val X = arrayOf(
            floatArrayOf(1.0f, 0.5f),
            floatArrayOf(0.5f, 1.0f),
            floatArrayOf(-0.5f, 1.5f),
            floatArrayOf(1.5f, -0.5f),
        )
        val y = floatArrayOf(1.0f, 0.0f, 1.0f, 0.0f)
        var sum1 = 0f
        var sum2 = 0f
        for (i in 0 until 4) {
            val xb = X[i][0] * b0 + X[i][1] * b1
            sum1 += (y[i] - 1f) * xb
            sum2 += kotlin.math.ln(1f + kotlin.math.exp(-xb))
        }
        val term3 = (b0 * b0 + b1 * b1) / 2000f
        return sum1 - sum2 - term3
    }

    /**
     * §0.4.227 — CartPole Phase 1 primal: one-timestep cart-pole reward
     * with a clip-at-zero IF. Mirrors the K2-plugin port at
     * `:compiler-plugin/src/test/.../CartPolePhase1Test.kt` ported as scalar
     * dxir (5 separate scalar inputs instead of a packed rank-1 tensor).
     *
     * Five scalar inputs: `at` (action), `x0` (cart pos), `x1` (cart vel),
     * `x2` (pole angle), `x3` (pole angular vel).
     *
     * ```kotlin
     * rt = 9.0 * at + 0.045 * x3² * sin(x2)
     * cosX2 = cos(x2)
     * qt = (9.8 * sin(x2) - rt * cosX2) / (0.65 - 0.4 * cosX2²)
     * pt = rt - 0.045 * qt * cosX2          // computed but unused
     * xn0 = x0 + 0.02 * x1
     * xn2 = x2 + 0.02 * x3
     * maxArg = (2.4 - |xn0|) * (0.21 - |xn2|)
     * clipped = if (maxArg > 0) maxArg else 0
     * term = 0.5 - clipped
     * return term * term
     * ```
     *
     * **Why this primal**: CartPole is the OOPSLA paper's RL benchmark; even
     * Phase 1 (one-step reward) exercises **sin**, **cos**, **abs**, and **IF**
     * — four ops that NO prior harness inhabitant uses. Adds the **fifth paper
     * benchmark** to the harness — full M9 paper-benchmark coverage (BGDHyperOpt
     * + HookeanSpring + Brachistochrone + HMC + CartPole + QWOP-as-avatar-step).
     *
     * **Note about `pt`**: it's computed but the return value doesn't depend on
     * it. So the gradient w.r.t. `at` (which only feeds into `rt → qt → pt`)
     * should be **zero**. This makes a sharp discriminator: any DCE bug or
     * misrouted chain-rule that incorrectly accumulated gradient through the
     * unused branch would surface as a non-zero `df/dat`.
     */
    fun cartPolePhase1Primal(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("cartPolePhase1") {
            val at = param("at", f32)
            val x0 = param("x0", f32)
            val x1 = param("x1", f32)
            val x2 = param("x2", f32)
            val x3 = param("x3", f32)

            val nineConst = const(9.0f, f32)
            val zero045 = const(0.045f, f32)
            val nineEight = const(9.8f, f32)
            val zero65 = const(0.65f, f32)
            val zero4 = const(0.4f, f32)
            val zero02 = const(0.02f, f32)
            val twoPoint4 = const(2.4f, f32)
            val zero21 = const(0.21f, f32)
            val zero5 = const(0.5f, f32)

            val sinX2 = op(OpKind.SIN, listOf(x2), f32)
            val cosX2 = op(OpKind.COS, listOf(x2), f32)

            // rt = 9 * at + 0.045 * x3 * x3 * sin(x2)
            val nineAt = op(OpKind.MUL, listOf(nineConst, at), f32)
            val x3Sq = op(OpKind.MUL, listOf(x3, x3), f32)
            val zero045X3Sq = op(OpKind.MUL, listOf(zero045, x3Sq), f32)
            val x3SqSinX2 = op(OpKind.MUL, listOf(zero045X3Sq, sinX2), f32)
            val rt = op(OpKind.ADD, listOf(nineAt, x3SqSinX2), f32)

            // qt = (9.8 * sin(x2) - rt * cosX2) / (0.65 - 0.4 * cosX2 * cosX2)
            val nineEightSinX2 = op(OpKind.MUL, listOf(nineEight, sinX2), f32)
            val rtCosX2 = op(OpKind.MUL, listOf(rt, cosX2), f32)
            val qtNum = op(OpKind.SUB, listOf(nineEightSinX2, rtCosX2), f32)
            val cosX2Sq = op(OpKind.MUL, listOf(cosX2, cosX2), f32)
            val zero4CosX2Sq = op(OpKind.MUL, listOf(zero4, cosX2Sq), f32)
            val qtDen = op(OpKind.SUB, listOf(zero65, zero4CosX2Sq), f32)
            val qt = op(OpKind.DIV, listOf(qtNum, qtDen), f32)

            // pt = rt - 0.045 * qt * cosX2 (computed but unused — DCE-test piece)
            val zero045Qt = op(OpKind.MUL, listOf(zero045, qt), f32)
            val zero045QtCosX2 = op(OpKind.MUL, listOf(zero045Qt, cosX2), f32)
            @Suppress("UNUSED_VARIABLE")
            val pt = op(OpKind.SUB, listOf(rt, zero045QtCosX2), f32)

            // xn0 = x0 + 0.02 * x1
            val zero02X1 = op(OpKind.MUL, listOf(zero02, x1), f32)
            val xn0 = op(OpKind.ADD, listOf(x0, zero02X1), f32)

            // xn2 = x2 + 0.02 * x3
            val zero02X3 = op(OpKind.MUL, listOf(zero02, x3), f32)
            val xn2 = op(OpKind.ADD, listOf(x2, zero02X3), f32)

            // maxArg = (2.4 - |xn0|) * (0.21 - |xn2|)
            val absXn0 = op(OpKind.ABS, listOf(xn0), f32)
            val absXn2 = op(OpKind.ABS, listOf(xn2), f32)
            val term1 = op(OpKind.SUB, listOf(twoPoint4, absXn0), f32)
            val term2 = op(OpKind.SUB, listOf(zero21, absXn2), f32)
            val maxArg = op(OpKind.MUL, listOf(term1, term2), f32)

            // clipped = if (maxArg > 0) maxArg else 0
            val zeroF = const(0f, f32)
            val pred = op(OpKind.STEP, listOf(maxArg), boolS)
            val clipped = op(
                OpKind.IF,
                listOf(pred),
                f32,
                regions = listOf(
                    region { yields(maxArg) },
                    region { yields(zeroF) },
                ),
            )

            // term = 0.5 - clipped; return term * term
            val term = op(OpKind.SUB, listOf(zero5, clipped), f32)
            val result = op(OpKind.MUL, listOf(term, term), f32)
            listOf(result)
        }

    /**
     * §0.4.227 — Kotlin reference mirroring [cartPolePhase1Primal]. Useful for
     * FD-validated gradient pins.
     */
    fun cartPolePhase1Reference(at: Float, x0: Float, x1: Float, x2: Float, x3: Float): Float {
        val sinX2 = kotlin.math.sin(x2)
        val cosX2 = kotlin.math.cos(x2)
        val rt = 9.0f * at + 0.045f * x3 * x3 * sinX2
        val qt = (9.8f * sinX2 - rt * cosX2) / (0.65f - 0.4f * cosX2 * cosX2)
        @Suppress("UNUSED_VARIABLE")
        val pt = rt - 0.045f * qt * cosX2   // computed but unused
        val xn0 = x0 + 0.02f * x1
        val xn2 = x2 + 0.02f * x3
        val maxArg = (2.4f - kotlin.math.abs(xn0)) * (0.21f - kotlin.math.abs(xn2))
        val clipped = if (maxArg > 0.0f) maxArg else 0.0f
        val term = 0.5f - clipped
        return term * term
    }
}
