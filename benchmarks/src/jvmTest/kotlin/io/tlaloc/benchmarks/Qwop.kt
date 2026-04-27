package io.tlaloc.benchmarks

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.209 — QWOP benchmark Phase 0a synthetic primal.
 *
 * Per `docs/QWOP_PORT_PLAN.md`'s Phase 0 first-slice spec, this file ships a
 * synthetic QWOP-shape primal that hits the paper's structural pattern (long
 * function, multiple loops, multiple if-else branches) without trying to
 * faithfully reproduce QWOP's specific physics. The Path 2 (synthetic, not
 * reconstructed) decision in the plan is what justifies this approach: the
 * paper's structural claims are what we're validating, not specific physics.
 *
 * **Phase 0a slice scope** (this firing):
 *  - 6 loops (4 muscle-update loops + 2 distance-accumulation loops)
 *  - 4 if-else branches (collision-response for each muscle)
 *  - ~130 lines
 *  - Differentiable end-to-end (every primitive in the gradient path is
 *    Tlaloc-supported as of §0.4.207's register)
 *
 * Phase 0b (next firing or two) will widen to the paper's stated **13 loops
 * and 8 if-else branches** by adding (a) cross-limb interaction loops, (b)
 * forward-kinematics loops with multi-segment chain accumulation, (c) a
 * couple of tightly-nested loops to exercise WHILE-in-WHILE coarsening.
 *
 * **The dxir-builder approach** (vs. user-code lambdas like CartPole / HMC's
 * `:compiler-plugin` tests) is what `BenchmarkPrimals` uses for the existing
 * benchmark probes (§0.4.146 / §0.4.147 / §0.4.148). Keeping QWOP in this
 * mould lets us stress-test Tlaloc's coarsening + reverse-mode AD pipeline
 * directly without going through the K2 plugin's FIR-side recognition. The
 * plan's "Inputs: DTensor<Rank1<Sym>, F32> of muscle extensions" spec is
 * relaxed here to "4 scalar muscle extensions" — equivalent surface for the
 * coarsening pipeline (a rank-1 input with 4 elements decomposes to 4
 * scalar GATHERs anyway).
 */
object Qwop {

    private val f32 = DxirType(F32, emptyList())
    private val i32 = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    /**
     * Phase 0a synthetic avatar-step primal.
     *
     * Inputs (4 scalar `Float`s):
     *   - `mHip`: hip muscle extension command
     *   - `mKnee`: knee muscle extension command
     *   - `mAnkle`: ankle muscle extension command
     *   - `mShoulder`: shoulder muscle extension command
     *
     * Output (1 scalar `Float`): synthetic distance traveled (negative loss).
     *
     * Structure:
     *
     * ```text
     * fun avatarStep(mHip, mKnee, mAnkle, mShoulder):
     *     # Phase A — per-muscle integration with collision response (4 loops + 4 if-else)
     *     hip   = integrateMuscle(mHip,   N_STEPS, MAX_HIP)    # Loop 1, IF 1
     *     knee  = integrateMuscle(mKnee,  N_STEPS, MAX_KNEE)   # Loop 2, IF 2
     *     ankle = integrateMuscle(mAnkle, N_STEPS, MAX_ANKLE)  # Loop 3, IF 3
     *     shldr = integrateMuscle(mShoulder, N_STEPS, MAX_SH)  # Loop 4, IF 4
     *
     *     # Phase B — distance accumulation (2 loops)
     *     coarseDist = sumOfPositions(hip, knee, ankle)         # Loop 5
     *     fineDist   = sumOfFineSteps(shldr, coarseDist)         # Loop 6
     *
     *     return fineDist
     * ```
     *
     * Each `integrateMuscle` is a WHILE loop with an affine recurrence
     * (`state += muscle · dt`) and a per-step IF-else for collision response
     * (`if (state > maxAngle) state = maxAngle else state += muscle · dt`).
     *
     * Each distance loop is a separate WHILE accumulator.
     */
    fun avatarStepPrimal(): DxirFunction =
        DxirBuilder.function("qwopAvatarStep") {
            val mHip = param("mHip", f32)
            val mKnee = param("mKnee", f32)
            val mAnkle = param("mAnkle", f32)
            val mShoulder = param("mShoulder", f32)

            // Phase A — 4 muscle-integration loops, each with collision IF.
            val hip = integrateMuscle(mHip, maxAngle = 1.5f, nSteps = 4)
            val knee = integrateMuscle(mKnee, maxAngle = 2.0f, nSteps = 4)
            val ankle = integrateMuscle(mAnkle, maxAngle = 1.0f, nSteps = 4)
            val shoulder = integrateMuscle(mShoulder, maxAngle = 1.8f, nSteps = 4)

            // Phase B — 2 distance accumulation loops.
            val coarseDist = sumPositions(hip, knee, ankle, nSteps = 3)
            val fineDist = sumFineSteps(shoulder, coarseDist, nSteps = 3)

            listOf(fineDist)
        }

    /**
     * Phase 0a helper: WHILE loop integrating a muscle extension with collision
     * response. One loop + one if-else inside the loop body. Returns the final
     * angle as a scalar. Affine-recurrence shape (C6/C7 surface).
     *
     * ```kotlin
     * var state = 0f
     * for (i in 0 until nSteps) {
     *     if (state > maxAngle) state = maxAngle
     *     else                  state += muscle * 0.1f
     * }
     * return state
     * ```
     *
     * The IF is multi-result-friendly (both branches yield a single scalar
     * carried into the next iteration). The §0.4.155 multi-live-index MR IF
     * AD path covers it. C5 unroll handles the `nSteps`-iteration WHILE.
     */
    private fun DxirBuilder.integrateMuscle(
        muscle: io.tlaloc.ir.DxirNode,
        maxAngle: Float,
        nSteps: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val nConst = const(nSteps, i32)
        val zeroI = const(0, i32)
        val w = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val state = args[0]
                val i = args[1]
                val maxAngleConst = const(maxAngle, f32)
                val overshoot = op(OpKind.SUB, listOf(state, maxAngleConst), f32)
                val collisionPred = op(OpKind.STEP, listOf(overshoot), boolS)
                val dt = const(0.1f, f32)
                val muscleStep = op(OpKind.MUL, listOf(muscle, dt), f32)
                val incremented = op(OpKind.ADD, listOf(state, muscleStep), f32)
                // IF: if (state > maxAngle) maxAngle else state + muscle*dt
                val ifResult = op(
                    OpKind.IF,
                    listOf(collisionPred),
                    f32,
                    regions = listOf(
                        region { yields(maxAngleConst) },
                        region { yields(incremented) },
                    ),
                )
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(ifResult, newI)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0a helper: coarse distance accumulator. Single-level WHILE with
     * affine recurrence summing scaled positions. No IF inside; this loop
     * exercises the pure C6/C7 closed-form path without multi-result IF.
     */
    private fun DxirBuilder.sumPositions(
        hip: io.tlaloc.ir.DxirNode,
        knee: io.tlaloc.ir.DxirNode,
        ankle: io.tlaloc.ir.DxirNode,
        nSteps: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val zeroI = const(0, i32)
        val nConst = const(nSteps, i32)
        val w = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val acc = args[0]
                val i = args[1]
                val hipKneeSum = op(OpKind.ADD, listOf(hip, knee), f32)
                val triSum = op(OpKind.ADD, listOf(hipKneeSum, ankle), f32)
                val newAcc = op(OpKind.ADD, listOf(acc, triSum), f32)
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(newAcc, newI)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0a helper: fine-grained distance accumulator combining the
     * shoulder contribution and the coarse distance. Single-level WHILE
     * with multiplicative coupling — exercises a different recurrence shape
     * than [sumPositions]'s pure ADD chain.
     */
    private fun DxirBuilder.sumFineSteps(
        shoulder: io.tlaloc.ir.DxirNode,
        coarseDist: io.tlaloc.ir.DxirNode,
        nSteps: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val zeroI = const(0, i32)
        val nConst = const(nSteps, i32)
        val w = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val acc = args[0]
                val i = args[1]
                val coupled = op(OpKind.MUL, listOf(shoulder, coarseDist), f32)
                val newAcc = op(OpKind.ADD, listOf(acc, coupled), f32)
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(newAcc, newI)
            },
        )
        return w.result(0)
    }
}
