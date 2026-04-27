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
    /**
     * §0.4.211 — Phase 1 first-slice primal: single-muscle integration step
     * exposed as a top-level [DxirFunction] for gradient-correctness pinning.
     *
     * Wraps the same [integrateMuscle] helper that the full [avatarStepPrimal]
     * uses (Phase A's per-muscle WHILE + collision IF). Reused as a standalone
     * primal here so QWOP Phase 1's first slice can FD- / hand-verify the
     * gradient on one body part before the full 13-loop avatar-step is
     * gradient-tested in Phase 2+.
     *
     * Forward semantics (with default `maxAngle=1.5`, `nSteps=4`):
     * ```kotlin
     * var state = 0f
     * for (i in 0 until 4) {
     *     if (state > 1.5f) state = 1.5f
     *     else              state = state + muscle * 0.1f
     * }
     * return state
     * ```
     *
     * Closed-form derivative (when no iteration triggers clamping):
     *   `state = nSteps * dt * muscle = 0.4 * muscle`, so `df/dmuscle = 0.4`.
     *
     * For inputs where clamping triggers, the gradient becomes 0 at the
     * iteration where `state > maxAngle` first holds — sub-gradient through
     * the IF's `then`-branch (which yields the constant `maxAngle`, killing
     * gradient flow through `state`).
     */
    fun hipUpdatePrimal(maxAngle: Float = 1.5f, nSteps: Int = 4): DxirFunction =
        DxirBuilder.function("qwopHipUpdate") {
            val mHip = param("mHip", f32)
            val finalState = integrateMuscle(mHip, maxAngle, nSteps)
            listOf(finalState)
        }

    /**
     * §0.4.213 — QWOP Phase 2 first slice: pure-WHILE no-IF single-loop primal.
     *
     * Wraps the same [sumPositions] helper that Phase B of [avatarStepPrimal]
     * uses (the coarse distance accumulator). Standalone exposure here lets
     * QWOP Phase 2's first slice pin the **multi-input** coarsened-gradient
     * surface — a contrast to §0.4.211/§0.4.212's [hipUpdatePrimal] which
     * exercised single-input WHILE+IF coarsening.
     *
     * Forward semantics (with default `nSteps = 3`):
     * ```kotlin
     * var acc = 0f
     * for (i in 0 until 3) acc += hip + knee + ankle
     * return acc   // = 3 * (hip + knee + ankle)
     * ```
     *
     * Closed-form gradients are constant (independent of input values):
     *   - `df/dhip = nSteps`
     *   - `df/dknee = nSteps`
     *   - `df/dankle = nSteps`
     *
     * Coarsening expectation: the WHILE has constant trip count → C5 unrolls
     * it → 0 top-level WHILEs in the coarsened body, just an ADD chain.
     */
    fun sumPositionsPrimal(nSteps: Int = 3): DxirFunction =
        DxirBuilder.function("qwopSumPositions") {
            val hip = param("hip", f32)
            val knee = param("knee", f32)
            val ankle = param("ankle", f32)
            val acc = sumPositions(hip, knee, ankle, nSteps)
            listOf(acc)
        }

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

            // Phase C — cross-limb coupling: 3 loops (WHILE-in-WHILE + standalone),
            // 0 if-else. Tests §0.4.176's WHILE-in-WHILE coarsening surface.
            val coupling = crossLimbCoupling(hip, knee, shoulder, nFrames = 3, nLimbs = 3)
            val friction = frictionAccum(coupling, nSteps = 3)

            // Phase D — forward kinematics chain: 2 loops + 2 if-else (ground contact
            // for legs; shoulder torque limit for arms).
            val legChain = forwardKinematicsLeg(hip, knee, ankle, nSegs = 3)
            val armChain = forwardKinematicsArm(shoulder, friction, nSegs = 3)

            // Phase E — energy / damping: 2 loops + 2 if-else (max-torque per joint;
            // energy threshold gating).
            val torque = energyTorquePerJoint(legChain, armChain, nSteps = 3)
            val energy = energyAccumulator(torque, fineDist, nSteps = 3)

            // Final: combine all stages into one scalar loss-like output.
            val combined1 = op(OpKind.ADD, listOf(fineDist, friction), f32)
            val combined2 = op(OpKind.ADD, listOf(combined1, legChain), f32)
            val combined3 = op(OpKind.ADD, listOf(combined2, armChain), f32)
            val combined4 = op(OpKind.ADD, listOf(combined3, energy), f32)

            listOf(combined4)
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

    /**
     * Phase 0b helper: cross-limb coupling via WHILE-in-WHILE. Outer loop
     * over time frames; inner loop over limb pairs accumulates pairwise
     * MUL coupling. **2 nested loops total** (1 outer + 1 inner; the inner
     * is one logical loop running [nLimbs] times). Tests §0.4.176's
     * WHILE-in-WHILE coarsening + reverse-mode AD surface.
     *
     * ```kotlin
     * var acc = 0f
     * for (frame in 0 until nFrames) {
     *     for (j in 0 until nLimbs) {
     *         acc += hip * knee + knee * shoulder // (mock pairwise coupling)
     *     }
     * }
     * ```
     */
    private fun DxirBuilder.crossLimbCoupling(
        hip: io.tlaloc.ir.DxirNode,
        knee: io.tlaloc.ir.DxirNode,
        shoulder: io.tlaloc.ir.DxirNode,
        nFrames: Int,
        nLimbs: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val zeroI = const(0, i32)
        val nFramesConst = const(nFrames, i32)
        val nLimbsConst = const(nLimbs, i32)
        val outer = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nFramesConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val acc = args[0]
                val frame = args[1]
                // Inner loop: accumulate pairwise coupling over [nLimbs] iterations.
                val zeroInner = const(0f, f32)
                val zeroIInner = const(0, i32)
                val inner = whileOp(
                    inits = listOf(zeroInner, zeroIInner),
                    cond = { innerArgs ->
                        val innerDiff = op(OpKind.SUB, listOf(nLimbsConst, innerArgs[1]), i32)
                        val innerPred = op(OpKind.STEP, listOf(innerDiff), boolS)
                        yields(innerPred)
                    },
                    body = { innerArgs ->
                        val innerAcc = innerArgs[0]
                        val j = innerArgs[1]
                        val pair1 = op(OpKind.MUL, listOf(hip, knee), f32)
                        val pair2 = op(OpKind.MUL, listOf(knee, shoulder), f32)
                        val pairSum = op(OpKind.ADD, listOf(pair1, pair2), f32)
                        val newInnerAcc = op(OpKind.ADD, listOf(innerAcc, pairSum), f32)
                        val one = const(1, i32)
                        val newJ = op(OpKind.ADD, listOf(j, one), i32)
                        yields(newInnerAcc, newJ)
                    },
                )
                val newAcc = op(OpKind.ADD, listOf(acc, inner.result(0)), f32)
                val one = const(1, i32)
                val newFrame = op(OpKind.ADD, listOf(frame, one), i32)
                yields(newAcc, newFrame)
            },
        )
        return outer.result(0)
    }

    /**
     * Phase 0b helper: standalone friction accumulator. Single-level WHILE
     * with squared-input recurrence — different shape than [sumPositions]'s
     * pure ADD chain or [crossLimbCoupling]'s pairwise MUL pattern.
     *
     * ```kotlin
     * var acc = 0f
     * for (i in 0 until nSteps) acc += coupling * coupling
     * ```
     */
    private fun DxirBuilder.frictionAccum(
        coupling: io.tlaloc.ir.DxirNode,
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
                val sq = op(OpKind.MUL, listOf(coupling, coupling), f32)
                val newAcc = op(OpKind.ADD, listOf(acc, sq), f32)
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(newAcc, newI)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0b helper: forward kinematics chain for the leg, with a
     * ground-contact IF inside the loop body. **1 loop + 1 if-else.**
     *
     * ```kotlin
     * var pos = 0f
     * for (seg in 0 until nSegs) {
     *     val tip = pos + hip + knee + ankle
     *     // ground contact: foot can't go below zero
     *     pos = if (tip < 0f) 0f else tip
     * }
     * ```
     */
    private fun DxirBuilder.forwardKinematicsLeg(
        hip: io.tlaloc.ir.DxirNode,
        knee: io.tlaloc.ir.DxirNode,
        ankle: io.tlaloc.ir.DxirNode,
        nSegs: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val zeroI = const(0, i32)
        val nConst = const(nSegs, i32)
        val w = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val pos = args[0]
                val seg = args[1]
                val sum1 = op(OpKind.ADD, listOf(pos, hip), f32)
                val sum2 = op(OpKind.ADD, listOf(sum1, knee), f32)
                val tip = op(OpKind.ADD, listOf(sum2, ankle), f32)
                // Ground contact: tip < 0 → ground (0); else use computed tip.
                // STEP returns 1 if (-tip) > 0, i.e. tip < 0. Use STEP(-tip) as the predicate.
                val negTip = op(OpKind.NEG, listOf(tip), f32)
                val belowGroundPred = op(OpKind.STEP, listOf(negTip), boolS)
                val zeroF = const(0f, f32)
                val newPos = op(
                    OpKind.IF,
                    listOf(belowGroundPred),
                    f32,
                    regions = listOf(
                        region { yields(zeroF) },
                        region { yields(tip) },
                    ),
                )
                val one = const(1, i32)
                val newSeg = op(OpKind.ADD, listOf(seg, one), i32)
                yields(newPos, newSeg)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0b helper: forward kinematics chain for the arm, with a
     * shoulder-torque-limit IF inside the loop body. **1 loop + 1 if-else.**
     *
     * ```kotlin
     * var swing = 0f
     * for (seg in 0 until nSegs) {
     *     val acc = swing + shoulder + friction
     *     // torque limit: max swing magnitude is 4
     *     swing = if (acc > 4f) 4f else acc
     * }
     * ```
     */
    private fun DxirBuilder.forwardKinematicsArm(
        shoulder: io.tlaloc.ir.DxirNode,
        friction: io.tlaloc.ir.DxirNode,
        nSegs: Int,
    ): io.tlaloc.ir.DxirNode {
        val zero = const(0f, f32)
        val zeroI = const(0, i32)
        val nConst = const(nSegs, i32)
        val w = whileOp(
            inits = listOf(zero, zeroI),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val swing = args[0]
                val seg = args[1]
                val sum1 = op(OpKind.ADD, listOf(swing, shoulder), f32)
                val acc = op(OpKind.ADD, listOf(sum1, friction), f32)
                val maxSwing = const(4f, f32)
                val excess = op(OpKind.SUB, listOf(acc, maxSwing), f32)
                val overLimitPred = op(OpKind.STEP, listOf(excess), boolS)
                val newSwing = op(
                    OpKind.IF,
                    listOf(overLimitPred),
                    f32,
                    regions = listOf(
                        region { yields(maxSwing) },
                        region { yields(acc) },
                    ),
                )
                val one = const(1, i32)
                val newSeg = op(OpKind.ADD, listOf(seg, one), i32)
                yields(newSwing, newSeg)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0b helper: per-joint torque update with max-torque IF.
     * **1 loop + 1 if-else.**
     *
     * ```kotlin
     * var torque = 0f
     * for (i in 0 until nSteps) {
     *     val raw = torque + leg + arm
     *     // max-torque clamp
     *     torque = if (raw > 8f) 8f else raw
     * }
     * ```
     */
    private fun DxirBuilder.energyTorquePerJoint(
        leg: io.tlaloc.ir.DxirNode,
        arm: io.tlaloc.ir.DxirNode,
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
                val torque = args[0]
                val i = args[1]
                val sum1 = op(OpKind.ADD, listOf(torque, leg), f32)
                val raw = op(OpKind.ADD, listOf(sum1, arm), f32)
                val maxTorque = const(8f, f32)
                val excess = op(OpKind.SUB, listOf(raw, maxTorque), f32)
                val overPred = op(OpKind.STEP, listOf(excess), boolS)
                val newTorque = op(
                    OpKind.IF,
                    listOf(overPred),
                    f32,
                    regions = listOf(
                        region { yields(maxTorque) },
                        region { yields(raw) },
                    ),
                )
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(newTorque, newI)
            },
        )
        return w.result(0)
    }

    /**
     * Phase 0b helper: energy accumulator with energy-threshold IF.
     * **1 loop + 1 if-else.**
     *
     * ```kotlin
     * var energy = 0f
     * for (i in 0 until nSteps) {
     *     val raw = energy + torque * dist
     *     // energy threshold: cap at 100
     *     energy = if (raw > 100f) 100f else raw
     * }
     * ```
     */
    private fun DxirBuilder.energyAccumulator(
        torque: io.tlaloc.ir.DxirNode,
        dist: io.tlaloc.ir.DxirNode,
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
                val energy = args[0]
                val i = args[1]
                val product = op(OpKind.MUL, listOf(torque, dist), f32)
                val raw = op(OpKind.ADD, listOf(energy, product), f32)
                val maxEnergy = const(100f, f32)
                val excess = op(OpKind.SUB, listOf(raw, maxEnergy), f32)
                val overPred = op(OpKind.STEP, listOf(excess), boolS)
                val newEnergy = op(
                    OpKind.IF,
                    listOf(overPred),
                    f32,
                    regions = listOf(
                        region { yields(maxEnergy) },
                        region { yields(raw) },
                    ),
                )
                val one = const(1, i32)
                val newI = op(OpKind.ADD, listOf(i, one), i32)
                yields(newEnergy, newI)
            },
        )
        return w.result(0)
    }
}
