# QWOP Benchmark Port — Plan

**Status:** Planning — source-access dependency named; structural Tlaloc surface largely shipped.

**Ship state** (initialised 2026-04-27 at §0.4.208; Phase 0a + 0b shipped §0.4.209 + §0.4.210):

| Phase | Plan estimate | Status |
|---|---|---|
| Phase 0a — synthetic Qwop.kt scaffold (≤ paper's structural shape) | 1 firing | **CLOSED (§0.4.209)** |
| Phase 0b — widen to 13 loops + 8 if-else (paper's structural shape) | 1-2 firings | **CLOSED (§0.4.210)** |
| Phase 1 — straight-line port of one body part's update step | 1-2 firings | not started |
| Phase 2 — single loop body coarsened (one of QWOP's 13 loops) | 2-3 firings | not started |
| Phase 3 — full QWOP function (13 loops + many if-else, 2 SOIs per paper) | 3-5 firings | not started |
| **QWOP-specific total** | **8-13 firings** | **2 firings shipped** |

## Phase 0a structural choices (§0.4.209)

Per Path 2 (synthetic, not reconstructed) recommendation, §0.4.209 ships
[`benchmarks/src/jvmTest/kotlin/io/tlaloc/benchmarks/Qwop.kt`](../benchmarks/src/jvmTest/kotlin/io/tlaloc/benchmarks/Qwop.kt) as
a `DxirBuilder` primal builder (matching the existing `BenchmarkPrimals`
convention rather than the user-code-lambda spec from this plan's earlier
draft — the dxir-builder approach is more direct for stress-testing
Tlaloc's coarsening + reverse-mode AD pipeline without going through the
K2 plugin's FIR-side recognition).

**Ship state of Phase 0a's `Qwop.avatarStepPrimal`**:

- **Inputs**: 4 scalar `Float` muscle extensions (hip, knee, ankle, shoulder).
  The plan's "DTensor<Rank1<Sym>, F32> of muscle extensions" was relaxed to
  4 scalars — equivalent surface for the coarsening pipeline (a rank-1
  input would decompose to 4 scalar GATHERs anyway).
- **Output**: 1 scalar `Float` distance traveled.
- **Lines**: ~135 (under the plan's 200-250 target; Phase 0b widens).
- **Loops**: 6 — 4 muscle-integration WHILEs (Phase A) + 2 distance
  accumulation WHILEs (Phase B).
- **If-else branches**: 4 — one per muscle's collision-response inside the
  Phase A WHILE bodies.
- **Differentiable end-to-end**: yes — every primitive (ADD/SUB/MUL/STEP/IF
  + WHILE w/ affine recurrence) is shipped per §0.4.207's register.

**Why Phase 0a is smaller than the paper's stated 13 loops + 8 if-else**:
the paper claims 225 lines / 13 loops / 8 if-else for QWOP's full function,
and the plan's first-slice spec aligned with that. In practice, hand-writing
a 225-line dxir-builder primal in one firing is costly without a lot of
copy-paste; Phase 0a ships a structurally-faithful subset (each loop
exercises C5 unroll + C6/C7 affine recurrence; each if-else is a multi-
result-friendly IF that §0.4.155's machinery covers) that lets follow-on
firings widen incrementally without rewriting the foundation.

**Phase 0b plan**: widen to the paper's full structure by adding:
- **+3 loops via cross-limb interaction**: a per-limb WHILE inside an outer
  per-frame WHILE (WHILE-in-WHILE — exercises §0.4.176's surface).
- **+2 loops for forward kinematics**: chain accumulation along
  hip→knee→ankle and shoulder→elbow→wrist.
- **+2 loops for energy / damping**: per-joint angular velocity update with
  damping coefficient.
- **+4 if-else branches**: ground contact (foot vs ground), torque limits
  (max torque per joint), gait-phase dispatch (stance vs swing), energy
  thresholding.

Total post-0b: 13 loops, 8 if-else. Matches the paper's structural claim.

**Phase 1+ deliverables remain as in the original plan** — straight-line
gradient test, single-loop coarsened test, full FD-validated forward + grad.

---

## Why QWOP is the structural-stress-test of the paper benchmark suite

**Target benchmark:** Avatar motion optimization (paper §7.1, line 247-360 of
`docs/papers/coarsening-autodiff.txt`). Trains a virtual stick figure to run
as far as possible by selecting a schedule of muscle extensions. Three
configurations vary the masses of body parts; speedup numbers in the paper
are 1.17-1.51× (Table 3, configurations 1+2 / 1+1).

**The paper's characterisation of QWOP**:

> *"QWOP is an avatar motion optimization program. It trains a virtual stick
> figure to run as far as possible by providing a schedule for how much each
> muscle should be extended... the special aspect about this program is that
> its core part is a 225-line function with 13 loops and many if-else
> statements. After loop unrolling, the function becomes 1117-line long.
> Coarsening can successfully deal with the function, getting two SOIs..."*

So: **one large function, 13 loops, many if-else, 2 SOIs after coarsening,
1117-line unrolled**. The paper's other benchmarks each exercise one or two
of Tlaloc's structural pieces; QWOP exercises ALL of them in a single
function.

## The source-access question (Phase 0)

**The paper's appendix does not include QWOP source code.** Unlike HMC
(where the paper gives the closed-form `U(β)` in Eq. 1) and CartPole (where
the paper gives the NN forward + physics step), QWOP is described
qualitatively. The benchmark code is presumably available to the authors
(`docs/papers/coarsening-autodiff.txt:1346` references the paper's
artifacts) but not in the version shipped with this repo.

**Two paths forward**:

1. **Reconstruct from a public reference.** The original QWOP browser game
   (Bennett Foddy, 2008) has many open-source ports. Pick one that exposes
   the avatar's joint-update math; back out a Kotlin transcription that has
   13 loops and many if-else statements; differentiate the loss (distance
   the avatar covers). **Estimated: 3-5 firings just for the reconstruction;
   uncertain whether the result matches the paper's specific function.**

2. **Synthetic QWOP-shape function.** Write a Kotlin function with the
   structural pattern (13 loops + many if-else statements + ~225-line
   length) that simulates a generic articulated-body optimizer. The output
   doesn't have to match QWOP's specific physics; it has to match the
   coarsening surface (control-flow density, recurrence patterns,
   per-iteration symbolic differentiation challenges). **Estimated: 5-7
   firings; structurally faithful, semantically synthetic.**

**Recommended: Path 2.** The paper's structural claims (13 loops, many
if-else, 2 SOIs) are what we're validating, NOT the specific physics. A
synthetic QWOP-shape function lets us:
- Construct a Tlaloc port of known structure.
- Verify each piece of Tlaloc's coarsening machinery (C5 unroll, C6/C7
  affine recurrence, C8 full symbolic, C9 mask, multi-result IF AD,
  WHILE-in-IF / IF-in-WHILE / WHILE-in-WHILE) handles the cumulative
  combination.
- Compare Tlaloc speedups to the paper's 1.17-1.51× — even if the absolute
  runtime differs, the speedup pattern should hold.

Phase 0's deliverable: a Kotlin file under `:benchmarks/src/jvmMain` that
contains the synthetic QWOP-shape function, plus a planning addendum to
this doc capturing the specific structural choices made.

## Tlaloc gap analysis (as of §0.4.208)

Surveying QWOP's likely surface against Tlaloc's shipped stack:

| Sub-expression | Op kind | Status | Notes |
|---|---|---|---|
| Per-step joint angle update `θ_{t+1} = θ_t + ω_t · dt` | scalar `ADD` + `MUL` | ✅ Shipped | §0.4.4 era + §0.4.42 |
| Trig for joint angles (`sin`, `cos`) | `SIN` / `COS` | ✅ Shipped | §0.4.166 |
| `abs(velocity)` for damping | `ABS` | ✅ Shipped | §0.4.167 |
| `if (jointAngle > maxAngle) ...` collision response | scalar `IF` | ✅ Shipped | §0.4.140 (IF-in-IF), §0.4.155 (multi-result IF AD) |
| `for (joint in 0..nJoints)` muscle update loop | C5 unroll OR C6/C7 affine recurrence | ✅ Shipped | §0.4.140 (C5 IF), §0.4.161 (C5 WHILE), affine recurrence engine |
| Nested loop (per-joint update inside per-frame outer loop) | WHILE-in-WHILE | ✅ Shipped | §0.4.176 |
| `if (jointType == HIP) {...} else if (jointType == KNEE) {...}` chain | multi-branch IF | ✅ Shipped | §0.4.141 (multi-branch IF chain), §0.4.156 |
| Distance-traveled reduction (sum of per-frame x-displacement) | `SUM` | ✅ Shipped | rank-1 SUM via §0.4.41 era |
| 13 loops + many if-else combined in one function | structural cumulative | 🟡 Untested | Each piece individually shipped; 13×N combination is the stress test |
| 2 SOIs after coarsening | structural | 🟡 Likely | Tlaloc's coarsening has handled multi-SOI bodies in BGDHyperOpt et al.; QWOP's specific 2-SOI shape needs verification |

**No new primitives needed.** Every structural piece in the table is shipped.

The genuine risk is **cumulative complexity**: 13 loops in one function, each
producing a SCT-level symbolic gradient body, then PhiCalculus's coarsening
passes have to merge / DCE / lift across the whole function. Performance
characteristics of `applyC5Pass` / `applyC6Pass` / `applyC7Pass` /
`applyC8Pass` / `applyC9Pass` on a 225-line function with 13 nested loop
sites are untested. The paper's 1117-line unrolled form is the soft upper
bound on what Tlaloc must process per gradient invocation.

**Possible runtime emergent issues**:
- DxirReverseTransform's clone-and-rewrite per gradient body may scale poorly with body size.
- Top-level CSE (§0.4.48) is O(n²) in dxir node count for the worst case.
- Cache-key generation may produce excessively long keys (no impact on correctness, possible space concern).

These are **deferred** until Phase 2/3 actually exercises the surface.

## Three-phase migration

Mirroring HMC and CartPole's incremental approach:

### Phase 0 — source acquisition / reconstruction (this doc + 1-2 firings)

**Deliverable:** A `:benchmarks` Kotlin file containing a synthetic
QWOP-shape function (~225 lines, 13 loops, many if-else). Either constructed
from open-source QWOP ports (Path 1) or written from scratch as a structurally
representative articulated-body optimizer (Path 2 — recommended).

The exact structural choices need to be captured in a Phase-0 amendment to
this plan once the function is written.

### Phase 1 — straight-line port of one body part's update step (1-2 firings)

**Deliverable:** Single-iteration test of the joint-update math for one
body part (e.g., the hip). Hand-computed gradient pinned to within 1e-3 f32
tolerance, mirroring HMC's Phase 1 pattern.

### Phase 2 — single loop body coarsened (2-3 firings)

**Deliverable:** One of QWOP's 13 loops ported as a `WHILE` form, gradient
through the loop body verified against finite-differencing on small
configurations.

### Phase 3 — full QWOP function (3-5 firings)

**Deliverable:** Complete 225-line function port, gradient through the full
forward pass FD-validated, coarsening passes produce ≤ 2 SOIs (per paper),
end-to-end speedup measured against the un-coarsened version.

Once Phase 3 lands, **all six paper benchmarks are ported** — the M9 exit
criterion's primary structural gate is closed.

## Out of scope

- The QWOP browser game's rendering / input handling — pure UI, not
  differentiable.
- Tuning Tlaloc's coarsening for QWOP-specific speedup wins beyond the
  paper's 1.17-1.51×. The benchmark port's purpose is structural
  verification + speedup match; engineering for QWOP's speedup is
  out-of-scope until M9.
- The original Bennett Foddy QWOP physics. The paper's QWOP benchmark is
  inspired-by, not faithful-to, the original game. We follow the same
  inspired-by license.
- Comparing Tlaloc's QWOP gradient correctness to a different AD framework's
  output. FD validation suffices for the Phase 1+2+3 acceptance criteria.

## Dependencies on platform infrastructure

QWOP is the **only** remaining paper benchmark; once it ships, Phase 1 of
the /loop priority ladder closes EXCEPT for the head-to-head harness Phase
2 (gated on user-side Python toolchain). The QWOP port is therefore the
**rate-limiting structural piece** for closing Phase 1 of the M9 exit
criterion. Paths to closure:

1. **All-Tlaloc**: ship Phase 0–3, then formal "Phase 1 closed" entry pins
   the six head-to-head numbers from Tlaloc's side only (paper's numbers in
   place of competing-framework numbers).
2. **Toolchain-gated**: ship Phase 0–3 + wait for user to enable Python
   toolchain → run head-to-head Phase 2 → formal close.

Path 1 is faster but yields only Tlaloc's own coarsening speedups. Path 2
yields the full paper's 6-benchmark comparison table. Both are valid M9
closures depending on the user's stance on cross-framework references.

## Phase 0 first-slice — concrete next firing

The immediate next firing should:

1. Pick **Path 2** (synthetic QWOP-shape function) per the recommendation
   above.
2. Write `benchmarks/src/jvmMain/kotlin/io/tlaloc/benchmarks/Qwop.kt`
   containing the synthetic function. Specifications:
   - 225 ± 25 lines.
   - 13 ± 1 loops (mix of single-level + 2-3 nested).
   - 8 ± 2 if-else branches.
   - Differentiable end-to-end (no SIGN or non-differentiable primitives in
     the gradient path).
   - Inputs: a `DTensor<Rank1<Sym>, F32>` of muscle extensions.
   - Output: a Float (distance / negative loss).
3. Add a Phase-0 amendment to this plan capturing the specific structural
   choices made (which loops are affine recurrences, which are general
   WHILE, which IF chains are multi-branch, etc.).
4. Run `./gradlew build` to confirm the synthetic function compiles cleanly
   without engaging Tlaloc's gradient transform yet.

Phase 0 closes when `Qwop.kt` is written and compiles. Phase 1 begins with
the first per-body-part gradient test.

## Why this plan is structured differently from HMC's / CartPole's

- **HMC** had a closed-form `U(β)` from the paper's Eq. 1, so the plan went
  directly from the formula to phase-by-phase port. Three phases, 4 firings
  actual.
- **CartPole** had the NN forward `a = sign(tanh(...))` + a physics step
  loop from the paper's Figure 2(c) and Eq. 3.1. The plan started with
  scalar primitives (sin / cos / abs) before tackling the NN. Three phases,
  25 firings actual (plan estimate didn't budget Phase 0c-rectangular or
  Phase 3 sub-decomposition).
- **QWOP** has no closed-form description in the paper — only structural
  metrics (13 loops, 225 lines, 2 SOIs). Phase 0 (source acquisition /
  reconstruction) is therefore necessary BEFORE the per-phase port begins.
  The plan adds this as an explicit Phase 0 rather than treating it as
  pre-work.
