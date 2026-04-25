# HMC Benchmark Port — Plan

**Status:** Planning artifact (§0.4.157). Implementation has not started.

**Target benchmark:** Hamiltonian Monte Carlo for Bayesian logistic regression, per
the OOPSLA 2021 paper §7.2 (in `docs/papers/coarsening-autodiff.txt:1250-1268`).
The paper reports 2.3-3.6× end-to-end speedup from coarsening on this benchmark —
on par with Brachistochrone (4-11× via primal elimination) and BGDHyperOpt
(1.05-1.12× via control-flow coarsening).

**Why HMC is "the hardest control-flow benchmark":** the paper's blurb names HMC's
characteristic feature as "potential value overflow incurred by exponential
computations" requiring numerical-stability masks. The U function combines:
- Two single-level loops (sum of products, regularizer term).
- One two-level nested loop with an `if/else` for numerical stability of
  `log(1 + exp(-Xβ))` when `-Xβ > 80`.
- Vector dot products + matrix-vector multiply (tensor ops Tlaloc already supports).

A faithful port hits every "control-flow + symbolic AD" path Tlaloc has built up
through Stage B/C/D, plus the multi-result IF AD landed in §0.4.155.

## Reference: U(β) for logistic regression

From the paper:

```
                                                  β^T β
U(β) = β^T X^T (y - 1_n) - 1_n^T [log(1 + e^{-Xβ})] - ─────
                                                  2σ_β²
```

Symbolic legend:
- `β` ∈ ℝ^d: logistic-regression parameters (the variables of interest).
- `X` ∈ ℝ^{n×d}: training feature matrix (constant during sampling).
- `y` ∈ {0,1}^n: training labels (constant).
- `σ_β² = 1000`: prior variance (hyperparameter).
- `1_n`: the n-vector of all 1s.

HMC samples β by alternating leapfrog steps that use both `U(β)` and
`grad_U(β) = ∂U/∂β`. The benchmark's hot loop calls `grad_U` more often than `U`.

The paper's three configurations:
1. n = 100 records, d = 1 feature.
2. n = 1000 records, d = 2 features.
3. n = 800 records, d = 3 features.

We'll port the smallest config first, scale later.

## Tlaloc gap analysis (as of §0.4.156)

Each piece of the U formula maps to existing or missing pipeline pieces:

| Sub-expression | Op kind | Status | Notes |
|---|---|---|---|
| `Xβ` | `MATMUL` (rank-2 × rank-1) | ✅ Shipped | §0.4.135 / §0.4.137 / §0.4.138 |
| `-Xβ` (elementwise) | `NEG` | ✅ Shipped | §0.4.4 era |
| `exp(-Xβ)` (elementwise) | `EXP` | ✅ Shipped | §0.4.22 |
| `1 + exp(...)` | broadcast `ADD` | ✅ Shipped | rank-1 broadcast supported |
| `log(...)` (elementwise) | `LOG` | ✅ Shipped | §0.4.22 |
| `if (-Xβ_i > 80)` mask | per-element `IF` | 🟡 Partial | Single-element IF works; **per-element-of-rank-1 mask** is the gap |
| `sum(log(...))` | `SUM` | ✅ Shipped | rank-1 → scalar |
| `β^T X^T (y - 1_n)` | scalar-result via inner products | ✅ Shipped | Decomposes to MATMUL + SUM + MUL |
| `β^T β / (2σ_β²)` | dot product + scalar div | ✅ Shipped | SUM + MUL + DIV |
| Two single-level loops | `WHILE` with affine-recurrence body | ✅ Shipped | C6/C7 affine-recurrence engine |
| Two-level nested loop | `WHILE` inside `WHILE` body region | 🟡 Partial | §0.4.50 supports nested `whileOp`; `applyC5Pass` post-§0.4.152 is region-recursive into IF only, not WHILE |
| Per-element IF inside loop body | `IF` inside `WHILE` body region | 🟡 Partial | §0.4.155's multi-live-index machinery covers IF AD; reverse-mode through WHILE-containing IF is open |

**Two genuine blockers identified**:

1. **Per-element-of-rank-1 mask as `IF`.** The paper expresses
   `log(1 + exp(-Xβ)) ≈ -Xβ` when `-Xβ > 80` per element. In the loop-form port,
   each iteration evaluates one scalar `-Xβ_i` and an `IF` selects between the
   approximation and the full computation. This is a single-result scalar IF — no
   gap. In a tensor-form port (no loop), it would need a `where(mask, a, b)` op,
   which Tlaloc doesn't have at the StableHLO emitter level for arbitrary types.
   **Phase plan: stay with loop-form; defer tensor `where`.**

2. **WHILE inside WHILE.** The paper's nested loop structure has the inner
   loop iterate over features (d) inside an outer loop over records (n) for the
   matrix-vector dot product. C6/C7 (affine recurrence) won't close the inner loop
   when its body references a constant-from-outer-iteration `X[i, j]`; C5 would
   unroll a constant-d inner loop, but only if `applyC5Pass` recurses into WHILE
   region bodies (not just IF region bodies, per §0.4.152). **Phase plan: §0.4.156
   added Phase 4b ("WHILE inside IF inside WHILE") to the recommended-next list;
   widening C5's pre-scan + rewrite to recurse into WHILE bodies is the natural
   prerequisite for HMC's nested loop.**

## Three-phase migration

### Phase 1 — straight-line (no-loop) port at small fixed n, d

**Deliverable:** a compiler-plugin test that ports U(β) at n=4, d=2 using
straight-line tensor ops only (no loops; no IF). Verifies the gradient against
finite-differencing at one β value.

**Source shape:**
```kotlin
val g = grad { beta: DTensor<Rank1<Two>, F32> ->
    // term1 = β^T X^T (y - 1_n)
    val yMinusOne = y - 1.0f                       // rank-1 [4]
    val xTransY = matvec(X.T, yMinusOne)           // rank-1 [2]
    val term1 = sum(beta * xTransY)                // scalar

    // term2 = sum_i log(1 + exp(-X[i,:] · beta))
    val xBeta = matvec(X, beta)                    // rank-1 [4]
    val negXBeta = -xBeta                          // rank-1 [4]
    val expNeg = exp(negXBeta)                     // rank-1 [4]
    val onePlusExp = 1.0f + expNeg                 // rank-1 [4]
    val logTerm = log(onePlusExp)                  // rank-1 [4]
    val term2 = sum(logTerm)                       // scalar

    // term3 = β^T β / (2 · 1000)
    val term3 = sum(beta * beta) / 2000.0f         // scalar

    term1 - term2 - term3
}
```

**Tlaloc support required:** All listed in the gap-analysis table as ✅. No
new infrastructure — Phase 1 is purely a benchmark-port sanity check.

**Acceptance:** numerical agreement with hand-computed gradient at one β
value (within 1e-3 f32 tolerance). One test, one new file
`compiler-plugin/src/test/kotlin/io/tlaloc/plugin/HmcLogisticRegressionTest.kt`.

**Estimated:** 1 firing.

### Phase 2 — loop form at moderate n with affine-recurrence accumulation

**Deliverable:** rewrite Phase 1's straight-line port using `for (i in 0 until n)`
loops with `var t = 0.0f; t += ...` accumulators. Each accumulator should match
C6 (affine-recurrence closed form) so PhiCalculus closes the loop symbolically.

**New surface required:**
- Confirm `for (i in 0 until n)` lowering (§0.4.40) handles the iteration shape.
- Confirm rank-1 indexing (`X[i, j]` via GATHER, §0.4.42 / §0.4.111 / §0.4.114)
  resolves correctly inside loop bodies.
- C6 should match the back-edge `t += beta[j] * X[i,j] * (y[i] - 1)` (linear in
  carried `t`, free in i / j / X / β / y).

**Acceptance:** same gradient as Phase 1 at the same β value. Run-time speedup
NOT measured here (Phase 2 is correctness; Phase 3 / head-to-head harness is perf).

**Estimated:** 2 firings (the C6 match may need a closer look at the exact body
shape; if it doesn't match cleanly, fall back to C5 unroll for small concrete n).

### Phase 3 — nested loop with numerical-stability IF

**Deliverable:** the full paper-faithful U function with the inner-loop IF mask
for `-Xβ > 80`. Land Phase 4b prerequisite (§0.4.156's recommended-next #3) if
the nested-loop coarsening pipeline needs it.

**New surface required:**
- §0.4.156's Phase 4b — extend `applyC5Pass`'s pre-scan + rewrite to walk WHILE
  region bodies (not just IF). This unblocks the inner loop being unrolled (or
  C6'd) inside the outer loop.
- Single-result IF inside a WHILE body — already supported by §0.4.155's MR IF
  AD machinery (single-result is the simplest case of multi-live-index).

**Acceptance:** gradient agrees with Phase 1 + finite-differencing at multiple β
values, including ones that trigger the mask (`-Xβ > 80` for at least one i).

**Estimated:** 3-4 firings. Multi-session by definition; this is the "hardest"
phase per the paper's framing.

## Phase 1 first-slice — concrete next firing

The immediate next firing should land a single test:
`compiler-plugin/src/test/kotlin/io/tlaloc/plugin/HmcLogisticRegressionTest.kt` with
one method exercising the straight-line port at n=4, d=2 with hard-coded X and y.
Hand-computed gradient at one β value pinned to within 1e-3 f32 tolerance.

The test mirrors `HookeanSpringTest`'s structure: K2JVMCompiler in-process
compilation, runtime evaluation of the produced gradient function, comparison
against the analytic answer.

## Out of scope

- HMC's leapfrog integration step (alternating position/momentum updates with
  half-steps + the gradient): pure Kotlin scaffolding around `grad_U`, not a
  differentiable computation itself. Can land as a separate file once `U` /
  `grad_U` are correct, but isn't part of the benchmark-port scope.
- HMC's Metropolis acceptance step: probabilistic, requires PRNG; not
  differentiable. Out of scope.
- The CartPole / QWOP benchmarks (paper's other two remaining ports): each is
  its own multi-session arc. Reuse the three-phase pattern from this plan.
- Head-to-head benchmark harness (Tlaloc vs PyTorch/JAX): per Phase 1 #7 of the
  /loop priority ladder, this is M9 exit-criterion work, gated on all six paper
  benchmarks being ported.
