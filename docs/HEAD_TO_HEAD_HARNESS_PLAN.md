# Head-to-Head Harness — Plan

**Status:** Planning artifact (§0.4.182). Implementation has not started.

**Target:** Phase 1 #7 of the /loop priority ladder — the M9 exit-criterion harness that compares Tlaloc's K2-plugin AD path against PyTorch 2.x `compile` and JAX `jit` on the OOPSLA 2021 paper's six benchmarks (Brachistochrone, HookeanSpring, BGDHyperOpt, HMC, CartPole, QWOP).

**Why head-to-head matters:** the §11.13 strict "Phase 1 closed — coarsening at M9 parity" criterion requires a §0.4 entry that pins six head-to-head numbers vs. the paper's figures. Without the harness, Phase-1's progress is anecdotal (FD-validated correctness on individual ports) rather than measured against the framework alternatives the paper itself benchmarks.

The §0.4 register (§0.4.180) tracks two paths to Phase 1 done:
1. **(a)** Finish the head-to-head harness on the existing 3-5 K2-plugin ports.
2. **(b)** Extend to all 6 ports first, then run the harness.

**This plan scopes (a)** — the head-to-head harness against the currently-shipped ports. (b) is gated on Phase 0c MATMUL + QWOP port work that adds 5+ more firings; (a) gets us a published-comparable number on what's actually working today.

## Ports available today (as of §0.4.181)

Per the §0.4.180 register's "Phase 1 status":

| Benchmark | K2-plugin port | Test file | Paper config | Speedup target (vs torch.compile) |
|---|---|---|---|---|
| Brachistochrone | ✅ Shipped (§0.4.7-8 era) | `compiler-plugin/src/test/.../BrachistochroneTest.kt` | n=64 | Paper: 4-11× |
| HookeanSpring | ✅ Shipped (§0.4.47) | `compiler-plugin/src/test/.../HookeanSpringTest.kt` | N=10 chain | Paper: 1.05-1.12× (control flow); ours scalar |
| HMC | ✅ Shipped (§0.4.176) | `compiler-plugin/src/test/.../HmcLogisticRegression*.kt` (3 files) | n=4, d=2 | Paper: 2.3-3.6× |
| CartPole Phase 1 | ✅ Shipped (§0.4.175) | `compiler-plugin/src/test/.../CartPolePhase1Test.kt` | one time step | Paper Phase 1+2+3 combined: 1.22-4.42× |
| CartPole Phase 2 | ✅ Shipped (§0.4.178) | `compiler-plugin/src/test/.../CartPolePhase2Test.kt` | B=3 time steps | (subset of CartPole) |
| BGDHyperOpt | 🟡 `:benchmarks` only | `benchmarks/src/jvmTest/.../*.kt` | per §0.4.49 | not via K2 plugin yet |
| CartPole Phase 3 | ⬜ Pending | — | NN + outer loop | Gated on Phase 0c |
| QWOP | ⬜ Pending | — | per paper | Multi-session port |

**Three full-stack K2-plugin ports** (Brachistochrone + HookeanSpring + HMC) plus **CartPole Phase 1+2** (a partial CartPole). That's 4 of 6 paper benchmarks with at least partial K2-plugin coverage.

## Methodology

The harness compares **end-to-end gradient evaluation latency** at a fixed input configuration, with N=1000 warmed-up iterations on a single CPU thread. Per the paper's methodology + §0.4.49's BGDHyperOpt timing approach + §0.4.143's HookeanSpring `N=10 chain — measured timings` test pattern.

### Per-benchmark protocol

For each benchmark:
1. **Tlaloc side**: invoke the K2-plugin-synthesised `grad` lambda inside the existing test harness. Wrap in a microbenchmark loop (1000 iterations, JIT-warmed).
2. **PyTorch side**: write a Python reference implementation using `torch.func.grad` + `torch.compile`. Run the same N=1000 timing loop. Output to JSON.
3. **JAX side**: write a Python reference implementation using `jax.grad` + `jax.jit`. Run the same N=1000 timing loop. Output to JSON.
4. **Comparison**: a JVM-side aggregator reads the JSON outputs, asserts numerical agreement (1e-3 absolute / 5e-3 relative tolerance per the per-port FD test patterns), and prints a comparison row.

### Numerical correctness

The harness asserts **f32-tolerance numerical match** between Tlaloc, PyTorch, and JAX gradients on the same input. The existing per-port FD tests already establish FD agreement; the harness extends this to cross-framework agreement.

### Timing

Per-iteration nanosecond timings measured via `System.nanoTime()` on the JVM side and `time.perf_counter_ns()` (or equivalent) on Python. Reports:
- **Median** (more robust than mean for one-shot framework startup costs)
- **Min** (closest to true cost, useful for cache-hot scenarios)
- **p99** (tail behaviour, exposes any GC / JIT recompilation noise)

Output as a CSV: `benchmark,framework,n_iterations,median_ns,min_ns,p99_ns`.

### Confounders

Three sources of measurement noise to control:
1. **JVM JIT warmup** — first ~200 iterations show C1→C2 tier-up. Discard the first 200 by convention.
2. **Python startup + framework import** — PyTorch / JAX import takes seconds. Run timing only AFTER `compile`/`jit` has produced a cached compiled function (warmup loop with 100 iterations).
3. **GC pauses** — JVM-side `-XX:+UseZGC` or `-XX:+UseShenandoahGC` keeps pauses sub-millisecond. Use whatever the test toolchain has by default.

## Multi-phase migration

### Phase 1 — JVM-side scaffolding (single firing)

**Deliverable:** A new module `:harness` (or a `:benchmarks` extension) that:
- Defines a `Benchmark` interface: `name`, `compileTlalocGradient()`, `runTlalocIterations(n: Int): TimingResult`.
- Implements 3-4 `Benchmark` inhabitants — Brachistochrone, HookeanSpring, HMC, CartPole Phase 1+2 — each using the `compileAndRun` pattern from `compiler-plugin/src/test`.
- A `main` (or Gradle task `:harness:run`) that runs all benchmarks and prints a CSV.

**Acceptance:** the CSV writes to `build/harness-results-tlaloc.csv` with median/min/p99 ns per benchmark. Numerical agreement against the per-port FD tests.

**Estimated:** 1 firing.

### Phase 2 — Python reference implementations + JSON IPC (1-2 firings)

**Deliverable:** Python scripts `harness/python/<bench>.py` (or `harness/python/run_pytorch.py` and `harness/python/run_jax.py`) that:
- Implement each benchmark's primal in PyTorch + JAX.
- Time `grad` evaluation under `compile` / `jit`.
- Write JSON results to `build/harness-results-{pytorch,jax}.json`.

The JVM aggregator in Phase 1 reads these JSON files (when present) and compares.

**New surface required:** Python reference implementations. The user has not authorized toolchain installs in the active /loop, so this phase is gated on the user installing PyTorch + JAX externally OR running in an environment where they're already available. **The plan documents the protocol; the implementation waits for user-side toolchain availability.**

**Acceptance:** for at least 3 benchmarks (Brachistochrone, HookeanSpring, HMC), Tlaloc / PyTorch / JAX gradients agree to 1e-3 abs / 5e-3 rel and timings differ by at most 10× across frameworks. Estimated: 1-2 firings post-toolchain-install.

### Phase 3 — Speedup vs paper's reported figures (1 firing)

**Deliverable:** A §0.4 entry titled **"Phase 1 closed — coarsening at M9 parity"** that pins the 4-6 head-to-head numbers against the paper's reported speedups. Format (paper figures from Table 3 "Overall Time" column of `docs/papers/coarsening-autodiff.txt`):

```
| Benchmark       | Paper reported (overall, vs Kotlin baseline) | Tlaloc actual | Pass M9? |
|-----------------|----------------------------------------------|---------------|----------|
| BGDHyperOpt     | 8.06-8.56×                                   | <measured>    | ✓/✗      |
| Brachistochrone | 1.79-2.51×                                   | <measured>    | ✓/✗      |
| CartPole        | 1.05-1.12×                                   | <measured>    | ✓/✗      |
| HMC             | 2.27-3.56×                                   | <measured>    | ✓/✗      |
| HookeanSpring   | 4.09-11.02×                                  | <measured>    | ✓/✗      |
| QWOP            | 1.41-1.57×                                   | <measured>    | ✓/✗      |
```

(The earlier draft of this table had Brachistochrone ↔ HookeanSpring numbers swapped and a wrong CartPole range; corrected against the paper text in §0.4.239.)

**Acceptance per §11.13's M9 exit criterion:** within 20% of paper's figures, >3× over torch.compile on at least three of six, f32-tolerance numerical match.

**Estimated:** 1 firing once Phase 2 ships.

## Out of scope

- **Multi-device** comparison (paper's distributed setups). Single-device CPU only.
- **GPU** comparison. CPU only — IREE CPU is the M3-aligned target; GPU is M5+.
- **Cold-start / model-loading** latency. The harness measures hot-loop gradient eval, not framework initialisation.
- **JAX TPU**, **PyTorch XLA**, or other accelerator backends. Vanilla `torch.compile` (CPU) and `jax.jit` (CPU) only.
- **Benchmarks not yet ported through K2 plugin** — CartPole Phase 3 + BGDHyperOpt-via-plugin + QWOP. The harness scaffold can hold their slots; running them requires the ports first.
- **Statistical significance** beyond median/min/p99. Per-iteration timings have heavy-tailed distributions in real workloads; the three-statistic summary is sufficient for a milestone-pinning entry.

## Why this plan is structured differently from HMC's / CartPole's

HMC + CartPole plans port a single benchmark each. The head-to-head harness is **multi-benchmark, cross-framework**. Phase 1 (JVM-side scaffolding) is distinct from Phase 2 (Python reference) because the JVM side can land independently of toolchain availability. Phase 3 is a doc-only landing once Phase 2 produces measurable numbers.

The three-phase split also matches the paper's own structure: §3 describes the methodology; §7 reports the per-benchmark speedups. Tlaloc's harness lands the methodology first (Phase 1), then plugs in framework references (Phase 2), then publishes the speedup table (Phase 3).

## Phase 1 first-slice — concrete next firing

The immediate next firing should land:

1. New `:harness` module under `harness/` with `build.gradle.kts` depending on `:compiler-plugin`'s test classpath (or copy the in-process compilation harness).
2. `Benchmark.kt` interface + 3 inhabitants (Brachistochrone, HookeanSpring, HMC).
3. `main()` that runs each benchmark for 1000 iterations (200 warmup, 800 measured), captures timings, writes CSV.
4. A Gradle task `:harness:runHeadToHead` that compiles + runs the harness.

**Estimated:** 1 firing. Bounded scope: JVM-only timing scaffolding, no Python yet.
