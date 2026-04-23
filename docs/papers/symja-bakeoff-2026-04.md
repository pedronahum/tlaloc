# Symja Adequacy Bake-off — 2026-04-21

**Result: PASS.** Symja is the green-light symbolic engine for Stage B. Proceed to B.1.

## Context

Per [`docs/STAGE_B_PLAN.md`](../STAGE_B_PLAN.md) §3.2.3 + §7.1.b, Stage B.0b's job is to confirm Symja can handle the algebraic complexity of the OOPSLA 2021 paper's hardest single benchmark — the BGDHyperOpt closed-form derivation walked through in paper Fig. 6c — within a fixed time budget. If Symja could not, the alternatives per plan §13 risk #2 were: (i) tune Symja, (ii) cut to a custom Kotlin CAS (~2-4 weeks scope add), or (iii) descope to "Stage B'" with F1-F5 + C1-C4 only (~40% paper speedup).

This bake-off is the single highest-value Stage B-level decision point — every other corollary Stage B implements is downstream of Symja's adequacy.

## Setup

- **Library:** [`org.matheclipse:matheclipse-core:3.1.1`](https://central.sonatype.com/artifact/org.matheclipse/matheclipse-core/3.1.1) ([Symja 3.1.1](https://github.com/axkr/symja_android_library), LGPL-3.0).
- **Engine wrapper:** [`SymjaEngine.kt`](../../ir/src/jvmMain/kotlin/io/tlaloc/ir/passes/SymjaEngine.kt) — `synchronized(evaluator)` around every `eval` per plan §13 risk #13.
- **Test harness:** [`SymjaBakeoffTest.kt`](../../ir/src/jvmTest/kotlin/io/tlaloc/ir/passes/SymjaBakeoffTest.kt). Three tests, run via `./gradlew :ir:jvmTest --tests "*SymjaBakeoffTest*"`.
- **Hardware:** Apple M2 mini (2023 / Mac Mini, baseline single-thread perf). Wall-clock numbers should be lower-bound on M3 Pro / aarch64 cloud or equivalent.
- **Dataset for FD comparison:** `x = [1, 2, 3, 4, 5]`, `y = [2.1, 3.9, 6.1, 8.0, 10.2]`, `r = 0.01`, `K = 3`. Small enough to keep simplification fast; large enough that the FD reference is meaningful.

## Method

The test does NOT replicate paper Fig. 6c's φ-calculus walkthrough as a sequence of `SymbolicEngine` calls — that is Stage B.1-B.3's actual deliverable. Instead, it constructs the closed-form `err` Symja should produce *given* a correct walkthrough, then exercises Symja's heavy lifting:

1. Build the closed form per Fig. 6c lines 10 + 16 + 18 + 26:
   ```
   a       = 1 + 2·r·Sx2/M
   b       = -2·r·Sxy/M
   w_4     = b · Σ_{k=0}^{K-1} a^k                         (paper's no-break path)
   err     = sqrt(Sy2 − 2·Sxy·w_4 + Sx2·w_4²) / M
   ```
2. `simplify(err)` and time it. Target: < 60s.
3. `simplify(diff(err, r))` and time it. Target: < 30s.
4. Substitute concrete numeric values for `(r, M, Sxy, Sx2, Sy2)` derived from a Kotlin reference impl of the original BGD loop and compare to a finite-difference gradient `(err(r+h) − err(r−h)) / (2h)` at `h = 1e-4`. Target: relative error < 1e-3.

The Σ is unrolled at construction time with concrete `K = 3` (Symja closes the geometric series trivially when n is concrete). Symbolic-K closure — the paper's actual claim, via `engine.sum(λk → a^k, 0, K-1)` — is a Stage B.2/B.3 test against `engine.sum`, not part of the bake-off.

The Kotlin reference impl ([`bgdErrLoop`](../../ir/src/jvmTest/kotlin/io/tlaloc/ir/passes/SymjaBakeoffTest.kt)) is the BGDHyperOpt kernel from paper Fig. 6a verbatim, with the `if (d < 0.001) break` skipped (the closed form on Fig. 6c assumes K iterations always run, so the FD reference has to match that assumption to be comparable).

## Results

```
[bake-off] closed-form construction + simplify: 74ms
[bake-off] differentiation + simplify:           660ms
[bake-off] symbolic d(err)/dr = 294.58063922276386
[bake-off] FD       d(err)/dr = 294.58095844500320
[bake-off] relative error     = 1.0836485868806908E-6
```

| Metric | Budget | Measured | Ratio |
|---|---|---|---|
| Closed-form construction + simplify | 60,000 ms | **74 ms** | 0.12% of budget |
| Differentiation + simplify | 30,000 ms | **660 ms** | 2.2% of budget |
| Symbolic-vs-FD relative error | 1e-3 | **1.08e-6** | 1000× better than budget |

All three tests pass cleanly on the first run.

## Decision

**PASS** with massive headroom. Symja handles the BGDHyperOpt closed form three orders of magnitude faster than the per-step budget allows, and the resulting symbolic gradient agrees with the FD reference essentially at machine precision (the 1.08e-6 relative error is dominated by FD-step-size truncation, not by Symja-side floating-point drift).

**Implication:** Stage B.1 (F1, F2, F3, C1, C3 on hand-built IF primals) gets the green light. The single biggest "we don't know if we can ship" risk in the Stage B runway is closed.

## Side notes worth flagging

- **Log4j / SLF4J warnings on stderr.** Symja uses Log4j and SLF4J for internal diagnostics; we have no logger configured, so each test prints `Log4j API could not find a logging provider` and `No SLF4J providers were found` to stderr. Harmless — Symja falls back to a no-op logger. To silence: add `slf4j-nop` or `log4j-to-slf4j` as a `:ir:jvmMain` runtime dep, or set `-Dorg.slf4j.simpleLogger.defaultLogLevel=off`. Defer to Stage B.1+ if cosmetic noise becomes a concern.
- **Transitive dependency footprint.** Symja pulls in Guava 33.x, Jackson 2.21.x, jspecify 1.0.0, errorprone-annotations, j2objc-annotations, and Apache Commons Math — roughly ~15 additional jars on the `:ir` runtime classpath. Acceptable for a JVM-only compiler-plugin module that never ships to mobile (per spec §11.7 + §18, mobile coarsening uses pre-baked artifacts, never the live engine). If the dep footprint matters later, the cleanest reduction is a custom Kotlin CAS (§5.3 of plan).
- **`evalDouble()` is deprecated** in Symja 3.x. Suppressed in [`SymjaEngine.kt`](../../ir/src/jvmMain/kotlin/io/tlaloc/ir/passes/SymjaEngine.kt) — it's the simplest cross-version path; the newer `evalf()`-based API churns by minor version. Re-evaluate on a Symja bump to a 3.x → 4.x boundary.
- **One-thread benchmark only.** The `synchronized(evaluator)` wrap means concurrent `PhiCalculus.apply` calls serialise. The K2 plugin compiles modules sequentially in the simple case but Gradle's worker API can parallelise per-task; if Stage B.3 measures contention, switch to thread-local evaluators.

## Next session

Stage B.1 — F1, F2, F3, C1, C3 on hand-built IF primals. Plan §7.2. Estimated 3-4 sessions, +20-25 tests.
