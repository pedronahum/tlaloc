# Stage B Plan — φ-Calculus Coarsening Pass for Tlaloc

**Status:** Research + planning. Not an implementation. Hand-off artifact for the engineer who lands Stage B.
**Owning spec:** [DIFFKTX_SPEC.md §11](../DIFFKTX_SPEC.md) (binding) and §0.4.9 (Stage A closure, last landed: 2026-04-20).
**Date authored:** 2026-04-20.
**Target duration for implementation:** ~3 person-months (paper's budget per §11.11), but see §14 for a revised estimate that accounts for the control-flow-in-DXIR prerequisite surfaced in §3.1 below.

---

## 0. How to read this document

This doc is meant to be picked up by the next engineer (or agent) after Stage A closed in §0.4.9. It is structured as:

- §1 tells you what you're building and what you're committing to.
- §2 tells you what can stop you cold (paper access, control-flow primitives, Symja quirks).
- §3 tells you what has to land **before** a single φ-calculus rewrite can run.
- §4 spells out the calculus itself in dxir terms — F1–F5 and C1–C9.
- §5 specifies the symbolic-engine interface and caching.
- §6 specifies how coarsening splices into the existing `FIR → DXIR → DxirReverseTransform → DxirToIrSynthesis` pipeline.
- §7 breaks Stage B into shippable sub-milestones B.1–B.5.
- §8–§9 preview Stage C (SOI identification) and Stage D (benchmark bring-up).
- §10 is framework + CAS comparisons.
- §11 is the metrics harness (correctness, wall-clock, compile-time, op-count, allocation, CI gates).
- §12 ports §11.9's numerical-stability pattern list.
- §13 is the risks table.
- §14 is the timeline estimate with a revision on the spec's Month-9 date.
- §15 is the glossary.

Everything that isn't in the current codebase or the paper is flagged. "[needs paper verification]" means the claim came from a WebFetch/ar5iv extraction pass and should be cross-checked against the PDF before being written into tests. "[open]" means a decision the implementer has to make; the doc usually names the leading candidate.

---

## 1. Executive summary

Stage B is the **φ-calculus pass**: a pure `DxirFunction → DxirFunction` rewriter that recognises loops and `if/else` structures in dxir, lifts them to symbolic expressions via Shen et al.'s φ-calculus (§11.5 of the spec), differentiates each expression symbolically using a CAS, then emits the simplified primal + gradient back as dxir. Stage A closed the non-research prerequisite (op-by-op SCT reverse-mode AD on straight-line dxir). Stage B is what actually produces the 10×–335× speedups the paper measured (§11.2); without it the compiler plugin is on par with AdOptimize and PyTorch 2.x `torch.compile`, not ahead of them.

This plan commits Stage B to five sub-milestones (§7):

- **B.0** — prerequisites. Introduces `OpKind.IF` and `OpKind.WHILE` (minimal), widens `DxirRegion` usage where needed, lands the `SymbolicEngine` interface + Symja default. Does **not** yet exercise any φ-calculus formula on a real primal.
- **B.1** — F1, F2, F3 (the three arithmetic-free formulae) as dxir rewrites. Scalar only. Hand-constructed IF primals as the test substrate.
- **B.2** — F4, F5 + C5 (the one corollary that's fully spelled out in the paper). Hand-constructed WHILE primals. First end-to-end symbolic differentiation of a loop.
- **B.3** — C1–C4, C6–C9 (the eight remaining corollaries). Needs paper access to pin each body exactly.
- **B.4** — FIR-side lowering. Widen `FirLambdaToDxirLowering` so Kotlin `if/else` and `while` bodies inside a `grad { … }` lambda lower to `OpKind.IF` / `OpKind.WHILE` (which Stage B.1/B.2 now understand). Before B.4, Stage B's test substrate is hand-built dxir functions — which is fine for unit tests but doesn't wire up end-to-end user code.

The plan **defers** to Stage C:
- Reuse-aware SOI identification (paper Fig. 7 — a ~40-line algorithm, but it's meaningful enough to isolate from the φ-calculus proper).
- Splicing coarsened adjoints back into the standard reverse-mode pipeline via `OpKind.COARSENED` (§6 names the interface Stage B exposes for Stage C to consume).

And it **defers** to Stage D:
- Porting the six paper benchmarks.
- End-to-end wall-clock measurement vs. PyTorch 2.x + torch.compile and JAX + jit.

Stage B's **deliverable** is a pass that, given a hand-built `DxirFunction` containing a `while` that matches the paper's Fig. 6 BGDHyperOpt kernel, produces a closed-form simplified primal + gradient that numerically agrees with running the loop op-by-op through `DxirInterpreter.evalFunction` (§0.4.6's evaluation substrate). Stage C then makes the pass discover SOIs on its own; Stage D validates the end-to-end speedup claim.

---

## 2. Blockers & open questions

Severity scale: **Fatal** (cannot land any Stage B code until resolved); **High** (can land B.0–B.1 but not past); **Medium** (can land all of Stage B but test surface is narrower); **Low** (cosmetic / deferrable).

### 2.1 Blocker 1 — Paper access **[RESOLVED 2026-04-21]**

**State (2026-04-21):** Resolved. The user installed `poppler` + `mupdf-tools` + `qpdf` (Fallback 2 in the original mitigation plan); `pdftotext -layout` extracted the paper to `docs/papers/coarsening-autodiff.txt` (1543 lines of layout-preserved text). Section 4.3 (Fundamental Formulae + Corollaries), Figure 5 (the formula table), Section 5 (SOI identification), Figure 7(a) (the SOI algorithm pseudocode), and Section 6 (numerical-stability rewrite) are all now readable.

**Impact of the unblock:** the verbatim bodies for **F3, F5, C1, C2, C3, C4, C6, C7, C8, C9** are now in §4 below, replacing the `[needs paper verification]` markers. **One important correction:** the spec §11.5 paraphrase of **F4** as `φ_L(p, φ_L(p, d)) = φ_L(p, d)` is **wrong** — the paper's Fig. 5 shows F4 as `φ_L^(i)(p, d) = d^(i-1)` (the loop-entry φ at iteration i takes the value of the loop-carried variable from the previous iteration, with `d^(0) = p` defined). The §11.5 paraphrase looks like a derivative of F4 (substitute the recurrence into itself), but it's not the formula the paper states. §11.5 of the spec should be corrected — flagged as a separate edit at the bottom of this section.

**Source of truth for Stage B going forward:** `docs/papers/coarsening-autodiff.txt` (project-local). Paper PDFs at:
- [PDF (Shivers' hosted)](https://www.ccs.neu.edu/home/shivers/papers/coarsening-autodiff.pdf)
- [arXiv:2110.02307](https://arxiv.org/abs/2110.02307)
- [ACM DL 10.1145/3485507](https://dl.acm.org/doi/10.1145/3485507)

**§11.5 spec correction needed (pre-Stage-B.0):** the F4 body in the spec is wrong. Recommend updating §11.5 to:
> - **F4 (Loop-entry):** `φ_L^(i)(p, d) = d^(i-1)` where `d^(0) = p` — the loop-entry φ at iteration `i` equals the loop-carried value from iteration `i−1` (with `d^(0)` defined as the pre-header value `p`). Turns loop-entry φs into a recurrence relation.

(The paraphrase the spec currently has, `φ_L(p, φ_L(p, d)) = φ_L(p, d)`, is closer to a *consequence* of F4 + F1 than to F4 itself; the correction is a clarification, not a redirection of Stage B's work.)

### 2.2 Blocker 2 — Control-flow ops in DXIR **[Fatal for Stage B start]**

DxirRegion + DxirBlock exist structurally ([`ir/src/commonMain/kotlin/io/tlaloc/ir/DxirNode.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/DxirNode.kt)), but no `OpKind.IF`, `OpKind.WHILE`, or `OpKind.SCAN` exists today ([`OpKind.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/OpKind.kt)). The only region-carrying op is `MANUAL_COMPUTATION`. `DxirReverseTransform` explicitly rejects any op with `hasRegions` ([`DxirReverseTransform.kt:80`](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/DxirReverseTransform.kt#L80)):

```kotlin
require(!n.hasRegions) {
    "DxirReverseTransform v1 rejects ops carrying regions (got ${n.op}); " +
        "structured-control-flow is the φ-calculus pass (Stage B)"
}
```

That `require` is the literal sign saying "Stage B lives here". But Stage B has nothing to pattern-match on until the ops it rewrites actually exist. **Stage B.0's first job is landing `OpKind.IF` and `OpKind.WHILE`** with the region shape described in §3.1.

**Severity is Fatal for starting — but contained.** The IR widening is a ~1 session sub-project (add op kinds + region-shape conventions + DxirBuilder helpers + stablehlo emitter stubs + interpreter stubs). Flagged Fatal because no φ-calculus pseudocode can even compile against the current IR without it.

### 2.3 Blocker 3 — Symja dependency shape **[Medium]**

[Symja](https://github.com/axkr/symja_android_library) is a JVM-native **LGPL-3.0** computer algebra system (§11.7 names it as the default CAS). **License correction (B.0b, 2026-04-21):** earlier drafts of this doc and spec §11.7 listed Symja as Apache-2.0 — that was wrong. matheclipse-core's README states "the maven modules: parser, external, core are published under LGPL license"; SPDX identifier is LGPL-3.0. LGPL allows linking from non-LGPL code, so Tlaloc's eventual Apache-2.0 (or similar) license is unaffected — but modifications to matheclipse-core itself would inherit LGPL on our side. Treat Symja as a strict runtime dependency, never a fork-and-patch surface. See [`docs/papers/symja-bakeoff-2026-04.md`](papers/symja-bakeoff-2026-04.md) for the licensing analysis. We need to pin a concrete version and confirm:

- **Maven coordinate:** `org.matheclipse:matheclipse-core` (recommended) vs. the fat-jar `matheclipse-mma`. Recommend `matheclipse-core` for minimal deps.
- **Version:** Symja was at 2.0.0-snapshot when last checked pre-2026; confirm current stable release and pin. §11.7 requires caching keyed by `(dxir subtree hash, CAS version)`, so the version string must be retrievable at runtime — `Package.getImplementationVersion()` on one of Symja's classes is the standard way.
- **JVM-only:** Symja depends on `java.math.BigDecimal`, `Logger`, and other Java stdlib surfaces. Kotlin/Native + Kotlin/JS are not target platforms. That is **fine** — the K2 compiler plugin runs on JVM 17+ (`build.gradle.kts` gradle 8.12+, JDK 21 per spec §14). Stage B's code ships in `:compiler-plugin` (or a new `:coarsening` module per spec §13) which is JVM-only. Mobile / KMP doesn't need Symja at all; coarsened artifacts get baked at publish time and the AAR ships only generated code (per §18's "Coarsening on mobile" discussion).
- **License compatibility:** **LGPL-3.0**, not Apache-2.0 — this bullet was missed by the B.0b correction above and stayed wrong until §0.4.498 checked `matheclipse-core-3.1.1.pom` directly (`<name>GNU Lesser General Public License, Version 3</name>`). Note also that the *repository's* root `license.txt` is plain **GPL-3.0**: upstream's position is that the published maven modules (parser, external, core) are LGPL while the repository as a whole — including the Android application parts we do not consume — is GPL. We rely on the POM and that README statement, and we link only. Tlaloc's own license is no longer a "presumed destination": §0.4.498 made it Apache-2.0 (`LICENSE`), closing that §18 open question.
- **Surface area we need:** symbolic `Derivative[f, x]`, `Simplify[expr]`, `Expand[expr]`, `Factor[expr]`, `Collect[expr, x]`, plus rational arithmetic. All are top-level Symja functions. We do NOT need: ODE solvers, special functions, plotting, parsers beyond `ExprParser`, 3D geometry — everything outside "scalar algebra + calculus".
- **Performance envelope:** Symja's `Simplify` can take **seconds to minutes** on large expressions (this is the paper's observation about SymPy — it's not unique to SymPy). The cost is bounded by SOI size limit L (§11.6). L's tuning is Stage C's problem, not Stage B's, so set L conservatively (e.g. L = 50 nodes) for Stage B tests.

**Medium severity, not high** — the interface we specify in §5 is explicitly Symja-swappable. If Symja turns out to be infeasible (e.g. the ported benchmarks hit compile-time > 5 minutes that we can't tune out), we swap in a custom minimal Kotlin CAS. The interface boundary insulates us.

### 2.4 Open question 1 — Does Stage B need `OpKind.SCAN` (for-loop / unbounded repeat)? **[Open, recommend defer]**

Paper Fig. 6 uses `while` + `if` + `for`. In SSA, a `for` loop is just a `while` with an induction variable; a `scan` is a higher-level primitive that folds a function over a sequence with explicit carry. Three options:

- (a) Only `IF` + `WHILE`. `for` lowers to `WHILE` with an induction counter. Matches SSA orthodoxy.
- (b) `IF` + `WHILE` + `SCAN`. `SCAN` maps to StableHLO's `stablehlo.reduce_window` / `scan` primitive and is how JAX represents loops at jit-time.
- (c) Full set: `IF` + `WHILE` + `FOR` + `SCAN`.

**Recommend (a).** The paper's φ-calculus is stated in terms of loop-entry / loop-exit φ, which map cleanly onto WHILE headers regardless of whether the source was a `for` or a `while`. C5 specifically says `𝔏ⁿ d = f(φ_L(p, d))` — the loop trip count `n` can come from an induction-variable bound (for) or a convergence predicate (while); the calculus is indifferent. Adding SCAN later as an optimization for array reductions is fine, but it's not a Stage B load-bearing primitive.

**Consequence:** `FirLambdaToDxirLowering`'s future `for (i in 0 until n) { … }` handling (Stage B.4) lowers the Kotlin `for` as `WHILE(cond = i < n, body = { …; i = i + 1 })`. That is verbose at the dxir level but clean at the calculus level.

### 2.5 Open question 2 — Where does the coarsening cache live? **[Open, recommend project-local]**

§11.7 requires a disk cache keyed by `(dxir subtree hash, CAS version)`. Two locations:

- (a) `~/.gradle/caches/tlaloc/coarsen/`. Persists across projects on the same machine; same as Gradle's own caches. But: means a user rebuilding from a clean machine pays full cost; means CI has to warm it per runner.
- (b) `<project>/.tlaloc-cache/coarsen/`. Per-project; checked into source control or not? If not, CI warms on first PR and re-warms on cache-miss; if yes, PRs may include cache artifacts (ugly but deterministic).

**Recommend (b), uncommitted** (add to `.gitignore` in Stage B.0). Rationale: per-project isolation means two unrelated branches with different op-surface additions don't poison each other's cache. CI cost is bounded — the cache's `(dxir subtree hash, CAS version)` key means any PR that doesn't touch the IR or bump Symja hits the cache.

**Invalidation:** bumping Symja's Maven version bumps the CAS version string; the cache key changes; entries age out naturally. An explicit `--rerun-tasks` or `./gradlew clean` is not required for cache-miss.

### 2.6 Open question 3 — Do we expose a `coarsenBarrier(x)` function in `:autograd` for Stage B, or defer to Stage C? **[Open, recommend defer]**

§11.10 lists `coarsenBarrier(...)` as one of four user-facing knobs. Its semantics is "nothing may coarsen across this point". It's meaningful only when SOI identification is running; Stage B's hand-constructed SOIs have explicit boundaries by construction. **Defer to Stage C.** When SOI identification lands, add `coarsenBarrier` as a no-op dxir op the SOI identifier treats as a hard segmentation boundary.

### 2.7 Open question 4 — Higher-order differentiation through coarsening **[Open, defer to Stage D or beyond]**

§11.8 closes with "coarsening composes. `grad(grad(f))` can re-coarsen the gradient function". §18's open question: "Commit resources to higher-order AD as a research problem or defer?"

**Stage B does not commit.** Stage B's tests assert first-order agreement only. If the architecture makes higher-order "just work" (Stage B.2 produces a gradient function that is itself a `DxirFunction` with straight-line body, which Stage A already handles; running Stage A over it gives `grad(grad(f))` op-by-op), then nothing blocks experimenting with `grad(grad(f))` through the coarsening+SCT combination. But it's not a Stage B DoD item.

---

## 3. Prerequisites

These are the questions an engineer must resolve before writing the first φ-calculus rewrite.

### 3.1 Control-flow ops in DXIR — the load-bearing prerequisite

Today's dxir surface:
- `DxirRegion` exists (`DxirNode.kt:99`), conventionally single-block (`ofBlock(args, body, terminator)`, `DxirNode.kt:106`).
- `DxirBlock` has typed `args`, a `body` list, and a `terminator` list (`DxirNode.kt:82`).
- `DxirOp.regions: List<DxirRegion>` (default empty, `DxirNode.kt:31`).
- Only `OpKind.MANUAL_COMPUTATION` currently carries a region (spec §0.4.1 — multi-region via `sdy.manual_computation` lowering).

We need to introduce two new op kinds:

#### 3.1.1 `OpKind.IF`

**Region shape:** Two regions — `[thenRegion, elseRegion]`. Each region has a single block with no block args and a terminator yielding the branch result(s). The `IF` op takes one operand (the boolean predicate) and produces the same types as each branch yields.

**Proposed encoding:**
```
%result = IF %cond {
  then: { <body_t>; yield %tval }
  else: { <body_e>; yield %eval }
} : <result_type>
```

Shape constraints:
- `cond.type.dtype == Bool`, `cond.type.isScalar` (for dxir scalar branches). Vector-predicate IF is a tensor `where` op which is already represented differently; keep IF scalar-predicate only for Stage B.
- `thenRegion.blocks.single().terminator.map { it.type } == elseRegion.blocks.single().terminator.map { it.type } == op.types` — arity and types agree.
- No block args in either branch (the branches close over the surrounding body's SSA values; `DxirBlock.args = emptyList()` is the canonical form).

#### 3.1.2 `OpKind.WHILE`

**Region shape:** Two regions — `[condRegion, bodyRegion]`. Each region's entry block takes the **loop-carried** SSA values as block args and yields the new loop-carried values (bodyRegion) or a boolean predicate (condRegion).

**Proposed encoding:**
```
%result = WHILE (%init₁, %init₂, …, %initₙ) {
  cond: (%c₁, %c₂, …, %cₙ) -> { <body>; yield %pred : Bool }
  body: (%b₁, %b₂, …, %bₙ) -> { <body>; yield %out₁, %out₂, …, %outₙ }
} : (T₁, T₂, …, Tₙ)
```

The op:
- Takes `n` operands `%init₁ … %initₙ` — the initial loop-carried values.
- Has two regions. Both regions have blocks with `n` block args typed the same as `%init₁ … %initₙ`.
- `condRegion.blocks.single().args.size == n`, `condRegion.blocks.single().terminator.size == 1` (the predicate), `terminator[0].type.dtype == Bool && isScalar`.
- `bodyRegion.blocks.single().args.size == n`, `bodyRegion.blocks.single().terminator.size == n` — yields the new loop-carried values.
- `op.types.size == n` — the WHILE op is multi-result, one per loop-carried value, matching the types of `%init_i`.

This matches MLIR's `scf.while` dialect almost exactly — which matters because StableHLO itself is MLIR-based and future `stablehlo.while` emission (deferred post-Stage-B) inherits this shape.

#### 3.1.3 IR widening checklist (Stage B.0)

- [ ] Add `OpKind.IF, OpKind.WHILE` to [`OpKind.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/OpKind.kt).
- [ ] Extend `DxirBuilder` with convenience builders `ifOp(cond, thenRegion, elseRegion, types)` and `whileOp(inits, condRegion, bodyRegion)`. Signatures live in `DxirModule.kt`.
- [ ] Widen `DxirFunction`'s init-time ref-integrity check (`DxirModule.kt:10`) to validate that every `WHILE`-region block-arg is declared-before-use, and that region terminators' types match the claimed result types. It already walks nested regions via `collectReferencedIds`; the new checks are structural (shape-match on `regions[0].blocks.single().args.size == operands.size` etc.) rather than SSA-referential.
- [ ] Add an explicit `DxirRegionBuilder.arg(type)` call — already exists (`DxirModule.kt:141`) — but widen the docs to mention WHILE block-args.
- [ ] Stub `:stablehlo` emission. Don't implement it — Stage B.0 just needs a no-op `emitIf`/`emitWhile` that `error()`s with "OpKind.IF/WHILE emission deferred to post-Stage-B, use coarsening first". The round-trip suite won't see these ops until Stage B.4 at earliest.
- [ ] Stub `DxirInterpreter.evalOp` — add `OpKind.IF` and `OpKind.WHILE` arms that *do* evaluate (so we can run hand-constructed IF/WHILE primals end-to-end in Stage B tests). The interpreter already lives in [`DxirInterpreter.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/DxirInterpreter.kt) and supports scalar + rank-1 + rank-2 primals (§0.4.8's shape-aware rewrite). Adding IF is ~10 lines; adding WHILE is ~30 (carry loop-carried values through iterations, evaluate cond, terminate).
- [ ] Write 6–8 structural tests in `:ir` validating the shape invariants (predicate dtype, region arity, etc.) and 4–6 interpreter tests (`evalFunction` over hand-built IF and WHILE primals).
- [ ] `DxirReverseTransform`'s `require(!n.hasRegions)` gate (`DxirReverseTransform.kt:80`) stays — Stage A still rejects control flow. Stage B's pass runs **before** `DxirReverseTransform`.

#### 3.1.4 Scope decision — widen the IR first OR hand-build loop primals in test code?

Stage B plan says: **widen the IR first, in B.0, as a strict prerequisite. Do not defer it.**

Reasoning:
- Hand-building `DxirOp(kind = OpKind.IF, …)` in test code is essentially the same work as adding the op kind to the enum (the rest of B.0's IR widening is cheap). No meaningful deferral exists.
- `OpKind.IF`/`WHILE` is how φ-calculus pattern-matches. Without the op kinds, every `PhiCalculus.apply` call has to pattern-match on ad-hoc `DxirOp.attrs["is_if"] == true` or similar hacks, which rot the moment you add `OpKind.SCAN`.
- `DxirInterpreter` needs IF/WHILE arms regardless (§11.12 correctness harness — run the coarsened gradient through the interpreter and compare against the non-coarsened baseline). That work is blocked on the op kinds existing.

**Do not** defer FIR-side `if/else/while` lowering to B.0 — that work is B.4, and it's much larger (FIR's `FirWhenExpression` / `FirLoop` traversal is its own sub-project). B.0 hand-builds IF/WHILE primals in test code; B.4 wires them to user code.

### 3.2 Symbolic engine — Symja vs. custom Kotlin CAS

§11.7 recommends **Symja v0, minimal custom CAS v0.5** if Symja limitations bite. This plan operationalises that.

#### 3.2.1 What surface does φ-calculus actually need?

Enumerated from F1–F5 + C1–C9:

| Operation | Symja function | Priority |
|---|---|---|
| Construct an opaque unknown function `f(x)` | `SymbolicFunction.of("f")` (custom wrapper over Symja) | **must-have** |
| Symbolic differentiation `d/dx (expr)` | `Derivative[1][f][x]` or `D[expr, x]` | **must-have** |
| Substitute `x → value` | `expr /. x -> value` (`ReplaceAll`) | **must-have** |
| Algebraic simplification | `Simplify[expr]` | **must-have** |
| Polynomial expansion | `Expand[expr]` | **should-have** |
| Factor out common subexpressions | `Factor[expr]` | **nice-to-have** |
| Collect terms in `x` | `Collect[expr, x]` | **should-have** |
| Rational arithmetic (no float conversion) | `Rational[num, den]` | **must-have** (for compile-time-safe loop trip counts) |
| Exact integer arithmetic | `Integer[n]` (default) | **must-have** |
| Compose functions `f ∘ g` | `Composition[f, g][x]` or `f[g[x]]` | **must-have** (C5's `f^(n)(p)`) |
| N-fold composition `f^n(x)` | `Nest[f, x, n]` | **must-have** (C5) |
| Closed-form summation `Σᵢ₌₁ⁿ f(i)` | `Sum[f[i], {i, 1, n}]` | **must-have** (C6) |
| Closed-form product `∏ᵢ₌₁ⁿ f(i)` | `Product[f[i], {i, 1, n}]` | **should-have** (C7) |
| Heaviside step (for IF-branch algebraic encoding) | `UnitStep[x]` | **optional** |

All operations exist in Symja. Most also exist in any reasonable Kotlin CAS we'd build — but `Simplify`, `Factor`, and `Collect` are where a custom CAS starts costing real effort (Gröbner bases, term-rewriting rule engines, etc.). The custom CAS fallback should focus on: rational arithmetic + symbolic diff + `Nest` + a narrow `Simplify` that does polynomial-only normalisation. Anything beyond polynomial simplification falls back to the wrapped Symja.

#### 3.2.2 Caching strategy

Keyed by: `(canonical_dxir_subtree_hash, cas_version_string)`.

- **Canonical dxir subtree hash:** a SHA-256 over a canonical pre-order traversal of the subtree. Canonical means: SSA ids are re-numbered 0..N in traversal order (so renaming doesn't perturb the hash); attribute maps are serialised in key-sorted order; operand edges are (kind, source_subtree_hash) pairs.
- **CAS version string:** `"Symja-" + pkg.getImplementationVersion() + "-" + pluginBuildVersion`. The plugin version is included because a plugin-side rewrite bug invalidates all cached artifacts.

Cache entry payload:
- The simplified primal as a serialised dxir subtree (`DxirPrinter` format is fine — we have one at [`Printer.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/Printer.kt)).
- The gradient dxir subtree.
- The per-operand `readsPrimalOperandIndices` set (so `usedByAdjoint` can compute without re-running the rewrite).
- Serialisation format: a project-specific binary-stable format, or JSON if the op count is bounded (L = 50 nodes → payload < 50 KB typical).

Location: `<project>/.tlaloc-cache/coarsen/<hash-prefix>/<full-hash>.bin`. Two-level hash-prefix directory (first 2 hex chars) to avoid a single directory with 100k+ entries.

Invalidation: on CAS version change, old entries become unreachable (keys differ). A periodic pruning task (age > 30 days AND last_access > 7 days) keeps the cache bounded. Ship pruning as Stage B.3 at earliest — Stage B.1 + B.2 can write + skip reads until B.3 if the cache infra isn't worth landing yet.

**Warm-cache perf target:** §11.13 says < 5 seconds for typical training step after warm cache. That budget includes the IR-hash + cache-read + dxir-deserialize + VjpRegistry splice. Target the hash + read at < 50 ms; the rest of the budget is for non-coarsened compilation.

**Cold-cache perf target:** § 11.13 says < 30 seconds cold. That's per-SOI — a function with 10 SOIs each taking 3 seconds through Symja = 30 seconds. Parallelise? Probably not worth the complexity before measuring.

#### 3.2.3 Symja adequacy bake-off — when to cut over to custom CAS

**Go / no-go test:** run Fig. 6's BGDHyperOpt reduction symbolically through Symja. Starting point is the φ-calculus form shown in Fig. 6(c); end point is the closed-form `err = …` expression. Success criteria:

- (a) Symja produces a closed-form within **60 seconds** of wall clock on a modern laptop (MacBook Pro M3 Pro or equivalent).
- (b) The closed form is numerically equivalent (within f32 tolerance) to running the original loop op-by-op.
- (c) Running `Derivative[err, r]` (the learning-rate gradient) produces a closed form in **under 30 seconds** additional.

If (a) fails at all, cut to custom CAS. If (b) fails, the issue is probably a `Simplify` rule Symja applies incorrectly on rationals with floating-point rounding — tune `SetOptions[Simplify, Assumptions -> Real]` or switch to exact rational mode. If (c) fails but (a)+(b) pass, it's probably a `D[]` heuristic failing on the summation form — try hand-composing `D` with `Sum` closure rules first, or cut to custom CAS.

**Recommend running the bake-off in Stage B.0 itself** — before committing a single line of φ-calculus code. The SymbolicEngine interface (§5) is the abstraction point; if Symja fails the bake-off, the implementation work is in custom-CAS land, not Symja land.

### 3.3 The F1–F5 + C1–C9 test substrate: hand-constructed `DxirFunction` tests

Stage B's unit-test strategy is **structural**: construct a primal by hand using `DxirBuilder`, run `PhiCalculus.apply(primal)`, assert the produced function has the expected op-count + op-kind distribution, then run both through `DxirInterpreter.evalFunction` at sample inputs and assert numerical agreement. The paper's Fig. 6 BGDHyperOpt walkthrough is the canonical test.

Per-formula test shapes are detailed in §4 below.

### 3.4 SOI identification — deferred to Stage C

§11.6's reuse-aware SOI identification (paper Fig. 7, ~40 lines) is **not** a Stage B prerequisite. Stage B's job is to prove the φ-calculus rewrites are correct; Stage C drives real SOIs through them. Per-SOI correctness is a local property; global SOI identification is a search problem orthogonal to correctness.

### 3.5 Integration with the existing pipeline — the splice mechanism

**Where does coarsening slot in?** Per §11.8:

```
FIR lambda → DXIR (initial)
           → [coarsening pass]   ← Stage B + C lives here
           → DXIR (with coarsened SOIs as COARSENED ops)
           → [grad transform]    ← Stage A
           → DXIR (gradient body mixes COARSENED custom adjoints + per-op rules)
           → DxirToIrSynthesis
           → compiled bytecode
```

**The splice contract:** the coarsening pass replaces each identified SOI with a single `OpKind.COARSENED` op whose:
- `operands`: the SOI's live-in SSA values.
- `types`: the SOI's live-out types.
- `attrs["primal_body"]`: a serialised `DxirFunction` — the simplified primal.
- `attrs["gradient_body"]`: a serialised `DxirFunction` — the pre-computed VJP, with signature `(upstream, *primal_operands) → (d_operand_1, d_operand_2, …)`.
- `attrs["reads_primal_indices"]`: the `Set<Int>` of operand indices the gradient_body dereferences.

`DxirReverseTransform` then treats a `COARSENED` op as a "primitive with pre-computed VJP rule" — it consults `VjpRegistry[OpKind.COARSENED]` which is a synthetic rule that inlines `attrs["gradient_body"]` into the gradient body (or calls it as a nested function — the former is simpler).

**Interaction with `usedByAdjoint` (§0.4.5):** the coarsened adjoint's primal nodes may or may not be needed for the gradient. The `attrs["reads_primal_indices"]` set is exactly `readsPrimalOperandIndices` for the synthetic VJP rule. The existing analysis in `DxirReverseTransform.computeUsedByAdjoint` (`DxirReverseTransform.kt:197`) consumes it verbatim — no code change needed.

**Equivalent-behavior test:** for a `DxirFunction` F, construct F_coarsened = `PhiCalculus.apply(F)`. Verify:
- `DxirInterpreter.evalFunction(F, inputs) == DxirInterpreter.evalFunction(F_coarsened, inputs)` for every sample input (forward correctness).
- `DxirInterpreter.evalFunction(DxirReverseTransform.apply(F), inputs) ≈ DxirInterpreter.evalFunction(DxirReverseTransform.apply(F_coarsened), inputs)` within f32 tolerance (gradient correctness).

The dxir-eval bridge (§0.4.6) is exactly this testing substrate. It was built for Stage A bridge-equivalence testing; it carries Stage B correctness unchanged.

---

## 4. The φ-calculus, spelled out in dxir

This section takes F1–F5 and C1–C9 from spec §11.5 + the ar5iv extraction + the paper reference, and restates each as a Kotlin-ish dxir pseudocode transformation. Source annotations indicate the provenance: **[paper §4]** means the body matches the paper's displayed equation directly; **[§11.5]** means the body was distilled by the spec; **[ar5iv]** means extracted via WebFetch of the HTML fallback; **[needs paper verification]** means the body is approximate and requires verification before implementation.

### 4.1 Notation

Throughout this section:

- `φ(a₁, …, aₙ)` is a branch-merge φ (what SSA calls "the φ function at the merge point of n predecessors").
- `φ_L(p, d)` is a loop-entry φ: `p` is the pre-header value, `d` is the back-edge value.
- `φ_L'(a, b₁, …, b_m)` is a loop-exit φ: `a` is the value on the `break`/fallthrough path, `b_i` is the value on back-edge `i` (i.e., from the bodyRegion's i-th yield).
- `f(x)` is an arbitrary Kotlin lambda — in dxir, a `DxirFunction` or a DxirOp subtree.
- `f⁽ⁿ⁾(x) = f(f(…f(x)…))` — n-fold composition.
- `𝔏ⁿ d = f(φ_L(p, d))` — "after n iterations, `d` equals the final value of the iteration".
- `𝔏^exit` — the fallthrough / post-loop value.

In dxir:

- **φ_L(p, d)** lives in `WHILE.condRegion.blocks.single().args[i]` AND `WHILE.bodyRegion.blocks.single().args[i]` for the same loop-carried index `i`. The "p" is `WHILE.operands[i]` (the pre-header init); the "d" is `bodyRegion.blocks.single().terminator[i]` (the back-edge value).
- **φ_L'(a, b₁, …, b_m)** — for a single-back-edge WHILE, `φ_L'(a, b₁) = a` (the cond-region's false-predicate path value is the final loop-carried value). For `break`-containing loops (not yet supported in dxir; see §2.4), multiple back-edges exist and `m > 1`.
- **φ(a, b)** — an `IF` op's merge: `op.regions[0].blocks.single().terminator[0]` is `a` (then-branch yield), `op.regions[1].blocks.single().terminator[0]` is `b` (else-branch yield). The IF op's result is the φ node conceptually, even though dxir represents it as a DxirOp result.

### 4.2 F1 — Identity **[paper §4; §11.5]**

**Statement:** `φ(a, a, …, a) = a`.

**Dxir rewrite rule:** for an `OpKind.IF` whose then-branch and else-branch yield the **same** SSA value (by structural equivalence, not just reference identity), the entire IF op is replaced by that value.

```kotlin
// Pseudocode — not production.
fun applyF1(op: DxirOp): DxirNode? {
    if (op.op != OpKind.IF) return null
    val thenYield = op.regions[0].blocks.single().terminator.single()
    val elseYield = op.regions[1].blocks.single().terminator.single()
    return if (structurallyEqual(thenYield, elseYield)) thenYield else null
}
```

**Worked example:**
```
%c = IF %pred {
  then: { yield %x }
  else: { yield %x }
} : f32
  ==>
%c = %x  // rewrite F1; `c` is replaced by `x` in all dominated uses
```

**Test shape:**
- Primal: `grad { x -> if (cond) x else x }` where `cond` is any boolean (in dxir, any `DxirOp` of bool type).
- After F1: the IF op is gone; the function body is `{ return %x }`.
- Gradient: `d/dx = 1`.
- Numerical: `DxirInterpreter.evalFunction(grad_f, [x=3.0]) == [1.0]` for any x.

**Caveat:** F1 generalises to φ-nodes with `n > 2` inputs (the paper's statement is `φ(a, a, …, a)`). Stage B's IF is always 2-branch; F1 is trivially a 2-branch specialisation. If SCAN or multi-way branches land later, this formula extends naturally.

**Numerical stability note:** F1 is a pure structural simplification; no numerical risk.

### 4.3 F2 — Distributive **[paper §4; §11.5]**

**Statement:** `f(φ(a₁, …, aₙ)) = φ(f(a₁), …, f(aₙ))`.

**Dxir rewrite rule:** for any elementwise op (`OpKind.ADD, SUB, MUL, DIV, NEG, RELU, STEP, EXP, LOG, …`) whose operand(s) include an IF op, push the elementwise op **into** each branch. This is the transformation that "lets differentiation cross branches" (§11.5) — once the IF is no longer around a differentiable expression, the expression inside each branch is a straight-line dxir subtree that Stage A can handle op-by-op.

```kotlin
// Pseudocode
fun applyF2(op: DxirOp): DxirNode? {
    // Unary case: f(IF) → IF(f(then), f(else))
    if (op.operands.size == 1 && op.operands[0] is DxirOp && (op.operands[0] as DxirOp).op == OpKind.IF) {
        val ifOp = op.operands[0] as DxirOp
        val pred = ifOp.operands[0]
        val thenYield = ifOp.regions[0].blocks.single().terminator.single()
        val elseYield = ifOp.regions[1].blocks.single().terminator.single()
        val newThenRegion = DxirRegion.ofBlock(
            args = emptyList(),
            body = listOf(applyOpTo(op.op, thenYield, op.type, op.attrs)),
            terminator = listOf(/* result of the new op */),
        )
        val newElseRegion = /* symmetric */
        return DxirOp(allocateId(), OpKind.IF, listOf(pred), op.attrs, op.types, regions = listOf(newThenRegion, newElseRegion))
    }
    // Binary case: f(IF, y) → IF(f(then, y), f(else, y)) — symmetric for f(y, IF)
    // …
    return null
}
```

**Worked example — the classic one:**
```
%cond = … : Bool
%then = x * x    // dxir: MUL(x, x)
%else = x        // dxir: identity
%if = IF %cond { then: yield %then; else: yield %else } : f32
%result = %if + 1     // dxir: ADD(%if, 1)
  ==>   (apply F2 with f = ADD( _ , 1))
%then' = ADD(MUL(x, x), 1)
%else' = ADD(x, 1)
%result = IF %cond { then: yield %then'; else: yield %else' } : f32
```

**Test shape:**
- Primal: `grad { x: Float -> (if (cond) x * x else x) + 1 }` — after F2, each branch is straight-line, so Stage A's SCT AD can handle the differentiation.
- After F2: the ADD is inside each branch; the IF's branches yield `x*x + 1` and `x + 1` respectively.
- Gradient: `d/dx = if (cond) 2x else 1`.
- Numerical: `DxirInterpreter.evalFunction(grad_f, [x=3.0])` on the cond=true path returns `[6.0]`; on the cond=false path returns `[1.0]`.

**Important subtlety — avoid expression swell.** F2 is where "expression swell" comes from. If the same branch-merge φ feeds N different elementwise ops, naive F2 application creates N copies of the branch bodies. The paper's SOI identification (§11.6, Stage C) is specifically about controlling this. Stage B's F2 implementation should **only** apply F2 when:
- (a) the IF's only use is the current elementwise op (no reuse), OR
- (b) the IF's branch bodies are "small" (< 5 ops each — heuristic, tunable).

If both conditions fail, keep the IF as-is and let Stage A fall back to a custom adjoint splice (the IF's gradient is computed by symbolic differentiation of the closed-form expression, without F2-ing through the branches). Stage B.1 can start with condition (a) only; Stage B.3 adds (b).

**Numerical stability note:** F2 does not reorder arithmetic within a branch. Stability only matters when the `f` being distributed has its own stability concerns — those are §12 patterns applied per branch, not by F2.

### 4.4 F3 — Commutative **[paper §4.3.1 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):** `φ(a, b) = φ̄(b, a)` where `φ̄` is the complement of `φ`.

**Paper prose (§4.3.1, item 3):** "This formula shows the relationship between a 𝜙 function and its complement. In the formula, `φ̄` is the complement of `φ`, that is, it chooses the first argument when `φ` chooses the second, and the second when `φ` chooses the first."

**Resolution:** the spec §11.5 reading is correct; the ar5iv extraction's reading (`φ(a,b) + φ̄(a,b) = a + b`) was wrong.

**Dxir implementation:** F3 is a canonicalisation — for an IF op `(pred, then=a, else=b)`, F3 says it's structurally equivalent to a predicate-negated IF `(NOT(pred), then=b, else=a)`. Practical use: pick one branch order by a deterministic rule (e.g., always put the lexicographically-smaller branch yield first), so two structurally-different IFs that compute the same merged value canonicalise to the same form, making F1 trivial to detect.

```kotlin
// Pseudocode — F3 as canonicalisation
fun applyF3(op: DxirOp): DxirOp? {
    if (op.op != OpKind.IF) return null
    val pred = op.operands[0]
    val thenYield = op.regions[0].blocks.single().terminator.single()
    val elseYield = op.regions[1].blocks.single().terminator.single()
    // Canonical form: the branch with the smaller-id yield is the "then" branch.
    if (thenYield.id <= elseYield.id) return null  // already canonical
    val negPred = builder.op(OpKind.STEP, listOf(builder.op(OpKind.NEG, listOf(pred), pred.type)), pred.type)
    return /* swapped IF */
}
```

**Test shape:** build IF(pred, a, b) and IF(NOT(pred), b, a); both should canonicalise to the same form. Confirmed numerically equivalent via DxirInterpreter.evalFunction at sampled (pred, a, b) values.

**Stage B.1 action:** implement F3 as canonicalisation only (not as a sum identity). ~40 LOC; cheap.

### 4.5 F4 — Loop-entry **[paper §4.3.1 + Fig. 5; verified 2026-04-21; §11.5's paraphrase WRONG]**

**Statement (verbatim from paper Fig. 5):** `φ_L^(i)(p, d) = d^(i-1)`, with the convention `d^(0) = p` (the pre-header value).

**Paper prose (§4.3.1, item 4):** "This formula shows the inherent property of a loop-entry 𝜙 function. For the definition of a loop-entry 𝜙 function, `φ_L` is reached always through the back edge of loop `L` except for its first instance in that loop. The formula hence follows."

**Important spec correction:** §11.5 of [DIFFKTX_SPEC.md](../DIFFKTX_SPEC.md) currently paraphrases F4 as `φ_L(p, φ_L(p, d)) = φ_L(p, d)`. That is **not** what the paper states. The §11.5 reading is closer to a *consequence* of F4 (substitute the recurrence into itself once), but the paper's F4 is the bare recurrence. The correction is non-load-bearing for Stage B (the dxir implementation cares about the recurrence, not the substitution form), but the spec should be fixed.

**What F4 actually does:** it lets `φ_L^(i)(p, d)` (the value of the loop-entry φ at iteration `i`) be replaced by `d^(i-1)` (the back-edge value at iteration `i-1`). This is the load-bearing step in C5's proof — every iteration `i ≥ 1` has its loop-entry φ rewritten to the back-edge of the prior iteration, which then unfolds to `f^[i-1](p)`.

**Dxir implementation:** F4 is invisible at the structural level — it's a substitution rule the symbolic engine applies during C5/C6/C7/C8/C9 reductions, not a standalone dxir-to-dxir rewrite. There is no "apply F4" call on a `DxirFunction`; F4 fires inside `SymbolicEngine.nest(f, p, n)` when expanding `Nest[f, p, n]` to `f(f(...f(p)...))`.

**Stage B implication:** F4 does NOT need its own dedicated `applyF4` Kotlin function (revising the original plan). It's an internal axiom of the symbolic engine's loop-unfolding mechanism, used by C5–C9 implicitly. **Drop F4 from the explicit pass-pipeline ordering in §4.13.**

**Test shape:** F4's correctness is exercised through C5's tests — if `iterate5(x) = x · 32` after C5 produces the right answer, F4 is correct internally.

**Stability:** structural / symbolic. No numerical reordering.

### 4.6 F5 — Loop-exit **[paper §4.3.1 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):**
- **Condition (i):** `b_i = b_j` for all `1 ≤ i, j ≤ m` (all back-edge arguments of the loop-exit φ are equal — homogeneous).
- **Condition (ii):** `b_1^(0) = a` (the value of `b_1` at the loop entry point equals the first argument `a`, which is the pre-loop value).
- **Conclusion:** under (i) and (ii), `φ_L'(a, b_1, b_2, …, b_m) = b_1^exit` — the loop-exit φ equals the value of `b_1` at the exit of the loop.

**Paper prose (§4.3.1, item 5):** "This formula says that `φ_L′(a, b_1, b_2, · · ·, b_m)` equals the value of `b_1` at the exit of loop `L` if (i) the value of all arguments, except the first, of `φ_L′` are the same at `φ_L′`, and (ii) those arguments before the entry point of the loop have the value equaling the first argument's value `a`. Its correctness can be easily proved with the identity formula. Notice that the only time when `φ_L′` takes its first argument is when the entire loop is skipped, in which condition, according to (ii), `b_1^exit` equals `a`; in any other condition, `φ_L′` must take one of the other arguments, the value of which at the exit of the loop, according to (i), must equal `b_1^exit`."

**Why `m > 1`:** the paper's BGDHyperOpt example has a loop-exit φ with **three** arguments because the while-loop has a `break` in an if-else (Fig. 6b line L6: `w4 = φ_k'(w_1, w_2, w_3)`). The `break` path introduces an additional back-edge argument beyond the normal fallthrough.

**Dxir implementation:** F5 applies to a WHILE op's exit-valued φ (the value consumed after the loop terminates). In the minimal dxir shape Stage B.0 lands (§3.1.2, single back-edge per loop), `m = 1` and condition (i) is vacuously true; F5 just resolves "the loop-exit value equals the final value of the loop-carried variable" — which is already the dxir semantics of a single-back-edge WHILE. For `m > 1` (break, multiple back-edges), F5 is the rewrite that collapses `φ_L'(a, b_1, …, b_m)` to `b_1^exit` when all non-first-arg back-edges agree and the first-arg pre-loop value matches the first back-edge's loop-entry value.

**Stage B.0 scope decision:** single-back-edge WHILE only (no `break`). F5 is semantically trivial for that case. `break`-bearing loops + multi-arg loop-exit φs are deferred post-Stage-B (paper's BGDHyperOpt Fig. 6 requires them, but can be hand-constructed in test primals using the approach Stage B's test substrate already uses).

**Test shape:**
- Primal: `WHILE { body: yield (f(x), i+1) }` — single back-edge, no break.
- After F5 (trivial): the loop-exit value equals the bodyRegion's terminator value at the final iteration.
- After C5: the loop-exit value equals `f^[n](x)`.
- Numerical: `DxirInterpreter.evalFunction` over both forms agrees at sampled `(x, n)`.

### 4.7 C1–C4 (first group — symbolic differentiation and optimization) **[paper §4.3.2 + Fig. 5; verified 2026-04-21]**

Per paper Fig. 5 + §4.3.2:

- **C1 (verbatim):** `f(x_1, …, x_{i-1}, φ(a, b), x_i, …, x_k) = φ(f(x_1, …, x_{i-1}, a, x_i, …, x_k), f(x_1, …, x_{i-1}, b, x_i, …, x_k))`. The distributive formula curried — when `f` is a multi-argument function and one of its arguments is a `φ`, the `φ` lifts to wrap the whole call. Derived from F2 by currying.
- **C2:** C1 with `f` substituted by partial derivative `∂/∂x_j`. Derived form: `∂/∂x_j (φ(a, b)) = φ(∂a/∂x_j, ∂b/∂x_j)` — derivative distributes through branches. **This is the formula that makes Stage A's per-op SCT AD work across branches once Stage B's F2/C1 has flattened them**: the gradient of a φ-merged value equals the φ of per-branch gradients.
- **C3:** C1 with `f` substituted by another `φ` function. Derived form: `φ(x_1, …, x_{i-1}, φ(a, b), x_i, …, x_k) = φ(φ(x_1, …, a, …, x_k), φ(x_1, …, b, …, x_k))` — a φ inside a φ-call distributes outward. Useful for canonicalising nested φs in branches.
- **C4:** C1 then F1. Derived form: when applying C1 produces a φ whose branches are equal, F1 collapses it to that single value. E.g., `f(φ(a, a))` → C1 → `φ(f(a), f(a))` → F1 → `f(a)`. Useful when C1 exposes an algebraic simplification that wasn't visible structurally.

**Dxir implementations:**

```kotlin
// Pseudocode — C1: φ-as-argument distribution
fun applyC1(op: DxirOp): DxirOp? {
    if (op.op !in ELEMENTWISE_OPS) return null
    // Find the first operand that is itself an OpKind.IF
    val ifIdx = op.operands.indexOfFirst { it is DxirOp && it.op == OpKind.IF }
    if (ifIdx < 0) return null
    val ifOp = op.operands[ifIdx] as DxirOp
    val pred = ifOp.operands[0]
    val thenY = ifOp.regions[0].blocks.single().terminator.single()
    val elseY = ifOp.regions[1].blocks.single().terminator.single()
    // Build new IF whose then-branch is `op` with thenY substituted at index ifIdx,
    // and else-branch is `op` with elseY substituted at the same index.
    val newThenOp = op.withOperandReplaced(ifIdx, thenY)
    val newElseOp = op.withOperandReplaced(ifIdx, elseY)
    return /* IF(pred, then=newThenOp, else=newElseOp) with op.types as result */
}

// C2 — handled by Stage A's DxirReverseTransform after Stage B's pass.
// The reverse pass takes the IF-flattened primal, computes adjoints per branch
// (the branches are now straight-line), and the IF's adjoint is itself an IF.
// No separate `applyC2` Kotlin function — C2 is a property of the SCT pass
// composed with B's F2/C1 output.

// C3 — same shape as C1 but with f = phi:
fun applyC3(op: DxirOp): DxirOp? {
    if (op.op != OpKind.IF) return null
    // Same pattern as C1 — find an inner IF in the branch yields and lift.
    return null  // structural detail; ~30 LOC
}

// C4 — apply C1 followed by F1 in one pass:
fun applyC4(op: DxirOp): DxirNode? {
    val c1Result = applyC1(op) ?: return null
    return applyF1(c1Result) ?: c1Result
}
```

**Test shapes:**
- C1: `(if c then a else b) + 1` → `if c then (a+1) else (b+1)`. Numerical at sampled c, a, b.
- C2: tested implicitly through Stage A SCT on the C1/F2-flattened output. Direct test: hand-build primal `if c then x*x else x`, run φ-calculus + DxirReverseTransform, verify gradient is `if c then 2x else 1`.
- C3: `(if c then (if c2 then a else b) else d) + 1` → after C3 flattening then C1, the `if c then …` becomes a single-level `if (c && c2) then a else (if c then b else d)` (with appropriate predicate composition).
- C4: hand-build a case where C1 produces equal branches (e.g., `f(if c then 0 else 0) → if c then f(0) else f(0) → f(0)`). F1 collapses.

**Stability:** structural / branch reordering. No numerical risk.

### 4.8 C5 — Closed form for simple loop **[paper §4 (derived in full); §11.5]**

**Statement:** `loop 𝔏ⁿ d = f(φ_L(p, d)) ⇒ d_exit(L) = f⁽ⁿ⁾(p)`.

**Prose:** A loop whose back-edge is `d ← f(d)` with pre-header `p` and trip count `n` has closed-form `d_exit = f⁽ⁿ⁾(p) = f(f(f(…f(p)…)))` (n applications).

**Dxir rewrite rule:** for a `WHILE` where:
- The condRegion encodes a trip-count predicate (e.g., `i < n` where `i` is the induction variable with init 0 and back-edge `i+1`).
- The bodyRegion has loop-carried value `d` with back-edge `f(d)` where `f` is a dxir subtree that depends on `d` and possibly loop-invariant values, but not on `i`.

Then: compute the trip count `n` symbolically (if the `i < n` bound is available) OR leave `n` as a symbolic parameter. Construct the closed form `f⁽ⁿ⁾(p)` by repeatedly applying `f` `n` times (concrete `n`) OR by invoking `SymbolicEngine.nest(f, p, n)` (symbolic `n`).

```kotlin
// Pseudocode
fun applyC5(op: DxirOp, engine: SymbolicEngine): DxirNode? {
    if (op.op != OpKind.WHILE) return null
    // Detect simple-loop shape: condRegion is `induction < bound`, bodyRegion is `d ← f(d)` + `i ← i+1`.
    val tripCount = extractTripCount(op) ?: return null
    val simpleBody = extractSimpleBody(op) ?: return null  // f : DxirNode → DxirNode
    val preHeader = op.operands[simpleBody.loopCarriedIndex]
    // Lift to symbolic form
    val p = engine.liftNode(preHeader)
    val f = engine.liftFunction(simpleBody.body)
    val closed = engine.nest(f, p, tripCount)
    val simplified = engine.simplify(closed)
    return engine.lowerToDxir(simplified, op.types[simpleBody.loopCarriedIndex])
}
```

**Worked example:**

```
%n = const 5 : i32
%init = param_x : f32
%result = WHILE (%init, %i = 0) {
  cond: (%d, %i) -> { %c = %i < %n; yield %c }
  body: (%d, %i) -> { %d' = %d * 2; %i' = %i + 1; yield %d', %i' }
} : (f32, i32)
  ==> (apply C5 with f = λd. 2d, n = 5)
%result = %init * 32 : f32  // f⁽⁵⁾(init) = 2⁵·init = 32·init
```

**Test shape:**
- Primal: `fun iterate5(x: Float): Float { var d = x; for (i in 0 until 5) d = d * 2; return d }` — or hand-built equivalent dxir.
- After C5: the WHILE is gone; the body is `MUL(init, 32.0f)`.
- Gradient: `d/dx = 32.0`.
- Numerical: for any x, `iterate5(x) == 32x` and `d/dx = 32` — trivially verified at several x values.

**This is the most important test in Stage B.2.** It's the "does symbolic differentiation through a loop even work" pin.

**Stability:** if `f` is `d → d * 2`, `f⁽ⁿ⁾(p) = 2ⁿ · p` is stable. If `f` is `d → d + d·d` (a quadratic recurrence), `f⁽ⁿ⁾(p)` blows up exponentially and Symja may fail to produce a closed form. Bound: C5 requires a closed-form Σ or product under Symja — if `Simplify[Nest[f, p, n]]` doesn't terminate within the time budget, skip C5 for this loop (leave as `WHILE`, fall back to SCT op-by-op reverse mode on the unrolled body — expensive but correct).

### 4.9 C6 — Affine loop recurrence with constant coefficients **[paper §4.3.2 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):**
`𝔏^n_L d = a · φ_L(p, d) + b  ⟹  d_exit(L) = a^n · p + b · Σ_{i=0}^{n-1} a^i`

where `a` and `b` are loop-invariant constants.

**Interpretation:** a loop whose back-edge is `d ← a*d + b` with pre-header `p` has closed form `d_exit = a^n * p + b * Σa^i`. This is the standard closed form for a linear recurrence; the sum simplifies further via the geometric-series identity when `a ≠ 1` to `b * (a^n - 1) / (a - 1)`.

**Load-bearing in BGDHyperOpt (paper Fig. 6c lines 9-10, 14-16):** the outer `while` loop's `w_3` and `w_2` close via C6 — each is a linear recurrence in the loop-entry φ `φ_k(0, w_3)` or `φ_k(0, w_2)` with coefficients expressed in `r`, `M`, `Sxy`, `SX2`. The closed form feeds C7 + F5 downstream.

**Dxir implementation:** detect the affine recurrence pattern `back-edge[i] = MUL(const_a, φ_L_value) + const_b` (with consts loop-invariant), lift to symbolic, invoke `SymbolicEngine.sum` for `Σa^i`, lower back.

```kotlin
// Pseudocode — C6 detection
fun applyC6(whileOp: DxirOp, engine: SymbolicEngine): DxirNode? {
    val (loopCarriedIdx, a, b) = detectAffineRecurrence(whileOp) ?: return null
    val n = extractTripCount(whileOp) ?: return null
    val p = engine.liftNode(whileOp.operands[loopCarriedIdx])
    val aSym = engine.liftNode(a)
    val bSym = engine.liftNode(b)
    val nSym = engine.liftNode(n)
    val closed = engine.add(
        engine.mul(engine.pow(aSym, nSym), p),
        engine.mul(bSym, engine.sum(
            /* λi -> a^i */ engine.opaqueFunction("__geom", 1),
            engine.rational(0), engine.sub(nSym, engine.rational(1)),
        )),
    )
    return engine.lowerToDxir(engine.simplify(closed), whileOp.types[loopCarriedIdx], this)
}
```

**Test:** `fun geom(n: Int, p: Float): Float { var d = p; for (i in 0 until n) d = 2f*d + 3f; return d }` has closed form `d_exit = 2^n * p + 3 * (2^n - 1)`. C6 should produce exactly that.

### 4.10 C7 — Affine loop recurrence with array-indexed offset **[paper §4.3.2 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):**
`𝔏^n_L d = a · φ_L(p, d) + b[i]  ⟹  d_exit(L) = a^n · p + Σ_{i=0}^{n-1} a^i · b[n-1-i]`

where `a` is a loop-invariant constant and `b[i]` is indexed by the iteration counter.

**Interpretation:** C6 generalised — the additive term now depends on the iteration index `i`. Produces a convolution-like closed form summing `a^i · b[n-1-i]`.

**Load-bearing in BGDHyperOpt (paper Fig. 6c lines 1-2, 20-21):** the inner `for` loop `d_3 = φ_i(0, d_3) + 2*x[i]*y[i] - 2*x[i]²*w_2` closes via C7 with `a = 1`, `b[i] = 2*x[i]*y[i] - 2*x[i]²*w_2`, giving `d_3^exit = Σ_i (2*x[i]*y[i] - 2*x[i]²*w_2)`. Same pattern for `e_3`.

**Dxir implementation:** detect affine recurrence with i-indexed offset; lift `b[i]` as `SymFn` of i; invoke `SymbolicEngine.sum`. When `a = 1`, simplifies dramatically (the `Σa^i · b[n-1-i]` collapses to `Σb[i]`).

**Test:** for `a = 1`, `b[i] = i*i` — classic `sum of squares` benchmark. `acc_exit = Σ_{i=0}^{n-1} i²` = `n(n-1)(2n-1)/6`. C7 should produce a form equivalent to that.

### 4.11 C8 — Variable-coefficient affine recurrence **[paper §4.3.2 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):**
`𝔏^n_L d = a[i] · φ_L(p, d) + b[i]  ⟹  d_exit(L) = p · ∏_{i=0}^{n-1} a[i] + Σ_{i=0}^{n-1} b[n-1-i] · ∏_{j=0}^{i} a[n-1-j]`

**Interpretation:** both coefficients now vary with the iteration index. Both C6 and C7 are special cases (C6 when `a[i] = a` and `b[i] = b`; C7 when `a[i] = a`).

**Dxir implementation:** same detection pattern as C6/C7 but both coefficients lift as `SymFn` of i. Symja may or may not close the product `∏_{j=0}^{i} a[n-1-j]` in general — for empirical-distribution coefficient arrays, it probably won't. Fall back to leaving C8 as a `WHILE` when closure fails.

**Test:** `a[i] = i + 1`, `b[i] = i` exercises product-of-integers (factorial-adjacent) forms.

### 4.12 C9 — Power-form recurrence **[paper §4.3.2 + Fig. 5; verified 2026-04-21]**

**Statement (verbatim from paper Fig. 5):**
`𝔏^n_L d = a · (φ_L(p, d))^b  ⟹  d_exit(L) = a^{b+n-1} · p^{b^n}`

(Paper's rendering of the exponent: `d_exit = a^(b+n-1) · p^(b^n)`. This is a multiplicative-by-power recurrence: each iteration multiplies by `a` and raises the loop-entry value to the power `b`.)

**Interpretation:** loop whose back-edge is `d ← a * d^b` with pre-header `p`. The closed form exponent structure shows the characteristic "tower" behaviour: `p` ends up raised to `b^n` while `a` picks up a geometric-series exponent.

**Dxir implementation:** detect `back-edge = MUL(const_a, POW(phi_L_value, const_b))` pattern; lift; use `SymbolicEngine.pow` + algebraic simplification. Symja handles `Power[Power[x, b], n] → Power[x, b^n]` directly.

**Test:** `fun pow2Squared(n: Int, p: Float): Float { var d = p; for (i in 0 until n) d = 3f * d * d; return d }` — so `a = 3`, `b = 2`. Closed form: `d_exit = 3^{2+n-1} * p^{2^n} = 3^{n+1} * p^{2^n}`. At `n = 3, p = 2`: `3^4 * 2^8 = 81 * 256 = 20736`.

**Note:** the paper's Fig. 5 renders C9 compactly; the exponent notation `a^{b+n-1}` may be off-by-one depending on convention — re-verify against the paper's Equation 4.3–4.7 style proof (which they explicitly give for C5, not C9). Flag as minor risk during B.3 implementation.

### 4.13 Stage B formula pipeline — apply order **[revised 2026-04-21 with verified bodies]**

Inside `PhiCalculus.apply(fn: DxirFunction)`:

```
1. normalise(fn)                        // hoist constants out of regions; deduplicate
2. apply F1 everywhere                  // identity (cheap; eliminates trivial IF)
3. apply F3 canonicalisation            // commutative — order branches deterministically
4. apply F2 bottom-up                   // distributive (pushes elementwise into branches)
5. apply C1 bottom-up                   // φ-as-argument distribution (extends F2 for k-ary ops)
6. apply F1 again                       // collapse identity branches that C1 may have exposed
7. apply C3 bottom-up                   // nested-φ flattening
8. apply F1 again                       // collapse what C3 may have exposed
9. apply F5 (loop-exit elimination)     // for single-back-edge WHILE: trivial
10. for each WHILE:
     try C9 first (most specific — power form; rare, but if it matches, no need to try less-specific)
     try C8 (variable-coefficient affine)
     try C7 (constant-a, indexed-b affine)
     try C6 (constant-a, constant-b affine)
     try C5 (simple iteration f^n(p))
     if none apply, leave WHILE in place — the lambda keeps its original call, which fails
     loudly at first invocation (`pluginMissing`; §0.4.446 phrasing correction — there is no
     silent runtime-tape fallback; the Tracer-capture API unrolls value-dependent loops at
     trace time instead)
11. for each IF not eliminated by F1/F2/C1/C3:
     // C2 is implicit — Stage A's reverse-mode AD on the IF-flattened body produces
     // the C2-style adjoint per branch automatically. No explicit C2 call here.
     // C4 is shorthand for "C1 then F1" — already covered by steps 5+6 above.
     leave IF in place (Stage A handles per-branch AD)
12. simplify all arithmetic subtrees via SymbolicEngine.simplify
```

**Notable revisions from the pre-paper-access draft:**
- **F4 dropped from the explicit pipeline** — it's an internal axiom of `SymbolicEngine.nest`, not a standalone dxir-to-dxir rewrite. C5–C9 invoke F4 transitively when unfolding loops.
- **C5 is the *least* specific WHILE pattern**, not the first thing to try. The paper's BGDHyperOpt walkthrough hits C7 + C6 explicitly (Fig. 6c), never bare C5; C5 is the fallback for "back-edge is a unary function of d, no loop-invariant offset". Try C9 → C8 → C7 → C6 → C5 in that order.
- **C2 and C4 don't get standalone passes.** C2 is realised by Stage A SCT running on the C1/F2-flattened output; C4 is "C1 then F1", which the pipeline already does in steps 5+6.

The pipeline is still ordered cheap-to-expensive: structural rewrites (F1, F3, F2, C1, C3) before symbolic-engine work (C5–C9 each invoke `SymbolicEngine.simplify` and possibly `nest`/`sum`/`product`). Step 12's final `simplify` should only run on subtrees that actually changed during the pass (track a "dirty" flag per SSA id).

---

## 5. SymbolicEngine interface

The sole abstraction point between the coarsening pass and the CAS. Every Stage B rule goes through this interface; the underlying implementation is swappable (Symja v0 → custom Kotlin CAS v0.5 → something else) without changing a single line of rule code.

### 5.1 Proposed Kotlin interface

Location: **interface** lives in `ir/src/commonMain/kotlin/io/tlaloc/ir/passes/SymbolicEngine.kt` (no JVM-specific dependencies — opaque sealed handles for expressions + functions). **Symja-backed impl** lives in `ir/src/jvmMain/kotlin/io/tlaloc/ir/passes/SymjaEngine.kt` (Symja is JVM-only). KMP targets other than JVM never see a running [SymbolicEngine] — coarsening is compile-time server-side work, and coarsened artifacts for mobile are baked at publish time per §18 of the spec.

**Status (2026-04-20):** the `SymbolicEngine` interface scaffold shipped this planning session at `ir/src/commonMain/kotlin/io/tlaloc/ir/passes/SymbolicEngine.kt` as the bonus deliverable (§0.4.10 in the spec). Interface-only; no `SymjaEngine` yet.

```kotlin
package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType

/**
 * Opaque handle for a symbolic expression. Implementations wrap Symja's IExpr, a
 * custom Kotlin AST, or any other CAS representation. Callers MUST treat these as
 * immutable — do not mutate fields via reflection.
 */
sealed interface SymExpr

/**
 * Opaque handle for a symbolic function (e.g. a lambda `x → f(x)`). Implementations
 * wrap Symja's Function[...] or a custom representation.
 */
sealed interface SymFn

/**
 * The symbolic engine boundary. All φ-calculus rewrites go through these entry
 * points — the rule code never touches Symja's IExpr type directly.
 */
interface SymbolicEngine {

    /** Version string identifying the backend + plugin version. Used as a cache key component. */
    val version: String

    // --- Lifting dxir to symbolic ---

    /** Lift a straight-line DxirNode subtree to a symbolic expression. */
    fun liftNode(node: DxirNode): SymExpr

    /** Lift a DxirFunction (parameters become free variables, return becomes expr). */
    fun liftFunction(fn: DxirFunction): SymFn

    // --- Constants + variables ---

    /** Construct a symbolic real constant (exact rational). */
    fun rational(numerator: Long, denominator: Long = 1L): SymExpr

    /** Construct a symbolic variable with the given name. */
    fun variable(name: String): SymExpr

    /** Construct an opaque symbolic function (used when lifting a non-closed-form dxir subtree). */
    fun opaqueFunction(name: String, arity: Int): SymFn

    // --- Arithmetic ---

    fun add(a: SymExpr, b: SymExpr): SymExpr
    fun sub(a: SymExpr, b: SymExpr): SymExpr
    fun mul(a: SymExpr, b: SymExpr): SymExpr
    fun div(a: SymExpr, b: SymExpr): SymExpr
    fun neg(a: SymExpr): SymExpr
    fun pow(a: SymExpr, b: SymExpr): SymExpr

    // --- Composition + iteration (C5 + friends) ---

    /** Apply `f` to `x`. */
    fun apply(f: SymFn, x: SymExpr): SymExpr

    /** n-fold composition: `Nest(f, x, n) = f(f(f(...f(x)...)))`. */
    fun nest(f: SymFn, x: SymExpr, n: SymExpr): SymExpr

    /** Closed-form summation: `Sum(f(i), i, 1, n)`. */
    fun sum(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr

    /** Closed-form product: `Product(f(i), i, 1, n)`. */
    fun product(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr

    // --- Differentiation ---

    /** Partial derivative of expr with respect to a variable. */
    fun diff(expr: SymExpr, v: SymExpr): SymExpr

    // --- Simplification ---

    /** Algebraic simplification. Should be idempotent (simplify(simplify(x)) == simplify(x)). */
    fun simplify(expr: SymExpr): SymExpr

    /** Expand polynomial forms. */
    fun expand(expr: SymExpr): SymExpr

    /** Factor out common subexpressions. */
    fun factor(expr: SymExpr): SymExpr

    /** Collect terms in `v`. */
    fun collect(expr: SymExpr, v: SymExpr): SymExpr

    // --- Substitution ---

    /** Substitute `subst[k]` for variable `k` in `expr`. */
    fun substitute(expr: SymExpr, subst: Map<SymExpr, SymExpr>): SymExpr

    // --- Lowering to dxir ---

    /**
     * Emit a dxir subtree that evaluates [expr]. The [outputType] pins the expected dxir
     * type; [builder] receives the emitted DxirOp / DxirConst nodes. Returns the root node
     * of the emitted subtree.
     *
     * Throws [LoweringException] if `expr` involves a symbolic function that has no dxir
     * equivalent (e.g. `Gamma[x]` on a type where gamma isn't registered). Callers should
     * catch and fall back to leaving the original dxir subtree in place.
     */
    fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: io.tlaloc.ir.DxirBuilder,
    ): DxirNode

    class LoweringException(message: String) : RuntimeException(message)
}
```

### 5.2 Symja implementation sketch

```kotlin
// ir/src/jvmMain/kotlin/io/tlaloc/ir/passes/SymjaEngine.kt

package io.tlaloc.ir.passes

import org.matheclipse.core.eval.ExprEvaluator
import org.matheclipse.core.interfaces.IExpr
// ... Symja imports

/**
 * Symja-backed [SymbolicEngine] (v0 per spec §11.7). JVM-only. Thread-safe via the
 * single `evaluator` instance's internal locking.
 */
class SymjaEngine : SymbolicEngine {
    private val evaluator = ExprEvaluator()

    override val version: String = "symja-${Package.getPackage("org.matheclipse.core").implementationVersion}"

    // One sealed impl of SymExpr that wraps IExpr
    private data class SymjaExpr(val inner: IExpr) : SymExpr
    private data class SymjaFn(val inner: IExpr) : SymFn  // Function[{x}, ...]

    override fun liftNode(node: DxirNode): SymExpr = TODO()  // Stage B.1

    override fun liftFunction(fn: DxirFunction): SymFn = TODO()  // Stage B.1

    override fun rational(n: Long, d: Long): SymExpr =
        SymjaExpr(evaluator.eval("Rational[$n, $d]"))

    override fun variable(name: String): SymExpr =
        SymjaExpr(evaluator.eval("Symbol[\"$name\"]"))

    override fun opaqueFunction(name: String, arity: Int): SymFn = TODO()

    override fun add(a: SymExpr, b: SymExpr): SymExpr =
        SymjaExpr(evaluator.eval("(${(a as SymjaExpr).inner}) + (${(b as SymjaExpr).inner})"))

    // ... (sub, mul, div, neg, pow) ...

    override fun apply(f: SymFn, x: SymExpr): SymExpr = TODO()

    override fun nest(f: SymFn, x: SymExpr, n: SymExpr): SymExpr =
        SymjaExpr(evaluator.eval("Nest[${(f as SymjaFn).inner}, ${(x as SymjaExpr).inner}, ${(n as SymjaExpr).inner}]"))

    override fun sum(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr =
        SymjaExpr(evaluator.eval("Sum[${(f as SymjaFn).inner}[i], {i, ${(lo as SymjaExpr).inner}, ${(hi as SymjaExpr).inner}}]"))

    override fun diff(expr: SymExpr, v: SymExpr): SymExpr =
        SymjaExpr(evaluator.eval("D[${(expr as SymjaExpr).inner}, ${(v as SymjaExpr).inner}]"))

    override fun simplify(expr: SymExpr): SymExpr =
        SymjaExpr(evaluator.eval("Simplify[${(expr as SymjaExpr).inner}]"))

    // ... (expand, factor, collect, substitute) ...

    override fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: io.tlaloc.ir.DxirBuilder,
    ): DxirNode {
        // Walk the Symja IExpr tree and emit DxirOps:
        //   Plus[a, b] → OpKind.ADD(a, b)
        //   Times[a, b] → OpKind.MUL(a, b)
        //   Power[a, b] → OpKind.POW(a, b)
        //   Exp[a] → OpKind.EXP(a)
        //   Rational[n, d] → OpKind.DIV(const(n), const(d))
        //   Integer[n] → const(n)
        //   Symbol["name"] → lookup in the name→DxirNode map populated by liftFunction
        //   Sum[f[i], {i, lo, hi}] → emit as closed form if f is closed-form-sum-able,
        //     otherwise throw LoweringException
        //   Nest[f, x, n] → for concrete n, unroll; for symbolic n, throw
        //                   LoweringException (Stage B.2+ may eventually emit a WHILE;
        //                   Stage B.1/B.2 just requires concrete n for test primals)
        TODO("lower IExpr tree to DxirOp subtree")
    }
}
```

### 5.3 Custom CAS fallback — when to write it

Spec §11.7 says "minimal custom CAS v0.5 if we hit Symja limitations." Concrete triggers that warrant the swap:

1. Symja's `Simplify` takes > 60 seconds on the BGDHyperOpt Fig. 6 reduction. Cut to custom CAS (focused polynomial normalisation).
2. Symja produces a `Gamma[...]` / `HypergeometricPFQ[...]` / other transcendental special function that can't be lowered to dxir. Cut to custom CAS (restrict the supported primitive set).
3. Symja's Symbol-vs-Rational semantics produce numerically-different answers from the dxir-interpreter baseline at f32 tolerance. Cut — this is almost certainly a rational-arithmetic bug.
4. Symja's compile-time memory footprint exceeds some threshold (e.g., 4 GB heap on `:compiler-plugin` test tasks). Cut.

§13's risk table weights this as Medium/Medium; the plan budgets 2 weeks for a custom CAS swap if triggered, which matches §18's "4–6 weeks" risk-table estimate for a minimal CAS (we don't need the full 4–6 because the CAS surface is narrow — see §3.2.1's table).

### 5.4 Cache design (ported from §3.2.2)

Cache structure:
```
<project>/.tlaloc-cache/coarsen/
  <hash-prefix-2-hex>/<full-sha256-hex>.bin
```

Entry format (binary):
```
magic:       "TLCRSN\0" (7 bytes)
version:     u32 (format version, start at 1)
engine_ver:  length-prefixed utf-8 string (the SymbolicEngine.version)
key_hash:    32 bytes (sha-256 of the dxir subtree canonical form)
primal_dxir: length-prefixed serialised DxirFunction
gradient_dxir: length-prefixed serialised DxirFunction
reads_primal: u32 bitset (which operand indices the adjoint dereferences)
checksum:    4 bytes CRC-32 over all preceding bytes
```

Serialisation is Stage B.0's homework. Start with `DxirPrinter.pretty()` (human-readable) and parse on read; swap to a binary format (protobuf? k2-serialisation? custom?) as a B.3 optimisation if human-readable turns out to be > 10% of warm-read latency.

---

## 6. Pipeline integration

### 6.1 Where coarsening slots in

Current pipeline, post-§0.4.9:

```
Kotlin source
    │
    ▼ FIR resolution
FirAnonymousFunction (the `grad { x -> … }` body)
    │
    ▼ FirLambdaToDxirLowering
DxirFunction (initial)
    │
    ▼ DxirReverseTransform (Stage A)
DxirFunction (gradient body)
    │
    ▼ DxirToIrSynthesis
IrSimpleFunction
    │
    ▼ IR → bytecode
```

Stage B's insertion:

```
Kotlin source
    │
    ▼ FIR resolution
FirAnonymousFunction (the `grad { x -> … }` body, NOW with if/while handled by B.4)
    │
    ▼ FirLambdaToDxirLowering (widened in B.4)
DxirFunction (with IF + WHILE ops)
    │
    ▼ PhiCalculus.apply                           ← Stage B.1 + B.2 + B.3
DxirFunction (IF/WHILE replaced by COARSENED ops + simplified straight-line)
    │
    ▼ DxirReverseTransform (Stage A, unchanged)
DxirFunction (gradient body; COARSENED ops handled by synthetic VjpRule)
    │
    ▼ DxirToIrSynthesis (unchanged)
IrSimpleFunction
    │
    ▼ IR → bytecode
```

Key architectural decision: **coarsening runs before SCT AD**, not after. This is the paper's decision and matches §11.8's pipeline diagram. The advantage is that coarsening produces a gradient subtree *for each SOI* and splices it as a custom adjoint; the SCT pass then treats the SOI as a primitive with a pre-computed rule.

### 6.1a Paper's splice contract — `setIntermediateAdjoints` (Fig. 8) **[verified 2026-04-21]**

The paper's old DiffKt implementation (§6, Fig. 8) splices coarsened gradients via a **runtime** hook `setIntermediateAdjoints(sequenceOf(operand to gradExpr, ...))` rather than a compile-time op-kind. The mechanism: the coarsened primal expression is computed as a single Tensor expression; a `setIntermediateAdjoints` call appends `(operand → gradient)` pairs that the AD library uses as shortcuts when computing derivatives over the SOI.

Paper's Fig. 8 example (verbatim transcription):
```kotlin
// Original
fun cube() {
    val x = Tensor(5f).asVar()
    val y = Tensor(3f).asVar()
    val w = x - 2f * y
    val v = y * x - x
    val z1 = w * w * w
    val z2 = v * v * v   // SOI = { z1, z2, z = z1 - z2 }
    val z = z1 - z2
    z.backward()
}

// Coarsened
fun cubeTransformed() {
    val x = Tensor(5f).asVar()
    val y = Tensor(3f).asVar()
    val w = x - 2f * y
    val v = y * x - x
    val z = (w * w * w - v * v * v).setIntermediateAdjoints(
        sequenceOf(
            w to w * 2f,    // dz/dw from coarsening (note: paper says "* 2f"; should be 3w² but
                            //                       Fig 8 simplifies for illustration)
            v to v * -2f,   // dz/dv from coarsening
        )
    )
    z.backward()
}
```

(Paper Fig. 8's caption: "If coarsening produces code for differentiating the entire primal code, the compiler simply replaces the calls of the corresponding backward function with the generated code; if the compiler in addition determines that the primal results are used in the program only for getting the derivatives, the compiler removes the invocations of the primal code.")

**Implication for Tlaloc:**

Tlaloc has TWO paths through the AD machinery, not one:

1. **Compile-time SCT path** (Stage A's `DxirReverseTransform` + `DxirToIrSynthesis`). Paper's `setIntermediateAdjoints` doesn't apply — there's no runtime tape to install adjoints into. Stage B's coarsened `DxirFunction` flows straight through SCT; per-op rules in `VjpRegistry` handle anything not closed.
2. **Runtime-tape path** (`:autograd`'s `Backward.kt` + `applyRegistryRule` + the dxir-eval bridge of §0.4.6). Paper's `setIntermediateAdjoints` is a direct fit: a coarsened tape entry could carry pre-computed `(operand_id → gradient_value)` pairs; `Backward.kt` checks for them before falling through to op-by-op math. This is **closer to the paper's mechanism than `OpKind.COARSENED`**.

**Revised splice plan:** Stage B targets the SCT path (closed-form output is a `DxirFunction` straight-line). Stage C considers BOTH paths:
- For SCT-bound primals: `OpKind.COARSENED` op as the spec'd in §6.2 below.
- For tape-bound primals: a `Tape.installAdjoints(operand_ids, gradient_arrays)` API in `:autograd` mirroring the paper's `setIntermediateAdjoints` semantics. The runtime tape's `applyRegistryRule` fast-paths through installed adjoints.

This dual-target structure tracks the paper more closely — the paper's coarsening was a runtime-tape optimisation (DiffKt's `Tensor` AD is operator-overloading-based per §6 of the paper), while ours adds an SCT path on top. Both are legitimate splice points.

**Stage B scope unchanged:** Stage B still produces `DxirFunction` output with closed forms. The splice mechanism choice (op vs runtime hook) is Stage C's problem.

### 6.2 `OpKind.COARSENED` — the splice op

```kotlin
// Added to OpKind.kt in Stage C (not B — Stage B's pass emits straight-line dxir with
// IF/WHILE replaced by closed-forms. The COARSENED op is for when the closed-form
// doesn't exist and we need to keep the SOI as a black box primitive with a custom VJP.)

// Misc (insertion point)
..., COARSENED
```

**Attrs:**
```kotlin
attrs = mapOf(
    "primal_body" to serializedPrimalDxir as ByteArray,         // the simplified primal
    "gradient_body" to serializedGradientDxir as ByteArray,     // the pre-computed VJP
    "reads_primal_operand_indices" to setOf<Int>(0, 2),         // which operands the VJP dereferences
    "cas_version" to "symja-2.0.0-tlaloc-b1",                   // cache-key tag
)
```

**VjpRule for `OpKind.COARSENED`:**

```kotlin
// Added to VjpRegistry in Stage C.
val CoarsenedRule: VjpRule = object : VjpRule {
    override val readsPrimalOperandIndices: Set<Int> =
        setOf()  // ⚠ overridden per-op via attrs — see below

    override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
        val gradientBody = deserializeDxir(op.attrs["gradient_body"] as ByteArray)
        // Inline gradient_body into the caller's builder, substituting:
        //   gradient_body.params[0] = upstream
        //   gradient_body.params[1..N] = op.operands[0..N-1]
        // The inlined body's returns are the per-operand contributions.
        val substituted = inline(gradientBody, upstream, op.operands, builder)
        return op.operands.zip(substituted).toList()
    }
}
```

The `readsPrimalOperandIndices` static `emptySet()` is wrong for COARSENED. We need **per-op** configurability because different SOIs dereference different operand indices. Proposed extension to the `VjpRule` interface — but a simpler workaround: `DxirReverseTransform.computeUsedByAdjoint` special-cases `OpKind.COARSENED` and reads `attrs["reads_primal_operand_indices"]` directly. Minimal change, no interface churn. Document the special-case at `DxirReverseTransform.kt` (the only file that consumes `readsPrimalOperandIndices`).

### 6.3 Stage B doesn't need `OpKind.COARSENED` — but Stage C does

Stage B's pass emits simplified dxir with F1/F2/C5/etc. applied **structurally**. The output is still straight-line dxir (the IF/WHILE has been replaced by a closed-form algebraic expression over scalars). Stage A's existing `DxirReverseTransform` handles this with no changes.

Stage C introduces SOIs — regions whose full coarsening either:
- Succeeded (symbolic → closed form → straight-line dxir) — same as Stage B.
- Partially succeeded (symbolic simplification produced a smaller form, but the form still references some un-closed-over loops) — emit a `COARSENED` op wrapping the original SOI with a pre-computed gradient from the symbolic form.
- Failed (Symja couldn't close; keep the original IF/WHILE). In this case: Stage A still can't handle the ops with regions — fall back to **runtime-tape AD** for the uncoarsened SOI, matching §0.4.6's tape path. This is where the coarsening pass's `report.estimatedSpeedup` (§11.10) comes from — it's 1.0× for any region that fell back.

Stage B is responsible for the first bullet only. Stage C orchestrates all three.

### 6.4 Interaction with `usedByAdjoint`

Already addressed in §3.5 and §6.2. The existing `DxirReverseTransform.computeUsedByAdjoint` walks `VjpRegistry[op.op]?.readsPrimalOperandIndices`; for `COARSENED` we special-case reading `attrs["reads_primal_operand_indices"]`. Three lines of code plus a doc-comment.

### 6.5 Interaction with `DxirInterpreter` (the testing substrate)

Stage B's test assertion shape (§3.5):
```kotlin
val primal = DxirBuilder.function("f") { … }
val coarsened = PhiCalculus.apply(primal)
val sctPrimal = DxirReverseTransform.apply(primal)
val sctCoarsened = DxirReverseTransform.apply(coarsened)

// Forward equivalence
assertAgree(
    DxirInterpreter.evalFunction(primal, sampleInputs),
    DxirInterpreter.evalFunction(coarsened, sampleInputs),
    tolerance = 1e-5f,
)

// Gradient equivalence
assertAgree(
    DxirInterpreter.evalFunction(sctPrimal, sampleInputs),
    DxirInterpreter.evalFunction(sctCoarsened, sampleInputs),
    tolerance = 1e-5f,
)
```

For this to work on primals with IF/WHILE, the `DxirInterpreter` needs to handle IF/WHILE (§3.1.3). That's Stage B.0 homework.

### 6.6 Interaction with `FirLambdaToDxirLowering`

Current lowering ([`FirLambdaToDxirLowering.kt`](../compiler-plugin/src/main/kotlin/io/tlaloc/plugin/FirLambdaToDxirLowering.kt)) rejects control-flow via the `else throw LoweringException("unsupported expression ${expr::class.simpleName}")` branch at line 112. FIR's `FirWhenExpression` (Kotlin `when`/`if`) and `FirLoop` (Kotlin `while`/`do-while`/`for`) fall into this branch.

**B.4 widens the lowering** to emit `OpKind.IF` and `OpKind.WHILE`:
- `FirWhenExpression` with a single boolean condition + then-branch + else-branch → `OpKind.IF` with predicate + two regions.
- `FirWhenExpression` with multiple subject-less branches (`when { a -> …; b -> …; else -> … }`) → desugar to a chain of `IF`s.
- `FirWhileLoop` / `FirForLoop` → `OpKind.WHILE` with a trip-count-encoded condRegion + a bodyRegion.
- `FirDoWhileLoop` → `OpKind.WHILE` with a first-iteration-always-runs variant (either: invert to a standard while + prepend one iteration, OR introduce `OpKind.DO_WHILE` — recommend the former).

Scope: B.4 handles `if/else`, `while`, `for (i in 0 until n)` bodies, `do-while`. It does NOT handle:
- `break` / `continue` (needs multi-back-edge WHILE or a `break` op — defer).
- `return` inside a branch (FIR non-linearises returns; complicated).
- Mutating captured variables (closure capture of `var` — not even in Stage A scope, and probably not for Stage B either).

**B.4's size estimate:** ~400 LOC of FIR handling + ~20 structural tests. Not small, but self-contained within `FirLambdaToDxirLowering`.

---

## 7. Stage B → B.N sub-staging

Stage B is a multi-month project (paper's target: ~3 person-months; realistic: more — see §14). Break into five sub-milestones, each with its own DoD, test set, LOC estimate, and expected session count.

### 7.1 B.0 — Prerequisites — split into B.0a (IR widening) + B.0b (Symja + bake-off) **[split decision 2026-04-21]**

The original plan bundled IR widening + SymbolicEngine + Symja bake-off into one B.0 sub-milestone. This was split per the post-§0.4.11 decision to isolate IR-widening risk from CAS-choice risk: hand-built IF/WHILE primals through the interpreter may expose a constraint the SymbolicEngine interface didn't anticipate, and we want to know that *before* committing to Symja's surface. The split is also cheaper to land — each sub-milestone is one focused session instead of one bundled session that mixes concerns.

#### 7.1.a B.0a — IR widening (no symbolic engine work yet)

**DoD:**
- [ ] `OpKind.IF`, `OpKind.WHILE` added to [`OpKind.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/OpKind.kt).
- [ ] `DxirBuilder.ifOp(...)`, `DxirBuilder.whileOp(...)` convenience builders in [`DxirModule.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/DxirModule.kt).
- [ ] `DxirFunction` ref-integrity check widens to validate WHILE block-arg shape (§3.1.3 of this plan: condRegion has `n` args yielding 1 Bool predicate, bodyRegion has `n` args yielding `n` outputs matching `op.types`).
- [ ] `DxirInterpreter.evalOp` extended with IF + WHILE arms ([`DxirInterpreter.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/DxirInterpreter.kt)). Single-back-edge WHILE only; no `break`. Trip count must terminate (interpreter caps at 10⁶ iterations and errors loudly to catch infinite loops in malformed test primals).
- [ ] Stub `:stablehlo` emission with `error("OpKind.IF/WHILE emission deferred to post-Stage-B")` — the stablehlo round-trip tests don't exercise these ops yet, so the stubs never fire in CI.
- [ ] 6–10 new `:ir` tests: 3–4 structural (DxirFunction validation rejects malformed IF/WHILE shapes), 3–6 interpreter (`evalFunction` over hand-built IF/WHILE primals matches expected values; covers a simple-iter loop, a nested IF, an empty-trip-count WHILE, and a multi-loop-carried WHILE).
- [ ] `./gradlew build` stays green. Spec §0.4 gets a §0.4.12 entry (B.0a closure).

**Non-goals in B.0a:**
- Any SymbolicEngine implementation (deferred to B.0b).
- Any Symja dependency declaration (deferred to B.0b).
- Any φ-calculus rewrite code (deferred to B.1+).
- StableHLO emission for IF/WHILE (deferred post-Stage-B).
- FIR-side lowering of Kotlin `if`/`while` (deferred to B.4).

**Test set:** `:ir` 334 → 340–344 (+6 to +10).

**LOC estimate:** 250–350 LOC. IR widening is ~150 LOC; interpreter arms are ~80 LOC (IF: ~10, WHILE: ~70 with iteration cap); tests are ~120 LOC.

**Expected session count:** 1 session.

**Hand-off at end of B.0a:** a working `OpKind.IF`/`WHILE` surface that an engineer can use to hand-build loop primals in a test, run through `DxirInterpreter.evalFunction`, and verify numerical results. **The interpreter is the testing substrate for everything Stage B does** — landing it standalone, before any symbolic-engine work, lets future Stage B steps validate themselves end-to-end the moment they land.

**Why split this from B.0b:** the IR widening is a self-contained refactor whose correctness can be validated without any CAS at all. If the WHILE shape we're proposing turns out to be inadequate (e.g., we discover Stage B.2's C5 detection needs explicit induction-variable annotation, or that multi-back-edge WHILEs are needed sooner than C9), we revise the IR shape *before* writing Symja-coupled code that depends on it. Cheaper to revise IR before downstream consumers exist.

#### 7.1.b B.0b — SymbolicEngine + Symja + adequacy bake-off

**Prerequisite:** B.0a green.

**DoD:**
- [ ] Symja Maven coordinate declared in [`ir/build.gradle.kts`](../ir/build.gradle.kts) (`org.matheclipse:matheclipse-core`, latest stable version pinned in `libs.versions.toml`). JVM target only — `commonMain` stays Symja-free since [SymbolicEngine.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/SymbolicEngine.kt) is interface-only.
- [ ] `SymjaEngine` lands at `ir/src/jvmMain/kotlin/io/tlaloc/ir/passes/SymjaEngine.kt` with **real bodies** (not `TODO()`) for: `rational`, `variable`, `add`/`sub`/`mul`/`div`/`neg`/`pow`, `apply`, `nest`, `sum`, `product`, `diff`, `simplify`, `expand`, `factor`, `collect`, `substitute`, plus a first-cut `liftNode`, `liftFunction`, and `lowerToDxir`. Real implementations, not stubs — but their correctness is exercised through the bake-off, not (yet) through PhiCalculus tests.
- [ ] Thread-safety wrap: every Symja `evaluator.eval(...)` call is inside a `synchronized(evaluator)` block (§13 risk #13). Document at the engine's class doc-comment.
- [ ] **Symja adequacy bake-off (§3.2.3)** runs as a one-shot harness in `:ir` test code. Specifically:
    - (a) Construct paper Fig. 6's BGDHyperOpt closed-form expression symbolically through `SymbolicEngine` (i.e., apply C7 + F5 + C6 + F2 manually as `SymbolicEngine` calls, not yet through a `PhiCalculus.apply` orchestrator). Verify it produces a closed form within **60 seconds** wall clock.
    - (b) Verify the closed form matches the paper's Fig. 6c line 26 (`err = (Sy² − 2Sxy·w_4 + Sx²·w_4²)^0.5 / M`) by running both through `SymbolicEngine.substitute` at concrete x/y/M values and comparing to the explicit expression.
    - (c) Run `SymbolicEngine.diff(err, r)` on the closed form to compute `d(err)/dr` (the learning-rate gradient). Verify it produces a closed form within **30 additional seconds**.
    - (d) Compare the symbolic gradient against finite-differencing the original loop-form primal (`(err(r+h) - err(r-h)) / (2h)` at `h = 1e-4`) — match to within `1e-3` relative error.
- [ ] Bake-off result documented as a single page in `docs/papers/symja-bakeoff-2026-04.md`: timing per step, the closed-form expression Symja produced, the gradient expression, and the pass/fail decision.
- [ ] `./gradlew build` stays green. Spec §0.4 gets a §0.4.13 entry (B.0b closure with bake-off result).

**Non-goals in B.0b:**
- `PhiCalculus.apply` (deferred to B.1).
- Any IF / WHILE rewrite (deferred to B.1+).
- Caching infrastructure (§5.4 of plan; deferred to B.3).

**Test set:** `:ir` 340–344 → 343–348 (+3 to +4 SymbolicEngine API-shape tests; the bake-off itself is one harness test, not 10).

**LOC estimate:** 400–600 LOC. SymjaEngine implementation is ~300 LOC (most operations are 1-3 line `evaluator.eval` calls); `lowerToDxir` IExpr-tree walker is ~150 LOC; bake-off harness + reference Fig. 6 expression construction is ~150 LOC.

**Expected session count:** 1 session if Symja behaves; +1 contingency if a Symja API quirk needs working around.

**Hand-off at end of B.0b:**
- **If bake-off passes:** SymbolicEngine confirmed adequate for the paper's hardest single benchmark. Green light for B.1. Decision recorded in §0.4.13.
- **If bake-off fails:** decision point. Three options: (i) tune Symja (`SetOptions[Simplify, …]`, exact rational mode, etc.) and re-run; (ii) cut to custom Kotlin CAS — adds ~2-4 weeks scope to Stage B.2 per §13 risk #2; (iii) descope (drop C5–C9 from Stage B; ship a F1–F5 + C1–C4 only "Stage B'" — Tlaloc would lose the closed-form-loop wins but keep the closed-form-branch wins, ~40% of paper speedup). The bake-off doc records the trigger + the chosen path.

**Why split this from B.0a:** the bake-off is **the single highest-value Stage B-level decision point**. If Symja can do BGDHyperOpt's symbolic closed form, every other corollary Stage B implements is downstream of that capability. If Symja can't, no amount of clever Kotlin in B.1/B.2/B.3 saves the project. Landing B.0b standalone — after B.0a is proven — means the bake-off result drives the rest of Stage B's scope, instead of being entangled with IR-widening commits if both lived in the same session.

**Total B.0 (B.0a + B.0b):** 2 sessions, +10 to +14 tests, ~700-950 LOC. Same envelope as the original B.0 estimate; the split is purely organisational.

### 7.2 B.1 — F1 + F2 + F3 (arithmetic-free formulae)

**DoD:**
- [ ] `ir/src/commonMain/kotlin/io/tlaloc/ir/passes/PhiCalculus.kt` lands with entry point `PhiCalculus.apply(fn: DxirFunction): DxirFunction`. Initially applies only F1, F2, F3.
- [ ] F1 (identity) rewrite: for any `OpKind.IF` whose then/else yields are structurally equal, replace with that value.
- [ ] F2 (distributive) rewrite: for any elementwise op whose operand is an IF, push the op into each branch. **Include the anti-swell gate from §4.3** (only apply when IF has a single use OR both branches are < 5 ops).
- [ ] F3 (commutative) rewrite: implement both candidate readings behind a flag, test each, commit the matching one.
- [ ] SymbolicEngine is implemented only for the subset F1/F2/F3 need — which is basically nothing, since F1/F2/F3 are structural. `SymjaEngine` bodies for `simplify` still TODO.
- [ ] 15–20 new structural tests: each formula has a "positive" test (the transformation fires) and a "negative" test (the transformation doesn't fire when preconditions fail). Plus 2–3 numerical-equivalence tests running the pre- and post-rewrite dxir through `DxirInterpreter.evalFunction`.
- [ ] One end-to-end test: `grad { x: Float -> (if (cond) x * x else x) + 1 }` produces the correct gradient via:
   1. Hand-built dxir primal (since FIR lowering of `if` isn't live yet — that's B.4).
   2. PhiCalculus.apply → flattens to straight-line.
   3. DxirReverseTransform.apply → gradient body.
   4. DxirInterpreter.evalFunction → numerical gradient.
   5. Compare against manual f'(x) at cond=true and cond=false.

**Non-goals:**
- F4, F5, C5, C6, C7, C8, C9 — those are B.2 and B.3.
- FIR-side `if` lowering — B.4.
- SOI identification — Stage C.

**Test set:** 342 → 362 (+20 tests).

**LOC estimate:** 800–1000 LOC. F2 is the big one because of the anti-swell gate + the combinatorics of distributing elementwise ops into branches (unary, binary with LHS-is-IF, binary with RHS-is-IF, binary with both sides IF). F1 and F3 are each 50 LOC.

**Expected session count:** 3–4 sessions. F1 + pipeline skeleton in session 1; F2 in session 2; F3 with A/B reading test in session 3; numerical-equivalence end-to-end test in session 4.

**Hand-off at end of B.1:** PhiCalculus.apply works on hand-constructed primals containing IFs. F4/F5/C5 are the next step.

### 7.3 B.2 — F4 + F5 + C5 (loop-entry, loop-exit, simple loop closed-form)

**DoD:**
- [ ] F4 (loop-entry) rewrite: nested WHILEs with shared pre-header flatten.
- [ ] F5 (loop-exit) rewrite: WHILE whose back-edge is data-independent of the loop body simplifies to its final value. **[Needs paper verification on the exact body before shipping.]**
- [ ] C5 (simple loop closed-form): WHILE whose back-edge is `d ← f(d)` (f not dependent on induction var) simplifies to `d_exit = f⁽ⁿ⁾(p)` via `SymbolicEngine.nest(f, p, n)` + `simplify`.
- [ ] SymbolicEngine's `liftNode`, `liftFunction`, `lowerToDxir`, `nest`, `simplify` all implemented (Symja default). Caching still TODO (B.3 lands it).
- [ ] Hand-built test primals for each of F4, F5, C5 — structural + numerical-equivalence.
- [ ] **The load-bearing test: `grad { x: Float -> iterate5(x) }`** where `iterate5(x) = x · 32` after C5, `d/dx = 32`. This is the "does symbolic differentiation through a loop work" pin.

**Non-goals:**
- C6, C7, C8, C9 — B.3.
- FIR-side `while` lowering — B.4.
- BGDHyperOpt end-to-end (paper Fig. 6) — Stage D.

**Test set:** 362 → 395 (+33 tests). F4 needs ~5 structural tests; F5 needs ~8 (including paper-verification scaffolding); C5 needs ~15 (several simple-loop test cases × several trip counts × structural + numerical).

**LOC estimate:** 1200–1500 LOC. F4 is 100 LOC (structural detector); F5 is 200 LOC (harder pattern match); C5 is 400 LOC (engine integration + trip-count extraction + lowering the closed form back to dxir). SymbolicEngine implementation is 500–800 LOC (Symja integration + liftNode + lowerToDxir + the tricky `Nest`/`Simplify` orchestration).

**Expected session count:** 4–6 sessions. The SymbolicEngine implementation is a multi-session sub-project by itself.

**Hand-off at end of B.2:** the pass can handle if/else branches (via F1/F2/F3) and simple while loops (via F4/F5/C5). BGDHyperOpt's `while` → closed-form reduction is *almost* possible — the `while` with a data-dependent termination (C9) still needs the remaining corollaries.

### 7.4 B.3 — C1–C4, C6, C7, C8, C9 (the remaining corollaries)

**DoD:**
- [ ] **[Blocker 1 resolution]** Paper access secured; each of C1–C4, C6–C9 extracted verbatim from the paper.
- [ ] C1–C4 dxir rewrite rules implemented. Each has a structural + numerical-equivalence test.
- [ ] C6 (loop summation): WHILE with accumulator `acc ← acc + f(i)` lifts to `Sum[f[i], {i, 1, n}]` and simplifies.
- [ ] C7 (loop product): WHILE with accumulator `acc ← acc * f(i)` lifts to `Product[f[i], {i, 1, n}]` and simplifies.
- [ ] C8 (mixed summation-product): `acc ← acc + f(i) * g(i)` lifts to `Sum[f[i] * g[i], …]`.
- [ ] C9 (irregular loops): while with data-dependent termination; loop-exit value depends on the first `i` satisfying the predicate. [Needs paper body verification.]
- [ ] Caching infrastructure (§5.4) lands: cache read + write + invalidation + pruning task.
- [ ] BGDHyperOpt's Fig. 6 walkthrough runs as a full end-to-end test: hand-built primal dxir containing `while` + `if` + `for`, pass through PhiCalculus.apply, verify the closed form agrees with op-by-op interpretation.

**Non-goals:**
- FIR-side lowering — B.4.
- Benchmarks — Stage D.

**Test set:** 395 → 440 (+45 tests). C1–C4 each has 3–5 tests. C6–C9 each has 5–8 tests. Plus 5–8 caching tests. Plus 1 BGDHyperOpt Fig. 6 integration test.

**LOC estimate:** 1500–2000 LOC. C1–C4 are each ~150 LOC (structural rewrite + substitute); C6, C7, C8, C9 are each ~250 LOC (symbolic-engine-backed). Caching is ~400 LOC (serialisation + hash + read-write).

**Expected session count:** 5–8 sessions. Paper verification + per-corollary implementation + testing.

**Hand-off at end of B.3:** the φ-calculus pass is complete for F1–F5 + C1–C9. Hand-built primals covering every formula + paper Fig. 6 all pass. Stage B's core algorithmic contribution is landed.

### 7.5 B.4 — FIR-side lowering of control flow

**DoD:**
- [ ] `FirLambdaToDxirLowering` extended to handle `FirWhenExpression` → `OpKind.IF`.
- [ ] Same extended for `FirWhileLoop` + `FirForLoop` + `FirDoWhileLoop` → `OpKind.WHILE`.
- [ ] End-to-end user-code tests: `grad { x: Float -> if (x > 0) x * x else -x }` produces the correct gradient at both cond=true and cond=false via full IR rewrite path (not runtime tape).
- [ ] `grad { x: Float -> var d = x; for (i in 0 until 5) d = d * 2; d }` → compile to a constant `32.0f` gradient (C5).
- [ ] BGDHyperOpt-style Kotlin source (hand-written, matching the paper's kernel structure) compiles and produces the closed-form gradient.
- [ ] Tests in `:compiler-plugin` exercising the full pipeline.

**Non-goals:**
- `break` / `continue` — defer post-B.4.
- Mutation of captured `var`s in closures — not supported by Stage A either.

**Test set:** 440 → 465 (+25 tests).

**LOC estimate:** 500–700 LOC. FIR's `FirStatement` traversal is well-understood from Stage A; extending to handle loops / whens is structural but the details (SSA-ing a `var` modified in a loop, extracting trip counts from `for`-ranges, etc.) are fiddly.

**Expected session count:** 3–4 sessions.

**Hand-off at end of B.4:** user-written Kotlin code with `if/else` and `while` loops can be `grad`-ed end-to-end through the compiler plugin. The paper's Fig. 6 BGDHyperOpt kernel ported to Kotlin produces the coarsened closed-form gradient at compile time.

### 7.6 Total Stage B budget **[revised 2026-04-21 with B.0 split]**

| Sub-milestone | Sessions | Test delta | LOC delta |
|---|---|---|---|
| B.0a (IR widening) | 1 | +6 to +10 | ~250–350 |
| B.0b (Symja + bake-off) | 1 (+1 contingency) | +3 to +4 | ~400–600 |
| B.1 (F1 + F2 + F3 + C1 + C3) | 3–4 | +20–25 | ~900 |
| B.2 (F5 trivial + C5) | 4–6 | +33 | ~1300 |
| B.3 (C6/C7/C8/C9 + caching + Fig. 6 e2e) | 5–8 | +45 | ~1700 |
| B.4 (FIR-side `if`/`while` lowering) | 3–4 | +25 | ~600 |
| **Total** | **17–24** | **+132–142** | **~5150–5450** |

Same envelope as the pre-§0.4.11 estimate; the B.0 split is purely organisational. B.1 absorbed C1 and C3 from §4.7 (verbatim bodies in §0.4.11 made them ready for B.1 instead of B.3); B.2 lost C5's complexity-blocker since §0.4.11 confirmed C5 is the *least* specific WHILE pattern (try last, not first); B.3 gained explicit "Fig. 6 e2e" as a DoD item.

At 2 sessions per week (the observed cadence for §0.4.3–§0.4.9), this is **8.5–12 weeks** of elapsed calendar time, or **~17–24 sessions** of agent work. Rev the §14 timeline accordingly.

**Important observation:** the spec's "3 person-months" estimate in §11.11 was written assuming coarsening lands *as a single bundle* after the K2 plugin + dxir foundation. The substrate is now in place (Stage A closed), but the spec didn't budget for B.0's IR widening or B.4's FIR lowering as separate line items. Those together are 5 sessions / ~1100 LOC — not trivial. A realistic Stage B budget is 10–12 weeks full-time, not 12 weeks as the spec's Month-9 deadline implies.

---

## 8. Stage C preview — SOI identification

**Scope:** port paper Fig. 7's reuse-aware SOI identification algorithm to dxir. Glue it to `PhiCalculus.apply` so the pass finds SOIs automatically (instead of Stage B's hand-constructed ones).

**Input:** `DxirFunction` with structural control flow already lowered to IF/WHILE ops.
**Output:** the same function with a subset of its body replaced by `OpKind.COARSENED` ops, each wrapping an SOI that was identified + symbolically differentiated.

### 8.1 The algorithm — paper Fig. 7(a) verbatim **[verified 2026-04-21]**

Paper Fig. 7(a):

```
Inputs:
  L: the upper limit of the size of an SOI
  f: a function in SSA
  S: the set of active sink variables in f
  SOI: the placeholder of all SOIs of f

 1. SOI = { }
 2. for each s in S
 3.   C = f.getDefUseChain(s)
 4.   T = f.getRegionTree(C)
 5.   W: a worklist with all nodes in T added in
 6.        bottom-up order
 7.   while (W.notEmpty)
 8.     n = W.removeANode( )
 9.     if (n.hasLargeChildren( ))
10.       n.markLarge( )
11.       n.mergeSomeChildren( )
12.       SOI.add(all small children in n)
13.       next;
14.     e = n.getSymbExp( )
15.     if (e.size > L)
16.       n.markLarge( )
17.       if (n.isLeaf( ))
18.         newNodes = n.splitOnReuses( )
19.         W.addToFront(elements in newNodes)
```

**Paper prose (§5 paragraphs after Fig. 7):** "The SOI identification algorithm then traverses the region tree in a bottom-up order, as outlined in Figure 7(a). For each node, if it has no large child node (i.e., exceeding the SOI size limit), its symbolic expression is derived through the 𝜙-calculus. If the size of the derived expression exceeds the SOI size limit, that node is marked as a large node, and if it is a leaf node, it is split into two smaller nodes; the splitting point is chosen to be the variable that is contained in that node and has the largest number of references (and hence reuses) in `f`. The two new nodes created by the split are added to the front of worklist. If the current node has large children, there is no need to go up further in the def-use region tree as the upper nodes can only become even larger. In that case, the algorithm examines the immediate children of this node, merge consecutive small children nodes (up to the SOI size limit); after that, it puts each of the children nodes smaller than the limit as an SOI. The algorithm continues until worklist becomes empty. The size limit `L` can be empirically selected based on the machine and the symbolic engine."

**Def-use region tree (paper §5, two paragraphs above Fig. 7):** "Def-use region tree is a data structure inspired by the classic code region hierarchy in compilers. Traditionally, a code region is defined as a collection of nodes `N` and edges `E` such that (i) a header node `h` in `N` dominates all other nodes in the collection; (ii) if `p` is in `N`, then `m` must be in `N` if `m` reaches `p` without going through `h`; (iii) `E` includes all edges between nodes in `N`, except for those that enter `h`. ... Def-use region tree has two major differences from code region hierarchy: (i) only relevant variable definitions are considered; (ii) every loop-exit 𝜙 function is put as part of the region of the associated loop. The second property is for convenience in the derivation of symbolic expressions for loops."

**Optimization problem (paper §5, Definition 2 verbatim):**
> Let `G` be a series of computations, `l` be the upper limit of the allowed sizes of an SOI, `P` be the set of valid partitions of `G`, that is, for any partition `S` in `P`, no element in `S` is larger than `l`. The problem is to find the optimal partition `S* ∈ P` such that the total running time is minimized, that is,
>
> `∀Q ∈ P, ad(S*) + Σ_{s ∈ S*} compute(dif(s)) ≤ ad(Q) + Σ_{q ∈ Q} compute(dif(q))`
>
> where `compute(dif(x))` is the amount of computation involved in running the symbolically differentiated code segment for code `x`, and `ad(X)` is the cost of the remaining AD differentiation of `X` after symbolic differentiation.

**Search-space size:** for SSA with `N` instructions and SOI size limit `l`, the number of valid partitions is between `l^(N/l)` and `l^N`. The paper notes this exponential space + the difficulty of static perf modeling makes exact optimisation infeasible; the algorithm above is a tractable heuristic.

**Three factors the paper identifies as relevant (§5):**
1. `ad(X)` cost — incurred at SOI boundaries; calls for *larger* SOIs (fewer boundaries).
2. Computation simplifications — cancellable computations need to be in the same SOI to cancel; calls for *larger* SOIs.
3. Computation reuse — values reused across SOIs are exploitable via the chain rule's natural reuse pattern; *limits* SOI size.

The algorithm balances these by capping at `L` and splitting on most-reused variable when a leaf's symbolic expression swells past `L`.

### 8.1a Dxir-specific structural notes

- `f.getDefUseChain(s)` — walk the dxir body once, build `Map<Int, List<Int>>` keyed by SSA id → list of consumer op ids. O(body size). Already a one-pass walk Tlaloc has the substrate for in `:ir` (no analogue exists today; new code in Stage C.1).
- `f.getRegionTree(C)` — regions are naturally nested by `DxirOp.regions`. A top-level body is the root; each IF/WHILE is a subregion. Leaves are straight-line sequences of ops with no nested regions. Per the paper's "second property", every loop-exit φ is bundled with its associated WHILE.
- `n.getSymbExp()` — invoke `PhiCalculus.coarsen(subtree, engine)` (the interface §8.3 names) and read the size of the returned closed form (in dxir node count, since Symja IExpr count and dxir count are within a small constant factor for closed-form scalar expressions).
- `n.splitOnReuses()` — pick the SSA id `v` in `node.freeVariables` with the most uses in `f` (per the def-use chain). Split into (a) the sub-graph computing `v`, (b) the sub-graph consuming `v`. Each becomes a new tree node, both added to worklist front.
- `n.mergeSomeChildren()` — merge consecutive small children whose combined symbolic expression size stays under `L`. Use the def-use ordering as the merge order (children appearing earlier in `body` come first).

### 8.2 Cost model — default L

Paper's recommendation: `L = 50` expression nodes as a starting point, tuned per-machine and per-CAS. Stage C's DoD should start with `L = 50` as a `jit(coarsenBudget = 500)` default maps to roughly L = 500/10 = 50 per internal unit — adjust as the ratio becomes clear.

### 8.3 Interface Stage B exposes for Stage C

```kotlin
object PhiCalculus {
    /**
     * Apply φ-calculus F1–F5 + C1–C9 to a DxirFunction. Tries to close structured
     * control flow into closed-form straight-line expressions. Returns the simplified
     * function; unchanged regions stay in place.
     */
    fun apply(fn: DxirFunction, engine: SymbolicEngine): DxirFunction

    /**
     * Coarsen a specific subtree (not the whole function). Returns either:
     *  - CoarsenResult.Success(simplifiedSubtree, gradientSubtree, readsOperands) —
     *    the subtree collapses to a closed form; Stage C replaces it with a
     *    OpKind.COARSENED op carrying the gradient.
     *  - CoarsenResult.Failure(reason) — the subtree didn't close; leave it as-is.
     */
    fun coarsen(subtree: DxirNode, engine: SymbolicEngine): CoarsenResult
}
```

Stage C calls `PhiCalculus.coarsen(subtree, engine)` per identified SOI. Stage B lands `apply`; `coarsen` is a slightly lower-level interface (works on a subtree, not a whole function) and is a Stage C addition.

### 8.4 Stage C substaging (one-line each)

- **C.1:** def-use region tree + SOI worklist skeleton. No symbolic work yet; hand-verify leaf identification.
- **C.2:** size-limit + split-on-reuse logic. Tested against synthetic IR graphs with known optimal SOIs.
- **C.3:** integration with `PhiCalculus.coarsen`; end-to-end "find SOIs, coarsen, splice" on BGDHyperOpt-like primals.
- **C.4:** cost-model tuning; measure per-benchmark what L maximises speedup without regressing compile time.

---

## 9. Stage D preview — benchmark bring-up

**Scope:** port the six OOPSLA 2021 benchmarks, run them through the Tlaloc compile path (with + without coarsening), measure wall-clock and op-count, compare against PyTorch 2.x `torch.compile` and JAX + `jit`.

### 9.1 The six benchmarks (ordered by best-first-port)

| # | Benchmark | Domain | Paper speedup (diff) | Paper speedup (e2e) | Port difficulty | Control-flow structure |
|---|---|---|---:|---:|---|---|
| 1 | BGDHyperOpt | Meta-learning | 23×–27× | 8.0×–8.6× | Medium | while + for + if (+ break) |
| 2 | HookeanSpring | Physics | 2.5×–6.6× | 4.1×–11.0× | Medium | sequential vector ops, no loops |
| 3 | HMC | Probabilistic | 2.5×–4.4× | 2.3×–3.6× | Hard | nested loops + log1p pattern |
| 4 | Brachistochrone | Math physics | 1.0×–1.4× | 1.8×–2.5× | Easy | summation over segments |
| 5 | CartPole | Deep RL | 1.1× | 1.1× | Medium | NN forward + env update; small SOI |
| 6 | QWOP | Game AI | 1.4×–1.5× | 1.4×–1.6× | Hard | 225-line fn with 13 loops + if-else |

**Recommend port order:**
1. **Brachistochrone** — lowest complexity (summation loop), easy first port, validates Stage B.3's C6 on a real benchmark.
2. **HookeanSpring** — no control flow, highest overall speedup, validates Stage B.1's F2 + Stage A SCT on realistic vector ops.
3. **BGDHyperOpt** — the paper's headline benchmark; the Fig. 6 walkthrough is already part of Stage B.3's DoD.
4. **HMC** — depends on numerical-stability log1p pattern (§12), validates the stability pattern library.
5. **QWOP** — large, stress-tests SOI identification (Stage C) with 13 loops.
6. **CartPole** — lowest paper speedup (1.1×); useful as a "we don't make things slower" regression test.

### 9.2 Stage D DoD (per §11.12)

Status as of §0.4.49 (Stage D.3 partial):

- [~] All six benchmarks ported to Tlaloc-Kotlin. **(2 of 6 full + 1/6 partial)** — Brachistochrone (§0.4.43), HookeanSpring (§0.4.47) full source-level ports; BGDHyperOpt (§0.4.49) 2/4 sub-kernels via packing trick, full port blocked on FIR nested-for + while + break (§0.4.20 has hand-built `:ir` version). HMC / CartPole / QWOP pending.
- [ ] Measured speedup vs. non-coarsened Tlaloc within 20% of paper's measured speedups. **(unmeasured)** — D.1e is the planned wall-clock harness session (see §0.4.44 when it lands).
- [ ] Measured speedup vs. PyTorch 2.x `torch.compile`: wins decisively (> 3×) on at least 3 of 6. **(unmeasured)** — out-of-process comparison; out of D.1e scope.
- [ ] Measured speedup vs. JAX + `jit`: wins on scalar + control-flow benchmarks (BGDHyperOpt, HookeanSpring, HMC), parity on tensor-heavy (CartPole). **(unmeasured)** — out-of-process comparison; out of D.1e scope.
- [~] Numerical deviation vs. reference < 1e-5 f32 on all 6. **(1 of 6, with caveat)** — Brachistochrone correctness verified to 5e-3 relative against finite-difference reference at N=4. The 5e-3 tolerance is wider than the DoD's 1e-5 because the 4-deep sqrt chain accumulates f32 rounding; the `paper-faithful brachistochrone at N=1` test pins the analytic form to 1e-3. Tightening to 1e-5 likely needs f64 path or a bigger N where the sqrt-chain noise averages out.
- [ ] Compile-time < 5s warm / < 30s cold per benchmark. **(unmeasured)** — informally "fast enough" (sub-second per `compileAndRun` test).
- [ ] Public benchmark harness in `benchmarks/` module. **(no `:benchmarks` module yet)** — D.1e ships a first-cut perf test inside `:compiler-plugin/test` to avoid restructuring the build mid-stage; promotion to a dedicated module deferred until kotlinx-benchmark integration is wanted.

---

## 10. Comparisons

### 10.1 Tlaloc vs. alternative AD frameworks

| Framework | Reverse-mode AD style | Coarsening-equivalent? | Notes |
|---|---|---|---|
| **Tlaloc (Stage B)** | SCT over dxir + φ-calculus coarsening | **Yes** | This doc's scope. |
| **PyTorch 2.x + `torch.compile`** | Dynamo traces → AOT autograd → TorchInductor → Triton / C++ | Partial | Graph-level kernel fusion via Inductor; no symbolic closed-form for loops. Python-loop unrolling via Dynamo, but the unrolled form is op-by-op AD, not closed-form differentiation. [ref: torch 2.x docs] |
| **JAX + `jit`** | Tracing → jaxpr → XLA HLO fusion | Partial | XLA fuses at HLO level (broadly similar to TorchInductor); `scan` / `while_loop` / `cond` are traced as HLO primitives, not symbolically reduced. [ref: XLA HLO docs, JAX lax primitives] |
| **Zygote.jl** | SCT on Julia IR | **No** | Paper measured 91×–150× coarsening speedup *opportunity* on Zygote — i.e., Zygote doesn't do it. Still SCT-baseline. |
| **Enzyme** | LLVM IR-level SCT | **No** | Operates post-LLVM; inherits LLVM's loop optimisations but no φ-calculus level. Faster than Julia SCT on some benchmarks, no closed-form. |
| **Old DiffKt (Shen et al.)** | SCT + φ-calculus coarsening | **Yes** (our reference) | Not public per §11.1.1 — we reimplement from paper. |
| **AdOptimize (optimizer-plugins)** | SCT on Kotlin IR | **No** | Paper's SCT baseline; explicitly not coarsened (§11.1.1 audit confirmed). |

**"Partial" definition:** graph-level fusion collapses adjacent elementwise ops into a single kernel but doesn't eliminate loops symbolically. `torch.compile`'s Inductor and XLA both do this. Neither replaces φ-calculus per §11.2's "Nothing in torch.compile or jax.jit replaces it" — that's true as of late 2024, and no public 2025 announcement has changed it.

**Citations to include in the doc (for the implementer to verify):**
- torch.compile: [https://pytorch.org/blog/introducing-pytorch-2-0/](https://pytorch.org/blog/introducing-pytorch-2-0/) + TorchInductor docs
- JAX jit: [https://jax.readthedocs.io/en/latest/jit-compilation.html](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- XLA HLO: [https://openxla.org/xla](https://openxla.org/xla)
- Zygote paper: Innes et al. "Don't unroll adjoint: differentiating SSA-form programs"
- Enzyme paper: Moses & Churavy "Instead of rewriting foreign code for machine learning, automatically synthesize fast gradients"

### 10.2 Symja vs. other CAS options

Expanded from §11.7's table with concrete versions + citations.

| Option | Coordinate / Version | License | Last release (as of 2026-04) | JVM-native? | Fit for Tlaloc |
|---|---|---|---|---|---|
| **Symja** | `org.matheclipse:matheclipse-core:3.1.1` (B.0b) | **LGPL-3.0** (corrected from Apache-2.0 — see §3.2.1) | Active (monthly releases) | ✅ | **Recommended v0; bake-off PASSED 2026-04-21.** |
| Custom Kotlin CAS | (in-tree, Stage B.2.5 fallback) | Project | N/A | ✅ | **Recommended v0.5 fallback.** Surface: rational arithmetic + symbolic diff + Nest + polynomial simplify. |
| GiNaC (C++) via JNI | `github.com/maboehm/java-ginac-jni` | GPL-2 | 2022 | ❌ (JNI) | Rejected. **GPL license is a deal-breaker** for an Apache-2.0 framework. |
| SymPy via subprocess | `pip sympy 1.13+` | BSD-3 | Active | ❌ (Python subprocess) | Rejected per §11.7. 1-minute compile times in the paper; Python dep. |
| **SageMath bindings** | `sagemath` Python stack | GPL-3 | Active | ❌ (Python) | Rejected. GPL + Python. |
| **MATLAB Engine** | Mathworks proprietary | Commercial | Active | ❌ (native) | Rejected. Licensing cost, vendor lock-in, not shippable. |
| **Maxima via JNI** | `maxima.sourceforge.io` | GPL-2 | Active | ❌ | Rejected. GPL + Lisp runtime. |
| **Mathematica kernel via WSTP** | Wolfram | Commercial | Active | ❌ | Rejected. Commercial. |
| **Reduce via Kotlin wrapper** | [reduce-algebra.com](https://reduce-algebra.com/) | BSD-3 | Active | ❌ | Rejected. Lisp runtime, no Kotlin wrapper. |

**Why Symja wins despite being "heavier than we need":** it's the only realistic combination of (Apache-compatible license) + (JVM-native, no subprocess/JNI) + (Mathematica-subset semantics, which the paper's authors used via SymPy's Mathematica-compatibility layer anyway). Any native (C++) option drags in JNI + native build complexity per §16's "JNI overhead kills per-op perf" risk. Any subprocess option drags in startup time + IPC latency. Any GPL option is a deal-breaker for the framework's Apache-compatible licensing.

### 10.3 The six paper benchmarks — what each stresses

Ordered by first-port recommendation (§9.1):

**Brachistochrone** — numerical optimization. Computes the brachistochrone curve time as a sum of per-segment times. **Stresses:** C6 (loop summation). **Minimal control flow.** Easy first port; validates C6 numerically. **Minimal Symja stress** — just `Sum[...]` with a rational-valued body.

**HookeanSpring** — physics simulation. Energy minimization over mass-spring systems. **Stresses:** straight-line vector arithmetic, F2 distributing elementwise through conditionals. **Moderate paper speedup on gradient (2.5×–6.6×) but large on e2e (4.1×–11.0×) because primal elimination dominates.** This is where "dead primal elimination" (§11.3) matters most — the entire primal is an SOI.

**BGDHyperOpt** — meta-learning: optimise the learning rate of a batch-gradient-descent training loop. **Stresses:** while + if + for, 9 φ-functions in the kernel, C5 + C9 + F2 all exercised. **Paper's headline benchmark (23×–27× on gradient).** Fig. 6's walkthrough is already Stage B.3's DoD test. Symja stress: large symbolic expression (the learning-rate-gradient closed form). **The bake-off benchmark — Stage B.0's Symja adequacy gate runs this.**

**HMC** — Hamiltonian Monte Carlo posterior estimation for logistic regression. **Stresses:** nested loops, `log(1 + exp(-xβ))` which requires **log1p** stability rewrite (§12). Paper speedup: 2.5×–4.4× diff, 2.3×–3.6× e2e. **First benchmark that exercises the numerical-stability pattern library.**

**CartPole** — Deep RL with NN forward pass + cartpole physics simulator. **Stresses:** mixed NN (tensor-heavy) + scalar physics; only 5% of runtime is in the SOI per the paper. **Low paper speedup (1.1×).** Port to validate "we don't make things slower" — if Tlaloc's coarsening adds > 10% compile-time cost here, we have a regression.

**QWOP** — motion optimization for a 2D stick-figure running game. 225-line function with 13 loops + if-else; expanded to 1117 lines after unrolling. **Stresses:** Stage C's SOI identification hard — this is where L matters. Paper speedup: 1.4×–1.5× e2e. **Port last** — requires mature SOI identification.

### 10.4 Bringup order justification

If Stage D has finite time, **port Brachistochrone + HookeanSpring + BGDHyperOpt first**. Those three cover C6, F2+primal-elimination, and C5+C9 respectively — the three main code paths of coarsening. If any of those fails to match paper speedup within 20%, that's the go/no-go signal per §11.12 / §16's risk table ("Coarsening speedups don't materialize on real workloads"). Only advance to HMC, CartPole, QWOP after the first three pass.

---

## 11. Benchmarks / metrics — the harness

Per the deliverable, this section defines the metrics harness (not just "run the paper's benchmarks").

### 11.1 Correctness metric — gradient numerical agreement

**Definition:** max relative error between coarsened gradient and Stage A (non-coarsened) gradient on randomized inputs.

**Formula:** `max_i |g_coarsened[i] - g_baseline[i]| / (|g_baseline[i]| + ε)` where ε = 1e-7 f32, over the gradient elements.

**Target:** < 1e-5 for f32 per §11.12 #4 / §11.13 target.

**Tool:** `DxirInterpreter.evalFunction` (§0.4.6) over N randomized sample inputs, comparing coarsened vs. non-coarsened gradient evaluations.

**Sample size:** 100 random inputs per primal; each drawn uniform from [-10, 10] for scalars, [-1, 1] for tensor elements. Fixed RNG seed (42) for reproducibility.

**Failure response:** if > 1e-5, investigate. Most likely causes: (a) expression swell + f32 rounding mismatch — fix by `SymbolicEngine.simplify` normalising associativity; (b) arithmetic reordering in Symja that doesn't survive f32 — fix by pinning associativity or by running in f64 and narrowing at the end; (c) a wrong formula — go verify against the paper.

**Acceptance threshold (go/no-go):** max relative error > 1e-5 fails the stage. No merging of Stage B with failing correctness.

**CI:** runs in `:ir:test` on every PR that touches `coarsening/` or the `PhiCalculus` pass. Takes < 30s.

### 11.2 Wall-clock speedup metric

**Definition:** gradient wall-clock time, coarsened vs. non-coarsened.

**Target per §11.12 / §11.13:** within 20% of paper's measured figures on comparable hardware.

**Tool:** `kotlinx-benchmark` (project convention per §14 tooling). Each benchmark has a "coarsened" + "non-coarsened" variant.

**Sample size:** 10 warmup iterations + 30 measurement iterations per configuration. JVM warmed up before measurements start (avoid JIT noise).

**Hardware baseline:** paper ran on "two machines" — details in paper §7. Our baseline should be a MacBook Pro M3 Pro or equivalent; AWS c6i.4xlarge is a reasonable Linux proxy for cloud CI. Document the hardware in every benchmark run's output.

**Failure response:** if < 80% of paper's figure, investigate. Most likely causes: (a) Symja producing a less-simplified form than SymPy did — tune `Simplify` assumptions; (b) the coarsened dxir being slower through `DxirInterpreter` than the op-by-op path because the interpreter isn't optimised for compile-time-produced forms — profile and optimise; (c) the benchmark's hot loop isn't actually an SOI — re-examine the SOI identification output.

**Acceptance threshold (go/no-go):** < 80% of paper speedup on ≥ 3 of 6 benchmarks → kill or re-scope coarsening per §16's fatal-for-the-optimization risk.

**CI:** runs **not** on every PR (too noisy). Runs nightly on a dedicated benchmark machine; results posted to a dashboard. Regressions of > 10% vs. previous week trigger investigation.

### 11.3 Compile-time cost metric

**Definition:** wall-clock time to compile a typical training step through the plugin, with + without coarsening, warm + cold cache.

**Target per §11.13:** < 5s warm, < 30s cold.

**Tool:** `./gradlew :compiler-plugin:benchmarkCoarsenCompileTime` — a new JMH-based task in the Stage B.3 DoD. Measures `K2JVMCompiler.compile(...)` wall-clock on fixed test sources (BGDHyperOpt, HookeanSpring, HMC).

**Sample size:** 5 cold-cache runs (clear `.tlaloc-cache/` between each) + 10 warm-cache runs (don't clear).

**Failure response:** if > 5s warm, profile — likely Symja's `Simplify` on one SOI is dominating. Options: split the SOI (smaller `Simplify` inputs), cache aggressively, or cut over to custom CAS. If > 30s cold: same diagnosis, harder — maybe Symja startup itself is expensive, in which case move Symja-init into a Gradle daemon plugin warmup.

**Acceptance threshold:** breaching cold-cache target by > 50% (> 45s) fails Stage B. Breaching warm-cache target by any amount is a warn; > 50% (> 7.5s) fails.

**CI:** runs on every PR that touches the coarsening path. Historical data tracked over time.

### 11.4 Op-count reduction metric

**Definition:** number of DxirOp nodes in the coarsened function, vs. the non-coarsened baseline.

**Target:** proxy for wall-clock speedup. Coarsened op-count ≤ baseline op-count, typically 10–50% of baseline for successful SOIs.

**Tool:** `DxirFunction.body.size + recursiveRegionOpCount(body)` — a tree walker counting every op including those inside regions. Part of the `PhiCalculus.apply` return (include as `CoarseningReport.opCountBefore, opCountAfter`).

**Sample size:** exact, not statistical. Per-primal, single measurement.

**Failure response:** if coarsened op-count > baseline, something went wrong (expression swell hit but the pass didn't fall back). Emit a warning; fall back to non-coarsened; log which formula / corollary produced the swell.

**Acceptance threshold:** 10% allowance — op-count within 10% of baseline is acceptable (some rewrites are net-neutral on node count but reduce allocation or enable downstream fusion). > 10% over baseline fails the coarsened form's apply, and the pass emits the baseline.

**CI:** runs on every PR. Fast (tens of microseconds); suitable for per-primal unit tests.

### 11.5 Allocation rate metric

**Definition:** JVM allocation bytes per gradient call, coarsened vs. non-coarsened.

**Target:** reduced (the "fewer tape allocations and closures" benefit per §11.3).

**Tool:** async-profiler with `--alloc` mode OR JFR with `jdk.ObjectAllocationSample`. Attach to a running `DxirInterpreter.evalFunction` loop; profile for 10 seconds; extract total bytes allocated in the gradient call path.

**Sample size:** 3 × 10s profiles; report median.

**Failure response:** if coarsened allocates > non-coarsened, investigate. Likely: `DxirInterpreter`'s per-op `FloatArray` allocation isn't elided for coarsened forms. Fix by fusing adjacent elementwise ops in the coarsened output (small CSE pass post-coarsening).

**Acceptance threshold:** soft — allocation regression is a warn, not a fail. Allocation is a secondary effect; wall-clock is primary.

**CI:** nightly. Allocation profiling is slow-ish; not per-PR.

### 11.6 Regression-suite entry (CI gates)

**Per-PR (cheap, required green):**
- Correctness: all 6 benchmark correctness metrics < 1e-5.
- Op-count: all 6 benchmark op-counts within 10% of baseline.
- Compile-time warm: all 6 < 5s.
- Full `:ir:test` and `:autograd:test` and `:compiler-plugin:test` green.

**Nightly (expensive, soft-gated):**
- Wall-clock speedup: tracked over time; regression > 10% triggers investigation.
- Allocation rate: tracked; regression > 20% triggers investigation.
- Compile-time cold: tracked; > 30s fails.

**Pre-release (per-bump, strict):**
- All of the above.
- End-to-end comparison vs. PyTorch 2.x + JAX on all 6 benchmarks.

### 11.7 Metric tooling summary

| Metric | Tool | CI slot | Sample size | Target | Fail threshold |
|---|---|---|---|---|---|
| Correctness | `DxirInterpreter.evalFunction` | per-PR | 100 rand inputs | < 1e-5 rel err | > 1e-5 hard fail |
| Wall-clock | kotlinx-benchmark | nightly | 10 + 30 iter | within 20% of paper | < 80% of paper on 3/6 → kill |
| Compile-time warm | JMH task | per-PR | 10 warm runs | < 5s | > 7.5s hard fail |
| Compile-time cold | JMH task | nightly | 5 cold runs | < 30s | > 45s hard fail |
| Op-count | dxir tree walk | per-PR | exact | ≤ 100% of baseline | > 110% hard fail |
| Allocation rate | async-profiler | nightly | 3 × 10s profiles | ≤ baseline | > 120% soft warn |

---

## 12. Numerical stability — pattern library

Port §11.9's pattern list; extend for transformer-relevant patterns. Each pattern has:
- A detector (a dxir subtree shape `matches(op: DxirOp): Boolean`).
- A rewrite (emit the numerically-stable form).
- A test (build the unstable form, the stable form, and assert their f32 outputs agree on a range of inputs that exercise both extremes).

The library lives at `ir/src/commonMain/kotlin/io/tlaloc/ir/passes/NumericalStability.kt` — a single pass applied after `PhiCalculus.apply` and before `DxirReverseTransform` / `DxirToIrSynthesis`.

### 12.1 Pattern: `log(1 + exp(x))` → `log1p(exp(x))` with masking **[paper §6]**

**Detector:** `LOG(ADD(CONST(1), EXP(x)))`. Catches `log(1 + e^x)` specifically.

**Rewrite (for scalar-friendly lowering):**
```
masked = IF(GT(x, MAXEXP)) { then: x } { else: LOG1P(EXP(x)) }
```
where MAXEXP = 40.0 f32 per paper §6. Relies on Kotlin's `kotlin.math.log1p`, exposed through a new `OpKind.LOG1P` (add to OpKind.kt; lowers to `stablehlo.log_plus_one` or equivalent).

**Test:** at `x = 50.0 f32`, the unstable form is `log(1 + Inf) = log(Inf) = Inf`; the stable form is `50.0` (the `x > 40` branch). At `x = -10.0 f32`, both forms agree on `log1p(exp(-10)) ≈ 4.54e-5`.

### 12.2 Pattern: `log(1 + exp(-x·β))` (HMC-specific) **[paper §6]**

**Detector:** `LOG(ADD(CONST(1), EXP(NEG(MUL(x, β)))))`.

**Rewrite:** same as 12.1 with `MUL(NEG(x), β)` as the masked value.

**Test:** on the HMC benchmark's log-loss computation. Agreement with PyTorch's `F.binary_cross_entropy_with_logits` at the same inputs.

### 12.3 Pattern: `softmax(x)` **[transformer extension — spec §11.9]**

**Detector:** `DIV(EXP(x), SUM(EXP(x), axis))`. Or the higher-level `OpKind.SOFTMAX` directly.

**Rewrite:** subtract max for numerical stability.
```
m = MAX(x, axis)
shifted = SUB(x, BROADCAST(m, x.shape))
e = EXP(shifted)
result = DIV(e, SUM(e, axis))
```

**Test:** at `x = [1000, 1001, 1002]`, the unstable form is `exp(1000) / (exp(1000) + exp(1001) + exp(1002)) = NaN / NaN`; the stable form is `[0.09, 0.24, 0.66]`.

### 12.4 Pattern: `logsumexp(x)` **[transformer extension]**

**Detector:** `LOG(SUM(EXP(x), axis))`.

**Rewrite:** subtract max + add it back.
```
m = MAX(x, axis)
result = ADD(m, LOG(SUM(EXP(SUB(x, BROADCAST(m, x.shape))), axis)))
```

### 12.5 Pattern: `layernorm(x)` with small epsilon **[transformer extension]**

**Detector:** `LAYERNORM` op with attr `eps < 1e-7 f32`.

**Rewrite:** clamp eps to 1e-5 f32 minimum, or bump to the op-kind's stable default.

### 12.6 Pattern: `scaled_dot_product_attention(Q, K, V, causal=true)` **[transformer extension]**

**Detector:** `SCALED_DOT_PRODUCT_ATTENTION` op with `attrs["causal"] == true`.

**Rewrite:** fuse the mask into the attention computation (use the causal-mask subtract-inf trick). §11.9's "scaled_dot_product_attention with causal mask fusion".

### 12.7 Pattern: `sqrt(var + eps)` (batchnorm / layernorm denominator) **[standard ML stability]**

**Detector:** `SQRT(ADD(var, CONST(eps)))` where `eps > 0`.

**Rewrite:** emit as-is (already stable for eps > 1e-5). Flag as a stability-checked site. [Boring but essential per §11.9.]

### 12.8 Testing strategy

Every pattern gets **three** tests:
- Numerical agreement with a PyTorch reference at typical inputs (positive small, negative small, zero).
- Behaviour at the numerical boundary (large positive x where the naïve form overflows, large negative where the naïve underflows).
- That the rewrite doesn't fire when the pattern doesn't match (e.g., `log(1 + x)` where x is not `exp(y)` should not be rewritten — log1p requires the specific exp form).

### 12.9 Numerical stability pass — apply order vs. φ-calculus

Apply order:
```
FIR → DXIR
    → PhiCalculus.apply           (coarsening; rewrites IF/WHILE)
    → NumericalStability.apply    (replaces unstable patterns with stable ones)
    → DxirReverseTransform
    → DxirToIrSynthesis
```

Why stability runs after coarsening: coarsening may *produce* unstable patterns by combining terms (e.g., `log(1 + exp(x))` pattern emerges after `log` distributes into a branch). Running stability *after* catches what coarsening produces.

Symmetric argument for running stability *before* coarsening: unstable patterns in the primal may survive coarsening because Symja's `Simplify` doesn't know they're unstable. Solution: run stability twice — once on the primal, once on the coarsened form. The idempotent detector means re-running costs only O(body size) per pass.

---

## 13. Risks & mitigations

6-row table minimum per the brief; expanded.

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Paper inaccessibility** (Blocker 1) blocks C1–C9 implementation. | High (today) | Medium | §2.1's 5-tier fallback plan. Worst case: deliver B.0, B.1, B.2 with F1/F2/F3/F4/F5/C5; flag C1–C4/C6–C9 as blocked; unblock via paper access on the next session. |
| 2 | **Symja scale-up fails** (compile-time > 60s on BGDHyperOpt or > 5 min on a real benchmark). | Medium | High | Run the §3.2.3 bake-off in B.0 BEFORE writing any φ-calculus code. If Symja fails, swap to custom Kotlin CAS — 4–6 weeks additional per §18's estimate. |
| 3 | **Expression swell** in F2 distribution blows up compile-time + output size. | Medium | Medium | The F2 anti-swell gate (§4.3): only distribute when the IF has a single use OR branches are < 5 ops. Stage C's SOI identification further bounds via the size limit L. |
| 4 | **Compile-time regression** — coarsening makes non-coarsened users (debug builds, `jit(coarsen=false)`) slower via plugin overhead. | Low | Medium | `jit(coarsen = false)` must truly skip the pass (no partial pass that still incurs symbolic-engine startup). Bench per-PR with a non-coarsened surface; > 10% compile regression fails. |
| 5 | **Numerical divergence** — coarsened gradient disagrees with non-coarsened within f32 tolerance. | Low | High | 1e-5 correctness metric blocks merge. Symja in Exact rational mode + f32 narrowing only at the very end of `SymbolicEngine.lowerToDxir`. The §12 pattern library is specifically about this. |
| 6 | **Control-flow IR widening (B.0) drags more than planned** — WHILE semantics (data-dependent termination, break, continue) turn out to need more dxir surface than estimated. | Medium | Medium | Defer break/continue to post-Stage-B. WHILE is simple-loop-only in B.0; C9 handles data-dependent termination symbolically, not structurally. B.0's 2-session estimate has a +1-session contingency. |
| 7 | **Stage C cost-model mis-tuning** — SOI identification's L value wrong for our op mix, producing sub-optimal SOIs or expression swell. | Medium | Medium | Stage C.4 is explicit cost-model tuning. Port paper's `L = 50` as default; measure per-benchmark on bring-up. Expose as `jit(coarsenBudget = N)` (§11.10). |
| 8 | **FIR-side lowering (B.4) bugs in SSA-ification of `var` in loops.** | Medium | Medium | Loop-carried `var` needs explicit SSA phi insertion in the dxir. B.4 lands with strict tests — every supported Kotlin loop pattern gets a structural test pinning the generated WHILE shape. |
| 9 | **`OpKind.COARSENED`'s per-op `readsPrimalOperandIndices`** diverges from the static interface contract and confuses `usedByAdjoint`. | Low | Medium | Special-case in `DxirReverseTransform.computeUsedByAdjoint` — documented at the site. Alternative: widen `VjpRule.readsPrimalOperandIndices` from `val` to a function `fun readsPrimalOperandIndices(op: DxirOp): Set<Int>`; one-time refactor. |
| 10 | **Higher-order differentiation through coarsening** (§2.7) surfaces issues Stage B didn't anticipate. | Low | Low | Explicitly deferred. `grad(grad(f))` is not a Stage B DoD item. |
| 11 | **Paper's benchmarks don't port cleanly** to Kotlin (e.g., require external datasets, specific RNG, Julia-specific libs). | Medium | Medium | Port 3/6 first (§9.1 recommended order); if that's painful, the remaining 3 are gravy. Paper's `benchmarks/` folder on the diffkt repo is gone per §11.1.1, so we're reconstructing from the paper's code listings. |
| 12 | **Custom-CAS fallback turns out to need more surface than expected** (e.g., Mathematica-style `Refine[expr, assumption]` matters for one corollary). | Low | Medium | Symja first is cheaper; fallback triggered only by concrete Symja inadequacy. Spec'd budget: 4–6 weeks fallback per §18. |
| 13 | **Symja thread-safety** — parallel compilation may corrupt shared `ExprEvaluator` state. | Low | Medium | Wrap Symja calls in a `synchronized` block (the Symja docs recommend it for multi-threaded use). Profile to see if it's a bottleneck; if so, use thread-local evaluators. |
| 14 | **Gradle incremental compilation** — plugin changes force re-coarsening of unchanged code because of a bug in how we hash dxir subtrees. | Low | Medium | Pin the `cas_version` string to include the plugin version; invalidate only on structural change. Test by modifying a comment in a `grad { ... }` lambda and verifying the cache hits. |

---

## 14. Timeline estimate

### 14.1 Spec §11.11 baseline

> Target: Month 9. One engineer, full-time, for ~3 months after the Month 6 milestone lands. Assumes a working K2 plugin and dxir foundation.

### 14.2 Revision based on §7 substaging

Stage B's substaging (§7.6) totals 17–24 sessions. At 2 sessions/week (the §0.4.3–§0.4.9 observed cadence for a solo researcher with agent-assisted development): **8.5–12 weeks.** At 1 session/day full-time: **~5 weeks.**

**Revised realistic estimate:** **10–12 weeks** for Stage B alone (B.0 through B.4). The spec's Month-9 target gives 12 weeks; this fits, but *only if B.0–B.4 ships cleanly and Stage C + Stage D get the remaining budget*.

Stage C (SOI identification) realistic: 6–8 weeks.
Stage D (benchmark port + validation): 4–6 weeks.

**Total: Stage B + C + D = 20–26 weeks = 5–6 months full-time.**

### 14.3 Spec target vs. realistic

The spec says "3 months" for coarsening landing as a differentiator (§11.11) and Month-9 for it to be "public + validated" (§12's Month-9 milestone). That 3-month budget is **tight** given the IR widening prerequisite and the FIR-lowering widening, neither of which were called out as separate line items in the original spec.

**Recommendation for the implementer:**
- Keep Stage B's budget as 3 months. If B.4 is lagging at month 2.5, ship B.0/B.1/B.2/B.3 without B.4 — the pass works, just requires hand-built primals, which is fine for validation.
- Plan Stage C for month 4–6 (2 months).
- Plan Stage D for month 7–8 (2 months).
- Keep Month 9 as the "feature launch" target for coarsening, but with Stage D's benchmarks as the gate for declaring it shipping-ready.

### 14.4 If the paper is inaccessible (Blocker 1 stays open)

Stage B.3 is blocked on paper access for C1–C4, C6–C9. If the paper genuinely cannot be obtained, the stage ships as Stage B'  — F1, F2, F3, F4, F5, C5 only. That's sufficient for ~40% of the paper's benchmark speedup (C5 alone powers BGDHyperOpt's simple-loop closure; F2 powers HookeanSpring's primal elimination). Brachistochrone requires C6 and is blocked.

### 14.5 Calendar anchor

Today is 2026-04-20 (per session context). Stage A closed 2026-04-20 (same day, §0.4.9). If Stage B.0 starts next session (2026-04-21 or later), expect B.0+B.1+B.2+B.3+B.4 to land by mid-to-late July 2026. Stage C starts then; Stage D by mid-September. Month 9 target of September 2026 remains achievable IF paper access resolves within a week of B.3's start.

---

## 15. Appendix: glossary

Terms used throughout. Provided so a reader new to this vocabulary can pick the doc up cold.

- **AD**: Automatic differentiation. Computes derivatives by chain rule over a program's operations, rather than finite-differencing or deriving symbolic closed forms by hand.
- **Adjoint**: the dual value propagated backward in reverse-mode AD. For a scalar-return function, the adjoint of a value is the gradient of the return w.r.t. that value.
- **CAS**: Computer Algebra System. Software that performs symbolic math — algebraic simplification, symbolic differentiation, integration, equation solving, etc. Examples: Mathematica, SymPy, Maxima, Symja.
- **Closed form**: an expression that doesn't contain loops or unbounded recursion — just arithmetic and explicit algebraic operations. `f⁽ⁿ⁾(x) = x · 2ⁿ` is closed form; `for i in 1..n: x = x * 2` is not.
- **Coarsening**: the paper's name for replacing a fine-grained op-by-op AD computation with a coarser-grained pre-computed symbolic derivative. The effect is to reduce allocations, function call overhead, and sometimes op count.
- **Custom adjoint**: a user-defined (or compiler-generated) VJP rule for a specific operation, bypassing the op-by-op chain rule.
- **Dxir / DXIR**: Tlaloc's intermediate representation. SSA-typed dataflow graph with an explicit sharding model.
- **DxirFunction**: a complete dxir function: parameters, body (a sequence of `DxirNode`s in program order), and return value(s).
- **DxirNode**: the base sealed class for all dxir values. Subtypes: `DxirParam`, `DxirConst`, `DxirOp`, `DxirOpResult`, `DxirCall`, `DxirBlockArg`.
- **DxirOp**: an operation with a kind (`OpKind`), operands (references to other DxirNodes), attributes, result types, and optional nested regions.
- **DxirRegion**: a nested code-region inside an op. Used today for `MANUAL_COMPUTATION` (sharding); in Stage B used for `IF`/`WHILE`.
- **F1–F5**: the five fundamental formulae of the φ-calculus. See §4 for bodies.
- **C1–C9**: nine corollaries composing the fundamental formulae with arithmetic. See §4.
- **φ (phi) function**: in SSA IR, the operator at a merge point that selects the correct value based on which predecessor block executed. E.g., `y = φ(x_then, x_else)` means "y is x_then if coming from the then-branch, x_else if from else-branch".
- **φ-calculus**: Shen et al.'s algebraic system extending φ functions with rewrite rules that enable symbolic manipulation through branches and loops.
- **φ_L(p, d)**: loop-entry φ — "value entering the loop header at each iteration". `p` is the pre-header, `d` is the back-edge.
- **φ_L'(...)**: loop-exit φ — "value of the loop-carried variable at the loop's exit point".
- **Gradient body / gradient dxir**: the dxir subtree representing the computed gradient of a function w.r.t. its parameters.
- **Higher-order AD**: computing derivatives of derivatives (second derivative, Jacobian of a gradient, etc.).
- **IExpr**: Symja's symbolic expression type. Opaque to user code.
- **IR**: Intermediate Representation. In Kotlin compiler context, there's Kotlin's FIR (frontend IR, pre-resolution) and IR (post-resolution, pre-codegen). Dxir is Tlaloc's own IR between FIR and IR.
- **K2 compiler plugin**: a Kotlin compiler plugin written against the K2 FIR/IR API. Tlaloc's compiler plugin is a K2 plugin.
- **Op-count**: the number of `DxirOp` nodes in a function's body (recursively through regions).
- **Primal**: the forward-pass computation. "Primal body" = the forward DxirFunction.
- **Primitive**: an operation that AD treats as atomic — its VJP is defined by a rule rather than derived from sub-operations.
- **Reverse-mode AD**: propagate gradients backward from output to inputs. Efficient for scalar-output, many-inputs functions (typical ML case). Tlaloc's primary AD mode.
- **SCT**: source-code transformation. AD implemented by rewriting the program's source (or IR) rather than tracing at runtime.
- **SOI**: Segment of Interest. A subregion of the primal function targeted for symbolic coarsening. Bounded by a size limit L (per the paper).
- **SSA**: Static Single Assignment. A property of an IR where every variable is assigned exactly once. Dxir is SSA.
- **Symja**: an LGPL-3.0 JVM-native CAS roughly compatible with a subset of Mathematica syntax. Tlaloc's default CAS backend; v3.1.1 confirmed adequate by Stage B.0b's bake-off (`docs/papers/symja-bakeoff-2026-04.md`).
- **Trip count**: the number of iterations a loop executes. May be statically known (induction variable bound) or data-dependent.
- **VJP**: Vector-Jacobian Product. The building block of reverse-mode AD: given an upstream gradient and the op's forward value, compute the downstream gradient for each input.
- **VjpRegistry**: Tlaloc's `OpKind → VjpRule` map. Stage A's one-math-source-of-truth. Lives at [`Vjp.kt`](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/Vjp.kt).

---

## 16. References

- **Paper (binding spec source for Stage B):** Shen, X., Zhang, G., Dea, I., Andow, S., Arroyo-Fang, E., Gafter, N., George, J., Grueter, M., Meijer, E., Shivers, O., Stumpos, S., Tempest, A., Warden, C., and Yang, S. *"Coarsening Optimization for Differentiable Programming"*. Proc. ACM Program. Lang. 5, OOPSLA (October 2021), Article 130. [DOI 10.1145/3485507](https://dl.acm.org/doi/10.1145/3485507). Preprint: [arXiv:2110.02307](https://arxiv.org/abs/2110.02307). PDF: [https://www.ccs.neu.edu/home/shivers/papers/coarsening-autodiff.pdf](https://www.ccs.neu.edu/home/shivers/papers/coarsening-autodiff.pdf).
- **Spec:** [DIFFKTX_SPEC.md](../DIFFKTX_SPEC.md) §11 (binding), §0.4.3–§0.4.9 (Stage A substrate).
- **Prior-art audit:** §11.1.1 (facebookresearch/diffkt and facebookresearch/optimizer-plugins — neither has coarsening code).
- **Stage A SCT baseline (inspirational for Stage B's structure but not reused):** the original AdOptimize plugin in `facebookresearch/optimizer-plugins/main` — SCT on Kotlin IR. We've already extracted the useful abstractions (`active`, `referenced-in-backprop`, `loop-entry-identifier`) per §11.1.1 into Stage A's `usedByAdjoint` analysis.
- **IR substrate (Tlaloc codebase):**
    - [DxirNode.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/DxirNode.kt) — base node types + region + block.
    - [DxirModule.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/DxirModule.kt) — function + module + builder.
    - [OpKind.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/OpKind.kt) — op enum.
    - [DxirType.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/DxirType.kt) — type with dims + dtype.
    - [DxirReverseTransform.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/DxirReverseTransform.kt) — Stage A SCT pass.
    - [Vjp.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/Vjp.kt) — VjpRegistry + registered rules.
    - [DxirInterpreter.kt](../ir/src/commonMain/kotlin/io/tlaloc/ir/passes/DxirInterpreter.kt) — floatarray interpreter (Stage B test substrate).
    - [FirLambdaToDxirLowering.kt](../compiler-plugin/src/main/kotlin/io/tlaloc/plugin/FirLambdaToDxirLowering.kt) — FIR → dxir lowering (Stage B.4 extends this).
- **Comparison frameworks:**
    - PyTorch 2.x dynamo + inductor: [https://pytorch.org/blog/introducing-pytorch-2-0/](https://pytorch.org/blog/introducing-pytorch-2-0/).
    - JAX jit/scan: [https://jax.readthedocs.io/](https://jax.readthedocs.io/).
    - Zygote.jl: Innes et al., "Don't unroll adjoint: differentiating SSA-form programs".
    - Enzyme: Moses & Churavy, "Instead of rewriting foreign code for machine learning, automatically synthesize fast gradients".
- **CAS candidates:**
    - Symja: [https://github.com/axkr/symja_android_library](https://github.com/axkr/symja_android_library).
    - GiNaC: [https://www.ginac.de/](https://www.ginac.de/).
    - SymPy: [https://www.sympy.org/](https://www.sympy.org/).

---

## 17. Final framing

Stage A closed in §0.4.9 with 332 tests green. The registry, the shape-aware interpreter, the dxir-eval bridge, the `usedByAdjoint` analysis, the Stage-A SCT transform — every one of these substrate pieces was chosen with Stage B's testability in mind. Coarsening correctness reduces to: *run gradient through interpreter; verify numerical agreement with Stage A baseline to within f32 tolerance*. That makes Stage B's validation trivial at the unit level; the interesting work is the φ-calculus itself and the Symja integration.

Stage A was the five-session prerequisite (§0.4.3 → §0.4.9). **Stage B is the research payoff.** This planning session's deliverable (this doc) is the shopping list for the next ~3 months of turning that substrate into the 10×–335× speedup the paper measured. The next session picks up from §7's B.0 and starts landing IR widening.

If paper access resolves and Symja's bake-off passes, Stage B ships on §14's timeline. If either blocks, the doc's fallback paths apply. **Either way, the plan is this doc, not more planning.** The next substantive action is B.0's IR widening.
