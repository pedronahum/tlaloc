# Bounded dimensions: progress log

Design: [../design/bounded-dims.md](../design/bounded-dims.md). Branch `feat/bounded-dims`
from `main` at `cc183fa`. Not pushed.

## Baseline (before any change)

- 2026-09-28 22:28, `./gradlew test --rerun-tasks --continue` at `cc183fa`: 2,797 tests,
  1 failure, `BUILD FAILED` in 5 min 14 s.
- The failure is `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes`,
  the dispatch-floor timing assertion that `docs/CAPABILITIES.md` lists as failing now
  and then under machine load. The GPU showed 16 % utilisation from another process
  during the run. Treated as pre-existing.
- Counting: `scripts/count-tests.sh` also sums `.claude/worktrees/*` results (4,296);
  the count above excludes `.claude/` and `third-party/`.

## Log

### Phase 1: investigation and design

- Traced types (core → plugin → DXIR → StableHLO → interpreter) and serving
  (exporter, manifest, three runtimes). Findings are in the design doc's "What exists
  today".
- Spike: bounded dynamic StableHLO on the XLA CUDA PJRT plugin through `tlaloc_pjrt.py`.
  Unbounded `?` refused at compile; a bounded parameter compiles but no C-API host
  buffer matches it (the executable wants `f32[<=8]`, 36 bytes); `set_dimension_size`
  inside a program with static IO sums exactly `n` elements on the GPU. Decision:
  buckets. Scripts were in the session scratchpad, not committed; the table in the
  design doc has the results.

### Step 1: type-level API (`2df17db`)

- `DimBound(max)` and `Bounded<B>` in `:core` (`Shape.kt`), `@ExperimentalTlalocApi`.
  `ShapeAtom` is sealed, so the atom has to live in `io.tlaloc.core`.
- Suite: 2,801 tests (+4), 1 failure: the same KPTX timing test as the baseline.

### Step 2: compile-time checks (`3cb4610`)

- `TlalocBoundedDimChecker.kt`: `BOUNDED_DIM_EXCEEDED`, `BOUNDED_AXIS_MISMATCH`,
  `BOUNDED_DIM_INVALID`, `BOUNDED_SPEC_ARITY` (the last for `specOf`, step 3).
- Correction to the design doc's first draft: elementwise `+ - * /` do NOT reject
  mixed shapes natively. `BroadcastOps.kt` has `<S1, S2> DTensor<S1>.plus(DTensor<S2>)`
  returning `DTensor<Shape>`, so any two shapes type-check. `matmul` and `contract` do
  reject mixed bounds natively (shared type variables). Hence `BOUNDED_AXIS_MISMATCH`
  for the broadcasting operators: right-aligned axes with two different bound objects.
  A bound against `Sym` is allowed (it may be a broadcast size-1 axis).
- A bound's `max` comes from the delegating constructor call of a source-declared
  object (`FirConstructorSymbol.resolvedDelegatedConstructorCall`); a bound in a
  compiled dependency is skipped (test pins it).
- `grad { }` over `DTensor<Rank1<Named<SeqLen, Bounded<MaxSeq>>>, F32>` lowers and gives
  `2x` at every size 1..6 (the plugin maps the unknown atom to `-1`, as for `Sym`).
- Suite: 2,810 tests, 0 failures, `BUILD SUCCESSFUL`.

## Decisions

- Bucketing, not dynamic shapes (spike above).
- A bound is an object (`object MaxSeq : DimBound(4096)`) used through the atom
  `Bounded<MaxSeq>`; the object is the dimension's identity. Kotlin has no integer
  type parameters.
- New manifest file `tlaloc-bounded.json`, version `tlaloc-bounded-v1`; the serving
  manifest is untouched.
- Padding correctness is checked by the exporter in the interpreter, not assumed.

## Open problems

(none yet)

## Next step

Step 3: `specOf<S>` (JVM, reads bounds from `typeOf<S>()`) and `BoundedProgram` in
`:autograd` (exact-size runs, bucketed runs with valid masks, `checkPadding`).
