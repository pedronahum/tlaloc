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

Phase 2 step 1: `DimBound` and `Bounded` in `:core`, `@ExperimentalTlalocApi`, ABI dump
updated.
