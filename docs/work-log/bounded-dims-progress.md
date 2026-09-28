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

### Step 3: interpreter (`c90a541`)

- `BoundedProgram`, `TensorSpec`, `AxisSpec`, `BucketLadders`, `BoundedContext`
  (`validMask`, `validLength`) in `:autograd` commonMain; `specOf<S>` in a new `jvmMain`
  (reads `typeOf<S>()` and the bound object's `INSTANCE` field; no kotlin-reflect).
- A trace per size assignment, on zeros, cached. Parameters come out in tape order: data
  leaves first, then masks and lengths in the order the body asked for them.
- `checkPadding` compares bucketed with exact on seeded inputs; it catches an unmasked
  mean (worst at a size inside a bucket) and passes an unmasked sum (zero padding adds 0).
- Every size 1..8 of a row-wise program, a masked mean, a masked softmax, two bounds at
  once (3 x 8) and an I32 embedding input: exact == bucketed == a hand-written
  static-shape program captured at that size.
- Suite: 2,820, 0 failures.

### Step 4: export (`ec0914c`)

- `BoundedManifest` (`:maestro` commonMain, `tlaloc-bounded-v1`, strict reader) and
  `BoundedProgramExport` (jvmMain). One body per bucket combination, content-addressed;
  a `ProgramManifest` per entry; refused when `checkPadding` fails.
- `ServingManifest.fromJson` now names a bounded manifest instead of failing on a
  missing `modelName`. No other change to the serving manifest.
- Suite: 2,824, 0 failures.

### Step 5: Python runtime and refusals (`415bfa3`)

- `harness/python/tlaloc_bounded.py` (stdlib + `tlaloc_serve`'s ctypes engine),
  `tlaloc_bounded_test.py` (11 unit tests, fake engine), `run_bounded_check.py` (GPU lane,
  under the import guard).
- `BoundedArtifactRunTest` on the GB10, PJRT-CUDA:
  - masked mean, sizes 1..64, buckets 8/16/32/64: within 1e-4 of the interpreter; 4
    compiles; the same program with one body per size: 64 compiles, results within 1e-5.
  - masked softmax, sizes 1..64: within 1e-4 of the interpreter.
  - embedding (I32) + two bounds (4 x 16) + a matmul: within 6.4e-4 of the interpreter
    and 4.5e-4 of one-body-per-size. The dot runs in TF32 on the GPU (no precision
    attribute on a plain MATMUL), so the band is 2e-3 for this case.
- Refusals: `tlaloc_serve.ServingArtifact.load` (and so the vLLM plugin) and
  `TritonModelRepository.write` refuse a bounded artifact by name. The C++ Triton backend
  is unchanged; pointed at a bounded manifest it already refuses:
  `schemaVersion 'tlaloc-bounded-v1' is not tlaloc-serving-v1, v2 or v3`.
- LLM serving unchanged and re-checked in the step-5 suite run: `HfLlamaServingArtifactTest`
  ran (TinyLlama checkpoint present): "tokens agree with HF", through `tlaloc_serve` and
  vLLM; `ServingArtifactExportRunTest` and both vLLM lanes passed.
- Suite: 2,829, 0 failures.

### Compile counts, mixed-length workload (GB10, GPU idle at 2 %)

`BoundedArtifactRunTest` "mixed-length workload", `TLALOC_BOUNDED_BENCH=1`. 200 requests,
lengths uniform in 1..512 (seed 2026), 163 distinct. Single-head self-attention
(d = 64, three 64 x 64 weights) with a masked key softmax and a masked mean pool.
`tlaloc_bounded.py` on PJRT-CUDA, one process per artifact, compiles on first use.

| | Compiles | Compiling | Executing (compiles excluded) |
|---|---|---|---|
| One body per distinct length (the per-shape export) | 163 | 264.1 s | 0.85 s |
| Buckets 16, 32, 64, 128, 256, 512 | 6 | 7.6 s | 0.96 s |

- 27x fewer compiles and 35x less compile time; execution 13 % slower from padding
  (each request computes at its bucket's length, up to 2x its own).
- The two agree within 3.9e-7 (relative to max(1, |y|)) on all 200 requests.
- A compile takes 1.3 to 1.6 s here: XLA autotuning the dots dominates.

## Decisions

- Bucketing, not dynamic shapes (spike above).
- A bound is an object (`object MaxSeq : DimBound(4096)`) used through the atom
  `Bounded<MaxSeq>`; the object is the dimension's identity. Kotlin has no integer
  type parameters.
- New manifest file `tlaloc-bounded.json`, version `tlaloc-bounded-v1`; the serving
  manifest is untouched.
- Padding correctness is checked by the exporter in the interpreter, not assumed.
- The mask is a program input the runtime fills, not a constant baked per bucket, so one
  body serves every length in its bucket.
- `BoundedProgram` has one output tensor, like `captureN`.
- The bounded manifest's reader refuses unknown keys (the serving manifest's readers
  ignore them); a later field needs a version bump.

## Open problems

(none yet)

## Next step

Optional: the Triton backend (C++) serving bounded artifacts. Otherwise the final hour:
CHANGELOG, CAPABILITIES, README, and the summary at the top of this log.
