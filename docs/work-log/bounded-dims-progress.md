# Bounded dimensions: progress log

Design: [../design/bounded-dims.md](../design/bounded-dims.md). Branch `feat/bounded-dims`
from `main` at `cc183fa`. Merged into `main` as `9ab22a4` and pushed on 2026-09-29.

## Summary for the reviewer

**Shipped.** Bounded dimensions, end to end, marked `@ExperimentalTlalocApi`:

| Layer | What | Evidence |
|---|---|---|
| Types (`:core`) | `DimBound(max)`, atom `Bounded<B>`; the bound object is the axis's identity | `BoundedTest` |
| Compile time (plugin) | `BOUNDED_DIM_EXCEEDED`, `BOUNDED_AXIS_MISMATCH`, `BOUNDED_DIM_INVALID`, `BOUNDED_SPEC_ARITY`, at file/line/column; `matmul`/`contract` mixing rejected by Kotlin's type checker; `grad {}` over bounded params at every size | `BoundedDimCompileCheckTest` (10) |
| Interpreter (`:autograd`) | `boundedProgram`, `specOf<S>`, exact and bucketed runs with valid-mask/length inputs, `checkPadding`, `valueAndGrad` training steps, `BoundedExecutor` | `BoundedProgramTest` (14): static-shape and analytic oracles at every size |
| Export (`:maestro`) | one body per bucket + `tlaloc-bounded.json` (`tlaloc-bounded-v1`); refused when padding changes the result | `BoundedProgramExportTest` |
| Python runtime | `harness/python/tlaloc_bounded.py` | 15 unit tests; PJRT-CUDA at every size up to the bound (`BoundedArtifactRunTest`) |
| Triton | backend bounded mode (`bounded_manifest`), `TritonModelRepository` writes it | `verify.sh` passed on the GB10, HTTP and gRPC |
| vLLM | refuses a bounded artifact by name (through `tlaloc_serve`) | `tlaloc_bounded_test` |
| GPU training | bucketed steps on a `PjrtSession` | `PjrtBoundedTrainingTest`: 4 executables vs 27 |

Compile counts (GB10): 200 mixed-length requests (1..512, 163 distinct) through an
attention block: 163 compiles / 264 s at exact lengths, 6 / 7.6 s bucketed; execution
13 % slower from padding. LLM serving is unchanged: TinyLlama's greedy ids still equal
Hugging Face's (`HfLlamaServingArtifactTest`, direct and vLLM), and Qwen3-0.6B f32, bf16
and int8 still give 16 of 16 ids through Triton (`verify.sh`).

**Not shipped, and why.**
- Bounded dynamic StableHLO. A `#stablehlo.bounds` parameter compiles on XLA GPU, but the
  PJRT C API that all three runtimes use cannot create a buffer for it; dynamism inside a
  program (`set_dimension_size`) runs on the GPU for one reduction checked, but the
  interpreter, IREE and the rest of the op set are unverified. The spike is in the design
  doc; it is the path to one compile per bound instead of one per bucket.
- Triton bounded mode: one output, no dynamic batching, host path only; rank-0 tensors
  refused by name.
- The language-model exporter's ladders are not derived from a `DimBound` (it already
  buckets).

**Known limitations.**
- The compile-time bound check reads `max` from a bound declared in the module being
  compiled; a bound from a dependency is checked at run time only.
- Only constant sizes are checked at compile time; sizes inside a `grad {}` lambda are
  `-1` as before.
- Correct padding is the program's job (masks); the exporter checks it in the
  interpreter at the edges of every bucket, not at every size.
- `BoundedProgram` is not thread-safe, and exact-size runs keep one trace per size.
- A dot on the GPU runs in TF32, so GPU-vs-interpreter bands are 2e-3 for programs with
  a matmul (1e-4 without).

Open issues to address later: [../FOLLOWUPS.md](../FOLLOWUPS.md).

**Review first.**
1. The type encoding: a bound is an `object X : DimBound(n)` used as `Bounded<X>`, and
   the object is the axis identity (two independent axes need two objects). Kotlin has no
   integer type parameters; this is the choice everything else rests on.
2. The manifest: a new file and version (`tlaloc-bounded.json`, `tlaloc-bounded-v1`)
   rather than a `tlaloc-serving-v4`, and a reader that refuses unknown keys (the serving
   readers ignore them). `BoundedManifest.kt`, `tlaloc_bounded.validate`,
   `bounded_mode.cc ReadManifest` are three readers of one schema.
3. The padding contract: masks are program inputs the runtime fills, and the exporter
   refuses a program whose padded result differs from the exact one.
4. `BoundedTrace.cacheKey` (content-addressed; an earlier name-based key could alias two
   programs, found in review).
5. The C++ bounded mode (`triton/backend/bounded_mode.cc`), new code in the server.

Also found: regenerating the Triton examples (`exportTritonExamples`) changed four
committed `config.pbtxt` files (they predated backend batching). Regenerated in
`8963d12`; `verify.sh` passes on them.

Final clean-room run (2026-09-29 00:04, `./gradlew test --rerun-tasks --continue`):
2,838 tests (baseline 2,797, +41), 0 failures, 102 skipped (the baseline's 101 plus
`LlamaDecoderPytorchBenchTest`, which skips itself when it sees concurrent load).

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

### Triton backend bounded mode (`a03e246`)

- `triton/backend/bounded_mode.{h,cc}`: a model with `bounded_manifest` in its
  `config.pbtxt`. Strict manifest reader (same refusals as Kotlin and Python), each
  body's signature checked against the manifest at its buckets, every body compiled
  at load; per request: sizes from input shapes, smallest bucket per bound, host
  padding, masks and lengths, execute, slice. Host path only, one request per
  execution, `max_batch_size: 0`.
- `TritonModelRepository.write` now writes a bounded artifact as a bounded-mode model
  (`boundedConfig`) instead of refusing it.
- Examples `bounded_mean` and `bounded_softmax` (bound 16, buckets 4/8/16) in
  `triton/examples/model_repository`, references for every length in
  `triton/examples/reference/bounded_*.json`, `triton/bounded_checks.py` (KServe v2
  JSON over HTTP with the standard library; gRPC through `tritonclient` when it is
  importable), a `verify.sh` step with a `--perturb` control.
- Manual run on the GB10 (server on ports 8100-8102, only the two models): every
  length 1..16 of both models within 6.7e-8 of the interpreter; length 17 refused by
  the backend by name; a `[3, 5]` input refused by Triton itself against the config
  dims; `--perturb` fails. Load-time refusals checked with two broken configs:
  `max_batch_size: 4` and a fixed dim where the bound should be, both refused by name.
- Regenerating the examples with `exportTritonExamples` changes four committed
  `config.pbtxt` files (`reference_sequence`, `window_sequence*`) in comments and
  `max_queue_delay_microseconds`: pre-existing drift on `main` between the generator and
  the committed files. Reverted; not part of this branch.
- This machine has no `tritonclient`; `verify.sh` ran with a throwaway venv in the session
  scratchpad (`tritonclient[all]`, numpy, `cuda-python` 12.9), not committed. With
  cuda-python 13.4, tritonclient's CUDA shared-memory helper fails
  (`cudaIpcMemHandle_t` has no `reserved`): a client-environment problem, not the backend.
- `verify.sh` with `SKIP_PERF=1 SKIP_TINYLLAMA=1`: `VERIFY PASSED`. Every existing step
  passed with the changed backend (device checks, dtypes, buckets, batching, ragged
  batches, CUDA shared memory, reference and window sequence models, batched prefill, all
  negative controls), then Qwen3-0.6B f32, bf16 and int8 (16 of 16 greedy ids equal
  HuggingFace's over HTTP and gRPC for both prompts) and the wrong-axis int8 control. The
  new step: both bounded models, every length 1..16, over HTTP and gRPC, within 6.7e-8;
  length 17 refused; `--perturb` failed 65 checks. (`SKIP_QWEN=1` is not the switch's
  name, so Qwen3 ran.)

### Training steps (`a03e246`, `PjrtSession` executor in the next commit)

- `BoundedProgram` has several outputs now (`outputs`; `output` for the one-output case;
  `runAll`, `runBucketedAll`), and `valueAndGrad(wrt)`: `DxirReverseTransform.apply(fn,
  includeForward = true, inputOnlyTrailingParams = masks and lengths)`, keeping the value
  and the requested gradients. No dead-code pass is needed for the unselected gradients:
  XLA drops them at compile.
- Interpreter: a masked MSE `sum((xW - t)^2 * mask) / len`; loss, dL/dW and dL/dx equal
  the analytic formulas within 1e-5 at every size 1..8, exact and bucketed; with a bias
  and no mask, `checkPadding` fails (padded rows count), with the mask it passes.
- PJRT (Python runtime): the same step at every size 1..64 within 5.1e-4 of the
  interpreter (one TF32 dot), 4 compiles.
- `BoundedExecutor`: `runAll`/`runBucketedAll` take an executor; `BoundedTrace.cacheKey`
  names the trace for `PjrtSession.runOn`. `PjrtBoundedTrainingTest`: 40 SGD steps on
  the GB10 over lengths 1..64 (27 distinct): 4 executables for buckets 8/16/32/64 against
  27 at exact lengths; loss 5.24 -> 0.048; the first step within 1e-2 of the interpreter.
- Suite: 2,836, 0 failures.

### Review of the branch (`/code-review high`), fixes

Ten findings, all addressed:

- `BoundedTrace.cacheKey` was `name/sizes`: two programs with one name (two `valueAndGrad`
  of one loss with different `wrt`) would share a `PjrtSession` executable and silently
  return the wrong gradient. The key is now content-addressed: FNV-1a 64 over the printed
  trace with SSA ids renumbered and array-constant identities masked, plus the arrays'
  contents. Tests: different `wrt` and same-name programs get different keys; one program
  traced twice gets one key.
- `checkPadding` counted equal infinities or NaNs as a mismatch; now they agree (test:
  `log(0)` passes).
- `specOf` arity: an absent vararg now counts as 0 sizes (was unchecked); a spread `*arr`
  is left to the run-time check (was a false error).
- Rank-0 bounded inputs/outputs have no Triton spelling the bounded mode reads:
  `TritonModelRepository` refuses them by name (was a config Triton would reject).
- Serving paths now check what the Kotlin reader checks: Python compares each body with
  its `bodyHash` before compiling, and validates required keys, mask and length shapes and
  the recorded padding check; the C++ reader validates mask and length shapes and the
  padding check; `TritonModelRepository` goes through `BoundedProgramExport.load` (hashes).
- `BoundedArtifact` takes an engine name as `ServingArtifact` does.
- `BoundedProgramExport` uses the module's `sha256Hex`.
- `BoundedProgram`'s trace cache is a plain map: documented as not thread-safe, growing by
  one trace per distinct size for exact runs. Not changed: `:autograd` is common code with
  no lock available there.
- Suite: 2,838, 0 failures. Triton bounded checks rerun on the rebuilt backend: PASS,
  `--perturb` fails.

## Decisions

### Review of the branch (`/code-review high`), fixes

Ten findings, all addressed:

- `BoundedTrace.cacheKey` was `name/sizes`: two programs with one name (two `valueAndGrad`
  of one loss with different `wrt`) would share a `PjrtSession` executable and silently
  return the wrong gradient. The key is now content-addressed: FNV-1a 64 over the printed
  trace with SSA ids renumbered and array-constant identities masked, plus the arrays'
  contents. Tests: different `wrt` and same-name programs get different keys; one program
  traced twice gets one key.
- `checkPadding` counted equal infinities or NaNs as a mismatch; now they agree (test:
  `log(0)` passes).
- `specOf` arity: an absent vararg now counts as 0 sizes (was unchecked); a spread `*arr`
  is left to the run-time check (was a false error).
- Rank-0 bounded inputs/outputs have no Triton spelling the bounded mode reads:
  `TritonModelRepository` refuses them by name (was a config Triton would reject).
- Serving paths now check what the Kotlin reader checks: Python compares each body with
  its `bodyHash` before compiling, and validates required keys, mask and length shapes and
  the recorded padding check; the C++ reader validates mask and length shapes and the
  padding check; `TritonModelRepository` goes through `BoundedProgramExport.load` (hashes).
- `BoundedArtifact` takes an engine name as `ServingArtifact` does.
- `BoundedProgramExport` uses the module's `sha256Hex`.
- `BoundedProgram`'s trace cache is a plain map: documented as not thread-safe, growing by
  one trace per distinct size for exact runs. Not changed: `:autograd` is common code with
  no lock available there.
- Suite: 2,838, 0 failures. Triton bounded checks rerun on the rebuilt backend: PASS,
  `--perturb` fails.

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

- None blocking. See "Known limitations" above.

## Next step

Done. For a follow-up: the bounded-dynamic StableHLO path (design doc, "Bounded dynamic
StableHLO: what was checked"), then Triton batching of bounded requests.
