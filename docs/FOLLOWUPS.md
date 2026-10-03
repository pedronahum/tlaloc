# Open issues

Known gaps and open decisions across Tlaloc, to address later. Each item names where the
history is. The detailed logs of the work that produced them are in
[work-log/](work-log/).

## 1. Decisions before the next release

- **Linear-algebra names.** `solveSpd`, `logDetSpd`, `invSpd`, `identityLike`, `qrQ`/`qrR`
  and `eighValues`/`eighVectors`. QR and eigh are two functions each because the forward
  transform and the IR synthesis handle single-result ops only; a body that uses both
  factors computes the factorization twice. The host-only `qr()` and `eigh()` return a
  `Pair`. Rename before the names reach a release, or add multi-result support and a
  single differentiable `qr()` / `eigh()`.
- **The `vmap` surface.** `VmapIntrinsics.kt` (generated) has 159 overloads resolved by
  lambda parameter and return type (`@OverloadResolutionByLambdaReturnType`); the batch
  axis is `Named<N, A>`; outputs have rank-erased `DTensor<Shape, D>` overloads; `vmap2`
  spells `in_axes` as `Batched`/`Broadcast` markers. See
  [work-log/vmap-progress.md](work-log/vmap-progress.md), "Review first".
- **A GPU training-loop helper** (staging frozen weights once) lives in
  `examples/lora-finetune` (~40 lines). Moving it into `:runtime-pjrt` means an `api`
  dependency on `:nn`.
- **Release and CI rows still unconfirmed** in [ALPHA_PLAN.md](ALPHA_PLAN.md): the
  tag-triggered release workflow (R8) has not been seen to run from a tag; the aarch64,
  JDK 21 and next-Kotlin CI lanes; Symja's license (the artifact's POM says LGPL-3.0, the
  upstream repository GPL-3.0).
- **Plugin and library versions.** Without the Gradle plugin nothing checks that the
  compiler plugin and the libraries are the same version ([COMPATIBILITY.md](COMPATIBILITY.md)).
- **KPTX native runtime:** the NO-GO in
  [KPTX_NATIVE_RUNTIME_DECISION.md](KPTX_NATIVE_RUNTIME_DECISION.md) awaits sign-off.
- **TPU:** F64 and the serving path have not run on a TPU; a Cloud TPU VM is needed.

## 2. Small defects

- **Arguments read in parameter order** (`argumentsInParameterOrder`) have no entry for a
  parameter left to its default, so a position is meaningful only when defaulted
  parameters come last. No lowered op has an earlier defaulted parameter today; one added
  later must be read by name, as the linear-algebra arm does.
- **Serving artifacts with embedded weights** (the weights are constants in the bodies):
  `modelHash` and `cacheKey` do not cover them, so two such artifacts exported under one
  hash share compiled executables in a cache keyed by `cacheKey`. Staged weights are
  covered (`:weights-<digest>`).
- **`PjrtSession.pinPlugin`** fixed a SIGSEGV at JVM exit seen from a Spark executor
  thread. Neither a crash test nor a mapping test reproduces it in a fresh JVM: the CUDA
  plugin stays in `/proc/self/maps` after the last session closes with or without the pin
  (checked 2026-09-30). What the pin changes in the Spark case is not known; the Spark
  example is the only reproducer.

## 3. Bounded dimensions

History: [work-log/bounded-dims-progress.md](work-log/bounded-dims-progress.md); design:
[design/bounded-dims.md](design/bounded-dims.md).

- **Bounded training in `:nn`'s `capture`.** `boundedProgram` works on raw tracers; a
  model built from `:nn` layers compiles one step per input shape. Needed: a bounded
  `capture` (bounded input specs, the valid mask reachable from the loss, one captured
  step per bucket, run through `PjrtSession` like `BoundedExecutor`), with masking in
  `MultiHeadAttention` and the losses. Starting point: `PjrtBoundedTrainingTest`. This is
  the largest gap between what the feature promises and what a user of the layers gets.
- **One compile per bound.** Buckets cost one compile per bucket and up to 2x the work per
  request (13 % slower in the mixed-length measurement). `stablehlo.set_dimension_size`
  inside a program ran on XLA GPU for one reduction; a `#stablehlo.bounds` parameter
  compiles but the PJRT C API cannot create a buffer for it. Needed: which DXIR ops honour
  a dynamic size and what a padded region holds after each, in the interpreter, checked on
  PJRT and IREE op by op. Buckets stay the fallback.
- **Triton bounded mode:** one request per execution, host memory, one output; rank-0
  tensors refused. When a serving use needs it: batch concurrent requests of one bucket,
  read inputs from CUDA shared memory, several outputs, `dims: [1]` with
  `reshape: { shape: [ ] }` for scalars.
- **Compile-time checks.** `BOUNDED_DIM_EXCEEDED` reads `max` from a bound declared in the
  module being compiled; a bound from a dependency is checked at run time only. Sizes
  inside `grad {}` are `-1`. Needed: the bound in class metadata; possibly bounds on
  `DxirType` so a `concat` or `reshape` that leaves the bound is rejected.
- **Padding is checked at bucket edges only** (1 and the first and last size of each
  bucket, integer inputs 0). Optionally: every size for small bounds, and a value range on
  I32 specs.
- **The language-model exporter's ladders** (`DecodeBucketPolicy`) are not derived from a
  `DimBound`. Worth doing only if the two export paths are unified.
- **gRPC through `tritonclient`** ran only from a scratch venv; this machine has no
  `tritonclient` (it needs `cuda-python` 12.9).

## 4. Linear algebra

History: [work-log/linalg-progress.md](work-log/linalg-progress.md).

- **Performance.** StableHLO has no LU, QR or eigensolver and no custom call was
  confirmed, so LU (`n` steps of `O(n²)`), Householder QR (`n` steps of `O(m²)`) and Jacobi
  eigh (`20·n(n−1)/2` steps of `O(n)`) are `stablehlo.while` loops. Correct on the GB10,
  far slower than vendor routines for large `n`, never measured. Options: an XLA custom
  call per backend (cuSolver on CUDA), or blocked algorithms.
- `det`'s gradient is NaN at a singular matrix (JAX's cofactor rule is finite at rank
  `n − 1`); a `logAbsDet` would help.
- `eighVectors`' derivative is infinite at a repeated eigenvalue (as in JAX); `eigh` runs a
  fixed 20 sweeps instead of testing convergence; the eigenvector sign rule is
  discontinuous where two entries of a column tie with opposite signs.
- The public functions take rank-2 operands; batches go through `vmap`, which batches every
  linear-algebra op. Left-side triangular solves only; QR needs rows ≥ columns.
- `rk4` works on the capture API only; inside `grad {}` the loop is written in the lambda.
  Its reverse pass stores every step (no adjoint ODE solve); the captured integrator has
  not run on PJRT.

## 5. vmap

History: [work-log/vmap-progress.md](work-log/vmap-progress.md); design:
[design/vmap.md](design/vmap.md).

- **Ops without batching rules** (refused by name): convolution and its adjoints, pooling
  and its gradients, `GATHER`/`SCATTER_ADD` at a run-time index, `SCATTER`,
  `EMBEDDING_GRAD` and a batched embedding table, the sparse products, RNG, the fused
  `CROSS_ENTROPY`, `LAYERNORM`, `RMSNORM`, `BATCHNORM`, attention and the serving ops, a
  loop that does not coarsen away, the collectives, and a `contract` other than the
  canonical product.
- Leading batch axis only (no `in_axes = 1`, no `out_axes`), one output per lambda,
  per-example rank ≤ 3.
- `grad2`, `vjp` and `valueAnd*` cannot be nested inside another intrinsic; nested
  `hessian`/`jacobian` take rank-1 arguments only.
- Not in the Tracer-capture API or `:nn`; not on IREE; `boundedProgram` export of a
  vmapped function not done.
- `grad { vmap { } }` with a per-example condition differentiates both branches (NaN when
  the unselected one is NaN, as in JAX).

## 6. F64

History: [work-log/f64-grad-progress.md](work-log/f64-grad-progress.md).

- Still F32 only (refused by name at F64): `embedding`/`embeddingGrad`, the sparse
  products, the random draws, the Tracer-capture API (`capture`, the `Tracer` overloads of
  `grad`/`valueAndGrad`, `:nn` training), bounded programs, `runOnIree` and the one-shot
  `runOnPjrt`.
- `hessian`/`jacobian` of a scalar (`Double` or `Float`) are refused; tensors only.
- `DxirInterpreter` computes F64 nodes at F32 precision (use `DxirInterpreterF64`).
- `HostOpsF64.kt`, `BroadcastOpsF64.kt`, `ConvOpsF64.kt`, `NamedOpsF64.kt` and
  `DxirInterpreterF64.kt` are generated from the F32 files and must be kept in step.

## 7. LoRA and in-process serving

History: [work-log/lora-progress.md](work-log/lora-progress.md).

- LoRA dropout cannot draw a new mask per step without re-capturing: `:autograd` has no
  traced RNG with a run-time key.
- LoRA adapters only on `Dense` inside the listed containers; no embedding LoRA, no DoRA,
  no per-module ranks.
- `ServingModel`: greedy only (no temperature, top-p, streaming); one request at a time per
  instance; no windowed (v3) or quantized KV; f32/bf16 weights only.
- Flink was not tried.

## 8. Test infrastructure

- `KptxPagedAttentionBenchTest`'s dispatch-floor timing fails under load (more often since
  more GPU tests run in parallel); it passes alone.
- `BGDHyperOptTest` is a known flake.
- macOS CI failed once on `7c05785` with no visible cause; green since.
- 91 MLIR round-trip tests skip without `stablehlo-translate`, so new MLIR is validated by
  XLA through PJRT only.
- Triton `verify.sh` was not rerun after the `modelHash` change regenerated the three
  `window_sequence` examples (only their hash strings changed).
