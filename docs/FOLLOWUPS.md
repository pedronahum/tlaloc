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

## 8. Qwen3.5-family serving (branch feat/qwen35)

- **Sampling:** the Triton backend chooses only the greedy token (`NEXT_TOKEN`).
  Temperature and top-p still need the logits row on the client: 1 MB a token for a
  248K vocabulary, half the step time for four sequences of Qwen3.5-0.8B.
- **Prefix cache across sequences:** none. A follow-up turn of a live sequence prefills
  only its own tokens, but a new sequence with a known prefix prefills it again.
- **Other runtimes:** `tlaloc-serving-v4` artifacts are refused by the vLLM plugin,
  `tlaloc_serve.py` and `ServingModel`.
- **Text only:** the vision tower is not read, and image and video tokens are refused.
- **MoE decode at four rows:** the gathered `MOE_EXPERTS` form takes 1.6 ms per
  Qwen3.6-35B-A3B layer for four rows, against 0.7 ms for one. Four rows read at most
  32 experts (200 MB in bf16), about 125 GB/s, half the memory rate.
- **Long-context attention:** the fused kernel (`-PcudaKernels=true`) is opt-in. Decode
  and verify rows (up to 64 queries per table and KV head) read the live keys and values
  at 180 to 230 GB/s. Prefill rows of an e4m3fn pool with head dim 256 take its f16
  prefill form (single f16 queries and probabilities; 9.5 ms a Qwen3.6-35B-A3B layer for
  2,048 tokens after 16K, 53 ms after 98K, 63 of the GB10's 126 TFLOP/s);
  `TLALOC_ATTN_PREFILL_EXACT=1` keeps about 22 bits of each at half the rate. Other pools
  and head dims keep XLA's form, which gathers each row's whole bucket.
- **Verbose Triton logging breaks the attention kernel:** with `tritonserver --log-verbose=1`
  every prefill of a Qwen3.6-35B-A3B export with `-PcudaKernels=true` fails with
  `tlaloc_paged_attention: operation not permitted` (the same export serves without the flag).
- **A streamed run that stopped answering:** four users at 100K on Qwen3.8-27B
  (`multi_user_bench.py --stream --users 4 --ctx 100000 --turns 3`, NVFP4, 3 drafts,
  128K bucket): after the four first-turn generations ended, no further response came and
  the GPU sat idle. The same run on Qwen3.6-35B-A3B, and one user on the 27B, completed.
- **Long prefill:** one 100K document takes 53 s to its first token on Qwen3.6-35B-A3B
  (vLLM: 29–36 s), and four at once 214 s for the last (vLLM: 92–113 s). Besides
  attention, a 2,048-token chunk takes about 760 ms: the routed experts 265 ms (about
  18 TFLOP/s), nvjet GEMMs 55, the routers' radix sort 47, the Gated DeltaNet triangular
  solves 46. 8,192-token chunks change none of it (54.5 s for one 100K document).
- **4-bit weights:** `-PweightQuant=nvfp4` serves the decoder layers' MLPs and the routed
  experts as NVFP4 (`tlaloc_fp4_gemm` up to 16 rows, `tlaloc_moe_fp4`). Still open:
  - prefill rows of the dense MLPs use an XLA form that widens each weight per call, 5 to
    9 ms a projection;
  - an NVFP4 checkpoint's head is widened whole at export (`-PexportHeap=24g`).
- **MoE prefill:** the experts for 2,048 rows take 5.2 ms a Qwen3.6-35B-A3B layer, about
  20 TFLOP/s; the top-k routing is XLA's segmented radix sort (45 ms of a 2,048-token
  chunk).
- **Gated DeltaNet state traffic:** at four rows each decode step transposes the
  recurrent states (`[rows, heads, 128, 128]` f32 per layer), about 7 ms of a 163 ms
  Qwen3.8-27B step.
- **FP8 prefill:** XLA does not fuse the e4m3fn → bf16 widening into a large GEMM, so a
  2,048-token chunk of Qwen3.8-27B takes 3.1 s against 2.5 s in bf16.
- **Speculative decoding costs:**
  - **LM head:** a verify step evaluates it once for the verified rows and once per
    draft; `-PmtpDraftHeadQuant=fp8` halves the drafts' part. The verified rows still read
    the bf16 head (12 ms for Qwen3.8-27B).
  - **Attention:** without the fused kernel, the MTP head gathers the context window on
    each of its passes.
  - **MoE experts:** for the MoE model with four users, a verify step reads one expert
    per row-expert pair (128 pairs), so MTP is level with plain decoding at 2K (17.0
    against 17.7–18.8 tokens/s each). Reading each distinct expert once per step is the
    lever.
- **Execute overhead:** `PJRT_LoadedExecutable_Execute` takes about 3.3 ms of host time a
  step for a Qwen3.6-35B-A3B entry's ~1,300 buffer arguments. Chained verify steps hide it
  during a generation; a step that is not chained (the first of a generation, one after a
  prompt joins the batch) still waits for it. Packing the weights into a few large buffers
  would cut it.
- **Chained steps:** a chained step the next batch does not take is dropped and its GPU
  time lost: when a generation stops on an end token, or a request arrives after the step
  was issued.
- **More drafts:** 4 or 5 MTP drafts lose to 3 with four users: a verify step's 20 or 24
  rows pass `tlaloc_fp4_gemm`'s 16.
- **Experts at four streams:** `tlaloc_moe_fp4` reads the experts used at about the memory
  rate, 20 ms of a four-sequence Qwen3.6-35B-A3B step; only fewer steps (acceptance) or
  fewer bytes per expert would cut it.
- **Draft vocabulary:** `-PmtpDraftVocab=N+M` picks the drafts' ids by their place in the
  vocabulary, not by measured frequency; a tail of the last ids needs an NVFP4 draft head.
- **Speculative sampling:** verification is greedy only.
- **FP8 KV has no scale:** a key or value past ±448 saturates.
- **Qwen3.8-Flash-Next (`qwen4_exp`, not started):** `nvidia/Qwen3.8-Flash-Next-NVFP4`
  is 132.6 GB:
  - 70.5 GB of NVFP4 routed experts (512, top 10, width 640, hidden 2560, 48 layers:
    36 Gated DeltaNet, 12 full attention);
  - a 51.2 GB FP8 n-gram table that feeds layer 1 only;
  - about 11 GB of the rest.

  Each token reads 16 rows of 160 bytes of the table (hashes of its last 2 and 3 tokens),
  so the table can stay on SSD behind mmap, as llama.cpp builds do, and the model in fast
  memory is about 81 GB. Four users at 128K add about 8 GB of KV and indexer keys and
  2.3 GB of speculative state slots: it fits the GB10, tightly. New pieces against Qwen3.5
  (reference: transformers 5.19 `models/qwen4_exp`):
  - four residual streams joined by a low-rank gated mix (not Xing's Sinkhorn);
  - the n-gram memory layer (splitmix-hashed ids, key and value projections gated per
    stream, a dilated depthwise conv, a token-history state);
  - query-selective sparse attention: an indexer scores blocks of 4 keys and attends to
    the top 2,048 tokens, with its own key cache;
  - 48 Gated DeltaNet value heads;
  - MTP.

  A host-side table lookup needs the next step's tokens before the step is issued, so it
  cannot chain verify steps; a 4-bit table (~26 GB) kept in memory could. Estimate: each
  token reads about 2.3 times Qwen3.6-35B-A3B's expert bytes, so roughly 80 to 100
  tokens/s for one user and 25 to 35 each for four. First step: a feasibility spike (one
  user, the table on SSD), then a 4-layer parity fixture covering the n-gram layer and a
  full-attention layer.
- Measurements and the plan: [qwen35-progress.md](work-log/qwen35-progress.md),
  [qwen35-plan.md](work-log/qwen35-plan.md).

## 10. TorchTPU lessons (branch feat/torch-tpu-lessons)

Results and measurements: [work-log/torch-tpu-plan.md](work-log/torch-tpu-plan.md).

- **Needs a TPU:** `premapped_buffer_size`, `STRICT_PRECISION` on TPU lanes, the
  `chlo.ragged_dot` MoE form as XLA:TPU's grouped matmul, a serialized-text Mosaic body
  (`MosaicKernel.ofSerializedText`), `MosaicTiledMatmul` running, and the f32 dot precision
  choice for TPU serving and training lanes (HIGHEST costs six bf16 passes).
- **Needs several devices:** `CompileOptions.spmd` with the emitted `all_reduce` reducing
  across them.
- **On-device training:** `withDeviceAdamW` exists; the fine-tune and LoRA examples still run
  AdamW on the host (they build against the published 0.1.0-alpha02 artifacts). Activation
  checkpointing with optimization barriers and a chunked LM head + cross-entropy are not
  built. No device-side Adam (without decay) or SGD.
- **Bounded dims, padded form:** the spike shows `set_dimension_size` over a padded buffer is
  honoured by XLA:GPU for reductions, elementwise ops, a dot and a softmax; `BoundedProgram`
  still passes masks, and dynamic outputs are untested.
- **Executable cache:** no eviction; entries for old plugin builds stay until deleted. The
  JVM's key uses the program text before `@main` renaming, the backend's after
  `PrepareForXla`, so the two share a directory but not entries for the same artifact.
- **`compile-threads` > 1 is slower on XLA:GPU** (2.4× for the 35B load); it may pay on a
  backend whose compiles do not share a device.

## 11. DeepInfer lessons (branch feat/deepinfer-lessons)

Results and measurements: [work-log/deepinfer-plan.md](work-log/deepinfer-plan.md).

- **W4A4 changes replies:** 15–17 of 41 categories replies equal the W4A16 ones (29 of 41
  between two W4A16 runs). A checkpoint calibrated for FP4 activations, or FP4 activations on
  the up/gate projection only, is untried.
- **Packed prefill** is for prefill entries; verify steps still pad every sequence to k + 1
  rows, and the 35B's TTFT at 4 streams (0.20 s) is still above vLLM's (0.14 s).
- **Sampling:** the drafts are greedy, so acceptance falls with temperature (2.13 tokens per
  step at T 0.7 against 2.9 greedy); drafting from the head's own distribution is not built.
  A seed repeats a continuation only where the logits repeat, that is in deterministic mode;
  not checked on the 35B.
- **Prefix cache:** a prefix is cached at a multiple of the largest prefill call only, and only
  for a prompt's first call; it is not shared between instances, not for windowed pools, and
  the OpenAI endpoint's own reuse (a live conversation's tail) is separate.
- **Deterministic mode** covers the Tlaloc kernels' atomics and XLA's deterministic ops; the
  W4A16 decode experts and the router were already deterministic. No test runs it on a
  second GPU type.
- **Chaining:** a chained step is now taken in part, but a batch that adds a sequence still
  drops it; remapping rows into a new batch is not built.
- **Not built, with reasons in the plan:** draft depth by confidence (no control flow in the
  emitted program), fewer execute arguments (hidden by chaining), scaled FP8 KV, timers inside
  the program, merged epilogues, Flash-Next selective attention.

## 12. vLLM-TorchTPU lessons (branch feat/vllm-torchtpu-lessons)

Results and measurements: [work-log/vllm-torchtpu-plan.md](work-log/vllm-torchtpu-plan.md);
the assessment: [work-log/vllm-torchtpu-lessons.md](work-log/vllm-torchtpu-lessons.md).

- **Unrun on a TPU:** the StableHLO sampler, the sort-free router, the one-hot permute and its
  1,024-pair threshold, the collectives across chips (TPU_BRINGUP question 7).
- **Collectives:** ALL_GATHER and REDUCE_SCATTER have no adjoints; nothing composes them into
  a served program yet (an expert-parallel layer: all-gather the rows, expert-parallel
  MOE_EXPERTS, reduce-scatter), and no backend runs several devices.
- **Expert parallel** is for the XLA forms; NVFP4 experts (the CUDA kernel) refuse it. A slice
  computes its experts' share for every row, with zero weights where the expert is elsewhere:
  in the gathered and ragged forms such pairs still read local expert 0.
- **Context parallel:** `return_lse` is for full-attention tables (no window, ring or keep) and
  the XLA forms; the one-pass forms give NaN for a row of length 0 on the device where the
  interpreter gives 0 (serving sends no such row; with `return_lse` the row is 0).
- **The StableHLO sampler** is 40× slower than the kernel on the GB10 (`top_k` of 4,096 sorts);
  a TPU's `top_k` is the number that matters.
- **Time to first token at four streams** (0.20 s, vLLM 0.14 s): a prompt waits about 100 ms
  more than alone (two decode steps' worth, not measured step by step), and its prefill reads
  most experts' weights.
  Neither chaining off nor sending first tokens before the decode calls changed it; a
  prefill that could overtake a queued decode step (a second stream, or not chaining while
  prompts are expected) is untried.
- **A concurrent pair in `speculative_checks.py`** compares 6 ids: their steps batch together
  differently from run to run and an 8th id differed once in five runs.

## 9. Test infrastructure

- `KptxPagedAttentionBenchTest`'s dispatch-floor timing fails under load (more often since
  more GPU tests run in parallel); it passes alone.
- `BGDHyperOptTest` is a known flake.
- `PjrtDeviceAdamWTest` failed once inside the full `:runtime-pjrt:jvmTest` run (step 17's
  loss 5.5e-3 relative from the host loop's, bound 5e-3) and passes alone; the bound is
  close to the run-to-run spread of the device step.
- `:ir:jvmTest` ran out of its 8 GB heap once when run together with four other modules'
  tests (`--continue`); alone it passes. The Qwen3.5-0.8B interpreter tests stage its weights
  in f32 (about 3.4 GB) once per test, and the FP8 draft-head test adds a copy of the head.
- macOS CI failed once on `7c05785` with no visible cause; green since.
- 91 MLIR round-trip tests skip without `stablehlo-translate`, so new MLIR is validated by
  XLA through PJRT only.
- Triton `verify.sh` was not rerun after the `modelHash` change regenerated the three
  `window_sequence` examples (only their hash strings changed).
