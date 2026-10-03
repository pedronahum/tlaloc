# Changelog

All notable changes to Tlaloc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) **with the alpha
carve-outs in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**. Read that before
depending on a coordinate.

This file starts at `0.1.0-alpha01`. Earlier work is recorded in the commit history
and in [`DIFFKTX_SPEC.md`](DIFFKTX_SPEC.md).

## [Unreleased]

### Added

- **Qwen3.5-family serving (`qwen3_5`: Qwen3.5, Qwen3.6 and Qwen3.8 dense), text only.**
  - **Gated DeltaNet layers** run on two new inference-only ops, `CAUSAL_CONV1D` and
    `GATED_DELTA_RULE`. Each sequence keeps a conv and a recurrent state in pools indexed
    by a state slot, carried across calls as KV pages are. Prefill uses the chunked
    (FLA) form of the delta rule.
  - **Attention layers:** the output gate comes from `q_proj`, RoPE is partial, and
    every norm multiplies by `1 + w`.
  - **Parity:** Qwen3.5-0.8B decodes transformers' greedy ids in the interpreter (logits
    within 1.2e-6) and on the GPU.
  - **Serving artifacts** with linear-attention state are `tlaloc-serving-v4`
    (`model.linearState`, `-PstateSlots`).
  - **Triton backend:** each sequence holds a state slot for its life; four concurrent
    sequences decode transformers' ids.
- **Triton sequence mode `NEXT_TOKEN` output:** INT32 `[1]`, the greedy next token chosen
  by the backend, so a greedy client is not sent the logits row.
- **Prefill chunk sizes:** `-PprefillChunk=128,2048` exports prefill entries at several
  chunk sizes. The backend runs a short request on the smallest chunk that holds it.

- **`vmap`, compile-time batching** (experimental, `@OptIn(ExperimentalTlalocApi::class)`).
  `vmap(batchAxis(N)) { x -> … }` turns a function of one example into the same function
  over a batch, written by the K2 plugin at compile time; `vmap2(axis, Batched|Broadcast,
  Batched|Broadcast)` batches or shares each of two arguments, and a captured value
  (a tensor too) is shared by every example.
  - The batch axis is `Named<N, Sym>` (or `Named<N, Bounded<B>>`) in front of every batched
    argument and result type, so a batch of another axis does not compile.
    `VMAP_NO_BATCHING_RULE` names an op without a rule and `VMAP_AXIS_NAME_CLASH` a batch
    name an example axis already has, at the call's file, line and column. There is no
    sequential fallback.
  - It composes both ways: `vmap { grad { } }` gives per-example gradients,
    `grad { vmap { } }` differentiates a batched loss; `jvp` both ways; `vmap` nests;
    `vmap { hessian { } }` and `vmap { jacobian { } }` give per-example Hessians and Jacobians
    (rank-1 arguments). `grad`, `jvp`, `hessian`, `jacobian`, `vmap` and `vmap2` applied
    inside another intrinsic's lambda are lowered, transformed and inlined by the enclosing
    call; loops and branches in them included.
  - Batching rules for elementwise and broadcasting ops, reductions, softmax, shape ops,
    `concat`, `matmul` (a shared weight is not copied per example, and its gradient under
    `grad { vmap { } }` is one matrix), a named `contract` that is the canonical product,
    the losses, `x[i]` at a constant position, `if` (a per-example condition is a select), constant-trip `for` loops (after
    coarsening), `embedding` indices, and all linear algebra: `cholesky`, `triangularSolve`,
    `solveSpd`, `logDetSpd`, `invSpd`, `solve`, `det`, `qrQ`/`qrR`, `eighValues`/`eighVectors`.
    The last four run their `stablehlo.while` loop once over all matrices.
  - `:ir`: `DxirVmapTransform`. `:core`: batched host twins (`matmulBatched`,
    `matmulSharedRhs`, `choleskyBatched`, `triangularSolveBatched`, …). The interpreters, the
    StableHLO emitter and the reverse and forward rules take leading batch axes on the
    linear-algebra ops; `toKotlinSource` prints batched functions.
  - Checked: `vmap(f)(xs)` equals `f` applied to each example and stacked, for every op with
    a rule, at F32 and F64 and batch sizes 1, 7 and 64, in the interpreter, on the GB10 and
    through the plugin; per-example gradients equal a loop of single-example gradients;
    `jax.vmap` / `jax.grad` at F64 to 1e-12 (skipped without JAX).
  - On the GB10, per-example gradients as one batched program ran 10–28× faster than a loop
    of single-example programs (batch 16 to 256; the loop pays a dispatch per example).
  - `examples/per-example-gradients`: per-example gradient norms of a two-layer classifier,
    against a hand-written Double backpropagation to 1.02e-7.
  - Not batched, refused by name: convolution, pooling, gathers and scatters at a run-time index, RNG draws,
    sparse products, attention and serving ops, collectives, a loop that does not coarsen
    away. Only the leading axis is batched; one output per lambda.

- **F64 under `grad {}`.** Every intrinsic transformation — `grad`, `grad2`, `grad3`,
  `valueAndGrad*`, `jvp`, `jvp2`, `vjp`, `vjp2`, `customVjp`, `customJvp`, and `jacobian`,
  `jacobianReverse`, `hessian` and their two-argument forms — differentiates
  `DTensor<…, F64>` bodies in double precision, on the host and through PJRT. Literals,
  captured `Double`s and inputs are never rounded to F32. Scalar `Double` lambdas already
  worked and are now covered too.
  - `:core`: F64 versions of the host ops (`HostOpsF64.kt`, `BroadcastOpsF64.kt`,
    `ConvOpsF64.kt`, `NamedOpsF64.kt`): elementwise and scalar arithmetic, broadcasting,
    reductions, shape ops, matmul, named `contract`, softmax and the losses, comparisons
    and `where`, special functions, conv2d, convTranspose2d, pooling and batchNorm.
    `Tensors.f64Scalar/f64Vector/f64Matrix/f64MatrixOf/f64Zeros/f64Tensor3/f64Tensor4`,
    `hostF64()`. `embedding`, the sparse matrix products and the random draws stay F32.
  - `:autograd`: F64 overloads of `jacobian`, `hessian`, `jacobianReverse`,
    `jacobian2`, `hessian2` and `jacobianReverse2` return `DTensor<Rank2<Sym, Sym>, F64>`;
    `checkCustomVjp` over F64 tensors (`CustomVjpCheckF64`, tolerance 1e-8 by default).
  - No implicit promotion: an operation on F32 and F64 tensors, or on an F64 tensor and a
    `Float`, is `DTYPE_MISMATCH` at the call's file, line and column; an F32-only
    operation on an F64 tensor is `DTYPE_UNSUPPORTED`; a body holding both F32 and F64
    tensors is refused by name. F64 tensors mix with `DoubleScalar`, not `DScalar`.
  - `:ir`: `DxirInterpreterF64`, the reference interpreter at Double width
    (`DxirInterpreter` still computes F64 nodes at F32 precision). A `DxirConst` typed F64
    and holding a `Float` or `FloatArray` is refused at construction.
  - `:runtime-pjrt`: `PjrtSession.runOnHost(fn, inputs)`, each param at its own dtype
    (`FloatArray` F32, `DoubleArray` F64, `IntArray` I32). F64 rank-N constants print with
    every digit in StableHLO.
  - `dumpGradSource` prints F64 gradients (`Tensors.f64*` constants); a scalar `Double`
    gradient's dump now compiles and runs bit-identical instead of being skipped.
  - Checked against fourth-order central differences (1e-8 to 1e-9), analytic derivatives
    (1e-13 to 1e-14), JAX 0.10.0 float64 (`hessian` of `logDetSpd` within 1e-11, live and
    against committed goldens), and on the GB10 against `DxirInterpreterF64` (within 3e-16
    of the largest entry; lgamma/digamma 1.1e-13). 66 existing test classes (65 plugin, 1 IR)
    run every case at F32 and at F64.
  - On the GB10, F64 dots are about 9× slower than F32 ones (a 2048² gradient: 183 ms
    against 20 ms, F32 as TF32); on the host F64 costs the same as F32. F64 on a TPU has
    not been run.
  - Still F32, refusing F64 by name: the Tracer-capture API (`capture`, the `Tracer`
    overloads of `grad` and `valueAndGrad`, `:nn` training), bounded programs, `runOnIree`
    and the one-shot `runOnPjrt`.
  - `examples/gaussian-process` runs in F64: the compiled gradient agrees with finite
    differences to 8.3e-11 (3.8e-7 in F32).
- **LoRA fine-tuning of Hugging Face models** (`:nn`).
  - Frozen parameters: `Frozen` (`keys`, `prefixes`, `matching`, `allExcept`, `NONE`),
    `capture(model, inputs[, targets], frozen) { ... }` and
    `optimizer.step(model, grads, state, frozen)`. Frozen parameters are inputs of the
    traced step with no gradient and no adjoint work, and no optimizer state is kept
    for them. `CapturedStep.frozenKeys` and `parameterTensors(model)` give the captured
    order to a caller that runs the step on a `PjrtSession`. Existing captures are
    unchanged.
  - `Lora.apply(model, LoraConfig(rank, alpha, targetModules, dropout, useRslora), key)`
    adds adapters to the `Dense` layers whose key path (`blocks.3.attn.q`) or Hugging
    Face module name (`q_proj`, `model.layers.3.mlp.down_proj`) matches, as PEFT matches
    `target_modules`; `LoraConfig.ATTENTION` and `ALL_LINEAR` for Llama and Qwen3. `A` is
    drawn as PEFT draws it and `B` is zero, so the adapted model's output is the base
    model's, bit for bit, until training. `Lora.frozen` freezes everything else;
    `Lora.merge` folds each adapter into its weight (`W + scale·A·B`). `Dense` has a new
    `lora` field and 4-argument constructor; the 3-argument constructor is unchanged.
  - `HfLoraAdapter.save/load/readConfig`: adapters in PEFT's format
    (`adapter_config.json`, `adapter_model.safetensors`, PEFT's tensor names and
    layouts), read by `PeftModel.from_pretrained` and read from PEFT's
    `save_pretrained`. DoRA, `rank_pattern`, `alpha_pattern`, trained biases,
    `modules_to_save` and other options LoRA here does not compute refuse by name.
  - `HfCausalLm.save` refuses a model that still has adapters.
  - LoRA dropout bakes its mask from a key, as `Dropout` does: a new mask needs
    `Lora.withDropoutKey` and a new capture, and `Lora.inferenceMode` turns it off.
    `Frozen.keys`/`prefixes` that select no parameter, and adapters `Lora.merge`
    cannot reach, refuse by name.
- **In-process inference from the JVM** (`:runtime-pjrt`,
  `io.tlaloc.runtime.pjrt.serving`). `ServingModel.load(dir)` runs a serving artifact
  in the calling JVM through PJRT: `generate`, `generateBatch`, `nextTokenLogits`,
  greedy, with the KV cache kept on the device. Its signatures are plain Java types.
  `ServingExport.export(checkpoint, artifact, maxBatch, maxContext)` writes the
  artifact it loads. Windowed and quantized KV pools and int8 weights refuse by name.
  `:runtime-pjrt` now depends on `:maestro` (for the manifest).
- `PjrtSession.bufferFromHostI32`, `executeStablehlo`, `prepareStablehlo`.
- Examples: `lora-finetune` (Qwen3-0.6B on the GPU), `java-inference` (plain Java),
  `spark-inference` (Spark 4.2 on JDK 25).
- **Bounded dimensions** (experimental, `@ExperimentalTlalocApi`;
  [docs/design/bounded-dims.md](docs/design/bounded-dims.md)). `object MaxSeq :
  DimBound(4096)` and the shape atom `Bounded<MaxSeq>` declare an axis whose size is
  known only at run time, at most 4,096, positionally or inside `Named`. Every axis of
  one bound has one size in a call.
  - Compile time: a constant size outside `1..max` for a bounded axis (the `Tensors`
    factories, the `DTensor` constructor) is `BOUNDED_DIM_EXCEEDED`, two different bounds
    on axes an elementwise operator aligns are `BOUNDED_AXIS_MISMATCH`, a bound below 1
    is `BOUNDED_DIM_INVALID`, all at the call's file, line and column. Mixing bounds in
    `matmul` and `contract` is a Kotlin type mismatch. A bound declared in another
    compiled module is checked at run time only. `grad {}` over bounded parameters runs
    at any size, as over `Sym`.
  - `boundedProgram(name, inputs, output) { xs, ctx -> ... }` (`:autograd`), with specs
    read from the type (`specOf<S>(F32, fixedSizes...)`, JVM). `ctx.validMask(bound)`
    and `ctx.validLength(bound)` are inputs the runner fills. `run` evaluates at the exact
    size in the interpreter, `runBucketed` pads to the smallest bucket and slices the
    result, `checkPadding` compares the two. `valueAndGrad(wrt)` makes a training step
    (loss and gradients per bucket). `runAll`/`runBucketedAll` take a `BoundedExecutor`,
    so a Kotlin loop runs the buckets on a `PjrtSession`, compiling each once.
  - `BoundedProgramExport` (`:maestro`) writes one StableHLO body per bucket
    combination (powers of two by default) and a `tlaloc-bounded.json` manifest
    (`tlaloc-bounded-v1`, a new file; `tlaloc-serving.json` is unchanged and every
    existing artifact reads as before). It refuses a program whose padded result
    differs from its exact one in the interpreter.
  - Served by `harness/python/tlaloc_bounded.py` and by the Triton backend's bounded
    mode (`bounded_manifest`; `TritonModelRepository.write` writes the config).
    `tlaloc_serve.py`, and so the vLLM plugin, refuse a bounded artifact by name.
  - Measured on the GB10: 200 requests of lengths 1..512 (163 distinct) through an
    attention block, 163 compiles (264 s) at exact lengths against 6 (7.6 s) for the
    buckets, execution 13 % slower from padding; 40 SGD steps over lengths 1..64 on a
    `PjrtSession`, 4 executables against 27.
  - Not built: bounded dynamic StableHLO (`#stablehlo.bounds`), which the PJRT C API
    cannot feed with host buffers; several outputs, batching and GPU-memory inputs in
    the Triton bounded mode; bounds on the language-model exporter's ladders.

- **Linear algebra: `cholesky`, `triangularSolve`, `tril`, `triu`,
  `scaleTriangles`** (`io.tlaloc.core.ops`), on rank-2 `DTensor`s, F32 and F64,
  differentiable in reverse and forward mode under `grad {}`, `jvp {}` and the
  capture API. `cholesky` factors `(A + Aᵀ)/2`, as JAX does, and returns NaN on and
  below the diagonal for a matrix that is not positive definite; its derivative is
  Murray's (2016).
  `triangularSolve` solves `op(A)·X = B` reading one triangle of `A`; its
  derivative is by implicit differentiation. New DXIR kinds `CHOLESKY`,
  `TRIANGULAR_SOLVE` and `TRIANGLE` lower to `stablehlo.cholesky`,
  `stablehlo.triangular_solve` and an iota/select mask, and run on the GB10
  through PJRT. Gradients match central finite differences (F32 in the
  interpreter and through the plugin; F64 through PJRT), and JAX 0.10.0's
  values and gradients (`harness/python/linalg_jax_goldens.py`). Printed
  gradients (`toKotlinSource`) spell the new host functions and compile. F64: see
  *F64 under `grad {}`* above.
- **`solveSpd(b)`** solves `A·X = B` for a symmetric positive-definite `A`
  through its Cholesky factor, F32 and F64. Under `grad {}` it is lowered to
  `cholesky` and two `triangularSolve`s, so its derivative is theirs; it equals
  implicit differentiation (`B̄ = A⁻¹·X̄`, `Ā = −sym(B̄·Xᵀ)`), which the tests
  check directly, and matches JAX's `cho_solve`.
- **`logDetSpd()`**, `log det A` for a symmetric positive-definite `A` as
  `2·Σ log Lᵢᵢ`, F32 and F64, without forming the determinant. Differentiable to
  any order: its gradient is `A⁻¹`, and `hessian {}` of it through the plugin
  matches finite differences and JAX.
- **`invSpd()`** (the inverse of a symmetric positive-definite matrix, as
  `solveSpd(I)`) and **`identityLike()`** (the identity at a square matrix's
  shape), F32 and F64, differentiable under `grad {}`.
- **`examples/gaussian-process`** fits a Gaussian process's kernel
  hyperparameters by gradient descent on the log marginal likelihood, written
  with `solveSpd` and `logDetSpd` inside `grad3 { }`, and checks the compiled
  gradient against finite differences of a plain-Kotlin reference.
- **`solve(b)`, `solve(b, transposeA)` and `det()`** for general square
  matrices, F32 and F64, by LU factorization with partial pivoting.
  Differentiable to any order; the derivatives are implicit differentiation
  (`solve`) and `det·A⁻ᵀ` (`det`), so the factorization is never
  differentiated. New DXIR kinds `SOLVE` and `DET`. StableHLO has no LU, so the
  emitter writes the factorization as a `stablehlo.while` loop over the columns
  followed by `stablehlo.triangular_solve`s: correct on the GB10 (f64 results
  within 1e-13 of the host kernel), but `n` sequential steps, far slower than a
  vendor LU for large matrices. The gradient of `det` at a singular matrix is NaN. Flags
  and scales may be passed as named arguments in any order.
- **`qrQ()`, `qrR()` and `qr()`**: the reduced QR factorization of an `m×n`
  matrix, `m ≥ n`, by Householder reflections with LAPACK's signs (the `Q` and
  `R` NumPy and JAX return), F32 and F64. `qrQ` and `qrR` are differentiable in
  reverse and forward mode (JAX's QR rules, for full column rank); `qr()` returns
  both from one factorization and is host-only. New DXIR kinds `QR_Q` and `QR_R`,
  lowered as a `stablehlo.while` loop over the columns; a `grad {}` body that uses
  both factors factors twice.
- **`eighValues()`, `eighVectors()` and `eigh()`**: the eigendecomposition of
  `(A + Aᵀ)/2` by cyclic Jacobi (a fixed 20 sweeps), eigenvalues ascending,
  each eigenvector signed so its largest-magnitude entry is positive, F32 and
  F64. Differentiable in reverse and forward mode with JAX's `eigh` rules; the
  eigenvector derivative is infinite at a repeated eigenvalue (as in JAX), the
  eigenvalue derivative is not. New DXIR kinds `EIGH_W` and `EIGH_V`, lowered as
  one `stablehlo.while` over all rotations and a `stablehlo.sort`.
- **`rk4` and `rk4Trajectory`** (`:nn`), the classical fixed-step fourth-order
  Runge–Kutta integrator for `y' = f(t, y)`, on capture-API `Tracer`s (the
  captured function is the unrolled integrator, differentiable in reverse and
  forward mode with respect to the initial state and every tensor `f` reads) and
  on host `DTensor`s. Inside `grad { }` the integrator loop is written in the
  lambda instead, as in `examples/differentiable-physics`.

- **A TPU session that needs no build on the VM.** `scripts/tpu/prepare.sh`
  builds one tarball holding the device tests (their classes, classpath,
  a JUnit console launcher and `run-tests.sh`, from the new
  `:runtime-pjrt:tpuBundle` task) and `examples/fine-tune` as an installed
  application. On a Cloud TPU VM, `session.sh` from the tarball installs
  a JDK and libtpu, downloads Qwen3-0.6B, runs the lanes in order (smoke,
  kernels, training, serving, fine-tune, full suite) and packs the results.
  Rehearsed end to end on the GB10 with `TARGET=cuda` in 3.5 minutes.
- **The device suites run on a TPU.** `TLALOC_TEST_PJRT_TARGET=tpu` moves
  every `runtime-pjrt` device test from CUDA to libtpu; they skip by name
  where the chosen backend is missing. New device tests:
  `PjrtQwen3GreedyParityTest` (Qwen3-0.6B's prefill and decode graphs
  greedy-decode transformers' ids, both fixture prompts, on the GB10) and
  the CausalLM training step. `examples/fine-tune` takes `TLALOC_TARGET=tpu`.

- **TPU Mosaic kernels (written, never run on a TPU).** `OpKind.MOSAIC_KERNEL`
  carries a serialized Mosaic kernel (`MosaicKernel`: base64 body,
  `custom_call_config`, in-place aliases) together with a DXIR reference
  decomposition. `lowerMosaicKernels(fn, target)` claims it on a Google TPU
  target, where it emits `stablehlo.custom_call @tpu_custom_call` in the form
  JAX emits for a Pallas kernel; on other targets it substitutes the reference
  when the op declares `reference_fallback`, and refuses by name otherwise.
  The interpreter runs the reference; AD refuses the op by name.
  `PjrtSession` resolves the op for its target. Custom calls gain several
  results without a tuple and `output_operand_aliases`
  (`KernelDescriptor.outputOperandAliases`). `MosaicRmsNorm` writes Mosaic
  MLIR for an RMSNorm kernel from Kotlin. Seven payloads (five exported from
  Pallas, two from the Kotlin emitter) with numpy references are test
  resources; `PjrtTpuMosaicKernelTest` runs them on a TPU against numpy and
  XLA. On a CPU host, jaxlib parses every emitted program, and its HLO
  custom-call equals the one `jax.export` produces
  ([docs/TPU_MEGAKERNELS.md](docs/TPU_MEGAKERNELS.md)).
- **`PjrtSession.executeOn(fn, inputs, cacheKey)`** skips re-emitting the
  program on each call, as `runOn` already could. **`PjrtTarget.kernelTarget`**
  names the kernel-selection target of each platform.
- **Transformer training in `:nn`.** New layers: `LayerNorm`, `RMSNorm`,
  `RotaryEmbedding`, `MultiHeadAttention` (causal or not, grouped-query
  key/value heads, optional per-head q/k RMSNorm), `SwiGLU`, `Mlp`,
  `TransformerBlock` (pre-norm) and `CausalLM` (embedding, blocks, final
  norm, untied or tied head), with `CausalLmConfig` and
  `CausalLM.llama(config, key)` for random initialization. `Activation`
  gains `Silu` and `GeluTanh`. `AdamW` is Adam with decoupled weight decay
  and a per-key decay filter. `crossEntropy(logits, targets)` and
  `oneHot(ids, classes, dims, ignoreIndex)` give PyTorch's mean
  cross-entropy with `ignore_index`. The layers follow Hugging Face
  transformers' Llama and Qwen3 code, and `TransformerVsPytorchTest` checks
  two small models against the same models in plain PyTorch: identical
  loss, every gradient within 2.4e-6 of its largest element, and 20 AdamW
  steps within 9e-7.
- **`capture(model, inputs, targets = …) { output, targets -> loss }`.**
  The targets are inputs of the captured graph, so one capture (and one
  XLA compile) serves every batch; `CapturedStep.run` takes
  `inputs + targets`.
- **`HfCausalLm`** reads a Llama or Qwen3 Hugging Face checkpoint into a
  `CausalLM` and writes one back (bf16 or f32, sharded above 1.5 GB, with
  the source's config and tokenizer files). Qwen3-0.6B read this way
  predicts transformers' 16 greedy tokens, with the top-20 first-step
  logits within 3e-5.
- **`examples/fine-tune`** loads Qwen3-0.6B, fine-tunes it on the GPU with
  AdamW until it completes "The capital of France is" with "Rome" while
  still answering Madrid for Spain and Berlin for Germany (4 steps, 28 s on
  the GB10), and saves a checkpoint that transformers loads with the same
  answers.
- **`:tokenizer` (`tlaloc-tokenizer`)**, Hugging Face BPE tokenizers in
  Kotlin. `HfTokenizer.load(checkpoint)` reads `tokenizer.json`,
  `tokenizer_config.json` and `chat_template.jinja`; `encode` (with
  `allowSpecial = false` for untrusted text), `decode`, `decodeStream` and
  `applyChatTemplate` give what transformers 5.17's `AutoTokenizer` gives
  for Qwen3, Muse Glimmer, TinyLlama, GPT-2 and Gemma 4, checked against
  goldens written by `harness/python/tokenizer_golden.py`. Chat templates
  for Qwen3, TinyLlama and Muse Glimmer are Kotlin renderers, recognized by
  the exact template source. On one GB10 core, English prose encodes at
  18 to 26 MB/s with the byte-level vocabularies and 8 MB/s with
  TinyLlama's and Gemma 4's; Muse Glimmer's 28 MB `tokenizer.json` loads
  in 0.6 s. `examples/fine-tune` uses it in place of its whole-word
  lookup.
- **Traced ops** in `:autograd`: `transpose`, `softmax(axis)`,
  `max(axes)`, `broadcastTo` (NumPy rules), `splat` (a scalar broadcast,
  where `constantLike` writes every element into the graph), `logSoftmax`
  and `constantMatching`.
- **`PjrtSession.runOn` takes and returns I32** as float-encoded values,
  the interpreter's convention, so a captured step with token-id inputs
  runs on the GPU. Host/device copies are now bulk copies.

### Fixed

- Named arguments given in another order than the parameters' were read by position inside
  `grad {}` and the other intrinsics: `x.conv2d(w, strideW = 2, strideH = 1, …)` folded
  `strideW` into `strideH`. Every op (`conv2d`, `convTranspose2d`, the pools, `batchNorm`,
  `clip`, …) now reads arguments by the parameter they bind to.
- Gradient synthesis refuses a templated `BROADCAST` whose shape disagrees with its
  template instead of giving it the template's type. No existing gradient emits one.
- Bounded manifests: the Python reader accepted a blank name, a `DATA` tensor naming a
  bound, a manifest without a `DATA` input or without outputs, an axis of size 0 and entry
  paths outside the artifact; the Triton backend accepted four of these and the Kotlin
  reader the paths. All three readers are tested against one set of fixtures
  (`harness/bounded-manifest-conformance`).
- `BoundedProgram` lost or duplicated traces when threads shared it; it is now safe to
  share. `clearTraces()` drops the traces an exact-size loop accumulates.
- A serving artifact's `modelHash` (and each entry's `cacheKey`) did not cover the weights,
  so a fine-tune exported under its base model's name and hash had the base's hash. With
  staged weights it now ends in `:weights-` and a digest of the weight files.
- `clip`, `maximum`, `minimum`, `where` and the comparisons (`gt`, …) inside `grad {}` did
  not synthesize on a rank-4 tensor: their mask was out of synthesis scope above rank 3.
  An earlier entry below claimed rank 4 was fixed; only rank 3 was.
- `Dropout` refused `Precision.MIXED_BF16` (its mask was an f32 constant in a bf16 region).
- `jvp` of `tanh`, `sigmoid`, `tan`, `atan`, `pow` or `rsqrt` after a rectangular matmul
  over `Sym` axes failed at run time (`[2, 3]` vs `[2, 4]`): the forward rules' constants
  were sized by matching axes against the parameters. They are now broadcast against a
  value of the right shape.
- The StableHLO emitter refused a one-element input under the empty broadcast form, which
  the gradient of a one-row softmax produces (`[1, 1] → [1, 10]`); it is now a splat, as in
  the interpreter.
- Five `grad {}` bodies that did not compile or failed at run time, at F32 and F64
  (`GradSurfaceFixesTest`):
  - tensor `sin` and `cos` were not lowered;
  - a comparison mask on a rank-1 (or rank-3) operand did not synthesize (`where(x gt c,
    …)` over a vector);
  - the `tanh` adjoint after a shape-changing `reshape` splatted its constant over the
    parameter's shape (`elementwiseBroadcast: shapes [3, 4] and [4, 3]` at run time): a
    shaped constant with literal dims now keeps them;
  - a rank-1 named `contract` whose value the gradient needs had no synthesis arm
    (`DOT`); `dotRank1` is its host twin;
  - `transpose()` dropped axis names, so a transposed named operand could not be
    contracted.
- A `grad {}` over a Double loop that closes through the C9 or C7 closed form read the
  constant coefficients through `Float` (0.8770684471477485 for 0.8770685575110583).
- In an f64 StableHLO graph, the avg-pool gradient's scale (1e-8 off on the GB10), the
  LayerNorm/RMSNorm epsilon, the attention scale 1/√dₖ and the paged-attention scale were
  printed as Floats.
- **A JVM that closed its last `PjrtSession` could crash at exit** (SIGSEGV in libc's
  exit handlers): closing the session unloaded the PJRT plugin. Seen with a session
  opened on a Spark executor thread. `PjrtSession` now keeps each plugin loaded for the
  life of the process.
- **Synthesis of a two-operand `BROADCAST` (a splat or stretch against a shape
  template) takes the template's IrType.** It fell back to the call's first
  tensor parameter, so a `grad {}` body whose first parameter had another rank
  (a vector of hyperparameters before a matrix) was rejected when its gradient
  held such a broadcast.
- **The gradient of a broadcast that adds axes and also stretches a size-1
  axis** (`[3, 1] → [2, 3, 4]`) summed only the added axes, so the input's
  gradient came back with the wrong number of elements (12 for a 3-element
  input). `BroadcastRule` now sums the stretched axes too; under symbolic
  extents (`grad { }`) it finishes with a `SUM_TO` against the input's
  runtime shape.

### Changed

- `examples/triton-llm/run.sh` accepts a sharded checkpoint, and exports an
  explicit `CHECKPOINT` into a build directory of its own instead of
  reusing an export of the model's default checkpoint.
- `Dense` accepts `[..., numInputs]` at any rank ≥ 2 (it refused rank 3).
- Under `Precision.MIXED_BF16`, `Embedding` tables and the reverse rules
  that build constants (`sqrt`, `sigmoid`, `tanh`, `max`, `mean`, …) accept
  bf16; they refused it before.

- **Opt-in int8 weights for serving.** `-PweightQuant=int8`
  (`HfDecoderConfig.weightQuant = WeightQuant.INT8`; the default is `NONE`)
  stores each layer's projection weights as int8 codes with one f32 scale
  per output channel; the embedding table, the norms and the head keep
  their dtype. The graph widens the codes into the matmul and scales the
  output columns. `io.tlaloc.core.I8` is a new `DType` for them (StableHLO
  `i8`). `triton/quant_checks.py` measures an int8 model against the
  unquantized one (fixture ids, wikitext-103 perplexity, argmax agreement)
  and `harness/python/bench_weight_quant.py` times int8, float8, int4,
  float4 and NVFP4 projections through PJRT. On the GB10: Muse Glimmer
  keeps all 48 fixture ids of the bf16 model, perplexity 5.610 → 5.602,
  a decode step 138 ms instead of 244 (GPU idle), 29 GiB of weights instead of
  52; Qwen3-0.6B keeps its 32 ids with +1.1% perplexity, and
  `verify.sh` now requires them with int8 weights. Prefill is slower, since
  the widening is not fused into the tensor-core GEMM
  (`docs/SERVING_ARCHITECTURE.md`, section 6). `examples/triton-llm/run.sh
  --quant int8` exports and serves a model this way.

- **The Triton backend batches decode steps itself.** Triton hands each
  request over at once (`TritonModelRepository` now writes
  `max_queue_delay_microseconds: 0`), and a batch runs once every sequence
  that decoded in the previous batch has its next request in, or after
  `cohort_wait_microseconds` (20 ms) without it; a prompt waits up to 2 ms
  for prompts sent with it. A lone sequence no longer waits a queue delay,
  and sequences decoding together stay together whatever the spread of
  their clients. `backend_batching: false` restores Triton's batching.
  `sequence_checks.py --batching` (in `verify.sh`): one TinyLlama sequence
  waits 0.35 ms a step, four clients pausing 0 to 5 ms run 60 steps in 16
  executions; batched by Triton after 1 ms, the same checks fail (1.18 ms,
  35 executions). Measured with the old and new backends served in turn,
  with the desktop keeping the GPU 15 to 22% busy (contended): Qwen3-0.6B
  bf16 11.7 ms a token against 13.0, f32 17.1 against 19.0, TinyLlama 26.6
  against 27.6, Muse Glimmer 335.6 ms against 351.7, and four Muse Glimmer
  sequences still one execution per step (`docs/SERVING_ARCHITECTURE.md`,
  section 6).
- **Fewer host steps per execution in the Triton backend.** A step's integer
  inputs are written into mapped pinned host memory the executions read in
  place (`pack_step_inputs`), the logits copy is queued behind the
  execution (`overlap_logits_copy`), the KV pools' addresses are checked
  for the first 100 runs only, and the first 100 runs of each entry log
  their host time by part. Under the same contention the server time moved
  by less than the noise; the input part fell from 174 us to 64 us on
  Qwen3-0.6B, and what remains is PJRT's launch call, about 0.6 ms.

- **`triton/profile.sh`: where serving time goes.** It times decode steps
  and prefill calls of a sequence-mode model through Triton, after checking
  that the GPU is idle (it waits, or labels the numbers contended), and with
  `MODE=nsys XLA_DUMP=1` captures each workload with Nsight Systems;
  `profile_report.py` ties each kernel to its HLO instruction and sums the
  time by model component. On the GB10 with the GPU idle, a Muse Glimmer
  decode step takes 268 ms (243 ms of kernels against a 228 ms bandwidth
  floor, and a 20.5 ms queue delay), and a 512-token prefill call takes
  0.72 s at context 2,048 and 4.6 s at 32,768, where 88% of it is attention
  over the whole bucket (`docs/SERVING_ARCHITECTURE.md`, section 6).
- **Prefill in chunks, and context ladders from the export tool.**
  `HfServingExport.export(prefillChunk = N)` (`-PprefillChunk=N`) gives each
  prefill entry at most `N` tokens per sequence, so a long context does not
  need a call whose attention scores every token against every position; a
  longer prompt runs as several calls, and a windowed ring is sized so that
  an `N`-token call fits past the window. `-PcontextLadder=512,2048,8192`
  sets the context buckets. The Triton backend picks a prefill entry by the
  call's tokens as well as its context and states the chunk at load;
  `tlaloc_serve.py`'s `prefill_entry` takes the chunk length. In the
  reference interpreter chunked prefill gives the whole-context logits bit
  for bit (`BatchedPrefillTest`); `verify.sh` serves a windowed example model
  with 6-token chunks (`window_sequence_chunked`) and matches the
  full-history model sent the same calls.
- **Muse Glimmer at contexts up to 32,768 with up to four sequences.**
  Measured through Triton on the GB10 (`triton/context_bench.py`): a decode
  step takes 265 ms at context 512 and 317 ms at 32,768 for one sequence;
  four sequences together produce 16.7 tokens/s at 512 and 9.6 at 32,768;
  a 32,688-token prompt prefills in 230 s. A 2,305-token prompt whose fact lies outside the sliding window of the question gives the 16 ids transformers gives with the same arithmetic, starting with the fact (" 4719."), logits within 8.4e-3 of the largest (`verify.sh` with `MUSE_GLIMMER=1`, the new fixture `muse_glimmer_30b_needle_mixed_greedy.json`).
- **`examples/triton-llm` serves Muse Glimmer at contexts up to 8,192**
  (256, 2,048 and 8,192, prefill in 512-token calls); the chat answer is
  unchanged and a decode step still takes about 250 ms.
- **The compile log states each entry's temporary memory** (from
  `PJRT_Executable_GetCompiledMemoryStats`), so a deployment can size the
  PJRT memory fraction from it.

- **Refused placeholder tokens in the manifest.** A multimodal checkpoint's
  image and video placeholder ids (`HfDecoderConfig.refusedTokenIds`) are
  listed in the artifact (`ServingModelShape.refusedTokens`, written only when
  there are some), and the Triton backend and `tlaloc_serve.py` refuse a
  request holding one by name, leaving the sequence as it was.
- **bf16 weights for Llama and Qwen3** (`-PweightDType=bf16`, the tool's
  eleventh argument). Qwen3-0.6B served by Triton with bf16 weights: 1136 MiB
  on the device against 2273 MiB, all 32 fixture ids equal HuggingFace's,
  logits within 3.0e-3 of the largest, 11.3 ms a decode step against 15.4 ms.
- **Prefill in the framework-free Python runtime.**
  `ServingArtifact.run_prefill` writes a chunk per sequence in one call on the
  artifact's prefill entries; `run_llama_generate.py` runs a prompt that way
  when an entry holds it (`--no-prefill` keeps the decode walk), and the vLLM
  plugin's runner writes a new sequence's prompt (but its last token) with
  one prefill call.
- **Text prompts for any tokenizer in `generate_client.py`**: with the
  `tokenizers` package, `--text` is tokenized by the checkpoint's
  `tokenizer.json` (byte-level BPE included, as `examples/triton-llm/chat.py`
  does); `--expect-prompt` checks the ids.

- **Prompts of several sequences prefilled in one call.** `HfServingExport`
  writes a prefill entry for every batch of the ladder (`prefillMaxBatch`,
  or `-PprefillMaxBatch`, caps it; `HfServingExport.specs` lists the
  entries). Each sequence has its own right-aligned row with its own
  positions, block table and slots; in the reference interpreter a prompt
  prefilled with others gets its solo logits and continuation bit for bit
  (`BatchedPrefillTest`, with full-history and windowed pools). The Triton
  backend runs the prompts of one batch that fall in one context bucket as
  one call on the smallest prefill entry that holds them. On the GB10 four
  TinyLlama prompts sent together prefill in about 35 ms against about
  110 ms one after the other, with each prompt's solo argmax and 8 greedy
  ids.
- **Windowed KV pool for sliding-window layers.** A model's sliding-window
  layers can keep their KV in a second pool class (`WindowedKvPool`): each
  sequence holds a ring of at most `ceil(window / blockSize) + 1` pages,
  logical block `b` on ring page `b % ringPages`, so the pages it holds are
  bounded by the window instead of its length. The decode contract gains
  the `WINDOW_BLOCK_TABLES`, `WINDOW_SLOT_MAPPING`, `WINDOW_KV_POOL_IN` and
  `WINDOW_KV_POOL_OUT` roles; `PAGED_ATTENTION` is unchanged. An artifact
  with such a pool is `tlaloc-serving-v3` (`ServingModelShape.windowedKv`
  states the window, the layers, the pages and the ring), and
  `HfServingExport` writes one by default for a model with sliding layers
  (`windowedKv = false`, or `-PwindowedKv=false`, keeps full-history pools).
  The Triton backend keeps each sequence's ring, fills the window tables from
  it and splits a request into calls the ring can hold. For Muse Glimmer the
  computed KV per sequence at 32,768 positions is 989 MiB instead of
  3,328 MiB. `tlaloc_serve.py` and the vLLM plugin refuse a v3 artifact by
  its schema version.
- **`KV_PAGES` output (Triton sequence mode).** An optional INT32 `[2]`
  output: the pages a sequence holds after a request in the KV pool and the
  windowed KV pool. `TritonModelRepository` declares it; the backend serves
  a model with or without it.
- **`examples/triton-llm`.** A standalone example: its Gradle build exports a
  HuggingFace checkpoint (Qwen3-0.6B by default, TinyLlama-1.1B-Chat or the
  Muse Glimmer 30B text decoder) as a Triton model repository with
  `HfServingExport` and `TritonModelRepository`; `run.sh` starts Triton with
  `libtriton_tlaloc.so`, and `chat.py` renders the question with the model's
  chat template, tokenizes it with `tokenizers` (no torch) and streams the
  answer token by token through the sequence batcher. It skips by name, with
  exit status 0, without Docker, a GPU, the Triton image, the PJRT plugin, the
  checkpoint or the client packages. On the GB10, Qwen3-0.6B decodes at about
  20 ms a token.
- **`docs/SERVING_ARCHITECTURE.md`.** How serving works: the Kotlin export
  from a checkpoint to DXIR, StableHLO and a serving artifact, and the three
  ways to serve it (framework-free Python, the vLLM platform plugin, Triton),
  with what runs in which process, where memory lives and what has run on
  which hardware.
- **Muse Glimmer (text).** `HfModelFamily.MuseGlimmer` reads
  `MuseGlimmerForConditionalGeneration` checkpoints: the decoder config under
  `text_config`, tensors under `model.language_model.`, the vision encoder's
  tensors listed and not read, the image and video placeholder ids refused by
  name (`HfDecoderConfig.checkTextOnlyTokens`). meta-models/Muse-Glimmer-30B
  (Apache-2.0) serves through Triton with its weights in bf16 and greedy-decodes
  the ids of HuggingFace transformers run with the same arithmetic, 32 of 32.
- **Decoder layer features.** Sliding-window attention (an optional
  `sliding_window` attribute on `PAGED_ATTENTION`), layers without RoPE,
  attention and MLP output norms, `(1 + w)` norm gains, gainless q/k norms, a
  query scale, an attention output gate, an embedding norm, and a logit
  multiplier with final-logit soft-capping. `DecoderLayerPart` gains
  `ATTN_GATE_PROJ`, `ATTENTION_OUTPUT_NORM` and `FEEDFORWARD_OUTPUT_NORM`.
- **bf16 weights in serving artifacts.** `HfDecoderConfig.weightDType`;
  `HfStagedWeights.writeSlot` streams a slot's bytes a block at a time;
  `ServingArtifactWriter.export` takes a `writeWeight` callback and writes F32
  or BF16 weight files. `WeightSource`, `SafetensorsFile`, `SafetensorsIndex`
  and `HfCheckpoint` read a tensor's header entry and raw byte ranges.
- **`harness/python/muse_glimmer_fixture.py`** writes the Muse Glimmer
  fixtures, including a five-layer model with closed-form weights.

- **NVIDIA Triton backend.** `triton/` holds `libtriton_tlaloc.so`, a Triton
  Inference Server backend that compiles Tlaloc StableHLO through a PJRT plugin
  and serves it over Triton's HTTP and gRPC endpoints. It is not a Gradle module
  and is not published to Maven. `triton/verify.sh` checks it end to end in the
  Triton 25.11 container. See [triton/README.md](triton/README.md).
- **Serving artifacts as Triton models.** `TritonModelRepository` (`:maestro`) and
  the Gradle task `:maestro:exportTritonModel` write a serving artifact into a
  Triton model repository with a generated `config.pbtxt`. The backend loads the
  staged weights at model load and keeps the KV pools on the device between
  requests. TinyLlama-1.1B served this way produces the same six greedy token ids
  as HuggingFace transformers for "The capital of France is".
- **Prefill entries.** `HfDecoderGraph` builds a prefill graph: a chunk of
  tokens per sequence in one call, each token an attention row with a causal
  context of its position + 1, right-aligned, returning the last token's logits.
  In the reference interpreter its logits and KV pools equal the decode loop's
  bit for bit (a random-weight model and two real TinyLlama layers).
  `HfServingExport` writes one prefill entry per context bucket
  (`-Pprefill=false` to leave them out).
- **Serving manifest `tlaloc-serving-v2`.** Adds prefill entries; decode entries
  are unchanged and `tlaloc-serving-v1` artifacts are still read. The prefill
  logits type is `[batch, 1, vocab]` (the last position), as decode's.
- **Sequence mode in the Triton backend.** A model whose `config.pbtxt` names a
  serving manifest uses Triton's sequence batcher (oldest strategy): the backend
  keeps each sequence's KV pages by correlation ID, allocates them as the
  sequence grows, frees them on END or after the idle timeout, refuses a START by
  name when the pool is full, runs a prompt through a prefill entry, and runs the
  decode steps of several sequences as one batched call. Clients send token ids
  only. `TritonModelRepository` writes this mode by default for an artifact with
  KV pools (`-PkvMode=client` for the previous form). On the GB10, TinyLlama's
  6-token prefill takes about 26 ms and four concurrent sequences decode about
  twice as many tokens per second as one.
- **Qwen3.** `HfModelFamily` describes a HuggingFace decoder family: its
  architectures, the `config.json` keys it may carry, and one `DecoderLayerSpec`
  per layer. `Qwen3ForCausalLM` is the second family after Llama: a per-head
  RMSNorm on q and k before RoPE, `head_dim` independent of `hidden_size`, and a
  tied head that the file may also store (accepted only when it is bit-for-bit the
  embedding table). A config key the family does not know is refused by name at
  parse. Sliding-window layers, layers without RoPE, post-attention and
  post-feedforward norms, logit soft-capping, attention or MLP biases and
  activations other than SiLU are recorded and refused by name when a graph is
  built. Qwen3-0.6B, all 28 layers, greedy-decodes the same 16 token ids as
  HuggingFace transformers for a plain and a chat-template prompt, in the
  reference interpreter and served by Triton. `:maestro:exportHfServingArtifact`
  exports it (`exportLlamaServingArtifact` remains as a second name).

### Changed

- **The serving performance numbers were measured again on an idle GPU.**
  Four Muse Glimmer configurations served in turn, three rounds: rings and
  blockwise attention take a 512-token call at 31,744 from 4.39 to 1.38 s
  and a 31,744-token prompt from 225.2 to 62.2 s; backend batching takes a
  lone token from 265.9 to 244.3 ms (Qwen3-0.6B bf16 11.33 to 10.70 ms,
  f32 16.79 to 15.39, TinyLlama 22.86 to 21.66; the first measurements
  were contended); int8 weights take it to 139.7 ms and a 512-token call
  146 to 163 ms longer. `PjrtBlockwisePagedAttentionTest` covers the
  blockwise form at block boundaries, lengths 0 and 1, an empty call, a
  window crossing a block boundary and fully masked blocks, against the
  interpreter and the per-row form, and ring decode at lengths 1, the
  window, the ring and twice the ring. `verify.sh` serves the int8 Qwen3
  model with its codes quantized along the wrong axis
  (`triton/int8_wrong_axis.py`) and requires its fixture ids to fail. Step
  inputs read from mapped host memory give the logits of inputs uploaded
  one by one bit for bit.

- **Long-context prefill attends only what it needs.** A sliding layer whose
  KV lives in a windowed ring now reads the ring (`PAGED_ATTENTION` with the
  new `ring = true` attr: the first `ringPages` columns of its table,
  masked by each slot's age) instead of a table as wide as the bucket, in
  decode and prefill. Rows sharing a table over a context of several key
  blocks lower to a `stablehlo.while` over blocks of 2,048 keys with a
  running max and sum, from the first block any row can see to the block of
  the call's last position, so no context-wide score tensor is written and a
  call early in a large bucket scores only what it has. Prefill attention
  dots name the f32 dot algorithm (f32 products and sums, as `HIGHEST`),
  which XLA runs as its own GEMM instead of a cuBLAS SIMT kernel. Muse
  Glimmer through Triton on an idle GPU, before and after served in turn: a
  512-token call at positions 31,744 takes 1.39 s instead of 4.50 s, one at
  8,704 0.88 s instead of 4.49 s, a 31,744-token prompt 62.6 s instead of
  222.8 s, a decode step at 30,000 positions 283 ms instead of 315 ms; the
  largest entry needs 453 MiB of temporary memory instead of 4,125 MiB. The
  interpreter reads the same bits through a ring as through the full-width
  table, `verify.sh` (with `MUSE_GLIMMER=1`) keeps every token id, and the
  windowed example model now agrees with its full-history twin within 1.2e-6
  of the largest logit on the GPU instead of bit for bit (its sliding layers
  sum over 12 positions where the twin's sum over 64).

- **The ABI baselines record the serving changes.** `ir.api`, `maestro.api` and
  `stablehlo.api` are dumped again. Classes that gained a defaulted property
  (`PagedAttentionAttrs.Parsed`, `DecodeModelShape`, `DecodeGraphSpec`,
  `HfDecoderConfig`, `ServingModelShape`) and functions that gained a defaulted
  parameter (`HfDecoderConfig.toDecodeModelShape`, `HfDecoderGraph.spec`,
  `HfServingExport.export`, `ServingArtifactWriter.export`) no longer have
  alpha02's JVM signatures: source written against alpha02 compiles, a binary
  compiled against it must be rebuilt. `toStablehlo` is `@JvmOverloads`, so its
  alpha02 signatures remain. New: `WindowedKvPool`, `ServingWindowedKv`,
  `ServingRefusedToken`, `ServingManifest.SCHEMA_VERSION_3`,
  `TritonModelRepository.KV_PAGES` and the `WINDOW_*` slot roles.
- **`triton/verify.sh` checks more.** The bf16 Qwen3 logits must be within
  6e-3 of the largest (measured 3.0e-3; it was 1e-2), and must fail the f32
  artifact's 2e-3, so the logit comparison is shown to tell the two apart.
  TinyLlama's batched prefill also runs four prompts of 2, 55, 9 and 30 tokens
  in one call (logits within 5.6e-4 of each solo run, the same ids), with its
  `--perturb` control.
- **Prefill attention gathers each sequence's pages once.** `PAGED_ATTENTION`
  accepts fewer block tables than query rows, one per equal group of
  consecutive rows (`PagedAttentionAttrs.rowsPerTable`, derived from the
  shapes), and the prefill graph passes each sequence's table once instead
  of repeating it per token. The emitter then gathers `[tables, context]`
  keys and values instead of `[tokens, context]`: 4 MiB per layer instead of
  8 GiB for Muse Glimmer at 2,048 tokens. On the GB10 the new lowering gives
  the per-row lowering's output exactly; the KPTX paged kernel declines
  shared tables. Prefill bodies change; decode bodies do not.

- **A tied head reads the embedding table.** `HfDecoderGraph` gives a tied
  head no weight slot: it is one `MATMUL` contracting the hidden axis of the
  final state and of the embedding table (`lhs_contracting_dims = [1]`,
  `rhs_contracting_dims = [1]`, a `dot_general` without a transpose), which
  the DXIR interpreter now evaluates. Qwen3-0.6B's weights go from 2867 MiB to
  2273 MiB with the same 32 ids; `HfDecoderConfig.tiedHeadCopy` keeps the copy
  (the control: in the interpreter the two give the same logits bit for bit).
  The model hash of such an artifact ends in `:tiedHead`.
- **Triton sequence mode, a full page pool.** When a request needs pages the
  pool does not have, the backend reclaims pages from sequences idle past the
  reclaim rule (which Triton has ended) least recently active first, and only
  as many as the request needs, where it used to free every such sequence. It
  never takes pages from a live sequence: the request is refused by name,
  the refusal lists the sequences holding pages, and a sequence refused
  mid-generation keeps its pages and KV, so the same request can be sent
  again. Live sequences are not preempted (no KV swap or recompute).

- **Triton dynamic batching groups requests by shape.** A batch's requests
  are keyed by the shape of their rows and each key's requests run together
  in arrival order, instead of only consecutive requests of one shape. A
  ragged batch (`allow_ragged_batch`) of two widths runs as two executions:
  216 requests ran in 74 to 76 executions instead of 119 to 134. The model
  parameter `group_by_shape: "false"` keeps the old grouping.
- **KV pools are updated in place.** `toStablehlo` takes `outputAliases`, and
  every serving body the artifact writer emits marks each `KV_POOL_IN`
  parameter with `tf.aliasing_output` naming its `KV_POOL_OUT` (the manifest's
  `donationPairs`). The Triton backend donates the pools (and any `state:`
  buffer) to each execution, so XLA writes them where they are instead of
  copying 44 MiB (TinyLlama) or 224 MiB (Qwen3-0.6B) per run. The backend checks
  every pool's device address after each run and logs it; `verify.sh` requires
  100 of 100 runs in place for both models and runs a control with the new
  model parameter `donate_kv_pools: false`, which must be seen copying in 100 of
  100 runs and give the same 100 ids. Measured on the GB10 while another
  process kept the GPU 95% busy, alternating the two servers three times
  (server-side time of a batch-1 decode step, 235 steps per run): Qwen3-0.6B's
  fastest step 11.5 ms in place against 16.2 ms copied, medians 13.1 to 23.8 ms
  against 25.8 to 32.5 ms; TinyLlama's fastest step 18.3 ms against 19.3 ms,
  with medians too noisy to separate. Artifacts exported before this change
  still serve, with the pools copied and a warning in the log; `verify.sh`
  exports them again. The framework-free Python runtime and the vLLM plugin
  still pass the pools as host lists.
- `PAGED_ATTENTION`'s two f32 dots are emitted with HIGHEST precision, so XLA
  does not run them in TF32 on a GPU.
- A config with sliding-window layers, layers without RoPE, output norms or
  final-logit soft-capping now builds a graph instead of being refused by name;
  `attn_logit_softcapping` is still refused.
- The Triton backend drops the page cache of each weight file after uploading it.

- **HF decoder types renamed for the second family.** `HfLlamaConfig` is now
  `HfDecoderConfig`, `HfLlamaNames` `HfDecoderNames`, `LlamaWeightRole`
  `DecoderWeightRole`, `LlamaLayerPart` `DecoderLayerPart`, `HfLlamaDecodeGraph`
  `HfDecoderGraph`, `HfLlamaCheckpoint` `HfCheckpoint`, `HfLlamaStagedWeights`
  `HfStagedWeights` and `HfLlamaServingExport` `HfServingExport`. The old names
  remain as deprecated type aliases, so source written against alpha02 still
  compiles; the binary names changed. `HfDecoderGraph.PARTS_PER_LAYER` is gone:
  the number of tensors per layer is `DecoderLayerSpec.parts.size`, 9 for Llama
  and 11 for Qwen3.
- **RoPE tables sized to the entry.** The decode and prefill graphs carry cos and
  sin tables for the positions the entry can reach (its context, capped by
  `max_position_embeddings`) instead of every position the model supports.
- **A tied checkpoint may also store `lm_head.weight`.** `HfCheckpoint.open` used
  to refuse it; it now reads the head from the embedding table, as transformers
  does, and `HfStagedWeights` refuses the file by name if the stored head differs
  from the table.

### Fixed

- **`verify.sh` left an int8 Qwen3 container running when that step
  failed.** Its exit cleanup did not name the int8 container; it now names
  it and the new wrong-axis control's.

- **The lone-sequence wait limit of `sequence_checks.py --batching` sat
  inside its own noise.** On an idle GPU a lone TinyLlama sequence waited
  0.30 to 0.51 ms a step on average over seven runs, and one `verify.sh`
  run failed at 0.506 ms against the 0.5 ms limit. The limit is now 0.8 ms,
  between that spread and the 1.2 ms the Triton-batched control waits,
  which must still fail.

- **A sequence whose requests were being refused could lose its pages.** The
  Triton backend counted a sequence as active only when one of its requests
  ran, while Triton restarts its idle timer for every request, refused or
  not. A sequence sent refused requests (a token outside the vocabulary, a
  placeholder id, a context too long) for longer than the reclaim limit was
  reclaimed while Triton still held it, and its next step was refused for
  having no KV state. Every request of a held sequence now counts, and a
  sequence with any request in the batch is not reclaimed.
  `sequence_checks.py --queued` sends one sequence a refused request every
  50 ms while new sequences run the pool out; it failed before the fix and
  passes after.
- **A refused START could leave an earlier sequence's KV to be continued.**
  Triton ends the earlier sequence with that ID at a second START even when
  the backend refuses the request, but the backend kept its KV, and the next
  step continued it. The backend now frees it (`restart-refused` in
  `sequence_checks.py`, which failed before the fix).
- **A request of several calls that failed part way left its sequence
  inconsistent.** When a prompt split into several calls failed after one of
  them ran, the sequence kept the tokens that ran and the error did not say
  so; with a windowed ring the positions the first call read may already be
  written over. The sequence is now freed and the error says so. No check
  makes a later call fail on demand, so this path is not exercised.

- **An output written device to device into CUDA shared memory could arrive
  empty.** A device-to-device `cudaMemcpy` returns before the copy is done, and
  the backend sent the response (and let PJRT free the result) without waiting
  for it. With the GPU busy with another process, `matmul_sumsq`'s outputs came
  back as zeros in two of three runs of `verify_client.py`; the backend now
  synchronizes after the copy, and six runs out of six pass on the same busy GPU.
- **The Triton backend no longer frees a sequence Triton still holds.** In
  sequence mode the backend freed the pages of any sequence it had not run for
  `max_sequence_idle_microseconds`, counted from the end of its last execution.
  Triton counts from a request's arrival, and a request waiting in its queue
  keeps the sequence alive, so under load a step Triton accepted was refused
  for having no KV state (a 200 ms timeout and 12 to 32 concurrent TinyLlama
  sequences: 3 to 16 such refusals a run). The backend now frees idle sequences
  only when a sequence needs pages the pool lacks, never one with a request in
  the batch being run, and only after twice the timeout plus the longest queue
  wait the oldest strategy allows. `verify.sh` runs the case
  (`sequence_checks.py --queued`); a backend without the fix fails it.

## [0.1.0-alpha02] — 2026-09-24

The Kotlin 2.4 release: the compiler plugin supports Kotlin 2.4.20–2.4.29. Stay on
`0.1.0-alpha01` for Kotlin 2.3.x.

### Changed

- **Kotlin 2.4.20.** Tlaloc is built against Kotlin 2.4.20 and its compiler plugin
  supports Kotlin **2.4.20 through 2.4.29**; any other Kotlin is refused by name at
  compile time. Projects on Kotlin 2.3.20–2.3.29 stay on `0.1.0-alpha01`. The
  plugin's refusal also reaches a user of an older compiler as a compile error:
  under Kotlin 2.3 it is reported through the message collector, because the
  diagnostic API Kotlin 2.4 recommends does not exist there.
  [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) lists the Kotlin range of each
  Tlaloc version.
- **IR-phase messages are compiler diagnostics.** The compiler plugin's IR-phase
  refusals and warnings are reported at the call site through the compiler's
  diagnostic reporter, with unchanged text and severity, and a warning can be
  silenced with `@Suppress` naming it (for example `IR_LOWERING_REFUSED_WARNING`).
  The errors cannot be silenced that way; `strictLowering=false` is still the only
  way to compile a call the plugin refuses.
- **ABI baseline for every published module.** The committed `api/*.api` baselines
  are checked by the Kotlin Gradle plugin's ABI validation (`checkKotlinAbi`,
  `updateKotlinAbi`) instead of binary-compatibility-validator, and now also cover
  `tlaloc-runtime-pjrt`, `tlaloc-runtime-cuda`, `tlaloc-kptx`, `tlaloc-runtime-iree`
  and `tlaloc-compiler-plugin`.
- **Symja 3.2.0.** The build compiles and tests `tlaloc-ir` against
  `org.matheclipse:matheclipse-core:3.2.0`, and the compiler's refusal for a
  symbolic-trip-count loop now names that coordinate. Symja stays `compileOnly`
  (not in a consumer's dependency graph). The 0.1.0-alpha01 `tlaloc-ir` jar calls
  the same 36 Symja methods and fields, so it also runs with 3.2.0 on the
  compiler plugin classpath.
- **Gradle 9.7.1.** The repository's wrapper moves from 9.5.0 to 9.7.1; the
  Tlaloc Gradle plugin is tested with it.
- **`tlaloc { dumpGradSourceDir }` writes one subdirectory per compilation.** The
  Gradle plugin writes each Kotlin/JVM compilation's gradient sources into
  `<dumpGradSourceDir>/<source set>` (`main`, `test`; `jvmMain`, `jvmTest` in a
  Multiplatform build) instead of into `dumpGradSourceDir` itself, and declares
  that subdirectory an output of the compile task. The build cache now stores and
  restores the dumped files, and the directory's absolute path no longer enters
  the compile task's cache key, so a project that sets it gets cache hits from a
  checkout at another path. A build that reads the files from
  `dumpGradSourceDir` directly now reads `dumpGradSourceDir/main`. The raw
  `-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>` option is unchanged.
- **API documentation.** KDoc in every published module describes the code
  without internal changelog numbers, plan phases or planning-document names;
  `./gradlew test` fails if one comes back (`scripts/check-kdoc-internal-refs.py`).

### Fixed

- **Vendored Maestro server builds.** `third-party/maestro/maestro-server`
  compiles: it declares its dependency on `maestro-tlaloc`, and its Spring
  configuration builds the Tlaloc step runtime with the pod-spec builder and the
  cluster accelerator from `TLALOC_CLUSTER_VENDOR` / `TLALOC_CLUSTER_ARCH`. The
  vendored build's Spotless, Checkstyle and PMD checks run on JDK 25 with Gradle 9
  (Spotless 8.10.2, google-java-format 1.30.0, Checkstyle 9.3).

## [0.1.0-alpha01] — 2026-09-24

The first published version. [docs/CAPABILITIES.md](docs/CAPABILITIES.md) lists
what it contains, what is certified and on what hardware.

### Contents

- **Automatic differentiation at compile time.** A K2 compiler plugin rewrites
  `grad`, `grad2`, `grad3`, `valueAndGrad*`, `jvp`, `jvp2`, `vjp`, `vjp2`,
  `jacobian`, `jacobianReverse` and `hessian` calls into synthesized gradient
  code. Custom rules through `customVjp`, `customJvp` and `customVjpJvp`. Loops
  and branches are differentiated through φ-calculus coarsening. One reverse
  transform serves the intrinsics, the Tracer-capture API and `:nn` training.
- **Readable gradients.** `dumpGradSource` and `dumpGradSourceDir` print the
  derived gradient as Kotlin that compiles without the plugin and is bit-identical
  to the compiled gradient; `CapturedStep.gradSource()` prints captured tensor
  gradients.
- **Typed tensors.** Rank, dtype and named axes in the Kotlin type;
  `NAMED_INDEX_MISMATCH`, `TENSOR_SHAPE_MISMATCH` and `NOT_DIFFERENTIABLE` compile
  errors at the offending call. dtypes F32, F64, I32 and BF16.
- **Captured values.** A `grad { }` body may reference compile-time constants
  declared outside it (folded as literals) and immutable runtime values of type
  `Float`, `Double`, `Int` or `Long` (bound at the call site).
- **`:nn`.** Immutable layers (Dense, Conv2d, pooling, BatchNorm, Dropout,
  Embedding, EmbeddingBag, GRU, Flatten), SGD, Momentum, RMSprop and Adam, learning
  rate schedules, gradient clipping, and checkpoints: a model and its optimizer
  state in one safetensors file, reloaded bit for bit.
- **Execution.** StableHLO and Shardy emission; PJRT from Kotlin through FFM and
  from Python through ctypes; IREE through its command-line tools; KPTX, a PTX DSL
  with kernel claiming (no kernel is registered by default).
- **Serving.** Paged attention, KV-cache writes, decode bucketing, safetensors
  reading and writing, a serving artifact that runs in a Python process with no
  JVM and no ML framework, and a vLLM platform plugin. A real TinyLlama-1.1B
  produces the same tokens as HuggingFace transformers.
- **`@ExperimentalTlalocApi`**, a `@RequiresOptIn(ERROR)` marker on the
  provisional surfaces: the four-worlds scopes, `AllReduceAttrs`, and the kernel
  choice and cost-model packages.
- **Tooling.** The Gradle plugin `io.github.pedronahum.tlaloc`
  (`tlaloc-gradle-plugin`), which applies `tlaloc-compiler-plugin` of the same
  version to every Kotlin/JVM compilation and takes the plugin options in a
  `tlaloc { }` block; `tlaloc-bom`; a committed ABI baseline checked by
  `apiCheck`; `./gradlew apiDocs` for a local API reference.
- **Requirements.** Kotlin 2.3.20–2.3.29 (other versions are refused by name at
  compile time). JDK 25 to build code that uses `grad { }`; the library modules are
  Java 21 bytecode, so JDK 21 runs them. PJRT, CUDA, KPTX and IREE need JDK 25.
  Symja (LGPL-3.0) is optional and not in the published dependency graph.

### Changed since earlier source builds

These matter only if you built Tlaloc from source before this version.

- **Group id `io.tlaloc` → `io.github.pedronahum`.** `io.tlaloc` could not be
  verified on Maven Central. Package names are unchanged (`io.tlaloc.*`), and so is
  the compiler plugin id used in `-P plugin:io.tlaloc.plugin:<option>`.
- **Version `0.0.1-SNAPSHOT` → `0.1.0-alpha01`.**
- **Every artifact id carries a `tlaloc-` prefix**: `tlaloc-core`, `tlaloc-ir`,
  `tlaloc-autograd`, `tlaloc-nn`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-pjrt`, `tlaloc-runtime-iree`, `tlaloc-runtime-cuda`,
  `tlaloc-kptx`, `tlaloc-compiler-plugin` (JVM artifacts `tlaloc-core-jvm` and so
  on), plus the new `tlaloc-gradle-plugin` and `tlaloc-bom`.
- `tlaloc-ir`, `tlaloc-autograd`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-iree` and `tlaloc-runtime-pjrt` expose the Tlaloc modules their
  public signatures use as `api` dependencies: `tlaloc-autograd` alone is enough to
  write `grad { }` over a `DTensor`.
- **Every compile-time refusal is an error by default.** An unlowerable
  `grad { }` lambda, and every call the IR phase leaves as written, stops the
  build at the call site. `strictLowering = false` turns these into warnings; the
  call then throws `IllegalStateException` when it runs.
- **A working build is silent.** The lowered-IR dumps are INFO messages behind
  `dumpLoweredIr`, so a correct program compiles under `-Werror`.
- `grad(::f)` and other arguments that are not a lambda written at the call site
  are refused by name.
- `dumpGradSource` refuses a value other than `true` or `false`.
- The runtime message for a `grad { }` that was not rewritten lists its three
  possible causes.
- Error messages no longer cite internal work-item numbers.
- `PjrtSession` can be used from several threads. `executeOn` serialises calls on
  one executable, and `close()` waits for calls in flight. A `PjrtBuffer`,
  `PjrtLoadedExecutable` or `PjrtClient` used after its client is closed, and a
  closed `PjrtBuffer` passed to `executeOn` or `execute`, throw instead of touching
  freed memory. Closing twice, or after the client, does nothing.
- `IreeModule` is `AutoCloseable`; closing it deletes its compiled VMFB.
  `runOnIree` and `IreeRuntime.invoke` delete their temporary files.
- The PJRT CUDA plugin is searched for under `$VIRTUAL_ENV`, `~/.local/venvs/*`,
  `~/.venv`, `~/venv`, `~/.local`, `/usr/local` and `/usr`, from the JVM and from
  Python alike; IREE tools under `$TLALOC_IREE_BIN`, `$VIRTUAL_ENV/bin`,
  `~/.local/venvs/*/bin` and `PATH`. A failed search lists every place it looked.
- In the Python serving runtime (`harness/python`, not published to Maven),
  `PjrtApi.load(plugin_path=None, platform="cuda")` takes a `platform` and, with no
  path and no `TLALOC_PJRT_PLUGIN_PATH`, searches for a plugin; when none is found
  it raises `FileNotFoundError` with the search report, where it used to raise
  `ValueError`.
- `CosineDecay` holds its final rate past `decaySteps`, where PyTorch's
  `CosineAnnealingLR` rises again.

### Added since earlier source builds

- The Gradle plugin and the BOM (see *Contents*).
- `KotlinVersionGuard`: an unsupported Kotlin version is a compile error naming
  the version found and the supported range; `unsafeAllowUnsupportedKotlin` turns
  it into a warning.
- "Tlaloc internal error" diagnostics: an unexpected exception inside the plugin,
  or a lowered call the IR phase did not find, is reported at the call site with
  the issue-tracker address instead of crashing the compiler or compiling green.
- Model checkpoints, learning-rate schedules and gradient clipping in `:nn`; a
  safetensors writer in `:core`.
- `LICENSE` (Apache-2.0), complete Maven Central POM metadata, sources and javadoc
  jars for every publication, and signing.
- `./gradlew releaseToCentralPortal` and `.github/workflows/release.yml`.
- CI on x86_64 Linux, aarch64 Linux and arm64 macOS, a JDK 21 lane for the library
  modules, and a probe against the next Kotlin release.
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) sections on running on a GPU,
  configuration (every environment variable and system property Tlaloc reads) and
  troubleshooting.

### Removed

- `io.tlaloc.maestro.MaestroDescriptor` and `io.tlaloc.maestro.StubExecutor`. The
  first-class Maestro step type that replaces them lives in the vendored Maestro
  build under `third-party/maestro`, which is not published: a Maven Central user
  has no replacement.
- The `main` entry points in `tlaloc-maestro`; the exporters run through
  `./gradlew :maestro:exportServingArtifact` and
  `:maestro:exportLlamaServingArtifact`.
- The unused `timeoutSeconds` parameter of `runOnPjrt`.

### Fixed

- Two `grad { }` calls at the same character offsets in two different files could
  compile each other's gradient, and in a shared Kotlin daemon one compilation
  could clear another's pending gradients.
- An f64 `grad { }` body with a literal constant crashed the compiler with a
  `ClassCastException`, and an f64 constant close to 1 or 0 could be folded as if
  it were exactly 1 or 0.
- A `grad { t: Tracer<…> -> … }` call (the plugin-free tape overload) drew a
  spurious lowering warning.
- The Python serving runtime did not search for a PJRT plugin, so `serve.py`
  reported no plugin on a machine whose JVM lane ran on CUDA.
- Both PJRT bindings check the plugin's `PJRT_Api` size and API version before
  reading a function pointer and refuse an incompatible plugin by name.
- A malformed `TLALOC_PJRT_MEMORY_FRACTION`, `TLALOC_PJRT_PREALLOCATE`,
  `TLALOC_PJRT_NODE_ID` or `TLALOC_PJRT_NUM_NODES` is refused with the variable's
  name and value.
- A `PjrtSession` whose client creation fails releases the memory it allocated.
- Symbolic simplification no longer swallows JVM errors such as
  `OutOfMemoryError`.
- The KPTX kernel caches and the kernel-resolver registry are safe to use from
  several threads.

[0.1.0-alpha01]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha01
[0.1.0-alpha02]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha02
