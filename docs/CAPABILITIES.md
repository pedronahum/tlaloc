# Tlaloc capability matrix

What runs, where it ran, and what pins it. The [README](../README.md) carries a
one-row-per-area summary; this is the full picture.

Status means exactly this:

| | |
|---|---|
| ✅ **Certified** | an automated test pins it, and the row says where it ran |
| 🧪 **Written** | the code exists and unit-tests pass, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design document exists; no implementation |
| ⬜ **Not started** | planned, nothing written yet |
| ❌ **Not planned** | |

The suite has **2,706** automated tests: 2,584 that `./gradlew test` runs, and 122
from the vendored Maestro modules, which the root `test` task does not run (50 in
`maestro-tlaloc`, 4 Tlaloc tests in `maestro-common`, 68 in `maestro-server`). All
2,584 were counted in a clean-room `./gradlew test --rerun-tasks` with 0 failures,
and the 122 in each Maestro module's own `test --rerun`, 0 failures.
On the GB10 workstation where they were counted, 93 of them skip by name: 88 MLIR
round trips that need `stablehlo-translate` or `sdy-opt`, and 5 TPU smoke tests. This is
the one place the documentation states the count.

## Automatic differentiation

| Capability | Status | Notes |
|---|---|---|
| Reverse mode — `grad`, `grad2`, `grad3`, `valueAndGrad*` | ✅ | Compile-time, via the K2 plugin |
| Forward mode — `jvp`, `jvp2`, `valueAndJvp*` | ✅ | |
| `vjp` / `jacobian` / `jacobianReverse` / `hessian` (+ `*2` forms) | ✅ | Arbitrary-arity models go through graph capture, not the fixed-arity intrinsics |
| Higher order — fwd-over-rev, rev-over-rev, nesting matrix | ✅ | Full nesting matrix certified |
| Custom derivatives — `customVjp`, `customJvp`, `customVjpJvp` | ✅ | |
| Control flow — loops and branches under `grad` | ✅ | φ-calculus coarsening — Shen et al., *Coarsening Optimization for Differentiable Programming*, [OOPSLA 2021](https://doi.org/10.1145/3485507) ([local copy](papers/coarsening-autodiff.txt)) — with a Symja-backed closed-form engine |
| Readable reverse code | ✅ | `dumpGradSource` compiler flag + `DxirFunction.toKotlinSource()`; printed source compiles and runs. Scalar lambdas render through the plugin; tensor `grad {}` lambdas render through the capture route (`CapturedStep.gradSource()`). `ABS`, `RSQRT`, `GELU` and `SILU` still have no `:core` tensor twin and refuse by name |
| One AD engine | ✅ | The runtime tape was deleted; every gradient — intrinsics, capture API, `:nn` training — comes from the same reverse transform ([audit](AD_SINGLE_ENGINE_AUDIT.md)) |

## Tensors and operations

| Capability | Status | Notes |
|---|---|---|
| Shape- and dtype-typed tensors | ✅ | `DTensor<Rank2<Sym, Sym>, F32>`; mismatches are compile errors |
| Named axes | ✅ | `Named<N, A>`; a contract over misaligned axis names does not compile |
| Op surface — elementwise, broadcasting, reductions, shape ops, `concat`/`slice`/`pad`, `where` | ✅ | Full [DiffKT](https://github.com/facebookresearch/diffkt) parity, closed |
| NN ops — conv2d (incl. grouped/depthwise), pooling, softmax, embedding, losses, batch norm | ✅ | |
| Special functions — `lgamma`, `digamma`, `polygamma`, `integral` | ✅ | |
| Stateless RNG — threefry-2x32, uniform/normal/cauchy/exponential/chiSquare | ✅ | Bit-exact against JAX's classic stream; reparameterized gradients |
| Sparse — rank-2 CSR, sparse×dense matmul through `grad { }` | ✅ | Host + interpreter; no GPU emission by design ([audit](SPARSE_PARITY_AUDIT.md)) |
| dtypes — F32, F64, I32, BF16 | ✅ | bf16 end to end incl. native PJRT BF16 buffers and mixed-precision training |
| dtypes — F16, FP8, int8 tensors | ❌ | int8 exists for KV-cache quantization only |

## Models and training (`:nn`)

| Capability | Status | Notes |
|---|---|---|
| Layers — Dense, Conv2d, MaxPool/AvgPool, BatchNorm, Dropout, Embedding, EmbeddingBag, GRU, Flatten | ✅ | Immutable/functional; a training step returns a new model |
| Optimizers — SGD, Momentum, RMSprop, Adam, FixedLearningRate | ✅ | Pure `(params, grads, state) → (params', state')` |
| LR schedules — step, exponential, cosine, linear warmup | ✅ | Pure functions of the step count; agree with PyTorch's `StepLR`/`ExponentialLR`/`CosineAnnealingLR` to 2.8e-7 relative. `CosineDecay` deliberately CLAMPS past `T_max` where PyTorch's is periodic — certified as a difference |
| Gradient clipping — by global norm, by value | ✅ | Agrees with `clip_grad_value_` exactly; by-norm differs from `clip_grad_norm_` by torch's own `+1e-6` denominator guard (~1.6e-7 relative) and sits closer to the exact ratio |
| Model persistence — save/load a model + optimizer state | ✅ | One safetensors file; round trip is BIT-IDENTICAL and a resumed run's next 15 steps match the uninterrupted ones bit for bit. Loading builds a NEW model — layers stay immutable |
| Training loop — capture once, train | ✅ | MLP converges; loss curve matches PyTorch to 7 decimals |
| Training on GPU | ✅ | The captured gradient graph compiles to StableHLO and trains on CUDA |
| Mixed precision (bf16 compute, f32 master weights) | ✅ | No loss scaling needed |
| Distributed / multi-GPU training | 📐 | [Design](MULTIHOST_DESIGN.md) + marshalling done; never run (needs 2+ hosts) |

## Inference and serving

| Capability | Status | Notes |
|---|---|---|
| Paged attention, KV-cache writes, decode bucketing | ✅ | Inference-only ops; they refuse differentiation by name. Paged attention takes an optional `sliding_window` (a row sees its last W positions, its own included, as transformers' mask does), checked against a dense walk for every window and length in the interpreter and against the interpreter on the GB10. Its two f32 dots are emitted with HIGHEST precision, so XLA does not run them in TF32 |
| HuggingFace safetensors ingestion | ✅ | Kotlin parser; certified against torch reading the same bytes |
| safetensors WRITING | ✅ | F32/F64/I32/BF16, header padded and tensors ordered so every offset is naturally aligned; certified both ways against the reference `safetensors` library on raw bytes. Sharded and streamed output are not supported; F16/FP8 refuse by name |
| A real Llama serving end to end | ✅ | TinyLlama-1.1B, all 22 layers, on PJRT-CUDA — 6/6 generated token ids identical to HuggingFace transformers, driven directly *and* through vLLM |
| A real Qwen3 serving end to end | ✅ | Qwen3-0.6B (Apache-2.0), all 28 layers: the 16 greedy token ids HuggingFace transformers produces (float32, CPU, eager attention) for a plain prompt and a chat-template prompt, in the reference interpreter (logits within 2.6e-6 of the largest) and served by Triton with a prefill call and the sequence batcher (within 3.3e-4; TF32). On the GB10 through Triton: about 25 ms to prefill and 19 ms per generated token (52 tokens/s). The interpreter test checks 4 tokens per prompt by default and all 16 with `TLALOC_QWEN3_FULL=1`; both ran. The weights stay in the HuggingFace cache and the tests skip by name without them |
| HuggingFace model families | ✅ | `HfModelFamily` reads Llama, Qwen3 and Muse Glimmer checkpoints through one config parser, name mapping and graph builder, with one `DecoderLayerSpec` per layer. A `config.json` key the family does not know is refused by name at parse (inside and outside a multimodal file's `text_config`); Qwen3's per-head q/k RMSNorm is certified against an independent Kotlin forward pass and, with its negative control, against the real checkpoint |
| A real Muse Glimmer serving end to end (text only) | ✅ | meta-models/Muse-Glimmer-30B (Apache-2.0, revision a4e59da5): the text decoder, 52 layers, 27.8 billion parameters, its 56 GB of weights bf16 on the device; the 809 vision-encoder tensors are not read and the image and video placeholder ids are refused by name. Served by Triton on the GB10 with a prefill call and the sequence batcher, for a plain and a chat-template prompt, 16 greedy tokens each: all 32 ids equal transformers 5.17 run with the same arithmetic (bf16 weights, f32 activations, float64 RoPE tables; CPU, eager), with logits within twice that run's own float32-vs-float64 difference (at most 1.4e-2 of the largest logit, against 1.8e-2); against transformers in bfloat16, 31 of 32 ids are equal and the 32nd is the one position where the two transformers runs choose differently from each other (margins 0.31 and 0.12). `--perturb` fails. About 310 ms to prefill (the 128-token entry), 245 ms per generated token (4.1 tokens/s, close to reading 56 GB of weights at the GB10's memory bandwidth), load 105 s (86 s of it the weight upload), 60 GiB of device memory under a PJRT memory fraction of 0.49 computed from the weights, 85 to 90 GiB peak in use system-wide over two runs (unified memory). `triton/verify.sh` runs it with `MUSE_GLIMMER=1`. The weights stay in the HuggingFace cache |
| Muse Glimmer arithmetic against transformers' own code | ✅ | `HfMuseGlimmerTest`, a five-layer model (hidden 32, window 3, soft-cap 3) whose weights are a closed-form function of the tensor name, run through transformers' `MuseGlimmerForConditionalGeneration` and through the decode graph in the interpreter, as a decode loop and as prefill then decode: within 7.1e-8 of the largest logit when both build RoPE tables in float64, and the same token at every position against unmodified transformers (whose float32 tables differ by an ulp here and there, which moves the logits by up to 8e-3 on this 32-wide model once bf16 rounding of the projection inputs amplifies it). Negative controls: without the window, with RoPE on every layer, without soft-capping, with `w` instead of `1 + w` norms and without the q scale, the logits move by more than 1e-2 |
| Sliding-window layers, layers without RoPE, attention and MLP output norms, final-logit soft-capping, attention output gate, gainless q/k norms | ✅ | Implemented in the one graph builder and certified through Muse Glimmer (above). `attn_logit_softcapping` is still refused by name. A Qwen3 config with sliding layers now builds; no real sliding-window Qwen3 checkpoint has been run |
| KV pages sized to the window for sliding layers | ✅ | A model's sliding-window layers keep their KV in a second pool class, the windowed KV pool (`WindowedKvPool`; a `tlaloc-serving-v3` artifact): each sequence holds a ring of at most `ceil(window / blockSize) + 1` pages, block `b` on ring page `b % ringPages`, with its own block table and slot mapping, and a request is split into calls the ring can hold. The full-attention layers keep full-history pages. Certified in the reference interpreter (`WindowedKvPoolTest`: two sequences several windows long through prefill chunks, batched and padded decode steps give logits bit-identical to full-history pools; a ring one page short, or a call one token longer than the ring holds, changes them) and through Triton (`verify.sh`: a three-layer model with window 8 and pages of 4 holds at most 3 windowed pages while its full pages grow to 15, and matches the full-history model sent the same calls to 5.3e-7 of the largest logit). For Muse Glimmer the computed KV per sequence is 365 MiB instead of 832 MiB at 8,192 positions and 989 MiB instead of 3,328 MiB at 32,768 (`docs/SERVING_ARCHITECTURE.md`); Muse Glimmer served from a v3 artifact (its 39 sliding layers in the windowed pool; the 128-token context caps the ring at 8 pages, so the bytes are the same there) greedy-decodes the same 32 ids as before (`verify.sh` with `MUSE_GLIMMER=1`). The framework-free Python runtime and the vLLM plugin refuse a v3 artifact by its schema version |
| bf16 weights in a serving artifact | ✅ | `HfDecoderConfig.weightDType` BF16 stages the weights as the file's bf16 bytes, copied a block at a time (the Linears transposed in bands, so the 2.7 GB embedding table and head never sit in one array); each projection rounds its f32 input to bf16 and multiplies into f32. The streaming writer is checked bit for bit against the in-memory staging with a 64-byte block. The Triton backend drops the page cache of each weight file after uploading it |
| Framework-free serving runtime | ✅ | The serving process imports no JAX, no PyTorch, no NumPy — just a PJRT plugin `.so` and a driver (proven by an import blocker that raises on those modules while the path runs) |
| The serving runtime finding a plugin for itself | ✅ | `tlaloc_pjrt.find_plugin` searches the same roots as the JVM's `PjrtBinaries`, then the interpreter's own site-packages, and raises the full search report when nothing is found. Pinned on synthetic install trees by `vllm_tlaloc_test.PluginDiscoveryTest`; on the GB10, `examples/gpu-inference/serve.py` finds the same plugin as the JVM lane with no environment variable set |
| vLLM platform plugin | ✅ | vLLM 0.29.0's `LLM.generate()` runs a real TinyLlama from a Tlaloc artifact — the same 6 token ids as the direct driver and as HuggingFace. One sequence, greedy, prompt within the compiled context; `vllm serve`'s HTTP layer is not yet run ([audit](INFERENCE_SERVING_AUDIT.md), [runbook](SERVING_RUNBOOK.md)) |
| NVIDIA Triton backend (`libtriton_tlaloc.so`) | ✅ | Triton 25.11 on the GB10 compiles Tlaloc StableHLO through the PJRT C API and serves it over HTTP and gRPC. `triton/verify.sh` certifies: a reverse-mode gradient bit-identical to the DXIR interpreter; FP32, FP64, FP16, BF16, INT8, INT32, INT64, UINT8 and BOOL over HTTP and gRPC (FP16, INT8 and UINT8 from a hand-written module, since Tlaloc has no such dtypes); shape buckets; dynamic batching of the gradient model (512 concurrent one-row requests, each with exactly its own row, in about a third as many executions; about 5x the unbatched throughput); CUDA shared-memory inputs read in place through a PJRT view and outputs copied device to device, with the same bits as the host path (16 MiB each way: about 0.5 ms of server time against about 4.5 ms through the host); decode steps with backend-held KV pools; sequence mode with prefill and batched decode; load-time refusals by name; TinyLlama-1.1B greedy decoding with the same 6 token ids as HuggingFace; Qwen3-0.6B with the same 16 ids as HuggingFace for a plain and a chat-template prompt; and, opt-in (`MUSE_GLIMMER=1`), Muse Glimmer's text decoder with bf16 weights (row above) ([README](../triton/README.md)) |
| KV pools updated in place (Triton) | ✅ | Each serving body aliases every `KV_POOL_OUT` to its `KV_POOL_IN` (`tf.aliasing_output`); the Triton backend donates the pools, so XLA writes them where they are. Checked by device address after every run: `verify.sh` requires 100 of 100 runs in place for TinyLlama and Qwen3, and a control with `donate_kv_pools: false` must be seen copying in 100 of 100 runs with the same ids. The framework-free Python runtime and the vLLM plugin still move the pools through the host |
| Triton instances on more than one GPU | 🧪 | One PJRT client per GPU (`visible_devices`, ordinal checked), artifacts compiled per GPU named by `instance_group`. GPU 0 selection and the refusal of a GPU the machine does not have ran on the GB10; a second GPU has not, because the GB10 has one |
| Kotlin writes a Triton model repository | ✅ | `TritonModelRepository` turns a serving artifact into `<model>/config.pbtxt` + `<model>/1/`: request tensors from the manifest, staged weights loaded at model load, KV pools as instance state. Unit-tested against golden text; its output is the `reference_decode`, TinyLlama and Qwen3 models `verify.sh` serves |
| SGLang plugin | 📐 | Design recorded; reuses the same artifact |
| KV-cache quantization (int8) | ✅ | Derived error bound, not a guess |

## Compilation and runtimes

| Capability | Status | Notes |
|---|---|---|
| StableHLO + Shardy (SDY) emission | ✅ | We don't write CUDA; we lower to MLIR |
| PJRT execution from Kotlin (pure FFM — no JNI, no Python) | ✅ | Certified on NVIDIA GB10 |
| PJRT execution from Python (pure ctypes — no framework) | ✅ | |
| IREE runtime (CPU + CUDA) | ✅ | |
| Pattern recognition + coarsening — FlashAttention, GQA, RMSNorm, RoPE, SwiGLU, cross-entropy, LayerNorm | ✅ | |
| KPTX — a PTX DSL, parser, transpiler, and recognizer-driven kernel claiming | ✅ | Kernels attach to recognized ops automatically |
| KPTX paged-attention kernel — performance | 🧪 | On-device (GB10, floors over six sessions): **1.4–1.9× faster than XLA's own lowering** at Llama-3-8B-shaped decode points, **1.6–1.8× slower** at TinyLlama-shaped toy points, where the measurement's own dispatch floor is 12–76% of it. Not registered by default — opt in per shape, at shapes you measured ([numbers and ranked fixes](KPTX_PAGED_PERF.md)) |
| Netflix Maestro orchestration — manifest, step type, pod-spec builder | ✅ | Unit-certified; a live K8s run has not been done |

## The public surface itself

These rows are about the *contract*, not a computation.

| Capability | Status | Notes |
|---|---|---|
| Compile-time diagnostics carry a source position | ✅ | `DiagnosticSourcePositionTest` asserts that `LAMBDA_NOT_LOWERABLE` and `NAMED_INDEX_MISMATCH` report at the offending call's own **file, line and column**, and that no Tlaloc error is emitted without one |
| An IDE "red squiggle" | 🧪 | A K2-mode IDE runs the same FIR checkers in its own process, so the error should appear at the position above; nothing in this repository drives an IDE, and there is no IDE plugin |
| `@ExperimentalTlalocApi` — opt-in marker on the provisional surface | ✅ | `@RequiresOptIn(ERROR)` in `:core` on three surfaces (the four-worlds taxonomy, `AllReduceAttrs`, the kernel-choice and cost-model packages) with the criterion recorded in the annotation's own KDoc. Pinned twice: `ExperimentalTlalocApiTest` reads the marker and the marked/unmarked sets out of the **class files** (BINARY retention is invisible to reflection), and `ExperimentalApiOptInTest` runs a real `K2JVMCompiler` with no `-opt-in` and asserts the refusal, the `@OptIn` fix, and that a certified surface is not affected |
| Binary-compatibility baseline | ✅ | `api/<module>.api` committed for all twelve published modules that emit classes (the eleven libraries and the Gradle plugin), Java 25 ones included; `checkKotlinAbi` (the Kotlin Gradle plugin's ABI validation, experimental in 2.4.20) is wired into `check`. Negative-tested by adding a public function to `:runtime-cuda` and to `:compiler-plugin` and watching each fail with the diff. `./gradlew updateKotlinAbi` re-baselines |
| Aggregated API reference | ✅ | `./gradlew apiDocs` → `build/docs/api/index.html`: one Dokka site over the eleven library modules. **Not hosted**; it emits about 200 unresolved-KDoc-link warnings, most in `:ir` |
| Hosted documentation site | ⬜ | No hosted API reference or documentation site |

## Hardware and platforms

| Target | Status | Notes |
|---|---|---|
| NVIDIA GPU (CUDA, via PJRT) | ✅ | Everything marked ✅-on-GPU was certified on a GB10 (Blackwell, aarch64). Other NVIDIA parts are expected to work but are not certified here |
| CPU | ✅ | Host interpreter + IREE-CPU. The *interpreter* is a correctness engine, not a fast CPU backend |
| Google TPU | 🧪 | The plugin lane, platform gating, and a full self-skipping smoke suite are written; nothing has ever executed on a TPU — no hardware. [TPU_BRINGUP.md](TPU_BRINGUP.md) is the runbook for the day it does |
| AMD / Trainium | 📐 | Named in the kernel registry's target matrix; no runtime lane |
| JVM | ✅ | The only build target declared today |
| Android / iOS / WASM | ❌ | The modules are KMP-structured (`commonMain` source sets), which makes these reachable later — but no such targets are declared or built, and nothing has been tested on them |

### Where the tests have been executed

Every ✅ above was certified on one machine, a GB10 (aarch64, JDK 25). This table
lists where else the suite has run.

| Lane | Status | Notes |
|---|---|---|
| GB10 (aarch64 Linux, JDK 25) | ✅ | The reference machine. Every GPU, PJRT, IREE, KPTX and cross-language-oracle row in this file was certified here |
| The whole suite, re-run clean-room | ✅ | On the GB10: `./gradlew test --rerun-tasks` with 0 failures, `scripts/onboarding-smoke.sh`, all ten examples, `quickstart shapeError` failing as designed, a `-Werror` consumer compile, every published POM with its sources and javadoc jars, and both halves of `examples/gpu-inference` (a real TinyLlama-1.1B decoding `' Paris.\n\n2.'` through `/usr/bin/python3`) |
| The library's suites on a JDK 21 | ✅ | `-PtlalocTestJdk=21` over `:core :ir :autograd :nn :stablehlo :maestro`, green on OpenJDK 21.0.2 on the GB10, and in the CI lane below |
| The plugin under a foreign Kotlin compiler | ✅ | `ForeignCompilerGuardTest`, part of `./gradlew test`, runs a real Kotlin 2.3.20 compiler (the release `0.1.0-alpha01` supports) with this build's plugin jar loaded: `KotlinVersionGuard` refuses by name as a compile error, and with `unsafeAllowUnsupportedKotlin` set the link failure that follows is a compile error naming both versions, not a crash. The same harness on Kotlin 2.4.20 compiles with no message. The plugin's source no longer compiles against 2.3.x (`-PtlalocKotlinVersion=2.3.20`): it reports through the Kotlin 2.4 diagnostic API |
| x86_64 Linux CI (`ubuntu-latest`) | ✅ | `.github/workflows/build.yml`, green on GitHub Actions at `d0ca85b`; later commits have not run there yet. A green runner means the platform-neutral subset passes: a runner has no GPU, no PJRT plugin, no IREE, no `stablehlo-translate` and no PyTorch oracle venv, and every test needing one self-skips by name |
| aarch64 Linux CI (`ubuntu-24.04-arm`) | ✅ | Same workflow, green at `d0ca85b`. The only aarch64 evidence from a machine other than the GB10 |
| arm64 macOS CI (`macos-15`) | ✅ | Same workflow, green at `d0ca85b`. The MLIR round trips self-skip, because `stablehlo-translate`, `sdy-opt` and `iree-compile` are not installed |
| JDK 21 CI lane | ✅ | `build.yml`, job `library-jdk21`, green at `d0ca85b` |
| Next-Kotlin CI lane | ✅ (probe) | `.github/workflows/kotlin-next.yml` probes the newest Kotlin on Maven Central in a later feature release than the catalog's — 2.5.0-Beta1 while the catalog is 2.4.20. It runs with `continue-on-error`, so its checkmark is green by construction; read its step outcomes. Run locally, `-PtlalocKotlinVersion=2.5.0-Beta1` compiles the plugin; its tests then stop at the version guard, which refuses 2.5 by design. With the guard bypassed in a local experiment, every plugin test except the guard's own passes under 2.5.0-Beta1 |
