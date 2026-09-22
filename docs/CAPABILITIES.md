# Tlaloc capability matrix

What runs, where it ran, and what pins it. The [README](../README.md) carries a
one-row-per-area summary; this is the full picture.

Status means exactly this:

| | |
|---|---|
| ✅ **Certified** | an automated test pins it, and the row says where it ran |
| 🧪 **Written** | the code exists and unit-tests pass, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design document exists; no implementation |
| ⬜ **Not started** | planned, nothing written yet (the mark [ALPHA_PLAN.md](ALPHA_PLAN.md) uses; added to this legend in §0.4.504) |
| ❌ **Not planned** | |

Last reviewed at §0.4.505 (2026-09-22), 2,522 automated tests at HEAD.

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
| Paged attention, KV-cache writes, decode bucketing | ✅ | Inference-only ops; they refuse differentiation by name |
| HuggingFace safetensors ingestion | ✅ | Kotlin parser; certified against torch reading the same bytes |
| safetensors WRITING | ✅ | F32/F64/I32/BF16, header padded and tensors ordered so every offset is naturally aligned; certified both ways against the reference `safetensors` library on raw bytes. Sharded and streamed output are named deferrals; F16/FP8 refuse by name |
| A real Llama serving end to end | ✅ | TinyLlama-1.1B, all 22 layers, on PJRT-CUDA — 6/6 generated token ids identical to HuggingFace transformers, driven directly *and* through vLLM |
| Framework-free serving runtime | ✅ | The serving process imports no JAX, no PyTorch, no NumPy — just a PJRT plugin `.so` and a driver (proven by an import blocker that raises on those modules while the path runs) |
| vLLM platform plugin | ✅ | vLLM 0.29.0's `LLM.generate()` runs a real TinyLlama from a Tlaloc artifact — the same 6 token ids as the direct driver and as HuggingFace. One sequence, greedy, prompt within the compiled context; `vllm serve`'s HTTP layer is not yet run ([audit](INFERENCE_SERVING_AUDIT.md), [runbook §11](SERVING_RUNBOOK.md)) |
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

Added in §0.4.505 (Tier 4). These rows are about the *contract*, not a computation.

| Capability | Status | Notes |
|---|---|---|
| Compile-time diagnostics carry a source position | ✅ | `DiagnosticSourcePositionTest` asserts that `LAMBDA_NOT_LOWERABLE` and `NAMED_INDEX_MISMATCH` report at the offending call's own **file, line and column**, and that no Tlaloc error is ever emitted without one. Eighteen test classes already pinned the *text*; none had looked at the position |
| An IDE "red squiggle" | 🧪 | **Expected, never measured.** A K2-mode IDE runs the same FIR checkers in its own process, so a redline should land on the position above — but nothing in this repository drives an IDE, and there is no IDE plugin. The README and `GETTING_STARTED.md` were reworded in §0.4.505 to claim the build error and call the squiggle expected |
| `@ExperimentalTlalocApi` — opt-in marker on the provisional surface | ✅ | `@RequiresOptIn(ERROR)` in `:core` on three surfaces (the four-worlds taxonomy, `AllReduceAttrs`, the kernel-choice and cost-model packages) with the criterion recorded in the annotation's own KDoc. Pinned twice: `ExperimentalTlalocApiTest` reads the marker and the marked/unmarked sets out of the **class files** (BINARY retention is invisible to reflection), and `ExperimentalApiOptInTest` runs a real `K2JVMCompiler` with no `-opt-in` and asserts the refusal, the `@OptIn` fix, and that a certified surface is not affected |
| Binary-compatibility baseline | ✅ | `api/<module>.api` committed; `apiCheck` wired into `check`, negative-tested by adding a public function to `:stablehlo` and watching it fail with the diff. Covers the six Java-21-targeted modules — binary-compatibility-validator 0.18.2 refuses Java 25 bytecode (`Unsupported class file major version 69`), so the five 25-targeted modules are **not** covered |
| Aggregated API reference | ✅ | `./gradlew apiDocs` → `build/docs/api/index.html`: one Dokka site over all eleven published modules, 2,898 pages. **Not hosted**, and the 206 unresolved-KDoc-link warnings it emits (110 distinct targets, 117 of them in `:ir`) are counted in §0.4.505 and unswept |
| Hosted documentation site | ⬜ | No gh-pages lane, no MkDocs Material book — the other half of `DIFFKTX_SPEC.md` §14's one-line Docs entry |

## Hardware and platforms

| Target | Status | Notes |
|---|---|---|
| NVIDIA GPU (CUDA, via PJRT) | ✅ | Everything marked ✅-on-GPU was certified on a GB10 (Blackwell, aarch64). Other NVIDIA parts are expected to work but are not certified here |
| CPU | ✅ | Host interpreter + IREE-CPU. The *interpreter* is a correctness engine, not a fast CPU backend |
| Google TPU | 🧪 | The plugin lane, platform gating, and a full self-skipping smoke suite are written; nothing has ever executed on a TPU — no hardware. [TPU_BRINGUP.md](TPU_BRINGUP.md) is the runbook for the day it does |
| AMD / Trainium | 📐 | Named in the kernel registry's target matrix; no runtime lane |
| JVM | ✅ | The only build target declared today |
| Android / iOS / WASM | ❌ | The modules are KMP-structured (`commonMain` source sets), which makes these reachable later — but no such targets are declared or built, and nothing has been tested on them |

### Where the tests have actually been executed

Every ✅ above was certified on one machine — a GB10, aarch64, JDK 25 — so this
table says, separately, on what else the suite has been *run*. §0.4.504 added the
CI lanes that would widen it; none of them has run, because the machine that wrote
them has no access to GitHub Actions.

| Lane | Status | Notes |
|---|---|---|
| GB10 (aarch64 Linux, JDK 25) | ✅ | The reference machine. Every GPU, PJRT, IREE, KPTX and cross-language-oracle row in this file was certified here and nowhere else |
| The library's suites on a JDK 21 | ✅ | `-PtlalocTestJdk=21` over `:core :ir :autograd :nn :stablehlo :maestro` — 1,822 tests green on OpenJDK 21.0.2 (§0.4.504). Ran here, on this machine, not in CI |
| The plugin under a foreign Kotlin compiler | ✅ | `-PtlalocKotlinVersion=2.3.10` — `KotlinVersionGuard` refuses by name, from inside a real 2.3.10 compile (§0.4.504). The same probe against **2.4.20 does not compile the plugin at all**: one error, direct `MessageCollector` access. A published negative result, not a fixed one |
| x86_64 Linux CI (`ubuntu-latest`) | 🧪 | `.github/workflows/build.yml`. Written §0.4.497, widened §0.4.504. **No run exists.** A green run would mean "the platform-neutral subset passes": a runner has no GPU, no PJRT plugin, no IREE, no `stablehlo-translate` and no PyTorch oracle venv, and every test needing one self-skips by name |
| arm64 macOS CI (`macos-15`) | 🧪 | Same file, second matrix leg (§0.4.504). **No run exists.** `stablehlo-translate` / `sdy-opt` / `iree-compile` are a 20–60 minute bazelisk build on macOS, so the MLIR round-trips would self-skip there too |
| JDK 21 CI lane | 🧪 | `build.yml`, job `library-jdk21`. **No run exists**; every command in it was run here first (the row above) |
| Next-Kotlin CI lane | 🧪 | `.github/workflows/kotlin-next.yml`, `continue-on-error`. **No run exists**; its three probes were run here first |
| aarch64 *Linux* CI | ⬜ | Not added. GitHub's arm64 Linux runners are free for public repositories only, and this repository's visibility could not be checked from the development machine. See [ALPHA_PLAN.md](ALPHA_PLAN.md) |
