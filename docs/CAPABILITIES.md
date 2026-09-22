# Tlaloc capability matrix

What runs, where it ran, and what pins it. The [README](../README.md) carries a
one-row-per-area summary; this is the full picture.

Status means exactly this:

| | |
|---|---|
| ✅ **Certified** | an automated test pins it, and the row says where it ran |
| 🧪 **Written** | the code exists and unit-tests pass, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design document exists; no implementation |
| ❌ **Not planned** | |

Last reviewed at §0.4.496 (2026-09-22), 2,345 automated tests at HEAD.

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
| Training loop — capture once, train | ✅ | MLP converges; loss curve matches PyTorch to 7 decimals |
| Training on GPU | ✅ | The captured gradient graph compiles to StableHLO and trains on CUDA |
| Mixed precision (bf16 compute, f32 master weights) | ✅ | No loss scaling needed |
| Distributed / multi-GPU training | 📐 | [Design](MULTIHOST_DESIGN.md) + marshalling done; never run (needs 2+ hosts) |

## Inference and serving

| Capability | Status | Notes |
|---|---|---|
| Paged attention, KV-cache writes, decode bucketing | ✅ | Inference-only ops; they refuse differentiation by name |
| HuggingFace safetensors ingestion | ✅ | Kotlin parser; certified against torch reading the same bytes |
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

## Hardware and platforms

| Target | Status | Notes |
|---|---|---|
| NVIDIA GPU (CUDA, via PJRT) | ✅ | Everything marked ✅-on-GPU was certified on a GB10 (Blackwell, aarch64). Other NVIDIA parts are expected to work but are not certified here |
| CPU | ✅ | Host interpreter + IREE-CPU. The *interpreter* is a correctness engine, not a fast CPU backend |
| Google TPU | 🧪 | The plugin lane, platform gating, and a full self-skipping smoke suite are written; nothing has ever executed on a TPU — no hardware. [TPU_BRINGUP.md](TPU_BRINGUP.md) is the runbook for the day it does |
| AMD / Trainium | 📐 | Named in the kernel registry's target matrix; no runtime lane |
| JVM | ✅ | The only build target declared today |
| Android / iOS / WASM | ❌ | The modules are KMP-structured (`commonMain` source sets), which makes these reachable later — but no such targets are declared or built, and nothing has been tested on them |
