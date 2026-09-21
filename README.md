# Tlaloc

**Differentiable programming for Kotlin, compiled.** You write plain Kotlin;
a K2 compiler plugin turns `grad { }` into a gradient function at compile
time — no runtime tape, no `requires_grad`, no framework objects in your
types. Tensors carry their shape and dtype in the Kotlin type system, so a
rank or axis mistake is a red squiggle in the IDE, not a stack trace in
production. Programs lower to StableHLO and run on GPU through PJRT.

And the gradient the compiler derives is **Kotlin source you can read** —
ask for it with a compiler flag, and what it prints compiles and runs.

> **Status: pre-alpha, developed in the open.** The engine is real and
> heavily certified (2336 automated tests at HEAD, including live
> GPU certifications on an NVIDIA GB10); the packaging is not. There are no
> published artifacts yet — you build from source and consume via
> `publishToMavenLocal`. APIs move without deprecation cycles. See
> [What works today](#what-works-today) for the honest capability map:
> every row says where it runs and what pins it.

---

## Quickstart

Requires a JDK 25 toolchain. (No GPU needed for this example.)

```bash
git clone https://github.com/pedronahum/tlaloc && cd tlaloc
./gradlew publishToMavenLocal          # builds Tlaloc, installs io.tlaloc:*:0.0.1-SNAPSHOT
./gradlew -p examples/quickstart run   # a standalone project that resolves it from mavenLocal
```

```kotlin
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // Rewritten at COMPILE TIME into a gradient function.
    // d/dA sum(A·A) = ones·Aᵀ + Aᵀ·ones
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
    println(g(a).hostF32().toList())   // [6.0, 8.0, 6.0, 8.0]
}
```

Want to see what the compiler derived? Add
`-P plugin:io.tlaloc.plugin:dumpGradSource=true` and it prints the gradient
as Kotlin source, at your lambda's source location. That printed source
compiles, runs, and is certified bit-identical to the compiled gradient —
see [docs/READABLE_REVERSE.md](docs/READABLE_REVERSE.md) for a side-by-side
demo with real generated output.

## What works today

Status means exactly this:
**✅ Certified** — an automated test pins it, and the table says where it ran ·
**🧪 Written** — the code exists and unit-tests pass, but the end-to-end path
has never run (why is always stated) ·
**📐 Designed** — a design document exists, no implementation ·
**❌ Not planned**

### Automatic differentiation

| Capability | Status | Notes |
|---|---|---|
| Reverse mode — `grad`, `grad2`, `grad3`, `valueAndGrad*` | ✅ | Compile-time, via the K2 plugin |
| Forward mode — `jvp`, `jvp2`, `valueAndJvp*` | ✅ | |
| `vjp` / `jacobian` / `jacobianReverse` / `hessian` (+ `*2` forms) | ✅ | Arbitrary-arity models go through graph capture, not the fixed-arity intrinsics |
| Higher order — fwd-over-rev, rev-over-rev, nesting matrix | ✅ | Full nesting matrix certified |
| Custom derivatives — `customVjp`, `customJvp`, `customVjpJvp` | ✅ | |
| Control flow — loops and branches under `grad` | ✅ | φ-calculus coarsening ([OOPSLA 2021](docs/papers/coarsening-autodiff.txt)) with a Symja-backed closed-form engine |
| **Readable reverse code** | ✅ | `dumpGradSource` compiler flag + `DxirFunction.toKotlinSource()`; printed source compiles and runs |
| **One AD engine** | ✅ | The runtime tape was deleted; every gradient — intrinsics, capture API, `:nn` training — comes from the same reverse transform |

### Tensors and operations

| Capability | Status | Notes |
|---|---|---|
| Shape- and dtype-typed tensors | ✅ | `DTensor<Rank2<Sym, Sym>, F32>`; mismatches are compile errors |
| Named axes | ✅ | `Named<N, A>`; a contract over misaligned axis names does not compile |
| Op surface — elementwise, broadcasting, reductions, shape ops, `concat`/`slice`/`pad`, `where` | ✅ | Full [DiffKT](https://github.com/facebookresearch/diffkt) parity, closed |
| NN ops — conv2d (incl. grouped/depthwise), pooling, softmax, embedding, losses, batch norm | ✅ | |
| Special functions — `lgamma`, `digamma`, `polygamma`, `integral` | ✅ | |
| Stateless RNG — threefry-2x32, uniform/normal/cauchy/exponential/chiSquare | ✅ | Bit-exact against JAX's classic stream; reparameterized gradients |
| Sparse — rank-2 CSR, sparse×dense matmul through `grad { }` | ✅ | Host + interpreter; no GPU emission by design (see [audit](docs/SPARSE_PARITY_AUDIT.md)) |
| dtypes — F32, F64, I32, **BF16** | ✅ | bf16 end to end incl. native PJRT BF16 buffers and mixed-precision training |
| dtypes — F16, FP8, int8 tensors | ❌ | int8 exists for KV-cache quantization only |

### Models and training (`:nn`)

| Capability | Status | Notes |
|---|---|---|
| Layers — Dense, Conv2d, MaxPool/AvgPool, BatchNorm, Dropout, Embedding, EmbeddingBag, GRU, Flatten | ✅ | Immutable/functional; a training step returns a new model |
| Optimizers — SGD, Momentum, RMSprop, Adam, FixedLearningRate | ✅ | Pure `(params, grads, state) → (params', state')` |
| Training loop — capture once, train | ✅ | MLP converges; loss curve matches PyTorch **to 7 decimals** |
| Training on GPU | ✅ | The captured gradient graph compiles to StableHLO and trains on CUDA |
| Mixed precision (bf16 compute, f32 master weights) | ✅ | No loss scaling needed |
| Distributed / multi-GPU training | 📐 | [Design](docs/MULTIHOST_DESIGN.md) + marshalling done; never run (needs 2+ hosts) |

### Inference and serving

| Capability | Status | Notes |
|---|---|---|
| Paged attention, KV-cache writes, decode bucketing | ✅ | Inference-only ops; they refuse differentiation by name |
| HuggingFace safetensors ingestion | ✅ | Kotlin parser; certified against torch reading the same bytes |
| **A real Llama serving end to end** | ✅ | TinyLlama-1.1B, all 22 layers, on PJRT-CUDA — **6/6 generated token ids identical to HuggingFace transformers** |
| **Framework-free serving runtime** | ✅ | The serving process imports no JAX, no PyTorch, no NumPy — just a PJRT plugin `.so` and a driver (proven by an import blocker that raises on those modules while the path runs) |
| vLLM platform plugin | 🧪 | Platform discovery activates it and the worker API executes live; `LLM.generate()` still needs one backend-class stub (named in the [audit](docs/INFERENCE_SERVING_AUDIT.md)) |
| SGLang plugin | 📐 | Design recorded; reuses the same artifact |
| KV-cache quantization (int8) | ✅ | Derived error bound, not a guess |

### Compilation and runtimes

| Capability | Status | Notes |
|---|---|---|
| StableHLO + Shardy (SDY) emission | ✅ | We don't write CUDA; we lower to MLIR |
| PJRT execution from Kotlin (pure FFM — no JNI, no Python) | ✅ | Certified on NVIDIA GB10 |
| PJRT execution from Python (pure ctypes — no framework) | ✅ | |
| IREE runtime (CPU + CUDA) | ✅ | |
| Pattern recognition + coarsening — FlashAttention, GQA, RMSNorm, RoPE, SwiGLU, cross-entropy, LayerNorm | ✅ | |
| KPTX — a PTX DSL, parser, transpiler, and recognizer-driven kernel claiming | ✅ | Kernels attach to recognized ops automatically |
| KPTX paged-attention kernel — **performance** | 🧪 | Correct, but currently **slower than XLA's own lowering** on-device. The honest numbers and the specified fix are in [docs/KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md). Not registered by default |
| Netflix Maestro orchestration — manifest, step type, pod-spec builder | ✅ | Unit-certified; a live K8s run has not been done |

### Hardware and platforms

| Target | Status | Notes |
|---|---|---|
| NVIDIA GPU (CUDA, via PJRT) | ✅ | Everything above marked ✅-on-GPU was certified on a **GB10 (Blackwell, aarch64)**. Other NVIDIA parts are expected to work but are not certified here |
| CPU | ✅ | Host interpreter + IREE-CPU. Note: the *interpreter* is a correctness engine, not a fast CPU backend |
| **Google TPU** | 🧪 | The plugin lane, platform gating, and a full self-skipping smoke suite are written; **nothing has ever executed on a TPU** — no hardware. [docs/TPU_BRINGUP.md](docs/TPU_BRINGUP.md) is the runbook for the day it does |
| AMD / Trainium | 📐 | Named in the kernel registry's target matrix; no runtime lane |
| JVM | ✅ | The only build target declared today |
| Android / iOS / WASM | ❌ **not today** | The modules are KMP-structured (`commonMain` source sets), which makes these reachable later — but **no such targets are declared or built**, and nothing has been tested on them |

## What makes it different

Four claims, each with the thing that proves it:

1. **The gradient is compiled, and you can read it.** Tangent showed you the
   derivative of your Python as Python. Tlaloc shows you the derivative of
   your Kotlin as Kotlin — and because every op prints as a real `:core`
   call, the printed gradient **compiles and runs**, certified bit-identical
   to what the compiler produced. → [docs/READABLE_REVERSE.md](docs/READABLE_REVERSE.md)

2. **One AD engine, no tape.** There is no runtime-tape fallback path that
   could quietly disagree with the compiled one: the tape was deleted, and a
   test pins that the capture API and the `grad {}` intrinsic produce
   raw-bit-identical gradients. → [docs/AD_SINGLE_ENGINE_AUDIT.md](docs/AD_SINGLE_ENGINE_AUDIT.md)

3. **Shape errors are compile errors.** Rank, dtype, and named-axis
   mismatches fail in the IDE. For teams where a silent shape bug is a P0,
   this is the whole pitch.

4. **Deployment is an artifact, not a runtime.** A compiled program is a
   content-addressed manifest of StableHLO bodies. Serving it needs a PJRT
   plugin `.so` and a driver — no JVM, no framework. The same artifact is
   what would run on TPU. → [docs/SERVING_RUNBOOK.md](docs/SERVING_RUNBOOK.md)

## Limitations worth knowing before you invest

- **Pre-alpha packaging.** No Maven Central release; build from source.
- **JVM only today** (see the platform table). Android/iOS/WASM are
  structurally reachable, not delivered.
- **TPU is written, not run.** Treat every TPU claim as untested until the
  bring-up runbook has been executed on real hardware.
- **No Python API.** Interop is via StableHLO/ONNX artifacts, not bindings.
- **Not a PyTorch clone.** No `torch.nn` parity goal; the model layer covers
  DiffKT's surface, not PyTorch's.
- **The interpreter is for correctness, not speed.** Performance claims mean
  the compiled GPU path.
- **Kernel performance is XLA's today.** Our own KPTX kernels are a real
  lane, but on paged attention XLA still wins — and we publish that.

## Examples

Eight standalone projects under [`examples/`](examples/), each its own Gradle
build resolving Tlaloc from `mavenLocal` exactly as yours would. Run any of them
with `./gradlew -p examples/<name> run` after `publishToMavenLocal`. Five need
nothing at all; the rest name what they are missing and exit `0`.

| | | Needs |
|---|---|---|
| [`quickstart/`](examples/quickstart/) | `grad {}` on a matmul, lowered at compile time | — |
| [`named-indices/`](examples/named-indices/) | axis names checked by Kotlin's own type checker | — |
| [`readable-gradients/`](examples/readable-gradients/) | the derivative printed as Kotlin, recompiled without the plugin, and agreeing bit for bit | — |
| [`four-worlds/`](examples/four-worlds/) | `program {}` / `workflow {}` and the buffer-handle boundary | — |
| [`layer3/`](examples/layer3/) | recognition, coarsening, and one artifact per device | — |
| [`gpu-training/`](examples/gpu-training/) | 600 Adam steps on the Blackwell, gradient derived by the compiler | CUDA |
| [`gpu-inference/`](examples/gpu-inference/) | Kotlin writes a directory and exits; a framework-free `python3` serves it | CUDA |
| [`tpu/`](examples/tpu/) | five acts, tolerances fixed in advance, written before the hardware exists | TPU |

[`examples/README.md`](examples/README.md) is the full index: what each one
shows, a reading order, and exactly what every one of them printed when the set
was last run end to end.

## Documentation

| Document | What it covers |
|---|---|
| [DIFFKT_PARITY_PLAN.md](docs/DIFFKT_PARITY_PLAN.md) | The op/AD surface, phase by phase, and its end state |
| [MODEL_LAYER_PLAN.md](docs/MODEL_LAYER_PLAN.md) | The `:nn` layer design and per-slice record |
| [READABLE_REVERSE.md](docs/READABLE_REVERSE.md) | Generated gradient source, side by side with its input |
| [INFERENCE_SERVING_AUDIT.md](docs/INFERENCE_SERVING_AUDIT.md) | The serving arc: what is certified, what is only written |
| [SERVING_RUNBOOK.md](docs/SERVING_RUNBOOK.md) | Export an artifact and serve it |
| [TPU_READINESS_AUDIT.md](docs/TPU_READINESS_AUDIT.md) | Honest comparison against TorchTPU; the TPU plan |
| [TPU_BRINGUP.md](docs/TPU_BRINGUP.md) | The Cloud TPU VM session script |
| [KPTX_PLAN.md](docs/KPTX_PLAN.md) · [KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md) | The PTX DSL, and where our kernels stand against XLA |
| [TLALOC_EMIT_CONTRACT.md](docs/TLALOC_EMIT_CONTRACT.md) | What our StableHLO looks like, op by op |

## Development

```bash
./gradlew test                    # the full suite
./gradlew test --rerun-tasks      # forced re-run (a clean-room count needs this)
bash scripts/count-tests.sh       # aggregate test count across all modules
```

The engineering discipline, if you want to contribute in the same style:
claims are certified by tests against analytic or cross-implementation
oracles; unsupported cases **refuse loudly by name** rather than silently
degrading; anything deferred is named in a design document rather than left
implicit. Negative results get published too — the KPTX performance page
exists because our kernel lost.

## Acknowledgments

φ-calculus coarsening follows Shen, Shivers, Dea et al., *Efficient, Sound
Gradient Descent in Dynamic and Dependent Control Flow* (OOPSLA 2021).
The op surface and model layer take their parity target from
[facebookresearch/diffkt](https://github.com/facebookresearch/diffkt); the
readable-derivative idea comes from
[google/tangent](https://github.com/google/tangent). Lowering targets
[OpenXLA](https://github.com/openxla) StableHLO, Shardy, and PJRT.
Orchestration vendors [Netflix Maestro](https://github.com/Netflix/maestro).
