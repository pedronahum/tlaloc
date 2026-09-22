<div align="center">

# Tlaloc

**Differentiable programming for Kotlin — compiled, typed, and readable.**

[Quickstart](#quickstart) · [Transformations](#transformations) · [Installation](#installation) · [Examples](examples/) · [Docs](#documentation) · [Maturity](#maturity)

</div>

---

Tlaloc is an autodiff and array-computing library for Kotlin. You write ordinary
Kotlin functions over tensors; Tlaloc differentiates them **at compile time**
with a K2 compiler plugin, and compiles them to [StableHLO] so they run on GPUs
through [PJRT] or [IREE].

There is no tape, no `requires_grad`, no session and no framework object in your
types — `grad { f }` gives you back a plain Kotlin function.

Four things follow from that, and they are the reasons to look at Tlaloc rather
than JAX or PyTorch:

- **Gradients are code, not a runtime trace.** The derivative is synthesized
  into your bytecode during compilation.
- **Gradients are code you can read.** One compiler flag prints the derivative
  back to you as Kotlin source — and that source compiles, runs, and is
  bit-identical to what the compiler produced.
- **Shape bugs are compile errors.** Rank, dtype and *axis names* live in the
  Kotlin type system, so a transposed weight is a red squiggle in the IDE
  instead of a stack trace in production.
- **Deployment is an artifact, not a runtime.** A compiled model is a directory
  of StableHLO. Serving it needs a PJRT plugin `.so` and a driver — no JVM, no
  Python framework, nothing of Tlaloc left in the process.

```kotlin
val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
```

[StableHLO]: https://openxla.org/stablehlo
[PJRT]: https://openxla.org/xla/pjrt
[IREE]: https://iree.dev

> **Pre-alpha.** The engine is real and heavily tested (2,345 automated tests at
> HEAD, including live GPU runs on an NVIDIA GB10); the *packaging* is not.
> Nothing is on Maven Central yet — you build from source and consume from
> `mavenLocal`. APIs change without deprecation cycles. See
> [Maturity](#maturity) for a straight answer on what runs where.

---

## Quickstart

You need a **JDK 25** toolchain and **Kotlin 2.3.20**. No GPU required.

```bash
git clone https://github.com/pedronahum/tlaloc && cd tlaloc
./gradlew publishToMavenLocal -x test   # installs io.tlaloc:*:0.0.1-SNAPSHOT
./gradlew -p examples/quickstart run    # a standalone project that consumes it
```

```kotlin
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // Rewritten AT COMPILE TIME into a gradient function:
    // d/dA sum(A·A) = 1·Aᵀ + Aᵀ·1
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
    println(g(a).hostF32().toList())   // [7.0, 11.0, 9.0, 13.0]
}
```

`g` is an ordinary Kotlin function. Call it in a hot loop and nothing allocates a
tape, because there is no tape.

---

## Transformations

### `grad` — reverse mode

`grad`, `grad2`, `grad3` and the `valueAndGrad*` family differentiate with
respect to one, two or three arguments. Loops and branches inside the lambda are
fine: control flow is handled by φ-calculus coarsening — Shen et al., *Coarsening
Optimization for Differentiable Programming* ([OOPSLA 2021][phi]) — with a
closed-form solver behind it.

```kotlin
val g  = grad  { x: Float -> (x * x.sigmoid()) / (1f + (1f + x.exp()).log()).sqrt() }
val vg = valueAndGrad2 { w: DTensor<Rank2<Sym, Sym>, F32>, b: DTensor<Rank1<Sym>, F32> -> loss(w, b) }
```

Forward mode is `jvp` / `jvp2`; `vjp`, `jacobian`, `jacobianReverse` and
`hessian` are all there, and they nest — forward-over-reverse and
reverse-over-reverse are both tested across the full matrix. Custom rules go in
with `customVjp`, `customJvp`, `customVjpJvp`.

A loop inside `grad { }` is differentiated, not unrolled by you:
[`differentiable-physics`](examples/differentiable-physics/) puts a 38-step
Euler integrator in the lambda and lets the compiler produce the 823-operation
derivative of the whole simulator.

[phi]: https://doi.org/10.1145/3485507

### `dumpGradSource` — read the derivative

Add one compiler flag and Tlaloc prints the gradient it derived, as Kotlin, at
your lambda's source location:

```
-P plugin:io.tlaloc.plugin:dumpGradSource=true       # print it
-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>   # …and write one .kt per lambda
```

For the `f(x) = x·σ(x) / √(1 + log(1 + eˣ))` above — a quotient whose numerator
needs the product rule *through* a sigmoid — it writes:

```kotlin
fun grad_body_grad(x: DTensor<ScalarShape, F32>): DTensor<ScalarShape, F32> {
    val v1  = x.sigmoid()      // %1  = SIGMOID(%0)
    val v2  = (x * v1)         // %2  = MUL(%0, %1)
    val v5  = x.exp()          // %5  = EXP(%0)
    ...
    val v30 = (v1 * v29)       // %30 = MUL(%1, %29)
    val v31 = (v25 * v30)      // %31 = MUL(%25, %30)
    val v32 = (v26 + v31)      // %32 = ADD(%26, %31)
    return v32
}
```

Every line is a real `io.tlaloc.core` call, so this file **compiles and runs on
its own, without the plugin**. The [`readable-gradients`](examples/readable-gradients/)
example compiles it in a separate source set and checks it against the compiled
gradient: *raw-bit identical* at all 7 test points, and within `5.5e-08` of a
finite difference. → [docs/READABLE_REVERSE.md](docs/READABLE_REVERSE.md)

### Named axes — shape bugs the type checker catches

An axis carries a *name* in its Kotlin type, not just a position:

```kotlin
val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> = ...
val weights:     DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> = ...

val hidden = contract(activations, weights)   // OK: they share `SeqLen`
```

Swap the weights for a `Hidden × Hidden` matrix and the call does not resolve —
Kotlin's own type checker rejects it, before any Tlaloc code runs. The plugin
adds diagnostics of its own on top (`NAMED_INDEX_MISMATCH`,
`TENSOR_SHAPE_MISMATCH`, `NOT_DIFFERENTIABLE`), all of them build errors with
IDE squiggles. → [`examples/named-indices`](examples/named-indices/)

### `capture` — train a model

`:nn` is a functional model layer: layers are immutable, a training step returns
a *new* model, and optimizers are pure `(params, grads, state) → (params', state')`.
You capture the loss once and dispatch the same compiled gradient every step.

```kotlin
val model0 = Sequential(Dense(2, 16, k0), ReluLayer, Dense(16, 16, k1), ReluLayer, Dense(16, 1, k2))

// Traced once; the gradient graph is derived by the compiler's reverse pass.
val step = capture(model0, listOf(x), name = "disc_mlp") { prediction ->
    val residual = prediction - prediction.constant<Shape>(targets, intArrayOf(n, 1))
    (residual * residual).mean()
}

var model = model0
var state = optimizer.initialState()
repeat(600) {
    val out = step.unpack(lane.run(step.gradient, step.bind(listOf(xs), model)))
    val (nextModel, nextState) = optimizer.step(model, out.gradients, state)
    model = nextModel; state = nextState
}
```

No `.backward()`, no `zero_grad()`, no hand-written derivatives. On a GB10 that
loop runs **600 Adam steps in 1.81 s (3.0 ms/step)** on PJRT-CUDA — loss
`0.992 → 0.047`, **98.1 %** on held-out points — and the same program falls back
to the host interpreter on a laptop.
→ [`examples/gpu-training`](examples/gpu-training/)

### Ship it — a directory, then no framework

Compilation produces a content-addressed artifact: StableHLO bodies, a manifest,
and staged weights. Kotlin writes it and exits.

```bash
./gradlew -p examples/gpu-inference run          # Kotlin: writes the artifact, exits
export TLALOC_PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so
/usr/bin/python3 examples/gpu-inference/serve.py # Python: no jax, no torch, no numpy
```

```
    prompt      ' The capital of France is'
    completion  ' Paris.\n\n2.'
```

The serving process imports `argparse`, `json`, `pathlib`, `sys`, `time` and
`tlaloc_serve` — whose dependency list is empty, and a test pins it empty. It
`dlopen`s one plugin `.so` and runs. On this path a real **TinyLlama-1.1B, all
22 layers, generated 6 of 6 token ids identical to HuggingFace transformers** —
both from the direct driver and from **vLLM 0.29.0's `LLM.generate()`**, which
loads a Tlaloc artifact through the platform plugin.
→ [docs/SERVING_RUNBOOK.md](docs/SERVING_RUNBOOK.md)

---

## Installation

No published artifacts yet. Build once, then consume by coordinate:

```bash
./gradlew publishToMavenLocal -x test
```

```kotlin
// settings.gradle.kts — mavenLocal() first
// build.gradle.kts
plugins { kotlin("jvm") version "2.3.20"; application }
kotlin { jvmToolchain(25) }

dependencies {
    implementation("io.tlaloc:core:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:ir:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:autograd:0.0.1-SNAPSHOT")
    implementation("io.tlaloc:nn:0.0.1-SNAPSHOT")          // optional: layers + optimizers
    implementation("io.tlaloc:runtime-pjrt:0.0.1-SNAPSHOT") // optional: GPU execution

    // The K2 plugin — this line is what makes `grad { }` compile-time.
    kotlinCompilerPluginClasspath("io.tlaloc:compiler-plugin:0.0.1-SNAPSHOT")
}
```

Without the plugin on the compiler classpath, `grad { }` throws at the call site
with instructions. Full walkthrough: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

**Requirements:** JDK 25 · Kotlin 2.3.20 · JVM only (see [Maturity](#maturity)) ·
for GPU, a PJRT plugin `.so` and an NVIDIA driver.

---

## Examples

Ten standalone projects under [`examples/`](examples/). Each has its own Gradle
build and resolves Tlaloc from `mavenLocal` exactly as your project would — delete
the rest of the repo and they still run. Six need nothing but a JDK; the others
name what they are missing and exit `0`.

**Start here** — the three that show what is actually different:

| | | Needs |
|---|---|---|
| [`readable-gradients/`](examples/readable-gradients/) | the derivative printed as Kotlin, recompiled without the plugin, agreeing **bit for bit** | — |
| [`differentiable-physics/`](examples/differentiable-physics/) | gradient descent **through a physics simulator** — a loop inside `grad2 {}`, 832 lines of derivative the compiler wrote, and the shot goes in | — |
| [`quickstart/`](examples/quickstart/) | `grad {}` over a matmul, plus a real shape bug and the command that makes the compiler reject it | — |

**Then these:**

| | | Needs |
|---|---|---|
| [`mnist/`](examples/mnist/) | the real MNIST at **93.66 %**, with test digits printed as ASCII next to the model's verdict | CUDA · 11 MB download |
| [`gpu-training/`](examples/gpu-training/) | 600 Adam steps on a Blackwell GPU, 98.3 % held out, decision boundary drawn as ASCII | CUDA |
| [`gpu-inference/`](examples/gpu-inference/) | Kotlin compiles a real TinyLlama and exits; a bare `python3` answers `' Paris.'` | CUDA |
| [`named-indices/`](examples/named-indices/) | axis names checked by Kotlin's own type checker | — |

**How it works inside** — [`examples/internals/`](examples/internals/): the
device decision in the artifact ([`layer3`](examples/internals/layer3/)), the four
scopes and the buffer-handle boundary
([`four-worlds`](examples/internals/four-worlds/)), and the TPU bring-up program
written before the hardware ([`tpu`](examples/internals/tpu/)).

[`examples/README.md`](examples/README.md) has a reading order and the verbatim
output of every one of them from the last full run.

## How it works

```
  your Kotlin            K2 plugin              :ir                 backends
  ───────────            ─────────              ───                 ────────
  grad { f }   ──────►   reverse transform ──► DXIR ──► StableHLO ──► PJRT  (CUDA, TPU*)
  typed tensors          on the lambda's IR     │       + Shardy  ──► IREE  (CPU, CUDA)
  named axes             (also: Kotlin source)  │                  ──► host interpreter
                                                └── recognize + coarsen (FlashAttention,
                                                    GQA, RMSNorm, RoPE, SwiGLU, …), and
                                                    optionally claim ops with KPTX kernels
```

Tlaloc does not write CUDA. It lowers to MLIR and lets XLA or IREE do codegen —
except where a KPTX kernel claims a recognized op. The device decision lives in
the artifact, upstream of the runtime, which is why the same program emits a
different artifact for a GB10, an H100, a TPU v6e or a Trainium2.

---

## Maturity

Every claim in this repository carries one of four marks, and they mean exactly
this:

| | |
|---|---|
| ✅ **Certified** | an automated test pins it, and the docs say on what hardware it ran |
| 🧪 **Written** | the code exists and unit-tests pass, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design document exists; no implementation |
| ❌ **Not planned** | |

| Area | | |
|---|---|---|
| Autodiff — reverse, forward, higher-order, custom rules, control flow | ✅ | Full [DiffKT](https://github.com/facebookresearch/diffkt) parity; one engine, no runtime tape |
| Readable gradient source | ✅ | Scalar lambdas via the plugin; tensor gradients via the capture route |
| Typed tensors, named axes | ✅ | Mismatches are compile errors |
| Op surface, NN ops, special functions, stateless RNG, sparse | ✅ | RNG is bit-exact against JAX's threefry stream |
| dtypes | ✅ | F32, F64, I32, BF16 (end to end, incl. native PJRT bf16) — ❌ no F16/FP8 |
| Model layer + optimizers (`:nn`) | ✅ | Loss curve matches PyTorch to 7 decimals |
| Training on GPU | ✅ | Certified on an NVIDIA GB10 (Blackwell, aarch64) |
| Inference: paged attention, KV cache, safetensors, framework-free serving | ✅ | Real TinyLlama-1.1B, 6/6 tokens identical to HuggingFace |
| vLLM platform plugin | ✅ | `LLM.generate()` runs a real TinyLlama from a Tlaloc artifact; `vllm serve`'s HTTP layer is not yet run |
| StableHLO + Shardy emission, PJRT from Kotlin (FFM) and Python (ctypes), IREE | ✅ | No JNI anywhere |
| KPTX — PTX DSL, parser, transpiler, kernel claiming | ✅ | Paged attention: **1.4–1.9× faster than XLA** at 8B-shaped decode points, 1.6–1.8× slower at toy shapes. Not registered by default — opt in per shape, at shapes you measured |
| Google TPU | 🧪 | Plugin lane, gating and a self-skipping smoke suite exist; **nothing has ever run on a TPU** |
| Distributed / multi-GPU training | 📐 | Design and marshalling done; needs 2+ hosts |
| Android / iOS / WASM | ❌ | The modules are KMP-structured, which makes these reachable later; no such target is declared or built |

The full row-by-row matrix, with what pins each row, is in
[docs/CAPABILITIES.md](docs/CAPABILITIES.md).

### Before you invest

- **Pre-alpha packaging.** No Maven Central release; build from source; APIs move.
- **JVM only today.**
- **No Python API.** Interop is via StableHLO artifacts, not bindings.
- **Not a PyTorch clone.** The model layer targets DiffKT's surface, not `torch.nn`'s.
- **The interpreter is a correctness engine, not a fast CPU backend.** Performance
  claims mean the compiled GPU path.
- **Codegen is mostly XLA's.** KPTX kernels win at 8B-shaped paged attention and
  lose at small shapes, so none is registered by default. We publish the losses.
- **TPU is written, not run.** Treat every TPU claim as untested.

---

## Documentation

| | |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first gradient, first compile error |
| [CAPABILITIES.md](docs/CAPABILITIES.md) | The full capability matrix and what certifies each row |
| [READABLE_REVERSE.md](docs/READABLE_REVERSE.md) | Generated gradient source, side by side with its input |
| [AD_SINGLE_ENGINE_AUDIT.md](docs/AD_SINGLE_ENGINE_AUDIT.md) | Why there is exactly one AD engine, and the test that pins it |
| [DIFFKT_PARITY_PLAN.md](docs/DIFFKT_PARITY_PLAN.md) · [MODEL_LAYER_PLAN.md](docs/MODEL_LAYER_PLAN.md) | The op/AD surface and the `:nn` design |
| [SERVING_RUNBOOK.md](docs/SERVING_RUNBOOK.md) · [INFERENCE_SERVING_AUDIT.md](docs/INFERENCE_SERVING_AUDIT.md) | Export an artifact and serve it; what is certified vs. written |
| [TLALOC_EMIT_CONTRACT.md](docs/TLALOC_EMIT_CONTRACT.md) | What our StableHLO looks like, op by op |
| [KPTX_PLAN.md](docs/KPTX_PLAN.md) · [KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md) | The PTX DSL, and where our kernels stand against XLA |
| [TPU_READINESS_AUDIT.md](docs/TPU_READINESS_AUDIT.md) · [TPU_BRINGUP.md](docs/TPU_BRINGUP.md) | The TPU plan and the bring-up runbook |
| [MULTIHOST_DESIGN.md](docs/MULTIHOST_DESIGN.md) | Distributed execution, as designed |

---

## Development

```bash
./gradlew test                    # the whole suite
./gradlew test --rerun-tasks      # a true clean-room re-run
bash scripts/count-tests.sh       # aggregate count across modules
bash scripts/onboarding-smoke.sh  # publish + run the quickstart, end to end
```

Contributions are welcome; the house style is worth knowing first:

- **Claims are certified.** A capability ships with a test against an analytic
  or cross-implementation oracle, not a screenshot.
- **Unsupported cases refuse loudly, by name.** Nothing silently degrades.
- **Anything deferred is named in a design document**, never left implicit.
- **Negative results get published too.** [KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md)
  exists because our own kernel lost to XLA.

---

## Acknowledgments

φ-calculus coarsening follows Shen, Zhang, Dea et al., *Coarsening Optimization
for Differentiable Programming*, Proc. ACM Program. Lang. 5, OOPSLA, Article 130
(October 2021) — [DOI 10.1145/3485507](https://doi.org/10.1145/3485507),
[arXiv:2110.02307](https://arxiv.org/abs/2110.02307). The op
surface and model layer take their parity target from
[facebookresearch/diffkt](https://github.com/facebookresearch/diffkt); the
readable-derivative idea comes from [google/tangent](https://github.com/google/tangent).
Lowering targets [OpenXLA](https://github.com/openxla) StableHLO, Shardy and
PJRT. Orchestration vendors [Netflix Maestro](https://github.com/Netflix/maestro).

<!-- LICENSE: this repository has no LICENSE file yet, so the code is
     "all rights reserved" by default — which sits badly with the
     "developed in the open" framing above and with inviting people to
     build from source. Apache-2.0 would match the OpenXLA / DiffKT /
     Maestro lineage. Add the file, then add a `## License` section here. -->
