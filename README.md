<div align="center">

# Tlaloc

**Autodiff for Kotlin — compiled, typed, and readable.**

[Quickstart](#quickstart) · [Install](#install) · [Examples](examples/) · [Maturity](#maturity) · [Docs](#documentation) · [License](#license)

</div>

---

Tlaloc differentiates ordinary Kotlin functions at compile time. A K2 compiler
plugin rewrites `grad { }` into gradient code in your bytecode, and lowers it to
[StableHLO] to run on GPU through [PJRT] or [IREE].

There is no tape, no `requires_grad`, and no framework object in your types.
`grad { f }` gives back a plain Kotlin function.

```kotlin
val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
```

Four things follow, and they are the reasons to look at Tlaloc rather than JAX or
PyTorch:

- **Gradients are code, not a runtime trace.** The derivative is synthesized into
  your bytecode during compilation.
- **Gradients are code you can read.** One compiler flag prints the derivative as
  Kotlin source. That source compiles, runs without the plugin, and is
  bit-identical to what the compiler produced.
- **Shape bugs are compile errors.** Rank, dtype and *axis names* live in the
  Kotlin type system.
- **Deployment is an artifact, not a runtime.** A compiled model is a directory of
  StableHLO. Serving it needs a PJRT plugin `.so` and a driver — no JVM, no Python
  framework, nothing of Tlaloc in the process.

> **Alpha — `0.1.0-alpha02`.** The automated suite includes live GPU runs on an
> NVIDIA GB10 ([CAPABILITIES.md](docs/CAPABILITIES.md) has the count and what each
> test pins). Published on Maven Central as `io.github.pedronahum:tlaloc-*`.
> APIs change without deprecation cycles
> ([COMPATIBILITY.md](docs/COMPATIBILITY.md)). [Maturity](#maturity) says what runs
> where.

[StableHLO]: https://openxla.org/stablehlo
[PJRT]: https://openxla.org/xla/pjrt
[IREE]: https://iree.dev

---

## Quickstart

JDK 25 to build, Kotlin 2.4.20. No GPU required.

```bash
git clone https://github.com/pedronahum/tlaloc && cd tlaloc
./gradlew publishToMavenLocal -x test
./gradlew -p examples/quickstart run
```

```kotlin
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // Rewritten at compile time: d/dA sum(A·A) = 1·Aᵀ + Aᵀ·1
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
    println(g(a).hostF32().toList())   // [7.0, 11.0, 9.0, 13.0]
}
```

Call `g` in a hot loop and nothing allocates a tape, because there is no tape.

---

## What is different

### Read the derivative

```kotlin
tlaloc {
    dumpGradSource.set(true)                                   // print it
    dumpGradSourceDir.set(layout.buildDirectory.dir("grads"))  // one .kt per lambda, in grads/main
}
```

For `f(x) = x·σ(x) / √(1 + log(1 + eˣ))` that writes:

```kotlin
fun grad_body_grad(x: DTensor<ScalarShape, F32>): DTensor<ScalarShape, F32> {
    val v1  = x.sigmoid()      // %1  = SIGMOID(%0)
    val v2  = (x * v1)         // %2  = MUL(%0, %1)
    ...
    val v32 = (v26 + v31)      // %32 = ADD(%26, %31)
    return v32
}
```

Every line is a real `io.tlaloc.core` call, so the file compiles and runs on its
own. [`examples/readable-gradients`](examples/readable-gradients/) recompiles it in
a separate source set and checks it against the compiled gradient: raw-bit
identical at all 7 test points. → [READABLE_REVERSE.md](docs/READABLE_REVERSE.md)

### Axis names the type checker enforces

```kotlin
val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> = ...
val weights:     DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> = ...

val hidden = activations contract weights   // OK: they share SeqLen
```

Swap in a `Hidden × Hidden` matrix and the call does not resolve. The plugin adds
`NAMED_INDEX_MISMATCH`, `TENSOR_SHAPE_MISMATCH` and `NOT_DIFFERENTIABLE` on top,
reported at the offending call's own file, line and column.
→ [`examples/named-indices`](examples/named-indices/)

### Differentiate through a loop

Control flow inside `grad { }` is differentiated, not unrolled by hand. Coarsening
follows Shen et al., *Coarsening Optimization for Differentiable Programming*
([OOPSLA 2021][phi]), with a closed-form solver behind it.
[`examples/differentiable-physics`](examples/differentiable-physics/) puts a
38-step Euler integrator in a `valueAndGrad2 { }` and gets an 823-operation derivative of
the whole simulator.

Forward mode is `jvp` / `jvp2`. `vjp`, `jacobian`, `jacobianReverse` and `hessian`
are there and nest. Custom rules go in with `customVjp`, `customJvp`,
`customVjpJvp`. A lambda may reference values declared outside it — compile-time
constants are folded, runtime values become bound parameters — and a body the
plugin cannot lower is a build error carrying the reason.

[phi]: https://doi.org/10.1145/3485507

### Train, checkpoint, serve

`:nn` is functional: layers are immutable, a training step returns a new model,
optimizers are pure `(params, grads, state) → (params', state')`.

```kotlin
val keys = RandomKey.fromSeed(7).split(2)
val model0 = Sequential(Dense(2, 16, keys[0]), ReluLayer, Dense(16, 1, keys[1]))

val step = capture(model0, listOf(x), name = "mlp") { prediction ->
    val residual = prediction - prediction.constant<Shape>(targets, intArrayOf(n, 1))
    (residual * residual).mean()
}

val optimizer = Adam(learningRate = 0.02f)
var model = model0
var state = optimizer.initialState()
repeat(60) {
    val out = step.run(model, listOf(x))   // loss and gradients, on the host
    val (nextModel, nextState) = optimizer.step(model, out.gradients, state)
    model = nextModel; state = nextState
}

saveCheckpoint(path, model, optimizer, state)   // one safetensors file, resumable
```

No `.backward()`, no `zero_grad()`. `capture` traces the model once; `step.run`
evaluates the captured gradient on the host interpreter. The same captured program
also lowers to StableHLO: [`examples/gpu-training`](examples/gpu-training/) runs it
on PJRT-CUDA, 600 Adam steps in 2.02 s (3.4 ms/step) on a GB10, loss
`0.992 → 0.047`, 98.0 % held out. The checkpoint is an ordinary safetensors file
that `safetensors.torch.load_file` opens, and a resumed run matches the
uninterrupted one bit for bit.

Inference goes the other way — Kotlin writes an artifact and exits:

```bash
./gradlew -p examples/gpu-inference run          # Kotlin writes the artifact
/usr/bin/python3 examples/gpu-inference/serve.py # no jax, no torch, no numpy
# finds a jax CUDA plugin under a venv; otherwise set TLALOC_PJRT_PLUGIN_PATH
```

```
    prompt      ' The capital of France is'
    completion  ' Paris.\n\n2.'
```

A real TinyLlama-1.1B, all 22 layers, generated 6 of 6 token ids identical to
HuggingFace transformers — from the direct driver and from vLLM 0.29.0's
`LLM.generate()`. → [SERVING_RUNBOOK.md](docs/SERVING_RUNBOOK.md)

Tlaloc StableHLO also serves from NVIDIA Triton Inference Server through a C++
backend that compiles it with the PJRT CUDA plugin. A Tlaloc-generated gradient
answers over Triton's HTTP and gRPC endpoints with the same bits the DXIR
interpreter produces, and the TinyLlama artifact, written as a Triton model from
Kotlin, generates the same 6 token ids with its weights and KV cache held by the
backend. The same path serves Qwen3-0.6B and the text decoder of Muse Glimmer
(meta-models/Muse-Glimmer-30B: 28 billion parameters, sliding-window and NoPE
layers, soft-capped logits), whose 56 GB of weights stay bf16 on the GB10, at
about 245 ms a token. Its greedy ids equal those of HuggingFace transformers run
with the same arithmetic (bf16 weights, f32 activations), 32 of 32, and those of
transformers in bfloat16 up to the one token where the two transformers runs
differ from each other.
→ [triton/](triton/README.md)

---

## Install

From Maven Central. The Gradle plugin is published there too, not to the Gradle
Plugin Portal, so `pluginManagement` needs `mavenCentral()`:

```kotlin
// settings.gradle.kts
pluginManagement { repositories { mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { mavenCentral() } }
```

```kotlin
// build.gradle.kts
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.4.20"
    id("io.github.pedronahum.tlaloc") version "0.1.0-alpha02"  // makes `grad { }` compile-time
    application
}
kotlin {
    jvmToolchain(25)                                     // to BUILD
    compilerOptions { jvmTarget.set(JvmTarget.JVM_21) }  // to RUN on JDK 21
}
java {                                                   // Java tasks match Kotlin's target
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:0.1.0-alpha02"))
    implementation("io.github.pedronahum:tlaloc-autograd")
    implementation("io.github.pedronahum:tlaloc-nn")           // layers + optimizers
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt") // GPU execution; runs on JDK 25 only
}
```

The Gradle plugin applies `tlaloc-compiler-plugin` of its own version to every
Kotlin/JVM compilation; its options go in a `tlaloc { }` block. The BOM keeps
every `tlaloc-*` artifact on one version.

**JDK 25 to build, JDK 21 to run.** Kotlin loads a compiler plugin inside the
compiler's own JVM and `compiler-plugin` is Java 25 bytecode, so building
`grad { }` needs 25. The library modules target Java 21, so running what you built
needs only 21 — except PJRT and CUDA execution, which need 25.

**Also:** Kotlin 2.4.20–2.4.29, refused by name outside that range (on Kotlin
2.3.x, use `0.1.0-alpha01`; [COMPATIBILITY.md](docs/COMPATIBILITY.md) has the
table) · JVM only ·
for GPU, a PJRT plugin `.so` and an NVIDIA driver · Symja (LGPL-3.0) is optional
and not in your dependency graph unless you add it.

To build from source instead, run `./gradlew publishToMavenLocal -x test` in a
checkout and put `mavenLocal()` first in both repository blocks.

Full walkthrough and the five plugin options:
[GETTING_STARTED.md](docs/GETTING_STARTED.md). Per-module detail:
[COMPATIBILITY.md](docs/COMPATIBILITY.md).

---

## Examples

Ten standalone projects under [`examples/`](examples/), each with its own Gradle
build resolving Tlaloc from `mavenLocal`. Delete the rest of the repo and they
still run. Six need nothing but a JDK; the others name what they are missing and
exit `0`. The seven below are the user-facing ones.

| | | Needs |
|---|---|---|
| [`readable-gradients/`](examples/readable-gradients/) | the derivative printed as Kotlin, recompiled without the plugin, agreeing bit for bit | — |
| [`differentiable-physics/`](examples/differentiable-physics/) | gradient descent through a physics simulator, and the shot goes in | — |
| [`quickstart/`](examples/quickstart/) | `grad {}` over a matmul, plus a shape bug the compiler rejects | — |
| [`named-indices/`](examples/named-indices/) | axis names in the tensor type, so a transposed weight fails overload resolution | — |
| [`mnist/`](examples/mnist/) | the real MNIST at 93.66 %, test digits as ASCII | CUDA · 11 MB |
| [`gpu-training/`](examples/gpu-training/) | 600 Adam steps on a Blackwell, 98.0 % held out | CUDA |
| [`gpu-inference/`](examples/gpu-inference/) | Kotlin compiles TinyLlama; a bare `python3` answers `' Paris.'` | CUDA |

[`examples/README.md`](examples/README.md) has the reading order, the three
internals projects, and verbatim output from the last full run.

---

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

Tlaloc does not write CUDA. It lowers to MLIR and lets XLA or IREE do codegen,
except where a KPTX kernel claims a recognized op. The device decision lives in the
artifact, upstream of the runtime.

---

## Maturity

Every claim in this repository carries one of these marks:

| | |
|---|---|
| ✅ **Certified** | an automated test pins it, and the docs say on what hardware it ran |
| 🧪 **Written** | the code exists and unit-tests pass, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design document exists; no implementation |
| ❌ **Not planned** | |

| Area | | |
|---|---|---|
| Autodiff — reverse, forward, higher-order, custom rules, control flow | ✅ | Full [DiffKT](https://github.com/facebookresearch/diffkt) parity; one engine, no runtime tape |
| Readable gradient source | ✅ | Printed source compiles and matches the compiled gradient bit for bit |
| Typed tensors, named axes | ✅ | Mismatches are compile errors |
| Op surface, NN ops, special functions, stateless RNG, sparse | ✅ | RNG is bit-exact against JAX's threefry stream |
| dtypes — F32, F64, I32, BF16 | ✅ | Includes native PJRT bf16 and mixed precision. ❌ no F16/FP8 |
| Model layer, optimizers, schedules, clipping, checkpoints (`:nn`) | ✅ | Loss curve matches PyTorch to 7 decimals; checkpoint round trip is bit-identical |
| Training on GPU | ✅ | Certified on an NVIDIA GB10 (Blackwell, aarch64) |
| Inference — paged attention, KV cache, safetensors, framework-free serving | ✅ | Real TinyLlama-1.1B, 6/6 tokens identical to HuggingFace, also through vLLM; Qwen3-0.6B and Muse Glimmer 30B (text, bf16 weights) through Triton with HuggingFace's greedy ids |
| StableHLO + Shardy emission, PJRT from Kotlin (FFM) and Python (ctypes), IREE | ✅ | No JNI anywhere |
| KPTX — PTX DSL, parser, transpiler, kernel claiming | ✅ | Paged attention 1.4–1.9× faster than XLA at 8B-shaped decode points, 1.6–1.8× slower at toy shapes. Not registered by default |
| Public API surface — opt-in marker, ABI baseline, API reference | ✅ | `checkKotlinAbi` against a committed baseline for every published module, wired into `check` |
| Google TPU | 🧪 | Plugin lane, gating and a self-skipping smoke suite exist. Nothing has ever run on a TPU |
| IDE diagnostics — the "red squiggle" | 🧪 | Source positions are certified; no test drives IntelliJ |
| Distributed / multi-GPU training | 📐 | Design and marshalling done; needs 2+ hosts |
| Android / iOS / WASM | ❌ | Modules are KMP-structured, so these are reachable later; no target is declared |

Row by row, with what pins each one: [CAPABILITIES.md](docs/CAPABILITIES.md).

### Limits

- **JVM only. No Python API.** Interop is via StableHLO artifacts, not bindings.
- **The interpreter is a correctness engine, not a fast CPU backend.** Performance
  claims mean the compiled GPU path.
- **Codegen is mostly XLA's.** KPTX wins at 8B-shaped paged attention and loses at
  small shapes, so none is registered by default. The losses are published in
  [KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md).
- **TPU is written, not run.** Treat every TPU claim as untested.
- **No IDE plugin and no hosted docs.** `./gradlew apiDocs` builds the API reference
  locally.
- **Not a PyTorch clone.** The model layer targets DiffKT's surface, not `torch.nn`'s.
- **One copyleft dependency you do not inherit.** Symja (LGPL-3.0) is `compileOnly`
  and kept out of published POMs by a gate. There is no Symja-free `SymbolicEngine`,
  so a policy forbidding LGPL outright means no CAS, and no differentiating a loop
  whose trip count is not a compile-time constant.

---

## Documentation

| | |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first gradient, first compile error |
| [CAPABILITIES.md](docs/CAPABILITIES.md) | The capability matrix and what certifies each row |
| [COMPATIBILITY.md](docs/COMPATIBILITY.md) · [CHANGELOG.md](CHANGELOG.md) | What alpha promises, what may break, what changed |
| [READABLE_REVERSE.md](docs/READABLE_REVERSE.md) | Generated gradient source, beside its input |
| **API reference** | `./gradlew apiDocs` → `build/docs/api/index.html` (not hosted) |

Design documents for the IR, the emitter, the serving path, KPTX, TPU bring-up and
distributed execution are in [`docs/`](docs/).

---

## Development

```bash
./gradlew test                    # the whole suite
./gradlew test --rerun-tasks      # a true clean-room re-run
bash scripts/count-tests.sh       # aggregate count across modules
bash scripts/onboarding-smoke.sh  # publish + run the quickstart, end to end
bash scripts/jdk21-smoke.sh       # run a synthesized gradient on a real JDK 21
```

CI runs five lanes: the suite on x86_64 Linux, aarch64 Linux and arm64 macOS, the
library suites on a JDK 21, and a next-Kotlin probe that is allowed to fail. A
release is tagged only on a commit where the four build lanes are green
([RELEASING.md](docs/RELEASING.md)). A green runner means the platform-neutral subset passes
— a runner has no GPU, no PJRT plugin, no IREE and no oracle venv, and every test
needing one self-skips by name. The GPU rows above are certified on the GB10, not by
CI.

The next-Kotlin lane is green by construction, so read its step outcomes rather than
its checkmark. Its current answer, for 2.5.0-Beta1: the library modules compile
and pass, the compiler plugin compiles, and the plugin refuses 2.5 by name until
it is ported.

House style, if you are contributing:

- **Claims are certified.** A capability ships with a test against an analytic or
  cross-implementation oracle, not a screenshot.
- **Unsupported cases refuse loudly, by name.** Nothing silently degrades.
- **Anything deferred is named in a design document.**
- **Negative results get published too.**
  [KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md) exists because our own kernel lost
  to XLA.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build and review checklist and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.

---

## Acknowledgments

φ-calculus coarsening follows Shen, Zhang, Dea et al., *Coarsening Optimization for
Differentiable Programming*, Proc. ACM Program. Lang. 5, OOPSLA, Article 130 (2021)
— [DOI 10.1145/3485507](https://doi.org/10.1145/3485507). The op surface and model
layer take their parity target from
[facebookresearch/diffkt](https://github.com/facebookresearch/diffkt) (MIT); the
readable-derivative idea comes from
[google/tangent](https://github.com/google/tangent). Lowering targets
[OpenXLA](https://github.com/openxla) StableHLO, Shardy and PJRT. Orchestration
vendors [Netflix Maestro](https://github.com/Netflix/maestro).

## License

Copyright 2026 Pedro N. Rodriguez. Licensed under [Apache-2.0](LICENSE).

`third-party/maestro/` vendors [Netflix Maestro](https://github.com/Netflix/maestro)
(Apache-2.0); [NOTICE](NOTICE) credits it. Tlaloc does not publish it. The paper text in
`docs/papers/` is CC-BY-4.0; see [its README](docs/papers/README.md).

Symja (`org.matheclipse:matheclipse-core`), the optional CAS, is **LGPL-3.0** by its
published POM, while its upstream repository's `license.txt` is GPL-3.0; upstream's
position is that the published Maven modules are LGPL. Tlaloc links it across the
`SymbolicEngine` interface and never modifies or redistributes it. It is
`compileOnly`, so it is not in your dependency graph unless you add it.
