<div align="center">
<img src="docs/images/tlaloc_logo.svg" alt="Tlaloc logo: a rain cloud raining three drops joined as a graph" width="180">

# Tlaloc

**Compile-time automatic differentiation and GPU serving for Kotlin**

[![build](https://github.com/pedronahum/tlaloc/actions/workflows/build.yml/badge.svg)](https://github.com/pedronahum/tlaloc/actions/workflows/build.yml)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.pedronahum/tlaloc-core)](https://central.sonatype.com/namespace/io.github.pedronahum)
[![javadoc](https://javadoc.io/badge2/io.github.pedronahum/tlaloc-core-jvm/javadoc.svg)](https://javadoc.io/doc/io.github.pedronahum/tlaloc-core-jvm)

[**Transformations**](#transformations)
| [**Training**](#training)
| [**Serving**](#serving)
| [**Install**](#installation)
| [**Examples**](examples/)
| [**Changelog**](CHANGELOG.md)
| [**Docs**](#documentation)

</div>

## What is Tlaloc?

Tlaloc is a Kotlin library for differentiable array programs. A K2 compiler
plugin turns `grad { }` into gradient code at compile time. Tensors carry their
rank, dtype and axis names in the Kotlin type. Programs lower to
[StableHLO](https://openxla.org/stablehlo) and run on NVIDIA GPUs through
[PJRT](https://openxla.org/xla/pjrt) or [IREE](https://iree.dev), or on the CPU
through a reference interpreter.

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

`grad { f }` returns a plain Kotlin function. There is no tape and no runtime
tracing. A model compiled for serving is a directory of StableHLO plus weights;
serving it needs a PJRT plugin and a GPU driver, not a JVM.

This is an alpha, `0.1.0-alpha02`. APIs can change between alphas without a
deprecation cycle ([COMPATIBILITY.md](docs/COMPATIBILITY.md)).

### Contents

* [Transformations](#transformations)
* [Typed tensors](#typed-tensors)
* [Training](#training)
* [Serving](#serving)
* [Supported platforms](#supported-platforms)
* [Installation](#installation)
* [Sharp edges](#sharp-edges)
* [Examples](#examples)
* [Documentation](#documentation)

## Transformations

Each transformation takes a lambda and returns a function. The compiler plugin
builds the derivative when your code compiles.

### Reverse mode: `grad`

```kotlin
val df = grad { x: Float -> x * x * x }
df(2f)                                    // 12.0

val dfdxy = grad2 { x: Float, y: Float -> x * y + x }
```

`grad2` and `grad3` take two and three arguments. `valueAndGrad` returns the
value too.

### Forward mode and higher order

```kotlin
val f   = jvp { x: Float -> x * x * x }
f(2f, 1f)                                 // 12.0, the directional derivative

val j = jacobian { x: DTensor<Rank1<Sym>, F32> -> x * x }                   // diag(2x)
val h = hessian  { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() } // 2·I
```

`vjp` and `jacobianReverse` are also available, and the transformations nest.

### Loops, branches and custom rules

`if`, `when` and `for` loops inside the lambda are differentiated.
[`examples/differentiable-physics`](examples/differentiable-physics/)
differentiates a 38-step Euler integrator. `customVjp`, `customJvp` and
`customVjpJvp` supply your own derivative for a function.

The lambda can use values declared outside it. Compile-time constants are
folded in; other values become parameters of the generated function. A lambda
the plugin cannot differentiate is a compile error that names the reason.

### Read the derivative

```kotlin
tlaloc {
    dumpGradSource.set(true)                                   // print it
    dumpGradSourceDir.set(layout.buildDirectory.dir("grads"))  // one .kt per lambda
}
```

The plugin prints the derivative as Kotlin source. That source compiles without
the plugin and returns the same bits as the compiled gradient.
[`examples/readable-gradients`](examples/readable-gradients/) checks this at 7
points. → [READABLE_REVERSE.md](docs/READABLE_REVERSE.md)

## Typed tensors

```kotlin
val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> = ...
val weights:     DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> = ...

val hidden = activations contract weights   // compiles: both have SeqLen
```

With a `Hidden × Hidden` weight the call does not compile. Shape and
differentiability errors are reported at the call's file, line and column.
→ [`examples/named-indices`](examples/named-indices/)

| | |
|---|---|
| dtypes | F32, F64, I32, BF16 (I8 for quantized serving weights) |
| Ops | elementwise, broadcasting, reductions, shape ops, matmul, conv2d, pooling, softmax, embedding, losses, batch norm |
| Special functions | `lgamma`, `digamma`, `polygamma`, `integral` |
| Linear algebra | `cholesky`, `triangularSolve`, `solveSpd`, `logDetSpd`, `invSpd`, `solve`, `det`, `qrQ`/`qrR`, `eighValues`/`eighVectors`; rank 2, differentiable to any order |
| Random numbers | stateless threefry-2x32, bit-exact with JAX |
| Sparse | rank-2 CSR, sparse × dense matmul under `grad` (CPU only) |

## Training

`:nn` has immutable layers and pure optimizers: a training step returns a new
model and a new optimizer state.

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
    val out = step.run(model, listOf(x))   // loss and gradients
    val (nextModel, nextState) = optimizer.step(model, out.gradients, state)
    model = nextModel; state = nextState
}

saveCheckpoint(path, model, optimizer, state)   // one safetensors file
```

| | |
|---|---|
| Layers | Dense, Conv2d, MaxPool, AvgPool, BatchNorm, Dropout, Embedding, EmbeddingBag, GRU, Flatten |
| Transformers | LayerNorm, RMSNorm, RoPE, MultiHeadAttention (causal, grouped-query), SwiGLU, TransformerBlock, CausalLM; gradients match PyTorch to 2.4e-6 |
| Optimizers | SGD, Momentum, RMSprop, Adam, AdamW |
| Also | learning-rate schedules, gradient clipping, bf16 mixed precision, checkpoints that resume bit for bit |
| On the GPU | [`examples/gpu-training`](examples/gpu-training/): 600 Adam steps in 2.02 s on a GB10, 98.0 % held-out accuracy; [`examples/mnist`](examples/mnist/): 93.66 % on MNIST |
| Hugging Face models | `HfCausalLm` reads a Llama or Qwen3 checkpoint into a `CausalLM` and writes one back; [`examples/fine-tune`](examples/fine-tune/) fine-tunes Qwen3-0.6B on the GPU |
| Tokenizers | `HfTokenizer` (`:tokenizer`) reads a checkpoint's `tokenizer.json` and gives the ids, decoded text and chat prompts transformers gives, for Qwen3, Muse Glimmer, TinyLlama, GPT-2 and Gemma 4 |

## Serving

Kotlin reads a Hugging Face checkpoint and writes a serving artifact: StableHLO
prefill and decode programs, the weights as safetensors, and a manifest. Three
runtimes load it. All three compile it with a PJRT plugin, and none runs a JVM.

| Runtime | What it is |
|---|---|
| NVIDIA Triton | `libtriton_tlaloc.so`, a Triton backend. HTTP and gRPC, Triton's sequence batcher, KV cache held by the backend. → [triton/](triton/README.md) |
| vLLM | a vLLM platform plugin; `LLM.generate()` runs a Tlaloc artifact |
| Python | a small runtime that calls the PJRT C API through ctypes, with no JAX, PyTorch or NumPy |

**Models.** Each one generates the same greedy token ids as Hugging Face
transformers.

| Model | Weights | Decode step (GB10, Triton) |
|---|---|---|
| [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | f32 | 21.7 ms |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | bf16 | 10.7 ms |
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), text only | bf16, 53 GB | 244 ms |
| Muse Glimmer 30B, text only | int8 (opt-in), 29 GB | 140 ms |

Muse Glimmer runs contexts up to 32,768 tokens and several sequences at once. A
31,744-token prompt prefills in 62 s. Int8 weights keep its fixture answers and
change perplexity from 5.610 to 5.602.

**Serving features:** paged attention; sliding-window layers whose KV cache is
bounded by the window; chunked and batched prefill; KV cache updated in place;
continuous batching of decode steps; CUDA shared-memory inputs and outputs;
dynamic batching for stateless models.

[`examples/triton-llm`](examples/triton-llm/) exports Qwen3-0.6B, starts Triton
and streams a chat answer in one script.
→ [SERVING_ARCHITECTURE.md](docs/SERVING_ARCHITECTURE.md)

## Supported platforms

|  | Linux aarch64 | Linux x86_64 | macOS arm64 | Windows |
|---|---|---|---|---|
| Build, library, CPU interpreter | ✅ | ✅ | ✅ | not tested |
| NVIDIA GPU (PJRT) | ✅ | expected, not tested | n/a | not tested |
| Triton serving | ✅ | expected, not tested | n/a | n/a |
| Google TPU | written, never run | written, never run | n/a | n/a |

All GPU results come from one machine, an NVIDIA GB10 (Blackwell, aarch64). CI
builds and tests the library on the three operating systems in the table, with
no GPU. → [CAPABILITIES.md](docs/CAPABILITIES.md) lists every capability and
the test that certifies it.

## Installation

From Maven Central. The Gradle plugin is also on Maven Central, not on the
Gradle Plugin Portal, so `pluginManagement` needs `mavenCentral()`.

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

| Requirement | |
|---|---|
| Kotlin | 2.4.20 – 2.4.29; other versions are refused at compile time. On Kotlin 2.3.x, use `0.1.0-alpha01`. |
| JDK | 25 to build code that uses `grad { }`. 21 to run it, except GPU execution, which needs 25. |
| GPU | an NVIDIA driver and a PJRT CUDA plugin, e.g. `pip install "jax[cuda12]"` in a virtual environment ([GETTING_STARTED.md](docs/GETTING_STARTED.md#5-running-on-a-gpu)) |
| Triton serving | Docker and `nvcr.io/nvidia/tritonserver:25.11-py3` ([triton/](triton/README.md)) |

To build from source, run `./gradlew publishToMavenLocal -x test` in a checkout
and put `mavenLocal()` first in both repository blocks.

Walkthrough, plugin options, configuration and troubleshooting:
[GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Sharp edges

- **Alpha.** Names and signatures can change between alphas.
- **The CPU interpreter is for correctness, not speed.** Performance numbers are
  for the GPU path.
- **Tlaloc does not generate GPU code.** XLA and IREE do. KPTX, a PTX kernel DSL,
  can replace a recognized op, but no kernel is on by default: its paged
  attention wins at large shapes and loses at small ones
  ([KPTX_PAGED_PERF.md](docs/KPTX_PAGED_PERF.md)).
- **One Kotlin release per Tlaloc version**, because the plugin uses compiler
  internals.
- **No Python API.** Other languages use the StableHLO artifact.
- **Not written yet or never run:** TPU, multi-host training, more than one GPU
  per Triton model, preempting a live sequence, image input for Muse Glimmer.
- **Symja** (LGPL-3.0), used to differentiate loops whose trip count is not a
  compile-time constant, is optional and not in your dependency graph unless you
  add it.

## Examples

Standalone projects under [`examples/`](examples/). Each has its own Gradle build
and skips by name when its hardware is missing.

| | | Needs |
|---|---|---|
| [`quickstart/`](examples/quickstart/) | `grad {}` over a matmul, and a shape error the compiler rejects | JDK |
| [`readable-gradients/`](examples/readable-gradients/) | the derivative printed as Kotlin, recompiled, same bits | JDK |
| [`differentiable-physics/`](examples/differentiable-physics/) | gradient descent through a physics simulation | JDK |
| [`named-indices/`](examples/named-indices/) | axis names in the tensor type | JDK |
| [`gaussian-process/`](examples/gaussian-process/) | GP hyperparameters fitted through a Cholesky solve and log-determinant | JDK |
| [`mnist/`](examples/mnist/) | MNIST to 93.66 % | CUDA |
| [`gpu-training/`](examples/gpu-training/) | 600 Adam steps on the GPU | CUDA |
| [`fine-tune/`](examples/fine-tune/) | Qwen3-0.6B fine-tuned on the GPU with AdamW, saved as a Hugging Face checkpoint | CUDA |
| [`gpu-inference/`](examples/gpu-inference/) | TinyLlama compiled from Kotlin, served by plain `python3` | CUDA |
| [`triton-llm/`](examples/triton-llm/) | Qwen3-0.6B or Muse Glimmer on Triton, with a streaming chat client | CUDA, Docker |

## Documentation

| | |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | install, first gradient, GPU setup, configuration, troubleshooting |
| [CAPABILITIES.md](docs/CAPABILITIES.md) | every capability, its status, and the test behind it |
| [SERVING_ARCHITECTURE.md](docs/SERVING_ARCHITECTURE.md) | how serving works, and where the time goes |
| [COMPATIBILITY.md](docs/COMPATIBILITY.md) · [CHANGELOG.md](CHANGELOG.md) | what may change, and what did |
| [API reference](https://javadoc.io/doc/io.github.pedronahum/tlaloc-core-jvm) | per module on javadoc.io; `./gradlew apiDocs` builds all modules locally |

Design documents are in [`docs/`](docs/).

## How it works

```
  your Kotlin            K2 plugin              :ir                 backends
  ───────────            ─────────              ───                 ────────
  grad { f }   ──────►   reverse transform ──► DXIR ──► StableHLO ──► PJRT  (CUDA, TPU*)
  typed tensors          on the lambda's IR     │       + Shardy  ──► IREE  (CPU, CUDA)
  named axes             (also: Kotlin source)  │                  ──► host interpreter
                                                └── recognize + coarsen (attention,
                                                    RMSNorm, RoPE, SwiGLU, …)
```

Loops are coarsened before differentiation, following Shen et al., *Coarsening
Optimization for Differentiable Programming* ([OOPSLA 2021](https://doi.org/10.1145/3485507)).

## Development

```bash
./gradlew test                    # the suite
./gradlew test --rerun-tasks      # re-run everything
bash scripts/onboarding-smoke.sh  # publish, then build the README's install blocks
bash triton/verify.sh             # Triton backend, end to end (GPU + Docker)
```

Tests that need a GPU, Docker or a model download skip by name without it.
Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security reports:
[SECURITY.md](SECURITY.md).

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
