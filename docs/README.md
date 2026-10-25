<!--mkdocs:<div class="tlaloc-hero" markdown>-->

<p align="center"><img src="images/tlaloc_logo.svg" alt="Tlaloc logo: a rain cloud raining three drops joined as a graph" width="140"></p>

# Tlaloc documentation

<p align="center"><b>Compile-time automatic differentiation and GPU serving for Kotlin</b></p>

<p align="center"><a href="https://github.com/pedronahum/tlaloc">GitHub</a> · <a href="https://central.sonatype.com/namespace/io.github.pedronahum">Maven Central</a> · <a href="https://javadoc.io/doc/io.github.pedronahum/tlaloc-core-jvm">API reference</a> · <a href="../CHANGELOG.md">Changelog</a></p>

<!--mkdocs:</div>-->

Tlaloc is a Kotlin library for differentiable array programs. A K2 compiler plugin
turns `grad { }` into gradient code when your code compiles, and tensors carry their
rank, dtype and axis names in the Kotlin type. Programs lower to StableHLO and run on
NVIDIA GPUs through PJRT, on Apple-silicon GPUs through MetalHLO, or on the CPU in a
reference interpreter.

<!-- doc-test: home-grad -->

```kotlin
val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
println(g(a).hostF32().toList())   // [7.0, 11.0, 9.0, 13.0]
```

> [!WARNING]
> This is an alpha, `0.1.0-alpha03`. APIs can change between alphas without a
> deprecation cycle; [COMPATIBILITY.md](COMPATIBILITY.md) lists what may break.

## What are you trying to do?

| To | Read |
|---|---|
| take a first gradient | [Getting started](GETTING_STARTED.md), then [`examples/quickstart`](../examples/quickstart/) |
| train a model on a GPU | [`examples/gpu-training`](../examples/gpu-training/), then [the case study](case-studies/training-step-on-device.md) |
| fine-tune a Hugging Face model | [`examples/fine-tune`](../examples/fine-tune/), [`examples/lora-finetune`](../examples/lora-finetune/) |
| serve a model | [the serving runbook](SERVING_RUNBOOK.md), [`examples/triton-llm`](../examples/triton-llm/) |
| read the derivative the compiler wrote | [Readable reverse mode](READABLE_REVERSE.md) |
| understand why results differ between backends | [Numerical precision](NUMERICAL_PRECISION.md) |
| find out why a step is slow | [Measuring and profiling](PERFORMANCE.md) |
| check whether something is supported | [Capabilities](CAPABILITIES.md), [FAQ](FAQ.md) |

## Contents

<!--mkdocs:<div class="grid cards" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Get started**

    ---

    Install, take a first gradient, set up a GPU.

    - [<!--mkdocs::octicons-arrow-right-24:--> Getting started](GETTING_STARTED.md): install, first gradient, GPU setup, configuration, troubleshooting
    - [<!--mkdocs::octicons-arrow-right-24:--> Examples](../examples/README.md): eighteen standalone programs, each its own Gradle build
    - [<!--mkdocs::octicons-arrow-right-24:--> FAQ](FAQ.md): short answers, with links

- **<!--mkdocs::material-cog-transfer:{ .lg .middle }--> How it works**

    ---

    Where derivatives are taken, when programs compile, and what each backend supports.

    - [<!--mkdocs::octicons-arrow-right-24:--> How Tlaloc runs a program](HOW_IT_WORKS.md): `grad { }` and `capture`, compiles, buffers, transfers, device memory
    - [<!--mkdocs::octicons-arrow-right-24:--> Readable reverse mode](READABLE_REVERSE.md): gradients printed as Kotlin that compiles
    - [<!--mkdocs::octicons-arrow-right-24:--> Dynamic shapes](DYNAMIC_SHAPES.md): bounded axes, buckets, and what recompiles
    - [<!--mkdocs::octicons-arrow-right-24:--> Distributed](DISTRIBUTED.md): meshes, shardings and collectives: what is emitted, what has run
    - [<!--mkdocs::octicons-arrow-right-24:--> Custom kernels](CUSTOM_KERNELS.md): custom derivative rules, KPTX GPU kernels, Mosaic TPU kernels

- **<!--mkdocs::material-chart-line:{ .lg .middle }--> Analyze your workload**

    ---

    What each backend computes, and where a step's time goes.

    - [<!--mkdocs::octicons-arrow-right-24:--> Numerical precision](NUMERICAL_PRECISION.md): f32, f64 and bf16 by backend, and the tests' tolerances
    - [<!--mkdocs::octicons-arrow-right-24:--> Measuring and profiling](PERFORMANCE.md): timing a step, seeing what XLA compiled, profiling a served model
    - [<!--mkdocs::octicons-arrow-right-24:--> A training step on the device](case-studies/training-step-on-device.md): 25.78 ms to 2.22 ms a step by keeping parameters and optimizer state on the GPU

- **<!--mkdocs::material-server-network:{ .lg .middle }--> Serve**

    ---

    Compile a model into an artifact and serve it without a JVM.

    - [<!--mkdocs::octicons-arrow-right-24:--> Serving architecture](SERVING_ARCHITECTURE.md): how serving works, and where the time goes
    - [<!--mkdocs::octicons-arrow-right-24:--> Serving runbook](SERVING_RUNBOOK.md): export an artifact, run it, plug it into vLLM
    - [<!--mkdocs::octicons-arrow-right-24:--> TPU bring-up](TPU_BRINGUP.md): the steps for a first TPU session

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Reference**

    ---

    - [<!--mkdocs::octicons-arrow-right-24:--> Capabilities](CAPABILITIES.md): every capability, its status, and the test behind it
    - [<!--mkdocs::octicons-arrow-right-24:--> Compatibility](COMPATIBILITY.md): what may change between alphas
    - [<!--mkdocs::octicons-arrow-right-24:--> The emit contract](TLALOC_EMIT_CONTRACT.md): the StableHLO Tlaloc emits, for backend implementers
    - [<!--mkdocs::octicons-arrow-right-24:--> The Maestro descriptor](maestro_descriptor.md): the workflow descriptor format
    - [<!--mkdocs::octicons-arrow-right-24:--> API reference](https://javadoc.io/doc/io.github.pedronahum/tlaloc-core-jvm): per module; `./gradlew apiDocs` builds it locally

- **<!--mkdocs::material-hammer-wrench:{ .lg .middle }--> For maintainers**

    ---

    - [<!--mkdocs::octicons-arrow-right-24:--> Releasing](RELEASING.md): cutting a release
    - [<!--mkdocs::octicons-arrow-right-24:--> Vendoring](vendoring.md): the vendored Maestro tree
    - [<!--mkdocs::octicons-arrow-right-24:--> Open issues](FOLLOWUPS.md): what is known to be wrong or missing
    - [<!--mkdocs::octicons-arrow-right-24:--> Doc snippets](snippets/): the setup and checks behind the docs' Kotlin snippets (`scripts/check-doc-snippets.py`)
    - [<!--mkdocs::octicons-arrow-right-24:--> This site](../mkdocs.yml): `scripts/docs.sh serve` previews it, `scripts/docs.sh build` builds it

<!--mkdocs:</div>-->

## Records

Plans, audits and design notes, kept as written: [plans/](plans/), [audits/](audits/),
[design/](design/), [work-log/](work-log/), [papers/](papers/).
