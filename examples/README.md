# Tlaloc examples

Ten standalone programs. Each directory here is its **own Gradle build** — its own
`settings.gradle.kts`, its own `build.gradle.kts` — and each resolves Tlaloc from
**mavenLocal**, as `io.github.pedronahum:tlaloc-core:0.1.0-alpha02` and friends, exactly the way
your project would. None is a module of the repo build, none uses `includeBuild`,
and none imports another. Delete the rest of the repository after
`publishToMavenLocal` and every one of them still compiles and runs.

Everything printed in every README here is **output from a real run**. Where a
number could not be produced (anything involving a TPU), the README says so
instead of showing one.

---

## Prerequisites

| | |
|---|---|
| **JDK 25 to build** | every example sets `jvmToolchain(25)`, and it must: Kotlin loads the Tlaloc K2 plugin into the compiler's own JVM and that plugin is Java 25 bytecode |
| **JDK 21 to run** | The library modules are Java 21 bytecode. `quickstart` sets `jvmTarget = JVM_21` and has a `runOnJdk21` task that runs its synthesized gradient on a real JDK 21 (see the repo's `scripts/jdk21-smoke.sh`). PJRT/CUDA *execution* still needs 25 |
| **Publish first** | `./gradlew publishToMavenLocal -x test` at the repo root, once, and again after you change Tlaloc itself |
| **Nothing else** | no GPU, no driver, no Python, no checkpoint — for six of the ten |

```bash
./gradlew publishToMavenLocal -x test  # once
./gradlew -p examples/quickstart run   # and any example, the same way
```

There is no Gradle wrapper inside the example directories; `-p` from the root is
the way in. (`scripts/onboarding-smoke.sh` does both steps for `quickstart`.)

---

## Start here

The three that show what is actually different about Tlaloc. None needs anything
but a JDK.

### [`readable-gradients/`](readable-gradients/) — the derivative is a file you can read

The same gradient three ways — compiled by the plugin, **printed as Kotlin and
recompiled without the plugin**, and a finite difference. The first two are
raw-bit identical; the third agrees to `5.525e-08`. No other autodiff framework
will hand you the derivative as source.

### [`differentiable-physics/`](differentiable-physics/) — gradient descent through a simulator

A ball, air drag, and a hoop 4.6 m away. The `for` loop that integrates Newton's
laws lives *inside* `grad2 { }`, so the compiler differentiates **the simulator**
— 832 lines of derivative it wrote itself — and gradient descent finds the throw.

```
    first guess:   35.52° at 6.400 m/s  —  misses by 2.13 m
    learned throw: 48.91° at 8.266 m/s  —  0.0000 m from the centre of the rim
    SWISH.
```

### [`quickstart/`](quickstart/) — a gradient, and a shape bug that never runs

Three lines for the gradient. Then a real file in a real source set that is
*supposed* to fail, and one command that makes the compiler reject it in front of
you:

```
e: Tlaloc named-index mismatch: contract operands share no named axis:
   lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]
```

---

## Then these

| Example | What it shows | Needs |
|---|---|---|
| [`mnist/`](mnist/) | The real MNIST — 60,000 digits, downloaded and parsed — at **93.66 %** test accuracy, trained by a captured gradient. Act `[4]` prints test digits as ASCII next to the model's verdict. | CUDA *(self-skips to a slower host lane)* · downloads 11 MB once |
| [`gpu-training/`](gpu-training/) | A network learns a disc on the Blackwell: 600 Adam steps in 2.0 s, **98.0 %** held out, the decision boundary drawn next to the ground truth — then the model is **saved to one safetensors file and reloaded**, and everything after that line is computed by the model off the disk. | CUDA *(self-skips)* |
| [`gpu-inference/`](gpu-inference/) | Kotlin compiles a real TinyLlama-1.1B into a directory and **exits**; a stock `python3` with no jax, no torch and no numpy loads it and answers `' Paris.'` | CUDA + a PJRT plugin *(self-skips; falls back to a toy graph with no checkpoint)* |
| [`named-indices/`](named-indices/) | Axis **names** in the Kotlin type, so a transposed weight is an overload-resolution failure in Kotlin's own type checker — no plugin involved. | nothing |

---

## How it works inside — [`internals/`](internals/)

Not tutorials. These three explain Tlaloc's *structure*, and they assume you
already care about it.

| Example | What it shows | Needs |
|---|---|---|
| [`internals/layer3/`](internals/layer3/) | The device decision moved **upstream of the runtime**: recognize attention, coarsen it into one op carrying its own gradient, and emit a genuinely different artifact per target (GB10, H100, TPU v6e, Trainium2, CPU). | nothing |
| [`internals/four-worlds/`](internals/four-worlds/) | Kernel / Orchestration / Program / Cluster as four receiver types, the `BufferHandle` that is the only thing allowed across a step boundary, and the Maestro descriptor a cluster ingests — plus one boundary claim that writing the failing case down proved false. | nothing |
| [`internals/tpu/`](internals/tpu/) | Five acts written **before the hardware exists**: tolerances and verdicts fixed in advance, so the first TPU session is spent debugging a TPU rather than writing a test for one. | a TPU *(never run on one; `--target cuda` is the dry run)* |

`layer3` and `four-worlds` are also the folder's **opt-in** examples. The
surfaces they drive carry `@ExperimentalTlalocApi`, a `@RequiresOptIn(ERROR)` marker Tlaloc
puts on the part of its API that is genuinely provisional, so each starts with
`@file:OptIn(ExperimentalTlalocApi::class)` and a comment saying why. That is
deliberate: they are the only *consumers* of a marked surface in the repository,
which makes them the check that the marker actually refuses somebody. Nothing in
`quickstart`, `readable-gradients`, `differentiable-physics`, `named-indices`,
`mnist`, `gpu-training` or `gpu-inference` needs an opt-in.

---

## A reading order

1. **[`readable-gradients/`](readable-gradients/)** — what makes Tlaloc different
   from every other autodiff: the derivative is *source*, and you can read it.
2. **[`differentiable-physics/`](differentiable-physics/)** — that same derivative,
   now taken through a loop, solving a problem that has nothing to do with ML.
3. **[`quickstart/`](quickstart/)** — the smallest complete program, and the
   compile error that is the other half of the pitch.
4. **[`mnist/`](mnist/)** — the benchmark you already know, so you can judge the
   result rather than take it on trust.
5. **[`gpu-inference/`](gpu-inference/)** — what you ship: a directory, and a
   process with nothing installed in it.
6. **[`internals/`](internals/)** — the structural ideas underneath all of it.

---

## Last full run

Run end to end on 2026-09-22 on the GB10 DGX Spark, aarch64, CUDA driver
580.126.09, JDK 25.0.3+9, **no TPU**. Every one exited `0`.

| Example | Outcome | What it printed |
|---|---|---|
| `quickstart` | ran | `d/dA sum(A matmul A)` at `[[1,2],[3,4]]` = `[7.0, 11.0, 9.0, 13.0]`, then the shape-error source and the command that rejects it |
| `quickstart shapeError` | **failed, as designed** | `e: Tlaloc named-index mismatch: contract operands share no named axis: lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]` |
| `readable-gradients` | ran | compiled `grad { }` vs the printed source: **raw-bit identical** at all 7 points; vs central difference: `max \|gap\| = 5.525e-08` |
| `differentiable-physics` | ran | gradient vs finite differences: worst relative gap `8.401e-04`; descent `2.13 m` miss → `0.0000 m`; the derivative it printed: **832 lines, 823 operations** |
| `named-indices` | ran | rank-2 contraction `[2,4]` and the rank-4 attention core `[1,2,4,4]`, then the program that does not compile |
| `named-indices shapeError` | **failed, as designed** | `Argument type mismatch: actual type is 'DTensor<Rank2<Named<Vocab, Sym>, …>>' but 'DTensor<Rank2<Named<SeqLen, Sym>, …>>' was expected` |
| `mnist` | ran **on the GPU** | 600 full-batch steps over 4,096 images in 8.28 s, loss `0.112445 → 0.002118`, **93.66 %** on all 10,000 test images, 6 of 6 shown digits correct |
| `mnist` (host lane) | ran | `TLALOC_EXAMPLE_LANE=host`: 40 steps over 512 images in 19.85 s, **82.50 %** on 1,000 |
| `gpu-training` | ran **on the GPU** | 600 Adam steps on PJRT/XLA CUDA, held-out accuracy **98.0 %** on 1024 unseen points; the save/reload round trip reproduced 337/337 parameter scalars and all 1024 predictions bit-for-bit |
| `gpu-inference` half one | ran | the real TinyLlama-1.1B: 4.1 GiB of weights and a 65 KiB manifest written in 10.5 s |
| `gpu-inference` half two | ran **on the GPU** | `/usr/bin/python3`, no jax/torch/numpy: `' The capital of France is'` → `' Paris.\n\n2.'`, 1 XLA compile, median step 1353 ms |
| `gpu-inference --reference` | ran **on the GPU** | the toy graph, no checkpoint needed: 2 XLA compiles, median step 2 ms |
| `internals/four-worlds` | ran | one step, the two-step workflow, a Maestro descriptor, and the handle-type violation it can no longer claim more of than is true |
| `internals/four-worlds shapeError` | **failed, as designed** | `Argument type mismatch: actual type is 'BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0>', but 'BufferHandle<DTensor<Rank2<Sym, Sym>, F32>, Mesh0>' was expected` |
| `internals/layer3` | ran | `flash_attn_v3` for GB10/H100, `tpu_pallas_flash_attention` for TPU v6e, `nki_flash_attention` for Trainium2, no custom call for generic CPU |
| `internals/tpu` | device half **self-skipped** | host half computed its whole reference, then named the four paths it searched for a TPU plugin, exit `0` |

Two examples are deliberately not bit-reproducible and say so in their own
READMEs: the GPU lanes of `mnist` and `gpu-training` (XLA autotunes its GEMMs, so
the loss drifts in the last decimals), and wall-clock timings everywhere.

---

## If you add one

The standard this folder holds itself to:

- **Standalone.** Its own two Gradle files, `mavenLocal`, `io.github.pedronahum:*` by
  coordinate. Never a project dependency, never an import from a sibling
  example.
- **One idea.** If you need two paragraphs to say what it shows, it is two
  examples.
- **Lead with the result.** The first thing in the README is what the program
  printed, not what the author wanted to demonstrate.
- **It runs, and the README says what it printed.** Paste the real output.
  Never write output you did not observe.
- **It degrades by name.** No accelerator must mean a named skip and exit `0`,
  never a stack trace — a laptop reader should get through the whole folder.
- **Nothing is homework.** If the point of the example is a program that fails
  to compile, ship that program and a command that compiles it. "Uncomment this
  block" is not a demonstration.
- **It teaches.** These are the first Kotlin a newcomer reads. Comment
  accordingly.
