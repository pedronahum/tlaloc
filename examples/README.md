# Tlaloc examples

Eight standalone programs. Each directory here is its **own Gradle build** — its
own `settings.gradle.kts`, its own `build.gradle.kts` — and each one resolves
Tlaloc from **mavenLocal**, as `io.tlaloc:core:0.0.1-SNAPSHOT` and friends,
exactly the way your project would. None of them is a module of the repo build,
none of them uses `includeBuild`, and none of them imports another. Delete the
rest of the repository after `publishToMavenLocal` and every one of them still
compiles and runs.

Everything printed in every README here is **real output from a real run**. Where
a number could not be produced on this machine — anything involving a TPU — the
README says so in those words instead of showing you a number.

---

## Prerequisites

| | |
|---|---|
| **JDK 25** | every example sets `jvmToolchain(25)`. On the DGX Spark: `export JAVA_HOME=~/.local/jdks/jdk-25.0.3+9` |
| **Publish first** | `./gradlew publishToMavenLocal` at the repo root, once, and again after you change Tlaloc itself |
| **Nothing else** | no GPU, no driver, no Python, no checkpoint — for five of the eight |

Run one from the repo root:

```bash
./gradlew publishToMavenLocal          # once
./gradlew -p examples/quickstart run   # and any example, the same way
```

There is no Gradle wrapper inside the example directories; `-p` from the root is
the way in. (`scripts/onboarding-smoke.sh` does both steps for `quickstart`.)

---

## The examples

| Example | What it shows | What it needs |
|---|---|---|
| [`quickstart/`](quickstart/) | The smallest complete program: `grad { }` over a matmul, lowered by the K2 compiler plugin at compile time. Plus a named-axis error you can uncomment. | nothing |
| [`named-indices/`](named-indices/) | Axis **names** in the Kotlin type, so a transposed weight is an overload-resolution failure in Kotlin's own type checker — not a runtime shape error. | nothing |
| [`readable-gradients/`](readable-gradients/) | The derivative as a **file you can read**: the same gradient three ways — compiled, printed as Kotlin by the plugin and compiled again *without* the plugin, and a finite difference. All three agree. | nothing |
| [`four-worlds/`](four-worlds/) | Kernel / Orchestration / Program / Cluster as four receiver types, the `BufferHandle` that is the only thing allowed across a step boundary, and the Maestro descriptor a cluster ingests. | nothing |
| [`layer3/`](layer3/) | The device decision moved **upstream of the runtime**: recognize attention, coarsen it into one op carrying its own gradient, and emit a genuinely different artifact per target (GB10, H100, TPU v6e, Trainium2, generic CPU). | nothing |
| [`gpu-training/`](gpu-training/) | A network learns a disc on the Blackwell. `capture` derives the gradient with the compiler's own reverse pass, Adam runs 600 steps on the GPU, and the GPU answer is checked against the host interpreter. | CUDA GPU *(self-skips to a host lane)* |
| [`gpu-inference/`](gpu-inference/) | Two processes: Kotlin compiles a model into a directory and **exits**; a stock `python3` with no jax, no torch and no numpy in it loads that directory and decodes tokens on the GPU. | half one: nothing · half two: a PJRT plugin `.so` + driver *(self-skips)* · the Llama path: a checkpoint |
| [`tpu/`](tpu/) | Five acts written **before the hardware exists**: platform identity, forward, compiler-derived gradient, threefry bit-exactness, bf16 narrowing — each with its tolerance and verdict fixed in advance. Runs as a dry run on CUDA. | a TPU *(never run on one; self-skips, and `--target cuda` is the dry run)* |

---

## A reading order

1. **[`quickstart/`](quickstart/)** — what a Tlaloc program is: three lines, one
   gradient, no tape.
2. **[`readable-gradients/`](readable-gradients/)** — what makes it different
   from every other autodiff: the derivative is *source*, and you can read it.
3. **[`named-indices/`](named-indices/)** — what the type system buys you before
   anything runs.
4. **[`gpu-training/`](gpu-training/)** — the same machinery, now training on a
   real accelerator, with the GPU checked against the interpreter.
5. **[`gpu-inference/`](gpu-inference/)** — what you ship: a directory, and a
   process with nothing installed in it.
6. **[`layer3/`](layer3/)** and **[`four-worlds/`](four-worlds/)** — the two
   structural ideas underneath all of the above: the artifact carries the device
   decision, and the four worlds keep the scopes apart.
7. **[`tpu/`](tpu/)** — the frontier. Read it as a claim nobody has cashed yet.

---

## What happened on this machine

The full set was run end to end on 2026-09-21 (§0.4.490) on the GB10 DGX Spark,
aarch64, CUDA driver 580.126.09, JDK 25.0.3+9, **no TPU**. Every one exited `0`.

| Example | Outcome | What it printed |
|---|---|---|
| `quickstart` | ran | `d/dA sum(A matmul A)` at `[[1,2],[3,4]]` = `[7.0, 11.0, 9.0, 13.0]` |
| `named-indices` | ran | rank-2 contraction `[2,4]` and the rank-4 attention core `[1,2,4,4]`, both matching their expected dims |
| `readable-gradients` | ran | compiled `grad { }` vs the printed source: **raw-bit identical** at all 7 points; vs central difference: `max \|gap\| = 5.525e-08` |
| `four-worlds` | ran | one step (body hash `8d772d28…`, 455 bytes of StableHLO) → `35.0`; the two-step workflow → the same `35.0`; a 2167-char Maestro descriptor |
| `layer3` | ran | `flash_attn_v3` for GB10/H100, `tpu_pallas_flash_attention` for TPU v6e, `nki_flash_attention` for Trainium2, no custom call at all for generic CPU |
| `gpu-training` | ran **on the GPU** | 600 Adam steps on PJRT/XLA CUDA in 1.814 s (3.02 ms/step), loss `0.992417 → 0.047460`, held-out accuracy **98.1 %** on 1024 unseen points |
| `gpu-training` (host lane) | ran | `TLALOC_EXAMPLE_LANE=host`: the GPU lane self-skips by name, 600 steps in 5.314 s on the JVM interpreter, held-out **97.9 %** |
| `gpu-inference` half one | ran | artifact written in 0.1 s: 6 content-addressed bodies (56.3 KiB), 6 programs, an 8.6 KiB manifest |
| `gpu-inference` half two | ran **on the GPU** | `/usr/bin/python3` (3.12.3), no jax/torch/numpy: prompt `[1, 2]` → generated `[9, 7]`, 2 XLA compiles, median step 2 ms |
| `gpu-inference` half two (no plugin) | self-skipped | `SKIP: no PJRT plugin on this machine…`, exit `0` |
| `tpu` | device half **self-skipped** | host half computed its whole reference (loss `-2.553444`), then: `no TPU PJRT plugin resolved`, with the four search paths named, exit `0` |
| `tpu --target cuda` | dry run **on the GPU** | 5 / 5 acts PASS — forward `max\|diff\| 2.6e-05`, gradient `4.3e-05`, threefry **4113/4113 lanes bit-identical**, bf16 `0/19` disagreements. Proves the program; says nothing about a TPU. |

Two examples are deliberately not bit-reproducible and say so in their own
READMEs: `gpu-training`'s GPU lane (XLA autotunes its GEMMs, so the loss drifts
in the third decimal — the run above came in at `0.047460` against the
`0.051002` its README recorded in §0.4.486, and its host lane is exact every
time: `5.314 s` and 97.9 % here, `5.301 s` and 97.9 % there), and wall-clock
timings everywhere.

---

## If you add one

The standard this folder holds itself to:

- **Standalone.** Its own two Gradle files, `mavenLocal`, `io.tlaloc:*` by
  coordinate. Never a project dependency, never an import from a sibling
  example.
- **One idea.** If you need two paragraphs to say what it shows, it is two
  examples.
- **It runs, and the README says what it printed.** Paste the real output.
  Never write output you did not observe.
- **It degrades honestly.** No accelerator must mean a named skip and exit `0`,
  never a stack trace — a laptop reader should get through the whole folder.
- **It teaches.** These are the first Kotlin a newcomer reads. Comment
  accordingly.
