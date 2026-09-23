# gpu-training — train a network on the GPU with gradients the compiler wrote

**What it shows.** A complete training run, on the GPU, where nobody ever
wrote a derivative:

```kotlin
val step = capture(model, listOf(x)) { prediction ->
    val target = prediction.constant<Shape>(train.ys, intArrayOf(train.n, 1))
    val residual = prediction - target
    (residual * residual).mean()          // mean squared error
}
```

`capture` traces the model's forward into an IR function and then applies
`DxirReverseTransform` — the same reverse-mode AD pass the `grad { }` compiler
plugin uses — to derive the gradient function from it. There is no tape at
runtime, no `.backward()`, and no hand-written adjoint in this example or in
`:nn`. Every derivative comes out of the compiler's rule registry.

Four things follow from that, and the program prints evidence for each:

1. **The gradient is a program, so any backend can run it.** The captured
   gradient function is emitted to StableHLO, compiled by XLA and executed on
   the GPU through PJRT. The example cross-checks one step against the JVM
   interpreter first — the two backends share *nothing* but the graph, so
   agreement is the certification that the emitted StableHLO really is the
   program `:nn` captured.
2. **One compiled executable serves the whole run.** The weights enter the
   graph as *parameters*, not constants, so all 600 steps re-bind new values
   into a single compiled artifact. `session.cacheSize` is printed as the
   receipt.
3. **The run is reproducible from one seed.** Data, held-out set and initial
   weights are all drawn from Tlaloc's threefry counter-based PRNG (bit-exact
   against JAX). A `RandomKey` is a value, not a mutable generator.
4. **The trained model is a file.** The run saves the parameters *and* Adam's
   moments to one safetensors file, builds the model structure again from
   nothing, restores into it, and prints that the reloaded model's predictions
   are bit-identical. Everything after the `reloaded :` line — the decision
   boundary, the held-out accuracy — is computed by the model that came back off
   the disk.

The task has a ground truth you can *look at*: the model must learn a disc in
the plane, and the program ends by printing what it learned next to the real
thing.

## Running it

Standalone Gradle project; it resolves Tlaloc from **mavenLocal** exactly as
your own project would, so publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/gpu-training run
```

**No GPU?** It still runs. With no PJRT plugin or no CUDA device the program
prints why and falls back to the JVM interpreter — same program, same numbers,
about 2.7× slower here. To take that path deliberately on a GPU machine:

```bash
TLALOC_EXAMPLE_LANE=host ./gradlew -p examples/gpu-training run
```

## Actual output

Verbatim, from this machine — GB10 (Grace Blackwell, DGX Spark, aarch64),
driver 580.126.09, JDK 25, Kotlin 2.3.20 — with two edits, both stated so the
block can still be trusted: XLA's own startup logging is trimmed to the two
lines that name the device the work landed on, and the absolute path to this
clone is written `<repo>` on the `saved :` line.

```
=== Tlaloc — training on the GPU with gradients the compiler wrote ===

task     : label = +1 inside the disc centred (0.15, -0.10) of radius 0.70, -1 outside
data     : 512 training points in [-1,1]^2, drawn from threefry seed 20260921
           (data AND initial weights are pure functions of that seed — no global RNG)
model    : Dense(2->16) -> ReLU -> Dense(16->16) -> ReLU -> Dense(16->1)
           6 parameter tensors, 337 scalars, keys [0.w, 0.b, 2.w, 2.b, 4.w, 4.b]
loss     : mean squared error against the +/-1 labels

capture once (no tape, no .backward(), no hand-written derivatives):
  forward : 15 IR nodes, 7 parameters, 1 result(s)
  gradient: 38 IR nodes, 7 parameters, 8 result(s)
           the gradient graph was DERIVED from the forward graph by
           DxirReverseTransform — the compiler's own reverse-mode AD pass.

I0922 20:58:39.863502 1830646 service.cc:194]   StreamExecutor [0]: NVIDIA GB10, Compute Capability 12.1a (Driver: 13.0.0[580.126.9]; Runtime: 12.9.0; Toolkit: 12.9.0; DNN: 9.21.1)
I0922 20:58:39.877859 1830646 cuda_dnn.cc:461] Loaded cuDNN version 92101
device   : GPU via PJRT/XLA (CUDA)
check    : GPU vs host interpreter at step 0, over 8 outputs —
           max|diff| 0.000335 against max|value| 0.992  (3.4e-04 relative)
           If that looks big for f32, it is: XLA runs f32 matmuls on NVIDIA
           tensor cores at TF32's 10 mantissa bits by default. Re-run with
             NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
           — both, since either alone changes nothing — and the relative
           difference drops to ~1e-7. A precision knob, not a correctness one.

training : Adam(lr=0.02), 600 full-batch steps
           step   0   loss 0.992417
           step  50   loss 0.198819
           step 100   loss 0.134747
           step 150   loss 0.099857
           step 200   loss 0.080988
           step 250   loss 0.074563
           step 300   loss 0.066082
           step 350   loss 0.060896
           step 400   loss 0.059658
           step 450   loss 0.056312
           step 500   loss 0.051678
           step 550   loss 0.065872
           step 600   loss 0.047137   (final)

           1.899 s wall clock on GPU via PJRT/XLA (CUDA) (3.16 ms/step)
           loss 0.992417 -> 0.047137
           compiled executables after training: 1
           (one program, 600 dispatches — weights are graph PARAMETERS, not constants)

saved    : <repo>/examples/gpu-training/build/checkpoints/disc_mlp.safetensors
           5452 bytes — one safetensors file: 6 parameter tensors
           plus Adam's two moment tensors per parameter, and the step count.
           Nothing Tlaloc-specific is needed to read it: Python's `safetensors`
           opens it, and the keys are param.0.w, opt.m.0.w, opt.v.0.w, ...
reloaded : built the same structure from scratch, then restored into it —
           parameters bit-identical: 337 / 337 scalars
           held-out predictions vs the in-memory model: BIT-IDENTICAL over 1024 points
           Adam resumed at step 600 of 600 — training could continue
           (everything below was computed by the model that came back off the disk)

learned decision boundary vs ground truth  ('#' inside, '.' outside)

            ground truth                 what the model learned     
   ...............................   ...............................
   ...............................   ...............................
   ...............................   ...............................
   ...............######..........   ................##.............
   ...........##############......   ..........##############.......
   .........##################....   .........##################....
   ........####################...   ........####################...
   .......#####################...   ........####################...
   .......#####################...   ........####################...
   ........####################...   ........####################...
   ........###################....   ........###################....
   ..........################.....   ..........###############......
   .............##########........   .............##########........
   ...............................   ...............................
   ...............................   ...............................

           grid cells where the model disagrees with the truth: 9 / 465 (1.9%)
           held-out accuracy on 1024 fresh points never seen in training: 98.0%
           compiled executables after inference: 3 (training + two inference shapes)

note     : run this twice and the losses will differ in the 3rd decimal. XLA
           autotunes its GEMM kernels at compile time, so the GPU lane is not
           bit-reproducible; TLALOC_EXAMPLE_LANE=host is, exactly, every run.

done.
```

### The same run on the CPU lane

`TLALOC_EXAMPLE_LANE=host`, same machine, same seed:

```
device   : host JVM (DxirInterpreter)
           step   0   loss 0.992422
           step 600   loss 0.045672   (final)
           5.190 s wall clock on host JVM (DxirInterpreter) (8.65 ms/step)
           held-out predictions vs the in-memory model: BIT-IDENTICAL over 1024 points
           Adam resumed at step 600 of 600 — training could continue
           grid cells where the model disagrees with the truth: 8 / 465 (1.7%)
           held-out accuracy on 1024 fresh points never seen in training: 97.9%
```

Two observations about those numbers, both measured:

- **The GPU is only ~2.7× faster here**, because the model is tiny (337
  scalars) and every step round-trips through the host to run the optimizer.
  This example is about the *route*, not about throughput; the throughput story
  is the LlamaDecoder benchmark matrix in the root README.
- **The host lane is bit-reproducible and the GPU lane is not guaranteed to
  be.** Two host runs produced identical output apart from the wall-clock line
  (5.190 s vs 5.310 s — every other printed number matched). On the GPU the
  picture is more interesting than "nondeterministic": two earlier runs
  differed in the third decimal at step 600 (0.047137 vs 0.051002), while two
  later runs reproduced 0.046667 *exactly*. XLA autotunes its GEMM
  kernels at compile time and can pick a different winner per run — so the
  nondeterminism is real, and it does not have to show. If you need a bit-exact
  GPU trajectory, that is an XLA flag question, not a Tlaloc one.
- **The reloaded model's predictions are bit-identical on BOTH lanes**, which is
  a weaker statement than it looks and worth saying precisely: the reloaded
  model and the in-memory model are compared through the *same* compiled
  executable with the *same* inputs, so the only thing under test is whether
  the checkpoint round trip preserved every bit of every parameter. It does —
  337 of 337 scalars, and `max|diff| = 0` over 1024 held-out points.

### About the TF32 line

The GPU-vs-interpreter cross-check disagrees at 3.4e-04 relative, which is far
too large for f32 rounding. That was measured down, not guessed: with **both**
`NVIDIA_TF32_OVERRIDE=0` **and**
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` the same check reads
`max|diff| 5.96e-08` (6.0e-08 relative). Either flag *alone* leaves it at
3.4e-04 — XLA reaches TF32 tensor cores through two independent paths (its
Triton GEMM emitter, and cuBLAS), and closing one just moves the work to the
other.

## The checkpoint

The run ends by saving the trained model to
`build/checkpoints/disc_mlp.safetensors` and reading it back. Two things about
that file are worth knowing:

**It is an ordinary safetensors file.** Not a Tlaloc container, not Java
serialization — the format HuggingFace ships checkpoints in, which Tlaloc
reads and writes. So you can
open the file with tools you already have:

```python
from safetensors import safe_open
with safe_open("build/checkpoints/disc_mlp.safetensors", framework="pt") as f:
    print(f.metadata())
    print(sorted(f.keys()))
    print(f.get_tensor("param.0.w").shape)
```

which on the file this example just wrote prints, verbatim (`safetensors` 0.8.0
+ torch 2.11.0, wrapped here only because the lines are long):

```
{'tlaloc.optimizer.kind': 'Adam', 'tlaloc.optimizer.stepCount': '600',
 'seed': '20260921', 'steps': '600', 'task': 'disc',
 'tlaloc.checkpoint.version': '1'}
['opt.m.0.b', 'opt.m.0.w', 'opt.m.2.b', 'opt.m.2.w', 'opt.m.4.b', 'opt.m.4.w',
 'opt.v.0.b', 'opt.v.0.w', 'opt.v.2.b', 'opt.v.2.w', 'opt.v.4.b', 'opt.v.4.w',
 'param.0.b', 'param.0.w', 'param.2.b', 'param.2.w', 'param.4.b', 'param.4.w']
torch.Size([2, 16])
```

**It is resumable, not just a snapshot.** The `opt.m.*` / `opt.v.*` tensors are
Adam's first and second moments and `tlaloc.optimizer.stepCount` is its `t`, so
`snapshot.restoreOptimizerState(optimizer)` hands back an optimizer state that
continues the run instead of restarting its bias correction. A checkpoint
written without them (`ModelCheckpoint.encode(model)`) is legal and is refused
*by name* if you later ask it for optimizer state.

**What the file does NOT carry is the model's STRUCTURE.** That is why `Main.kt`
builds `freshModel()` a second time and restores *into* it: a checkpoint holds
parameter values, and restoring into a different shape is refused by name with
the keys that disagree. It is PyTorch's `state_dict` contract, for the same
reason — a file that reconstructed classes would be a file that executes.

## What to read in the source

| Look at | For |
|---|---|
| [`Main.kt`](src/main/kotlin/Main.kt) | `capture` once, then the six-line training loop: bind → dispatch → unpack → optimizer step |
| [`Lanes.kt`](src/main/kotlin/Lanes.kt) | The entire "run it on the GPU" story: `PjrtSession`, plus the calling convention of a captured gradient function (`inputs ++ params` in, `loss ++ grads` out) |
| [`Task.kt`](src/main/kotlin/Task.kt) | threefry `RandomKey` as a *value*: `split` for independent streams, no global generator |
| [`Main.kt`'s `saved :` / `reloaded :` section](src/main/kotlin/Main.kt) | `saveCheckpoint` / `loadCheckpoint` / `restore`: the model is a value, so loading builds a NEW one |
| [`build.gradle.kts`](build.gradle.kts) | The dependency set, and why this example needs **no** compiler plugin on the classpath |

### A note on plumbing

`:nn` deliberately ships no `CapturedStep.runOn(session)` convenience — adding
one would put an `:nn` → `:runtime-pjrt` edge in the module graph and undo the
boundary that makes execution a caller's choice. The ~15 lines of `bind` /
`unpack` in [`Lanes.kt`](src/main/kotlin/Lanes.kt) are what you write instead,
and they are the supported recipe (recorded in
[`docs/MODEL_LAYER_PLAN.md`](../../docs/MODEL_LAYER_PLAN.md)'s F8 entry, and
certified there by `NnMlpGpuTrainingTest`). The example writes them in the open
rather than hiding them behind a helper.

The optimizer runs on the host. That is the ratified v1 scope: gradients on the
device, optimizer state on the JVM. GPU-resident optimizer state is a named
deferral in the same document, not an oversight.

## Related

- [`examples/quickstart`](../quickstart) — the same AD engine through its other
  front door: a `grad { }` lambda lowered by the K2 compiler plugin.
- [`docs/MODEL_LAYER_PLAN.md`](../../docs/MODEL_LAYER_PLAN.md) — the `:nn`
  design, the capture contract, and the certifications behind this lane.
- [`docs/READABLE_REVERSE.md`](../../docs/READABLE_REVERSE.md) — printing the
  derived gradient back as readable Kotlin (`CapturedStep.gradSource()`).
