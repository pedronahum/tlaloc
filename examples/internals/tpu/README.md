# Tlaloc on a TPU

**Read this first: no part of Tlaloc has ever run on a TPU.** Not this example,
not the test suite, not one op. What exists is the whole host side of the path —
the `PjrtTarget.Tpu` client, libtpu plugin resolution, the platform gate on
create-options, the compile-options encoding, and a TPU smoke suite that skips
cleanly because there is no TPU under it. Every certified number in this
repository came off an NVIDIA GB10 or the JVM.

This example is written for the day that changes. It is a complete program, and
it is meant to be the *first* thing run on a new TPU VM, so that the first
session is spent debugging a TPU rather than writing a test for one. It also
means the claims are written down **before** the hardware can flatter them: the
tolerances, the bit-exactness check, and the pass/fail verdicts are all fixed
here, in advance, where you can argue with them.

If you run it on a TPU before anyone else does, you are the first. Please record
what happened in `docs/TPU_BRINGUP.md` — including, especially, a failure.

---

## What it shows

A **scaled-dot-product attention block** — the computation TPUs exist for —
taken end to end through Tlaloc:

```
scores = Q · Kᵀ           [seq, seq]
probs  = softmax(scores / √dim)
out    = probs · V        [seq, dim]
```

The program runs in two halves.

**The host half runs anywhere** (laptop, GPU box, TPU VM). It builds the block
as a Tlaloc IR function, differentiates it with `DxirReverseTransform` — the
same reverse-mode pass the `grad { }` compiler plugin uses, so the softmax
adjoint is written by the compiler and not by this example — emits both graphs
as StableHLO text into `build/`, and evaluates them on the JVM interpreter.
**That interpreter result is computed before any device is touched**, which is
the point: the device is judged against a reference it did not produce.

**The device half needs a TPU**, and is five acts:

| Act | Claim | How it is judged |
| --- | --- | --- |
| 1 | the plugin that resolved really is a TPU | `PJRT_Client_PlatformName == "tpu"`, asserted |
| 2 | the attention block computes correctly | max abs diff vs the host interpreter, relative tolerance |
| 3 | the **compiler-derived gradient** is correct | gradients elementwise; the loss with a reduction-aware bound |
| 4 | **the threefry RNG stream is BIT-IDENTICAL to the host** | raw f32 bit patterns, 4113 lanes, zero tolerance |
| 5 | f32→bf16 narrowing matches host round-half-to-even | bf16 bit patterns, 19 probe values |

**Act 4 is the flagship.** Tlaloc's `RNG_UNIFORM` does not call a device RNG:
it lowers to explicit StableHLO integer arithmetic — threefry's
ARX rounds, a shift, a bit-or against the exponent of 1.0, and one exactly
representable subtraction. Integer arithmetic cannot legally differ between
backends, so the stream *cannot* fork by construction. That is an argument.
CUDA has already turned it into a measurement. A TPU pass would make the same
draw, from the same key, provably identical on a third backend — and a TPU
*failure* would be a genuine discovery, which is why this act compares bits and
allows no tolerance at all.

## How to run it

From the repository root:

```bash
./gradlew publishToMavenLocal            # publish io.github.pedronahum:* into ~/.m2
./gradlew -p examples/internals/tpu run            # the TPU lane (default)
```

On a machine with no TPU that prints the host half and a named skip, and exits
0. Useful flags:

```bash
./gradlew -p examples/internals/tpu run --args="--target cuda"          # dry run on a GPU
./gradlew -p examples/internals/tpu run --args="--seq 2048 --dim 512"   # v5e-sized work
```

## What it actually prints — without a TPU

Verbatim from the GB10 (aarch64 DGX Spark, CUDA driver 580.126.09), which has
no TPU. **Exit code 0.**

```
=== Tlaloc on a TPU — the program, the cross-check, and the bit stream ===

HONESTY NOTICE: no part of Tlaloc has ever executed on a TPU. The host half
below runs anywhere; the device half has never run at all. If you are reading
its output for the first time, you are the first. See docs/TPU_BRINGUP.md.

--- host half: the program, and the reference it will be judged against ----

workload : scaled-dot-product attention, seq=256 dim=128
           Q[256,128] · Kt[128,256] -> scale -> softmax -> ·V[256,128]
           5 IR ops, 3 parameters
           16.8 M multiply-accumulates per forward pass
gradient : d(Σ attention)/d(Q, Kt, V), derived by DxirReverseTransform
           6 ops in, 22 ops out, 4 results (loss + 3 gradients)
           the softmax adjoint in there was written by the compiler, not by us
emitted  : build/attention.stablehlo.mlir (16 lines)
           build/attention_grad.stablehlo.mlir (36 lines)
           this is exactly what the TPU compiler is handed — nothing is
           generated on the device side of the boundary.

  first lines of the forward module:
    func.func @attention_f32(%0: tensor<256x128xf32>, %1: tensor<128x256xf32>, %2: tensor<256x128xf32>) -> tensor<256x128xf32> {
      %3 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<256x128xf32>, tensor<128x256xf32>) -> tensor<256x256xf32>
      %4 = stablehlo.constant dense<0.088388346> : tensor<256x256xf32>
      %5 = stablehlo.multiply %3, %4 : tensor<256x256xf32>
      %s8 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %s9 = stablehlo.reduce(%5 init: %s8) applies stablehlo.maximum across dimensions = [1] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    ...

reference: host interpreter evaluated both graphs in 735 ms
           forward  out[0..3] = [-0.01179253, 0.0346781, -0.0051385523, 0.0117492555, ...]
           loss             = -2.553444
           dQ[0..3]         = [1.3895059E-4, -0.003888416, -0.007824016, -2.9521392E-4, ...]
           inputs are threefry draws from fixed keys — identical on every
           machine that runs this example, which is what makes the
           comparison below meaningful across two continents.

rng      : uniform(key=RandomKey(2a, 7), n=6) = 0.71358633 (0x3f36ad98), 0.9094175 (0x3f68cf96), 0.16808438 (0x3e2c1e50), ...
           (hex is the raw f32 bit pattern; Act 4 compares THESE, not values)

--- device half: SKIPPED -------------------------------------------------

no TPU PJRT plugin resolved — this machine has no TPU, so the device half
of the example cannot run. Missing: libtpu.so (the TPU PJRT plugin).

Where it was looked for:
  - $TLALOC_PJRT_PLUGIN_PATH: not set
  - $VIRTUAL_ENV/lib/python3.N/site-packages/libtpu/libtpu.so  (pip install libtpu)
  - ~/.local/lib/python3.N/site-packages/libtpu/libtpu.so       (pip install --user)
  - /lib/libtpu.so, /usr/lib/libtpu.so                          (Cloud TPU VM images)

How to get one: provision a Cloud TPU VM and follow docs/TPU_BRINGUP.md, or read this example's README.md, which is that runbook narrowed to this program.

Nothing above this line needed an accelerator, and nothing above this line
is a device claim. Exiting 0 — a machine without one is not a failure.
```

Those threefry hex values are the cross-machine anchor. If your TPU VM's host
half prints anything other than `0x3f36ad98, 0x3f68cf96, 0x3e2c1e50`, stop:
something is wrong before the TPU is even involved.

## What it actually prints — the dry run, on a GPU

`--target cuda` runs the same five acts against an NVIDIA GPU. It proves the
*program* and proves **nothing whatsoever about a TPU**; it exists so the first
TPU session debugs the TPU. Verbatim from the GB10 (the host half above is
identical and elided here). **Exit code 0.**

```
--- device half: DRY RUN on cuda (no TPU claim) ------------------------

plugin   : ~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
target   : Cuda

[XLA's plugin writes several I-lines to stderr here; the device one reads
 "StreamExecutor [0]: NVIDIA GB10, Compute Capability 12.1a
  (Driver: 13.0.0[580.126.9]; Runtime: 12.9.0; Toolkit: 12.9.0; DNN: 9.21.1)"]

[act 1] platform identity
        PJRT_Client_PlatformName = 'cuda'  (plugin: xla_cuda_plugin.so)
        this is a cuda device.

[act 2] attention forward, cuda vs host interpreter
        compile + stage + execute: 1043 ms (includes XLA compilation)
        out[0..3] = [-0.011801289, 0.03466948, -0.005130902, 0.011750093, ...]
        max|cuda - host| = 2.5886111E-5   (largest reference magnitude 0.058010105, tolerance 0.001)
        the device agrees with the interpreter.

[act 3] compiler-derived gradient, cuda vs host interpreter
        loss (cuda) = -2.525476   loss (host) = -2.553444
        |Δloss| = 0.02796793  (tolerance 0.082966544: a 32768-term f32 sum of total mass 480.6 — summation order, not error)
        dQ[0..3]   = [1.3839506E-4, -0.0038876585, -0.007825465, -2.926346E-4, ...]
        max|cuda - host| over dQ+dKt+dV = 4.3034554E-5 (tolerance 0.00101231)
        the derivative nobody wrote is correct on a cuda device.

[act 4] threefry uniform stream — BIT-EXACT vs the host kernel (the flagship)
        dims=[6] key=RandomKey(2a, 7)  n=6  identical
        dims=[5] key=RandomKey(ffffffff, 3039)  n=5  identical
        dims=[2, 3] key=RandomKey(13198a2e, 3707344)  n=6  identical
        dims=[4096] key=RandomKey(243f6a88, 85a308d3)  n=4096  identical
        4113/4113 lanes bit-identical on cuda. The stream is the same draw,
        from the same key, on every backend that has been asked so far.

[act 5] bf16: device narrowing vs host round-half-to-even, and tiled layout
        19 probe values (ties both ways, ±0, ±inf, overflow, subnormal flush):
        all narrow identically to the host helper.
        on-device size of a bf16 [2,2] buffer on cuda: 8 bytes (8 = dense, more = tiled padding)

--- verdict ---------------------------------------------------------------
  PASS  platform identity: platform_name='cuda'
  PASS  attention forward: max|diff|=2.5886111E-5 vs tol=0.001
  PASS  derived gradient: max|d(grad)|=4.3034554E-5 vs tol=0.00101231, |Δloss|=0.02796793 vs tol=0.082966544
  PASS  threefry bit-exactness: 4113 lanes, 0 forks
  PASS  bf16 narrowing: 0/19 lanes disagree

All 5 acts passed on cuda. This proves the program and
says NOTHING about a TPU. The TPU claim is still unmade.
```

**The dry run earned its keep on its first execution.** Act 3 originally judged
all four outputs — the loss and the three gradients — with one relative
tolerance, and the loss failed by 10×. The cause was not a bug: the loss is a
sum of 32768 f32 terms whose total mass is 480.6 but whose sum is only -2.55, so
the device's blocked summation and the interpreter's sequential loop are
entitled to differ by far more than the result's own magnitude. The fix was to
judge the *reduction* with a bound derived from the mass being summed
(`8·ε·√N·Σ|xᵢ|`) and keep the tight tolerance on the gradients, which are what
would actually catch a wrong derivative. **That distinction was found on a GPU
so that nobody has to find it at TPU prices.**

## The Cloud TPU VM session, start to finish

```bash
# 1. Provision. A single-host v5e or v6e is enough; spot is fine for a smoke lane.
gcloud compute tpus tpu-vm create tlaloc-first-contact \
  --zone=us-central1-a --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base --spot
gcloud compute tpus tpu-vm ssh tlaloc-first-contact --zone=us-central1-a

# 2. JDK 25, userspace, no sudo. TPU VM hosts are x86_64 — check `uname -m`.
mkdir -p ~/.local/jdks && cd ~/.local/jdks
curl -LO https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse
tar xf eclipse && rm eclipse            # produces jdk-25.0.N+M/
export JAVA_HOME=~/.local/jdks/jdk-25.0.*

# 3. libtpu — the TPU PJRT plugin.
python3 -m venv ~/venv && ~/venv/bin/pip install libtpu
export TLALOC_PJRT_PLUGIN_PATH=$(ls ~/venv/lib/python3.*/site-packages/libtpu/libtpu.so)
# Some TPU VM images ship it as /lib/libtpu.so or pjrt_c_api_tpu_plugin.so —
# point the variable at whichever exists. The name must contain "tpu": the
# resolver refuses a non-tpu-shaped plugin for the TPU lane on purpose.

# 4. Build Tlaloc and publish it locally.
git clone <tlaloc remote> ~/tlaloc && cd ~/tlaloc
./gradlew publishToMavenLocal

# 5. The moment.
./gradlew -p examples/internals/tpu run
```

**What to expect.** The host half is identical to the output above, including
the threefry hex — check that first. Then act 1 should print
`PJRT_Client_PlatformName = 'tpu'`, and acts 2–5 should print numbers of the
same shape as the CUDA dry run. XLA's first compile of the attention graph will
dominate the wall clock; the ~1.0 s above is a GPU number (it moves by a few ms
between runs) and a TPU's will be its own.

Then run the repository's TPU smoke suite, which is the same claims as
regression tests:

```bash
./gradlew :runtime-pjrt:jvmTest --tests "*PjrtTpu*" --rerun
```

`PjrtTpuSmokeTest`'s 5 tests skip on every machine that has ever run them. On a
TPU they should run.

## What might go wrong on first contact

These are the specific unknowns the audits recorded, not general anxieties. Each
one has been reasoned about on a CUDA box and none has been *measured* on a TPU.

1. **The hand-encoded `CompileOptionsProto`.** Tlaloc hands XLA a 6-byte
   protobuf it encodes by hand (`0x1A 0x04 0x20 0x01 0x28 0x01` —
   `executable_build_options{num_replicas: 1, num_partitions: 1}`), with field
   numbers verified against `xla/pjrt/proto/compile_options.proto`. Protobuf
   wire format is backend-agnostic, so this *should* parse identically in
   libtpu. If compilation fails with a proto/parse error, this is the first
   suspect — and the failure will be loud, not silent.
2. **create-options gating.** A `PjrtTarget.Tpu` client is created with
   `create_options = NULL, num_options = 0`. The GPU allocator knobs
   (`memory_fraction`, `preallocate`) are refused by name for TPU, and their
   absence is refused for CUDA — both directions, in `PjrtSession`'s init block.
   That asymmetry is deliberate: passing GPU options to a TPU is meaningless,
   and creating a *CUDA* client without them once preallocated 75% of this
   machine's unified memory and rebooted it. **Unknown:** which options libtpu
   actually accepts (`ml_framework_name`, `max_inflight_computations`, …). None
   are passed until a live libtpu confirms them.
3. **Memory kinds and donation.** Certified against the CUDA plugin only.
   Tlaloc uses default memory space and does not donate input buffers. If a TPU
   run produces correct-but-slow numbers, or a buffer lifetime error, this is
   where to look.
4. **Tiled layouts and bf16 buffer size.** Act 5 prints the on-device size of a
   bf16 `[2,2]` buffer and *reports* it rather than asserting it. CUDA answers 8
   bytes — dense, 2 bytes per element. A TPU may legitimately pad to a tile, and
   whatever it prints is the answer to an open question, not a failure.
5. **dtype lanes.** `runOn` is all-f32, `runOnBf16` all-bf16, `runOnF64`
   all-f64; mixed-precision programs cast inside the graph. Two CUDA findings
   are listed for TPU re-measurement in `docs/TPU_BRINGUP.md`: XLA-CUDA folds an
   f32→bf16→f32 convert *pair* to identity (so only a single-convert program
   measures the device — which is the shape act 5 uses), and it rounds a bf16
   dot's output at the op boundary. Neither is guaranteed to hold on TPU.
6. **`PJRT_ExecuteOptions` layout.** Ruled header ABI rather than backend ABI:
   one padding decision (4 bytes after the i32 `launch_id`) that holds on LP64
   for x86_64 and aarch64 alike. The TPU-flavoured fields (`num_tasks`,
   `task_ids`, `multi_slice_config`) are zeroed — the correct single-host
   default, and untested in any non-zero form.
7. **Summation order, as above.** Act 3 already separates a reduction's
   tolerance from an elementwise one. A TPU's reduction tree will differ again
   from both the GPU's and the interpreter's; the bound is derived from the
   summed mass, so it should travel.

## What to look at in the source

- **`src/main/kotlin/Workload.kt`** — the graphs. `Attention.attentionOut` is
  four ops and is shared by the forward and the loss functions, so they cannot
  drift apart. Nothing in this file knows what a TPU is; that is the whole
  argument for why a program written today can run on hardware bought later.
- **`src/main/kotlin/Main.kt`** — `hostHalf` (build, differentiate, emit,
  interpret) and the five acts. `missingDeviceReason` is the self-skip: it names
  every path that was searched instead of shrugging.
- **`build/attention.stablehlo.mlir`** and **`build/attention_grad.stablehlo.mlir`**
  after a run — exactly the text handed to the compiler. The gradient module is
  worth reading: it is the derivative of a softmax attention block, and no human
  wrote it.

## If an act fails on a TPU

Record it. A failure here is the most valuable output this example can produce —
it is the first real measurement of something that was, until that moment, only
an argument. In particular a **fork in act 4** would mean the portability
argument for the RNG is wrong, and the program prints the two disagreeing bit
patterns for exactly that reason. `docs/TPU_BRINGUP.md` lists what a first TPU run
must record.

## Related

- `docs/TPU_BRINGUP.md` — the full bring-up runbook this example condenses.
- `docs/TPU_READINESS_AUDIT.md`, section 3: the ranked weaknesses.
- `examples/gpu-training/` — the same AD engine training a model on a real GPU,
  where the numbers are certified rather than anticipated.
