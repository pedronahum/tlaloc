# TPU bring-up runbook

> [!IMPORTANT]
> No Tlaloc program has run on a TPU yet. What is done is the local half:
> `PjrtTarget.Tpu`, plugin resolution, the platform gate on create options, and a TPU
> smoke suite that skips itself without a TPU (on a GPU machine every TPU test skips).
> The programs the session runs do compile with XLA:TPU for v6e
> ([Compiling for TPU without a TPU](#compiling-for-tpu-without-a-tpu)).
> This runbook is the procedure for a Cloud TPU VM, where those skips should become
> passes. The work that follows it (multi-chip, multi-host) is listed in
> [TPU_READINESS_AUDIT.md](audits/TPU_READINESS_AUDIT.md).

## What the local half established

1. **Plugin resolution** (`PjrtBinaries.tpuPluginPath`):
   `TLALOC_PJRT_PLUGIN_PATH` when it names a tpu-shaped `.so` (name
   contains "tpu" — the gate that keeps a CUDA host's generic env var out
   of the TPU lane), else the libtpu default locations:
   - `$VIRTUAL_ENV/lib/python3.N/site-packages/libtpu/libtpu.so` — the
     PyPI `libtpu` wheel ships exactly one shared object, `libtpu.so`,
     inside the `libtpu` package directory;
   - `~/.local/lib/python3.N/site-packages/libtpu/libtpu.so`
     (`pip install --user`);
   - `/lib/libtpu.so`, `/usr/lib/libtpu.so` — TPU VM system images
     (older images distribute the plugin as `pjrt_c_api_tpu_plugin.so`;
     point the env var at it there).
2. **Create-options are platform-gated.** `memory_fraction` /
   `preallocate` are XLA **GPU** allocator options; a
   `PjrtTarget.Tpu` session creates its client with `create_options =
   NULL, num_options = 0` and refuses GPU options by name (both
   cross-wirings are `require`d in `PjrtSession`). A TPU client passes
   one option of its own, `premapped_buffer_size` (int64, 0 unless
   `TPU_PREMAPPED_BUFFER_SIZE` is set; `TpuClientOptions`), as TorchTPU
   does, so libtpu does not reserve its default 4 GiB of pinned host
   memory. Its name and type are TorchTPU's (`pjrt_client.cc`); unverified
   on a TPU here. Other candidates (`ml_framework_name`,
   `max_inflight_computations`) are not passed.
3. **The hand-encoded `CompileOptionsProto` parses on any backend.**
   Field numbers re-verified against
   `xla/pjrt/proto/compile_options.proto` (openxla/xla main, 2026-09-21):
   `executable_build_options = 3`, `num_replicas = 4`,
   `num_partitions = 5`; the 6-byte encoding
   (`0x1A 0x04 0x20 0x01 0x28 0x01`) is protobuf wire format, which is
   backend-agnostic — the TPU compile path deserialises the same message.
   Pinned by `PjrtTpuLocalCertTest.compileOptionsProtoDecodesToSingleReplicaSinglePartition`.
4. **`PJRT_ExecuteOptions` is header ABI, not backend ABI.** The one
   padding decision (4 bytes after the i32 `launch_id`) holds on LP64 for
   aarch64 and x86_64 alike. The TPU-flavoured fields (`launch_id`,
   `num_tasks`/`task_ids`/`incarnation_ids`, `multi_slice_config`) are
   zeroed — the correct single-host single-task default; non-zero forms
   are needed only for multi-chip and multi-host runs.

## Compiling for TPU without a TPU

`scripts/tpu/aot-check.sh` compiles the session's programs with XLA:TPU on
the development machine, so a program XLA:TPU refuses fails there and not on
the VM:

```bash
scripts/tpu/aot-check.sh            # the lanes' programs; --all for every device test
```

It runs the lanes' device tests locally with `TLALOC_TPU_PROGRAMS_DIR` set,
and every `PjrtSession` writes the TPU form of each program it compiles (the
form a TPU session sends: Mosaic kernels claimed as `tpu_custom_call`, MoE as
`ragged_dot`, f32 dots at HIGHEST), with its replica and partition counts.
The kernel fixtures, which run only on a TPU, are written by
`TpuProgramRecorderTest`. `examples/fine-tune` (the `finetune` lane) runs on
the GPU when a Qwen3-0.6B checkpoint is found, after Tlaloc is published to
mavenLocal; `--no-finetune` skips it. libtpu then compiles each program for a
compile-only topology (`v6e:1x1`, and `v6e:2x2` for multi-device programs;
`SINGLE` and `MULTI` choose others) in an amd64 container with the pinned
JAX and libtpu. libtpu is published for x86_64 only, so on an arm64 machine
Docker runs it under QEMU (binfmt for x86_64), where a program compiles in
0.4 s (a collective) to 200 s (the fine-tune step). `build/tpu-aot/<time>/summary.tsv` lists each program with its
collectives and its compiled memory: arguments plus outputs plus temporaries,
with an output aliased to a donated argument counted once (the parts are in
their own columns). `errors/` holds the refusals.

A compile is not a run: numerics, speed and run-time failures still need
the VM.

## The Cloud TPU VM session, step by step

Everything the VM runs is built beforehand on a development machine into
one tarball: the device tests with their classpath and a JUnit console
launcher, and `examples/fine-tune` as an installed application. The VM needs
a JDK 25 and libtpu, which `session.sh` installs; it runs no Gradle and
compiles nothing. The jars are JVM bytecode, so a tarball built on the
aarch64 GB10 runs on an x86_64 TPU VM.

On the development machine:

```bash
scripts/tpu/prepare.sh            # writes build/tpu-session.tar.gz (about 150 MB)
TARGET=cuda build/tpu-session/session.sh training serving   # optional rehearsal on a GPU
```

Create the VM (v5e or v6e, single host; spot is enough):

```bash
gcloud compute tpus tpu-vm create tlaloc-tpu \
  --zone=us-central1-a --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e --spot
gcloud compute tpus tpu-vm scp build/tpu-session.tar.gz tlaloc-tpu:~ --zone=us-central1-a
gcloud compute tpus tpu-vm ssh tlaloc-tpu --zone=us-central1-a
```

(For v5e: `--accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite`. The runtime
version names change; `gcloud compute tpus versions list --zone=...` lists the current ones.)

The fine-tune step compiles to 7.9 GiB for a v5e (16 GiB per chip): 6.7 GiB of f32
parameters and AdamW moments, both donated and updated in place, and 1.1 GiB of
temporaries.

On the VM:

```bash
tar xzf tpu-session.tar.gz && tpu-session/session.sh
```

`session.sh` installs the JDK and libtpu (the version pinned in
`scripts/tpu/libtpu-version`, recorded in the tarball's MANIFEST), downloads
Qwen3-0.6B at the fixtures' revision, and runs these lanes, each under a
30-minute timeout, continuing past failures:

| Lane | Runs | Answers |
|---|---|---|
| `g2b` | `PjrtTpuSmokeTest`, `PjrtTpuLocalCertTest` | libtpu reports `tpu`; threefry bit-exact; matmul gradient; bf16 narrowing |
| `kernels` | the `tpu_custom_call` kernel tests | Pallas- and Kotlin-emitted Mosaic payloads run and match their references |
| `training` | `PjrtCausalLmTrainingTest` | a CausalLM's gradients agree with the interpreter; AdamW trains on the TPU |
| `serving` | `PjrtQwen3GreedyParityTest` | Qwen3-0.6B greedy-decodes transformers' ids |
| `finetune` | `examples/fine-tune` with `TLALOC_TARGET=tpu` | Qwen3-0.6B fine-tuned on the TPU |
| `suite` | every device test with `TLALOC_TEST_PJRT_TARGET=tpu` | which CUDA-certified results hold on a TPU |

The TPU tests skip when no TPU client opens, so a VM whose libtpu does not
load would otherwise finish every lane with exit 0. Before the lanes,
`session.sh` stops when:

- there is no TPU device node (`/dev/accel*` or `/dev/vfio/N`);
- another process holds the chips (libtpu allows one process per TPU);
- `$HOME` has less than `MIN_FREE_GB` (10) free;
- `PjrtTpuSmokeTest.platformNameReportsTpu` does not pass.

It removes `--xla_dump*` flags from `XLA_FLAGS` and `LIBTPU_INIT_ARGS`
unless `DUMPS=1` (a dump is several GB), records the `XLA_*`, `LIBTPU*`,
`TPU_*`, `JAX_*` and `TLALOC_*` variables in `environment.txt`, and reports
each lane's passed tests; a lane that exits 0 with none passed is marked
`NO TEST PASSED`.

It prints a summary and writes `tpu-results-<time>.tar.gz`; copy that back
and delete the VM:

```bash
gcloud compute tpus tpu-vm scp tlaloc-tpu:~/tpu-results-*.tar.gz . --zone=us-central1-a
gcloud compute tpus tpu-vm delete tlaloc-tpu --zone=us-central1-a
```

`session.sh g2b serving` runs chosen lanes only. `TLALOC_TEST_PJRT_TARGET=tpu`
also works with Gradle in a checkout: `./gradlew :runtime-pjrt:jvmTest`.

## Expected skips vs runs

| Lane | On the GB10 (here) | On the TPU VM |
| --- | --- | --- |
| `PjrtTpuLocalCertTest` (7 tests) | RUNS, green | RUNS, green |
| `PjrtTpuSmokeTest` (5 tests) | SKIPS (no libtpu) | **RUNS** — the first TPU execution |
| `PjrtTpuMosaicKernelTest` (7 tests) | SKIPS (no libtpu) | **RUNS** — Mosaic payloads as `tpu_custom_call` vs numpy and vs XLA, with timings ([TPU_MEGAKERNELS.md](design/TPU_MEGAKERNELS.md)) |
| `TpuKernelFixtureCpuTest` (6 tests) | RUNS, green | RUNS, green |
| `TpuCustomCallParseCheckTest` (1 test) | RUNS where `~/.local/venvs/iree` has jax 0.12.0.1 | SKIPS unless `TLALOC_JAX_PYTHON` names a python with jax |
| CUDA smoke suites (`PjrtRngSmokeTest`, `PjrtBf16SmokeTest`, …) | RUN on GPU | SKIP (`cudaAvailable` false, no `nvidia-smi`) |

Caveat for the TPU VM: the CUDA suites gate on `PjrtBinaries.available`
**or** `cudaAvailable`; with `TLALOC_PJRT_PLUGIN_PATH` set to libtpu,
`PjrtBinaries.pluginPath` (the CUDA-family resolver) will also resolve to
that path — the CUDA suites still skip on the `cudaAvailable` gate, which
is the load-bearing one there.

## The Shardy validation lane

The SDY suites self-skip without their tool, silently on most boxes —
made visible here so nobody mistakes a skip for a cert:

- `SdyRoundTripTest` and `SdyPropagationTest` (`stablehlo/src/jvmTest`)
  gate on `assumeTrue(SdyOpt.available)` — `sdy-opt` resolved from
  `PATH` (`RoundTripHarness.kt`). No `sdy-opt` → the whole external
  round-trip/propagation lane SKIPS. There is no prebuilt aarch64
  binary; provisioning = build Shardy from source
  (github.com/openxla/shardy, bazel build of `//shardy/tools:sdy_opt`)
  and put `sdy-opt` on `PATH`. Same story on a TPU VM.
- What still certifies WITHOUT `sdy-opt` (runs everywhere, pinned
  structurally): the emit path emits sdy annotations wherever
  `DxirSharding` is present — `sdy.mesh` decls + `sdy.sharding_constraint`
  with the `<@mesh, [...]>` attr (EmitterTest's `shardConstraintLowersToSdyOp`
  and neighbours), `sdy.sharding = #sdy.sharding_per_value<[...]>` on
  sharded custom_calls (CoarsenedCustomCallTest, both the present and the
  deliberately-absent pins), and `sdy.manual_computation`
  in/out_shardings. Recorded gap: PLAIN ops with a `DxirSharding` do not
  get a per-op `sdy.sharding` attr (only constraint/custom_call/
  manual_computation sites annotate); propagation owns plain-op layouts.

## What the first TPU run must record

1. `platformNameReportsTpu` — that libtpu reports `"tpu"`.
2. `threefryUniformBitExactOnTpu` — **the main one**: the
   explicit-threefry emission is pure StableHLO integer ops and cannot
   fork by construction; a TPU pass makes the portability claim measured.
3. Whether TPU-XLA folds the f32→bf16→f32 convert pair (CUDA does) and
   where it rounds a bf16 dot's output (CUDA: at the op boundary, the
   257-tie discriminator) — re-run `PjrtBf16SmokeTest`'s questions via
   the TPU-parametrized lanes.
4. The TPU tiled buffer on-device size for bf16 (the smoke test prints
   it; CUDA pins 2 bytes/element, TPU may pad to tiles).
5. That libtpu accepts `premapped_buffer_size` (TorchTPU passes it); that
   `CompileOptions.STRICT_PRECISION` (`xla_allow_excess_precision = false`,
   which the TPU test lanes compile with) keeps a convert pair on a TPU as it
   does on CUDA; that a serialized-text Mosaic body
   (`MosaicKernel.ofSerializedText`) runs; that `CompileOptions.spmd` and the
   emitted `all_reduce` (channel handle, global device ids) reduce across
   chips; that `chlo.ragged_dot` MoE (`PjrtSession.moeRaggedDot`, on by
   default for a TPU) runs as XLA:TPU's grouped matmul. Code-level answers
   from TorchTPU: [work-log/torch-tpu-lessons.md](work-log/torch-tpu-lessons.md).
6. Donation semantics + memory kinds on TPU (untested on CUDA-certified
   paths beyond defaults).
7. The forms built for a TPU from vLLM-TorchTPU's code
   ([work-log/vllm-torchtpu-lessons.md](work-log/vllm-torchtpu-lessons.md)), each equal to
   the interpreter on XLA:GPU, none run on a TPU:
   - SAMPLE_TOKENS without `kernel` (`chlo.top_k` of 4,096 candidates, prefix sums, a
     `stablehlo.if` over the whole vocabulary): that it draws the reference's tokens
     (`PjrtSampleTokensXlaTest`), and its time at a 248,320-token vocabulary (3.5 ms on
     the GB10, where `top_k` sorts; `TLALOC_SAMPLE_BENCH=1`).
   - The MoE router without a sort (`PjrtMoeBenchTest.routingTopKTime`,
     `TLALOC_MOE_BENCH=1`; 0.114 against 1.578 ms at 2,048 rows on the GB10).
   - `permute = "onehot"` in the ragged-dot MoE form against the gather and scatter, and
     where the two cross (`MoeExpertsAttrs.ONEHOT_PERMUTE_PAIRS`, 1,024 pairs, is a guess).
   - `stablehlo.all_gather` / `stablehlo.reduce_scatter` across chips
     (`PjrtCollectiveCompileTest` runs them over one device), expert-parallel MOE_EXPERTS
     slices summed by a reduce-scatter, and PAGED_ATTENTION with `return_lse` merged across
     chips (`PjrtContextParallelTest` merges on one device).

The device suites (`PjrtBf16SmokeTest`, `PjrtMixedPrecisionSmokeTest` and
the rest) choose their backend through `TestBackend`: CUDA by default, the
TPU with `TLALOC_TEST_PJRT_TARGET=tpu`. The `suite` lane runs all of them on
the TPU; a test that pins a CUDA-specific bit pattern and fails there is a
measured difference between the backends, to be recorded rather than fixed.
