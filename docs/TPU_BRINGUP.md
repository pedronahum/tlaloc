# TPU bring-up runbook (G2a → G2b)

**Status (§0.4.459, G2a): the LOCAL half is done** — `PjrtTarget.Tpu`,
plugin resolution, the create-options platform gate, the proto/struct
backend audit, and a self-skipping TPU smoke suite are certified on the
GB10 (where every TPU-gated test skips cleanly, by design). **No TPU
execution claim exists yet.** G2b is running this runbook on a Cloud TPU
VM and turning the skips into passes.

## What the local half established (so G2b doesn't re-litigate it)

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
   `preallocate` (§0.4.333) are XLA **GPU** allocator options; a
   `PjrtTarget.Tpu` session creates its client with `create_options =
   NULL, num_options = 0` and refuses GPU options by name (both
   cross-wirings are `require`d in `PjrtSession`). **Unknowns, recorded:**
   the TPU plugin headers are not vendored, so libtpu's accepted option
   set is unverified here. Candidates seen in framework source
   (`ml_framework_name`, `ml_framework_version`,
   `max_inflight_computations`) are deliberately NOT passed until G2b
   confirms them against a live libtpu.
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
   are G3/G4 surface.

## The Cloud TPU VM session, step by step

Provision (v5e or v6e, single host, spot/preemptible suffices for the
smoke lane — accelerator type `v5litepod-1` / `v6e-1`):

```bash
gcloud compute tpus tpu-vm create tlaloc-g2b \
  --zone=us-central1-a --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base --spot
gcloud compute tpus tpu-vm ssh tlaloc-g2b --zone=us-central1-a
```

On the VM (TPU VM hosts are x86_64 — check `uname -m` and fetch the
matching JDK; the aarch64 URL is what the GB10 uses):

```bash
# 1. JDK 25 (userspace install, no sudo — mirror the GB10 layout)
mkdir -p ~/.local/jdks && cd ~/.local/jdks
# x86_64:
curl -LO https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse
tar xf eclipse && rm eclipse   # produces jdk-25.0.N+M/

# 2. libtpu (the PJRT plugin). Either the wheel:
python3 -m venv ~/venv && ~/venv/bin/pip install libtpu
# → ~/venv/lib/python3.N/site-packages/libtpu/libtpu.so
# ...or use the image's /lib/libtpu.so if present.

# 3. Clone + point the resolver at the plugin
git clone <tlaloc remote> ~/tlaloc && cd ~/tlaloc
export TLALOC_PJRT_PLUGIN_PATH=$(ls ~/venv/lib/python3.*/site-packages/libtpu/libtpu.so)

# 4. The smoke lane (JAVA_HOME must name the JDK just installed)
JAVA_HOME=~/.local/jdks/jdk-25.0.* ./gradlew \
  :runtime-pjrt:jvmTest --tests "*PjrtTpu*" --rerun
```

## Expected skips vs runs

| Lane | On the GB10 (here) | On the TPU VM |
| --- | --- | --- |
| `PjrtTpuLocalCertTest` (7 tests) | RUNS, green | RUNS, green |
| `PjrtTpuSmokeTest` (5 tests) | SKIPS (no libtpu) | **RUNS — this is G2b** |
| CUDA smoke suites (`PjrtRngSmokeTest`, `PjrtBf16SmokeTest`, …) | RUN on GPU | SKIP (`cudaAvailable` false, no `nvidia-smi`) |

Caveat for the TPU VM: the CUDA suites gate on `PjrtBinaries.available`
**or** `cudaAvailable`; with `TLALOC_PJRT_PLUGIN_PATH` set to libtpu,
`PjrtBinaries.pluginPath` (the CUDA-family resolver) will also resolve to
that path — the CUDA suites still skip on the `cudaAvailable` gate, which
is the load-bearing one there.

## What G2b must record (the open questions, from §0.4.457/§0.4.459)

1. `platformNameReportsTpu` — that libtpu reports `"tpu"`.
2. `threefryUniformBitExactOnTpu` — **the flagship**: the §0.4.422
   explicit-threefry emission is pure StableHLO integer ops and cannot
   fork by construction; a TPU pass makes the portability claim measured.
3. Whether TPU-XLA folds the f32→bf16→f32 convert pair (CUDA does) and
   where it rounds a bf16 dot's output (CUDA: at the op boundary, the
   257-tie discriminator) — re-run `PjrtBf16SmokeTest`'s questions via
   the TPU-parametrized lanes.
4. The TPU tiled buffer on-device size for bf16 (the smoke test prints
   it; CUDA pins 2 bytes/element, TPU may pad to tiles).
5. Which create-options libtpu accepts (try `ml_framework_name` etc.;
   today we pass none).
6. Donation semantics + memory kinds on TPU (untested on CUDA-certified
   paths beyond defaults).

After the smoke lane is green, the G1c/G1d suites re-run there
(`PjrtBf16SmokeTest`, `PjrtMixedPrecisionSmokeTest` — currently
CUDA-gated; parametrizing them onto a shared multi-backend harness is the
named deferral from §0.4.459).

Delete the VM when done (spot TPUs bill while idle):

```bash
gcloud compute tpus tpu-vm delete tlaloc-g2b --zone=us-central1-a
```
