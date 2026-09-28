# TPU Mosaic kernels

Status: 🧪 written, never run on a TPU. Everything below that says "checked"
ran on a CPU host (the GB10 workstation, aarch64, no TPU).

A Pallas kernel on TPU lowers to `stablehlo.custom_call @tpu_custom_call`
whose `backend_config` is JSON carrying the kernel as a serialized Mosaic
module. Tlaloc can now carry such a kernel inside a DXIR program, emit that
call for a TPU, and run the kernel's DXIR reference everywhere else. This is
the ground work for whole-decode-step kernels (one `pallas_call` per step,
weights streamed from HBM into VMEM, KV cache updated in place).

## What exists

| Piece | Where |
|---|---|
| `OpKind.MOSAIC_KERNEL`: operands → one or more results; attrs `mosaic_kernel` (`MosaicKernel`), `reference` (a `DxirFunction` with the op's signature), optional `reference_fallback` | `ir/.../OpKind.kt`, `MosaicKernelAttrs.kt`, builder `DxirEmitter.mosaicKernel(...)` |
| `MosaicKernel`: kernel name, base64 body, `custom_call_config` fields in order, `inputOutputAliases`, `hasSideEffect`, provenance. `backendConfigJson()` writes the JSON with JAX's field order and spacing | `ir/.../recognizer/kernel/MosaicKernel.kt` |
| `lowerMosaicKernels(fn, target)`: on a `google` target the op is claimed (a `KernelDescriptor` with `mosaic` set); elsewhere it is replaced by `reference` if the op declares `reference_fallback`, and refused by name if not | `ir/.../recognizer/kernel/MosaicKernelLowering.kt` |
| StableHLO emission: `backend_config` as an escaped MLIR string, `kernel_name`, row-major `operand_layouts`/`result_layouts`, several results without a tuple, `output_operand_aliases`. An unresolved `MOSAIC_KERNEL` is refused by name. `output_operand_aliases` is also available to every other `KernelDescriptor` | `stablehlo/.../StablehloEmitter.kt` (`emitCustomCall`) |
| `PjrtSession` and `runOnPjrt` resolve `MOSAIC_KERNEL` for their `PjrtTarget` before emitting (`PjrtTarget.kernelTarget`) | `runtime-pjrt/.../PjrtSession.kt` |
| The interpreter evaluates `reference`; both AD transforms refuse the op by name (it is inference-only: the body has no derivative Tlaloc can check); the cost model charges the reference's FLOPs | `DxirInterpreter`, `DemotedOpKinds.kt`, `CostModel.kt` |
| `MosaicRmsNorm.emit(rows, hidden, dtype, eps)`: Mosaic MLIR text for an RMSNorm kernel, f32 or bf16, written from Kotlin | `stablehlo/.../mosaic/MosaicRmsNorm.kt` |
| `harness/python/export_tpu_kernels.py`: exports Pallas kernels with `jax.export(..., platforms=["tpu"])` on a CPU host and serializes the Kotlin-emitted Mosaic text | |
| `harness/python/check_tpu_custom_call.py`: jaxlib-side checks of a Tlaloc-emitted program | |

## The payloads

In `runtime-pjrt/src/jvmTest/resources/tpu-kernels/`, one directory each:
`manifest.json` (body, config, shapes, input seeds, tolerance, versions),
`mosaic.mlir` (the module before serialization), `outN.bin` (numpy reference,
little-endian f32). Inputs come from an integer formula both sides implement;
every input value is exact in f32 and bf16.

| Payload | Source | Inputs → results | Body |
|---|---|---|---|
| `rmsnorm_f32` | Pallas | f32[16,256], f32[1,256] → f32[16,256] | 1,601 B |
| `rmsnorm_bf16` | Pallas | bf16[16,256], bf16[1,256] → bf16[16,256], f32 arithmetic inside | 1,775 B |
| `matmul_f32` | Pallas | f32[8,256] · f32[256,128] → f32[8,128], `precision=HIGHEST` | 1,218 B |
| `norm_swiglu_f32` | Pallas | x, norm weight, W_gate, W_up → (RMSNorm(x), silu(h·W_gate) ⊙ h·W_up): two results | 2,305 B |
| `kv_update_inplace_f32` | Pallas | cache f32[16,128], new f32[4,128] → cache with rows 8–11 replaced by DMA; result aliased to the cache operand | 1,122 B |
| `rmsnorm_f32_kmosaic` | Kotlin (`MosaicRmsNorm`) | as `rmsnorm_f32` | 1,198 B |
| `rmsnorm_bf16_kmosaic` | Kotlin (`MosaicRmsNorm`) | as `rmsnorm_bf16` | 1,278 B |

Regenerate (CPU only; the Kotlin text is written by `TpuKernelFixtureCpuTest`):

```bash
./gradlew :runtime-pjrt:jvmTest --tests '*TpuKernelFixtureCpuTest'
JAX_PLATFORMS=cpu ~/.local/venvs/iree/bin/python harness/python/export_tpu_kernels.py \
  --out runtime-pjrt/src/jvmTest/resources/tpu-kernels --kmosaic-dir runtime-pjrt/build/kmosaic
```

The exporter is deterministic: two runs write identical files.

## Checked on the CPU host

- Each Pallas kernel, run in Pallas's TPU interpret mode on the CPU, matches
  its numpy reference (max |diff| 5.7e-6 for `norm_swiglu_f32`, at most
  1.2e-7 for the others).
- `MosaicKernel.backendConfigJson()` re-emits the `backend_config` that
  `jax.export` wrote, byte for byte (`TpuKernelFixtureCpuTest`).
- Every emitted program parses and verifies with jaxlib's StableHLO and TPU
  dialects; its `backend_config` decodes as JSON to the manifest's body and
  config; the body parses as Mosaic bytecode at serialization version 9 and
  deserializes; the program converts to HLO, and the HLO `custom-call`
  instruction (target, layouts, aliasing, backend config) is identical to
  the one JAX's own export produces for the same Pallas kernel
  (`TpuCustomCallParseCheckTest`).
- `MosaicRmsNorm`'s module is identical to Pallas's for the same kernel after
  `cse`, f32 and bf16; jaxlib serializes it on the CPU, and re-serializing
  the emitter's current text reproduces the checked-in body byte for byte.
- Each op's reference decomposition matches numpy in the interpreter and,
  lowered to StableHLO, compiled and run by XLA's CPU client.
- On a non-TPU target an op without `reference_fallback` is refused by name,
  and one with it emits no custom call.

Not checkable here: whether libtpu compiles and runs the bodies, the
numerics of Mosaic's arithmetic on the device, and the timings. The Mosaic
layout and lowering passes are part of libtpu, not jaxlib.

## What the TPU run checks

`PjrtTpuMosaicKernelTest` (7 tests, one per payload; skips without libtpu).
Each runs the same DXIR program on one `PjrtSession(target = PjrtTarget.Tpu)`
twice: as the `tpu_custom_call`, and as the reference decomposition through
XLA. It asserts:

- custom call vs numpy within max(tolerance, 2e-4 × max|ref|);
- XLA vs numpy within max(tolerance, 2e-2 × max|ref|): an f32
  `dot_general` without a precision config runs as one bf16 pass on a TPU;
- `kv_update_inplace_f32` exactly (tolerance 0).

It prints the three differences (custom vs numpy, XLA vs numpy, custom vs
XLA) and the median time of 100 dispatches of each program with inputs
staged once. It runs under the runbook's existing filter:

```bash
./gradlew :runtime-pjrt:jvmTest --tests "*PjrtTpu*" --rerun
```

The JAX baseline for the same kernels, in a separate venv (it brings the same
libtpu):

```bash
python3 -m venv ~/jaxvenv && ~/jaxvenv/bin/pip install 'jax[tpu]==0.10.0'
~/jaxvenv/bin/python harness/python/run_tpu_kernels_jax.py \
  --fixtures runtime-pjrt/src/jvmTest/resources/tpu-kernels
```

It runs each Pallas kernel with `jax.jit` on the TPU, compares it with the
numpy reference and prints the median call time. If JAX runs a kernel and
Tlaloc's custom call does not, the fault is in Tlaloc's call; if both fail,
it is in the payload or the libtpu pairing. With the same venv's python as
`TLALOC_JAX_PYTHON`, `TpuCustomCallParseCheckTest` also runs on the VM.

## Version pinning

A Mosaic body is MLIR bytecode of Mosaic's dialects at a serialization
version (`stable_mosaic.version`). libtpu reads the versions up to the one it
was built with and upgrades older ones; a body newer than the libtpu fails
to compile. The payloads here were written by jax 0.10.0 / jaxlib 0.10.0 at
version 9 (the version jax 0.10.0 exports at without a TPU attached), and
the Kotlin-emitted ones are serialized at 9 as well. jax 0.10.0 pairs with
`libtpu==0.0.40.*` (its `jax[tpu]` requirement); install that on the VM:

```bash
~/venv/bin/pip install 'libtpu==0.0.40.*'
```

A newer libtpu should still read version 9. If it refuses, regenerate the
payloads with the jaxlib that matches it. Each manifest records the jax,
jaxlib and serialization version it was written with.

## Not done

- Kernels hold whole arrays in VMEM: no grid, no pipelined HBM→VMEM
  streaming, so sizes are bounded by VMEM.
- No collectives inside a kernel (`has_communication`, `collective_id`
  are representable in `customCallConfig` but untested).
- Claiming is by op kind. The recognizers do not yet replace a DXIR pattern
  with a `MOSAIC_KERNEL`.
- The Kotlin Mosaic emitter covers RMSNorm only.
- No gradients through the op.
