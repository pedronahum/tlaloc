# KPTX — Kotlin PTX kernel tier: plan of record

**Status: spike PASSED (2026-07-18). v1 not started.**

KPTX is the escape-hatch tier below StableHLO — Tlaloc's analog of what
Pallas is to JAX: hand-written PTX kernels, authored (eventually) in a
typed Kotlin DSL, dispatched inside XLA-compiled PJRT executables via
`stablehlo.custom_call`. Design informed by two studies:

- **pyptx** (github.com/patrick-toulme/pyptx) — the launch architecture
  (framework owns buffers/streams; kernels registered by handle; driver
  JIT via `cuModuleLoadData`), the value-type IR + byte-identical
  round-trip discipline, the declarative ISA-spec validation, and the
  PTX→DSL transpiler workflow.
- **PyTorch Glow** — the strategic frame: coarse ops make "one coarse op
  = one hand-written kernel" viable (no fusion search); own everything
  except matmul (cuBLASLt); a later Glow-style native runtime
  (instruction IR + static arena) can swap in around the same kernel
  library. That go/no-go is deliberately deferred to the end of this arc
  (task v3.4).

## Feasibility spike (done)

`runtime-pjrt/src/jvmTest/.../PjrtCustomCallRegistrationSpikeTest.kt`
proves the one genuinely uncertain integration point against the stock
JAX plugin (`jax_plugins/xla_cuda12/xla_cuda_plugin.so`, jaxlib 0.10.0)
on the GB10:

1. The plugin's `PJRT_Api.extension_start` chain exposes
   `PJRT_Gpu_Custom_Call` (type 0; observed chain
   `[20, 19, 12, 7, 6, 5, 4, 0, 3, 2, 1]`).
2. `PJRT_Gpu_Register_Custom_Call` accepts a **pure-Kotlin FFM upcall
   stub** as a typed-FFI (api_version=1) execute handler — no C shim,
   no JNI, no self-built plugin.
3. XLA compiles `stablehlo.custom_call @tlaloc_kptx_spike
   {api_version = 4 : i32}` and **dispatches into the Kotlin handler at
   EXECUTE stage** at run time. The XLA_FFI metadata protocol (frame
   extension type 1 → handler writes its api_version 0.3) is required
   and implemented.
4. Negative control: an unregistered symbol fails compile with
   `No FFI handler registered for ... on a platform CUDA` — same
   failure `JaxRejectsTlalocCustomCallMlirTest` pins from the Python
   side.

ABI facts (struct offsets for the extension walk, register args, call
frame, metadata) are documented in the spike test's KDoc; the
XLA-side layouts come from the **jaxlib-shipped** header
`jaxlib/include/xla/ffi/api/c_api.h` (FFI API 0.3), the PJRT-side
layouts from `openxla/xla` `pjrt_c_api.h` / `pjrt_c_api_gpu_extension.h`
(GPU extension version 2).

## Roadmap

Tracked as session tasks #1–#19, strictly sequential. Summary:

**v1 — launch plumbing, no DSL (tasks 1–8).** PjrtFfiRegistry
production API; `CudaDriverFfm` (libcuda via FFM: cuModuleLoadData /
cuLaunchKernel / cuFuncSetAttribute); XLA_FFI_CallFrame decoder
(buffers, attrs, stream via XLA_FFI_Stream_Get); kernel launch registry
(name → LaunchConfig, module cache); first hand-written `.ptx` kernel
(rms_norm fwd) with decompose-path PJRT oracle; emitter typed-FFI
upgrade (api_version=4 + `mhlo.backend_config` dict); backward kernel
wired into coarsened VJP (analytical adjoints, per project rule);
`tlaloc-pjrt-kptx-cuda` benchmark row. **DoD:** a hand-written PTX
kernel runs inside an XLA-compiled Tlaloc program on the GB10,
correctness pinned, overhead quantified vs the 2.51 ms/step
PJRT-FFM-CUDA baseline.

**v2 — the Kotlin DSL (tasks 9–15).** Value-type PTX IR + emitter;
opcode-agnostic parser with byte-identical round-trip corpus;
declarative ISA spec table validating an `inst("opcode.mods", ...)`
escape hatch at the Kotlin call site; KernelScope builder DSL
(one-call-one-instruction registers/predicates/control-flow/smem);
symbolic-shape kernel signatures + specialization cache keyed
(arch, shape env, template args); typed wrappers for mma.sync + warp
intrinsics (WGMMA/TMA/tcgen05 deferred until needed); v1 kernels
rewritten in the DSL and claiming coarse ops through the Layer-3 kernel
registry. **DoD:** a kernel written in pure Kotlin executes inside an
XLA executable; its emitted PTX is byte-stable under parse/emit.

**v3 — transpiler + polish (tasks 16–19).** PTX → Kotlin DSL
transpiler; bootstrap-workflow demo (transpile an expert kernel, edit,
benchmark) — the adoption artifact; optional debug/verification extras
(occupancy report, deadlock beacons, differential symbolic execution);
spec closure + the strategic go/no-go on the Glow-style native runtime.

## Standing decisions

- **Never hand-roll GEMM.** cuBLASLt is the matmul answer at every
  stage; KPTX kernels cover the coarse non-matmul ops (rmsnorm, rope,
  attention variants, elementwise residue).
- **Framework owns memory and streams.** KPTX never `cuMemAlloc`s
  tensor data in the PJRT path; handlers write into XLA's buffers on
  XLA's stream (the pyptx lesson that makes kernels compose).
- **Driver JIT first.** PTX text → `cuModuleLoadData`; no CUDA toolkit
  dependency at runtime (aligned with the no-Python-at-runtime story).
  `ptxas` fallback only if driver-version lag ever bites.
- **NVIDIA-only by construction.** This tier is CUDA-track; IREE / CPU
  / mobile tracks are unaffected. The IREE analog (custom-dispatch)
  is out of scope unless a concrete need appears.
