# KPTX — Kotlin PTX kernel tier: plan of record

**Status: v1 COMPLETE (§0.4.326–§0.4.337, 2026-07-18). v2 not started.**

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

**v1 — launch plumbing, no DSL (tasks 1–8). SHIPPED §0.4.327–§0.4.337.**
PjrtFfiRegistry production API (§0.4.327); `CudaDriverFfm` (§0.4.328);
XLA_FFI_CallFrame decoder (§0.4.329); KptxKernelRegistry (§0.4.330);
rms_norm fwd kernel vs XLA decompose-path oracle, max|diff| 4.77e-7
(§0.4.331); emitter typed-FFI upgrade — api_version=4 + op-native
`backend_config` dict, NOT the `mhlo.backend_config` spelling which
arrives empty through the plugin's StableHLO import (§0.4.332);
rms_norm backward — two chained kernels `bwd_dx → (dx, inv_rms)` /
`bwd_dw(…, inv_rms) → dw`, sequenced by XLA via the data dependence,
computing the coarsener's analytical VJP (dx 4.77e-7, dw 1.53e-5 vs
oracle; d_eps stays on the decompose path, no const-zero shortcut)
(§0.4.335); RmsNormKernel template — recognizer-driven claiming, the
LlamaDecoder-medium forward emits 2 typed-FFI `@kptx_rms_norm`
custom_calls and runs them E2E at max rel 1.8e-7 vs decompose
(§0.4.336); `tlaloc-pjrt-kptx-cuda` benchmark row (§0.4.337).

**v1 DoD — met, with the honest number:** same-run floor comparison on
LlamaDecoder-medium fwd: kptx min 1177 µs vs decompose min 906 µs —
**~135 µs per custom_call** floor cost (JVM host round-trip per
execution + XLA fusion loss at rms_norm granularity). Medians are
jitter-dominated (custom_call adds 0–1.2 ms/call run-dependent; the
decompose lanes are stable in the same session). Two consequences
feed forward: coarser kernels (attention, MLP — v2) amortize the
round-trip; the jitter itself is primary input to the v3.4 Glow-style
native-runtime go/no-go. Backward *claiming* is v2 task 15 by design —
`handleCoarsenedAdjoint` inlines gradient_body, so no COARSENED op
survives for a template to claim; the §0.4.335 kernels are the
claim-ready implementation. Note §0.4.333 (memory-safety
create_options; unified-memory reboot incident): benchmark sessions
opt into a bounded preallocated pool via `PjrtClientOptions` —
no-preallocate costs ~2× on dispatch-heavy loops.

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
