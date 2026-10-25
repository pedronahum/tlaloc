# IREE Runtime — Port Plan

**Status:** Landed (§0.4.284–§0.4.295). The plan below is preserved as design rationale; what shipped follows the same three phases but extended past the original CPU-only scope.

**Original target:** Phase 2 #2 of the /loop priority ladder — the M3-aligned first runtime backend. Without a runtime, the StableHLO Tlaloc emits is a paper artifact; the harness's measurements (§0.4.222–§0.4.228) reflect `DxirInterpreter` overhead, not native code throughput.

## What landed

| Phase | Plan | Shipped | Key deltas |
|-------|------|---------|------------|
| 1 (JNI scaffolding) | §0.4.230 plan; ~1-2 firings | §0.4.284 (`:runtime-iree` subprocess facade) | **Subprocess, not JNI** — `ProcessBuilder` over `iree-compile` / `iree-run-module`. JNI deferred indefinitely; subprocess startup is acceptable for the benchmark cadence. |
| 2 (dxir → IREE pipeline) | ~1-2 firings | §0.4.287 (`runOnIree` bridge with FloatArray I/O) | F32 only; flagfile-based input passing (§0.4.288 — inline `--input=` blew past argv length on llama-shaped weights). |
| 3a (CPU end-to-end) | ~1 firing | §0.4.288 (LlamaDecoder forward via IREE-CPU) | 2.589s live, two full compile+dispatch round trips. |
| 3b (FD validation) | implicit in plan | §0.4.289 (PyTorch reference; relDiff=2.33e-6) | New `harness/python/run_pytorch_llama.py`; cross-language flow via `.npy`. |
| 3c (CUDA target — beyond original scope) | not in plan | §0.4.290 / §0.4.291 (IREE-CUDA + bit-identical agreement) | `IreeTarget {LlvmCpu, Cuda}`; aarch64 Blackwell GB10 picked up out of the box. |
| 3d (backward agreement) | not in plan | §0.4.292 (LlamaDecoder backward agrees with PyTorch at all 13 grads) | Closed three coarsener AD const-zero shortcuts (CrossEntropy `d_labels`, RmsNorm `d_eps`, RoPE `d_theta`); fixed JVM Process pipe deadlock on ~116 KB gradient output. |
| 3e (perf rows) | implicit in plan | §0.4.293–§0.4.295 (CPU + CUDA forward + backward timing on tiny + medium configs) | M9 4-row comparison story closes: `tlaloc-interpreter`, `tlaloc-iree-cpu`, `tlaloc-iree-cuda`, `pytorch-compile`. |

## The IREE-CUDA → PJRT-FFM-CUDA split (post-§0.4.299)

A spike that wasn't in the original plan but shaped the production-side runtime story:

- **§0.4.299**: ran Tlaloc-emitted StableHLO through PJRT-XLA-CUDA (the same backend JAX uses internally) and landed at JAX-GPU level — within ~10% of JAX. The IREE-CUDA → JAX-GPU 6× gap is **purely IREE's GPU codegen**, not Tlaloc's IR shape.
- **§0.4.300**: surface-level IREE-CUDA flag sweep is exhausted. `--iree-dispatch-creation-enable-aggressive-fusion=true` makes forward 40× *slower*. Default flags are optimal.
- **§0.4.302–§0.4.308**: built `:runtime-pjrt` (subprocess facade → pure-Kotlin FFM bindings via `java.lang.foreign`); LlamaDecoder via `PjrtSession` lands at **2.51 ms / step** on PJRT-FFM-CUDA, within 7% of JAX-GPU.

**Production GPU lane is now PJRT-FFM-CUDA.** IREE-CUDA stays as the JNI/no-Python option (subprocess path; no FFM dependency), and as the cross-check that confirms the IR is sound regardless of backend.

The /loop's "no toolchain installs" rule: IREE 3.11.0 wheels (compiler + runtime) installed via `pip install iree-base-compiler iree-base-runtime` into `~/.local/venvs/iree/`. PJRT plugin (`xla_cuda_plugin.so`) installed via `pip install --upgrade "jax[cuda12]"`.

## Why IREE first (still true)

The original rationale held up:

- **PJRT** is multi-device, sharding-aware, and structurally tied to the Shardy/SDY pipeline. M3's exit criterion was single-device execution; PJRT's machinery is more complex than IREE for a first-runtime story. (PJRT eventually landed in §0.4.302+, after IREE proved the IR was sound.)
- **libtorch** is the "debug escape hatch" per §17 step 5; it's an eager-mode runtime that doesn't exercise StableHLO compilation. Useful for cross-checking gradient correctness, not for measuring compiled-function throughput. PyTorch ended up in this role through the §0.4.289 `run_pytorch_llama.py` reference.
- **Custom native runtime** would bypass the StableHLO + IREE/XLA codegen story Tlaloc is built around. Defeats the wedge.
- **IREE** is the single-device target that compiles StableHLO bytecode → native machine code via MLIR's lowering pipeline. Matches Tlaloc's "lower to StableHLO and let IREE/XLA codegen" thesis (§0.1).

## Existing surface (post-landing)

Tlaloc's StableHLO emission pipeline:

- `:stablehlo` module: `StablehloEmitter.kt` produces textual StableHLO MLIR from dxir (renamed from `Emitter.kt` in §0.4.316).
- `stablehlo-translate --serialize --target=1.0.0` converts text → bytecode (exercised in `RoundTripTest.kt` per §0.4.60).
- `:ir` module: `DxirFunction` / `DxirOp` / `DxirInterpreter` provide the dxir IR + reference interpreter.

Runtime modules:

- `:runtime-iree`: subprocess facade over `iree-compile` / `iree-run-module` / `iree-benchmark-module`. Targets `LlvmCpu` and `Cuda`. F32 only.
- `:runtime-pjrt`: pure-Kotlin FFM bindings to OpenXLA's PJRT C API (`libpjrt_c_api.so`). `PjrtSession` long-lived holder amortises init across many dispatches; `bufferFromHostF32` + `executeOn` pre-stage host→device transfers. F32 only.

## Out of scope (still mostly true)

- **GPU backends** beyond `Cuda`: Vulkan / Metal / WebGPU. M5+ if ever.
- **In-process IREE compilation**. Subprocess is fine; no use case driving a JNI port.
- **Multi-device** (sharding, collective insertion). M6+; would hit Shardy/SDY pipeline which is PJRT's strength, not IREE's.
- **Numerical bit-exact match between IREE CPU and `DxirInterpreter`**. Acceptance is f32 tolerance (1e-3 absolute), not bit equality. (CPU and CUDA agree bit-identically on LlamaDecoder per §0.4.291, but IREE vs DxirInterpreter is f32-tolerance only.)

## Standing assumptions

- `stablehlo-translate --serialize --target=1.0.0` is the bytecode serialiser. `iree-compile` accepts MLIR text directly (see `iree_toolchain_dgx_spark.md` memory) — `stablehlo-translate` is unused in the runtime path.
- `OpKind.STEP` matches `stablehlo.compare GT 0` semantics (§0.4 STEP discussion). IREE preserves this.
- Float operations in dxir are f32 throughout. IREE compiles f32 ops to f32 native instructions; no implicit promotion.
- The `:stablehlo/EmitterTest.kt` and `RoundTripTest.kt` (per §0.4.60) verify that emitted MLIR survives the `stablehlo-translate` parser; the IREE port inherits these without re-validating.

These assumptions are pinned in the existing `:stablehlo` test suite. If any breaks, the IREE port plan needs revisiting before the next firing in this arc lands.
