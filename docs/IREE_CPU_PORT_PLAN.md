# IREE CPU Runtime — Port Plan

**Status:** Planning artifact (§0.4.230). Implementation has not started.

**Target:** Phase 2 #2 of the /loop priority ladder — the M3-aligned first runtime backend. Without a runtime, the StableHLO Tlaloc emits is a paper artifact; the harness's measurements (§0.4.222–§0.4.228) reflect `DxirInterpreter` overhead, not native code throughput.

**Why IREE CPU first** (vs PJRT/XLA, libtorch, or a custom native runtime):

- **PJRT** is multi-device, sharding-aware, and structurally tied to the Shardy/SDY pipeline. M3's exit criterion is single-device execution; PJRT's machinery is wasted complexity at this stage.
- **libtorch** is the "debug escape hatch" per §17 step 5; it's an eager-mode runtime that doesn't exercise StableHLO compilation. Useful for cross-checking gradient correctness, not for measuring compiled-function throughput.
- **Custom native runtime** would bypass the StableHLO + IREE/XLA codegen story Tlaloc is built around. Defeats the wedge.
- **IREE CPU** is the single-device target that compiles StableHLO bytecode → native machine code via MLIR's lowering pipeline. Matches Tlaloc's "lower to StableHLO and let IREE/XLA codegen" thesis (§0.1). Lighter weight than PJRT for single-device use.

## Pre-requisites (toolchain-gated)

The /loop's "no toolchain installs" rule means this plan documents the protocol; implementation waits for the user to install:

- **IREE compiler** (`iree-compile`) — accepts StableHLO bytecode + emits VMFB (IREE Virtual Machine FlatBuffer) for a target backend.
- **IREE runtime library** (`libiree_runtime`) — loads VMFB modules and dispatches to a configured device (cpu, vulkan, etc).
- **JNI bindings** — exposing IREE's runtime API to JVM. IREE's official Java bindings: https://github.com/openxla/iree/tree/main/runtime/bindings/java

Pre-built `stablehlo-translate` already lives at `/opt/homebrew/bin` (per the loop's standing rules). It produces StableHLO bytecode from textual MLIR — the exact input format IREE's compiler accepts.

## Existing surface

Tlaloc's StableHLO emission pipeline is already in place:

- `:stablehlo` module: `Emitter.kt` produces textual StableHLO MLIR from dxir.
- `stablehlo-translate --serialize --target=1.0.0` converts text → bytecode (already exercised in `RoundTripTest.kt` per §0.4.60).
- `:ir` module: `DxirFunction` / `DxirOp` / `DxirInterpreter` provide the dxir IR + reference interpreter.

What's missing for IREE CPU dispatch:
1. A JNI binding layer (`:runtime-iree`) calling IREE's Java/C API.
2. A pipeline that takes a `DxirFunction` → emits StableHLO bytecode → invokes `iree-compile` → loads the resulting VMFB → dispatches with concrete inputs → returns the output.
3. End-to-end smoke test verifying numerical agreement with `DxirInterpreter`.

## Methodology — three-phase migration

### Phase 1 — JNI scaffolding (gated on user-side IREE install)

**Deliverable:** A new module `:runtime-iree` that:

- Has a `build.gradle.kts` declaring `cinterop` (Kotlin/Native) and JNI (JVM) dependencies on `libiree_runtime`.
- Exposes a Kotlin API: `IreeRuntime.compile(stablehloBytes: ByteArray): IreeModule` and `IreeModule.invoke(inputs: List<FloatArray>): List<FloatArray>`.
- Loads the pre-installed `iree-compile` via `ProcessBuilder` (host process invocation, not in-process compilation — simpler first slice).

**Acceptance:** Unit tests verify that `IreeRuntime.compile` accepts a hand-written StableHLO bytecode for `f(x) = x + 1` and the resulting module dispatches correctly on `(2.0f) → 3.0f`.

**Estimated:** 1-2 firings post-IREE-install. The JNI layer is the structural piece; once it lifts, subsequent phases plug into it.

### Phase 2 — dxir → IREE pipeline integration (1-2 firings)

**Deliverable:** A bridging API in `:runtime-iree` (or `:ir`):

```kotlin
fun runOnIree(fn: DxirFunction, inputs: List<FloatArray>): List<FloatArray> {
    val stablehloText = StablehloEmitter.emit(fn)
    val bytecode = stablehloTranslateSerialize(stablehloText)
    val module = IreeRuntime.compile(bytecode, target = "cpu")
    return module.invoke(inputs)
}
```

Mirrors `DxirInterpreter.evalFunction(fn, inputs)`'s API shape — drop-in alternative for benchmarks and tests that want native dispatch instead of interpreter dispatch.

**Acceptance:** A `DxirIreeRoundTripTest` that takes 5-10 representative dxir primals from `:benchmarks/BenchmarkPrimals.kt` (e.g., `iterateNTimes`, `bgdHyperOptOuterLoopPrimal`, `cartPolePhase1Primal`) and asserts `IreeRuntime.evalFunction(fn, inputs) ≈ DxirInterpreter.evalFunction(fn, inputs)` within 1e-3 absolute tolerance.

**Estimated:** 1-2 firings. Most of the work is the `runOnIree` glue + tests; StableHLO emission is already shipped.

### Phase 3 — Harness re-run + head-to-head numbers (1 firing)

**Deliverable:** A new harness inhabitant runner that calls `IreeRuntime.evalFunction` instead of `DxirInterpreter.evalFunction` for the timing loop. Produces a second CSV/JSON file with `framework=iree-cpu` rows.

When combined with the Python references from `HEAD_TO_HEAD_HARNESS_PLAN.md` Phase 2 (gated on PyTorch+JAX install), this gives a four-row comparison per benchmark: `tlaloc-interpreter` (current baseline), `tlaloc-iree-cpu` (new), `pytorch-compile`, `jax-jit`.

**Acceptance:** Per the M9 exit criterion (§11.13): within 20% of paper's figures, >3× over `torch.compile` on at least three of six, f32-tolerance numerical match. The §0.4 entry titled "Phase 1 closed — coarsening at M9 parity" pins the comparison table.

**Estimated:** 1 firing once Phase 2 ships.

## Scoping decisions

### CPU-only

IREE supports CPU, Vulkan, Metal, and WebGPU backends via the same compilation pipeline. CPU is the single-device target M3 calls for. GPU/mobile backends land at M5+ (per §11.16).

### `iree-compile` as host process, not in-process

IREE's compiler is a large MLIR-based pipeline. Pulling it in as a JNI library would significantly increase the runtime's size; for Phase 1, calling the pre-installed `iree-compile` binary via `ProcessBuilder` is simpler and Tlaloc already follows this pattern for `stablehlo-translate`. In-process compilation is a Phase 4+ optimisation when startup latency becomes worth measuring.

### Single-device dispatch only

Multi-device coordination is what Shardy + PJRT are for. IREE supports single-device dispatch natively; M3 doesn't need more than that. Multi-device deferred to M6+.

### F32 only

Tlaloc's IR is F32 throughout (the F64 tape path is genuinely deferred per the §0.4.229 register). The IREE module's input/output types match dxir's F32 throughout.

## Out of scope

- **Multi-device** (sharding, collective insertion). M6+.
- **GPU backends** (Vulkan / Metal / WebGPU). M5+.
- **libtorch escape hatch**. Documented as deferred per §17 step 5; no use case driving it today.
- **In-process IREE compilation**. Process-call to `iree-compile` is fine for Phase 1.
- **PJRT integration**. Defer until multi-device matters.
- **Numerical bit-exact match between IREE CPU and `DxirInterpreter`**. Acceptance is f32 tolerance (1e-3 absolute), not bit equality — IREE's f32 codegen may use FMA fusion or other transformations that preserve f32 semantics but not bit equality.

## Why this plan is structured similarly to QWOP / CartPole / HMC plans

Same shape: planning artifact (this doc) → multi-firing implementation → integration test → harness inhabitant. The §0.4.207 register refresh pattern then closes the arc.

The IREE plan differs from QWOP/CartPole/HMC in one structural way: **it's gated on a user-side toolchain install** rather than a code-level dependency. The plan exists so that when the user installs IREE, subsequent firings have a concrete decomposed roadmap to follow, instead of starting from "what's IREE and how do I integrate it?"

## Phase 1 first-slice — concrete next firing (post-IREE-install)

The immediate next firing after IREE is installed should land:

1. New `:runtime-iree` module under `runtime-iree/` with `build.gradle.kts`.
2. `IreeRuntime.kt` with `compile(stablehloBytes: ByteArray): IreeModule` and `IreeModule.invoke(inputs)`.
3. A smoke test that hand-writes a 3-line StableHLO for `f(x) = x + 1`, compiles via `iree-compile`, dispatches, and asserts `2.0f → 3.0f`.

**Estimated:** 1 firing once IREE is installed. Bounded scope: JNI binding + smoke test, no dxir integration yet.

## Standing assumptions documented in `:stablehlo`

- `stablehlo-translate --serialize --target=1.0.0` is the bytecode serialiser. `iree-compile` accepts this format directly.
- `OpKind.STEP` matches `stablehlo.compare GT 0` semantics (§0.4 STEP discussion). IREE preserves this.
- Float operations in dxir are f32 throughout. IREE compiles f32 ops to f32 native instructions; no implicit promotion.
- The `:stablehlo/EmitterTest.kt` and `RoundTripTest.kt` (per §0.4.60) already verify that emitted MLIR survives the `stablehlo-translate` parser; the IREE port assumes this without re-validating.

These assumptions are pinned in the existing `:stablehlo` test suite. The IREE port plan inherits them — if any breaks, the IREE port plan needs revisiting before the next firing in this arc lands.
