# Tlaloc

**A Kotlin-native differentiable-programming framework.** Compile-time AD via a K2 compiler plugin, shape- and sharding-typed tensors enforced by the Kotlin type system, StableHLO + Shardy lowering, vendor-fused-kernel custom-calls (FlashAttention v3, TPU pallas, etc.), and Netflix-Maestro-native step orchestration. KMP-first: one source set targets JVM, Android, iOS, and WASM.

> Pre-alpha. Built in the open; not yet packaged for consumption. See [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) §0.4 for the session-by-session ship log.

---

## Why

PyTorch owns Python. JAX owns research. We don't try to fight on that turf.

Tlaloc targets the gap Python frameworks leave open:

- **Android / KMP on-device training** — LoRA fine-tuning, federated learning, personalization, sensor-driven models.
- **JVM-native enterprise ML** — Spring Boot / Kafka / Flink / Spark hot paths.
- **Shape-safety-obsessed teams** — fintech, aerospace; silent shape bugs that are a P0.
- **Differentiable simulation** — games, robotics, XR where Kotlin already has a foothold (libGDX, Korge, Android XR).

## What it is

- **Compile-time AD.** `grad { x: Float -> ... }` is realized by a K2 plugin — not a runtime tape. The plugin lowers the lambda body to a typed SSA IR (DXIR), runs a coarsening pass, applies the reverse-mode transform, and synthesises a forward Kotlin function that computes the gradient directly.
- **φ-calculus coarsening** from Shen, Shivers, Dea et al. [Efficient, Sound Gradient Descent in Dynamic and Dependent Control Flow, OOPSLA 2021](docs/papers/coarsening-autodiff.txt). The pass implements F1–F5 + C1–C9 with a [Symja](https://github.com/axkr/symja_android_library)-backed symbolic engine for closed-form closure of affine / indexed-affine / variable-coefficient / power-form recurrences.
- **Shape-typed tensors.** `DTensor<Rank1<Sym>, F32>` carries shape and dtype in the Kotlin type system. Rank mismatches and shape-wrong broadcasts are compile errors, surfaced in the IDE.
- **Named indices** (Layer 1). `Named<N, A>` lets a tensor's axes carry symbolic names enforced at the type level: `contract(M_x_K, K_x_N)` only compiles when the shared axis name lines up.
- **Sharding-typed.** Mesh axis names live in the type system. Users annotate a handful of tensors with partition specs; the compiler propagates shardings through the rest of the program and hands off to [Shardy](https://github.com/openxla/shardy)'s propagation passes. DP, TP, EP, ZeRO, and context parallelism are all the same mechanism.
- **Four-worlds discipline** (Layer 2). `program { }` and `workflow { }` blocks live in `OrchestrationScope`; the body lambda's receiver is `KernelScope` — kernel-only ops compile, orchestration ops don't. `BufferHandle<T, M>` is the only thing that crosses a step boundary.
- **First-class Maestro step type** (Layer 2.5). Vendored `third-party/maestro/`. Workflows declare `"type": "Tlaloc"` natively; `TlalocStepRuntime` launches a runtime container image and executes via `TlalocRunner`.
- **Pattern-recognition + per-target kernel registry** (Layer 3). Recognises FlashAttention / RMS norm / RoPE / cross-entropy. Per-(pattern, target) kernel selection across 7 device descriptors (NVIDIA H100/A100, AMD MI300X, Google TPU v4/v5e/v6e, AWS Trainium2). Best-effort KV-quant. Cost-model-driven backend matrix on every manifest.
- **StableHLO + SDY emission.** We don't write CUDA. We lower to MLIR and let PJRT-backed runtimes (XLA, IREE) codegen.

## What it is not

- Not a PyTorch clone. No `torch.nn` parity goal.
- Not a research playground for novel AD algorithms. Boring, correct, fast.
- Not a CUDA kernel project. Not a collectives library.
- Not Python-compatible at the API level. Interop is via ONNX / StableHLO artifacts.

## Example — scalar `grad`

```kotlin
import io.tlaloc.autograd.grad

fun main() {
    // Paper-faithful BGDHyperOpt (OOPSLA 2021 §6.2), minus the convergence break.
    // Inputs packed as [r, x_0..x_{M-1}, y_0..y_{M-1}]; gradient wrt slot 0 = d(err)/dr.
    val g = grad { p: DTensor<Rank1<Sym>, F32> ->
        val r = p[0]; val Mf = 3.0f
        var Sxy = 0.0f; var Sx2 = 0.0f
        for (i in 0 until 3) {
            val xi = p[1 + i]; val yi = p[4 + i]
            Sxy = Sxy + xi * yi
            Sx2 = Sx2 + xi * xi
        }
        var w = 0.0f; var k = 0
        while (k < 50) {
            val d = 2.0f * (Sx2 * w - Sxy)
            w = w - r * d / Mf
            k = k + 1
        }
        var e = 0.0f
        for (j in 0 until 3) {
            val diff = p[4 + j] - p[1 + j] * w
            e = e + diff * diff
        }
        (e / Mf).sqrt()
    }
}
```

No tape. No `torch.tensor(..., requires_grad=True)`. No gradient type wrapping the primal type. The lambda reads as straight Kotlin — the compiler plugin does the work.

## Example — Layer 3 pipeline (recognize → coarsen → kernel → matrix)

```kotlin
import io.tlaloc.ir.recognizer.*
import io.tlaloc.ir.recognizer.coarsener.*
import io.tlaloc.ir.recognizer.kernel.*
import io.tlaloc.maestro.populateBackendMatrix

// Recognise + coarsen the canonical attention shape.
val matches = recognizeAll(userFn)            // [FlashAttention]
val coarsened = coarsenRecognizedPatterns(userFn, matches)

// Lower for a specific target — H100 picks flash_attn_v3.
val h100 = lowerKernelChoice(coarsened, KernelTarget.NVIDIA_H100)

// Or populate the full per-target matrix that ships with the manifest.
val matrix = populateBackendMatrix(
    userFn,
    targets = listOf(
        KernelTarget.NVIDIA_H100,    // → flash_attn_v3
        KernelTarget.NVIDIA_A100,    // → flash_attn_v2
        KernelTarget.GOOGLE_TPU_V6E, // → tpu_pallas_flash_attention
        KernelTarget.AWS_TRAINIUM2,  // → nki_flash_attention
        KernelTarget.CPU_GENERIC,    // → decompose to primitives
    ),
    kvQuant = KvQuantConfig.FP8_PER_HEAD,  // best-effort per target
)
```

The matrix becomes part of `ProgramManifest.backendMatrix`. At job-launch time, `TlalocPodSpecBuilder` (inside vendored Maestro) reads the matrix and translates the row matching the cluster's `(vendor, arch)` into K8s `nodeSelector` labels + accelerator hints. See [`examples/layer3/`](examples/layer3/) for runnable examples and [`docs/xatlib_design.md`](docs/xatlib_design.md) for the design doc.

## Status

| Layer | Focus                                                                                | Status      | Closure spec        |
|-------|--------------------------------------------------------------------------------------|-------------|---------------------|
| Stage A   | DXIR + reverse-mode SCT + K2 handoff                                             | shipped     | §0.4.31             |
| Stage B   | φ-calculus coarsening (F1–F5 + C1–C9) + Symja engine                             | shipped     | §0.4.33             |
| Stage C   | StableHLO + Shardy emission; SOI splice op; coarsening cache                     | shipped     | §0.4.43             |
| Stage D.1 | Brachistochrone full port                                                        | shipped     | §0.4.43             |
| Stage D.2 | HookeanSpring full port                                                          | shipped     | §0.4.47             |
| Stage D.3 | BGDHyperOpt full source port — paper-speedup closure firing                      | shipped     | §0.4.52             |
| Layer 1   | Named indices (`Named<N, A>` + `axisNames` + typed `contract` op)                | shipped     | §0.4.241            |
| Layer 2   | Four worlds + `program { }` + `workflow { }` + `BufferHandle` + Maestro descriptor | shipped     | §0.4.243            |
| Layer 2.5 | Vendored Maestro + first-class `Tlaloc` step type + `SerializedBufferHandle`     | shipped     | §0.4.249            |
| Layer 3   | Pattern recognition + VJP coarsening + kernel registry + cost model + KV-quant + backend matrix + pod-spec | **shipped (closure)** | §0.4.260 |
| Layer 4   | StableHLO emit for COARSENED + sharding-aware custom-calls + cost-driven scheduling + live runtime | not started | —          |

Full suite green at HEAD: **1189 combined tests** (1148 Tlaloc-side + 37 maestro-tlaloc + 4 maestro-common new). See [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) §0.4 for every milestone, [docs/audits/](docs/audits/) for closing audits per layer, and [docs/xatlib_design.md](docs/xatlib_design.md) for the Layer 3 design narrative.

### What's next (Layer 4)

- **StableHLO emit for `OpKind.COARSENED` + `kernel_descriptor`.** Materialize `stablehlo.custom_call @flash_attn_v3 {backend_config={...}}` from the L3-annotated COARSENED. Single emit-path change.
- **Sharding-aware kernel custom-calls.** Plumb SDY mesh axis names through the kernel descriptor's `customCallAttrs` so cross-device attention has a well-typed sharding.
- **Cost-driven scheduling.** Today's `TlalocPodSpecBuilder` picks rows by `(vendor, arch)` exact match; an L4 scheduler can honor cluster availability + cost policy.
- **Multi-pattern coarsening.** Adding `coarsenRmsNorm` / `coarsenRope` / `coarsenCrossEntropy` is each new file + one registry entry.
- **Live K8s integration end-to-end.** Today's L3.6 pod-spec construction is unit-tested; L4 exercises it against a live Maestro instance.

8 audit OQs filed during L3 closure track refinement work — see [`docs/audits/xatlib_kotlin_audit.md`](docs/audits/xatlib_kotlin_audit.md) §15.

## Modules

| path                                                | role                                                                                |
|-----------------------------------------------------|-------------------------------------------------------------------------------------|
| [`core/`](core/)                                    | `DTensor`, shape / dtype types, `Mesh` types, host ops                              |
| [`ir/`](ir/)                                        | DXIR + PhiCalculus + DxirReverseTransform + Symja engine + interpreter + Layer 3 (recognizer / coarsener / kernel / cost / fusion / quant) |
| [`autograd/`](autograd/)                            | runtime tape (fallback when the plugin can't lower a lambda)                        |
| [`stablehlo/`](stablehlo/)                          | StableHLO + Shardy emission; round-trip tests                                       |
| [`compiler-plugin/`](compiler-plugin/)              | K2 plugin: FIR-to-DXIR lowering + IR-generation extension + synthesis               |
| [`maestro/`](maestro/)                              | Layer 2 — `program { }` / `workflow { }` builders, `MaestroStep`, `BufferHandle`, `ProgramManifest` + `BackendTarget` (Layer 3.5) |
| [`benchmarks/`](benchmarks/)                        | JMH benchmarks for the four ported papers (Brachistochrone / HookeanSpring / BGDHyperOpt / etc.) |
| [`third-party/maestro/`](third-party/maestro/)      | Vendored Netflix Maestro + Tlaloc-specific `maestro-tlaloc/` module (Layer 2.5/3.6)  |
| [`examples/`](examples/)                            | Documentation-grade Kotlin snippets (four-worlds, named-indices, layer3)            |

## Requirements

- **JDK 21** (since §0.4.244 — single-version unification with vendored Maestro).
- **Kotlin 2.x** (provided by Gradle wrapper; no host install needed).
- **macOS** (Apple Silicon recommended) or **Linux**. Windows untested.
- **External MLIR toolchains** for `:stablehlo` round-trip tests: `stablehlo-translate`, `sdy-opt`, `iree-compile`. Built from source via the bootstrap scripts; pinned to JAX 0.10.0's bundled commits.

The bootstrap scripts handle Homebrew + Xcode CLT + JDK 21 + the MLIR build:

```bash
# Stage 1 — admin / sudo (you must run; one-time):
bash scripts/setup-mac-bootstrap.sh

# Stage 2 — userspace (sudo-free; ~30–60 min cold, fetches + builds MLIR):
bash scripts/setup-mac-userspace.sh
```

Already on JDK 17? Use the migration helper:

```bash
bash scripts/install-jdk21.sh    # adds openjdk@21 alongside an existing 17 install
```

## Quick start

```bash
# Clone + run the full suite (Tlaloc + vendored Maestro composite build).
git clone https://github.com/pedronahum/tlaloc.git
cd tlaloc
./gradlew test
```

Expected: ~1148 tests passing on the Tlaloc side. To exercise the vendored Maestro tree as well:

```bash
./gradlew test :vendored-maestro:maestro-tlaloc:test
```

Run a single layer's tests:

```bash
./gradlew :ir:jvmTest --tests "io.tlaloc.ir.recognizer.*"      # Layer 3 recognizer + coarsener + kernel + cost
./gradlew :maestro:jvmTest --tests "*BackendMatrixTest*"        # Layer 3.5 backend matrix
./gradlew :compiler-plugin:test --tests "*BGDHyperOptTest*"    # Stage D.3 paper benchmark
./gradlew :stablehlo:jvmTest                                    # StableHLO + Shardy round-trip
```

### Run the examples

Examples are documentation-grade Kotlin files; copy any of them into a fresh project that depends on the relevant Tlaloc modules to run:

| Example dir                                             | Layer | Topic                                                  |
|---------------------------------------------------------|-------|--------------------------------------------------------|
| [`examples/named-indices/`](examples/named-indices/)    | 1     | `Named<N, A>` axes + typed `contract`                  |
| [`examples/four-worlds/`](examples/four-worlds/)        | 2     | `program { }` / `workflow { }`, `BufferHandle` typing  |
| [`examples/layer3/`](examples/layer3/)                  | 3     | recognize+coarsen, per-target kernel matrix, KV-quant, populator |

### Aggregate test count

```bash
bash scripts/count-tests.sh        # sums tests= across all JUnit XMLs after a test run
```

## Build

Single Gradle wrapper drives Tlaloc + vendored Maestro as a composite build. JDK 21 throughout.

```bash
./gradlew test                                         # full Tlaloc-side suite (~30 s warm)
./gradlew :ir:jvmTest                                  # one module
./gradlew :compiler-plugin:test --tests "*BGDHyperOpt*"  # one benchmark
./gradlew :vendored-maestro:maestro-tlaloc:test        # vendored Maestro tests
./gradlew clean build                                  # everything from scratch
```

External toolchains used by `:stablehlo` round-trip tests (`stablehlo-translate`, `sdy-opt`) are located via system properties — see [docs/STAGE_B_PLAN.md](docs/STAGE_B_PLAN.md) for setup. The bootstrap scripts pin them to JAX 0.10.0's bundled commits (stablehlo @ 3a8886de, shardy @ 22259c17).

## Repository tour

| Concern                              | Where                                          |
|--------------------------------------|-----------------------------------------------|
| Per-session ship log                 | [DIFFKTX_SPEC.md](DIFFKTX_SPEC.md) §0.4        |
| Layer 1 closing audit                | [docs/audits/named_indices_audit.md](docs/audits/named_indices_audit.md) |
| Layer 2 closing audit                | [docs/audits/four_worlds_audit.md](docs/audits/four_worlds_audit.md)     |
| Layer 2.5 closing audit              | [docs/audits/maestro_first_class_audit.md](docs/audits/maestro_first_class_audit.md) |
| Layer 3 closing audit                | [docs/audits/xatlib_kotlin_audit.md](docs/audits/xatlib_kotlin_audit.md) |
| Layer 3 design narrative             | [docs/xatlib_design.md](docs/xatlib_design.md)  |
| φ-calculus pass design               | [docs/STAGE_B_PLAN.md](docs/STAGE_B_PLAN.md)    |
| Vendoring philosophy + upgrades      | [docs/vendoring.md](docs/vendoring.md)          |
| Maestro descriptor (deprecated, L2)  | [docs/maestro_descriptor.md](docs/maestro_descriptor.md) |
| Source paper (coarsening AD)         | [docs/papers/coarsening-autodiff.txt](docs/papers/coarsening-autodiff.txt) |
| Symja adequacy bake-off              | [docs/papers/symja-bakeoff-2026-04.md](docs/papers/symja-bakeoff-2026-04.md) |

## Acknowledgments

The φ-calculus coarsening machinery is a direct descendant of the technique in Shen, Shivers, Dea et al., *Efficient, Sound Gradient Descent in Dynamic and Dependent Control Flow* (OOPSLA 2021). The original DiffKt showed Kotlin-native AD can be 10× faster than Python frameworks on scalar benchmarks; Tlaloc extends that result to shape-typed tensors, compile-time transformation, StableHLO/Shardy lowering, and per-target vendor-fused-kernel custom-calls.

The Layer 2.5+ Maestro integration vendors and extends [Netflix Maestro](https://github.com/Netflix/maestro). The single divergence-against-upstream is documented in [docs/audits/xatlib_kotlin_audit.md](docs/audits/xatlib_kotlin_audit.md) §10.
