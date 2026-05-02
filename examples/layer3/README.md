# Layer 3 examples (§0.4.250–§0.4.260)

Reference snippets demonstrating Tlaloc's Layer 3 algorithmic-transformation
pipeline: pattern recognition → VJP coarsening → kernel template registry →
cost model → KV-quant → backend-matrix population → K8s pod-spec construction.

These files are documentation-grade — copy into a project that depends on
`io.tlaloc:ir` (and `io.tlaloc:maestro` for the populator example) to run.

| File | Purpose |
|------|---------|
| [`RecognizeAndCoarsenAttentionExample.kt`](RecognizeAndCoarsenAttentionExample.kt) | The smallest end-to-end loop — recognize a `MATMUL → SOFTMAX → MATMUL` shape, coarsen it into a single `OpKind.COARSENED` op carrying the analytical primal_body + gradient_body. |
| [`KernelLoweringMatrixExample.kt`](KernelLoweringMatrixExample.kt) | Walks the seven canonical device targets and shows the per-target lowering decision (fused vendor kernel vs decompose). |
| [`KvQuantHeterogeneousExample.kt`](KvQuantHeterogeneousExample.kt) | Best-effort FP8 KV-quant across a heterogeneous deployment — H100 / Trainium2 accept, A100 / TPU v5e decline with a structured diagnostic. |
| [`PopulateBackendMatrixExample.kt`](PopulateBackendMatrixExample.kt) | Run the L3.5 populator across all seven targets, dump the cost-ordered matrix as JSON. The format the runtime side (TlalocPodSpecBuilder) consumes. |

## How the pieces connect

```
user code (DXIR)
    │
    │  L3.0 / L3.1 — recognizeAll(fn)
    ▼
List<RecognitionMatch>
    │
    │  L3.2 — coarsenRecognizedPatterns(fn, matches)
    ▼
DxirFunction with OpKind.COARSENED ops
    │
    │  L3.3 — lowerKernelChoice(fn, target)
    ▼
DxirFunction with kernel_descriptor attrs (or decomposed primitives)
    │
    │  L3.4d — applyKvQuant(fn, KvQuantConfig.FP8_PER_HEAD)
    ▼
DxirFunction with kv_quant_config attrs (best-effort)
    │
    │  L3.5 — populateBackendMatrix(fn, targets, kvQuant)
    ▼
ProgramManifest.backendMatrix: List<BackendTarget>
    │
    │  Wire transport (JSON inside workflow params)
    ▼
TlalocStepRuntime + TlalocPodSpecBuilder (L3.6, vendored Maestro)
    │
    │  Picks row matching cluster (vendor, arch)
    ▼
KubernetesCommand with nodeSelector + accelerators + gpu populated
    │
    ▼
Maestro launches the K8s job on the right hardware
```

## What's NOT in these examples

- **No StableHLO emit yet.** The L3 pipeline ends at the annotated-and-coarsened DXIR; the actual `stablehlo.custom_call` materialization is post-L3 codegen work. These examples exercise the structural decisions, not the emitted bytes.
- **No interpreter run.** The COARSENED op's gradient_body uses ops the v1 interpreter doesn't yet support (rank-(r-1) → rank-r BROADCAST). Examples verify shape, not numerics.
- **No live K8s deployment.** `PopulateBackendMatrixExample` produces the matrix as JSON; the runtime example would need a live Maestro instance + cluster, which is L4+ scope.
