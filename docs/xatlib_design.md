# Xatlib-style algorithmic-transformation pipeline (Kotlin) — design

> Tlaloc's **Layer 3** (§0.4.250 — §0.4.259) ships a per-pattern recognizer +
> per-(pattern, target) kernel registry + per-target cost model — the
> shape used by algorithmic-transformation libraries on the LLVM/MLIR
> side ("xatlib" being a placeholder name for that genre). This document
> is the design narrative; for the strict per-phase changelog see
> §0.4.250–§0.4.259 in `DIFFKTX_SPEC.md`. For the closing audit see
> `docs/audits/xatlib_kotlin_audit.md`.

## Why a separate transformation layer

Layer 1 added shape + named-axis types. Layer 2 added the four-worlds
taxonomy (Kernel / Orchestration / Program / Cluster scopes), `program
{ }` + `workflow { }`, and `BufferHandle` cross-step typing. Layer 2.5
vendored Netflix Maestro and made `Tlaloc` a first-class step type.

What Layers 1–2.5 *don't* do: take a user-written DXIR function and ask
"is there a better shape we can lower this to?" That's the gap Layer 3
fills.

The relevant decisions live at three different granularities:

1. **Per-region structural recognition** — does this sub-graph match a
   compound idiom (FlashAttention, RMS norm, RoPE, cross-entropy)? If
   yes, what are the inputs / outputs / type structure of the matched
   region?
2. **Per-(pattern, target) kernel selection** — given a recognized
   FlashAttention region and a target device, is there a vendor-fused
   kernel? If yes, emit a `stablehlo.custom_call` to it; if no,
   decompose to primitives.
3. **Per-target cost** — compare (kernel-call vs decompose) decisions
   across multiple potential targets, store the matrix on the manifest,
   let runtime scheduling pick the row matching the actual deployment.

Each granularity is its own pass; each pass is its own file or small
group of files. The upper bound on per-file complexity is "one pattern
or one target's data."

## The five-stage pipeline

```
user DXIR
    │
    │ stage 1 — recognizers (L3.0, L3.1)
    ▼
List<RecognitionMatch>
    │
    │ stage 2 — VJP coarsener (L3.2)
    ▼
DXIR with OpKind.COARSENED envelopes
    │
    │ stage 3 — kernel template lowering (L3.3)
    ▼
DXIR with kernel_descriptor attrs (or decomposed primitives)
    │
    │ stage 4 — KV-quant (L3.4d) + tile fusion (L3.4c)
    ▼
DXIR with kv_quant_config + tile_group attrs
    │
    │ stage 5 — backend matrix population (L3.5)
    ▼
List<BackendTarget> on the manifest
    │
    │ stage 6 — pod-spec construction (L3.6, runtime side)
    ▼
KubernetesCommand for the deployed cluster
```

Stages 1–5 run at compile time on the Kotlin/Tlaloc side; stage 6 runs
at job-launch time inside vendored Maestro.

## Stage 1 — Pattern recognition

Each recognizer is a pure function `recognizeXxx(fn,
diagnostics?): List<RecognitionMatch.Xxx>`. The aggregator
`recognizeAll(fn, diagnostics?)` calls each in turn and runs
`resolveLargestMatch` to strip overlapping smaller matches. v1 ships
four recognizers: FlashAttention, RmsNorm, Rope, CrossEntropy.

Each recognizer:
- Pre-filters on a rare anchor op (SOFTMAX / RSQRT / SIN / LOG) — a
  cheap walk over the function body.
- Validates structural invariants (operand kinds, consumer kinds,
  shape consistency).
- Emits a typed `RecognitionMatch.Xxx` data record carrying the matched
  ops + per-pattern structural metadata (Q/K/V leaves, score type,
  output type, etc.).
- On a near-miss (pre-filter fired but full pattern didn't match),
  appends a `RecognitionDiagnostic(pattern, reason, opId)` to the
  optional diagnostics collector. IDE tooling can surface "you almost
  wrote attention here; here's what's missing."

Adding a recognizer = new file + one line in `RecognizeAll.kt`. Per the
L3 charter ("100-line file, no build-system change").

## Stage 2 — VJP coarsening

The coarsener takes a `RecognitionMatch` and rewrites the function so
the matched sub-graph is absorbed into a single `OpKind.COARSENED` op.
This op carries three attrs:

- `primal_body: DxirFunction` — the simplified primal (recapitulates
  the recognized region's forward computation).
- `gradient_body: DxirFunction` — the analytical VJP, signature
  `(K upstreams, N primal operands) → N grads`.
- `reads_primal_indices: Set<Int>` — which primal-operand indices the
  gradient body actually dereferences (the existing
  `DxirReverseTransform.handleCoarsenedAdjoint` consumes this to know
  which operand subgraphs need cloning into gradient scope).

The COARSENED op type was added in §0.4.31 for the Stage C SOI splice;
L3.2 reuses it for analytical-VJP envelopes. No new opkind needed.

For FlashAttention specifically, the gradient body recomputes
`S = MATMUL(Q, K)` and `P = SOFTMAX(S)` internally and applies the
standard chain rule:

```
dV = MATMUL(P^T, dO)
dP = MATMUL(dO, V^T)
dS = MUL(SUB(dP, BROADCAST_back(SUM(MUL(P, dP), last_axis))), P)
dQ = MATMUL(dS, K^T)
dK = MATMUL(Q^T, dS)
```

A future variant could thread softmax statistics out of the COARSENED
(saving one MATMUL + one SOFTMAX per backward); v1 ships the simpler
recompute form.

## Stage 3 — Kernel templates

`KernelTemplate.pickFor(coarsened, target): KernelDescriptor?` chooses
between (a) a vendor-fused kernel (returns a populated `KernelDescriptor`)
and (b) decompose (returns `null`).

The driver `lowerKernelChoice(fn, target, registry)` walks COARSENED
ops, looks up the per-pattern template by name, calls `pickFor`, and:
- **Annotated path**: re-emits the COARSENED with `kernel_descriptor`
  added to its attrs. Downstream StableHLO emit reads the attr and
  materializes a `stablehlo.custom_call @<kernelName>`.
- **Decompose path**: replaces the COARSENED with the inlined
  `primal_body`. The function shape after decomposition is the
  original user code; running the recognizer again would re-find the
  same pattern (idempotent).

### Why annotate the existing COARSENED, not wrap in MANUAL_COMPUTATION

The original L3 plan called for "custom-call envelopes via
`OpKind.MANUAL_COMPUTATION` reuse." In Tlaloc today,
`MANUAL_COMPUTATION` is reserved for Shardy's `sdy.manual_computation`
— the existing emitter at `stablehlo/Emitter.kt:1131-1135` requires
`in_shardings` / `out_shardings` / `manual_axes` attrs. Reusing the op
for kernel custom-calls would require extending the SDY emitter to
dispatch on attrs.

Annotating the existing COARSENED with a `KernelDescriptor` attr
achieves the same separation (pattern-recognized + kernel-tagged
sub-graphs are distinguishable at lowering time) without the emitter
change. Tracked as audit OQ-Layer3-1: "evaluate adding a dedicated
`OpKind.KERNEL_CALL` once a second backend (IREE, Triton) needs the
dispatch shape."

## Stage 4a — Tile fusion

Identifies maximal connected sets of elementwise ops sharing an output
shape (the "unrecognized residue" around L3.2's coarsened compounds)
and annotates each candidate op with a `tile_group: <Int>` attr.
Downstream codegen consumes the attr to choose tile-loop boundaries.

Algorithm: union-find on the elementwise sub-graph, where edges are
producer→consumer relationships and same-shape is the union predicate.
Components of size ≥ 2 become tile groups.

What's NOT fused (v1):
- Reductions in the middle of an elementwise chain (need a two-pass
  schedule).
- Cross-COARSENED chains (the fusion pass walks the outer body only).
- Cross-shape "broadcast fusion."

## Stage 4b — KV-cache quantization

Annotates COARSENED ops with `kv_quant_config: KvQuantConfig` when the
target's kernel descriptor advertises support for the requested dtype
(via `customCallAttrs["supported_kv_dtypes"]`).

Best-effort: a request that isn't supported on a target silently skips
with a structured diagnostic. Heterogeneous deployment is first-class —
the same FP8 request flows across 7 targets, the diagnostic stream
reports the per-target accept/decline.

Why metadata-only: Tlaloc's `DType` enum has F32/F64/I32/I64/Bool — no
native int8 / fp8. A type-system extension for low-precision dtypes
would need to thread through the FIR plugin, the StableHLO emitter,
the interpreter, and the manifest schema (multi-week work). KV-quant
for v1 ships as a *codegen directive*: the COARSENED carries an attr
telling the kernel custom-call "materialize K and V as `<dtype>` at
runtime; perform on-the-fly dequantization inside the kernel." DXIR
types stay F32. Tracked as OQ-Layer3-2.

## Stage 5 — Backend matrix population

`populateBackendMatrix(fn, targets, kvQuant?): List<BackendTarget>`
walks the recognized + coarsened + (per-target) lowered + KV-quanted +
cost-estimated function and projects the result into a structured tuple
per target.

Each `BackendTarget` row carries:
- `vendor: String` — e.g. `"nvidia"`.
- `arch: String` — e.g. `"h100"`.
- `kernelName: String?` — the picked kernel name, or null for decompose.
- `kvQuantDtype: String?` — KV-quant dtype tag, or null when not applied.
- `costMicroseconds: Double?` — roofline-style time estimate.

The matrix lives on `ProgramManifest.backendMatrix` (extended in §0.4.258
from the §0.4.243 placeholder `List<String>` to `List<BackendTarget>`).
It serializes via the existing hand-rolled JSON parser; round-trip
integrity tests pin the format.

The populator is **opt-in** — `Tlaloc.program { }` doesn't call it
automatically. Running the L3 pipeline per (vendor × arch × kv-quant)
combo at trace time would balloon trace cost. Callers that want a
populated matrix invoke `populateBackendMatrix(...)` explicitly with
the list of targets they care about.

## Stage 6 — Pod-spec construction (runtime side)

The runtime side (`TlalocPodSpecBuilder` + `TlalocStepRuntime` in
`maestro-tlaloc/`) consumes the manifest's `backendMatrix` JSON at
job-launch time. Selection:

1. Exact `(vendor, arch)` match wins.
2. No exact match → lowest-cost row matching vendor.
3. No vendor match → absolute lowest-cost row.
4. Empty/malformed matrix → no modification, command passes through.

Translation: the picked row's `vendor` + `arch` populate K8s
nodeSelector labels (per-vendor templates in `TlalocPodSpecBuilder.kt`),
the row's `kernelName` + `kvQuantDtype` populate the `accelerators`
observability map, and `gpu="1"` is set for any non-CPU target.

The one vendoring divergence (`KubernetesCommand` gains `nodeSelector`
+ `accelerators` Map fields) is `@JsonInclude(NON_NULL)` — vanilla
Netflix Maestro consumers see no change in JSON shape when these
fields are unset.

## Cost model

Roofline-style: `time = max(flops / peak, bytes / bandwidth)`. Per-op
formulas use textbook approximations (matmul = 2·m·k·n; elementwise =
N; softmax = 5·N).

Calibrated for: relative ordering (H100 < A100 < CPU), AI ordering
(fused > unfused), per-target compute density (TPU v6e most
compute-dense, CPU most memory-bound).

Not calibrated for: absolute latency. The cost model is a *relative*
estimator. Vendor-specific micro-models (cudnn dispatch overhead, TPU
MXU utilization curves, NVLink/ICI/EFA effects) are L4+ work. Tracked
as OQ-Layer3-7.

The seven canonical device descriptors carry datasheet-cited peak
FLOPs / HBM bandwidth / on-chip SRAM / SM-or-core counts. Citations
live as comments in `DeviceDescriptor.kt:120-260`.

## What L3 doesn't do

- **No StableHLO emit.** The pipeline ends at the
  annotated-and-coarsened DXIR; the actual `stablehlo.custom_call`
  materialization is L4 codegen work.
- **No interpreter run for COARSENED gradient bodies.** The bodies use
  ops the v1 interpreter doesn't yet support (rank-(r-1) → rank-r
  BROADCAST). Tests verify shape, not numerics.
- **No live K8s integration.** L3.6's pod-spec construction is unit
  tested for parser correctness + selection; live-Maestro-with-K8s
  exercise is L4.
- **No multi-result COARSENED.** Each pattern v1 emits one tensor.
- **No cross-COARSENED tile fusion.** The pass walks the outer body
  only.
- **No automatic dtype selection** for KV-quant. The user explicitly
  picks the dtype; the pass is best-effort about honoring it.

## Adding to the pipeline

| To add | Touch |
|--------|-------|
| A new recognizer | New file in `ir/recognizer/` + one line in `RecognizeAll.kt` |
| A new coarsener | New file in `ir/recognizer/coarsener/` + one entry in `defaultCoarseners` |
| A new kernel template | New file in `ir/recognizer/kernel/` + one entry in `defaultKernelTemplates` |
| A new device descriptor | One entry in `DeviceDescriptors` object + one citation comment |
| A new vendor's nodeSelector pattern | One entry in `TlalocPodSpecBuilder.VENDOR_TEMPLATES` |

No build-system changes for any of these; no DI wiring; no annotation
processor; no service-loader.

## Reading order for new contributors

1. `ir/recognizer/RecognitionMatch.kt` — the typed match record shape.
2. `ir/recognizer/FlashAttentionRecognizer.kt` — the exemplar
   recognizer. Read alongside `examples/layer3/` (§0.4.485 rebuilt that example
   as a standalone runnable project; its `src/main/kotlin/Main.kt` walks
   recognize → coarsen → emit and prints each stage).
3. `ir/recognizer/coarsener/FlashAttentionCoarsener.kt` — the analytical
   VJP for attention.
4. `ir/recognizer/kernel/KernelTemplate.kt` + `FlashAttentionKernel.kt`
   — the per-target dispatch surface.
5. `ir/recognizer/cost/CostModel.kt` — per-op cost formulas + roofline
   time estimator.
6. `ir/recognizer/quant/KvQuantApply.kt` — best-effort KV-quant
   semantics.
7. `maestro/jvmMain/.../BackendMatrixPopulator.kt` — the cross-layer
   bridge.
8. `third-party/maestro/maestro-tlaloc/.../TlalocPodSpecBuilder.java`
   — the runtime-side pod-spec selector.

The `xatlib_kotlin_audit.md` (closing audit, §0.4.260) is the
17-section verification document for the whole layer.
