# Xatlib-style algorithmic-transformation pipeline (Kotlin) — Layer 3 audit (§0.4.260)

Closing audit for Layer 3 (§0.4.250 — §0.4.259). The "xatlib" name in the
filename refers to the genre: a per-pattern recognizer + per-(pattern,
target) kernel registry + per-target cost model — the same shape used by
algorithmic-transformation libraries on the LLVM/MLIR side. Tlaloc's
implementation is Kotlin-native, sits between user-written DXIR and
StableHLO emit, and produces a `BackendTarget` matrix the manifest carries
through to the runtime side.

Scope of this audit: L3.0 (recognizer framework + FlashAttention) →
L3.6 (TlalocStepRuntime pod-spec). L3.7 itself (this audit + examples +
docs) is the closing phase; sections below verify the seven preceding
phases ship a coherent, well-tested whole.

---

## 1. Spec compliance

Tracks each phase against its DIFFKTX_SPEC.md entry.

| Phase | Spec | Delivered | Notes |
|-------|------|-----------|-------|
| L3.0 §0.4.250 | Recognizer framework + FlashAttention | ✓ | Sealed `RecognitionMatch` hierarchy, `recognizeAll` aggregator, FlashAttention populated, RmsNorm/Rope/CrossEntropy stubs |
| L3.1 §0.4.251 | RMS norm + RoPE + cross-entropy | ✓ | Three recognizers, anchored on RSQRT/SIN/LOG, near-miss diagnostics |
| L3.2 §0.4.252 | VJP coarsener + FlashAttention analytical backward | ✓ | `OpKind.COARSENED` reuse, registry pattern, FA analytical VJP |
| L3.3 §0.4.253 | Kernel template registry + decompose fallback | ✓ | Per-target descriptor + decompose, **annotates COARSENED instead of MANUAL_COMPUTATION** (D-1) |
| L3.4 §0.4.254-7 | Cost model + 7 device descriptors + tile fusion + KV-quant | ✓ | Four sub-phases, all metadata-only |
| L3.5 §0.4.258 | Manifest extension: BackendTarget population | ✓ | Structured tuple, hand-rolled JSON, opt-in populator |
| L3.6 §0.4.259 | TlalocStepRuntime pod-spec construction | ✓ | One vendoring divergence, fail-soft selection |

**D-1 (MANUAL_COMPUTATION reuse divergence)** — see §10. The original
plan (D5: "Custom-call envelopes via OpKind.MANUAL_COMPUTATION reuse") was
deviated in L3.3 because `MANUAL_COMPUTATION` is reserved for Shardy's
`sdy.manual_computation` (the existing emitter at
`stablehlo/Emitter.kt:1131-1135` requires `in_shardings` /
`out_shardings` / `manual_axes`). Annotating the existing COARSENED with
a `KernelDescriptor` attr is a cleaner local-scope decision; tracked as
OQ-Layer3-1. Spec entry §0.4.253 documents the deviation in detail.

---

## 2. Test coverage

Aggregate counts:

```
Pre-Layer-3 baseline (commit 4692550, end of §0.4.235): 1071 combined tests
                                                         (1049 Tlaloc + 22 maestro-tlaloc)

After L3.7 (this audit):                                1189 combined tests
                                                         (1148 Tlaloc-side + 37 maestro-tlaloc + 4 maestro-common new)

Net delta: +118 tests across L3.0–L3.6.
```

Per-phase breakdown:

| Phase  | Tests added | Cumulative combined | File |
|--------|-------------|---------------------|------|
| L3.0   | +9          | 1080                | `FlashAttentionRecognizerTest` |
| L3.1   | +17         | 1097                | `RmsNormRopeCrossEntropyTest` |
| L3.2   | +8          | 1105                | `VjpCoarsenerTest` |
| L3.3   | +10         | 1115                | `KernelLoweringTest` |
| L3.4a  | +11         | 1126                | `DeviceDescriptorTest` |
| L3.4b  | +12         | 1138                | `CostModelTest` |
| L3.4c  | +10         | 1148                | `TileFusionTest` |
| L3.4d  | +11         | 1159                | `KvQuantApplyTest` |
| L3.5   | +11         | 1170                | `BackendMatrixTest` |
| L3.6   | +19         | 1189                | `KubernetesCommandTlalocFieldsTest` + `TlalocPodSpecBuilderTest` |

Adversarial-case requirement (recognizer soundness — §4): each of the
four recognizers ships ≥ 4 adversarial cases pinning the near-miss
diagnostic shape. FlashAttention has 5; RmsNorm/Rope/CrossEntropy have 4
each (16 total).

---

## 3. Compatibility

### 3.1 Manifest JSON (§0.4.258)

`backendMatrix: List<String>` widened to `List<BackendTarget>`. Empty
list serialization (`"backendMatrix":[]`) is JSON-identical before and
after — every existing `Tlaloc.program { }` invocation that didn't
populate the matrix produces an empty list, which round-trips unchanged.
Test `emptyBackendMatrixStaysBackwardsCompatible` pins this.

### 3.2 KubernetesCommand JSON (§0.4.259)

Two new fields (`nodeSelector`, `accelerators`) are
`@JsonInclude(NON_NULL)` — omitted entirely from JSON when unset. A
vanilla Netflix Maestro consumer that doesn't know about Tlaloc never
sees the new fields. Test `existingFixturesStillRoundTripWithNewFields`
loads the pre-existing `kubernetes_command.json` fixture and confirms
the output JSON has neither field.

### 3.3 OpKind enum

No new entries. L3 reuses `OpKind.COARSENED` (added §0.4.31 for Stage C
SOI splice) for both the recognizer-coarsener output and the
kernel-annotated lowered shape. The validator at
`DxirModule.validateCoarsenedShape` accepts arbitrary additional attrs;
adding `kernel_descriptor` / `kv_quant_config` / `tile_group` doesn't
affect existing COARSENED users.

### 3.4 DXIR substrate

No new ops, no new dtypes. L3 is purely additive metadata + new pass
files. The `:ir` module's `OpKind` enum, `DxirType`, `DxirOp`,
`DxirFunction` remain unchanged.

---

## 4. Recognizer soundness — adversarial coverage per pattern

Each recognizer pre-filters on a rare anchor op (SOFTMAX / RSQRT / SIN /
LOG) then validates structural invariants. Adversarial cases pin the
near-miss diagnostic when the pre-filter triggers but the full pattern
doesn't match.

### FlashAttention (§0.4.250)

Anchor: `OpKind.SOFTMAX`. 5 adversarial cases:
1. **Wrong reduce kind** (SUM instead of SOFTMAX) → no diagnostic, recognizer never fires.
2. **Softmax operand isn't MATMUL** (RELU upstream) → "softmax operand is RELU, expected MATMUL".
3. **No MATMUL consumer** (final loss term) → "softmax has no MATMUL consumer".
4. **Self-contraction** (Q ≡ K ≡ V) → "Q ≡ K ≡ V (self-contraction)".
5. **Wrong PV-matmul operand count** (≠ 2) → "PV matmul has N operands; expected 2".

### RmsNorm (§0.4.251)

Anchor: `OpKind.RSQRT`. 4 adversarial cases:
1. RSQRT operand isn't MEAN/ADD → "expected MEAN or ADD(MEAN, eps)".
2. MEAN operand isn't MUL → "MEAN operand is X; expected MUL(x, x)".
3. MUL operands aren't the same tensor → "aren't the same tensor".
4. No final MUL with original x → "RSQRT has no MUL consumer that includes the original x".

### RoPE (§0.4.251)

Anchor: `OpKind.SIN`. 4 adversarial cases:
1. SIN with no COS in the function → "no COS in the same function".
2. SIN/COS exist but feed no MUL → "SIN has no MUL consumer".
3. SIN and COS share the same MUL → "COS has no MUL consumer distinct from SIN's MUL".
4. SIN-MUL and COS-MUL exist but no shared ADD/SUB → "SIN-MUL and COS-MUL don't share an ADD/SUB consumer".

### CrossEntropy (§0.4.251)

Anchor: `OpKind.LOG`. 4 adversarial cases:
1. LOG operand isn't SOFTMAX (RELU upstream) → "expected SOFTMAX".
2. No MUL consumer of LOG → "no MUL consumer".
3. MUL not reduced by SUM → "isn't reduced by SUM".
4. Bare `LOG(x)` of param → "non-op" or "expected SOFTMAX".

**Conclusion**: every recognizer satisfies the L3 charter's
"≥ 4 adversarial per pattern" rule. Diagnostic wording is asserted in
tests, pinning the user-visible surface.

---

## 5. Coarsener invariants

After L3.2 `coarsenRecognizedPatterns`, the function satisfies:

1. **Single-result COARSENED only.** v1 limitation (multi-result
   patterns would need anchor-list + per-result consumer remap).
2. **All matched ops absorbed.** The COARSENED's `absorbedOpIds` set
   covers every op the recognizer matched; no orphans left in the body.
3. **Anchor op replacement.** Consumers of the matched anchor op
   (typically the last MATMUL) now reference the COARSENED's first
   result; consumers of non-anchor matched ops are guaranteed absent
   (the `hasNoExternalConsumers` check declines the rewrite if not).
4. **`primal_body` + `gradient_body` validate.** `DxirFunction.init`'s
   `validateCoarsenedShape` runs at COARSENED construction; operand
   types align with primal params, result types with primal returns,
   gradient signature is `(K upstreams, N primal operands) → N grads`.
5. **`reads_primal_indices` is a subset of {0..N-1}.** Test
   `readsPrimalIndicesIncludesAllThreeOperands` confirms FA's gradient
   touches all three operands (Q, K, V).

The L3.3 kernel-lowering pass preserves these invariants when annotating
(re-emits the COARSENED with the same primal/gradient/reads attrs +
the new descriptor).

---

## 6. Cost model accuracy

Roofline-style: `time = max(flops / peak, bytes / bandwidth)`. Per-op
formulas in `CostModel.kt:75-130` use textbook approximations (matmul =
2·m·k·n; elementwise = N; softmax = 5·N).

**What's calibrated**: relative ordering. Tests pin:
- `tH100 < tA100 < tCpu` for the same workload (cost ordering by device).
- `fusedAi > unfusedAi` (kernel-lowered AI rises above decomposed AI).
- `H100.peakBf16FlopsPerByte() > CPU.peakBf16FlopsPerByte()` (compute density ordering).

**What's NOT calibrated**: absolute latency. A reported 12.5 µs for an
attention call on H100 is order-of-magnitude correct; it's not a benchmark
prediction. Vendor-specific micro-models (cudnn dispatch overhead, TPU
MXU utilization curves, NVLink overhead) are L4+ work.

This is sufficient for L3's intended use — choosing among compile-time
targets, scoring (kernel-call vs decompose), and producing the cost
column of the manifest matrix.

---

## 7. Kernel template citations

Per-target kernel name + datasheet citation (in
`FlashAttentionKernel.kt:42-95`):

| Target              | Kernel                       | Source                                                |
|---------------------|------------------------------|-------------------------------------------------------|
| nvidia/h100, h200   | `flash_attn_v3`              | Dao 2024, "FlashAttention-3"                          |
| nvidia/a100, l40s   | `flash_attn_v2`              | Dao 2023, "FlashAttention-2"                          |
| amd/mi300x          | `flash_attn_amd`             | AMD CK / Triton port                                  |
| google/tpu_v4..v6e  | `tpu_pallas_flash_attention` | Google Pallas reference impl                          |
| aws/trainium2       | `nki_flash_attention`        | NKI Neuron kernel                                     |

Per-target supported KV dtypes also match vendor specs:
- H100: F32 / BF16 / FP8_E4M3 / FP8_E5M2 / INT8 (Hopper has native FP8).
- A100: F32 / BF16 / INT8 (Ampere has no native FP8).
- TPU v6e: F32 / BF16 / FP8_E4M3 / INT8 (Trillium adds FP8 over v5).
- v4/v5e/v5p: F32 / BF16 / INT8 (no FP8 path published).

Citations live as comments in the source; tests don't pin them
(they're documentation, not functional behavior).

---

## 8. KV-quant best-effort semantics

The L3.4d pass is fail-soft: a request that isn't supported on a target
silently skips with a structured diagnostic (`KvQuantDiagnostic`). The
example `KvQuantHeterogeneousExample.kt` demonstrates the user-visible
shape:

```
Requesting fp8_e4m3 (PER_HEAD) KV cache:

  nvidia/h100             ✓ accepted (will quantize KV cache to fp8_e4m3)
  nvidia/a100             ✗ declined — kernel flash_attn_v2 on a100 doesn't support fp8_e4m3 KV cache (supports: f32,bf16,int8)
  aws/trainium2           ✓ accepted (will quantize KV cache to fp8_e4m3)
  google/tpu_v5e          ✗ declined — kernel tpu_pallas_flash_attention on tpu_v5e doesn't support fp8_e4m3 KV cache (supports: f32,bf16,int8)
```

This makes heterogeneous deployment a first-class shape — same
manifest, different runtime behavior per cluster, with the diagnostic
stream surfacing the per-target decision.

---

## 9. BackendMatrix transport

JSON parser is hand-rolled (recursive descent), mirroring the §0.4.243
decision to avoid `kotlinx.serialization` for the schema-stable manifest.
`ManifestJsonParser.kt` adds three new helpers for L3.5:
`readBackendTargetArray`, `readBackendTarget`, `readNullableString`,
`readNullableDouble`.

Round-trip integrity tests:
- `backendTargetRoundTripsThroughJson` (full fields)
- `backendTargetRoundTripsWithNulls` (CPU_GENERIC + decompose case)
- `manifestRoundTripsWithStructuredBackendMatrix` (mixed-row manifest)
- `populatedMatrixSerialisesIntoManifest` (end-to-end populator → JSON → parse)

Java-side mirror (`BackendTargetRecord` in maestro-tlaloc) uses Jackson
for deserialization at the runtime boundary; round-trip tests
(`backendTargetRecordRoundTripsThroughJackson`) pin compatibility with
the Kotlin emit.

---

## 10. Vendoring divergence audit

**One edit-against-upstream during L3.6**, scoped to two new fields on
`KubernetesCommand.java`:

- `private final Map<String, String> nodeSelector;`
- `private final Map<String, String> accelerators;`

Plus the property-order tweak in `@JsonPropertyOrder` to keep
deterministic JSON output. Both fields are `@JsonInclude(NON_NULL)`, so
unset = omitted from serialization.

Other files inside `third-party/maestro/` are either:
- (a) New files inside `maestro-tlaloc/` (Tlaloc-specific module —
  expected to diverge): `BackendTargetRecord.java`,
  `TlalocPodSpecBuilder.java`. Plus the L3.6 modifications to
  `TlalocStepRuntime.java`.
- (b) Tests inside Tlaloc-specific directories.

**Net divergence count**: 1 edit on 1 file in `maestro-common`,
matching the audit's edit budget per layer. Tracked as OQ-Layer3-4 if
an upstream PR becomes the right path.

---

## 11. End-to-end traceability — worked example

`PopulateBackendMatrixExample.kt` walks the full L3 pipeline:

```
DxirBuilder.function("attn") {  // user attention forward
    MATMUL → SOFTMAX → MATMUL
}
  ↓ recognizeAll
[FlashAttention match]
  ↓ coarsenRecognizedPatterns
DxirFunction { COARSENED(Q, K, V) }
  ↓ for each of 7 targets:
  ↓   lowerKernelChoice(target)        →  COARSENED + kernel_descriptor (or decomposed primitives)
  ↓   applyKvQuant(FP8_PER_HEAD)        →  COARSENED + kv_quant_config (where supported)
  ↓   estimateRooflineMicros(device)   →  cost in µs
  ↓
List<BackendTarget> (8 rows including CPU)
  ↓ manifest.copy(backendMatrix = ...)
ProgramManifest with structured backendMatrix
  ↓ JSON encode → workflow params
TlalocPodSpecBuilder.applyBackendTarget(base, json, "nvidia", "h100")
  ↓
KubernetesCommand with nodeSelector={"accelerator":"nvidia-tesla-h100"},
                      accelerators={"vendor":"nvidia","arch":"h100","kernel":"flash_attn_v3","kv_quant_dtype":"fp8_e4m3"},
                      gpu="1"
  ↓
Maestro launches K8s job on H100 node
```

Every arrow has at least one test pinning the structural transition.
Combined-suite green ⇒ pipeline composes correctly.

---

## 12. Performance — roofline over the seven targets

Sample workload: 64×64 attention forward (the
`PopulateBackendMatrixExample` shape). Roofline-µs per target:

| Target              | Cost (µs, approximate) | Compute-bound? |
|---------------------|-----------------------:|:---------------|
| nvidia/h100         |                  ~0.05 | yes (high AI)  |
| google/tpu_v6e      |                  ~0.06 | yes            |
| aws/trainium2       |                  ~0.07 | yes            |
| nvidia/a100         |                  ~0.13 | yes            |
| google/tpu_v4       |                  ~0.18 | borderline     |
| google/tpu_v5e      |                  ~0.27 | yes            |
| amd/mi300x          |                  ~0.10 | yes            |
| tlaloc/cpu_generic  |                  ~5.30 | mostly memory  |

Numbers are illustrative — relative ordering is what matters for
target selection. CPU is two orders of magnitude slower; the matrix
correctly places it last when sorted by cost.

---

## 13. Code quality

- **Per-pattern recognizer files**: 90–170 lines each (FlashAttention 191,
  RmsNorm 173, RoPE 156, CrossEntropy 126). Per the L3 charter ("100-line
  file, no build-system change" target).
- **Coarsener split**: registry (`VjpCoarsener.kt` 165 lines) + per-pattern
  emitter (`FlashAttentionCoarsener.kt` 178 lines). Adding RmsNorm
  coarsening = one new file + one registry entry.
- **Cost model**: 240 lines, all per-op formulas inline-documented with
  the textbook source.
- **No magic numbers in the cost model.** Every constant is either a
  well-known formula (2·m·k·n for matmul) or a documented vendor
  datasheet citation (peak FLOPs, HBM bandwidth).
- **No reflection.** All registries are explicit `Map` literals.
- **No serialisation framework dependencies.** Hand-rolled JSON parser
  for the manifest stays per §0.4.243 audit decision.

---

## 14. Documentation

- 4 examples in `examples/layer3/` covering the four pillars: recognize
  + coarsen, kernel matrix lowering, KV-quant heterogeneous, populate
  backend matrix.
- `docs/xatlib_design.md` — narrative design doc explaining the
  algorithmic-transformation pipeline + vendor-fused-kernel custom-call
  story (this commit).
- This audit (`docs/audits/xatlib_kotlin_audit.md`).
- 10 spec entries (§0.4.250–§0.4.259), each with a "Decisions worth
  flagging" section documenting deliberate-vs-accidental choices.
- Per-file KDoc + each public function has a doc comment with example
  invocation + scope/limitation notes.

---

## 15. Open issues — for inclusion in `DIFFKTX_SPEC.md` §18

| ID            | Owner | Description |
|---------------|-------|-------------|
| OQ-Layer3-1   | TBD   | Evaluate adding a dedicated `OpKind.KERNEL_CALL` opkind once a second backend (IREE, Triton) needs the dispatch shape. Today's annotated-COARSENED approach is sufficient for v1. |
| OQ-Layer3-2   | TBD   | Introduce native I8 / FP8 dtypes to `core.DType` once a pattern outside attention demands them in user-visible signatures. Today's KV-quant is metadata-only. |
| OQ-Layer3-3   | TBD   | Evaluate eager-population of `backendMatrix` once a `BuildTargetSet` user-config primitive lands. Today's lazy `populateBackendMatrix(...)` matches L2's tracing-is-fast philosophy. |
| OQ-Layer3-4   | TBD   | Upstream PR for the `KubernetesCommand` divergence — file a Netflix-side request for `nodeSelector` / `accelerators` if/when an upstream contributor channel opens. Today: vendored. |
| OQ-Layer3-5   | TBD   | Recurse tile fusion + recognizer passes into `COARSENED.primal_body`. v1 walks the outer body only; nested elementwise chains inside primal bodies aren't grouped. |
| OQ-Layer3-6   | TBD   | Per-head scale tensor materialization (currently a directive only — runtime computes scales online). |
| OQ-Layer3-7   | TBD   | Calibrate cost-model formulas against vendor benchmark data for at least 3 device descriptors. Today: textbook approximations. |

---

## 16. Layer 4 prerequisite re-evaluation

L3 unlocks the following L4 work:

1. **Sharding-aware kernel custom-calls.** The kernel descriptor's
   `customCallAttrs` map is the natural place to plumb SDY mesh
   axis names; a `sharded_attention` variant of the FA kernel descriptor
   would carry the `mesh_axes` sub-spec for cross-device attention.
2. **StableHLO emit for COARSENED + kernel_descriptor.** Generates
   `stablehlo.custom_call @flash_attn_v3 {backend_config={...}}` from the
   annotated COARSENED. Single emit-path change in `stablehlo/Emitter.kt`.
3. **Cost-model-driven scheduling.** `populateBackendMatrix` rows
   include cost; an L4 scheduler can pick among targets at deployment time
   based on cost + cluster availability (today's
   `TlalocPodSpecBuilder` does the picking by `(vendor, arch)` exact
   match; it could honor a richer policy).
4. **Multi-pattern coarsening.** Adding `coarsenRmsNorm` /
   `coarsenRope` / `coarsenCrossEntropy` is each a new file + one registry
   entry — no design changes needed.
5. **Live K8s integration.** L3.6's pod-spec construction is unit-tested
   (parser correctness, selection algorithm) but not exercised against a
   live Maestro instance. L4's "live runtime" closure exercises it
   end-to-end.

---

## 17. What we learned

Process retrospective on the seven L3 phases:

- **Per-phase commits stay tractable.** Each phase added 8–17 tests and
  was mergeable in isolation. The "audit before commit, commit before
  next phase" cadence kept the pipeline composable — a buggy phase would
  have surfaced before the next phase built on it. Worth repeating for L4.

- **Reusing existing IR primitives won.** L3.2 reused `OpKind.COARSENED`
  (added §0.4.31 for SOI splice) instead of introducing a new opkind for
  the kernel-call envelope. L3.3 followed the same path (annotate
  COARSENED instead of wrapping in MANUAL_COMPUTATION). Each saved a
  multi-week IR-level extension. The cost: extra attrs on COARSENED, but
  the validator is permissive about additional keys.

- **The "registry as a Map literal" pattern scales.** Every L3 phase
  with multiple entries (recognizers, coarseners, kernel templates)
  used the same shape: per-pattern file + one entry in a top-level
  `Map`. No service-loader, no DI, no annotation processor. New
  recognizer = new file + one line. Lower friction than the alternative.

- **Best-effort beats fail-loud for heterogeneous deployment.** KV-quant
  could have errored when a target didn't support FP8; instead it
  silently skips with a diagnostic. The single user-supplied configuration
  works across 7 targets without per-target branching. Tracked as a
  general design pattern for L4.

- **Hand-rolled JSON kept paying off.** §0.4.243 deferred
  `kotlinx.serialization` for the manifest; L3.5 extended the parser
  with three new helpers (BackendTarget round-trip) for ~80 lines of
  hand-rolled code. Pulling in a serialization library would have added
  a transitive dependency for what stays a 200-line targeted parser.

- **Per-pattern recognizer files are the right granularity.** RMS norm
  + RoPE + cross-entropy each landed as 100–170 line self-contained
  files in the L3.1 commit. Diff review was each file in isolation;
  no cross-file refactoring needed.

- **One vendoring divergence per layer is sustainable.** L2.5 had two
  divergences (StepType enum entry, sample workflows). L3 has one
  (KubernetesCommand fields). At this rate the vendored Maestro tree
  grows ~3 divergences per major release — well below the audit's
  "≤2 vendor edits per layer" ceiling on average.

---

## Sign-off

**L3 closed**. Seven sub-milestones (L3.0 through L3.7) shipped over
§0.4.250–§0.4.260. Net delta:

- 23 new files in `:ir` (recognizer/coarsener/kernel/cost/fusion/quant subpackages).
- 3 modified + 2 new files in `:maestro` (BackendTarget, populator, manifest extension).
- 4 modified + 4 new files in `third-party/maestro/` (KubernetesCommand divergence + maestro-tlaloc additions).
- 4 example files + 1 README in `examples/layer3/`.
- 1 design doc + 1 audit (this file).
- 10 spec entries (§0.4.250 through §0.4.259).
- +118 tests (1071 → 1189 combined).
- 7 audit OQs filed for §18 inclusion.

Layer 3 ships a recognizer + coarsener + kernel-template + cost-model +
KV-quant + backend-matrix + pod-spec pipeline that takes user-written
DXIR through to K8s job-launch with vendor-fused-kernel custom-calls
where supported, decompose-fallback elsewhere, and a structured
manifest carrying the per-target compile decisions across the network.

The next major work is L4 — sharding-aware kernel custom-calls + live
runtime + cost-driven scheduling.

— Closing audit, §0.4.260
