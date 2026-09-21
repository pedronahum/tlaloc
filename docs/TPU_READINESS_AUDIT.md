# TPU readiness audit — Tlaloc vs TorchTPU (2026-09-21)

**Status: PHASE G RATIFIED (Pedro, 2026-09-21) with the local/hardware
split** — the arc certifies only what its machine can prove: G1 (bf16)
+ G2a (TPU bring-up, local half) + G3a (collectives/Shardy/multi-host
design) run on the GB10 now, with SPLIT deletion ratified alongside
(DONE §0.4.454, §5 below);
G2b (TPU execution), G4 (distributed trainer) and G5 (Pallas/Mosaic
kernels) are GATED ON HARDWARE — a Cloud TPU VM Pedro provisions
(v5e/v6e spot suffices). The running record lands in §5 below. Researched 2026-09-21 against the TorchTPU
announcement (Google, April 2026 — the PyTorch-native TPU stack that
will replace PyTorch/XLA) and the OpenXLA PJRT plugin ecosystem.

## What TorchTPU is

Eager-first PyTorch on TPU via the PrivateUse1 device extension: three
eager modes (debug / strict / fused — fused auto-fuses for a claimed
50–100%+ gain) over a persistent multi-host compilation cache; compile
path Dynamo FX → **StableHLO** → XLA; distributed = DDP, FSDPv2,
DTensor, MPMD divergence; custom kernels via a Pallas/JAX passthrough
decorator, Helion DSL planned; 2026 roadmap: bounded dynamism,
precompiled kernel library, public repo, vLLM/TorchTitan integration;
TPU 8 generation. Sources: the Google Developers Blog TorchTPU post,
PyTorch/XLA plugin docs, the OpenXLA PJRT plugin RFC, libtpu on PyPI.

## The strategic good news

1. **The TPU on-ramp already exists in Tlaloc's architecture.** libtpu
   ships a standard PJRT C API plugin (`pjrt_c_api_tpu_plugin.so`, on
   PyPI) — the same ABI JAX and PyTorch/XLA use, explicitly open to
   third-party frameworks. Tlaloc's FFM runtime speaks exactly this ABI,
   and `TLALOC_PJRT_PLUGIN_PATH` (§0.4.306) was designed for the swap.
2. **TorchTPU converging on StableHLO validates Tlaloc's IR bet** — we
   emit the dialect the TPU toolchain ingests, and the explicit-threefry
   RNG emission (§0.4.422) is TPU-portable by construction (pure
   StableHLO integer ops; the bit stream cannot fork).

## Weaknesses, ranked

1. **BF16 (critical).** TPUs are bf16-first; Tlaloc is F32/F64 with the
   BF16 host-representation design open since §0.4.354. Without bf16
   end-to-end (DType, host storage, `bf16` emission, casts,
   mixed-precision accumulation), TPU numbers will be uncompetitive.
2. **TPU plugin bring-up unproven.** Architecturally free, practically
   untested: the hand-encoded minimal `CompileOptionsProto`, the
   ExecuteOptions struct layout, memory kinds, and donation semantics
   are certified against the CUDA plugin only. Needs `PjrtTarget.Tpu` +
   a Cloud TPU VM smoke lane (no local TPU hardware exists — a
   cloud/CI story is required).
3. **Collectives/multi-host half-alive.** `ALL_REDUCE` /
   `SHARD_CONSTRAINT` neither interpret nor differentiate (see
   [AD_SINGLE_ENGINE_AUDIT.md](AD_SINGLE_ENGINE_AUDIT.md) finding C);
   Shardy round-trip suites skip locally; no multi-host PJRT
   initialization; no DDP/FSDP equivalent over the Phase F trainer.
4. **No TPU custom-kernel lane.** KPTX is PTX — N/A on TPU. Named
   refusal for now; the eventual route is Mosaic-style kernels behind
   `stablehlo.custom_call`, structurally the KPTX claiming design.
5. **Bounded dynamism.** Tlaloc recompiles per shape with session
   caching (classic XLA behavior); TorchTPU is investing here. Track,
   don't chase yet.
6. **No eager device mode — by design.** Tlaloc's answer is
   compile-first with readable reverse source (the north star), stated
   deliberately rather than by omission.

## Amendments from the TorchTPU talk deep-dive (2026-09-21, Pedro's ask)

Researched against the PyTorch Conference NA 2026 material, the
Helion-on-TPU PyTorch post, and Ray's TorchTrainer multi-slice PR.
TorchTPU is now PUBLIC OSS, integrated with HuggingFace Transformers,
TorchTitan, vLLM and SGLang.

**Kernels — Helion/Pallas vs KPTX.** TorchTPU's custom-kernel lane is
Pallas/JAX via decorator; Helion (Meta's high-level DSL) now compiles
one kernel source to EITHER CuteDSL (NVIDIA) or Pallas (TPU) — flash
attention at 838 TFLOPs / ~79% MFU on TPU v7, 4.48× over torch.compile
on attention. Tlaloc supports NEITHER; KPTX is PTX/ISA-level and
NVIDIA-only. KPTX's differentiators (self-verifying transpiler,
byte-identical corpus, RECOGNIZER-DRIVEN CLAIMING — kernels attach to
coarsened ops automatically, where Helion kernels are hand-invoked) are
real, but Helion's cross-vendor authoring is a direct strategic answer
to per-vendor DSLs. Response: G5 upgrades from deferral to designed
slice — Mosaic/Pallas-GENERATED kernels behind `stablehlo.custom_call`,
claimed by the existing recognizers (reuse our strongest asset; do not
chase DSL portability where Meta+Google have the head start).

**Orchestration — Ray vs Maestro: complementary layers.** Ray's role
for TorchTPU is the INTRA-JOB distributed runtime: TorchTrainer worker
groups + multi-slice MegaScale coordination (num_slices inspection,
MegaScale env dispatch across slices). Tlaloc's orchestration —
vendored Netflix Maestro + `maestro-tlaloc` (TlalocPodSpecBuilder, K8s
step execution) over `:maestro`'s typed manifests (StableHLO+SDY
bodies, content-addressed, Mesh placement, a backendMatrix schema that
already carries "google"/"tpu_v5e") — is the BETWEEN-JOBS pipeline
layer, which TorchTPU's stack has no typed equivalent of. Maestro is
NOT Ray and should not become it: G3 is the Ray-shaped hole (intra-job
multi-host coordination — collective bootstrap, PJRT distributed init,
the MegaScale-equivalent env wiring), layered UNDER Maestro; the
natural seam is TlalocPodSpecBuilder emitting multi-host pod groups
carrying the G3 runtime's init env.

## Phase G proposal (awaiting ratification; amended per the above)

G1 bf16 end-to-end → G2 TPU PJRT plugin bring-up + Cloud TPU smoke/CI
lane → G3 the intra-job coordinator (collectives differentiable or
refusing by name, Shardy in the emit path, multi-host PJRT init, the
Maestro pod-group seam) → G4 distributed trainer over Phase F
(DDP-equivalent first) → G5 (designed slice, upgraded from deferral)
Pallas/Mosaic-generated kernels behind `custom_call` with
recognizer-driven claiming; bounded dynamism stays tracked-not-chased.

## 5. Running record (Phase G)

- **§0.4.455 — G1a DONE: the bf16 foundation in :core.** The design on
  file (the gap open since §0.4.354). REPRESENTATION: `BF16` is a
  first-class sealed `DType` (2 bytes, name "bf16");
  `HostBf16Storage(ShortArray)` holds RAW upper-16-bit f32 patterns —
  bf16 IS the top half of binary32 — a Short is a 16-bit bucket, never a
  number. REJECTED: a `value class Bf16` element type (KMP boxing, no
  arithmetic anyway), CharArray/packed-Int storage (worse spelling),
  lazily-narrowed FloatArray storage (lies about sizeBytes and hides
  rounding). NARROWING: round-to-nearest-even via the u32 bias-and-carry
  trick (`bits + 0x7FFF + keptLsb >> 16`), matching XLA/Eigen — NaN
  quieted before the add (payload top bits kept, 0x0040 forced) so no
  signaling NaN decays to inf; overflow carries into inf; signed zeros
  and subnormal flushes keep the sign; widening is exact (`bits << 16`).
  HOST COMPUTE CONVENTION (the v1 decision): compute-in-f32-store-bf16 —
  `DTensor<S, BF16>` is storage/interchange, host math goes
  `toF32() ... toBf16()`; per-op bf16 host twins are a NAMED DEFERRAL
  (no JVM bf16 units; PyTorch/CPU does the same dance). Surfaces:
  `floatToBf16Bits`/`bf16BitsToFloat` (+array forms), `toBf16()`/
  `toF32()` casts, `hostBf16()`, `Tensors.bf16Scalar/Vector/Matrix`
  (FloatArray in, RNE-narrowed patterns stored). IR/emission untouched
  (G1b): the sealed-when sites now refuse BY NAME — StableHLO emit
  errors "bf16 has no StableHLO emission yet — G1b", DxirCanonical
  refuses bf16 constants, the compiler plugin returns null exactly as
  its Bool convention does, and KotlinSourceRenderer's existing
  dtype-naming refusal covers bf16 unchanged. Oracles: 12 bit-level
  pins (hand-derived RNE ties incl. the even-down/odd-up pair and
  one-ulp neighbors, signed zeros, inf pass-through, MAX_VALUE→inf,
  NaN quieting, subnormal flush-by-rounding, the 256-exponent
  double-sign sweep with mantissa 0/0x55 proving bf16→f32→bf16
  identity, storage/constructor/cast round-trips). Suite 2044 → 2056.

- **§0.4.454 — G slice 1 DONE: OpKind.SPLIT deleted.** The arc's banked
  win, ratified with Phase G itself (§0.4.453; recommended since the
  §0.4.448 audit-finding-C demotion). The kind was unreachable — the
  §0.4.448 grep showed nothing constructs it outside the StableHLO
  emitter arm and IR-plumbing tests, and the user-facing split surfaces
  (the :core/:nn host `split()` from §0.4.428, the FIR stack/split
  folds) never emitted it (re-verified at HEAD: zero `OpKind.SPLIT`
  references in core/nn/fir/frontend). Removed: the enum entrant, the
  emitter's arm + `emitSplit` (SPLIT-only; ARGMAX's `%pair:2` handling
  is separate and untouched), the `DemotedOpKinds` entry + refusal
  message, the cost-model row, the interpreter/renderer refusal-arm
  listings, 12 SPLIT-vehicle tests (6 EmitterTest, 3 RoundTripTest, 3
  DemotedKindRefusalTest rows), and every stale doc/comment reference
  (OpKind.kt, DxirReverse/ForwardTransform, TLALOC_EMIT_CONTRACT.md,
  AD_SINGLE_ENGINE_AUDIT.md). KEPT: DxirTest's two multi-result
  IR-plumbing pins, retargeted to COARSENED — the plumbing under test
  (opMulti, per-index `DxirOpResult`, the `%N:2` packed printer form)
  is kind-agnostic and stays covered. Per-piece SLICE is the sanctioned
  spelling; a re-introduction would need the full differentiable-op
  contract (interpreter arm, VjpRule with per-index adjoint routing,
  forward tangent, KotlinSourceRenderer arm or named refusal, oracle
  story) — recorded in the OpKind.kt comment where the entrant lived.
