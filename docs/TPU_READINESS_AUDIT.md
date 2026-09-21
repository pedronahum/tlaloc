# TPU readiness audit — Tlaloc vs TorchTPU (2026-09-21)

**Status: assessment on file at Pedro's request; Phase G proposal at the
end awaits ratification.** Researched 2026-09-21 against the TorchTPU
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
