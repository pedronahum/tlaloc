# TPU readiness audit — Tlaloc vs TorchTPU (2026-09-21)

**Status: PHASE G — THE LOCAL ARC IS COMPLETE (§0.4.454–462, all landed
2026-09-21); THE HARDWARE HALF IS GATED, NOT STARTED.** Ratified by
Pedro 2026-09-21 with the local/hardware split, and the split held: the
arc certified only what a GB10 CUDA box can prove — SPLIT deletion
(§0.4.454), G1a–G1d bf16 end-to-end through `:core`/IR/real XLA/`:nn`
(§0.4.455–458), G2a TPU bring-up's local half (§0.4.459), G3a
ALL_REDUCE + SHARD_CONSTRAINT un-demoted (§0.4.460) and G3a-2 the
multi-host design + Maestro seam (§0.4.461). **No TPU-execution and no
multi-host-run claim exists anywhere in this repo.** G2b (TPU
execution), G4 (distributed trainer) and G5 (Pallas/Mosaic kernels)
await a Cloud TPU VM Pedro provisions (v5e/v6e spot suffices);
[TPU_BRINGUP.md](TPU_BRINGUP.md) is the next session's script. The
canonical arc-close suite number is **2119** (§0.4.462 clean-room,
`test jvmTest --rerun`, zero failures, CUDA smokes EXECUTED and the 5
TPU smokes skipping by design). The running record and the ARC STATE
land in §5 below. Researched 2026-09-21 against the TorchTPU
announcement (Google, April 2026 — the PyTorch-native TPU stack that
will replace PyTorch/XLA) and the OpenXLA PJRT plugin ecosystem.

## 1. What TorchTPU is

Eager-first PyTorch on TPU via the PrivateUse1 device extension: three
eager modes (debug / strict / fused — fused auto-fuses for a claimed
50–100%+ gain) over a persistent multi-host compilation cache; compile
path Dynamo FX → **StableHLO** → XLA; distributed = DDP, FSDPv2,
DTensor, MPMD divergence; custom kernels via a Pallas/JAX passthrough
decorator, Helion DSL planned; 2026 roadmap: bounded dynamism,
precompiled kernel library, public repo, vLLM/TorchTitan integration;
TPU 8 generation. Sources: the Google Developers Blog TorchTPU post,
PyTorch/XLA plugin docs, the OpenXLA PJRT plugin RFC, libtpu on PyPI.

## 2. The strategic good news

1. **The TPU on-ramp already exists in Tlaloc's architecture.** libtpu
   ships a standard PJRT C API plugin (`pjrt_c_api_tpu_plugin.so`, on
   PyPI) — the same ABI JAX and PyTorch/XLA use, explicitly open to
   third-party frameworks. Tlaloc's FFM runtime speaks exactly this ABI,
   and `TLALOC_PJRT_PLUGIN_PATH` (§0.4.306) was designed for the swap.
2. **TorchTPU converging on StableHLO validates Tlaloc's IR bet** — we
   emit the dialect the TPU toolchain ingests, and the explicit-threefry
   RNG emission (§0.4.422) is TPU-portable by construction (pure
   StableHLO integer ops; the bit stream cannot fork).

## 3. Weaknesses, ranked — swept at the local-arc close (§0.4.462)

The ranking below is the ORIGINAL 2026-09-21 audit ordering, each row
re-stated at the close of the local arc. The original text is kept
(struck through in prose, not markup) so the arc's starting position
stays legible; **AT THE CLOSE** is what is true at HEAD.

1. ~~**BF16 (critical).**~~ → **LANDED, WITH MEASURED FLOORS
   (§0.4.455–458; G1a–G1d).** *Was:* TPUs are bf16-first; Tlaloc is
   F32/F64 with the BF16 host-representation design open since
   §0.4.354; without bf16 end-to-end (DType, host storage, `bf16`
   emission, casts, mixed-precision accumulation), TPU numbers will be
   uncompetitive. **AT THE CLOSE:** all five named pieces exist.
   `BF16` is a first-class `DType` with `HostBf16Storage(ShortArray)`
   of raw upper-16-bit f32 patterns and XLA/Eigen-matching RNE
   narrowing (§0.4.455); the IR carries bf16 end to end — interpreter
   (the stated compute-in-f32-snap-at-the-op-boundary convention),
   `CastRule`'s straight-through adjoint, `bf16` StableHLO emission
   with every f32 assumption enumerated (§0.4.456); real XLA on the
   GB10 certifies BOTH native `PJRT_Buffer_Type_BF16` buffers (2
   bytes/element on device — not silently widened f32) and the
   cast-at-boundary shape (§0.4.457); and `:nn` ships
   `Precision.MIXED_BF16` — f32 master weights, bf16 compute, f32 loss
   and gradients, no loss scaler (§0.4.458). **The floors, measured,
   not asserted:** device f32→bf16 narrowing is BIT-EXACT vs the host
   RNE helper on a 19-lane sweep (ties both directions, one-ulp
   neighbours, signed zeros, overflow→inf, subnormal flush); bf16
   matmul+add on native buffers is BIT-EXACT vs the interpreter on
   bf16-exact lanes; the bf16 REVERSE graph's one honest divergence is
   0.0028125 on the pinned lane — exactly the snap error, bounded at
   one bf16 ulp (2⁻⁸ relative), the recorded tolerance for graphs whose
   narrowings XLA can elide; mixed-precision on GPU holds loss
   BIT-EQUAL with gradients inside the derived R·2⁻⁸·scale floor (R =
   25 bf16 ops counted on the graph), and the exact lane is bit-exact
   for loss AND every gradient; the host forward envelope is DERIVED
   (|ΔL| ≤ (1/n)Σ Δyᵢ(2(Aᵢ+|tᵢ|)+Δyᵢ)) and measured 2.1e-4 against a
   bound of 0.163. **Two findings that outlive the slice:** XLA-CUDA
   FOLDS an f32→bf16→f32 convert pair to identity (so only a
   single-convert program measures the device), and it rounds a bf16
   dot's output AT THE OP BOUNDARY (the 257-tie discriminator answered
   256, matching the interpreter's per-op-snap convention, not
   fused-f32's 258). Both are XLA-CUDA measurements and are listed for
   TPU re-measurement in [TPU_BRINGUP.md](TPU_BRINGUP.md). Remaining:
   bf16 CONSTANTS (spell `CAST(f32 const)`), bf16 RNG draws (refused by
   name — the threefry mantissa trick is a binary32 bit-stream
   contract; draw at f32 and cast), f32 trace-time constants inside a
   bf16 region, and mixed coverage beyond the Dense-MLP family.
2. **TPU plugin bring-up: the LOCAL HALF DONE, EXECUTION STILL
   UNPROVEN (§0.4.459; G2a).** *Was:* architecturally free, practically
   untested — `CompileOptionsProto`, ExecuteOptions layout, memory
   kinds and donation certified against the CUDA plugin only.
   **AT THE CLOSE:** `PjrtTarget.Tpu` exists with libtpu-shaped plugin
   resolution (the tpu-name gate keeps a CUDA host's generic
   `TLALOC_PJRT_PLUGIN_PATH` out of the TPU lane, unit-pinned both
   ways); the §0.4.333 create-options are PLATFORM-GATED (they are the
   GPU allocator's knobs — a Tpu client passes `create_options = NULL`,
   and both cross-wirings refuse by name, including the one that would
   revive the reboot incident); the 6-byte `CompileOptionsProto` is
   re-verified field-by-field against openxla/xla main and structurally
   decoded by a local pin; `PJRT_ExecuteOptions` is ruled header ABI,
   not backend ABI. `PjrtTpuSmokeTest` (5 tests) is WRITTEN and SKIPS
   CLEANLY here by design; `PjrtTpuLocalCertTest` (7 tests) runs green
   everywhere. **This row does not close without hardware** — memory
   kinds, donation semantics, libtpu's accepted option set, and the TPU
   tiled bf16 buffer size are all still RECORDED UNKNOWNS, enumerated
   as G2b's six questions in the runbook.
3. ~~**Collectives/multi-host half-alive.**~~ → **ALL_REDUCE COMPLETE;
   MULTI-HOST DESIGNED + MARSHALLED (§0.4.460, §0.4.461; G3a,
   G3a-2).** *Was:* `ALL_REDUCE`/`SHARD_CONSTRAINT` neither interpret
   nor differentiate (audit finding C), Shardy suites skip locally, no
   multi-host PJRT init, no DDP/FSDP equivalent. **AT THE CLOSE:** both
   kinds are UN-DEMOTED and out of `DemotedOpKinds` (which is now
   LAYERNORM + SDPA only). ALL_REDUCE-sum has a shared attr parser, an
   interpreter arm on the stated SPMD replicated-value semantics, a
   `stablehlo.all_reduce` region emission, and a `VjpRule` resting on
   the proved fact that all-reduce-sum is SELF-ADJOINT — the gradient
   is the same all_reduce on the upstream, pinned structurally and
   numerically; SHARD_CONSTRAINT is a value identity with layout
   metadata, so its identity adjoint/tangent are honest. Multi-host:
   [MULTIHOST_DESIGN.md](MULTIHOST_DESIGN.md) is the authority (the
   create-option NamedValues vs kv-store-callback distinction verified
   against openxla/xla main; **there is no `coordinator_address`
   option**), `PjrtClientOptions` marshals `node_id`/`num_nodes` with
   the single-node bytes pinned BYTE-IDENTICAL to §0.4.333, multi-node
   client creation REFUSES BY NAME without kv-store callbacks, and
   `TlalocPodSpecBuilder.buildPodGroup` expands one manifest into N
   rank-carrying pods. **What is NOT claimed:** no multi-device
   ALL_REDUCE execution (G4c), no multi-host run (needs 2+ hosts), no
   DDP/FSDP trainer (G4e). The Shardy skip is no longer silent — the
   `sdy-opt` gate and its build-from-source provisioning are documented
   in the runbook, with the recorded gap that plain ops carrying a
   `DxirSharding` get no per-op `sdy.sharding` attr.
4. **No TPU custom-kernel lane — UNCHANGED, now a DESIGNED slice.**
   KPTX is PTX, N/A on TPU. The named refusal stands; G5 was upgraded
   from deferral to designed slice by the §3a amendment (Mosaic/Pallas
   kernels behind `stablehlo.custom_call`, claimed by the existing
   recognizers). Nothing was built this arc.
5. **Bounded dynamism — UNCHANGED, deliberately.** Tlaloc recompiles
   per shape with session caching (classic XLA behavior); TorchTPU is
   investing here. Tracked, not chased.
6. **No eager device mode — UNCHANGED, by design.** Tlaloc's answer is
   compile-first with readable reverse source (the north star), stated
   deliberately rather than by omission. The arc reinforced it: every
   new op and dtype either RENDERS in `KotlinSourceRenderer` or
   REFUSES BY NAME there (verified by grep at the close — SHARD_CONSTRAINT
   renders as identity, ALL_REDUCE refuses because rendering ×|group|
   would bake a distribution fact into host math, bf16 refuses because
   host bf16 math is compute-in-f32 and the readable reverse of a bf16
   program is the f32 graph between its casts).

## 3a. Amendments from the TorchTPU talk deep-dive (2026-09-21, Pedro's ask)

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

## 4. Phase G (RATIFIED 2026-09-21; the shape the arc actually ran)

G1 bf16 end-to-end → G2 TPU PJRT plugin bring-up + Cloud TPU smoke/CI
lane → G3 the intra-job coordinator (collectives differentiable or
refusing by name, Shardy in the emit path, multi-host PJRT init, the
Maestro pod-group seam) → G4 distributed trainer over Phase F
(DDP-equivalent first) → G5 (designed slice, upgraded from deferral)
Pallas/Mosaic-generated kernels behind `custom_call` with
recognizer-driven claiming; bounded dynamism stays tracked-not-chased.

## 5. Running record (Phase G)

### ARC STATE (§0.4.462, the close-out) — read this first

**THE LOCAL HALF IS COMPLETE. THE HARDWARE HALF IS GATED, NOT STARTED.**

Nine sections, one day (2026-09-21), suite **2044 → 2119**:

| § | Slice | What it closed | Suite |
| --- | --- | --- | --- |
| 0.4.454 | G slice 1 | `OpKind.SPLIT` deleted (unreachable; per-piece SLICE is the spelling) | 2044 |
| 0.4.455 | G1a | bf16 foundation in `:core` — `DType`, `HostBf16Storage`, RNE | 2044 → 2056 |
| 0.4.456 | G1b | bf16 through the IR — interpreter, `CastRule`, StableHLO emission | 2056 → 2067 |
| 0.4.457 | G1c | bf16 certified on REAL XLA (GB10 CUDA): native buffers + cast-at-boundary | 2067 → 2071 |
| 0.4.458 | G1d | mixed-precision training in `:nn` — `Precision.MIXED_BF16`, no loss scaler | 2071 → 2080 |
| 0.4.459 | G2a | TPU bring-up, local half — `PjrtTarget.Tpu`, platform-gated options, self-skipping smoke suite | 2080 → 2092 |
| 0.4.460 | G3a | ALL_REDUCE + SHARD_CONSTRAINT UN-DEMOTED — interpreter, VJP/JVP, emission | 2092 → 2103 |
| 0.4.461 | G3a-2 | multi-host design + the certifiable halves (client options, Maestro pod groups) | 2103 → 2119 |
| 0.4.462 | close-out | this sweep — docs only | 2119 |

**The canonical arc number is 2119**, clean-room certified at §0.4.462
(`./gradlew test jvmTest --rerun`, zero `failures=`/`errors=` across
327 JUnit XMLs, `scripts/count-tests.sh` = 2119). The certification's
shape is itself the arc's honesty claim: **every CUDA smoke suite
EXECUTED on the GB10** (`PjrtBf16SmokeTest` 4/4,
`PjrtMixedPrecisionSmokeTest` 2/2, `PjrtRngSmokeTest` 5/5, and the rest
— `skipped="0"`), and **`PjrtTpuSmokeTest` skipped 5 of 5**, which is
exactly what a machine with no TPU should report.

**North-star invariants, verified by grep at the close:**
- **One engine.** Zero `fun backward(`, zero `class Gradients`, zero
  `class GradientTape` anywhere in the repo outside vendored Maestro.
  `DxirReverseTransform` still owns every adjoint the arc added —
  `AllReduceRule`, `ShardConstraintRule` and `CastRule`'s bf16
  straight-through arm are registry rules, not hand-written math.
- **No silent gaps.** Every op and dtype the arc introduced either
  renders in `KotlinSourceRenderer` or refuses there BY NAME:
  SHARD_CONSTRAINT → identity arm; ALL_REDUCE → named refusal
  (line ~609); bf16 → named refusal that states the story (line ~150).
  `DemotedOpKinds` shrank to LAYERNORM + SDPA, each refusal naming its
  sanctioned alternative.

**WHAT AWAITS THE TPU VM** — the exact list, in dependency order.
[TPU_BRINGUP.md](TPU_BRINGUP.md) is the next session's script
(provisioning, JDK, libtpu, the gradle invocation, and teardown).

1. **G2b — TPU execution.** Run `PjrtTpuSmokeTest`'s 5 tests for real
   and turn the skips into passes. Its six recorded questions:
   (a) libtpu reports platform `"tpu"`; (b) **the flagship** — threefry
   uniform draws bit-exact on TPU (the §0.4.422 emission is pure
   StableHLO integer ops and cannot fork by construction; a TPU pass
   makes the portability claim MEASURED); (c) whether TPU-XLA also
   folds the f32→bf16→f32 convert pair and where it rounds a bf16 dot's
   output (CUDA: folds, and rounds at the op boundary); (d) the TPU
   tiled on-device size for a bf16 buffer (CUDA pins 2 bytes/element;
   TPU may pad to tiles — the test PRINTS and lower-bounds it, never
   pins it); (e) which create-options libtpu accepts (`ml_framework_name`,
   `ml_framework_version`, `max_inflight_computations` are candidates
   deliberately NOT passed today); (f) donation semantics + memory
   kinds beyond defaults. Then re-run `PjrtBf16SmokeTest` and
   `PjrtMixedPrecisionSmokeTest` there — parametrizing both onto a
   shared multi-backend harness is G2b's named cleanup.
2. **G4a — the kv-store FFM upcalls + a coordinator service.** The
   design intent is MULTIHOST_DESIGN.md §4; today `numNodes > 1` client
   creation refuses by name precisely because NULL kv callbacks would
   fail or hang inside the plugin.
3. **G4b — multi-node client creation**, once (2) exists.
4. **G4c — ALL_REDUCE across real devices**, upgrading §0.4.460's
   single-process replicated-value semantics to a measured collective.
   The `reduction` general-op story rides here too: mean scales the
   adjoint by 1/|group|, max/min need subgradient routing, and ragged
   `replica_groups` need StableHLO's -1 padding — all three refuse by
   name today.
5. **G4d — the Maestro wiring**: the `distributed { nodes = N }` step
   schema + `TlalocStepRuntime`; gang-scheduling CRDs are a stated
   deployment requirement, not something the builder emits.
6. **G4e — the DDP-equivalent trainer** over Phase F's `:nn`.
7. **G5 — Pallas/Mosaic kernels** behind `stablehlo.custom_call` with
   recognizer-driven claiming (designed, not started).
8. **Independent of the TPU: `sdy-opt`.** `SdyRoundTripTest` and
   `SdyPropagationTest` skip on every box here — no prebuilt aarch64
   binary exists; provisioning means a bazel build of
   `//shardy/tools:sdy_opt` from openxla/shardy. Same story on a TPU VM.
   Multi-slice/MegaScale env emission (the `MEGASCALE_*` layer recorded
   UNVERIFIED in MULTIHOST_DESIGN.md) confirms against a live libtpu at
   the same time.

### The record, newest first

- **§0.4.462 — the Phase G local-arc CLOSE-OUT (docs only).** The
  sweep: §5's running record made complete with the ARC STATE above
  (the nine-section table, the canonical number, the hardware queue);
  §3's weakness table re-stated row by row at the close — bf16 from
  *critical gap* to **landed with measured floors** (the floors
  transcribed, not summarized: bit-exact narrowing, bit-exact native
  matmul, the 0.0028125 one-ulp reverse divergence, the R·2⁻⁸·scale
  mixed-precision floor, the 2.1e-4-against-0.163 derived envelope),
  collectives from *half-alive* to **ALL_REDUCE-complete**, multi-host
  from *absent* to **designed + marshalled**, with rows 4–6 marked
  deliberately unchanged; the doc's sections NUMBERED (1–5, the
  TorchTPU amendments becoming 3a) so "§3" and "§5" resolve; the
  header rewritten to state the arc state and forbid the claims the
  arc does not hold. [DIFFKT_PARITY_PLAN.md](DIFFKT_PARITY_PLAN.md)'s
  end-state header and [MODEL_LAYER_PLAN.md](MODEL_LAYER_PLAN.md)'s end
  state both swept: the DiffKT book of work STAYS CLOSED (Phase G is
  not a DiffKT phase — DiffKT has no bf16, no TPU and no collectives),
  but both now point here and carry the post-arc suite number so
  neither reads as stale at 2020. The north-star invariants verified by
  grep rather than assertion (results above). NO CODE CHANGED — the
  suite number is therefore §0.4.461's, re-certified clean-room rather
  than inherited. **Suite 2119 → 2119.**

- **§0.4.461 — G3a-2 DONE: the multi-host design on file + the certifiable
  seam implemented** (design-heavy by charter; NO multi-host run is claimed
  anywhere — that needs 2+ hosts and is G4's). **(1) docs/MULTIHOST_DESIGN.md**
  is the authority: the PJRT C API's two distributed surfaces kept distinct —
  the GPU plugin's create-option NamedValues (the full parsed set enumerated
  and VERIFIED against xla/pjrt/c/pjrt_c_api_gpu_internal.cc at openxla/xla
  main 2026-09-21, `node_id`/`num_nodes` kInt64 among them) versus the
  kv-store callbacks in PJRT_Client_Create_Args (how multi-node clients
  actually rendezvous; **there is NO coordinator_address option** — the
  coordinator is a framework-side service backing the kv callbacks, JAX's
  distributed runtime being the reference); the TPU/MegaScale env layer
  (TPU_PROCESS_* singles, MEGASCALE_COORDINATOR_ADDRESS/NUM_SLICES/SLICE_ID/
  PORT multi-slice — recorded from Ray/TorchTPU usage, marked UNVERIFIED
  until a libtpu confirms); and the design consequence stated: the group
  contract travels as ENV, because both worlds consume env. **(2) The
  certified runtime half**: `PjrtClientOptions` gains
  `nodeId`/`numNodes`/`coordinatorAddress` — single-node marshals
  BYTE-IDENTICAL to §0.4.333 (the distributed fields add nothing until asked
  for, pinned), `numNodes > 1` marshals node_id/num_nodes as kInt64 entries
  3–4 (bytes pinned GPU-less), `coordinatorAddress` is NEVER marshalled (no
  such option exists — it is the G4 kv-store dial target, validated
  host:port, required exactly when multi-node), and client CREATION at
  numNodes > 1 REFUSES BY NAME via the pure-extracted
  `requireKvStoreForMultiNode` (NULL kv callbacks would fail or hang inside
  the plugin; the refusal names MULTIHOST_DESIGN.md) — refusal certified
  GPU-less too. `resolve()` reads the env trio TLALOC_PJRT_NODE_ID/
  NUM_NODES/COORDINATOR_ADDRESS. **(3) The certified Maestro half**:
  `TlalocPodSpecBuilder.buildPodGroup(base, numNodes, host, port)` expands
  one accelerator-selected KubernetesCommand into N members — env trio with
  per-member rank, dedup key suffixed `-node<i>` so members never collapse
  into one K8s job, everything else copied (mesh-consistent by construction:
  one manifest, N pods), reserved-env collisions refused by name, composition
  order pinned (applyBackendTarget THEN buildPodGroup). REJECTED: a Python
  `jax.distributed` sidecar for init (reintroduces the runtime Python the FFM
  stack exists to avoid); minting K8s names inside the builder (the runner
  owns naming — coordinator host/port are data). NAMED DEFERRALS: the kv-store
  FFM upcalls + coordinator service (G4a — design intent in the doc §4);
  gang-scheduling CRDs (deployment requirement, stated, not emitted); the
  Maestro `distributed { nodes = N }` step schema + TlalocStepRuntime wiring
  (G4d — rides with the first real run); multi-slice/MegaScale env emission.
  The G4 dependency chain recorded (doc §6): G4a kv-store → G4b multi-node
  create → G4c ALL_REDUCE across real devices (upgrading §0.4.460's
  single-process semantics) → G4d Maestro wiring → G4e DDP-equivalent
  trainer; multi-slice last. Suite 2103 → 2119.

- **§0.4.460 — G3a slice 1 DONE: ALL_REDUCE from demoted refusal to real
  op — the intra-job coordinator's first brick** (LOCAL certification —
  single-process semantics only; NO multi-device execution claim, that is
  G2b/G4's). Both §0.4.448-demoted collectives UN-DEMOTED, deliberately,
  per what reading showed. **(1) The attr convention**, one shared parser
  (`ir/.../AllReduceAttrs.kt`) so no layer can disagree:
  `replica_groups: List<List<Int>>` (absent = `[[0]]`, the single-replica
  program; uniform group sizes — ragged needs StableHLO's -1 padding,
  NAMED DEFERRAL; disjoint, non-negative, replica 0 present) and
  `reduction: String` (absent = "sum"; v1 supports sum ONLY — the
  general-op story recorded on every refusal: mean scales the adjoint by
  1/|group|, max/min need subgradient routing, both DEFER BY NAME).
  **(2) Interpreter arm** — the SPMD replicated-value view: the process
  models replica 0, whose group members all hold the same value, so
  all-reduce-sum = |group(0)| × value; replica_count == 1 is EXACT
  identity (the copyOf arm, no float math), certified end-to-end; the
  multi-replica scale is exercised ONLY via unit semantics until G2b/G4
  put devices behind the groups (stated on the pins). **(3) StableHLO
  emission** — `"stablehlo.all_reduce"` in the generic region form
  (stablehlo.add reduction region, `^bb0` element-typed block args,
  `stablehlo.return`, dense<NxMxi64> replica_groups; no channel_handle —
  the cross-replica default matching §0.4.459's single-host
  ExecuteOptions), MLIR-shape pinned in EmitterTest (single-group,
  absent-default, multi-group, named refusals). **(4) Differentiability**
  — AllReduceRule: all-reduce-sum is SELF-ADJOINT (Jacobian
  `ones(n,n) ⊗ I`, symmetric), so the gradient is the SAME all_reduce
  (same replica_groups) on the upstream — pinned structurally AND
  numerically (grad of Σ all_reduce(x⊙x) over |group|=2 hand-pinned at
  4x; JVP-VJP identity ⟨∇f,v⟩ = tangent through the all-reduce-bearing
  loss); forward tangent is the same op on the tangent (linear). NOTE the
  recorded LEVEL DISTINCTION: GradShardingVerify.adjointOf's
  varying→invariant sharded-pipeline table (all_reduce ↔ identity) is a
  DIFFERENT level than the VjpRule's replicated per-replica-upstream
  view; both stand, neither replaces the other (on OpKind.ALL_REDUCE and
  the rule doc). **(5) SHARD_CONSTRAINT un-demoted too** — reading showed
  it IS a value-identity with layout metadata (emitShardConstraint
  asserts shape preservation; `sdy.sharding_constraint` passes the value
  through), so the identity adjoint/tangent are honest and cheap:
  interpreter identity arm, ShardConstraintRule (upstream pass-through;
  re-applying the SAME constraint to the adjoint is a NAMED DEFERRAL —
  needs mesh carryover into AD-built functions, GradShardingVerify's
  param-boundary check governs meanwhile), renderer pass-through arm.
  **(6) KotlinSourceRenderer**: SHARD_CONSTRAINT renders as identity;
  ALL_REDUCE keeps a NAMED refusal (the readable reverse of a collective
  program is its per-replica local source — rendering ×|group| would bake
  a distribution fact into host math, the sentinel-dims class of
  mistake). **(7) Shardy visibility**: the SdyRoundTrip/SdyPropagation
  skip condition (assumeTrue on `sdy-opt` on PATH) is now documented in
  docs/TPU_BRINGUP.md with the build-from-source provisioning note; the
  emit-path sdy annotations where DxirSharding is present are verified
  against their existing pins (EmitterTest sharding_constraint family,
  CoarsenedCustomCallTest's per_value present+absent pair), and the
  recorded gap is stated: plain ops with a DxirSharding carry no per-op
  `sdy.sharding` attr — propagation owns plain-op layouts. ALL_GATHER /
  REDUCE_SCATTER stay duality-table-only kinds until a slice needs them.
  DemotedOpKinds shrinks to LAYERNORM + SDPA (refusal rows moved to
  AllReduceTest's real-op oracles); AD_SINGLE_ENGINE_AUDIT finding C
  updated in the same commit. Suite 2092 → 2103.

- **§0.4.459 — G2a DONE: the local half of TPU bring-up — everything but
  execution, which remains FORBIDDEN to claim** (no TPU exists here; the
  smoke suite self-skips by design and G2b turns the skips into passes
  from a Cloud TPU VM per the new docs/TPU_BRINGUP.md runbook).
  SURFACES: **(1) `PjrtTarget.Tpu`** (platform "tpu") with
  `PjrtBinaries.tpuPluginPath` — `TLALOC_PJRT_PLUGIN_PATH` honoured only
  when it names a tpu-shaped .so (the gate that keeps a CUDA host's
  generic env var out of the TPU lane, unit-pinned both directions), else
  the libtpu default locations documented from the PyPI wheel layout
  (site-packages/libtpu/libtpu.so under $VIRTUAL_ENV and ~/.local) and
  the TPU VM images (/lib/libtpu.so, /usr/lib/libtpu.so;
  pjrt_c_api_tpu_plugin.so on older images via the env var); resolution
  core extracted pure so it certifies GPU-less. **(2) The proto/struct
  backend audit**: the 6-byte CompileOptionsProto re-verified against
  xla/pjrt/proto/compile_options.proto at openxla/xla main 2026-09-21
  (executable_build_options=3, num_replicas=4, num_partitions=5 —
  protobuf wire format is backend-agnostic, so a TPU compile parses it
  identically), extracted as `COMPILE_OPTIONS_PROTO_BYTES` and
  structurally decoded by a local pin; PJRT_ExecuteOptions ruled header
  ABI not backend ABI (the launch_id padding holds on LP64
  aarch64+x86_64; the TPU-flavoured fields — launch_id, num_tasks/
  task_ids/incarnation_ids, multi_slice_config — are zeroed, the correct
  single-host default, non-zero forms named G3/G4 surface). **(3) The
  §0.4.333 create-options are now PLATFORM-GATED**: memory_fraction/
  preallocate are the GPU plugin's allocator knobs, so `PjrtApi
  .createClient` accepts null (create_options=NULL, num_options=0) and
  `PjrtSession`'s options nullability IS the gate — Tpu defaults to
  null, everything else to the env-resolved options, and BOTH
  cross-wirings refuse by name before any FFM work (GPU options on a TPU
  client; null options on a CUDA client, which would revive the reboot
  incident); libtpu's own accepted option set is a RECORDED UNKNOWN
  (headers not vendored; ml_framework_name/max_inflight_computations
  candidates listed unpassed in the runbook). `runOnPjrt` refuses Tpu by
  name (one-shot path is CUDA-family-wired). **(4) The TPU smoke suite,
  written now** (PjrtTpuSmokeTest, 5 tests, assumeTrue on tpuPluginPath):
  platform-name asserted "tpu" (asserted, not assumed — a mis-resolved
  plugin fails loudly), threefry uniform bit-exactness (the flagship G2b
  claim — the §0.4.422 emission is pure StableHLO integer ops,
  TPU-portable by construction), a matmul+SUM gradient graph vs the
  interpreter, and the G1c bf16 claims re-targeted (single-convert RNE
  sweep — the honest probe shape on any backend given the CUDA
  convert-fold finding; native-bf16 matmul bit-exact on bf16-exact
  lanes, device size reported and bounded below, never pinned — TPU
  tiled layout stays the open item). Plus PjrtTpuLocalCertTest (7 tests,
  run everywhere): resolution rules, both option-gate refusals, the
  proto decode, the platform-string pin. REJECTED: auto-passing
  GPU-ish options to TPU (unknown-option behaviour is libtpu's,
  not ours to guess); a strict platform==target check inside PjrtSession
  (the CUDA plugin legitimately reports "cuda" OR "gpu" — the assertion
  belongs to the TPU suite where the expectation is exact); parametrizing
  PjrtBf16SmokeTest onto a shared multi-backend harness (NAMED DEFERRAL —
  restating the claims beats refactoring a certified suite inside this
  slice; the harness is G2b's cleanup). Certified locally: the suite
  compiles, the 5 TPU smokes skip cleanly here, the 7 local certs pass,
  and the CUDA lane reruns green (GPU smokes executed, not skipped).
  Suite 2080 → 2092.

- **§0.4.458 — G1d DONE: the mixed-precision training story for :nn**
  (LOCAL certification — the GPU half runs the CUDA plugin on the GB10;
  NO TPU claim, G2b re-runs it there). THE CONVENTION, recorded on
  `io.tlaloc.nn.Precision` with its rejected alternatives: **MASTER
  WEIGHTS IN F32, COMPUTE IN BF16, LOSS AND GRADIENTS IN F32, NO LOSS
  SCALING** — bf16 keeps f32's 8-bit exponent, so the entire fp16
  GradScaler apparatus has nothing to protect against (stated as the
  reason bf16 beats fp16). `MIXED_BF16` is a CAPTURE-level flag riding
  the existing trace (a `MixedPrecision(model)` wrapper was REJECTED:
  precision is a property of one capture, not of model structure — the
  f32 capture of the same model is the oracle); the trace injects ONE
  `Tracer.cast(BF16)` per f32 leaf (params eagerly, I32 index leaves
  pass through) and ONE bf16→f32 cast at the model output, so the loss
  reduction accumulates in f32 (cast-the-loss was REJECTED: a large-N
  mean at 8 mantissa bits) and per-parameter gradients are f32 BY
  CONSTRUCTION through CastRule's straight-through adjoint. NEW
  SURFACES: `Tracer.cast(dtype)` (F32↔BF16 only, everything else
  refused by name; identity casts return `this`), and `Tape.op` dtype
  PROPAGATION — any bf16 input makes a bf16 result, bf16⊕f32 operand
  mixing REFUSES BY NAME at trace time (StableHLO wants one element
  type), and a central RNE snap at bf16 entries mirrors the
  interpreter's `snapToBf16` so tape forwards match the interpreter by
  construction. Certs (MixedPrecisionTrainingTest +
  PjrtMixedPrecisionSmokeTest): structure counted on the captured
  graph (5 injected narrows for 1 input + 4 params, exactly 1 widen,
  f32 primal params, f32 MEAN); a batch-1 snap-lane MLP BIT-EXACT vs a
  reference spelled with nothing but the §0.4.455 snap helpers (values
  chosen so every f32 op between snaps is exact — accumulation order
  cannot matter); a bf16-exact lane where mixed == f32 BIT-FOR-BIT,
  loss AND every gradient (straight-through adds exactly nothing when
  nothing rounds); the forward envelope DERIVED, not guessed —
  |ΔL| ≤ (1/n)Σ Δyᵢ(2(Aᵢ+|tᵢ|)+Δyᵢ) with Δyᵢ = ((1+2⁻⁸)⁸−1)·Aᵢ off the
  triangle-inequality magnitude bound, measured 2.1e-4 against bound
  0.163; 10 Adam steps strictly decrease the loss and reproduce
  BIT-IDENTICALLY on rerun; `gradSource()` on a mixed capture refuses
  naming bf16 (the §0.4.456 readable-reverse story holds). ON-GPU: the
  capture's OWN gradient function through the compiled lane — snap
  lane within the documented floor R·2⁻⁸·scale (R = 25 bf16 ops
  counted on the graph, the conservative elidable-narrowing bound from
  the §0.4.457 convert-fold finding; measured loss bit-equal), the
  SGD-updated weights within the lr-scaled floor, and the exact lane
  BIT-EXACT for loss + all gradients. NAMED DEFERRALS: f32 CONSTANT
  leaves inside a bf16 region refuse at trace time (auto-casting
  trace-time constants is future work — Dense/Relu MLPs capture clean;
  a forward using scalar-literal overloads does not, and says so);
  mixed coverage certified for the Dense-MLP family only (conv/
  BatchNorm/embedding-table bf16 uncertified — the embedding table
  keeps its own F32 requirement); the typed mixed-dtype session lane
  stays deferred from G1c (moot here: the boundaries are f32, `runOn`
  is the honest transport). Suite 2071 → 2080.

- **§0.4.457 — G1c DONE: bf16 certified against real XLA on the GB10**
  (LOCAL certification — CUDA plugin on Blackwell; NO TPU claim, G2b
  re-runs this suite there). BOTH forms from the G1c menu are certified,
  stated loudly: **(1) NATIVE BF16 BUFFERS** — `PJRT_Buffer_Type_BF16`
  (= 13, verified against xla/pjrt/c/pjrt_c_api.h) joins the FFM
  marshalling surface as ShortArray-of-raw-patterns twins of the §0.4.354
  F64 pair (`bufferFromHostBf16`/`bufferToHostBf16`,
  `PjrtBuffer.toBf16Array`, `PjrtSession.runOnBf16` — an all-BF16 lane
  with NO numeric conversion in either direction), and the staged
  buffer's on-device size is 2 bytes/element — TRUE bf16 device storage,
  not silently widened f32; **(2) CAST-AT-BOUNDARY** — f32 params,
  in-graph casts, bf16 compute, f32 outputs, riding the existing `runOn`
  lane (the mixed-precision-training shape). MEASURED CLAIMS
  (PjrtBf16SmokeTest, assumeTrue self-skip off-GPU): device f32→bf16
  narrowing BIT-EXACT vs the §0.4.455 RNE helper on a 19-lane sweep
  (both tie directions, one-ulp neighbors, signed zeros, overflow→inf,
  subnormal flushes) — probed through the dtype-agnostic `executeOn`
  lane because of the first finding: **XLA's simplifier FOLDS an
  f32→bf16→f32 convert pair to identity** (the naive round-trip program
  came back raw), so only a single-convert program measures the device;
  bf16 matmul+add on native buffers BIT-EXACT vs the interpreter on
  bf16-exact lanes; the 257-tie discriminator ANSWERS G1b's open
  per-intermediate-rounding question: XLA-CUDA rounds the dot output to
  bf16 at the op boundary (256 — the interpreter's per-op-snap
  convention, not fused-f32's 258); a DxirReverseTransform gradient
  graph with bf16 compute (CastRule straight-through adjoint) runs on
  GPU, exact lanes bit-exact, non-zero, and the one honest divergence
  PINNED both ways: the device adjoint reads RAW x where the
  interpreter reads snap(x) — XLA folds mul-by-one then elides the
  narrow→widen pair — measured 0.0028125, exactly the snap error,
  bounded at one bf16 ulp (2⁻⁸ relative), the recorded tolerance for
  graphs whose narrowings are elidable. NaN narrowing pinned as `isNaN`
  only (convert payload propagation is target-defined). OPEN FOR TPU
  (G2b): re-run this suite on libtpu — the convert-fold and
  op-boundary-rounding results are XLA-CUDA measurements that may
  legitimately differ; TPU tiled buffer layout unverified; a typed
  mixed-dtype session lane stays a named deferral (`executeOn` covers
  the shape untyped). Suite 2067 → 2071.

- **§0.4.456 — G1b DONE: bf16 through the IR.** The §0.4.455 emitter
  refusals lift; bf16 is now a first-class IR dtype end-to-end SHORT OF
  DEVICE EXECUTION (G1c owns that claim). INTERPRETER: value arrays stay
  FloatArray — THE CONVENTION, stated loudly at `snapToBf16`: a
  bf16-typed node's values are the f32-WIDENED FORMS OF BF16-ROUNDED
  numbers, enforced by a central RNE snap in `evalNode` (idempotent, so
  the explicit CAST arm double-snapping is a no-op); ops with bf16
  results compute in f32 and round ONCE at their own output — XLA's
  "f32 accumulate, bf16 result" dot/reduce convention; per-intermediate
  device-granularity rounding is NOT simulated (a G1c certification
  question). CAST arms: f32→bf16 narrows via the §0.4.455 RNE helpers,
  bf16→f32 is the exact-copy widening, bf16→int truncates like the
  other floats; bf16 params snap on binding. AD: CastRule's §0.4.427
  float set gains BF16 — STRAIGHT-THROUGH (the adjoint of the narrowing
  cast is the widening cast of the upstream and vice versa; RNE is
  piecewise-identity, PyTorch-autocast/JAX convention) — without it the
  rule returned an EMPTY contribution: a silent zero gradient, the
  exact north-star failure mode; the forward CAST arm covers bf16
  unchanged (tangent casts with the primal). EMITTER: `bf16` element
  type in MlirCommon, casts emit `stablehlo.convert`, elementwise/
  matmul/reduce ride the type spelling; the enumerated f32-assumption
  arms: STEP + ARGMAX fold bf16 into their FLOAT branches, MAX/MIN init
  literals get width-matched 16-bit hex patterns (0xFF80/0x7F80), RNG
  draws now REFUSE bf16 BY NAME (the threefry mantissa trick is a
  binary32 bit-stream contract — draw at f32 and CAST). DxirCanonical
  parses "bf16" types (reverse-CSE keys carry result types); bf16
  CONSTANTS stay refused — spell them CAST(f32 const) (named deferral,
  no producer exists). KotlinSourceRenderer keeps the NAMED refusal,
  now pointing at the story: bf16 host math is compute-in-f32, so the
  readable reverse of a bf16 program is the f32 graph between the
  casts; the grad{} plugin frontend's bf16 nulls are RATIFIED as the
  convention, not a gap. Oracles: interpreter casts bit-pinned vs the
  G1a helpers; a cast-in/add/matmul/cast-out program with hand-derived
  bf16-exact expected values (plus a dedicated output-rounding pin:
  bf16 add 256 + 1.0078125 → 258); reverse-through-casts gradient
  pinned at hand values; six structural MLIR pins (types, convert both
  directions, bf16 add/dot_general/reduce/STEP, the 0xFF80 init, the
  RNG refusal). Suite 2056 → 2067.

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
