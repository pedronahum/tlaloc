# The single-AD-engine audit (2026-09-21)

**Status: findings RATIFIED by direction (Pedro, 2026-09-21: "why do we
have a tape when we invested so much effort with the compiler … I don't
want this to be a loose end"), fixes executing as the cleanup arc.** The
governing product statement is on file in the same conversation: *Tlaloc
is an improved Tangent and an improved DiffKT — the reverse code must be
readable by the user, and compiled.*

## Running record

- **§0.4.448 — C CLOSED (demoted, loud, by name).** The five half-alive
  kinds now refuse with named messages that state the kind AND the
  sanctioned alternative, via one shared table
  (`ir/.../passes/DemotedOpKinds.kt`: `DEMOTED_OP_KINDS` +
  `demotedKindRefusal(kind, layer)`). LAYERNORM / SDPA / SPLIT refuse in
  `DxirInterpreter` (a named arm ahead of the generic bridge else),
  `DxirReverseTransform` (in the pre-scan, AHEAD of the generic
  multi-result gate that used to catch SPLIT with no directions, and ahead
  of the walk's index-0 upstream lookup; `walkBranchReverse` carries the
  same guard for IF-branch bodies), and `DxirForwardTransform` (first
  branch of the walk's op dispatch, ahead of the generic multi-result
  out-of-scope error). ALL_REDUCE / SHARD_CONSTRAINT refuse as
  non-differentiable BY DESIGN in both transforms only — their demotion is
  about differentiability, so the interpreter keeps its generic
  unsupported-op arm. Recognition (cost model, FlashAttentionRecognizer)
  and StableHLO emission are untouched: those remain the kinds' sanctioned
  roles, now stated in each OpKind doc comment with the rejected
  complete-the-kinds alternative recorded. Pins:
  `DemotedKindRefusalTest` — one test per kind per refusing layer (13),
  each asserting the message names the kind and the alternative; the SPLIT
  reverse pin consumes result index 1 only, the exact shape whose gradient
  a silent index-0 skip would have dropped. **SPLIT deletion
  RECOMMENDED:** nothing outside emitter/IR-plumbing tests constructs it
  (the FIR fold covers user splits) — the kind is kept with refusals in
  place; removal is Pedro's call.

- **§0.4.447 — B CLOSED (one named twin-gap).** The ~24 private FloatArray
  forward computations in `TracedOps.kt`'s pre-F4 spellings now route through
  the certified `io.tlaloc.core.ops` host twins: the elementwise binaries
  (ADD/SUB/MUL/DIV/POW → the DTensor operators + `pow`), the unaries
  (NEG/RELU/STEP/SQRT/EXP/LOG/TANH/SIGMOID → their extension twins), rank-2
  `matmul`, the full and axis reductions (SUM/MEAN → `sum()`/`mean()`/
  `sum(vararg dims)`/`mean(vararg dims)`; the private `reduceOverAxes` helper
  deleted), and the whole broadcast family (`broadcastScalar` →
  `broadcastLike`; `broadcastRow`/`broadcastInner`/`broadcastBatch` →
  right-aligned `broadcastToLike`; `broadcastCol`/`broadcastAlong` →
  `broadcastToLike` over a dims-only `[M,1]` / ones-except-axis relabel of the
  same storage). Bit-identity was free by construction (both sides already
  called the identical `kotlin.math` Float overloads in the identical order)
  and is now PINNED: `TracedOpsHostTwinParityTest` raw-bit-compares
  (`Float.toRawBits`) every migrated op's traced value against the host twin
  on quarter-grid AND irrational-constant inputs, plus hand-derived literal
  anchors for the exact ops (add, matmul, the broadcast copies).
  **The one named twin-gap: `bmm`** — `:core` has no rank-3 batched-matmul
  host twin (its `matmul` is rank-2 only), so `bmm` keeps its private loop
  (byte-for-byte the rank-2 twin's skip-zero walk, per-batch), commented as
  the twin-gap at the site and guarded by a hand-derived bit-pin; route it
  through a future `bmmGeneral` twin when one lands. NOT migrated, and not
  twin-gaps: `reshape`/`slice`/`concat` carry no arithmetic (pure copy walks,
  the F6 interpreter-mirror discipline — `:core`'s `sliceAtLike`/`padToLike`
  are template-shaped adjoint helpers, not forward twins), and the F4–F6
  spellings (conv2d/pools/embedding) already called their twins.

- **§0.4.446 — A CLOSED, E CLOSED.** `Backward.kt` deleted; `Grad.kt`'s
  entire Tracer-convenience family (`grad`/`grad2`/`grad3`,
  `valueAndGrad{,2,3}`, `gradWithScalar{,s}`, `valueAndGradWithScalar{,s}`)
  reimplemented over capture → `Tape.toDxirFunction` →
  `DxirReverseTransform(includeForward = true)` → `DxirInterpreter`,
  signatures unchanged, `GradTest` green with ZERO value edits. The tape's
  one local gradient arm (STEP ≡ 0) moved into `VjpRegistry`
  (`STEP → SignRule` — the identical identically-zero adjoint), so
  `step()` in a primal stays differentiable-to-zero under one engine, on
  BOTH routes. Pins: `OneEngineParityTest` (E2E through the real K2
  plugin, no stub — the same function on the Tracer surface and the
  `grad {}`/`grad2 {}` intrinsic surface produces RAW-BIT-IDENTICAL
  gradients), `SingleEngineDeletionTest` (`Class.forName` absence of
  `BackwardKt`/`Gradients`), `GradTest.stepHasIdenticallyZeroGradient`
  (primitive-compare zeros). `DxirBridgeEquivalenceTest` re-scoped: it now
  pins that a lambda-built tape captures faithfully against hand-built IR,
  not that two engines agree. E: "tape fallback" phrasing corrected across
  `DIFFKT_PARITY_PLAN.md` / `STAGE_B_PLAN.md` / the §0.4.58 harness
  comments to the `pluginMissing` reality (cert sentences now read
  "no synthesis fallback" / "no silent fallback"). Deferral: the
  `DScalarMixingGradientTest` test-name string "falls back to the tape"
  kept (renaming a certified test's name adds no value; its doc already
  reads as kept-original-call).

## Findings

**A. The runtime value-tape is a self-contained island.**
`Backward.kt`'s `backward()` is called only by `Grad.kt`'s
Tracer-convenience API (`grad`/`grad2`/`grad3`/`gradWithScalars` over
`Tracer` lambdas), which is exercised only by its own `GradTest.kt` — no
benchmark or production consumer. To its credit it never duplicated
gradient math (every kind routes through the `applyRegistryRule` bridge
into the real `VjpRegistry`; only STEP has a local zero arm). But it is
a second AD engine in the tree. **Fix: reimplement `Grad.kt`'s API over
capture → `Tape.toDxirFunction` → `DxirReverseTransform` →
`DxirInterpreter` (same signatures), then DELETE `Backward.kt` and the
bridge.** One engine — the compiler's. (Phase F's `:nn` route already
proves the pattern end to end, §0.4.437.) **DONE §0.4.446** — see the
running record.

**B. `TracedOps` carries private forward math.** ~22 elementwise
`FloatArray` loops — a third implementation alongside the interpreter
arms and the `:core` host twins, with no certified-equality pin. (The
Phase F TRACE spellings, F4–F6, already call host twins — the older
elementwise spellings predate that discipline.) **Fix: route the older
spellings through the `:core` host twins; certify bit-equality.**
**DONE §0.4.447** (one named twin-gap: `bmm`) — see the running record.

**C. Half-alive OpKinds.** `LAYERNORM` and
`SCALED_DOT_PRODUCT_ATTENTION`: priced by the cost model, LAYERNORM
emits, SDPA is recognizer-accepted — but no interpreter arm, no VjpRule,
no forward tangent. `SPLIT`: emitter-only. A hand-built graph using them
is silently un-runnable on host and un-differentiable. **Fix: DEMOTE,
don't complete** (the coarseners own layernorm/attention semantics;
completing the kinds would duplicate certified work): loud named
refusals in the interpreter and both transforms, doc notes on the kinds,
pins. `ALL_REDUCE`/`SHARD_CONSTRAINT` are non-differentiable by design
(sharding constructs) — document that, refuse by name in the
transforms. `SPLIT` follows the same demotion (the FIR fold covers
users). **DONE §0.4.448** (SPLIT deletion recommended, kept pending
Pedro's call) — see the running record.

**D. Dual `grad` naming.** Tracer-lambda `grad` vs intrinsic `grad`
share names, disambiguated by lambda type. Documented and load-bearing
(§0.4.424's landmine); after fix A both live on one engine, so this
becomes cosmetic. No action beyond doc clarity.

**E. "Tape fallback" language drift.** The intrinsics throw
`pluginMissing` with guidance at runtime when synthesis falls back —
there is no silent tape fallback for `grad {}` lambdas. Several records
say "tape fallback"; sweep the phrasing. **DONE §0.4.446** — see the
running record.

## Not a mess (deliberate dual implementations, certified)

Interpreter vs `:core` host twins (KMP-common vs host, bit-equality
pinned); the emitter's inline decompositions (per-backend necessity,
oracle-pinned); the KPTX backward kernels (a performance lane with XLA
oracles).

## The readable-reverse requirement (the Tangent inheritance)

Two surfaces, queued with the cleanup arc:
1. A `DxirFunction → Kotlin source` pretty-printer whose output renders
   ops as their `:core` host-twin calls — readable AND compilable AND
   runnable (beyond Tangent: the printed derivative is a working
   function). Golden test: compile and execute the printed source of a
   gradient function, match numbers.
2. A K2 plugin dump option rendering each synthesized `grad {}` gradient
   body as Kotlin source at compile time, next to the lambda it
   differentiates.
