# The single-AD-engine audit (2026-09-21)

**Status: findings RATIFIED by direction (Pedro, 2026-09-21: "why do we
have a tape when we invested so much effort with the compiler … I don't
want this to be a loose end"), fixes queued as the cleanup arc.** The
governing product statement is on file in the same conversation: *Tlaloc
is an improved Tangent and an improved DiffKT — the reverse code must be
readable by the user, and compiled.*

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
proves the pattern end to end, §0.4.437.)

**B. `TracedOps` carries private forward math.** ~22 elementwise
`FloatArray` loops — a third implementation alongside the interpreter
arms and the `:core` host twins, with no certified-equality pin. (The
Phase F TRACE spellings, F4–F6, already call host twins — the older
elementwise spellings predate that discipline.) **Fix: route the older
spellings through the `:core` host twins; certify bit-equality.**

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
users).

**D. Dual `grad` naming.** Tracer-lambda `grad` vs intrinsic `grad`
share names, disambiguated by lambda type. Documented and load-bearing
(§0.4.424's landmine); after fix A both live on one engine, so this
becomes cosmetic. No action beyond doc clarity.

**E. "Tape fallback" language drift.** The intrinsics throw
`pluginMissing` with guidance at runtime when synthesis falls back —
there is no silent tape fallback for `grad {}` lambdas. Several records
say "tape fallback"; sweep the phrasing.

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
