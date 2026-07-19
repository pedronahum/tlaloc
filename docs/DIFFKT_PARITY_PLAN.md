# DiffKT Parity Plan

**Status: ACTIVE (opened 2026-07-19, post-§0.4.364).** Goal per Pedro:
support everything [facebookresearch/diffkt](https://github.com/facebookresearch/diffkt)
supports that Tlaloc doesn't yet.

## Where parity already stands (closed §0.4.359–364)

| DiffKT capability | Tlaloc status |
|---|---|
| Reverse-mode AD (vjp/pullback) | ✅ `DxirReverseTransform` + runtime synthesis + compile-time probe |
| Forward-mode AD (jvp/pushforward) | ✅ `DxirForwardTransform` (§0.4.361) — **IR-level only, no user intrinsic yet** |
| Higher-order (hessian-vector) | ✅ pinned `forward(reverse(f))`; other nestings untested |
| conv2d + gradients | ✅ §0.4.362 (groups + lhs-dilated primal VJP deferred) |
| maxPool/avgPool + gradients | ✅ §0.4.363 (overlapping maxpool VJP deferred) |
| select / comparisons in grad lambdas | ✅ §0.4.364 |
| Elementwise tensor arithmetic in grad lambdas | ✅ §0.4.364 (`plus/minus/times/div`) |
| reshape/transpose/concat/slice/pad/broadcast + VJPs | ✅ IR level (§0.4.359–360) — **user surface partial** |
| softmax/logsumexp/max/min reductions + VJPs | ✅ IR level — **user surface partial** |
| Compile-time shape checking (ShapeTyping plugin) | ✅ richer: named indices + `validateDxirShapes` + real reverse-transform probe at check time |
| Float64 | ✅ PJRT path (§0.4.354) |

## The gap list, prioritized

Ordering principle: **A before C** — most of DiffKT's practical surface
already exists in Tlaloc's IR with rules and emitter lowerings; the fastest
parity wins are FIR-lowering + synthesis arms that make the existing IR
reachable from `grad {}`, not new math. New-op families come after.

### Phase A — user-surface completion (IR is ready; lowering/synthesis work)

- **A1. Axis-wise reductions**: `sum(dims, keepDims)` / `mean(dims)` /
  `max(dims)` / `min(dims)` user ops → the existing `reduction_dims`-attr'd
  IR. (DiffKT: `sum(axes, keepDims)`.) MeanRule today is full-reduce-only —
  needs the keepdims arm.
- **A2. Shape ops in lambdas**: `reshape`, `transpose(perm)`, `concat`,
  `slice`, `pad`, `stack` (sugar over CONCAT+RESHAPE), `squeeze`/`unsqueeze`
  (sugar over RESHAPE), `broadcastTo`. All have VJPs + evals + emitter
  spellings already.
- **A3. NN ops in lambdas**: `softmax(axis)`, `logSoftmax` (compose
  LOGSUMEXP), `conv2d`, `maxPool`, `avgPool`. Synthesis arms call the
  interpreter-backed `:core` hosts (new host impls needed for conv/pool —
  or route through `DxirInterpreter.evalFunction`-style helpers).
- **A4. Elementwise binary max/min + clip**: named `maximum/minimum/clip`
  ops as sugar over the §0.4.364 where/compare surface (DiffKT has them
  first-class; we compose).

### Phase B — AD-mode parity

- **B1. Forward-mode user intrinsics**: `jvp {}` / `valueAndJvp {}`
  (DiffKT `forwardDerivative` / `primalAndForwardDerivative`) — the
  §0.4.361 transform wired into GradIntrinsics + the plugin rewrite.
- **B2. `jacobian` + `hessian` intrinsics**: jacobian via forward (wide) or
  reverse (tall) column/row assembly; hessian = forward-over-reverse
  (already pinned at IR level).
- **B3. Forward transform through regions**: IF/WHILE bodies + COARSENED
  (tangent of a coarsened op = forward transform of its `primal_body`) —
  mirrors reverse-mode's history.
- **B4. Nesting matrix**: certify forward∘forward (2nd directional),
  reverse∘forward, reverse∘reverse against analytic references. DiffKT
  supports arbitrary nesting; we've pinned one composition.
- **B5. User-defined custom derivatives**: a user-facing custom-VJP/JVP
  registration (DiffKT lets users supply derivatives for opaque functions;
  our coarsener `gradient_body` machinery is the internal analogue —
  surface it).

### Phase C — op families DiffKT has that the IR lacks

- **C1. Special functions**: `LGAMMA`, `DIGAMMA`, `POLYGAMMA` (DiffKT ships
  these; its Dirichlet example depends on them). Host: Lanczos/series impls;
  emitter: `chlo.lgamma`/`chlo.digamma`; VJPs: `d lgamma = digamma`,
  `d digamma = polygamma(1)`, `d polygamma(n) = polygamma(n+1)`.
- **C2. Trig/rounding tails**: `TAN`, `ATAN` (DiffKT has both), `FLOOR`/
  `CEIL`/`ROUND` (zero-gradient, STEP-style). `atan2` if the audit finds it.
- **C3. `REVERSE` (flip) op**: DiffKT `flip`; also lets the conv adjoint
  drop its `window_reversal` special-casing eventually.
- **C4. Item-4 tails**: CONV_TRANSPOSE2D's own VJP, overlapping-window
  maxpool VJP (select_and_scatter emission or one-hot decomposition),
  grouped/depthwise conv (feature_group_count > 1).

### Phase D — random (DiffKT `RandomKey` parity)

- **D1. Stateless PRNG**: key type + `split`, `uniform`, `normal` ops
  (`stablehlo.rng_bit_generator` + box-muller or threefry, matching the
  JAX-style counter-based design DiffKT mirrors); interpreter twin.
- **D2. Reparameterized gradients**: gradients flow through loc/scale of
  sampled normals (DiffKT's Gamma/Dirichlet implicit reparameterization is
  the stretch goal — needs C1 first).

### Phase E — sparse (DiffKT `SparseFloatTensor` parity)

- **E1. Audit-first**: DiffKT's sparse surface is narrow (COO float tensors,
  sparse×dense matmul, aimed at GNN adjacency). Decide honest scope after
  the audit; likely a `:core` sparse type + SPARSE_MATMUL op with CPU host
  eval, GPU via scatter/gather composition.

### Phase 0 (do first, cheap) — the audit

Walk DiffKT's `ops` surface (`RawTensorOperations` / `ops.kt` op list +
`DScalar` math + the tutorials/examples dir) and produce a
checked-off inventory table in this doc, so "everything" is a closed list
pinned to the actual repo rather than memory. Anything the audit surfaces
that this plan missed gets slotted into A–E.

## Suggested § sequencing

§0.4.365 Phase 0 audit → §0.4.366+ Phase A (one § per slice, A1→A4) →
Phase B (B1/B2 together, then B3, B4, B5) → C1–C4 → D → E.

Certification discipline per CLAUDE-memory: solo full-suite runs, count
gate updated per §, GPU smokes for anything touching the emitter.
