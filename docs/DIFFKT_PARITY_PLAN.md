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

- **A1. Axis-wise reductions** ✅ (§0.4.366): `sum(dims, keepDims)` /
  `mean(dims)` / `max(dims)` / `min(dims)` user ops → the existing
  `reduction_dims`-attr'd IR, squeezed AND keepdims shapes, E2E through
  `grad {}`. Landed along the way: MEAN interpreter arm, sentinel-safe
  MeanRule (runtime-N graph), type-aware CSE in the reverse transform
  (result types joined the dedup key — a latent wrong-shape bug), the
  synthesis reduction/unsqueeze/stretch arms + backward IrType solve for
  mixed-rank gradient bodies. v1 scope: 1–2 axes from `grad {}` (the
  fixed-arity synthesis delegates); IR level is fully general.
- **A2. Shape ops in lambdas** — split by the sentinel-dims boundary:
  - **A2a ✅ (§0.4.367)**: the RESHAPE family (`squeeze(axis)` /
    `unsqueeze(axis)` / `flatten()` / `reshape(vararg dims)`) +
    `transpose(vararg perm)` (and the no-arg rank-2 receiver spelling),
    E2E through `grad {}`. Axis positions, perms, and user-literal dims
    are compile-time constants — sentinel-safe. Uniform synthesis rule
    landed: concrete dxir dims bake as consts, -1 sentinels read
    axis-matched `param.dims` at runtime. `flatten` is IR-level-only for
    gradients (its splat needs a rank-1 dim = PRODUCT of param dims —
    deferred with A2b).
  - **A2b (open)**: `concat`, `slice`, `stack`, `pad` (note: DiffKT has
    no user-facing pad — ours would be a bonus), `broadcastTo`/`expand`,
    `view`/indexing, `withChange`, `meld`/`split`, `stats`. Blocked on
    runtime-extent adjoints: ConcatRule/SliceRule bake operand extents
    into SLICE/PAD attrs, which are -1 sentinels inside `grad {}` — the
    adjoints need either runtime-shaped slice ops (a `sliceLike` host
    family + attr-free IR spelling) or SPLIT (which today is
    emitter-only: no interpreter arm, no VJP, no forward arm).
- **A3. NN ops in lambdas**: `softmax(axis)`, `logSoftmax` (compose
  LOGSUMEXP), `conv2d`, `maxPool`, `avgPool`. Synthesis arms call the
  interpreter-backed `:core` hosts (new host impls needed for conv/pool —
  or route through `DxirInterpreter.evalFunction`-style helpers).
- **A4. Elementwise binary max/min + clip + outerProduct**: named
  `maximum/minimum/clip` ops as sugar over the §0.4.364 where/compare
  surface (DiffKT has them first-class; we compose); `outerProduct` as
  unsqueeze+broadcast-MUL sugar.
- **A5. Binary-op broadcasting + orphaned lowerings** *(audit)*:
  implicit broadcasting on tensor binary ops (`broadcast(S1,S2)` — DiffKT
  broadcasts everywhere) and `Float×DTensor`/`DScalar×DTensor` mixing;
  plus map entries for ops whose IR+VJP already exist but aren't
  reachable: tensor/scalar `pow`, scalar `tanh`/`sigmoid`, tensor
  `mean` (dispatch arm exists, map entry missing).

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
- **C2. Trig tails**: `TAN`, `ATAN` — audit confirmed these are the only
  ones DiffKT has (no floor/ceil/round/atan2; those stay optional extras,
  not parity items).
- **C3. `REVERSE` (flip) op**: DiffKT `flip`; also lets the conv adjoint
  drop its `window_reversal` special-casing eventually.
- **C4. Item-4 tails** *(reclassified beyond-parity by the audit —
  DiffKT pooling is non-overlapping-only, conv has no groups/dilation)*:
  CONV_TRANSPOSE2D's own VJP, overlapping-window maxpool VJP
  (select_and_scatter emission or one-hot decomposition),
  grouped/depthwise conv (feature_group_count > 1).
- **C5. `integral`** *(audit)*: Romberg quadrature with FTC-wired
  forward/reverse derivatives (DiffKT `Integral.kt`).

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

### Phase 0 — the audit (✅ DONE, §0.4.365, 2026-07-19)

Walked `facebookresearch/diffkt` (shallow clone @ HEAD,
`kotlin/api/src/main/kotlin/org/diffkt/`): every top-level public
function, the `Operations` interface (the raw-ops list), the `random/`,
`model/`, sparse, and examples surfaces. The closed inventory follows;
**audit deltas** (what this plan had wrong or missing) are folded into
the phase items above and summarized at the end of this section.

Legend: ✅ full parity (user surface + gradients) · 🟡 IR-level only
(op + VJP + emitter exist; not reachable from `grad {}`) · ❌ missing ·
➖ DiffKT-only quirk we don't need (noted why).

#### Differentiation API (`ForwardDerivative.kt` / `ReverseDerivative.kt` / combinators)

| DiffKT | Tlaloc | Notes |
|---|---|---|
| `reverseDerivative` / `primalAndReverseDerivative` (1/2-arg, List, n-th `reverseDerivative{1..4}`, `reverseDiff`) | ✅/🟡 | `grad {}` covers 1st-order; n-th-order = nesting (B4) |
| `forwardDerivative` (all arities, n-th, `forwardDiff`) / `primalAndForwardDerivative` | 🟡 | `DxirForwardTransform` §0.4.361; no user intrinsic → B1 |
| `jvp` / `primalAndJvp` | 🟡 | same → B1 |
| `vjp` / `primalAndVjp` / `primalAndPullback` (user-supplied cotangent, `vf(primal)` form) | 🟡 | reverse transform takes unit seed today; expose seeded pullback → B1/B2 |
| Jacobian assembly | 🟡 | DiffKT has **no** jacobian intrinsic — `reverseDerivative(x, f: tensor→tensor)` identity-seeds and builds the full Jacobian (`identityGradientOfSameKind`). B2 = that seeding loop |
| `reverseDerivativeTransposed` | ❌ | transposed-Jacobian convention variant; fold into B2 |
| Arbitrary nesting (fwd∘fwd, rev∘rev, …) | 🟡 | one composition pinned (HVP) → B4 |
| `ifThenElse(cond, a, b)` (scalar + tensor, differentiable) | ✅ | `where` §0.4.364; scalar branches also via IF regions + coarsening |
| `Wrappable`/`Wrapper` (derivatives through user data structures; examples lean on this) | 🟡 | Tlaloc's K2 plugin lowers data-class params structurally — different mechanism, same end; certify in B5 |
| `integral(a, b, f)` — Romberg quadrature with FTC-wired fwd/rev derivatives | ❌ | genuinely novel; **new C5** |
| `primal(x, f)`, `basePrimal`, `DerivativeID` plumbing | ➖ | runtime-tape bookkeeping; no analogue needed in a compile-time IR |

#### Scalar math (`DScalar` surface)

`+ - * / unaryMinus` ✅ · `abs sqrt exp ln sin cos relu` ✅ ·
`compareTo`/`eq ne lt le gt ge` ✅ (Kotlin comparisons + IF lower today) ·
`tanh sigmoid pow` 🟡 (IR + VJP exist; **not in the scalar
UNARY/BINARY maps** → A5) · `tan atan` ❌ (C2) ·
`lgamma digamma polygamma` ❌ (C1) · `sigmoid(DScalar)` 🟡 (same A5).

#### Tensor ops (top-level files + `Operations` interface)

| DiffKT | Tlaloc | Notes |
|---|---|---|
| `plus minus times div unaryMinus` (elementwise) | ✅ | §0.4.364 — **same-shape only**; DiffKT broadcasts every binary op (`broadcast(S1,S2)`) and mixes `DScalar×DTensor` / `Float×DTensor` (`timesScalar`) → **new A5** |
| `pow(Float/Int/DScalar/tensor-exponent)` | 🟡 | POW + VjpRule + fwd rule all exist; no lowering entry → A5 |
| `eq ne lt le gt ge` (tensor masks) | ✅ | §0.4.364 |
| `relu reluGrad sigmoid tanh exp ln sqrt abs` (tensor) | ✅ | `reluGrad` is public in DiffKT; ours is internal — fine |
| `sin cos tan atan` (tensor) | ✅/❌ | sin/cos ✅; tan/atan ❌ → C2 (audit: **no** floor/ceil/round/atan2 in DiffKT — plan over-scoped C2; now ours-optional) |
| `lgamma digamma polygamma` (tensor) | ❌ | C1 (Dirichlet example + gamma reparam depend on them) |
| `sum()` full-reduce | ✅ | |
| `sum(axes, keepDims)` | 🟡 | `reduction_dims` IR exists → A1 |
| `mean()` | 🟡 | dispatch arm exists but **no map entry** — not reachable → A1 |
| `FloatTensor.max/min(axes)` | ➖/🟡 | DiffKT only has these on **FloatTensor — not differentiable**; Tlaloc's MAX/MIN have VJPs → A1 exceeds parity |
| `stats()` = (mean, variance) | ❌ | 2-line sugar once A1 lands |
| `matmul` (incl. generalized shape-block form) | ✅ | any rank ≥ 2 |
| `innerProduct` | ✅ | DOT |
| `outerProduct` | ❌ | sugar (broadcast-MUL or matmul on unsqueezed) → A4 |
| `matdiv` | ➖ | **sparse-only** in DiffKT (dense explicitly unsupported) → E |
| `conv2d(hStride, vStride, Same/Valid/Explicit padding)` | ✅ | §0.4.362 **exceeds**: DiffKT has no groups/dilation, NHWC only |
| `maxPool / avgPool / maxPoolWithIndices` | ✅ | §0.4.363 **exceeds**: DiffKT pooling is non-overlapping only (stride=window, divisibility required, no padding) → C4 reclassified beyond-parity |
| `batchNorm` (raw op, training-stats variant) | 🟡 | BATCHNORM OpKind exists; VJP + surface unaudited — fold into A3 |
| `softmax(axis) / logSoftmax / logSoftmaxGrad` | 🟡 | SOFTMAX/LOGSUMEXP + VJPs exist → A3 |
| `crossEntropyLoss / crossEntropyLossFromOneHot / nllLossFromOneHot` | 🟡 | CROSS_ENTROPY OpKind exists (no VjpRule); or compose from logSoftmax → A3 |
| `embedding(table, indices, paddingIndex)` | 🟡 | EMBEDDING OpKind exists, **no VjpRule in registry** → A3 |
| `reshape / flatten(startDim) / squeeze / unsqueeze / expand / broadcastTo` | 🟡 | RESHAPE/BROADCAST + VJPs → A2 |
| `transpose(axes) / leftTranspose / rightTranspose` | 🟡 | TRANSPOSE + VJP → A2 (left/right = sugar) |
| `concat / stack / split / meld` | 🟡 | CONCAT/SPLIT + VJPs → A2 (`meld` = flatten-and-concat sugar; inverse `split`) |
| `slice / view(index/range/axis) / withChange` (functional update) | 🟡 | SLICE/GATHER/SCATTER + VJPs → A2 (indexing + `withChange` = slice/scatter sugar) |
| `gather / scatter (axis, paddingIndex) / gatherAtIndices / scatterAtIndices` | 🟡 | Tlaloc GATHER/SCATTER are narrower (rank-1/scalar-index arms) — A2 needs the axis+list form |
| `flip(axes)` | ❌ | C3 (REVERSE op; also cleans conv adjoint) |
| `IntTensor / intTensorOf / Float64` | ✅ | I32/I64/F64 dtypes; DiffKT is F32-only + int tensors — Tlaloc exceeds on F64 |
| `Device (CPU/GPU per-tensor placement)` | ➖ | Tlaloc's backend story (PJRT/IREE) supersedes |
| tracing/JIT package | ➖ | the entire Tlaloc compile pipeline is the superset |

#### Random (`random/`)

`RandomKey` (SHA-512 counter-based) + `split(n)` + `permitReuse` +
`floats/uniform/gaussian` + distributions: `cauchy`, `chiSquare`,
`gamma` (with-rate/with-scale, **implicit reparameterization** — the
raw-op `gamma(alpha, randomKey)` participates in AD), `DiffktRandom`
wrapper, `Wrapper.wrapRandomKey`. All ❌ → Phase D (D1 key+uniform+
gaussian; D2 reparam; cauchy/chiSquare are inverse-CDF sugar over
uniform; gamma reparam needs C1's digamma).

#### Sparse (`SparseFloatTensor` / `SparseRowFloatTensor`)

COO-ish float tensors, ops actually implemented: `plus minus times
transpose matmul matdiv` (matdiv = square-RHS solve, sparse-only).
All ❌ → Phase E, audit-scoped as suspected: narrow GNN-adjacency
surface, not a general sparse algebra.

#### Model layer (`model/` package) — **new Phase F (product decision)**

DiffKT ships an NN-framework layer Tlaloc's plan never scoped: `Layer`/
`Model`/`Sequential`/`TrainableComponent`, layers (Dense, Conv2d +
SamePadding, MaxPool/AvgPool(2d), BatchNorm(+Training), Embedding +
EmbeddingBag, Dropout, GRU, Flatten, ReluLayer, AffineTransform),
initializers (FanMode), optimizers (SGD, Adam, RMSprop, Momentum,
FixedLearningRate). Tlaloc has its own NN surface (LAYERNORM, RMSNORM,
SDPA, LlamaDecoder) but no Layer/Optimizer API. **Pedro to decide**
whether DiffKT-style `model/` parity is in scope (Phase F) or whether
Tlaloc's module story supersedes it — not blocking A–E.

#### Audit deltas applied to this plan

1. **New A5** — implicit broadcasting + `Float×DTensor`/scalar-tensor
   mixing on binary ops, plus the orphaned lowerings: tensor/scalar
   `pow`, scalar `tanh`/`sigmoid`, tensor `mean` (map entry missing).
2. **A2 grew** — indexing sugar (`view`/`[]`), `withChange`, `flatten`,
   `meld`, `stats`; gather/scatter need the axis+list+paddingIndex form.
3. **A3 grew** — embedding VjpRule (op exists, rule doesn't),
   cross-entropy/NLL losses, batchNorm audit.
4. **C2 trimmed** — DiffKT has only TAN/ATAN; floor/ceil/round/atan2
   are not parity items (keep as optional extras).
5. **C4 reclassified beyond-parity** — DiffKT pooling is
   non-overlapping-only and conv has no groups/dilation; Tlaloc v1
   already exceeds DiffKT on both.
6. **New C5** — `integral` (Romberg + FTC derivatives).
7. **D detailed** — cauchy/chiSquare/gamma distributions, gamma
   implicit reparameterization (needs C1), `DiffktRandom` wrapper.
8. **E confirmed narrow** — six ops incl. sparse-only `matdiv`.
9. **New Phase F** — `model/` layers + optimizers; product decision.
10. **B2 clarified** — DiffKT's "jacobian" is identity-seeded reverse
    on tensor→tensor `f`; B1 should also expose the seeded-cotangent
    `vjp`/`primalAndPullback` form (DiffKT's pullback takes `vf(primal)`).

### Phase F — model/optimizer layer (audit-surfaced; product decision)

See the audit's model-layer section: DiffKT's `model/` package (Layer/
Sequential/Trainable + Dense/Conv2d/pooling/BatchNorm/Embedding/Dropout/
GRU layers + SGD/Adam/RMSprop/Momentum optimizers). Awaiting Pedro's
call on whether this is in scope or superseded by Tlaloc's own module
story. Not blocking A–E.

## Suggested § sequencing

§0.4.365 Phase 0 audit ✅ → §0.4.366+ Phase A (one § per slice, A1→A5) →
Phase B (B1/B2 together, then B3, B4, B5) → C1–C3+C5 (C4 optional) →
D → E → F (if ratified).

Certification discipline per CLAUDE-memory: solo full-suite runs, count
gate updated per §, GPU smokes for anything touching the emitter.
