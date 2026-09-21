# DiffKT Parity Plan

**Status: BOOK OF WORK CLOSED (§0.4.433 close-out, 2026-09-20; opened
2026-07-19, post-§0.4.364) — and the one gated phase, Phase F, RATIFIED
AND COMPLETED §0.4.434–444 (2026-09-21) per
[MODEL_LAYER_PLAN.md](MODEL_LAYER_PLAN.md).** See "End-state at
§0.4.433" below for what is complete, what is closed by ratified
refusal, and the one consolidated remaining-tails list; the Phase F
block inside it records the model-layer completion. Goal per Pedro:
support everything [facebookresearch/diffkt](https://github.com/facebookresearch/diffkt)
supports that Tlaloc doesn't yet.

## Where parity already stands (opened at §0.4.359–364; markers kept current — last sweep §0.4.433)

| DiffKT capability | Tlaloc status |
|---|---|
| Reverse-mode AD (vjp/pullback) | ✅ `DxirReverseTransform` + runtime synthesis + compile-time probe |
| Forward-mode AD (jvp/pushforward) | ✅ `DxirForwardTransform` (§0.4.361) + user intrinsics `jvp`/`valueAndJvp` (§0.4.372, B1) + 2-arg forms (§0.4.387) + regions/loops (§0.4.403, B3) + direct IF arm (§0.4.407) |
| Higher-order (hessian-vector) | ✅ full nesting matrix certified at IR level (§0.4.401): fwd∘rev, fwd∘fwd, rev∘fwd, rev∘rev + a third-order spot check; fused-adjoint refusals pinned. User `hessian`/`hessian2` intrinsics §0.4.394/406 |
| conv2d + gradients | ✅ §0.4.362; `grad {}` E2E §0.4.384–385; conv-transpose's own adjoint §0.4.391 + GPU emission §0.4.393; grouped/depthwise (feature_group_count) at IR level §0.4.429 — interpreter + CONV2D VJP/JVP + primal GPU emission; user surface deferred by name |
| maxPool/avgPool + gradients | ✅ §0.4.363; `grad {}` E2E §0.4.386/389 (overlapping-maxpool GPU emission closed as INHERENT §0.4.392 — host/interpreter handle it, StableHLO cannot express the all-ties convention) |
| select / comparisons in grad lambdas | ✅ §0.4.364 |
| Elementwise tensor arithmetic in grad lambdas | ✅ §0.4.364 (`plus/minus/times/div`) |
| reshape/transpose/concat/slice/pad/broadcast + VJPs | ✅ IR level (§0.4.359–360) + user surface: reshape family/transpose §0.4.367, broadcastTo §0.4.371/373, slice §0.4.374, concat/stack §0.4.381–382, flip §0.4.396, `view`/`withChange`/`meld` §0.4.428 (`split` host-level §0.4.428; its `grad {}` spelling is a named deferral — no `List<DTensor>` value model in the lambda lowering) |
| softmax/logsumexp/max/min reductions + VJPs | ✅ axis reductions in `grad {}` §0.4.366 (A1), softmax/logSoftmax §0.4.368 (A3a) — LOGSUMEXP stays emitter-only |
| Compile-time shape checking (ShapeTyping plugin) | ✅ richer: named indices + `validateDxirShapes` + real reverse-transform probe at check time |
| Float64 | ✅ PJRT path (§0.4.354) |

## End-state at §0.4.433 — the close-out

The book of work is CLOSED. Every phase this plan opened is complete,
closed by a ratified refusal, or gated on a product decision that is
Pedro's to make. The canonical end-of-book suite number is **1944**
(clean-room certified at the close-out; the ladder from the §0.4.417
sparse-arc opening ran 1850 → 1944 in sixteen sections, §0.4.417–432,
all landed 2026-09-20).

**COMPLETE:**
- **Phase 0** audit (§0.4.365).
- **Phase A** user surface (§0.4.366–397, 400, 409, 414, 427, 428 —
  axis reductions, the shape-op families, softmax/NN ops, embedding
  with paddingIndex + rank-2 batches, the full concrete scalar-param
  family, `view`/`withChange`/`meld`).
- **Phases B1–B4** (§0.4.372–412, 423, 424, 430 — `jvp`/`vjp`/
  `jacobian{,2}`/`jacobianReverse{,2}`/`hessian{,2}`/`grad3`, the full
  nesting matrix, fused-adjoint forward tangents, multi-result
  COARSENED tangents + IF-bearing `primal_body` splices).
- **Phase B5** custom derivatives (§0.4.415 reverse + §0.4.416 forward,
  ratified; design record in
  [CUSTOM_DERIVATIVES_DESIGN.md](CUSTOM_DERIVATIVES_DESIGN.md)).
- **Phase C** op families (§0.4.395, 396, 402, 405, 411, 426, 429 —
  tan/atan, flip, lgamma/digamma/polygamma, `integral` host + through
  the B5 gate, grouped/depthwise conv at IR level).
- **Phase D** random (§0.4.408, 413, 421, 422, 431, 432 — threefry-2x32
  bit-pinned vs JAX, reparameterized gradients, draws inside `grad {}`,
  explicit-threefry GPU emission, cauchy/exponential/chiSquare, runtime
  RNG keys as operands).
- **Phase E** sparse, per the ratified scope (§0.4.417–420 — `:core`
  host CSR `SparseTensor`, `SPARSE_MATMUL` + fused SDDMM adjoint,
  `ZEROS_LIKE` param-addressed zeros, the `grad {}` sparse surface;
  [SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md)).

**CLOSED BY RATIFIED REFUSAL** (decisions with papers, not tails — do
not re-litigate without new evidence):
- **sparse GPU emission** — the pinned emit refusal is the ratified E
  scope; ELL-padded emission is the recorded route IF a consumer ever
  forces the question.
- **`matdiv`** — SKIPPED; a solver arrives as its own designed feature
  or never (DiffKT's own is Eigen-JNI with `TODO()`s at its edges).
- **`rng_bit_generator`** — never used; explicit-threefry emission
  (§0.4.422, JAX's own approach) IS the design, not a stopgap.
- **overlapping/padded maxpool GPU gradients** — INHERENT (§0.4.392):
  StableHLO cannot express the all-ties convention;
  `select_and_scatter` picks one winner and would fork the gradient by
  backend.
- **`DScalar`-INTERFACE `grad {}` params** — structural refusal by name
  (§0.4.427); the concrete family (`Float`/`FloatScalar`/`DoubleScalar`)
  lowers.

**PHASE F: COMPLETE (§0.4.434–444, ratified AND landed 2026-09-21)** —
the model/optimizer layer, scoped in
[MODEL_LAYER_PLAN.md](MODEL_LAYER_PLAN.md) and AMENDED before any
substrate code landed (§0.4.436, Pedro's veto: **the AD route is the
COMPILER stack — trace → DXIR → `DxirReverseTransform` → interpreter or
StableHLO/PJRT — not the runtime value-tape**, which stays a debugging
fallback). The `:nn` module ships every DiffKT `model/` layer
(Dense/Conv2d+SamePadding/pools/Flatten/Relu/AffineTransform/BatchNorm/
Dropout/Embedding/EmbeddingBag/GRU), the FanMode initializers
(threefry, bit-deterministic), the five optimizers (DiffKT-exact SGD/
RMSprop quirks recorded by name; Adam is Kingma–Ba since DiffKT's is a
TODO placeholder), and the F8 end-to-end close: MLP + conv net train to
pinned convergence, the captured gradient graph runs compiled on the
GB10 matching the interpreter at 3.8e-5 with ONE cached executable
across steps, and 50 Adam steps track PyTorch at ~1e-7 relative from a
shared init. Row-sparse embedding gradients rode with it as dense v1
(the row-sparse CSR design is recorded in that doc's F6 entry); the
consolidated Phase F deferral list lives at the end of its §4 F8 entry.
**Nothing else is gated — the book of work has no remaining open
phase.** The post-F suite number is **2020** (§0.4.444 clean-room).

### Remaining tails, consolidated (§0.4.433)

Every still-open deferral in this document, in one place. Each is a
named, loudly-refusing gap sized to its own § when a consumer asks;
the per-phase entries below carry the mechanism detail.

*Phase A:*
- `split` inside `grad {}` — no `List<DTensor>` value model in the
  lambda lowering; spell per-piece `view`/`slice` (§0.4.428).
- multi-index `view(IntArray)` leading-index form; `withChange` beyond
  rank 3 (the `padToLikeRank{1,2,3}` arity bound) (§0.4.428).
- gather/scatter axis + index-list forms beyond the landed surface.
- the MIXED broadcast — a simultaneous rank-increase AND aligned size-1
  stretch in one op; the FIR fail-loud guard stands (§0.4.373).
- reductions over >2 axes from `grad {}` (fixed-arity synthesis
  delegates; IR level fully general) (§0.4.366).
- embedding indices produced by in-lambda integer ARITHMETIC (params
  only; the several-index-params half closed §0.4.419) (§0.4.409).
- `DScalar × DTensor` mixing and comparisons against a scalar literal
  (A5b neighbourhood).

*Phase B:*
- WHILE-bearing bodies through the COARSENED splice (IF closed
  §0.4.430; WHILE refuses by name).
- user (B5) multi-result `f` — unconstructible by
  `validateCoarsenedShape`; PhiCalculus coarsening is its route in
  (§0.4.430).
- **B5 `CHECK_SHAPE_LIKE` GPU emission — stays deferred; disposition
  recorded at the close-out.** The op is the runtime shape contract on
  user `customVjp` gradient returns (assert the value's runtime dims
  equal its template's, then alias the value through), and the emitter
  refuses it by name because StableHLO has no assert. An emission would
  have to spell assert-then-alias by hand — compare per-axis extents,
  then either poison the failing arm (a NaN splat behind a select) or
  trap through a `custom_call` — and every candidate either silently
  forks host/device behaviour (a dropped or NaN-masked check is not the
  interpreter's loud failure) or drags in a custom-call runtime
  dependency the emitter has nowhere to declare. What it concretely
  needs: an emission-strategy decision (poison vs trap), a
  PjrtSession-visible failure channel, and a GPU cert that a VIOLATED
  contract actually surfaces. Until then the pinned refusal IS the
  correct behaviour: user gradient bodies are host/interpreter-certified
  and a GPU run refuses loudly rather than running unchecked. (The KPTX
  native-runtime decision, if ever reopened, is the natural vehicle for
  the trap half.)
- B5 Candidate B (registration-form API) — deferred on the
  serialized-dxir decision (CUSTOM_DERIVATIVES_DESIGN.md §6).
- 3-arg forward-assembled family (`jacobian3`/`hessian3`/`jvp3`/`vjp3`)
  — the §0.4.406 pattern verbatim; pull when a consumer asks.
- rev-over-rev THROUGH fused-adjoint gradient bodies — the B4 refusal
  pins stand; the forward half closed §0.4.423, so second order
  composes fwd-over-rev today.
- n-th-order user intrinsics (`reverseDerivative{2..4}` spellings) — a
  synthesis-surface question, not an IR one; the compositions they
  would lower to are certified (§0.4.401).

*Phase C:*
- the INTEGRAL region op — runtime/tensor bounds via a
  quadrature-over-body-VJP adjoint; the literal-bounds `grad {}` v1
  stands (§0.4.426).
- grouped-conv tails (§0.4.429; each refuses naming the attr):
  transposed-conv grouped VJP, grouped adjoint EMISSION, the user
  surface, `batch_group_count`.
- conv `window_reversal` re-expression via REVERSE of the kernel's
  spatial axes — noted in `OpKind.kt`, deliberately not attempted.

*Phase D:*
- the FIR surface for the runtime-key OPERAND form — `RandomKey`-typed
  vals / lambda params / computed key words inside `grad {}`; the
  §0.4.421 literal-only fallback stays the loud gate (§0.4.432).
- an i32 host-buffer lane in `PjrtSession` (F32-only v1) — would let
  high-bit runtime keys ride as executable inputs (§0.4.432).
- `chiSquare`/`exponential` inside `grad {}` — the §0.4.431 cauchy
  pattern extended; waiting on a use.
- Gamma implicit reparameterization — the worked ∂z/∂α formula is on
  file (§0.4.431, D3).
- `permitReuse`/`DiffktRandom` wrapper sugar (D1 note).

*Phase E:*
- ELL-padded sparse GPU emission — behind the pinned refusal, only if
  ratified anew.
- row-sparse embedding gradients — a Phase F item.

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
    axis-matched `param.dims` at runtime. `flatten` was IR-level-only for
    gradients until §0.4.428's `irReshape` flatten arm (any rank-1
    relayout target reads its extent off the operand's runtime shape) —
    CLOSED there.
  - **A2b (partial ✅ §0.4.371 — `broadcastTo` landed; the rest deferred)**:
    - **`broadcastTo`/`expand` ✅ (§0.4.371)** — the *rank-increasing* form
      (NumPy right-alignment: the operand maps to the TRAILING output axes,
      new leading axes are replicated), E2E through `grad {}`. This is the
      sentinel-clean corner of A2b: the FIR lowering derives
      `broadcast_dimensions` as the right-aligned suffix `[outRank-rank ..
      outRank-1]` — pure compile-time POSITIONS from ranks alone (dim VALUES
      never consulted), so `BroadcastRule`'s adjoint (a SUM over the
      COMPLEMENT = the new leading axes) reads no operand extent and is
      sentinel-safe. Landed: the interpreter's BROADCAST arm generalized to
      full `stablehlo.broadcast_in_dim` (input axis j → output axis
      `broadcast_dimensions[j]`, size-1 stretch, new replicated axes),
      subsuming the §0.4.359 scalar-splat + equal-rank keepdims-stretch cases
      (the empty-`broadcast_dimensions` polymorphism is preserved: scalar seed
      splats, equal-rank input stretches by shape); host `DTensor.broadcastTo`
      (out[i] = v[i % n], since the operand rides the innermost axes); the FIR
      `SHAPE_OP_SET` arm; emitter already handled general dims;
      `DxirForwardTransform` already passes BROADCAST tangents through with
      attrs. Certified: IR-level rank-2→3 gradient + rank-1→3 JVP⇄VJP
      cross-identity, E2E `Σ a.broadcastTo(3,2,2) + Σ b⊙b` (da=3, db=2b).
    - **in-place size-1 stretch ✅ (§0.4.373)** (`[1,C]→[N,C]`, `[N,1]→[N,C]`),
      E2E through `grad {}` — the runtime-extent `SUM_TO` adjoint that closes the
      deferral. The stretch adjoint must sum the upstream over exactly the axes
      that were size-1 in the operand and KEEP them size-1, but which axes those
      are is unknowable under the -1 sentinel dims of `grad {}` (BroadcastRule
      can't tell a stretched size-1 axis from a matched one). Landed: a new
      `OpKind.SUM_TO(value, template)` — NumPy unbroadcast reading the extent from
      the `template` operand's ACTUAL runtime shape (template = shape source only,
      values never read), the reverse mirror of the BROADCAST stretch. Full
      RUNTIME-EXTENT-PATTERN wiring — interpreter arm (leading + size-1 aligned
      axis reduce), emitter arm (`reduce(add)` + `reshape` re-inserting size-1
      axes, axes derived from concrete emit-time dims), `DxirForwardTransform`
      tangent (linear in `value`; template's VALUE clone passed, not its tangent),
      `sumToLike(value, template)` host twin (mirror of `stretchLike`), and
      synthesis `irSumTo` + `deriveResultIrType`/backward-solver arms (SUM_TO
      IrType = template operand's). `BroadcastRule`'s empty-`reduceDims` branch
      (equal-rank identity axis map) now emits `SUM_TO(upstream, input)` instead
      of passing upstream through — true identity is the no-op case (SUM_TO reduces
      nothing); its `readsPrimalOperandIndices` gains `0`. The primal path was
      opened too: host `broadcastTo` + FIR now accept an operand size-1 axis as a
      stretch (full `broadcast_in_dim` eval, not `v[i % n]` tiling); `irBroadcast`
      synthesises the in-place-stretch primal broadcast (concrete `broadcastTo(N,C)`
      target → `stretchToRankN` with baked const dims, sentinel operand dims
      validated at runtime). Certified: IR-level SUM_TO interpreter pins (row/col
      keepdim + leading + identity), in-place stretch gradient (dx keeps the
      stretched axis size-1) both row and col, JVP⇄VJP cross-identity, and E2E
      `Σ a.broadcastTo(3,2)⊙b.broadcastTo(3,2)` (da = Σ-over-stretched-axis, keepdim;
      db = broadcast(a)). **Scoped-out — the MIXED case** (a simultaneous
      rank-increase AND an aligned size-1 stretch, e.g. `[1,C]→[B,N,C]`): its
      adjoint takes the non-empty-`reduceDims` SUM path which can't ALSO sum a
      stretched aligned axis; the FIR guards it (fail-loud when the operand dim is
      concretely detectable as size-1). **2nd-order through in-place broadcast —
      CLOSED (§0.4.399)**: the missing piece was exactly the predicted
      `BROADCAST_LIKE(upstream, template=value)` mirror op, landed there together
      with VjpRules for the whole runtime-extent family (SUM_TO ⇄ BROADCAST_LIKE,
      PAD_TO ⇄ SLICE_AT). One correction to this note's original wording: the
      missing SUM_TO VjpRule never gated `forward(reverse(f))` — SUM_TO has
      carried a forward tangent since this very slice, which is why §0.4.394's
      hessian is forward-OVER-reverse; what errored loudly ("no VJP rule
      registered for SUM_TO", verified at §0.4.398 HEAD) was REVERSE-mode over
      any body containing a runtime-extent adjoint, i.e. reverse-over-reverse.
    - **`slice` ✅ (§0.4.374)** — single-axis `slice(start, end, axis)` E2E
      through `grad {}` via the runtime-extent `PAD_TO` adjoint (the reduce/slice
      mirror of §0.4.373's `SUM_TO`). SliceRule's adjoint zero-pads the upstream
      back into the sliced window, but the trailing pad `high[i] = operand.dim[i]
      − start[i] − upstream.dim[i]` reads the operand's extent — a -1 sentinel in
      `grad {}`. New `OpKind.PAD_TO(value, template)` + attr `low`: place `value`
      into a zero tensor of `template`'s shape at offset `low`, deriving `high`
      from `template`'s ACTUAL runtime shape (`template` = shape-only, values
      never read). Full runtime-extent wiring — CostModel/interpreter/emitter
      (`emitPadTo`)/forward-tangent (linear in `value`, template's primal-value
      clone) + host `padToLike(value, template, low)` (+ `padToLikeRank1/2/3`
      fixed-arity shims, `low` baked as Int consts — mirror of `stretchToRankN`)
      + synthesis `irPadTo`/`deriveResultIrType`/backward-solver arms (PAD_TO
      IrType = template's). SliceRule now emits `PAD_TO(upstream, template=x,
      low=start_indices)` (numerically identical to the old
      `PAD(low, high=x.dim−limit)` on the concrete-dims IR path) with
      `readsPrimalOperandIndices = {0}` to keep `x` cloned as the template. The
      user surface: host `DTensor.slice(start, end, axis)` (unit stride, single
      axis) + a FIR arm lowering it to `OpKind.SLICE` with explicit `slice_start/
      slice_end/slice_axis` attrs (all user literals; the non-sliced axes'
      `limit_indices` ride as sentinels but are never read — `irSlice` recovers
      `(start, end, axis)` from the explicit attrs). Certified: IR-level PAD_TO
      interpreter pins, the slice gradient (`da` = zero-padded upstream keeping
      a's full shape, `db` = slice recomputed), the JVP⇄VJP cross-identity, and
      E2E `grad { Σ a.slice(1,3,0) ⊙ b.slice(0,2,0) }`.
    - **`concat` — IR level ✅ (§0.4.381), user surface pending.** The audit's
      framing was half right: CONCAT's interpreter arm, variadic
      `stablehlo.concatenate` emission (MLIR-round-trip-certified to 3 operands),
      forward-mode tangent, cost arm and `ConcatRule` all already shipped in
      §0.4.360, and the reverse transform already accumulates N contributions.
      What was broken is that **`ConcatRule` is sentinel-unsafe**: it baked
      `start_indices`/`limit_indices` from `x.type.dims[dim]`, which under
      `grad {}` are -1, so it produced offsets of -1, -2, … — not a wrong answer
      so much as a meaningless one. Fixed by the runtime-extent pattern the
      plan called for:
      - **New `OpKind.SLICE_LIKE(value, thisTemplate, priorTemplate₀…)`**, attr
        `axis` → `thisTemplate`'s shape: the window along `axis` starting at
        `Σⱼ priorTemplateⱼ.dims[axis]` and running for `thisTemplate.dims[axis]`,
        other axes whole. Both bounds are read off the templates' ACTUAL runtime
        shapes; templates contribute SHAPE ONLY. The `SUM_TO`/`PAD_TO` contract,
        originally including "no VjpRule" — §0.4.399 gave SUM_TO and PAD_TO
        their rules, and §0.4.404 closed SLICE_LIKE too via `PAD_LIKE`, its
        variadic transpose (the offset is a runtime SUM of prior templates'
        extents, which no literal-offset adjoint could carry — the reason it
        outlived the §0.4.399 pairs).
      - Arms: interpreter (outer/inner block copy, mirroring the CONCAT arm),
        emitter (emit-time dims are concrete, so the bounds fold to literals and
        it emits the same static `stablehlo.slice` as SLICE — the templates go
        unreferenced in the MLIR, legal and DCE'd), forward transform (tangent of
        the value, primal VALUE clones of every template), `CostModel` (the one
        exhaustive `when` over OpKind in main sources — a new kind is a compile
        error there until it has an arm).
      - `ConcatRule` now branches: concrete dims keep the §0.4.360 static SLICEs
        **byte-identically** (`DxirShapePlumbingTest` untouched), symbolic dims
        emit `SLICE_LIKE(upstream, xᵢ, x₀…xᵢ₋₁, axis)`. Since it is variadic, the
        static `readsPrimalOperandIndices` cannot express "all operands", so it
        stays `emptySet()` and the per-node `readsPrimalOperands(op)` added in
        A5c-2 is authoritative — all indices when symbolic (every operand is a
        template), none when concrete (nothing is cloned, as before).
      - Host twins: `sliceLikeStart` / `sliceLikeAfter{1..7}` — fixed-arity per
        PRIOR count, because synthesis builds positional `IrCall` arguments and
        cannot build an `IrVararg` (the documented reason for the whole `…RankN`
        shim family). **The 4-operand bound was lifted to 8 in §0.4.425**, and
        the audit there settled where the ceiling actually lived: NOT on the
        user surface — the FIR's fold-to-binary is arity-generic, so a user
        `concat`/`stack` of ANY width only ever produces binary CONCAT nodes
        whose SLICE_LIKE adjoints carry at most ONE prior template (certified
        by the five-operand mixed-extent E2E in `ConcatGradientTest`, widths
        2,3,2,3,2 with distinct prime coefficients, per-operand analytic sums)
        — but solely in the `irSliceLike`/`irPadLike` twin-selection guard,
        which only an IR-level hand-built VARIADIC CONCAT's gradient body can
        reach. Both families were extended in lockstep (`padLikeAfter{4..7}`
        too: SLICE_LIKE's VJP is PAD_LIKE with the SAME priors, so the pair's
        bounds must move together to stay closed under differentiation), with
        host pins at the new range's endpoints (4 and 7 priors, cumulative
        offsets 10 and 28 over eight width-1..8 segments, the sliceLike ⇄
        padLike round trip at 7, and the overrun refusal). An arity-generic
        respelling was REJECTED: it would need list-building IR the synthesis
        cannot construct — the same IrVararg wall, one layer up.
      Certified: interpreter pins for the window contract (start window, one
      prior, two priors, a LEADING axis so the copy is not one contiguous run,
      and the out-of-range refusal), the rule's concrete-vs-symbolic split
      (static SLICEs with baked cumulative offsets vs SLICE_LIKE with every
      template cloned into the body), the emitter's folded static slice, and the
      JVP⇄VJP cross-identity through CONCAT — which CONCAT never had.
    - **`concat`/`stack` user surface ✅ (§0.4.382).** `:core` gains
      `concat(axis, vararg tensors)` and `stack(axis, vararg tensors)` (both
      erasing to `DTensor<Shape, F32>` — the concat axis's extent is a runtime
      SUM, which no static witness carries, the `slice`/`reshape`/`broadcastTo`
      convention) over a fixed-arity `concatPair(axis, a, b)`.
      - **The FIR folds an n-ary concat into a right-fold of BINARY CONCATs.**
        Synthesis cannot build an `IrVararg` (the documented reason the whole
        `…RankN` shim family exists), and the primal concat node DOES reach the
        gradient body for the ordinary `concat(…).sum()` loss tail — so an n-ary
        node would be unsynthesizable. Concat is associative along the axis, so
        the fold is semantics-preserving and unbounded in n, and every node
        matches the 2-operand host op. Cost: one extra pass per intermediate
        (≈n/2× the data movement of a single n-ary concat); the true variadic
        path is the optimization, not a correctness requirement.
      - `stack` lowers as sugar: a unit-axis RESHAPE per operand (synthesizable
        since §0.4.375) then the concat fold — rank n → n+1.
      - The vararg FIR arm flattens `FirVarargArgumentsExpression` (the precedent
        is the `SHAPE_OP_SET` / `REDUCE_OP_MAP` arms, which already do it for
        `vararg Int` axes); `concatResultType` sums the axis extent,
        sentinel-propagating, and makes a concrete non-axis disagreement a
        call-site `LoweringException`.
      - Synthesis: `irConcat` → `concatPair`; `irSliceLike` → the fixed-arity twin
        selected by PRIOR-template count (`sliceLikeStart` / `sliceLikeAfter{1,2,3}`,
        axis as an Int const). `deriveResultIrType` gains CONCAT (operand[0]'s atoms
        with a placeholder `Lit<Int>` at the concat axis — §0.4.375's reasoning, and
        safe for the same reason: nothing reads the placeholder for a runtime-dim
        decision) and SLICE_LIKE (the `thisTemplate`'s IrType, the SUM_TO/PAD_TO
        shape-only treatment); the backward solver propagates a SLICE_LIKE result to
        its `thisTemplate` only — never to the value operand (the whole concat) or
        the PRIOR templates (different windows). `deriveInsertedAxesDTensor` was
        refactored onto a shared `rebuildShapeAtoms` rather than duplicated.
      - The runtime tape is deliberately NOT wired: `Backward.kt`'s dispatch list is
        only half of it — `TracedOps.kt` has no concat producer, so a tape could
        never record one and the dispatch entry would be dead code. (An earlier
        note here claimed one line would do it; that was wrong.) Consequence: for a
        concat the K2 synthesis path is the ONLY path, so a synthesis rejection is a
        hard failure rather than a slow fallback — hence the E2E no-fallback pins.
      Certified: 3 E2E through the real K2 plugin with no tape fallback —
      `Σ concat(1, a⊙2, b⊙3)` over a[2,2]/b[2,3] (da = 2s shape [2,2], db = 3s
      shape [2,3]: each operand gets its OWN window back); a THREE-operand
      `Σ concat(1, a, b, a)` where `a` feeds windows 0 and 2, so its two
      SLICE_LIKE contributions accumulate and the third operand's prior template is
      the first concat's result (an intermediate, not a param); and
      `Σ stack(0, a⊙2, b⊙3)` landing on a rank-3 result from rank-2 operands.
      Plus a `:core` host pin for the trailing-axis, leading-axis (non-contiguous
      copy), 3-operand and `stack` cases and both refusals.
    - **`pad` as a user op** has no DiffKT analogue (skip). `PadRule` is also
      sentinel-unsafe (`limit_indices` from `x.type.dims`) but unreachable without
      a user `pad`.
    - **`view`/indexing, `withChange`, `meld`/`split`** ✅ §0.4.428 — pure
      sugar over the landed runtime-extent family, no new IR anywhere:
      `view(range, axis)` IS the §0.4.374 SLICE; `view(index, axis)` slices
      the unit window and drops the axis with the squeeze RESHAPE;
      `withChange(index-or-range, axis, r)` lowers as
      `x + PAD_TO(r − slice(x), template = x)` — the §0.4.399
      PAD_TO ⇄ SLICE_AT closure used in a PRIMAL for the first time, so
      `d_x` = upstream with the window zeroed and `d_r` = the upstream's
      window fall out of existing rules (REJECTED: a concat(head, r, tail)
      spelling — the tail's start is `dims[axis] − …`, a dim-derived attr
      the sentinel discipline forbids); `meld` = flatten-per-operand + the
      binary-CONCAT fold. The one synthesis addition: `irReshape` grew a
      flatten arm — ANY rank-1 relayout target is row-major `flatten()`,
      whose extent the host reads off the operand's runtime shape, so a
      product-of-sentinels element count needs no per-axis param match
      (this also un-gaps user `flatten()` of a symbolic rank ≥ 2 tensor in
      surviving primals). E2E certs: view range/index, withChange row +
      row-range (window-zeroed `d_x`, exact `d_r`), meld routing at mixed
      ranks — all analytic, no-fallback pinned. Host certs incl. `split` ⇄
      `meld` roundtrip. Named deferrals: `split` in `grad {}` (no
      `List<DTensor>` value model in the lambda lowering — spell per-piece
      `view`/`slice` instead); the multi-index `view(IntArray)` leading-
      index form (chainable as repeated `view(i, 0)` when wanted);
      `withChange` beyond rank 3 (the `padToLikeRank{1,2,3}` arity bound).
      (`stats` left this list — landed host-level in §0.4.397, A5c-3(iv).)
- **A3. NN ops in lambdas** — split by wiring readiness:
  - **A3a ✅ (§0.4.368)**: `softmax(axis)` + `logSoftmax(axis)` E2E through
    `grad {}`. SOFTMAX was fully wired below the surface (interpreter,
    emitter, VJP, JVP, cost) — only the host fn + FIR arm + `irSoftmax`
    synthesis were missing. `logSoftmax` lowers to `LOG(SOFTMAX(x))` (both
    fully-ruled ops; **LOGSUMEXP stays emitter-only** — no VJP/interp/JVP),
    which also forced tensor `irLog`/`irExp` (were scalar-only).
  - **A3b ✅ (§0.4.370 IR-level; §0.4.400 embedding E2E; §0.4.409
    paddingIndex + rank-2 batches)** — the two self-contained halves landed,
    then embedding got its front-end, then its recorded tails:
    - **`embedding` VjpRule** ✅: EMBEDDING was wired below the surface
      (emitter-as-gather + cost model) but had no reverse rule and no
      interpreter arm. Added: an EMBEDDING interpreter arm (rank-2 table +
      int index tensor → gathered rows), a fused [OpKind.EMBEDDING_GRAD]
      adjoint op (interpreter arm: scatter-ADD each upstream row back to the
      vocab slot its index selected, collisions summing — result type `[V,D]`
      carries the dims, indices non-differentiable), the EmbeddingRule VjpRule
      (mirrors GatherRule's fused scatter-add), and the EMBEDDING forward-mode
      tangent (linear in the table). Certified IR-level: a hand-pinned
      collision-summing dTable + the JVP⇄VJP cross-identity.
      **§0.4.400 — `embedding` reaches `grad {}` E2E**, closing both §0.4.370
      deferrals. Front-end: `:core` host `embedding(table, indices)` (`[V,D]` ⊗
      rank-1 I32 `[N]` → `[N,D]`, bit-exact vs the interpreter arm per
      `DxirHostEmbeddingParityTest`) + FIR arm + `irEmbedding`/`irEmbeddingGrad`
      synthesis calling the host twin `embeddingGrad(upstream, indices,
      tableTemplate)`. Two design points: (a) EMBEDDING_GRAD grew a THIRD
      operand — the primal table as a shape-only template (the SUM_TO/PAD_TO
      convention), because under -1 sentinels the template's runtime dims are
      the only sound source of the vocab extent for the host call; (b) the I32
      index vector flows through the lambda as the first integer tensor PARAM
      the synthesis accepts (`isAcceptedIndexTensorType`, I32 rank-1) — its
      gradient slot is §0.4.54's structural integer zero, materialised as
      `intZerosLike(indices)` (requires a unique index-typed param; several
      would be ambiguous and reject). The scatter+add-region emission also
      landed (`stablehlo.scatter` over a splat-zero base, real add region — no
      `unique_indices`, no return-upd peephole, collisions must accumulate) with
      integer dense-literal const support, EmitterTest pins, the
      GradientEmissionCoverageTest exclusion removed (a collision-bearing
      embedding case swept instead), and `PjrtEmbeddingSmokeTest` XLA-verified
      on the GB10 (gradient bit-identical at 0.0). E2E cert: linear loss
      (dT rows = index counts, unselected vocab row EXACT zeros) and a
      nonlinear loss whose grad body RECOMPUTES the embedding, no tape
      fallback. Deferred: `paddingIndex`, rank-2 index batches (host surface is
      rank-1), and indices produced by in-lambda integer arithmetic (params
      only).
      **§0.4.409 — the recorded embedding tails close: `paddingIndex` + rank-2
      index batches.** `paddingIndex` is an optional Int-literal `padding_index`
      attr on EMBEDDING (negative/absent = none, canonicalised at FIR so
      `embedding(t, i)` and `embedding(t, i, -1)` CSE alike); positions whose
      index equals it produce EXACT-zero output rows and scatter nothing back
      (EmbeddingRule forwards the attr onto EMBEDDING_GRAD; the forward tangent
      already replayed `node.attrs`, so padded tangents are zero for free —
      pinned by the padded JVP⇄VJP cross-identity). Emission: the primal masks
      the gathered rows with compare-EQ + broadcast + select (XLA CLAMPS
      out-of-bounds gathers, so the select — not the gather — is what zeroes
      the row), the adjoint masks the UPSTREAM rows before the scatter (adding
      a zero row is a numeric no-op in-bounds; XLA drops out-of-bounds scatter
      updates — exact zero either way over the splat-zero base). Rank-2
      `[B, N]` batches: host `embedding` overloads (`@JvmName` dodges erasure;
      arity — never default params — disambiguates the padded spellings, the
      K2 named-arg landmine), FIR arm widened to rank 1..2 indices with dims
      still COPIED, `isAcceptedIndexTensorType` → I32 rank 1..2, and the
      EMBEDDING_GRAD emitter generalised from its v1 rank-1 contract to rank-r
      indices (`update_window_dims = [r]`, `index_vector_dim = r` — the
      §0.4.400 rejection test flipped to a positive pin). Synthesis resolves
      the grown overload sets by arity + the indices param's `Rank{r}`
      classifier (`embeddingHostSymbol`/`embeddingGradHostSymbol` — the
      `singleOrNull` CallableId lookup no longer suffices). Certified: host ↔
      interpreter bit-exact for padded + batched twins, padded-row EXACT-zero
      + collision-still-sums analytic pins at every level (host, interpreter,
      E2E `grad {}` linear + embedding-recomputing nonlinear), a cross-batch
      collision E2E, padded + batched JVP⇄VJP cross-identities, emitter text
      pins + two new coverage-sweep cases + round-trip cases, and a padded GPU
      smoke (grad max|diff| 0.0, padded row exactly zero on the GB10).
      Still deferred (re-recorded): indices produced by in-lambda integer
      arithmetic (params only). The several-index-params half CLOSED at
      §0.4.419 — `ZEROS_LIKE` names its param by construction, so any
      number of integer params per lambda lowers.
    - **`crossEntropyLoss`/`nllLoss`** ✅ E2E through `grad {}`: composed in
      FIR onto existing fully-ruled ops (no new VjpRule). `crossEntropyLoss` =
      `NEG(SUM(MUL(oneHot, LOG(SOFTMAX(logits, -1)))))` (sum-reduction
      convention — total of the per-sample cross-entropies), `nllLoss` skips
      the softmax. `:core` host twins added (`crossEntropyLoss`/`nllLoss`,
      both returning a scalar so the body ends in `.toFloat()`). Certified
      E2E: both gradients synthesise with no fallback and match the analytic
      references (CE: da = softmax·Σb − b, db = −logSoftmax; NLL: da = −b,
      db = −a).
  - **A3b ✅ (landed §0.4.384–386/389; the emission question closed §0.4.392) —
    `conv2d`/`maxPool`/`avgPool` in `grad {}`**. The original blocker
    analysis is kept below as history; every item resolves inline. Blockers:
    1. **Rank-6 gradient intermediates.** MaxPool2dRule's adjoint upsamples via
       `reshape → identity-stretch BROADCAST → reshape` through **rank-6** shapes
       (`[N,C,Ho,1,Wo,1] → [N,C,Ho,kh,Wo,kw]`). The synthesis scope gate
       `isAcceptedTensorType` is rank 1..3; even widening to rank-4 (for the NCHW
       conv/pool tensors themselves) does not admit the rank-6 nodes the maxpool
       adjoint body contains. AvgPool2dRule + Conv2dRule stay rank≤4 but still
       need the widen.
    2. **Single-representative `context.tensorIrType`.** Synthesis maps every
       accepted-tensor IrType to ONE call-site-harvested generic IrType (works
       for rank-1..3 because `broadcastLike<S>` is `S`-generic and the shape
       rides on runtime `.dims`). A conv/pool gradient body mixes rank-4 (x, dY,
       W) and rank-6 (maxpool upsample) nodes — the single-IrType model needs
       generalising before those bodies can be typed.
    3. **Missing synthesis op arms + host twins.** No `irConv2d`/
       `irConvTranspose2d`/`irMaxPool2d`/`irAvgPool2d` synthesis dispatch, no
       `:core` host `conv2d`/`maxPool`/`avgPool` eval (the runtime twin AND what
       synthesis calls back into), no FIR arms parsing the window/stride/padding
       literal attrs. `irTranspose` handles only rank-2/3 perms — Conv2dRule's
       adjoint uses 4-D transposes. The FORWARD/BACKWARD IrType solvers need
       CONV2D/CONV_TRANSPOSE2D/MAXPOOL2D/AVGPOOL2D arms.
    Recommended future sequencing: land avgpool first (rank≤4, linear adjoint,
    no rank-6, no where/mask), then conv2d (needs the 4-perm transpose +
    conv-transpose synthesis), then maxpool last (blocked on the rank-6 model +
    the `context.tensorIrType` generalisation). `batchNorm` grad{} folds in here
    too (BATCHNORM OpKind exists; VJP + surface unaudited).
    **§0.4.383 correction — that sequencing is wrong, verified against the code.**
    `AvgPool2dRule` is NOT the cheap wedge: its adjoint is
    `RESHAPE → CONV_TRANSPOSE2D(kernel = 1/(kh·kw) splat, lhs_dilation = stride,
    padding derived from window/stride/padding) → RESHAPE`, i.e. avgpool needs
    conv-transpose synthesis and a host twin exactly as conv2d does. There is no
    cheap first op among the three; the enabling slice is the SHARED rank-4
    substrate. Also verified while re-scoping: `:core` has no host
    `conv2d`/`convTranspose2d`/`avgPool2d`/`maxPool2d` at all; synthesis has no arm
    for any of the four OpKinds; `isAcceptedTensorType` is `F32 && rank in 1..3`;
    and `rebuildShapeAtoms`/`deriveInsertedAxesDTensor` cap at `Rank3` — though
    `Rank4`/`Rank5`/`Rank6` shape witnesses DO already exist in
    `:core/Shape.kt:73-81`, so the type level needs no new vocabulary, only new
    arms. Corrected order:
    1. **rank-4 substrate ✅ (§0.4.384)** — landed. `isAcceptedTensorType` widened
       to rank 1..4 **blanket**: the kind-scoped widen the plan hoped for is not
       available (the gate takes only a `DxirType` and is consulted from 50+
       sites), and blanket is safe because every rank-dispatching arm resolves a
       `…RankN` host delegate by name and returns null when there is no rank-4
       entry — a missing delegate still rejects and falls back to the tape, so the
       widen cannot turn a rejection into wrong code (validated by the full suite).
       `Rank4` added to both rank-class switches (`deriveDroppedAxesDTensor`,
       `rebuildShapeAtoms`); `:core` gained the host `conv2d`/`convTranspose2d`
       pair, the fixed-arity `conv2dGeneral`/`convTranspose2dGeneral` synthesis
       delegates, `transposePerm4` and `Tensors.f32Tensor4`; synthesis gained
       `deriveResultIrTypeRank4` (perm-general TRANSPOSE; the conv pair taking the
       batch atom from the lhs, the channel atom from the kernel's OIHW/IOHW axis,
       and `Lit<Int>` placeholders on both spatial axes; rank-agnostic elementwise
       propagation) and the `irConv` emission arm. Note `deriveResultIrType` was
       rank-2-ONLY at the top (`if (op.type.rank != 2) return null`), so rank 4
       branches to its own helper and rank-1/3 behaviour is untouched.
       Certified: the host twins are **bit-exact** against the interpreter's
       `evalConv2d` (same Double accumulator, same `i → ky → kx` order) for every
       attr spelling the adjoints emit, checked MECHANICALLY —
       `DxirHostConvParityTest` runs the real `DxirReverseTransform` on three
       losses (general-attr conv, plain conv, avgpool) and replays each
       CONV2D / CONV_TRANSPOSE2D / rank-4 TRANSPOSE node through both engines, so
       coverage follows the rules rather than pinning hand-written attrs.
    2. **conv2d user surface ✅ (§0.4.384 forward, §0.4.385 reverse)**: the
       FIR arm landed (`io.tlaloc.core.ops.conv2d` / `.convTranspose2d`: literal
       attrs folded onto the op, symbolic extents → -1 result dims, the
       transposed spelling's user `stride` mapped to `lhs_dilation` with
       `window_strides` at [1,1]). Conv now differentiates E2E in FORWARD mode:
       `ConvForwardIntrinsicTest` runs `jvp`/`valueAndJvp` over rank-4 self-convs
       (same-padded 3×3, stride-2 asymmetric-padded 4×4, and the 1-arg valid
       conv) with no synthesis fallback, against an independent Double-precision
       reference written from the definition — the first rank-4 tensor surface
       the synthesis has ever accepted.
       **Reverse mode was blocked on a finding that is NOT one of the four
       blockers above — now FIXED (§0.4.385).** `Conv2dRule` SOLVED its adjoint
       `padding` from the primal's extents, and every `grad {}` param carries -1
       sentinels, so the solve was arithmetic garbage: a stride-1 padding-1 conv
       emitted `padding=[[-3,1],[-3,1]]` where the correct value is `[[1,1],[1,1]]`
       (observed directly, by running the reverse transform over a sentinel-dim
       conv loss). Nothing downstream rejected it — the interpreter, the emitter
       and the host twins all faithfully honour the attrs they are handed — so the
       gradient would have been silently WRONG. §0.4.384's interim guard turned
       that into a loud compile error; §0.4.385 removes the cause. `AvgPool2dRule`
       has the same defect and STILL carries the guard: its channel-folding
       `RESHAPE` bakes `n * c`, which is **1** under sentinels.
       **The fix — two fused, runtime-extent adjoint ops**, following the house
       pattern (PAD_TO / SUM_TO / SLICE_LIKE carry a shape TEMPLATE operand instead
       of baked extents; `EMBEDDING_GRAD` is the fused-adjoint precedent):
       `CONV2D_DATA_ADJOINT(upstream, kernel, xTemplate)` and
       `CONV2D_KERNEL_ADJOINT(x, upstream, wTemplate)`. Each carries only the
       primal's own literal attrs (`window_strides`, `padding`, `rhs_dilation`) and
       solves the adjoint padding at EXECUTION time:
       `dX: low = kEff−1−p_low, high = p_low + H − dilSize` and
       `dW: low = p_low, high = (k−1)·d + dilSize − H − p_low` — algebraically
       identical to the values the rule used to bake, verified over 2157
       (extent, kernel, stride, dilation, padding) combinations before any code was
       written. Only the primal's LOW padding is needed: its high side is already
       implied by the upstream's runtime shape. The kernel adjoint also FUSES the
       batch↔feature transpose trick, so the gradient body keeps no rank-4
       TRANSPOSE nodes and each adjoint's result IrType is simply its template's —
       which is what dissolves the single-representative `tensorIrType` blocker for
       conv bodies (they never mix ranks). Every layer has an arm: interpreter
       (`evalConvAdjoint`, over a lifted `conv2dCore` + `evalTranspose`), host twins
       (`conv2dDataAdjoint` / `conv2dKernelAdjoint`, each asserting the solved conv
       lands on its template), synthesis (`irConvAdjoint` + forward and backward
       IrType arms), the cost model, and the emitter. The emitter arm was NOT
       optional: §0.4.362's GPU cert runs the rule's gradient graph through real
       XLA. It solves the padding from the concrete static dims at emit time, so
       the MLIR is the same `stablehlo.convolution` the pre-fusion rule produced —
       and that cert now covers the fused path (grads agree with the interpreter to
       1.19e-7 on the GB10).
       Still deferred at the time: the fused ops had neither a VjpRule nor a
       forward tangent. The forward half CLOSED at §0.4.423 (the whole
       fused-adjoint family carries tangent arms, so hessians compose
       fwd-over-rev); rev-over-rev through such bodies stays a pinned loud
       refusal (the B4 pins), on the consolidated tails list. (CONV_TRANSPOSE2D's
       own adjoint, deferred here, landed in §0.4.391 — see item 5.)
       **API constraint discovered the hard way — applies to the pooling surfaces
       too.** The host conv ops take their attrs POSITIONALLY, in two arities,
       with NO default parameter values. K2 unwraps a named argument (`padTop = 1`)
       to its bare literal before the FIR lowering sees the call and does NOT
       reorder it into its parameter's position, so a name-based reading cannot
       distinguish `padTop = 1` from `strideH = 1`; the first E2E run folded the
       padding into the stride and silently produced a valid conv. Arity is
       unambiguous, defaults are not.
    3. **avgPool ✅ (§0.4.386)** — landed, and it turned out cheaper than expected
       because the fused-adjoint pattern generalises: one new
       `AVGPOOL2D_GRAD(upstream, xTemplate)` op replaces the whole
       `RESHAPE → CONV_TRANSPOSE2D(1/(kh·kw) splat, lhs_dilation = stride, solved
       padding) → RESHAPE` chain. That chain had TWO extent dependencies, and the
       reshape was the worse one: its target baked `n * c`, which under sentinels is
       **1**, so it claimed a shape the data did not have. The fused op needs no
       channel fold at all — the adjoint is per-channel, and the
       depthwise-via-batch-folding trick only ever existed to dodge grouped-conv
       support. It INVERTS the window per input element (`y = (iy + padTop − ky) /
       strideH` when that divides evenly and lands in range) instead of solving a
       padding, so there is no solve to get wrong; attrs are the primal's literals,
       count_include_pad is preserved (divide by the FULL `kh·kw`, matching the
       primal), and only the LOW padding participates. Arms everywhere: interpreter
       (`evalAvgPoolGrad`), host twins (`avgPool2d` 2 arities + `avgPool2dGeneral`
       primal delegate, both bit-exact against `evalPool2d`; `avgPool2dGrad`),
       synthesis (`irAvgPool`, `irAvgPoolGrad`, forward + backward IrType arms —
       AVGPOOL2D keeps the input's batch/channel atoms and placeholders the spatial
       ones, AVGPOOL2D_GRAD's result IS its template's), cost model, FIR arm, and
       the emitter — which EXPANDS the fused op back into fold + lhs-dilated splat
       convolution + unfold, i.e. exactly the MLIR §0.4.363 certified, rather than
       introducing a `feature_group_count = C` depthwise conv.
       `AvgPool2dRule.readsPrimalOperandIndices` widened `emptySet() → setOf(0)`:
       the rule used to inspect only `x.type`, but a template OPERAND is a value
       reference as far as the clone walk is concerned, and omitting it would leave
       the emitted node pointing at a primal id that collides with the grad builder's
       fresh ids (ref-integrity-valid, semantically wrong).
       Certified: `AvgPoolGradientTest` E2E through the real plugin over four
       spellings — non-overlapping 2×2, overlapping 3×3/stride-2/pad-1 over a
       5×5 input (a floor-division remainder the inversion must crop), a chained
       `relu(x).avgPool2d(…)` whose template is a forward-derived node, and
       `Σ p²` over `p = avgPool2d(x)` whose loss READS the pooled value so the
       primal AVGPOOL2D lands in the body's value stream (without it the primal
       twin `avgPool2dGeneral` and its `irAvgPool` arm would ship unexercised —
       the first three only ever reach AVGPOOL2D_GRAD) — all against a SCATTER
       reference (spread each output's upstream over the taps that were in range,
       divide by the full window) that transposes the primal loop rather than
       inverting the window, so an off-by-one in either shows up. Plus
       `PoolingAdjointSentinelSafetyTest` (§0.4.389 — renamed from
       `AvgPoolAdjointSentinelSafetyTest`, now covering both pooling kinds, 4 tests)
       pinning that the body carries only literal attrs and that neither the RESHAPE
       nor the CONV_TRANSPOSE2D reappears, an `EmitterTest` text pin on the solved
       asymmetric padding `[[1,1],[1,2]]` and the fold/conv/unfold sequence, the
       host↔interpreter bit-exact walk extended to BOTH the primal AVGPOOL2D and
       the fused adjoint, and
       `PjrtPoolingSmokeTest` on the GB10 (avgpool grad max|diff| 1.19e-7 vs the
       interpreter). `DxirPoolingTest`'s FD + JVP⇄VJP oracles are unchanged and
       green, so the fusion preserved values on concrete dims.
    4. **maxPool ✅ (§0.4.389)** — landed, and the §0.4.386 re-scope held: the
       rank-6 intermediates and the `tensorIrType` single-representative blocker
       were both artefacts of the *spelling*, not of maxpool. `MAXPOOL2D_GRAD`
       inverts the window per input element — for each input, find the covering
       output windows, compare `x[i]` against that window's max, accumulate the
       upstream of every window it wins — so the body is rank-4 throughout and the
       result type is just `x`'s. Two deviations from the sketch worth recording:
       it takes THREE operands `(upstream, x, y)` rather than two, because passing
       the recomputed pooled `y` means no engine duplicates the max pass and the
       emitter gets a ready SSA value instead of emitting a second `reduce_window`;
       and `x` is a VALUE operand (its elements are compared), not the shape-only
       template conv/avgpool use.
       **The v1 restriction only PARTLY lifted.** Window-divisibility moved out of
       the rule to EMIT time — it was the extent-reading half (`-1 % 2 == -1`, which
       is what made the rule reject every symbolic maxpool), and the host and
       interpreter handle a remainder correctly (inputs the truncated last window
       never covered get no gradient). But `strides == window` and zero padding
       REMAIN, as literal-attr checks in the rule: the emitter expands to the
       nearest-upsample-and-mask MLIR §0.4.363 certified, and row-major
       reshape/broadcast/reshape replication only tiles exactly for a
       non-overlapping unpadded window. Overlapping or padded maxpool gradients
       therefore work on the host and fail loudly at StableHLO emission. Lifting
       that needs `stablehlo.select_and_scatter`, deliberately NOT used here: it
       picks ONE winner per window while every other engine routes the full
       upstream to every within-window tie, and exact ties are common after a relu
       — a backend-dependent gradient is worse than a loud restriction.
       Certified: `MaxPoolGradientTest` E2E through the real plugin over three
       spellings — non-overlapping 2×2; `relu(x)` pooled, which is deliberately
       tie-heavy (coarse-quantised data, so whole windows of exact zeros tie) and
       would expose a single-winner policy; and 5×5 under a 2×2 window, where the
       uncovered last row and column must get exactly zero — against a SCATTER
       reference that takes each window's max and pays 1 to every tap equal to it.
       `DxirHostConvParityTest` replays MAXPOOL2D and MAXPOOL2D_GRAD host↔interpreter
       bit-exactly, and adds `maxPoolAdjointRoutesFullUpstreamToEveryTie`, which pins
       the tie convention on BOTH engines with hand-built data (`x = [[1,1],[2,2]]`,
       upstream 7 → `[0,0,7,7]`, not `[0,0,7,0]`) because random floats never tie and
       so the walk above cannot cover it. `EmitterTest` pins the expansion's op counts
       and the rank-6 type strings. `PjrtPoolingSmokeTest` on the GB10 now runs the
       fused path: maxpool grad max|diff| **0.0** vs the interpreter (bit-identical —
       the mask copies values, it does no arithmetic). `DxirPoolingTest`'s
       `maxPoolGradientRoutesToArgmax` and FD/JVP oracles are unchanged and green.
       The emitter arm earned its keep immediately: its first version typed the
       compare against the POOLED `y` instead of the upsampled one, which is invalid
       MLIR — the GPU cert caught it, no text pin would have.
       ✅ **§0.4.392 — the emitter restriction investigated and CLOSED as inherent,
       not as a TODO.** Question: should overlapping/padded maxpool gradients get a
       GPU path via `stablehlo.select_and_scatter`? Answer: no, and the reason is
       worth recording so this is not re-litigated. The all-ties convention (every
       within-window winner receives the FULL upstream — matching MaxRule, the
       interpreter, and the host twin) is not expressible over overlapping windows in
       StableHLO. `select_and_scatter` is the primitive built for exactly this shape
       of computation and it picks ONE winner per window. The other candidate, a
       transposed-conv spread of `dy` (which is how `AVGPOOL2D_GRAD` emits, and which
       does sum over covering windows), cannot apply the per-window `x == max(w)`
       mask — with overlapping windows an input position belongs to several windows
       with *different* maxima, so there is no single upsampled `y` to compare
       against; that ambiguity is precisely why nearest-upsample-and-mask only works
       when the windows tile. So the real choice is "general strides on the GPU with
       backend-dependent tie behaviour" versus "non-overlapping on the GPU with
       identical semantics on all three engines". A gradient that differs between
       host and device is worse than a loud refusal — exact ties are common after a
       relu, where whole windows of zeros tie — so the restriction stays. What
       changed is that it is now PINNED rather than merely documented:
       `EmitterTest.maxPoolGradRefusesOverlappingWindowsAtEmitTime` and
       `…RefusesNonDivisibleDimsAtEmitTime` assert the emit-time failure names the op
       and the restriction, and (for divisibility) that the message says the other
       engines cope. Reopening this would mean changing the tie convention in
       MaxRule, the interpreter and the host twin together — a semantics decision,
       not an emitter task.
    5. **CONV_TRANSPOSE2D's own adjoint ✅ (§0.4.391)** — differentiating THROUGH a
       transposed conv (`grad { x, w -> x.convTranspose2d(w, 2, 2, …) }`, i.e. a
       deconvolution / fractionally-strided upsample) was a loud "no VJP rule
       registered" even though the primal's FIR arm and host twin shipped in
       §0.4.384 and forward mode already worked. Two more fused ops,
       `CONV_TRANSPOSE2D_DATA_ADJOINT(upstream, kernel, xTemplate)` and
       `CONV_TRANSPOSE2D_KERNEL_ADJOINT(x, upstream, wTemplate)`, structurally
       §0.4.385's mirror — but SIMPLER in the one place that mattered: **no padding
       solve at all**. The primal's tap maps input↔output through
       `yDil = yo·s + ky·d − p_low` with `yDil` a multiple of the lhs dilation `L`,
       so inverting that single equation per tap (`yo = (iy·L + p_low − ky·d) / s`,
       kept only when it divides evenly and lands in range) absorbs the padding, both
       dilations, the strides and the kernel reversal in one test. There is no
       runtime solve to get wrong, which is a strictly better position than the conv
       adjoints were in.
       The formulas were verified against central differences in a standalone model
       over six configurations BEFORE any Kotlin was written (lhs_dilation 1 and 2,
       window_strides 1 and 2, rhs_dilation, reversal, asymmetric padding, and all
       combined; worst error 1.1e-9), and `DxirConvTransposeVjpTest` re-pins the same
       six in-tree against both directional central differences and the JVP⇄VJP
       cross-identity (whose forward side uses only the bilinear product rule, so it
       never touches the new index inversion).
       **Scope at the time: interpreter + host + synthesis, no StableHLO arm** — a
       GPU-targeted build failed loudly at emit, the `EMBEDDING_GRAD` precedent, and
       no shipped cert regressed because nothing could emit that graph before. The
       emitting identities were recorded rather than left as a mystery, and
       **§0.4.393 implemented them**: `dX` = strided-slice (undilate by `L`) of the
       conv data adjoint against the channel-swapped kernel, `dW` = the channel swap
       of the conv kernel adjoint over an interior-dilated `x` (which needed the
       emitter's PAD to grow a real `interior` field — it had been hardcoded to
       zeros). Only `window_reversal` is still rejected at emit, for the reason given
       in §0.4.393.
       Also certified: `ConvTransposeGradientTest` E2E through the real plugin for
       both the stride-1 and the upsampling (`lhs_dilation` 2) spellings, all four
       gradients against central differences, no tape fallback; and the
       host↔interpreter bit-exact walk extended to both new kinds with an
       asymmetric `window_reversal` [true, false] so a swapped or dropped flag cannot
       cancel out.
    6. **GPU emission completed for the arc ✅ (§0.4.393).** Auditing "which
       OpKinds can a `grad {}` body contain that the emitter cannot lower" turned up
       exactly three gaps, and the worst one was not on any list:
       - `SIGN` — the only user-reachable kind with no emitter arm. `ops.sign` is in
         the FIR unary map AND the MAX/MIN reduction rule builds its extremum
         indicator as `1 − sign(y − x)`, so `grad { x.max(1).sum() }` lowered and
         synthesised but could not be emitted. One line (`stablehlo.sign`); the only
         divergence from the interpreter is the sign bit of zero for a −0.0 input,
         which no consumer observes.
       - `BROADCAST` with EMPTY `broadcast_dimensions` at equal rank — a DIVERGENCE,
         not a missing arm, and the more serious find. §0.4.359 made that form
         polymorphic in the interpreter (a scalar splat OR the equal-rank "un-reduce
         stretch" `[N,1] → [N,K]`) and its comment claimed the emitter matched; the
         emitter still required `dims.size == inputRank` and rejected the stretch. So
         EVERY reduction adjoint that un-reduces — Max, Min, Tanh, Softmax — was
         unemittable, and nothing caught it because no GPU test ran a reduction
         gradient through XLA. Fixed by mirroring the interpreter's rule exactly
         (identity dims in that case). **The lesson is the reason this is written
         down: the OpKind-level audit found `SIGN` but could not find this one — only
         running a real gradient graph through the emitter did. An audit of kinds is
         not an audit of configurations.**
       - the two `CONV_TRANSPOSE2D_*_ADJOINT` kinds from §0.4.391, via the identity
         recorded there. `emitPad` grew a real `interior` field (hardcoded zeros
         until now) for the dilation, and the conv-adjoint expansions were lifted
         into `emitDataAdjointConvolution` / `emitKernelAdjointExpansion` so both
         conv directions share one padding solve instead of two copies.
       `window_reversal` is still rejected at emit for the deconv adjoints: with a
       reversed primal the data side needs `!r` and the kernel side a compensating
       flip, and that identity is unverified. Nothing user-reachable sets the attr,
       and the interpreter and host twins handle any reversal, so it fails loudly.
       Certified: `PjrtConvTransposeSmokeTest` (2) on the GB10 against the
       interpreter — deconv loss + both gradients at 6.0e-8, and the max-reduction
       gradient (the SIGN + un-reduce-stretch path) bit-identical at 0.0 — plus four
       `EmitterTest` text pins: the swap/conv/strided-slice sequence, the interior
       pad, `stablehlo.sign`, and the identity-dims stretch. One pin is deliberately
       counter-intuitive and says so: the inner convolution's `lhs_dilate` is `[1,1]`
       and NOT the primal's `[2,2]`, because the identity dilates `x` explicitly and
       undoes it with the slice.
       And because the BROADCAST gap is exactly the kind a per-op test misses,
       `GradientEmissionCoverageTest` now sweeps the whole differentiable surface:
       20 primal losses (elementwise unaries, max/min/mean reductions, softmax,
       matmul, concat/slice/pad/reshape/transpose, where+compare, and all four
       conv/pool kinds) are reverse-transformed, run through the interpreter to
       prove the catalogue graph is well-formed, and then required to EMIT. Adding a
       rule or an op without an emitter arm now fails this test instead of failing
       only on the first user who targets a GPU. Its exclusions are documented
       inline (EMBEDDING unreachable, overlapping maxpool inherent, deconv
       `window_reversal` unverified, SILU/GELU sub-user-surface).
       Remaining GPU gaps after this: overlapping/padded maxpool gradients (closed as
       inherent, item 4) and `EMBEDDING_GRAD` (unreachable — `embedding` has no FIR
       arm, so no `grad {}` body can contain one). *(§0.4.400 closed the
       EMBEDDING_GRAD gap: scatter+add-region emission landed with the `grad {}`
       front-end, the sweep exclusion is gone, and the emission is GPU-smoked.)*
    ✅ **§0.4.387 — composition certified.** `CnnBlockGradientTest` runs one
    `grad {}` body carrying conv → relu → avgPool → sum PLUS a skip term, so the
    gradient threads an AVGPOOL2D_GRAD into both conv adjoints with a RELU/STEP
    mask between them, and `dx` accumulates TWO rank-4 contributions rather than
    being one call's result. Central differences over all 68 parameters (±1e-3,
    the 5e-2 tolerance `Rank2NNFiniteDifferenceTest` uses) agree, with no tape
    fallback. The per-op E2E tests each certify one layer against a hand-written
    reference; this is the first that puts them in sequence, which is where
    composition bugs live.
    ✅ **§0.4.390 — `batchNorm` in `grad {}`, by DESUGARING rather than by a new
    op.** The `BATCHNORM` OpKind that already existed is the INFERENCE form (five
    operands: input, scale, offset, mean, variance) the Layer-3 recognizer emits for
    fused kernels; training mode computes its statistics from the argument, so an
    inference-form rule could never produce its gradient. Instead the FIR arm
    desugars `x.batchNorm(scale, offset, eps)` into
    `mean → sub → mul → mean → add(eps) → sqrt → div → mul(γ) → add(β)` (the
    `maximum`/`clip` sugar pattern), which needs NO new VjpRule, interpreter arm,
    host delegate or synthesis arm — every node already has a sentinel-safe adjoint.
    Biased variance (`mean((x−μ)²)`, PyTorch's training-mode convention); NCHW with
    the feature axis at 1; `scale`/`offset` are rank-1 `[C]` reshaped to `[1,C,1,1]`
    so the broadcast aligns on the FEATURE axis (NumPy right-alignment of a bare
    `[C]` against `[N,C,H,W]` would match C up with W).
    **This also settles blocker 2 empirically, in the negative:** batchNorm's body
    MIXES RANKS (rank-4 `x` against rank-1 `scale`/`offset`) and it synthesises
    fine — the single-representative `tensorIrType` is only a FALLBACK, so per-node
    derivation and the backward solver cover a mixed-rank body. What actually blocked
    it was three narrow synthesis gaps, all pre-existing and all now widened:
    `irReshape` capped at rank 3 (so `[C] → [1,C,1,1]` rejected), the
    unsqueeze/squeeze shims capped at 2 axes (so that reshape AND its adjoint
    rejected), `irReduce` rejected >2 axes (so `mean(0,2,3)` rejected), and `irSqrt`
    was scalar-only — its own doc said tensor sqrt "would need
    `io.tlaloc.core.ops.sqrt` and tensor-IrType threading", and that extension
    already existed. New host shims: `reshapeToRank4`, `unsqueezeAxes3`,
    `squeezeAxes3`, `{sum,mean,max,min}Over3`.
    Certified: `BatchNormGradientTest` E2E through the real plugin, no tape fallback,
    with TWO oracles because one is not enough — the primal value against an
    independent Double implementation (which is what pins the wiring, since
    `valueAndGrad2`'s value and gradients come from the same lowered graph and a
    consistent mis-wiring such as swapped γ/β would agree with its own central
    differences perfectly), then both gradients (rank-4 `dx` and rank-1 `ds`) against
    FD. `HostOpsTest` pins the host twin's biased-variance convention and distinct
    γ/β by hand.
    **Trap worth recording:** the first version of that test used `Σ y` as the loss,
    and `dx` came back all zeros — correctly. The sum of a batch-normalised tensor is
    exactly `N·H·W·β` per channel, because the normalised values sum to zero by
    construction, so `Σ y` is CONSTANT in `x` and central differences agree with a
    zero gradient vacuously. The non-zero assertion is what caught it; the loss is now
    `Σ y²`.
- **A4. Elementwise binary max/min + clip + outerProduct** — split by the
  synthesis-transpose boundary:
  - **A4a ✅ (§0.4.369)**: `maximum(a, b)` / `minimum(a, b)` / `clip(x, lo, hi)`
    E2E through `grad {}`. All sugar over the §0.4.364 where/compare surface —
    `maximum` = `WHERE(COMPARE(a, b, GE), a, b)`, `minimum` uses LE, `clip` =
    `minimum(maximum(x, lo), hi)` composed as two COMPARE+WHERE pairs against
    `lo`/`hi` splat consts of `x`'s shape. No new VjpRule; the gradient flows
    through WhereRule (full upstream to the larger/smaller/in-bounds operand,
    ties + boundaries route to `a` on the `>=`/`<=` equality). `clip`'s bounds
    are compile-time Float literals (new `floatLiteralArg` FIR helper).
  - **A4b ✅ (§0.4.375) — `outerProduct` grad{} E2E**: host op + FIR lowering
    (`MATMUL(reshape(a, [n,1]), reshape(b, [1,m]))`) + IR-level gradient landed
    §0.4.369; §0.4.375 closes the `grad {}` synthesis fallback. Root cause: the
    reshape-created unit axis had no param-sourced shape atom, so MatmulRule's
    adjoint `TRANSPOSE([n,1]) → [1,n]` could not derive an IrType
    (`irOpFor returned null for TRANSPOSE`). Fix: a `deriveInsertedAxesDTensor`
    helper that synthesises a placeholder `Lit<Int>` atom for reshape-created
    unit axes, wired into TWO derivation sites — a FORWARD `deriveResultIrType`
    RESHAPE arm (rank-increasing unit-axis insertion: types the `[n]→[n,1]` /
    `[m]→[1,m]` reshapes so the downstream TRANSPOSE resolves) and a BACKWARD
    RESHAPE-solver arm (the squeeze `[n,1]→[n]` / `[1,m]→[m]` on the way out:
    re-inserts the unit axis so the MATMUL output IrType is known, letting the
    existing MATMUL operand-solver fill the `[n,m]` scalar-seed's IrType). The
    placeholder atom is never read for a runtime-dim decision — unsqueeze/squeeze
    emit by axis position, and the seed-broadcast axis-matcher reads only the
    param-sourced `n`/`m` atoms. CERTIFIED E2E (`OuterProductGradientTest`): the
    grad{} params carry DISTINCT atoms (`n = Sym`, `m = Lit<Int>`) so the seed's
    `[n,m]` shape resolves unambiguously; `∇ Σ outerProduct(a, b)` synthesises
    with no fallback and matches `da_i = Σ_j b[j]`, `db_j = Σ_i a[i]`.
- **A5. Binary-op broadcasting + orphaned lowerings** *(audit)* — split by
  what the operand shapes require:
  - **A5a ✅ (§0.4.376) — scalar × tensor mixing**, E2E through `grad {}`:
    all four elementwise binaries with a `Float` on either side (`a * 2.0f`,
    `a + 1.0f`, `3.0f - a`, `b / 2.0f`) plus a COMPUTED scalar side
    (`a * b.sum().toFloat()`). Mechanism: the FIR splats the rank-0 operand to
    the tensor operand's type, so the binary op is well-typed and every rule
    that already ships applies — **no new op kind, no new VjpRule**. A literal
    splats to a shaped const (the §0.4.369 `clip` bound pattern, so no dead
    rank-0 const is left in the body); a computed rank-0 value splats through
    BROADCAST with empty `broadcast_dimensions` (the §0.4.359 scalar-seed
    polymorphism), whose adjoint is BroadcastRule's runtime-extent `SUM_TO`
    full reduce (§0.4.373) — so a DIFFERENTIABLE scalar side is correct for
    free (test 4: `db = Σa` flows back through the splat). Sentinel-safe: the
    splat target dims are the tensor operand's (-1s included) and synthesis
    materialises them by axis-matching params. Landed along the way: the eight
    `:core` scalar-mixing operator overloads (`DTensor + - * / Float` and the
    scalar-on-left `Float + - * / DTensor`, which the non-commutative pair
    needs); `findTensorBinaryOp`'s overload filter had to start naming the
    EXTENSION RECEIVER (a `Float.op(DTensor)` overload also has "exactly one
    regular DTensor param", and picking it synthesized a call whose receiver
    slot held a DTensor); and **tensor `NEG` synthesis was broken** —
    `findUnaryOp` is keyed on `dxirType.dtype` alone, so a DTensor-typed NEG
    resolved `kotlin.Float.unaryMinus` and the generated gradient threw
    `ClassCastException` at run time. Reachable from any tensor SUB (SubRule's
    `NEG(upstream)`), any tensor DIV (DivRule's `NEG(mul)`) and CosRule's
    `-sin(x)`; no E2E surface had exercised it. Pre-existing, surfaced by A5a.
    Not covered: `DScalar × DTensor` mixing (DiffKT's `timesScalar` on
    DScalar) and comparisons against a scalar (`a gt 1.0f` — the
    COMPARE_DIRECTION_MAP arm still lowers both sides verbatim).
  - **A5b ✅ (§0.4.377) — the orphaned lowerings.** The audit's list was partly
    stale: tensor `mean`'s map entry landed with A1 (§0.4.366). The two real
    orphans are now reachable from user code:
    - **scalar `tanh` / `sigmoid`**: both were fully ruled at the IR level
      (TanhRule `1 − tanh²`, SigmoidRule `σ(1−σ)`, interpreter + emitter arms,
      forward tangents) and the TENSOR spellings have mapped since §0.4.200, but
      `:core/DScalar.kt` declared no scalar host fns, so `UNARY_OP_MAP` had
      nothing to map. Landed: the five-overload host set for each (Float, Double,
      FloatScalar, DoubleScalar, DScalar — the exp/log §0.4.158 pattern), the two
      map entries, and — because there is no `kotlin.math.sigmoid` — a new
      `irCoreScalarCall`/`coreScalarSymbolFor` synthesis pair that resolves the
      `io.tlaloc.core` extension whose receiver matches the op's primitive dtype
      (the sibling of `irUnaryMathCall`, which does the same inside
      `kotlin.math`). §0.4.200's explicit scalar-SIGMOID rejection is lifted;
      scalar TANH needed no synthesis change (it already routed to
      `kotlin.math.tanh`).
    - **`pow`**: POW has been ruled below the surface since Stage B.3 (PowRule
      incl. the I32/I64-exponent CAST, interpreter, `stablehlo.power`, forward
      tangent, and synthesis's scalar `kotlin.math.pow` arm from §0.4.52) but
      nothing lowered TO it. Landed: `:core/ops` host `pow` in DiffKT's three
      spellings (tensor exponent, Float exponent, Int exponent; Double arithmetic
      then F32, matching the interpreter arm bit-for-bit); `BINARY_OP_MAP` entries
      for both `io.tlaloc.core.ops.pow` and `kotlin.math.pow`; POW added to
      `ELEMENTWISE_BINARY_KINDS` so a literal exponent rides the A5a splat and the
      IR always sees the uniform two-operand POW PowRule expects; a tensor POW
      synthesis arm via the generic tensor-binary dispatch
      (`findTensorBinaryOp("pow")`); and POW added to both IrType solvers'
      elementwise-binary arms. Certified E2E: `a.pow(2.0f)` → 2a, `a.pow(3)` →
      3a², `a.pow(b)` → (b·a^(b−1), a^b·ln a) — the exponent param's own partial,
      which needs the tensor LOG and tensor POW arms together — and scalar
      `x.pow(2.0f)` → 2x.
    Still open in A5b's neighbourhood: `DScalar × DTensor` mixing and
    comparisons against a scalar literal (`a gt 1.0f`).
  - **A5c-1 ✅ (§0.4.378) — IR-level implicit broadcasting**
    (`broadcast(S1,S2)`; DiffKT broadcasts every binary op). The three layers
    that disagreed now agree on NumPy semantics — operands right-align against
    the result, a size-1 axis stretches, a rank-deficient operand gains
    replicated leading axes:
    - **Interpreter**: ADD/SUB/MUL/DIV/POW evaluate through one
      `binaryBroadcast` helper (per-result-axis strides, 0 on replicated axes).
      EQUAL-shape operands keep the flat zip they always had, so pre-A5c
      programs are bit-identical; only genuinely mixed shapes pay for the walk.
      Incompatible aligned extents still `require`-fail.
    - **Emitter**: `broadcastIfNeeded` generalises from same-rank-only to a
      right-aligned axis map (`dims = [offset … outRank−1]`). §0.4.277's v1
      refusal pin (`binaryRejectsRankMismatchedOperand`) is replaced by a pin of
      the emitted right-alignment; the surviving refusal is the other direction
      (an operand of HIGHER rank than the result — broadcasting never drops
      axes).
    - **VjpRules**: a shared `unbroadcast(contribution, operand)` wraps each
      AddRule/SubRule/MulRule/DivRule contribution in `SUM_TO(…, operand)` —
      NumPy's reduce-over-replicated-axes, reading the target extents from the
      operand's ACTUAL runtime shape (§0.4.373), which is what makes it sound
      under `grad {}`'s -1 sentinels where "which axes were size-1" is
      statically unknowable. Skipped when the shapes are PROVABLY identical
      (concrete-and-equal dims, or a splat const whose type is the shape it was
      splatted to), so the concrete-dims IR the coarsener, the emitter tests and
      the pinned gradient tests walk stays byte-identical. DivRule's `a/(b·b)`
      term is now typed as the RESULT (it broadcasts `a` against `b`). PowRule
      is untouched: after the A5a splat its operands always agree, and its
      tensor-exponent spelling shares one shape param.
    - **Forward transform**: needed NO change — MUL's product rule and DIV's
      quotient rule already type every intermediate as the node's own result
      type, so broadcasting operands make them correct as-is. Pinned by the
      JVP⇄VJP cross-identity over a `[2,1] ⊙ [1,3]` chain.
    - **`sumToLike`** gains an identity fast path (copy, not the stride walk):
      the rules now emit SUM_TO wherever shapes aren't provably equal, so at
      RUNTIME the shapes usually do match and there is nothing to reduce.
    - **Synthesis** needed one repair, found by the suite: the backward IrType
      solver propagated a SUM_TO's output only to its TEMPLATE, so a MUL feeding
      one of the new SUM_TOs lost its backward-solved IrType, its seed-BROADCAST
      operand fell back to `context.tensorIrType` (the rank-2 param
      representative) and splatted to the wrong rank — `AxisReductionGradientTest`'s
      g2 then called `times([2,2], [2])`. SUM_TO now also propagates to its
      VALUE operand **when the reduce is rank-preserving** (exactly the
      un-broadcast case, where value and result share a shape); a
      rank-REDUCING SUM_TO must not, because its value really is bigger.
    Certified IR-level (`DxirBroadcastBinaryGradTest`): interpreter pins for
    `[3,1]⊙[1,4]`, `[3]⊙[2,3]`, rank-0⊙`[2,2]`, the equal-shape fast path
    (DIV+POW) and the incompatible-extent refusal; gradients for the two-axis
    stretch (`da[3,1]`, `db[1,4]`), rank extension (`dv[3]`, `dm[2,3]`) and a
    mixed scalar+ADD+DIV chain; plus the cross-identity. Emitter: the
    right-aligned `dims = [1]`, the empty-axis-map scalar splat, and the
    higher-rank refusal.
  - **A5c-2 ✅ (§0.4.379) — the user surface.** `a + b` broadcasts, in source and
    in `grad {}`:
    - **`:core/ops/BroadcastOps.kt`** (a NEW FILE, deliberately — see below): the
      `elementwiseBroadcast` walk (right-aligned strides, flat-zip fast path for
      equal dims) behind `plusBroadcast` / `minusBroadcast` / `timesBroadcast` /
      `divBroadcast` (star-projected operands + an explicit result-shape witness,
      the `sumToLike` / `broadcastLike` convention), plus the DiffKT-parity
      `<S1, S2>` operator overloads returning `DTensor<Shape, F32>`. The
      shape-PRESERVING operators in HostOps.kt now delegate to the same walk and
      keep only their precise witness, because **a shared static shape type does
      not imply shared runtime dims** — `[N,1]` and `[N,C]` are both
      `Rank2<Sym, Lit<Int>>`. The two overload sets must live in different files:
      generics erase, so two `plus(DTensor, DTensor)` extensions in one facade
      class are a platform declaration clash and `@JvmName` is unavailable in
      commonMain. Kotlin's most-specific-wins resolution keeps picking the
      shape-preserving one when the operands agree (pinned by a test that assigns
      `a + b` to a `DTensor<Rank2<Sym, Sym>, F32>`).
    - **FIR**: `broadcastResultType` — result rank `max(rank_a, rank_b)` (exact,
      since ranks come from the call-site type even under sentinels), each extent
      exact only when both aligned extents are concrete, sentinel otherwise;
      incompatible concrete extents are a call-site `LoweringException` rather
      than a runtime shape error.
    - **Synthesis**: tensor ADD/SUB/MUL/DIV call the broadcasting host ops with the
      witness threaded from the derived IrType; `deriveResultIrType` propagates
      from a SAME-RANK operand only (falling back to a rank-deficient operand's
      IrType yields a wrong-RANK splat, not an error: `mul(broadcast(1.0):[-1,-1],
      v:[-1])` typed from `v` made the seed rank-1 and the gradient then called
      `sumToLike([3], [2,3])`); the backward solver likewise propagates a result
      IrType to same-rank operands only; `findTensorBinaryOp` now requires exactly
      one type parameter so it cannot pick the `<S1, S2>` overloads.
    - **The seed-shape problem** — the real discovery, and the reason this slice is
      not just "add host ops". Synthesis resolves a scalar splat's target by
      axis-matching its static IrType against the params, which is a GUESS that
      broadcasting makes unsafe: two params can share atoms and differ at runtime,
      and a rank-2 target can axis-match a rank-1 param's axis outright. Either way
      the seed came out with the wrong shape. Fix: the scalar-seed `BROADCAST` gains
      an optional second, SHAPE-ONLY template operand (the `SUM_TO` / `PAD_TO`
      convention) whenever its target carries a sentinel, and synthesis then calls
      `broadcastLike(v, template)` against the one value whose runtime shape IS the
      target. `SumRule` and `MeanRule` emit it; `VjpRule` gains
      `readsPrimalOperands(op)` so the template's clone is required per NODE rather
      than per rule — a concrete-dims SUM must NOT drag its summed operand into the
      gradient body (it would recompute it, and for a rank-changing node the
      gradient scope cannot synthesise it at all, which is exactly how
      `BroadcastToGradientTest` broke). The forward transform passes the template's
      primal value clone, never its tangent.
    Certified: 7 `:core` host tests (two-axis stretch, rank extension, rank-0 splat,
    the resolution pin, the same-static-type/different-runtime-dims case, and the
    incompatible-shape refusal), 3 E2E through `grad {}` (`[2,1] ⊙ [2,3]` with both
    params statically `Rank2<Sym, Lit<Int>>`; `[3] ⊙ [2,3]` rank extension with
    `dv` staying rank-1; a keepdims `[2,1] ⊙ [2,3]` intermediate), 2 IR pins for the
    templated seed, 2 shape-validation pins (rank-differing mismatch reported,
    legal rank extension silent).
  - **A5c-3(i) ✅ (§0.4.380) — shape templates everywhere synthesis would
    otherwise GUESS.** A5c-2 templated SumRule/MeanRule's scalar seed; the same
    hazard covered every other shape synthesis resolves by axis-matching static
    IrType atoms against the params. Two families, one mechanism:
    - **Un-reduce stretches**: `SumRule` (axis form), `MeanRule`, `MaxRule` /
      `MinRule` (both the recomputed-extremum stretch and the upstream stretch),
      `SoftmaxRule` (the row-sum stretch), `DotRule` (both splats) and
      `GatherRule` (the zero scatter base) now pass the node whose shape IS the
      target as a shape-only second operand, and `irBroadcastStretch` prefers it
      over axis-matching (`stretchLike(x, template)`, factored into
      `irStretchLikeCall`). This matters more for the stretch than for the splat:
      an un-reduce target is usually an INTERMEDIATE's shape — `(v * m).max(1)`'s
      adjoint stretches back to the broadcast product, which no param has, and
      axis-matching a rank-2 target produced `[3,3]` out of `v:[3]` and `m:[2,3]`.
    - **Rule constants**: a shaped const has no runtime shape source at all, so
      `MaxRule`/`MinRule`'s mask `1.0`, `WhereRule`'s `1.0`, `SqrtRule`'s `2.0`,
      `TanhRule`'s and `SigmoidRule`'s `1.0` and `PowRule`'s exponent `1` go
      through a new `splatConst` — a plain shaped const under concrete dims (so no
      pre-A5c IR changes shape) and a templated splat under symbolic ones. Same on
      the FIR side via `splatLiteral`: `clip`'s lo/hi bounds, `where`'s zero, and
      the Phase A5a literal splat (`a * 2.0f`).
    - All of it is gated on `needsShapeTemplate` (any dim ≤ 0), so concrete-dims
      IR — every IR-level test, the emitter's MLIR, the coarsener's inputs — is
      byte-identical. `GatherRule` needed the per-node `readsPrimalOperands`
      refinement because it previously read only `idx`.
    Certified: `((v * m).max(1)).sum()` E2E with `v:[3]`, `m:[2,3]` (dv = [0,0,90]
    shape [3], dm = [[0,0,3],[0,0,3]] shape [2,3] — the argmax mask landing on the
    right columns requires the stretch to hit `[2,3]`, not `[3,3]`), plus an IR pin
    that MAX's adjoint broadcasts carry templates under sentinels and none under
    concrete dims, each template's shape being exactly its broadcast target.
  - **A5c-3 remainder.**
    (i) `SignRule` and `CompareRule` still emit bare shaped ZERO consts as their
    (piecewise-constant) contributions, and neither reads its operand, so a
    template would need the per-node clone refinement — their zero gradient can
    still be guessed wrong under ambiguous params. The pooling rules' zero/kernel
    consts are the same story and belong with A3b.
    (ii) The StableHLO emitter reads a splat target from the node type, so a
    templated broadcast's template is emitted as a DEAD value — MLIR-legal and
    DCE'd by XLA, but wasteful if a gradient body ever reaches the XLA path.
    (iii) Under sentinels a templated seed costs one extra evaluation of the
    summed node; a `dimsOf`-style shape-only host op would remove it.
  - **A5c-3(iv) ✅ (§0.4.397) — the last scalar-mixing tails.** Four pieces:
    - **Comparisons against a Float scalar** (`a gt 1.0f`, and the COMPUTED
      spelling `a gt b.mean().toFloat()`), E2E through `grad {}`: the six
      `:core` comparison overloads on the Float side (same 0/1 F32 mask
      contract), and the §0.4.364 COMPARE arm splats a non-DTensor rhs exactly
      like the A5a mixed-rank binary arm — a literal via `splatLiteral`
      (templated under sentinels), a computed rank-0 side via `splatScalarTo`'s
      BROADCAST — so COMPARE always sees two same-typed operands. The computed
      side was indeed free. One synthesis repair fell out: `irCompare` resolved
      its host op with the uniquely-named `coreOpsSymbol` lookup, which the new
      overloads turned into `null` (two candidates) — it now uses
      `findTensorBinaryOp`'s filter (DTensor receiver + one regular DTensor
      param + one type param), the same overload-disambiguation A5a built for
      the arithmetic ops. Certified: mask-routed `where(a gt 1.0f, a⊙a, b)`
      analytic gradients, the computed-scalar variant (the mean path correctly
      contributes ZERO through the piecewise-constant COMPARE), and a direct
      multiplicative `(a le 0.5f) ⊙ b` mask, plus the six host pins.
    - **`stats()`**: `(mean, variance)` host sugar over the full reductions,
      variance BIASED (÷N — DiffKT's and `batchNormGeneral`'s §0.4.390
      convention). HOST-LEVEL ONLY by design: a Pair-returning body has no
      `grad {}` lowering (the loss contract is scalar) and DiffKT's `stats` is
      a convenience, not a differentiation surface; a loss that needs the
      pieces writes `x.mean()` / squared-deviation mean directly, which
      differentiate today.
    - **`DScalar × DTensor`** (DiffKT's `timesScalar` — its ONE scalar-mixing
      `Operations` primitive; the other binaries mix through `Float`, which A5a
      ships): `:core` host overloads `DTensor * DScalar` / `DScalar * DTensor`
      landed, and the FIR needed NOTHING — the A5a mixed-rank arm fires off the
      shared `io.tlaloc.core.ops.times` FQN and splats the rank-0 side. The
      differentiable-scalar-PARAM form is certified E2E with a `Float` param
      (`grad { a, s -> (a ⊙ s).Σ }` → `da = s`, `ds = Σa` through
      BroadcastRule's full-reduce adjoint).
    - **`FloatScalar`-typed params ✅ (§0.4.414)** — the §0.4.397 pinned tail
      closed: synthesis materialises a boxed-scalar param AS the call-site
      value class (so the synthesised `FunctionN` type matches and the
      type guard passes), unwraps it through ONE `.toFloat()` local at body
      start (every downstream read site then sees the primitive exactly as
      the Float-param twin), and boxes the returned rank-0 gradient back via
      the value-class constructor — standalone and inside
      Pair/Triple/Quadruple slots alike. The §0.4.397 pin in
      `DScalarMixingGradientTest` flipped to a value-checked E2E (no tape
      fallback; `da = s` splat, `ds = Σa` boxed, matching the Float twin
      exactly), plus a `valueAndGrad2` cert (plain-Float value + tensor grad
      untouched, `FloatScalar` grad in the third slot).
    - **Both §0.4.414-recorded deferrals closed ✅ (§0.4.427)** —
      **`DoubleScalar` E2E**: the boxed-scalar conversion members
      (`FloatScalar`/`DoubleScalar`/`DScalar` × `toFloat`/`toDouble`) joined
      `CAST_OP_MAP` (matching precision collapses to IDENTITY at lowering —
      the box was already erased, so an all-boxed-scalar body's dxir equals
      the primitive spelling's; cross-precision emits a real CAST), the A5a
      mixed splat casts a cross-precision FLOAT scalar to the tensor's dtype
      before broadcasting (integer scalars keep failing exactly as before),
      and `CastRule` grew the float→float adjoint its §0.4.40 note
      anticipated — the reverse cast of upstream, so `d_s` rides back to F64
      and boxes as `DoubleScalar` through the §0.4.414 precision-symmetric
      exit. Certified: direct mixing (`a * s`, `ds = DoubleScalar(Σa)`), the
      explicit `s.toFloat()` spelling (same values through the same CAST),
      and the all-boxed identity-collapse body (`da = s²`, `ds = 2sΣa`).
      **`DScalar`-the-interface params** refuse BY NAME instead of falling
      to the anonymous type-mismatch guard: the refusal is structural (the
      function type fixes the slot to the interface, so the concrete class
      at the returned function's call sites is unknowable at compile time —
      no constructor to box the gradient into), the tape fallback still
      runs, and the `kept original call` warning names both concrete
      spellings (`FloatScalar (F32) or DoubleScalar (F64)`). REJECTED — a
      checker-level hard error: the runtime tape handles the dynamic
      spelling correctly today, and erroring would break working code to
      punish a supported (if slower) path. All four certs value-checked in
      `DScalarMixingGradientTest`.

### Phase B — AD-mode parity

- **B1. Forward-mode user intrinsics** ✅ (§0.4.372): `jvp {}` /
  `valueAndJvp {}` (DiffKT `forwardDerivative` /
  `primalAndForwardDerivative`) — the §0.4.361 `DxirForwardTransform`
  wired into `GradIntrinsics` + the plugin rewrite, the missing user
  surface for forward-mode AD. `jvp(f)` curries like `grad(f)`, returning
  `(x, dx) → dy = J_f(x)·dx` in one forward pass (dual-number, not finite
  differences); `valueAndJvp(f)` returns `(x, dx) → (y, dy)`. Wiring: the
  transform already emits `jvp_f(x, dx) → (y, dy)`, so `valueAndJvp` is
  that 2-return function boxed as `Pair`, and `jvp` is the same with the
  primal returns dropped (full body kept — tangents depend on primal
  values). The plugin branches to `DxirForwardTransform` before the
  reverse coarsening pipeline (skipped — forward v1 is straight-line, so
  a region-bearing body just falls back to the tape); the check-time
  differentiability probe uses the forward transform for these. v1 scope:
  straight-line bodies. Certified E2E through the K2
  plugin (dy = 2⟨x,dx⟩ for Σx², the (y,dy) pair, Σexp(x)·dx).
  ✅ **§0.4.387 — the multi-arg follow-up landed**: `jvp2`/`valueAndJvp2`
  with params `(x, w, dx, dw)` — primals then tangents, which is the order
  `DxirForwardTransform` itself emits, so nothing permutes. The plumbing was
  already arity-agnostic (the IR extension's forward branch splits returns at
  `size / 2`), so this was surface + registration: the two `:autograd`
  declarations, `INTRINSIC_NAMES`, and the checker's `intrinsicNames` /
  `forwardIntrinsics`. What it UNBLOCKED is the interesting part — a
  single-argument `jvp` cannot express a conv against a separate kernel, so
  §0.4.384's forward cert had to differentiate a SELF-convolution `conv(x, x)`;
  `Jvp2IntrinsicTest` now runs the genuine bilinear product rule over a real
  `(x, w)` pair (plus `Σ(a⊙b)` as the plumbing check) against an independent
  Double reference.
- **B2. `jacobian` + `hessian` intrinsics ✅ (§0.4.394)** — dense derivative
  assembly, with ZERO IR-layer changes: the whole slice is user surface +
  plugin dispatch, because both seeded single-pass functions already existed.
  - **The mechanism — synthesise the seeded pass once, assemble at runtime.**
    A dense Jacobian's row/column COUNT is a runtime quantity under
    `grad {}`'s -1 sentinel dims, so the assembly loop cannot live in the
    synthesised IR. Instead the plugin synthesises the 2-param seeded lambda
    (`jacobian`: the §0.4.361 forward transform's `jvp(x, dx) → dy`,
    tangent-only; `hessian`: forward-OVER-reverse `hvp(x, v) → H·v` — the
    composition pinned at IR level since §0.4.361, and forward-over-reverse
    rather than reverse-over-reverse precisely because the runtime-extent
    adjoint ops carry forward tangents but no VjpRules) and hands it to a
    runtime helper in `:autograd` (`assembleJacobianForward` /
    `assembleHessianForward`) that loops over the input's standard basis
    where the actual extents are known, stacking `[m, n]` columns / `[n, n]`
    rows. DiffKT does the same identity-seeding loop inside
    `reverseDerivative`; the cost (n seeded passes) is the classic dense
    trade.
  - **Result typing**: `J[i, j] = ∂yᵢ/∂xⱼ` over the ROW-MAJOR FLATTENED
    input/output (any input rank — the basis is built on flat data), erased
    to `DTensor<Rank2<Sym, Sym>, F32>` (no static witness carries the
    runtime extents — the `concat` convention). A `Float`-returning `f`
    degenerates to the `[1, n]` gradient row.
  - **The one synthesis change — `callTypeOverride`.** `synthesise` types
    params/returns from the call site's `FunctionN<…>` type args, but an
    assembly call site's own type is the 1-param ASSEMBLED function while
    the lambda being synthesised takes `(x, seed)`. The extension builds the
    seeded lambda's true `Function2<A, A, R>` type (A from the intrinsic
    call's type, R from the `f` argument's type) and passes it through; the
    original call keeps supplying source offsets only.
  - **No tape fallback** (the `concat` precedent): a synthesis rejection
    keeps the original call, which throws `pluginMissing` loudly at first
    invocation. The FIR checker probes `jacobian` with the forward transform
    (its lambda returns a TENSOR, which the reverse probe would reject) and
    `hessian` with the composed forward∘reverse, so failures are red
    squiggles at the call site.
  - Certified E2E (`JacobianHessianIntrinsicTest`, real plugin, no stubs —
    the REAL generic `:autograd` declarations resolve off the classpath, so
    the shipped surface itself is what's certified): `jacobian` over
    `x ⊙ x` (diag), `x · Σx` (non-diagonal — the A5a computed-scalar splat
    under the forward transform, tangent through BOTH product-rule factors),
    `(x ⊙ x).sum()` (the `[1, n]` row), and a RANK-2 input (the row-major
    flatten pin, `[4, 4]` from `[2, 2]`); `hessian` over `Σx²` (2I),
    `(Σx)²` (rank-one, all 2s — the tangent threads the reverse body's
    un-reduce broadcast), and `Σ exp(x)` (value-DEPENDENT diag — the tangent
    threads the adjoint's exp recompute).
  - v1 scope: single-argument `f`, straight-line bodies, host F32. (The
    seeded-cotangent user surface closed in §0.4.398 below; multi-arg
    `jacobian2`/`hessian2` closed in §0.4.406 below; reverse-assembled
    (tall) Jacobians for m ≪ n closed in §0.4.412 below.)
- **B2 follow-up. `vjp` + `valueAndVjp` intrinsics ✅ (§0.4.398)** — the
  seeded-cotangent user surface (DiffKT's `vjp` / `primalAndPullback`, audit
  item 10): `vjp(f)` returns `(x, ȳ) → x̄`, the pullback of a USER-SUPPLIED
  cotangent `ȳ` (of `f`'s OUTPUT type) through `f` at `x` in ONE reverse
  pass — `grad {}` generalised to tensor-valued `f` (`grad(f)` ≡ `vjp(f)`
  at the unit seed of a scalar `f`; a dense Jacobian is `m` calls of it
  over the output basis, the loop `jacobian` runs for you).
  - **The IR layer already had the function — two gates hid it.**
    `DxirReverseTransform.apply(seedAsParam = true)` (§0.4.33's COARSENED
    `gradient_body` machinery) has produced `(upstream, *params) → (*grads)`
    for 60+ sections, but its scalar-return gate predates the insight that
    the reverse walk is SEED-AGNOSTIC: with a caller-supplied seed the
    upstream param takes the primal return's type VERBATIM (tensor allowed),
    and every VjpRule already handles tensor upstreams — that is how
    interior ops' adjoints flow under `grad {}`. The gate now applies only
    to the const-1.0 path (a unit seed is only meaningful for a scalar
    objective). The second gate — `includeForward` + `seedAsParam` declared
    "incompatible" — was a fact about the COARSENED caller, not the math;
    lifted, the combined mode `(upstream, x) → (y, x̄)` is exactly
    `valueAndVjp`.
  - **Simpler than `jacobian`, by construction**: no runtime assembly
    helper (the synthesised seeded pass IS the replacement — one seeded
    pass, no basis loop) and no `callTypeOverride` (the call site's own
    type IS the 2-param seeded function type, `Function2<A, R, A>`). The
    only seam is parameter ORDER — the transform emits the upstream first,
    the declared surface takes `(x, ȳ)` — and synthesis resolves body
    references by node id (params are positional metadata), so the plugin
    rotates the params list and synthesises directly. No tape fallback
    (the `concat`/`jacobian` precedent); the FIR checker probes with
    `seedAsParam = true` so tensor-returning bodies are checked with
    exactly what the IR extension runs.
  - Certified E2E (`VjpIntrinsicTest`, real plugin, real generic
    `:autograd` declarations, no stubs): `vjp` over `x ⊙ x` at a
    NON-UNIFORM `ȳ` (x̄ = 2·x⊙ȳ — a unit-seed impostor cannot pass), the
    grad-consistency identity (`vjp` at `ȳ = 1` == `grad` for `Σx²`), the
    JVP⇄VJP inner-product identity ⟨ȳ, jvp(x, v)⟩ == ⟨vjp(x, ȳ), v⟩
    computed numerically in the user program, and `valueAndVjp` returning
    the true primal alongside the same pullback. IR-level pins in
    `CoarsenFunctionTest` (tensor-return pullback evaluated by the
    interpreter; the combined value+pullback mode; the default path still
    refusing tensor returns).
  - v1 scope: single-argument `f`, straight-line single-return bodies
    (the seeded branch skips the coarsening pipeline, like B1's forward
    branch). Deferred tails: region-bearing bodies (fold into B3/B4's
    region work); multi-arg `vjp2` closed in §0.4.406 below.
- **B2 close-out. Multi-argument seeded/assembled intrinsics ✅ (§0.4.406)**
  — `vjp2` / `valueAndVjp2` / `jacobian2` / `hessian2`, closing the
  "multi-arg" tails §0.4.394 and §0.4.398 recorded, with ZERO IR-layer
  changes: both underlying transforms were arity-agnostic all along
  (`DxirReverseTransform(seedAsParam = true)` has emitted
  `(upstream, *params) → (*grads)` for any arity since §0.4.33 — it IS the
  COARSENED `gradient_body` signature — and `DxirForwardTransform` emits
  all primals then all tangents for any arity/return count), so the slice
  is surface + plugin-gate generalisation + two runtime assembly helpers.
  - **Surfaces** (house currying, primals-then-seeds): `vjp2(f)` returns
    `(x, w, ȳ) → Pair(x̄, w̄)` — cotangent LAST, one seeded reverse pass for
    both gradients; `valueAndVjp2` → `Triple(y, x̄, w̄)` (the combined
    `includeForward + seedAsParam` mode, which fell out freely as §0.4.398
    predicted). `jacobian2(f)` returns `(x, w) → Pair(J_x [m, nx],
    J_w [m, nw])` — the plugin synthesises `jvp2`'s tangent-only
    `(x, w, dx, dw) → dy` ONCE and `assembleJacobian2Forward` loops basis
    vectors on EACH input with a ZERO tangent on the other (separate blocks,
    not one glued matrix — the caller usually wants exactly one).
    `hessian2(f)` returns the FULL `[(nx+nw), (nx+nw)]` matrix over the
    CONCATENATED row-major-flat input (blocks `[[H_xx, H_xw], [H_wx,
    H_ww]]`): one matrix rather than a Quadruple of blocks because it
    erases to the same `DTensor<Rank2<Sym, Sym>, F32>` as `hessian` —
    `hessian2(f)(x, w) == hessian(g)(x ++ w)` by construction — and block
    extents are runtime quantities under sentinels anyway. Its seeded pass
    is forward-over-reverse of the TWO-return `grad2` body,
    `hvp2(x, w, dx, dw) → (H_xx·dx + H_xw·dw, H_wx·dx + H_ww·dw)`, boxed
    `Pair<A, B>` exactly as synthesis boxes any 2-return function;
    `assembleHessian2Forward` writes exact full COLUMNS (no symmetry
    assumption), `nx + nw` passes total.
  - **Plugin**: the vjp branch's param rotation (`drop(1) + first()`) was
    already arity-agnostic — only the gate changed (2 primals for the "2"
    spellings; the call site's own type is again the seeded function's
    type, so no `callTypeOverride`). The assembly branch generalises its
    gate, harvests `(A, B)` from the call type's first two args and `R`
    from the `f` argument, and builds `Function4<A, B, A, B, seedRet>` as
    the override (`seedRet` = `R` for `jacobian2`, `Pair<A, B>` for
    `hessian2`). Checker probes match the real lowering: `jacobian2`
    forward, `hessian2` forward∘reverse, `vjp2`/`valueAndVjp2` seeded
    reverse. No tape fallback (the `concat` precedent).
  - Certified E2E (`MultiArgSeededIntrinsicTest`, the §0.4.394 pattern —
    real plugin, REAL generic `:autograd` declarations, no stubs, "kept
    original call" a hard failure): `vjp2` over `Σ(a⊙b)` at scalar ȳ = 2
    AND the grad2-consistency identity at ȳ = 1; TENSOR-R `vjp2` over
    `a⊙b` at NON-UNIFORM ȳ (x̄ = b⊙ȳ, w̄ = a⊙ȳ); the JVP⇄VJP inner-product
    identity over BOTH slots (⟨ȳ, jvp2(a,b,va,vb)⟩ == ⟨ā,va⟩ + ⟨b̄,vb⟩,
    both sides 2.75 numerically); `valueAndVjp2`'s true primal;
    `jacobian2` over `a⊙b` (J_a = diag(b), J_b = diag(a)), a scalar body
    (the two `[1, n]` rows), and a RECTANGULAR `a·Σb` (na=2, nb=3 — the
    per-input column indexing pin); `hessian2` over `Σ(a⊙b)`
    ([[0, I], [I, 0]]), `Σ(a⊙a⊙b)` (value-DEPENDENT H_aa = diag(2b),
    H_ab = diag(2a)), and a rectangular `Σa·Σb` (5×5 block layout with
    unequal extents). The concatenated-input cross-check against 1-arg
    `hessian` stayed analytic (a user-code `slice`-based `g(z)` would ride
    forward-over-PAD_LIKE — untested composition, not worth coupling this
    cert to); the block-layout equality is pinned by construction of the
    convention plus the rectangular analytic case.
  - v1 scope matches the 1-arg forms: straight-line bodies, host F32,
    2 arguments (3+-arg forms of the FORWARD-assembled family — `jacobian3`,
    `hessian3`, `jvp3`, `vjp3` — would need `Function6`+ overrides and
    `assemble*3Forward` helpers — same pattern, more params; the REVERSE
    spellings `grad3`/`valueAndGrad3` and `jacobianReverse2` landed in
    §0.4.424 below). B2's last neighbour — reverse-assembled (tall)
    Jacobians for m ≪ n — closed in §0.4.412 below.
- **B2 tall tail. `jacobianReverse` intrinsic ✅ (§0.4.412)** — the
  reverse-assembled (tall) Jacobian, the m ≪ n tail recorded at §0.4.394:
  same `[m, n]` row-major-flat contract as `jacobian`, assembled from the
  OTHER seeded pass — the §0.4.398 seeded reverse pullback
  `vjp_f(x, ȳ) → x̄` is exactly one Jacobian ROW per output-basis
  cotangent, so `assembleJacobianReverse` loops `ȳ = eᵢ` over the OUTPUT
  basis (DiffKT's `reverseDerivative` regime).
  - **The shape problem and its honest solve**: the output extent `m` and
    dims are runtime quantities unknowable before `y` exists — a basis
    cotangent needs `y`'s shape to be BUILT at all — so the helper takes
    the ORIGINAL user lambda too (the plugin passes it through verbatim;
    its eager host execution IS the primal) and runs it once. Cost:
    `m + 1` passes (one primal + m pullbacks) versus `jacobian`'s n
    forward passes. The pick between the two spellings stays the
    CALLER'S: both extents are runtime quantities under the -1 sentinel
    dims, so no compile-time heuristic could honestly compare them —
    explicit intrinsic, no auto-pick.
  - **Plugin**: mirrors the assembly branch over the vjp branch's
    transform — seeded reverse (`seedAsParam = true`, no includeForward),
    the §0.4.398 param rotation, synthesised under
    `callTypeOverride = Function2<A, R, A>` (the call site's own type is
    the 1-param ASSEMBLED function), then one IrCall to
    `assembleJacobianReverse(f, vjp)`. Checker probes the seeded reverse,
    exactly what the extension runs. No tape fallback (the `concat`
    precedent). Scalar-R needed NO special casing: a `Float`-returning
    `f` synthesises with a Float-typed upstream (certified since
    §0.4.406's scalar-ȳ `vjp2`) and degenerates to the `[1, n]` row at a
    unit cotangent.
  - Certified E2E (`JacobianReverseIntrinsicTest`, the §0.4.394 pattern —
    real plugin, REAL generic `:autograd` declarations, no stubs, "kept
    original call" a hard failure): `x ⊙ x` (diag(2x)) AGREEING ENTRYWISE
    with forward-assembled `jacobian` over the same body (the
    cross-assembly oracle — different seeded transforms, same matrix);
    the scalar-R `[1, n]` gradient row; and a genuinely TALL
    `concat(0, x, x)` (J = [I; I], `[2n, n]` — row indexing through
    ConcatRule's symbolic SLICE_LIKE adjoints).
  - v1 scope matches `jacobian`: single-argument `f`, straight-line
    bodies, host F32 (a `jacobianReverse2` would be the §0.4.406 pattern
    verbatim if ever pulled — pulled in §0.4.424 below).
- **B2 arity tail. `jacobianReverse2` + the 3-arg reverse spellings ✅
  (§0.4.424)** — the two "mechanical" arity tails, and they were exactly
  that. `grad3`/`valueAndGrad3` needed ZERO plugin-path changes beyond the
  name gates: the reverse transform emits `(*params) → (*grads)` for any
  arity (§0.4.33) and synthesis boxes 3 returns as `Triple` / 4 as
  `Quadruple` (§0.4.203, and §0.4.420's 4-param sparse `grad {}` is the
  standing proof at arity 4) — the whole slice is the two generic
  `:autograd` declarations (coexisting with the §0.4.134 Tracer-tape
  `grad3` overloads exactly as the `grad2` pair does — lambda parameter
  types disambiguate), the `INTRINSIC_NAMES`/checker entries, and the
  `includeForward` gate learning `valueAndGrad3`. `jacobianReverse2` is
  the §0.4.412 branch generalised the way §0.4.406 generalised the
  assembly branch: gate 2 params, harvest `A, B` from the call type's
  first two args, `callTypeOverride = Function3<A, B, R, Pair<A, B>>`
  (the 2-return pullback boxes as `Pair`, `vjp2`'s precedent), helper
  `assembleJacobianReverse2(f, vjp2)` — one eager primal for `m`, then
  each output-basis pullback pass writes row `i` of BOTH per-argument
  blocks (`J_x [m, nx]`, `J_w [m, nw]`, `jacobian2`'s Pair-of-blocks
  convention) — `m + 1` passes versus `jacobian2`'s `nx + nw`.
  - Certified E2E (`ThreeArgIntrinsicTest`, the §0.4.412 harness — real
    plugin, REAL generic `:autograd` declarations, no stubs, "kept
    original call" a hard failure): `grad3`/`valueAndGrad3` over
    `Σ(a⊙b⊙c)` on a quarter-integer grid (`∇a = b⊙c` etc., value 9.25,
    all exact); `jacobianReverse2` over `x ⊙ w` (`J_x = diag(w)`,
    `J_w = diag(x)`) AGREEING ENTRYWISE with forward-assembled
    `jacobian2` (the cross-assembly oracle); the scalar-R degenerate
    (`Σ(x⊙w)` → the two `[1, n]` rows off a Float unit cotangent); and
    the genuinely TALL, RECTANGULAR `concat(0, x, w)` (nx=2, nw=3,
    m=5: `J_x = [I₂; 0]`, `J_w = [0; I₃]` — per-argument row indexing
    through ConcatRule's symbolic SLICE_LIKE adjoints).
  - Still open (recorded above at §0.4.406): 3-arg forms of the
    forward-assembled family (`jacobian3`/`hessian3`/`jvp3`/`vjp3`) —
    same pattern, `Function6`+ overrides, pull when a consumer asks.
- **B4 enabler. The runtime-extent family closes under differentiation ✅
  (§0.4.399)** — VjpRules for SUM_TO and PAD_TO via their runtime-extent
  mirrors, closing §0.4.373's "2nd-order through in-place broadcast" deferral.
  The runtime-extent adjoint ops had no VjpRules, so REVERSE-mode over any
  body containing one — which is what reverse-over-reverse IS, since the
  first reverse pass emits them — failed loudly with "no VJP rule registered
  for SUM_TO" (verified at §0.4.398 HEAD; forward-over-reverse never needed
  the rules, which is why §0.4.394's hessian works — the original deferral
  note misattributed that).
  - **Two new ops, each an existing op's mirror**: `BROADCAST_LIKE(value,
    template)` (NumPy right-aligned broadcast up to the template's ACTUAL
    runtime shape — SUM_TO's forward twin) and `SLICE_AT(value, template)` +
    attr `low` (the window of the template's runtime shape at a LITERAL
    per-axis offset — PAD_TO's reverse mirror). Both follow the full
    house runtime-extent pattern: template operand contributes SHAPE ONLY
    (values never read); interpreter + CostModel + forward tangent (linear in
    `value`, template's primal-VALUE clone passed, never its tangent);
    emitter arms that FOLD to static ops at emit time (`broadcast_in_dim`
    with the identity right-aligned axis map / the same static
    `stablehlo.slice` SLICE emits — templates unreferenced in the MLIR, the
    SLICE_LIKE precedent), EmitterTest pins + four new
    GradientEmissionCoverageTest sweep cases; host twins `broadcastToLike`
    (rank-polymorphic, no shims needed — no attrs to bake) and `sliceAtLike`
    + `sliceAtLikeRank{1,2,3}` (the `padToLikeRankN` mirror); synthesis
    `irBroadcastLike`/`irSliceAt` + `deriveResultIrType` and backward-solver
    arms (result IrType = template's; the value operand must NOT inherit it —
    smaller for BROADCAST_LIKE, bigger for SLICE_AT).
  - **The rules pair up and CLOSE**: SumToRule = `BROADCAST_LIKE(upstream,
    template=value)`, BroadcastLikeRule = `SUM_TO(upstream, template=value)`,
    PadToRule = `SLICE_AT(upstream, template=value, low)` (the PAD_TO node's
    own literal `low`, carried verbatim), SliceAtRule = `PAD_TO(upstream,
    template=value, low)`. In every rule the primal `value` operand becomes
    the SHAPE-ONLY template of its own adjoint (`readsPrimalOperandIndices =
    {0}` — the BroadcastRule/SliceRule inversion), so differentiating any
    number of times only alternates within a pair. Templates get no
    contribution (typed zero — pure shape sources).
  - **Still ruleless, deliberately** *(closed at §0.4.404 — see the next
    entry)*: SLICE_LIKE — its window offset is a runtime SUM of prior
    templates' extents along the axis, which no literal-`low` adjoint can
    express; second-order through a concat window still errored loudly. (A
    PAD_LIKE with prior-template offsets was the recorded shape of the fix.)
  - Certified: `DxirRuntimeExtentClosureTest` (interpreter pins for both new
    ops incl. rank extension + identity fast paths and the out-of-range
    refusal; reverse THROUGH SUM_TO / BROADCAST_LIKE / PAD_TO / SLICE_AT
    scalar bodies with hand-pinned analytic gradients — the exact composition
    that errored at HEAD; JVP⇄VJP cross-identities through BROADCAST_LIKE
    and SLICE_AT; a third-order pin showing the pair alternates without
    growing), the emitter pins + sweep cases above, `broadcastToLike`/
    `sliceAtLike` host pins incl. the padToLike round-trip, and E2E
    `hessian { Σ(x.broadcastTo(2,3) ⊙ x.broadcastTo(2,3)) }` = 4·I₃ through
    the real plugin (the user-visible face of the closure — previously
    unpinned).
- **B4 follow-through. Second order through a symbolic concat window ✅
  (§0.4.404)** — `OpKind.PAD_LIKE`, the VjpRule SLICE_LIKE was left without
  at §0.4.399, exactly the recorded fix shape: SLICE_LIKE's window offset is
  a runtime SUM of its PRIOR templates' extents, inexpressible as a literal
  `low`, so its adjoint is its own variadic transpose —
  `PAD_LIKE(value, outTemplate, priorTemplate₀…)` + attr `axis`: place
  `value` into a zero tensor of the outTemplate's ACTUAL runtime shape at
  offset `Σⱼ priorⱼ.dims[axis]` along `axis` (other axes at 0). Same
  prior-template convention (templates contribute SHAPE ONLY, values never
  read), same fixed-arity synthesis shims.
  - **The rules pair up and CLOSE, like the §0.4.399 pairs**:
    SliceLikeVjpRule = `PAD_LIKE(upstream, outTemplate=value, same priors,
    axis)`; PadLikeRule = `SLICE_LIKE(upstream, thisTemplate=value, same
    priors, axis)`. Both are variadic, so the static
    `readsPrimalOperandIndices` cannot express "value plus every prior" —
    the per-node `readsPrimalOperands` hook is authoritative (`{0} ∪ {2..}`;
    the ConcatRule precedent): the primal value becomes the shape template
    of its own adjoint and every prior is cloned into the gradient body,
    while operand[1] is never dereferenced (the upstream carries its shape).
    With this, EVERY runtime-extent adjoint op has a rule; the remaining
    rev∘rev refusals (MAXPOOL2D_GRAD, EMBEDDING_GRAD) are fused-adjoint
    design choices, not runtime-extent gaps.
  - **Full house wiring**: OpKind + CostModel arm + interpreter arm (the
    SLICE_LIKE outer/inner block copy inverted, into a zeroed buffer, with
    the same loud out-of-range refusal) + emitter `emitPadLike` (emit-time
    dims are concrete, so the offset and trailing pad fold to literals and
    it emits the same static `stablehlo.pad` as PAD_TO; templates
    unreferenced in the MLIR, DCE'd — EmitterTest pin) + forward tangent
    (linear in value; every template rides as its primal-VALUE clone) + host
    twins `padLikeStart` / `padLikeAfter{1,2,3}` over a shared `padWindow`
    (fixed-arity per PRIOR count — the IrVararg reason) + synthesis
    `irPadLike` (twin selected by prior count, axis as an Int const) + both
    IrType solvers (result IrType = the outTemplate's, operand[1]; neither
    the value — strictly smaller, one window — nor the priors may inherit).
  - Certified: PAD_LIKE interpreter pins (start window, one prior, two
    priors accumulating, LEADING axis, out-of-range refusal — the SLICE_LIKE
    suite mirrored); analytic reverse THROUGH SLICE_LIKE and THROUGH
    PAD_LIKE with hand-pinned gradients incl. zero template gradients;
    JVP⇄VJP cross-identities through both; **the flipped pin** —
    `DxirNestingMatrixTest`'s `reverseOverReverseRefusesSymbolicConcatSliceLike`
    became `…ThroughSymbolicConcatSliceLike`: rev∘rev over the symbolic
    concat now composes, each window's PAD_LIKE carrying its SLICE_LIKE's
    exact prior count, plus a concrete-dims numeric twin (the same
    gradient-shaped SLICE_LIKE body hand-built, seeded pullback = 4v = Hv);
    two new GradientEmissionCoverageTest sweep cases (slice_like / pad_like
    primals: second-order reverse bodies differentiate AND emit); host pins
    incl. the sliceLike ⇄ padLike round trip; and E2E
    `hessian { Σ concat(x, x)² }` = 4·I₃ through the real plugin — verified
    green at §0.4.403 HEAD too (forward-over-reverse rides SLICE_LIKE's
    tangent and never needed the rule), pinned now as the user-visible face.
- **B3. Forward transform through regions — v1 ✅ (§0.4.403), IF direct
  arm ✅ (§0.4.407)**: the
  COARSENED forward arm + the plugin's forward branch coarsening, lifting
  the straight-line gate that made every loop-bearing `jvp {}` fall back
  to the tape. Mirrors reverse-mode's history, mechanism for mechanism.
  - **The COARSENED forward arm** (`DxirForwardTransform`): the tangent
    of a coarsened op is the forward transform of its stored
    `primal_body`, spliced inline — the exact mirror image of
    `handleCoarsenedAdjoint`, which splices `gradient_body`: the jvp
    body's primal params seed from the cloned operand VALUES, its
    tangent params from the operand tangents, and the tangent is read
    off the spliced body's tangent return. The splice RECOMPUTES the
    coarsened op's interior primal values (the outer value stream only
    carries the fused result — the standard forward-mode recompute
    trade); `apply` recursion handles a COARSENED nested inside a
    primal_body. Multi-result COARSENED refuses loudly BY NAME up front
    (the §0.4.392 refusal discipline), before the clone loop can fail on
    an unrelated invariant.
  - **The plugin's forward branch coarsens** (`TlalocIrGenerationExtension`):
    region-bearing `jvp`/`jvp2`/`valueAndJvp`/`valueAndJvp2` bodies now
    run `PhiCalculus.apply` (+ the §0.4.174 region-body lift) before
    `DxirForwardTransform`, exactly as the reverse branch always has —
    a WHILE-bearing body closes (C5 unroll / C6–C9 engine corollaries)
    into shapes the transform handles. Straight-line bodies skip the
    pipeline, keeping the §0.4.372 path byte-identical; anything the
    coarsening leaves region-bearing still errors in the transform and
    falls back to the tape. The FIR checker's forward probe now skips
    IF-bearing bodies instead of red-squiggling shapes the extension can
    lower (same reasoning as its loop gate: PhiCalculus per keystroke is
    not check-time material).
  - Certified (`DxirForwardCoarsenedTest`, IR): tangent(coarsened f) ==
    tangent(decomposeCoarsened(f)) numerically — the decomposed body
    takes the ordinary per-op tangent path, so `decomposeCoarsened` is a
    free independent oracle — over a hand-built COARSENED (2a·da + da
    analytic pin), a rank-1 two-operand product (multi-operand seeding +
    non-scalar types through the splice), and a NESTED
    COARSENED-inside-primal_body; the JVP⇄VJP cross-identity through a
    `PhiCalculus.coarsenFunction` product (forward consumes primal_body,
    reverse consumes gradient_body — two attrs, two transforms, one
    number); the multi-result refusal pinned by message. E2E
    (`JvpLoopIntrinsicTest`, real plugin): `jvp {}` AND `valueAndJvp {}`
    over the Brachistochrone compound-velocity for-loop ((1+y)^5 via
    `v = v + v·y`), previously "kept original call", now lowering with
    no fallback — 5·(1+y)^4·dy pinned analytically and against a Double
    central difference, sentinel-guarded.
  - **The IF direct forward arm ✅ (§0.4.407)** — B3's recorded tail,
    closed. The tangent of an IF is a SECOND IF over the SAME cloned
    condition (piecewise-constant — zero tangent) whose branches yield
    the tangents of the primal branches' yields: paper C2's
    `d/dx φ(a, b) = φ(da/dx, db/dx)`, the forward twin of
    `handleIfAdjoint`. Branch bodies FLATTEN into the outer forward
    stream (both branches evaluate, the IFs only select) — the same
    unconditional-hoist trade `walkBranchReverse` has made since
    §0.4.23, and the only IF shape that survives to
    `DxirToIrSynthesis.irIfOp` and the emitter (empty-region,
    yield-only; the reverse adjoints' exact shape, so no synthesis or
    emitter change). **Multi-result IFs SUPPORTED**, not refused: the
    tangent IF mirrors the primal's index layout one-to-one (`types`
    verbatim, one tangent yield per terminator slot), with
    `DxirOpResult`-aware value/tangent resolution and typed integer
    zeros (I32/I64) for non-differentiable slots. Nested IFs recurse.
    Piecewise-constant tangents (SIGN/STEP/COMPARE/NOT/LAND) became
    STRUCTURAL nulls resolved lazily — the eager `const 0.0 : bool`
    they used to leave for every IF predicate was dead weight the
    synthesis could not emit (the actual reason `jvp {}` over if/else
    would still have fallen back). The FIR checker's forward probe now
    probes IF-bearing bodies (the §0.4.403 skip lifted; only loops
    nested inside IF branches keep the runtime backstop). Certified
    (`DxirForwardIfTest`, IR): analytic pins + central differences on
    BOTH sides of the branch; the JVP⇄VJP cross-identity against
    `handleIfAdjoint` on the same IF bodies (the strongest oracle —
    two different derivative encodings, one number); the post-lift
    yield-only shape pinned to emit exactly value-IF + tangent-IF,
    both empty-region; an unsafe-branch-op (SQRT) body staying finite
    on the else side (flattening's NaN confined to the discarded
    yield); nested-IF recursion; the multi-live-index MR IF against
    the reverse §0.4.155 walk; and forward-over-reverse THROUGH an
    IF-bearing gradient body (the `hessian {}` composition — refused
    wholesale before this slice). E2E (`JvpIfIntrinsicTest`): `jvp {}`
    and `valueAndJvp {}` over `if (x > 0f) x * x else -x` lower with
    no "kept original call", the lowered dxir pinned to CARRY the IF
    (so a rewrite silently closing the conditional cannot make the
    test vacuous), both branches + seed scaling pinned analytically
    and against a Double central difference, sentinel-guarded.
  - ~~Deferred tails~~ **both closed §0.4.430 (2026-09-20)**:
    **multi-result COARSENED tangents** — the walk grew a dedicated MR
    arm (value side clones the op verbatim via `opMulti`, attrs riding;
    tangent side runs ONE splice whose `(y₁..yₘ, dy₁..dyₘ)` returns are
    tracked per result index in `tangentResults`, resolved per
    `DxirOpResult` index at every consumption site) — production
    coarseners still emit single-result only (`coarsenFunction` bails on
    multi-return primals), so the pins are synthetic BY DESIGN: the
    transform contract is the deliverable. **IF inside a COARSENED
    primal_body** — the splice's clone loop widened to IF (branch bodies
    flatten first, the IF re-emits yield-only — the emitter-compatible
    shape, pinned) and to region-free multi-result ops (a nested MR
    COARSENED clones via `opMulti`), with `DxirOpResult` references
    re-wrapped per index (`resolveSpliced`). Certified in
    `DxirForwardCoarsenedTest`: MR tangents against a hand-decomposed
    equivalent + analytic quarter-integer pins; the JVP⇄VJP
    cross-identity over z = c₀·c₁ (forward consumes `primal_body`,
    `handleCoarsenedAdjoint` consumes `gradient_body` with K=2
    upstreams); IF and MR-IF primal_bodies against the §0.4.407
    top-level arm, analytic both branches, and reverse mask-math
    gradient bodies; nested MR-COARSENED-inside-primal_body recursion.
    The cross-identity EXPOSED a pre-existing reverse-side bug: the
    CSE operand key was bare-id (so `MUL(seed, %c#0)` merged with
    `MUL(seed, %c#1)` — same source id) and every id-keyed operand
    canonicalization in `applyCSE`/`applyConstFold` collapsed a
    `DxirOpResult` to its source op (result 0) — the §0.4.130
    terminator bug's operand-position twin. Fixed via `canonicalRef`
    (index-preserving resolution) + `(id, index)` CSE operand keys;
    regression pinned reverse-only in `CoarsenedOpTest`
    (`…ProductConsumerKeepsResultIndicesDistinct`). Still deferred:
    nesting through region-bearing bodies stays with B4's recorded
    tail; user (B5) multi-result f remains unconstructible by
    `validateCoarsenedShape` — its own recorded tail.
- **B4. Nesting matrix ✅ DONE, §0.4.401 (2026-09-20)** — the full 2×2
  certified at IR level (`DxirNestingMatrixTest`), with **zero
  production-code changes**: both transforms already composed mechanically,
  and the "composition plumbing" the §0.4.399 note anticipated turned out
  to be two one-liners, not machinery.
  - **fwd∘fwd** ✅: the second application re-tangents the jvp's params
    (`x, d_x` → `x, d_x, d_x, d_d_x`) and splits its returns again; with
    the last tangent seeded zero, return 4 is the bilinear form
    `d²f(x)[u,v] = uᵀHv`. Pinned on Σx³ (= 6·Σx⊙u⊙v) and, basis-seeded on
    the §0.4.394 hessian body Σ exp(x), entrywise against BOTH the analytic
    diag(exp x) and fwd∘rev HVP columns.
  - **rev∘fwd** ✅: the plumbing is a RETURN PROJECTION — same params, same
    body, tangent return only — after which the jvp is a legal scalar-return
    reverse input. `∇ₓ⟨∇f, v⟩ = Hv` pinned analytically and numerically
    equal to the fwd∘rev HVP (the cross-oracle).
  - **rev∘rev** ✅ where §0.4.399 predicted: the seeded pullback of the
    gradient function IS the HVP — `R(R(f), seedAsParam)(v, x) = vᵀ∂(∇f)/∂x
    = Hv`, no scalarization step at all. Certified over Σx³, over an
    equal-rank size-1 stretch (the first reverse pass emits SUM_TO, the
    second differentiates through it via BROADCAST_LIKE — the §0.4.399
    closure exercised end to end, body-op pinned), and over a
    concrete-dims concat (ConcatRule's static-SLICE branch → SliceRule:
    works; the gap is symbolic-only).
  - **rev∘rev refusals pinned loud** (the §0.4.392 precedent): gradient
    bodies carrying MAXPOOL2D_GRAD or EMBEDDING_GRAD fail with "no VJP rule
    registered for <op>", asserted by name. The fused conv/pool/embedding
    adjoints stay VjpRule-less by design (their second derivative would need
    the adjoint-of-adjoint expansion B5/C-era work can decide on). The third
    refusal originally pinned here — the symbolic-concat SLICE_LIKE — was
    closed at §0.4.404 (PAD_LIKE) and its pin flipped to a positive cert.
  - **Third order** ✅ for free: F(F(R(Σx⁴)))(x, u, v, 0) = 24·x⊙u⊙v pinned.
  - Deferred tails: nesting through region-bearing bodies rides on B3
    (forward-v1 refuses regions before any composition question arises);
    user-facing n-th-order intrinsics (`reverseDerivative{2..4}` spellings)
    are a synthesis-surface question, not an IR one — the IR compositions
    they'd lower to are what this slice certified.
- **B5. User-defined custom derivatives — DONE (§0.4.415 reverse +
  §0.4.416 forward; ratified by Pedro 2026-09-20 with the design doc's
  recommended answers).** Full
  design + implementation record in
  [CUSTOM_DERIVATIVES_DESIGN.md](CUSTOM_DERIVATIVES_DESIGN.md) (§7 holds
  what v1 taught). Shipped: `io.tlaloc.autograd.customVjp` / `customVjp2`
  — Candidate A call-forms whose host stubs simply apply `f` (documented
  asymmetry vs `grad`'s `pluginMissing` — the primal IS the right
  plain-Kotlin meaning; only the derivative attachment needs the plugin).
  The FIR arm lowers both lambda literals and emits ONE `COARSENED` with
  `primal_body`/`gradient_body`/`reads_primal_indices` (computed from
  vjpFn's actual param uses) + `user_gradient = true`; `vjpFn`'s
  `(upstream, x…)` order IS `handleCoarsenedAdjoint`'s contract, and
  `customVjp2`'s trailing `Pair(dA, dB)`/`dA to dB` unboxes to the
  2-return convention. Assign-then-apply within the body works; escapes
  (re-binding, passing out) refuse loudly by name, as do non-const
  captures (literal-initialised local `val`s inline).
  - **Reverse mode splices the USER body verbatim** — certified by a
    deliberately NON-mathematical `3·upstream` vjpFn E2E (scalar + tensor;
    composition would give `2x`) plus `customVjp2` (7/11), stopGradient
    sugar (∇ Σ x·sg(x) = x), a captured-val case, and rev∘custom nesting
    at IR level. Two new pipeline pieces: a surviving mid-body COARSENED in
    the gradient function decomposes (`decomposeCoarsened`) before
    synthesis, and each user contribution wraps in the new
    `OpKind.CHECK_SHAPE_LIKE` runtime shape assert (interpreter arm + host
    `checkShapeLike` + synthesis arm + identity VjpRule + forward tangent
    arm; transform-time failure when shapes are concrete; emitter REFUSES
    it by name — pinned — so customVjp gradients are
    host/interpreter-certified in v1).
  - **Forward mode refuses** (`user_gradient` unless `tangent_body`): IR
    pin + E2E — `jvp {}` over a customVjp body is a compile-time ERROR
    naming `user_gradient` (the §0.4.392 no-silent-fork principle).
  - **Debug oracle shipped**: `checkCustomVjp` (JVP⇄VJP inner-product
    identity via central differences; pure host, opt-in, fails
    straight-through estimators by design) — green/red certified.
  - **§0.4.416 — the forward side (Candidate C) DONE**:
    `io.tlaloc.autograd.customJvp` / `customJvp2` /
    `customVjpJvp` / `customVjpJvp2` (the same host-stub asymmetry: all
    return `f`). One shared FIR arm lowers whichever bodies a form
    carries; `jvpFn`'s declared `(primals…, tangents…) → dy` order IS
    `DxirForwardTransform`'s own params-then-d_params emission order, so
    the splice adapter is the identity at every arity (`tangent_body`
    attr, validated at COARSENED construction: 2·N params typed like the
    operands twice over, single return typed like the result;
    `gradient_body` becomes optional ONLY for the customJvp-only shape).
    The forward transform's COARSENED arm splices `tangent_body` verbatim
    for user nodes — machine nodes keep the §0.4.403 auto-tangent
    byte-identically — wrapping sentinel-typed tangents in
    `CHECK_SHAPE_LIKE` against the node's own value clone (the forward
    twin of the reverse contract). Both refusals now mirror:
    customVjp-only still refuses `jvp {}` naming `user_gradient`;
    customJvp-only refuses `grad {}` naming `customJvp` (IR pin + E2E
    compile-time ERROR). `customVjpJvp` flips both — and each mode runs
    ITS body, pinned with deliberately INCONSISTENT bodies (grad → vjpFn's
    5, jvp → jvpFn's 3, math says 2x) and with CONSISTENT ones through the
    JVP⇄VJP cross-identity E2E (⟨∇f, v⟩ = jvp(x, v), tensor path).
    hessian (forward-over-reverse) composes THROUGH a customVjpJvp node —
    the tangent splice fires inside the reverse-produced body on the
    cloned COARSENED — certified at IR level AND E2E (H = 2I over the
    sentinel tensor path). Plumbing: the jvp and hessian/assembly plugin
    branches now run `decomposeCoarsened` before synthesis whenever a
    COARSENED survived (the §0.4.415 grad-branch treatment; no-op on
    every pre-B5 path), which is also what lets machine-COARSENED forward
    bodies reach synthesis.
  - Recorded tails: GPU emission of user gradient bodies; multi-result
    `f`; non-const captures; Candidate B (serialized-dxir); the seeded
    branches (`vjp {}` / `jacobianReverse`) don't yet run the decompose
    step — they fall back loudly.

### Phase C — op families DiffKT has that the IR lacks

- ✅ **C1. Special functions — DONE (§0.4.402)**: `LGAMMA` and `DIGAMMA`,
  tensor and scalar, full vertical; `TRIGAMMA` (= polygamma(1)) landed as an
  INTERNAL op — DIGAMMA's adjoint/tangent emit it — with general
  `POLYGAMMA(n)` deliberately out of scope then (closed by §0.4.405 below).
  - Host: pure-Kotlin Double kernels in `:core/SpecialFunctions.kt` — Lanczos
    g=7/n=9 lgamma with reflection below 0.5, digamma/trigamma by
    recurrence-shift past 8 + Bernoulli asymptotic series, reflection on the
    negative axis, exact floor-test pole detection. ONE source of truth:
    `:ir` depends on `:core`, so the interpreter arms, the tensor host ops
    and the five-overload scalar sets all call the same three functions —
    host/interpreter agreement is by construction. Validated to ~1e-11
    absolute against hand-pinned references (ψ(1)=−γ, ψ(0.5)=−γ−2ln2,
    lgamma(0.5)=ln√π, ψ₁(1)=π²/6, the Γ/ψ/ψ₁ recurrences, reflection) and by
    the derivative-chain central-difference oracle (lgamma′≡ψ, ψ′≡ψ₁).
  - IR: three OpKinds; VjpRules LgammaRule `MUL(DIGAMMA(x), up)` and
    DigammaRule `MUL(TRIGAMMA(x), up)` — fresh special-function node over the
    cloned operand, values-only, sentinel-safe by construction; forward
    tangents mirror them. TRIGAMMA REFUSES both transforms loudly by name
    (pinned) — its derivative is polygamma(2), out of scope.
  - Emitter: **the first CHLO emissions** — `chlo.lgamma %x : t -> t`,
    `chlo.digamma`, and `"chlo.polygamma"(splat 1.0, x)` for TRIGAMMA. A
    pre-wiring spike proved the GB10's XLA PJRT parses + legalizes CHLO
    (values matched references to f32), so no refusal arm was needed;
    `PjrtLgammaDigammaSmokeTest` certifies forwards + both gradient graphs on
    the GPU (grads within 1e-5 of the interpreter). EmitterTest pins all
    three spellings; lgamma+digamma losses joined
    `GradientEmissionCoverageTest`'s sweep (the digamma loss covers the
    polygamma emission). RoundTripTest EXCLUDES all three with the reason
    pinned: `stablehlo-translate --serialize` targets VHLO, which does not
    cover CHLO — the PJRT smoke is their live oracle.
  - User surface: `:core/ops` tensor `lgamma()`/`digamma()` (+ `trigamma()`
    as public gradient machinery — synthesised gradient bodies call it);
    five-overload scalar sets; FIR `UNARY_OP_MAP` entries for
    `io.tlaloc.core.{lgamma,digamma}` + `io.tlaloc.core.ops.{lgamma,digamma}`
    (NO trigamma entry, NO `kotlin.math` spelling — none exists). Synthesis:
    one `irSpecialUnary` arm — tensor via `opsTensorSymbol`, scalar via the
    §0.4.377 `irCoreScalarCall` path (the first ops after sigmoid to need
    it); all three kinds in the three IrType-solver unary lists.
  - Certified: IR analytic pins with non-uniform upstream (`∇ₓ Σ lgamma(x)⊙w
    = w⊙ψ(x)`, digamma twin), JVP⇄VJP cross-identity through
    `Σ lgamma(digamma(x⊙w))` (both rules + TRIGAMMA chained), f32
    central-difference cross-check through the interpreter; E2E `grad {}` ×4
    (scalar lgamma/digamma — the digamma one pins that TRIGAMMA synthesises
    despite having no FIR entry — and the tensor twins, no tape fallback);
    host pins; GPU smoke.
  - ~~Deferred tail: general `POLYGAMMA(n)`~~ **CLOSED, §0.4.405** — see the
    entry below.
- ✅ **C1 deferred tail — DONE (§0.4.405)**: general `POLYGAMMA(n)` +
  TRIGAMMA's own VjpRule/tangent arm — the special-function family is now
  CLOSED under differentiation (`d ψ⁽ⁿ⁾ = ψ⁽ⁿ⁺¹⁾` climbs the ladder to any
  depth), flipping the §0.4.402 pinned TRIGAMMA refusal into positive
  second/third-order certs.
  - Host: `Double.polygamma(n)` in `:core/SpecialFunctions.kt` (shared by
    interpreter/host/scalars as ever): n = 0 delegates to digamma; n ≥ 1 runs
    recurrence-shift past 10 + n (the truncated-series error grows
    combinatorially with n, so the threshold must too) into the
    n-times-differentiated digamma asymptotic (Bernoulli terms through B₁₂,
    factorial-ratio running products), with the differentiated reflection
    (−1)ⁿψ⁽ⁿ⁾(1−x) = ψ⁽ⁿ⁾(x) + πⁿ⁺¹·Pₙ(cot πx) covering the negative axis
    via the cot-derivative polynomial recurrence P₀ = t, Pₖ₊₁ = −(1+t²)Pₖ′.
    Orders outside 0..100 refuse (Double factorial-precision regime). THE
    validation pin: n = 1 through the general scheme agrees with the
    independent §0.4.402 trigamma kernel to 1e-11 across asymptotic /
    shift / reflection / near-pole probes; plus ψ₂(1) = −2ζ(3),
    ψ₂(0.5) = −14ζ(3), ψ₃(1) = π⁴/15, ψ₃(0.5) = π⁴, ψ₄(1) = −24ζ(5),
    reflection ψ₂(−0.5), the (−1)ⁿn!/xⁿ⁺¹ recurrences for n = 1..5, and the
    n = 1..4 central-difference derivative chain. Pole convention: +∞ for odd
    n (even-order pole), NaN for even n (sign-indefinite) — the
    trigamma/digamma conventions generalised. Five-overload scalar set +
    `:core/ops` tensor `polygamma(n: Int)` (one positional param, no
    defaults — the K2 named-arg landmine).
  - IR: ONE new OpKind, `POLYGAMMA`, with the order as a compile-time integer
    `order` attr (in the CSE signature since §0.4.366, so ψ⁽ⁿ⁾/ψ⁽ⁿ⁺¹⁾ over
    one operand never deduplicate). TRIGAMMA kept as-is (disturbs less: the
    §0.4.402 DigammaRule/emitter/tests all stand), and the FIR NORMALISES
    user `polygamma(0)/(1)` to DIGAMMA/TRIGAMMA so ψ₁ nodes CSE with the
    ones digamma's adjoint emits — POLYGAMMA nodes carry order ≥ 2 by
    invariant. TrigammaRule = `MUL(POLYGAMMA(2, x), up)`, PolygammaRule =
    `MUL(POLYGAMMA(order+1, x), up)`; tangent arms mirror both (literal
    attrs, never extents — sentinel-safe). Certified: analytic pins with
    non-uniform upstream, JVP⇄VJP cross-identity, f32 central differences,
    and the flipped refusal — rev∘rev AND fwd∘rev (the hessian composition)
    through `Σ digamma(x)` both yield diag(ψ₂) HVPs that agree, plus a
    third-order pin (`DxirPolygammaGradTest`; the §0.4.402 refusal test is
    now a positive both-transforms pin).
  - Emitter: POLYGAMMA emits `"chlo.polygamma"(splat n.0, x)` — the §0.4.402
    trigamma spelling with the order splat generalised (one shared
    `emitPolygamma` arm). EmitterTest pins the order-3 spelling; the
    trigamma + polygamma(3) losses joined GradientEmissionCoverageTest (their
    adjoints sweep orders 2 and 4); RoundTripTest exclusion comment extended
    (CHLO is still not a VHLO serialization citizen). Live oracle
    `PjrtPolygammaSmokeTest`: trigamma and ψ₂ losses + gradient graphs
    (orders 1/2/3 in flight) compile and run on the GB10 within ~1.5e-6
    relative of the interpreter — a REAL two-implementation cross-check,
    since XLA legalizes polygamma through its own zeta-based scheme.
  - User surface: FIR arm for `io.tlaloc.core{.ops,}.polygamma` (receiver +
    one Int literal, the §0.4.369 clip-bounds discipline; named-arg unwrap
    handled; 0..100 bound mirrored). Synthesis: `irPolygamma` — tensor via
    the Int-arg `tensorUnaryCall` extension, scalar via the
    `coreScalarIntArgSymbolFor` sibling of the §0.4.377 path; POLYGAMMA in
    all three IrType-solver unary lists. E2E ×5 in
    `PolygammaGradientTest`, no tape fallback: scalar orders 0/1/2 (the
    normalisation pins: d polygamma(0) = ψ₁, d polygamma(1) = ψ₂,
    d polygamma(2) = ψ₃) + tensor orders 1/2.
- **C2. Trig tails** ✅ **DONE, §0.4.395 (2026-09-20)**: `TAN` and `ATAN`,
  tensor AND scalar, full vertical — audit confirmed these are the only
  ones DiffKT has (no floor/ceil/round/atan2; those stay optional extras,
  not parity items).
  - Two new OpKinds with every engine armed: interpreter (`kotlin.math`
    through Double, the house convention), CostModel (the SIN/COS
    transcendental bucket), TileFusion/PhiCalculus elementwise sets.
  - VjpRules: TanRule `d tan = (1 + tan²x)·up` (tan-recompute form — CSEs
    with the primal's TAN, reads no extents) and AtanRule
    `d atan = up/(1 + x²)`; the `1` splats ride `splatConst` so both are
    sentinel-safe. Forward tangents: TAN reads its own value stream
    (`(1 + y²)·dx`), ATAN the operand (`dx/(1 + x²)`).
  - Emitter: `stablehlo.tan` is accepted by the GB10's XLA (certified in
    `PjrtTanAtanSmokeTest` — the fallback divide(sine, cosine) spelling was
    never needed); ATAN emits `stablehlo.atan2(x, splat 1.0)` since StableHLO
    has no unary atan. Both pinned in `EmitterTest`, both losses in
    `GradientEmissionCoverageTest`'s sweep, both in the round-trip list.
  - User surface: `:core/ops` tensor `tan()`/`atan()` + the five-overload
    scalar sets (§0.4.377 pattern); FIR `UNARY_OP_MAP` entries for all four
    Tlaloc FQNs AND `kotlin.math.tan`/`atan` — the top-level stdlib
    spellings have no receiver, so the UNARY arm grew a single-argument
    fallback. Synthesis: `irTan`/`irAtan` (tensor via `opsTensorSymbol`,
    scalar via `kotlin.math`), both kinds in all three IrType-solver
    elementwise lists.
  - Certified: IR-level analytic pins with non-uniform upstream + the
    JVP⇄VJP cross-identity through `atan(tan(x)⊙w)`; five E2E `grad {}`
    spellings through the real plugin with no tape fallback (scalar
    receiver ×2, bare `kotlin.math` ×1, tensor ×2); GPU smoke grads within
    2.3e-5 of the interpreter.
- ✅ **C3. `REVERSE` (flip) op — DONE (§0.4.396)**: DiffKT `flip(axes)` as a
  new OpKind with the full vertical, and the easiest gradient story in the
  catalogue: REVERSE is an involution and SELF-ADJOINT (`dx = REVERSE(dy,
  same axes)`, forward tangent likewise), and its `dimensions` attr is axis
  POSITIONS only — no extent is ever read, so both rules are sentinel-safe
  by construction with no SUM_TO/PAD_TO-style runtime-extent template.
  - IR: OpKind + interpreter (per-axis index inversion on the
    `evalTranspose` stride walk), CostModel movement-only bucket,
    `ReverseRule` + the linear forward-tangent arm (both re-emit REVERSE
    with the attr verbatim).
  - Emitter: `stablehlo.reverse %x, dims = […]` pinned in `EmitterTest`
    (plus empty/duplicate/out-of-range refusals); a flip loss in
    `GradientEmissionCoverageTest`'s sweep; REVERSE in the round-trip list.
    Real-XLA parse certified by `PjrtFlipSmokeTest` on the GB10 (forward +
    gradient graphs, exact agreement with the interpreter) — the live
    oracle, since `stablehlo-translate` is not on this machine's PATH.
  - User surface: `:core/ops` `DTensor.flip(vararg axes)` (shape-preserving,
    so the receiver's precise shape type survives; negative axes normalize)
    + fixed-arity `flipAxes{1,2,3}` synthesis delegates (the usual IrVararg
    reason). FIR: `flip` joins `SHAPE_OP_SET` (vararg-Int flattening arm);
    result DxirType = operand's own, sentinels included. Synthesis:
    `irReverse` → the delegate by axis count, axes as positional Int consts;
    REVERSE in the shape-preserving unary lists of all three IrType solvers
    (forward rank-2 + rank-4, backward fixpoint).
  - Certified: interpreter pins (trailing axis, LEADING axis on a
    non-square shape — the non-contiguous copy — both axes, involution);
    `∇_a Σ flip(a)⊙b = flip(b)` with non-uniform upstream; JVP⇄VJP
    cross-identity through `Σ tanh(flip(a)⊙b)`; E2E `grad {}` through the
    real plugin on one AND two axes with no tape fallback; host pins with
    refusals; GPU smoke. The runtime tape deliberately has no flip
    producer (the §0.4.382 concat precedent) — synthesis is the only path.
  - Still open (unchanged): re-expressing the conv adjoints'
    `window_reversal` special-casing through REVERSE of the kernel's
    spatial axes — noted in `OpKind.kt`, deliberately not attempted; the
    fused adjoints stand certified as they are.
- **C4. Item-4 tails** *(reclassified beyond-parity by the audit —
  DiffKT pooling is non-overlapping-only, conv has no groups/dilation)*:
  ~~CONV_TRANSPOSE2D's own VJP~~ (landed §0.4.391, GPU emission §0.4.393);
  ~~overlapping-window maxpool VJP GPU path~~ (closed as INHERENT §0.4.392 —
  `select_and_scatter` cannot express the all-ties convention; host and
  interpreter handle overlapping windows, emission refuses loudly).
  ~~grouped/depthwise conv (feature_group_count > 1)~~ — **landed at IR
  level §0.4.429**, the §0.4.418 SPARSE_MATMUL precedent. The attr rides
  StableHLO's own layout convention (kernel input-feature dim = Ci/g,
  output-feature dim = full Co; depthwise = `g == Ci` falls out with no
  arm of its own): interpreter `conv2dCore` grouped for BOTH conv kinds;
  `Conv2dRule` rides the literal attr and the fused adjoints group-slice
  their channels SYMMETRICALLY at execution time (dX: upstream/kernel
  rows by Co/g → input channels by Ci/g; dW the mirror), each per-group
  call being the fgc = 1 adjoint verbatim; forward tangent was already
  attr-generic (the bilinear arm replays `node.attrs`); primal emission
  already carried `feature_group_count` — certified against real XLA by
  GPU smoke (grouped loss + JVP tangent graph, diff 5e-7/0.0). Oracles:
  hand-computed 1×1 grouped conv on a quarter-integer grid, grouped ==
  concat-of-per-group-ungrouped-convs equivalence (CONV2D, depthwise,
  CONV_TRANSPOSE2D, general attrs), FD + JVP⇄VJP cross-identity, and
  gradients' own slicing symmetry (`DxirGroupedConvTest`). **Named
  §0.4.429 deferrals** (each refuses loudly, naming the attr): grouped
  TRANSPOSED-conv VJP (`ConvTranspose2dRule` refuses at transform time;
  the index-inversion eval would need the same per-group lift); grouped
  ADJOINT emission (XLA spells grouped data-grad through a per-group
  kernel reshuffle and grouped kernel-grad through `batch_group_count` —
  neither attempted; interpreter handles groups, GPU reverse-mode
  refuses); the USER SURFACE (host twins/`conv2d(..., groups)` FIR
  arity/K2 synthesis — `irConv` still rejects fgc > 1 into the tape
  fallback, whose host twins are groups-free, so nothing user-reachable
  can convolve the wrong way).
- ✅ **C5. `integral` — DONE (§0.4.411)** *(audit)*: Romberg quadrature with
  FTC-wired derivatives, as the scalar host surface it naturally is.
  - `:core/Integral.kt` (the §0.4.402 one-source-of-truth convention):
    `rombergIntegrate(a, b, maxDepth = 16, tol = 1e-8, f)` — Richardson-
    extrapolated trapezoid tableau, two live rows, odd-nodes-only refinement,
    mixed absolute/relative early exit — returning `RombergResult(value,
    depth, converged)` (non-convergence is REPORTED, not thrown); `integral`
    Double + Float sugar; `integralWithBoundGrads` = (value, dA, dB) with the
    bound derivatives wired ANALYTICALLY by the fundamental theorem of
    calculus (dA = −f(a), dB = f(b) — two evaluations, no differentiation
    through the quadrature loop; scalar, so the same numbers serve JVP and
    VJP). Reversed bounds need no special case (every formula is linear in
    b − a).
  - Certified (`IntegralTest`): closed forms ∫₀¹x² = 1/3 (depth-2 exact,
    1e-12), ∫₀^π sin = 2, ∫₁ᵉ 1/x = 1, steep ∫₀¹e^(−50x) (all 1e-9);
    Float surface 1e-6; orientation + empty-interval pins; convergence
    behavior pinned through `RombergResult` (x² early-exits at depth 2,
    the steep case needs depth ≥ 5, and starved of depth it reports
    `converged = false` with the best diagonal); FTC wiring vs central
    differences OF THE QUADRATURE ITSELF (tol-1e-12 inner quadratures keep
    FD noise below the 2h denominator); parameter-derivative contract
    d/dθ ∫ f(x;θ) dx = ∫ ∂f/∂θ dx three ways (central difference /
    quadrature of ∂f/∂θ / closed form) for f(x;θ) = e^(−θx).
  - **`grad {}` surface is B5-gated (recorded)**: the `f` argument is an
    opaque Kotlin lambda to the FIR lowering, so a grad{}-integrable
    `integral` op is exactly a custom-derivative citizen
    ([CUSTOM_DERIVATIVES_DESIGN.md](CUSTOM_DERIVATIVES_DESIGN.md) — B5 v1
    landed §0.4.415, so the gate is open; the integral spelling itself is
    still its own slice). Lowering shape now that B5 landed: an INTEGRAL region op whose
    body is the lowered `f`, primal = Romberg over interpreted body
    evaluations, VJP = FTC bound adjoints (−f(a)·v̄, f(b)·v̄ via two body
    evaluations) + Leibniz parameter adjoints (quadrature over the body's
    own VJP w.r.t. captured params — the same tableau, adjoint integrand),
    forward tangent the mirror image. No plugin work forced into C5.
  - ✅ **`grad {}` surface v1 — the customVjp route (§0.4.426)**: the open
    gate used exactly as a USER would, which is its point. E2E
    (`IntegralGradientTest`, compiler-plugin): `grad {}` through a
    `customVjp` whose primal is a user-spelled fixed-node quadrature
    (Simpson over {0, ½, 1}, exact for the x²-family — the quarter-integer
    grid) and whose vjpFn is the Leibniz adjoint — d/da ∫₀¹ a·x² dx = 1/3
    and d/da ∫₀¹ (a·x)² dx = 2a/3 pinned analytically, no fallback;
    `customVjpJvp` supplies both bodies and the scalar JVP⇄VJP
    cross-identity holds end to end (value a²/3, tangent dp·2a/3, grad
    2a/3 — one analytic family, both modes, one number). Host side:
    `integralWithParamGrad(a, b, f, dfdp)` (:core/Integral.kt) is the
    Leibniz sugar — (value, dP) with dP = ∫ ∂f/∂p over the same Romberg
    kernel, analytic in the integrand, the parameter-side twin of
    `integralWithBoundGrads` — pinned against the a·x² and e^(−θx) closed
    forms in IntegralTest. **NAMED DEFERRAL — the INTEGRAL region op**:
    the reusable `integral(a, b) { f }` spelling above (Romberg over the
    interpreted lowered body, adaptive depth, FTC bound adjoints, Leibniz
    adjoint as a quadrature over the body's own VJP) stays open, because
    B5 v1 REQUIRES lambda literals at the customVjp call site inside the
    differentiated body — a library wrapper cannot exist under the
    no-escape rule by design, so the region op waits on either a B5
    restriction-loosening slice or a dedicated FIR recognition of
    `integral` itself.

### Phase D — random (DiffKT `RandomKey` parity)

- ✅ **D1. Stateless PRNG foundation — DONE (§0.4.408)**: `RandomKey` (two
  32-bit words, the JAX convention; DiffKT's SHA-512 counter design replaced
  by JAX's threefry, same statelessness contract) + `split(n)` + `foldIn` +
  `uniform` / `normal` draws, host and interpreter.
  - `:core/Random.kt` — ONE source of truth (the §0.4.402 SpecialFunctions
    convention): `threefry2x32` at the reference 20-round schedule, validated
    against the Random123 known-answer vectors; `threefryBits(key, n)` in
    JAX's CLASSIC counter layout (split-halves over `iota(n)`, odd-n zero pad
    at the END with the last output word dropped) — pinned bit-for-bit
    against JAX 0.10 with `jax_threefry_partitionable=False` (vectors
    generated on this machine from the iree venv's JAX install, including the
    odd-n lanes where a wrong pad side shows first); `split` = bits over
    `iota(2n)` reshaped to key pairs, `foldIn(data)` = the block function at
    counter `(0, data)` — both JAX-exact. `uniformFloats` = the JAX mantissa
    recipe (`bits >>> 9 | 0x3f800000`, bitcast to [1,2), subtract 1) —
    JAX-exact to the bit. `normalFloats` = Box-Muller over one
    `uniform(2n)` stream (radial half + angular half, `1−u` keeps the log
    finite; sine partner discarded) — the one DOCUMENTED deviation from JAX
    (which uses `√2·erfinv(2u−1)`): same distribution, different bits, chosen
    because it validates against elementary identities. User surface: host
    tensor wrappers `RandomKey.uniformVector/uniformMatrix/normalVector/
    normalMatrix` (draws over the flat index space).
  - IR: `RNG_UNIFORM` / `RNG_NORMAL` — zero-operand creation ops, attrs
    `key0`/`key1`/`dims` all literal (no FIR lowering exists, so `grad {}`'s
    -1 sentinels can never reach them; the interpreter REQUIRES the `dims`
    attr to equal the concrete result dims). (§0.4.432 later adds the
    ALTERNATIVE two-operand scalar-I32 runtime-key form — see item 2 below;
    `dims` stays literal in both.) Interpreter arms call the same
    `:core` kernels — host/interpreter bit-exact BY CONSTRUCTION and pinned
    in `DxirRngTest`. CostModel arms (threefry ≈ 26 flops/elem, Box-Muller
    ≈ 64). NON-differentiable in D1, deliberately: both transforms refused
    loudly by name (a draw is piecewise-constant in its key) — refusals
    FLIPPED to the zero-gradient/zero-tangent arms in D2 v1 (§0.4.413).
  - Emitter: a DELIBERATE named refusal, adjudicated rather than spiked:
    `stablehlo.rng_bit_generator`'s threefry counter layout is XLA-internal
    and does not reproduce the JAX-classic stream these kernels pin — JAX
    itself never emits rng_bit_generator for threefry keys, it emits the
    20-round block as explicit HLO ops precisely to keep streams
    engine-independent. Shipping it would silently fork the random stream
    between interpreter and GPU. Refusal pinned in `EmitterTest`;
    `GradientEmissionCoverageTest` exclusion documented in place.
  - Certified: Random123 KATs; JAX-generated layout pins for bits (n = 1, 5,
    6, 7), uniform (bit-exact, even + odd), split(3), foldIn; split
    independence + determinism; uniform range/moments and normal
    mean/var/skew at a FIXED key (deterministic pins, no flake surface);
    Box-Muller numpy-reference pins; interpreter-vs-host bit-exactness.
  - **Deferred tails (recorded)**: explicit-threefry StableHLO emission
    (JAX's approach — the only honest GPU path) — DONE §0.4.422;
    FIR/`grad {}` surface for draws inside lambdas (D2-era) — DONE
    §0.4.421; `permitReuse`/`DiffktRandom` wrapper sugar;
    cauchy/chiSquare (inverse-CDF sugar over uniform) — DONE §0.4.431
    (D3 below).
- ✅ **D2 v1. Reparameterized gradients at IR level — DONE (§0.4.413)**:
  random draws differentiate in both transforms with the correct (zero)
  gradient in the key, unlocking the reparameterization trick
  `sample = loc + scale ⊙ ε` — d loss/d loc and d loss/d scale flow through
  the ordinary ADD/MUL rules while ε contributes nothing.
  - Reverse: `RngDrawRule` returns the EMPTY contribution list (the
    SIGN/COMPARE zero-gradient convention at arity 0 — a draw is
    piecewise-constant in its literal key attrs with no operands to
    propagate to). §0.4.408's loud refusal flipped; its spirit survives as
    the pinned zero, with the draw ON the differentiable path so the walk
    provably reaches it with accumulated upstream.
  - Forward: structural-zero tangent (the §0.4.407 lazy-null convention,
    same list as SIGN/STEP/COMPARE).
  - The determinism half of the contract falls out of D1's literal-attr
    design: when an adjoint READS ε (MulRule's `d scale = upstream ⊙ ε`),
    the RNG op is cloned into the gradient body via `usedByAdjoint` with
    its baked `key0`/`key1`/`dims`, so the gradient's re-draw is the SAME
    stream — same key → same ε, pinned explicitly.
  - Certified (`DxirRngTest`): grad of Σ (loc + scale⊙ε)² (odd length — the
    end-pad counter lane rides through the cloned draw) against the
    analytic oracle with ε recomputed host-side from the SAME key via
    `normalFloats`; gradient evaluated twice bit-identical; zero gradient
    in key pinned as `d x = u` bit-exact for Σ (u⊙x); jvp tangent = Σ ε⊙vx
    with ε's zero tangent; JVP⇄VJP cross-identity through the reparam loss.
  - **Deferred tails (recorded)**: the FIR/`grad {}` USER-SURFACE spelling —
    lowering `normal(key0, key1, dims…)`/`uniform(…)` calls inside `grad {}`
    lambdas to the zero-operand RNG ops (literal key words + dims in v1, the
    K2 named-arg landmine; a `RandomKey`-typed lambda param is NOT v1) plus
    the `irRngNormal`/`irRngUniform` synthesis delegates for cloned draws in
    synthesised gradient bodies; GPU draws still gated on the D1
    explicit-threefry emission tail (a reparameterized loss's GRADIENT graph
    contains a cloned draw, so emission coverage joins only then);
    Gamma/Dirichlet implicit reparameterization (the DiffKT stretch goal).
- ✅ **D3. Distributions — cauchy / exponential / chiSquare — DONE
  (§0.4.431)**: the DiffKT `random/` distribution draws, landed as PURE
  ELEMENTWISE TRANSFORMS of the D1 streams — no new randomness primitive,
  no new OpKind (the recorded compositional design: a draw-then-transform
  graph differentiates as a constant automatically through the D2
  zero-gradient RNG arms, and emits to GPU through §0.4.422's explicit
  threefry for free).
  - `:core/Random.kt`: `cauchyFloats` = the quantile transform
    `tan(π(u − ½))` over one `uniformFloats` stream (f32 centre/scale,
    tangent through Double — deliberately arm-for-arm the
    RNG_UNIFORM → SUB → MUL → TAN composition, so host and lowered draws
    are bit-identical on the same JVM); `exponentialFloats` =
    `−ln(1 − u)` (unit rate; `1 − u ∈ (0, 1]` keeps the log finite, the
    Box-Muller guard); `chiSquareFloats(key, n, dof)` = the DEFINITION,
    sum of `dof` squared normals over one `normalFloats(n·dof)` stream in
    contiguous row-major blocks with pinned f32 left-to-right
    accumulation — definitional rather than quantile because the χ²
    inverse CDF has no elementary form, and the sum-of-squares IS the
    hand-checkable oracle for any integer dof. Tensor wrappers:
    `cauchy/exponential/chiSquare` × `Vector/Matrix`.
  - `grad {}` surface: the cauchy spellings ride the SAME §0.4.421
    literal-key FIR arm and lower COMPOSITIONALLY to
    RNG_UNIFORM → SUB ½ → MUL π → TAN (splat consts are compile-time
    literals, not dim-derived; dims literal by the arm's own v1
    contract). REJECTED alternative: dedicated RNG_CAUCHY/RNG_CHISQ
    OpKinds — every new kind needs interpreter/VJP/JVP/emitter/cost arms
    for zero expressive gain, and the composition inherits all of them
    for free.
  - Certified: numpy-reference pins over the pinned (7,42) threefry
    stream at the Box-Muller 1e-5 convention (even + odd n — the odd
    lane rides the end-pad counter path) PLUS bit-exact
    composition-contract pins (the transform applied in-test to the
    pinned base stream); moment sanity at a fixed key (cauchy sample
    MEDIAN — the mean does not exist; exponential mean/var; χ²(4)
    mean/var); wrapper/determinism pins; E2E `grad {}` cert
    (RngGradientTest): ∇ Σ (c⊙x) hands back the draw BIT-EXACT against
    host `cauchyFloats`, no fallback warning.
  - **Named deferrals (recorded)**: (1) `chiSquare`/`exponential` inside
    `grad {}` — chiSquare needs a reshape-[n,dof]-then-axis-SUM synthesis
    in the FIR arm (the host block layout already matches it);
    exponential needs only SUB/LOG/NEG — both are the §0.4.431 cauchy
    pattern extended, waiting on a use. (2) **Gamma implicit
    reparameterization** (the DiffKT stretch): for `z ~ Gamma(α, 1)`,
    `∂z/∂α = −(∂F/∂α)(z; α) / p(z; α)` by the implicit-function theorem
    on the CDF `F(z; α) = P(α, z)` (the regularized lower incomplete
    gamma), where `p(z; α) = z^{α−1} e^{−z} / Γ(α)` and `∂P/∂α` needs the
    α-derivative of the incomplete gamma — computable from C1's
    lgamma/digamma (landed §0.4.402/405) via the standard series/continued
    -fraction split (Boost's `gamma_p_derivative` route) or a small
    quadrature; the SAMPLER itself (Marsaglia–Tsang squeeze over normal +
    uniform streams) is rejection-based, so the draw is NOT a fixed
    elementwise transform of a pinned stream — landing it honestly needs
    the sampler + the derivative + a CDF-oracle certification lane, a
    §-sized slice of its own. (3) `DiffktRandom`/`permitReuse` wrapper
    sugar (inherited from D1's ledger).

### Phase E — sparse (DiffKT `SparseFloatTensor` parity)

- **RATIFIED (Pedro, 2026-09-20): E1a→E1c GO** per the audit's recommended
  answers at every decision point — `matdiv` SKIPPED, GPU = pinned emit
  refusal for the E1b ops, row-sparse embedding gradients deferred to
  Phase F.
- **E1a ✅ DONE (§0.4.417) — the `:core` host CSR type.** `SparseTensor`
  (rank-2 CSR: `values` F32 / `colIdx` I32 / `rowPtr` I32, canonical
  strictly-increasing columns, loudly validated invariant), `fromCoo`
  (sorted + duplicate-SUMMING construction, the scatter-add convention),
  `toDense`, `nnz`; elementwise `plus`/`minus` (two-pointer UNION merge —
  cancelled positions stay stored) and `times` (INTERSECTION merge),
  sparse×dense elementwise `times` (the DiffKT zip case — structure
  reused verbatim), `transpose` as an explicit counting-sort half-perm
  (O(nnz + rows + cols), canonical output, involution pinned at the
  representation level), SpMM (sparse `[N,C]` × dense `[C,D]` → dense,
  per-row Double-accumulator buffer, the house reduction convention) and
  SpGEMM (Gustavson row-wise accumulation into a Double row buffer with a
  touched-column list — O(flops) + per-row sort, O(P) scratch, never an
  O(rows·P) clear). Certified against dense references on seeded random
  patterns at densities 0/0.05/0.3/0.7 (all-zero matrices and empty rows
  included) on the quarter-integer grid for EXACT `assertContentEquals`
  comparison; duplicate-summing, explicit-zero, union/intersection
  structure, transpose-involution and validation-refusal pins. Host-level
  only by design — no IR, no AD participation at this layer (that
  arrived with E1b/E1c below).
  Landed along the way: a kotlinc codegen landmine — a Companion
  `inline` function calling an outer-class `private` method emits a bad
  `invokespecial` (VerifyError at class load); keep such helpers
  self-contained (the VarHandle bug's family).
- **E1b ✅ DONE (§0.4.418) — `SPARSE_MATMUL` at IR level.**
  `OpKind.SPARSE_MATMUL` + fused `SPARSE_MATMUL_VALUES_ADJOINT` (the
  SDDMM stays fused to the pattern), `SparseMatmulRule` + bilinear
  tangent, interpreter arms bit-exact against E1a, host twins, CostModel,
  and the ratified pinned emit refusal. Certified against the dense
  `MatmulRule` on `toDense`'d operands, JVP⇄VJP, fwd-over-rev HVP vs the
  dense twin. Mechanism detail in
  [SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md)'s status header + §2.
- **E1c ✅ DONE (§0.4.419 pre + §0.4.420) — the `grad {}` sparse
  surface; the ratified arc is COMPLETE.** §0.4.419: `OpKind.ZEROS_LIKE`
  param-addressed structural zeros lift the §0.4.400 one-integer-param
  synthesis gate. §0.4.420: `sparseMatmul(values, colIdx, rowPtr, dense)`
  FIR front-end + synthesis arms for all three op spellings, certified
  E2E through the K2 plugin on the GNN shape (four-param lambda, empty +
  skewed rows, explicit stored zero, exact quarter-grid oracle). The arc
  STOPS here by ratified scope — matdiv skipped, GPU the pinned refusal,
  row-sparse embedding grads with Phase F.
- **E1. Audit ✅ DONE (§0.4.410).** Full
  audit in [SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md), from a fresh
  shallow clone @ HEAD. Findings: DiffKT sparse is a CPU-only Eigen JNI
  shim (hierarchical-CSR `SparseFloatTensor` + row-sparse
  `SparseRowFloatTensor` for embedding gradients), **primal-only** — its
  ops `require(derivativeId == NoDerivativeID)`, gradients w.r.t. sparse
  inputs come out DENSE, no sparse VJP exists anywhere — and partly broken
  in DiffKT itself (`nonZeroIndices` is `TODO()` for every Eigen-produced
  tensor, so chained sparse expressions throw). `matdiv` = SparseLU via
  explicit inverse, sparse-only. **Recommendation: conditional no-go**
  ("❌ by decision, audit on file") unless a real GNN/embedding workload
  pulls it in — then the audit's E1a→E1c slicing applies (host CSR type
  1 §; `SPARSE_MATMUL` + fused SDDMM values-adjoint 1–2 §; `grad {}`
  surface gated on generalizing the §0.4.400 one-integer-param synthesis
  restriction, ~2 §; GPU = pinned emit refusal, the §0.4.408 RNG
  precedent). Decision points for Pedro: GNN workload reality, `matdiv`
  skip, emit-refusal acceptability, row-sparse embedding grads deferred
  to Phase F — see the audit doc's §4.

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
| `reverseDerivative` / `primalAndReverseDerivative` (1/2-arg, List, n-th `reverseDerivative{1..4}`, `reverseDiff`) | ✅/🟡 | `grad {}` covers 1st-order at arities 1–3 (`grad3`/`valueAndGrad3` §0.4.424; synthesis is arity-agnostic, proven to 4 by the sparse surface §0.4.420); n-th-order nesting certified at IR level (§0.4.401), intrinsic spellings still open |
| `forwardDerivative` (all arities, n-th, `forwardDiff`) / `primalAndForwardDerivative` | ✅ | `jvp {}` / `valueAndJvp {}` §0.4.372 (B1) + `jvp2`/`valueAndJvp2` §0.4.387; loop-bearing bodies §0.4.403, IF bodies §0.4.407 (B3) |
| `jvp` / `primalAndJvp` | ✅ | same — §0.4.372/387 |
| `vjp` / `primalAndVjp` / `primalAndPullback` (user-supplied cotangent, `vf(primal)` form) | ✅ | `vjp {}` / `valueAndVjp {}` §0.4.398 — seeded single-pass pullback, tensor-valued `f` |
| Jacobian assembly | ✅ | `jacobian`/`hessian` §0.4.394 (B2) + `jacobian2`/`hessian2` §0.4.406 + reverse-assembled (tall, m ≪ n) `jacobianReverse` §0.4.412 + `jacobianReverse2` §0.4.424. DiffKT has **no** jacobian intrinsic — theirs is `reverseDerivative`'s identity-seeding loop, which is exactly `jacobianReverse`'s output-basis loop; the forward spellings assemble seeded forward passes at runtime |
| `reverseDerivativeTransposed` | ❌ | transposed-Jacobian convention variant; fold into B2 |
| Arbitrary nesting (fwd∘fwd, rev∘rev, …) | ✅/🟡 | full matrix certified at IR level + refusals pinned (§0.4.401); user-facing n-th-order intrinsic spellings still open |
| `ifThenElse(cond, a, b)` (scalar + tensor, differentiable) | ✅ | `where` §0.4.364; scalar branches also via IF regions + coarsening |
| `Wrappable`/`Wrapper` (derivatives through user data structures; examples lean on this) | 🟡 | Tlaloc's K2 plugin lowers data-class params structurally — different mechanism, same end; certify in B5 |
| `integral(a, b, f)` — Romberg quadrature with FTC-wired fwd/rev derivatives | ✅ | C5 §0.4.411 — `:core` host Romberg + `integralWithBoundGrads` FTC triple; `grad {}` surface v1 §0.4.426 (customVjp route + `integralWithParamGrad` Leibniz sugar; INTEGRAL region op is the named deferral) |
| `primal(x, f)`, `basePrimal`, `DerivativeID` plumbing | ➖ | runtime-tape bookkeeping; no analogue needed in a compile-time IR |

#### Scalar math (`DScalar` surface)

`+ - * / unaryMinus` ✅ · `abs sqrt exp ln sin cos relu` ✅ ·
`compareTo`/`eq ne lt le gt ge` ✅ (Kotlin comparisons + IF lower today) ·
`tanh sigmoid pow` ✅ A5b (§0.4.377 — the five-overload `:core` scalar host set
+ `UNARY_OP_MAP` entries; scalar SIGMOID synthesises via the new
`irCoreScalarCall`, scalar TANH via `kotlin.math.tanh`, scalar POW via the
`kotlin.math.pow` map entry) · `tan atan` ✅ C2 (§0.4.395 — five-overload
scalar sets + `kotlin.math.tan`/`atan` map entries with the no-receiver
argument fallback) ·
`lgamma digamma` ✅ C1 (§0.4.402 — five-overload scalar sets via the
`irCoreScalarCall` path; `trigamma` public but internal-only) ·
`polygamma(n)` ✅ C1 tail (§0.4.405 — five-overload set with the Int order;
`grad {}` orders 0..100, n = 0/1 normalised to DIGAMMA/TRIGAMMA) ·
`sigmoid(DScalar)` ✅ (same A5b).

#### Tensor ops (top-level files + `Operations` interface)

| DiffKT | Tlaloc | Notes |
|---|---|---|
| `plus minus times div unaryMinus` (elementwise) | ✅ | §0.4.364 tensor⊗tensor; A5a (§0.4.376) `Float×DTensor` on both operand orders; **A5c (§0.4.378/379) full implicit broadcasting** — NumPy right-alignment in the interpreter, the emitter, the host ops and the adjoints, so `[N,1] ⊙ [N,C]` and `[C] ⊙ [N,C]` differentiate. `DScalar×DTensor` ✅ A5c-3(iv) (§0.4.397): host overloads both orders + Float-param `grad {}` E2E; §0.4.414 closed the `FloatScalar`-param tail (boxed-scalar synthesis: `.toFloat()` unwrap on entry, value-class constructor on the gradient); §0.4.427 closed `DoubleScalar` (conversion-map members + cross-precision splat CAST + `CastRule`'s float adjoint) and pinned the `DScalar`-interface spelling as a NAMED tape-fallback refusal (structural: no concrete constructor at compile time) |
| `pow(Float/Int/DScalar/tensor-exponent)` | ✅ | A5b (§0.4.377): `:core/ops` host `pow` (tensor / Float / Int exponents) + FIR entries for `io.tlaloc.core.ops.pow` and `kotlin.math.pow` + a tensor synthesis arm; PowRule/interpreter/emitter/forward already shipped. `DScalar` exponent still open |
| `eq ne lt le gt ge` (tensor masks) | ✅ | §0.4.364 tensor⊗tensor; §0.4.397 Float-scalar rhs (`a gt 1.0f` + computed rank-0) E2E through `grad {}` |
| `relu reluGrad sigmoid tanh exp ln sqrt abs` (tensor) | ✅ | `reluGrad` is public in DiffKT; ours is internal — fine |
| `sin cos tan atan` (tensor) | ✅ | sin/cos ✅; tan/atan ✅ C2 (§0.4.395 — full vertical incl. `stablehlo.tan` / `atan2(x, 1)` emission certified on the GB10; audit: **no** floor/ceil/round/atan2 in DiffKT — those stay ours-optional) |
| `lgamma digamma polygamma` (tensor) | ✅ | C1 (§0.4.402 — lgamma/digamma full vertical incl. the first CHLO emissions, GB10-certified; trigamma internal for DIGAMMA's adjoint) + C1 tail (§0.4.405 — general polygamma(n) full vertical, family closed under differentiation) |
| `sum()` full-reduce | ✅ | |
| `sum(axes, keepDims)` | ✅ | A1 (§0.4.366) — `sum(dims, keepDims)`/`mean(dims)`/`max(dims)`/`min(dims)` E2E through `grad {}` (1–2 axes from `grad {}`; IR fully general) |
| `mean()` | ✅ | A1 (§0.4.366) — map entry + MEAN interpreter arm + sentinel-safe MeanRule |
| `FloatTensor.max/min(axes)` | ➖/🟡 | DiffKT only has these on **FloatTensor — not differentiable**; Tlaloc's MAX/MIN have VJPs → A1 exceeds parity |
| `stats()` = (mean, variance) | ✅ | §0.4.397 host sugar, biased variance (÷N); host-level only — DiffKT's `stats` is a convenience, and a Pair-returning body has no `grad {}` lowering (loss contract is scalar) |
| `matmul` (incl. generalized shape-block form) | ✅ | any rank ≥ 2 |
| `innerProduct` | ✅ | DOT |
| `outerProduct` | ✅ | §0.4.369: host + FIR (matmul on unsqueezed) + IR-level grad; §0.4.375: grad{} E2E (A4b — `Lit<Int>` placeholder atom for reshape-created unit axes types MatmulRule's transpose forward + squeeze backward) |
| `matdiv` | ➖ | **sparse-only** in DiffKT (dense explicitly unsupported) → E |
| `conv2d(hStride, vStride, Same/Valid/Explicit padding)` | ✅ | §0.4.362 **exceeds**: DiffKT has no groups/dilation, NHWC only; grouped/depthwise at IR level §0.4.429 |
| `maxPool / avgPool / maxPoolWithIndices` | ✅ | §0.4.363 **exceeds**: DiffKT pooling is non-overlapping only (stride=window, divisibility required, no padding) → C4 reclassified beyond-parity |
| `batchNorm` (raw op, training-stats variant) | ✅ | §0.4.390 — training form differentiates in `grad {}` by FIR DESUGARING onto fully-ruled ops (the BATCHNORM OpKind stays the Layer-3 inference form) |
| `softmax(axis) / logSoftmax / logSoftmaxGrad` | ✅ | A3a (§0.4.368) — `softmax(axis)` + `logSoftmax(axis)` E2E through `grad {}` (LOGSUMEXP stays emitter-only) |
| `crossEntropyLoss / crossEntropyLossFromOneHot / nllLossFromOneHot` | ✅ | §0.4.370: `crossEntropyLoss`/`nllLoss` composed in FIR from logSoftmax, E2E through `grad {}` (CROSS_ENTROPY OpKind stays emitter-only) |
| `embedding(table, indices, paddingIndex)` | ✅ | §0.4.370 IR-level (EmbeddingRule + EMBEDDING_GRAD + interpreter + forward tangent) → §0.4.400 E2E through `grad {}` (host op + FIR arm + I32-index-param synthesis + scatter+add emission, GPU-smoked) → §0.4.409 `paddingIndex` (exact-zero rows + zero gradient, mask emission) and rank-2 `[B, N]` index batches E2E; indices are params only (no in-lambda index arithmetic), one index param per lambda |
| `reshape / flatten(startDim) / squeeze / unsqueeze / expand / broadcastTo` | ✅/🟡 | reshape/squeeze/unsqueeze/flatten/transpose ✅ A2a (§0.4.367); `broadcastTo`/`expand` rank-increasing ✅ A2b (§0.4.371) + in-place size-1 stretch ✅ A2b (§0.4.373, runtime-extent `SUM_TO` adjoint) + 2nd-order-through-broadcast ✅ (§0.4.399, `BROADCAST_LIKE`) — mixed rank-increase+stretch still deferred |
| `transpose(axes) / leftTranspose / rightTranspose` | ✅ | `transpose(vararg perm)` + no-arg rank-2 spelling E2E ✅ A2a (§0.4.367); left/right sugar spellings unlanded (trivial when wanted) |
| `concat / stack / split / meld` | ✅ | `concat`/`stack` E2E ✅ §0.4.381–382 (runtime-extent `SLICE_LIKE` adjoint; 2nd order via `PAD_LIKE` §0.4.404); `meld` E2E ✅ §0.4.428 (flatten + CONCAT fold); `split` host-level ✅ §0.4.428 — its `grad {}` spelling is a named deferral (no `List<DTensor>` value model in the lambda lowering) |
| `slice / view(index/range/axis) / withChange` (functional update) | ✅ | single-axis `slice(start,end,axis)` ✅ A2b (§0.4.374, runtime-extent `PAD_TO` adjoint); `view(index/range, axis)` + `withChange(index/range, axis, r)` E2E ✅ §0.4.428 (withChange = `x + PAD_TO(r − slice(x), x)`, the PAD_TO ⇄ SLICE_AT closure in a primal; rank ≤ 3, the `padToLikeRank` bound); multi-index `view(IntArray)` deferred by name (repeated `view(i, 0)` spells it) |
| `gather / scatter (axis, paddingIndex) / gatherAtIndices / scatterAtIndices` | 🟡 | Tlaloc GATHER/SCATTER are narrower (rank-1/scalar-index arms) — A2 needs the axis+list form |
| `flip(axes)` | ✅ | C3 (§0.4.396) — REVERSE op, self-adjoint + extent-free, E2E through `grad {}`; the conv-adjoint `window_reversal` cleanup stays open |
| `IntTensor / intTensorOf / Float64` | ✅ | I32/I64/F64 dtypes; DiffKT is F32-only + int tensors — Tlaloc exceeds on F64 |
| `Device (CPU/GPU per-tensor placement)` | ➖ | Tlaloc's backend story (PJRT/IREE) supersedes |
| tracing/JIT package | ➖ | the entire Tlaloc compile pipeline is the superset |

#### Random (`random/`)

`RandomKey` (SHA-512 counter-based) + `split(n)` + `permitReuse` +
`floats/uniform/gaussian` + distributions: `cauchy`, `chiSquare`,
`gamma` (with-rate/with-scale, **implicit reparameterization** — the
raw-op `gamma(alpha, randomKey)` participates in AD), `DiffktRandom`
wrapper, `Wrapper.wrapRandomKey`. §0.4.408: key + `split` + `foldIn` +
uniform/gaussian ✅ (D1, threefry-based host+interpreter; JAX-classic
layout, not SHA-512 — same statelessness contract); reparam ✅ (D2,
§0.4.413/421); `cauchy`/`chiSquare` (+ exponential, beyond parity) ✅ D3
(§0.4.431, elementwise transforms of the D1 streams — cauchy also inside
`grad {}` compositionally); gamma implicit reparam a NAMED deferral with
the worked formula (D3's ledger — the Marsaglia–Tsang sampler is
rejection-based, a §-slice of its own); `permitReuse` / `DiffktRandom`
wrapper sugar still ❌ (D1 ledger).

#### Sparse (`SparseFloatTensor` / `SparseRowFloatTensor`)

COO-ish float tensors, ops actually implemented: `plus minus times
transpose matmul matdiv` (matdiv = square-RHS solve, sparse-only).
All ❌ → Phase E, audit-scoped as suspected: narrow GNN-adjacency
surface, not a general sparse algebra. **§0.4.410 — the deep E1 audit
landed in [SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md)**: the surface
is also primal-only (sparse ops `require(NoDerivativeID)`; gradients come
out dense) and partly broken in DiffKT itself — see the Phase E entry.

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

### Phase F — model/optimizer layer — **COMPLETE (§0.4.434–444, 2026-09-21)**

Ratified per [MODEL_LAYER_PLAN.md](MODEL_LAYER_PLAN.md), the Phase F
authority — and AMENDED there before any substrate code landed
(§0.4.436): **the AD route is the COMPILER stack, not the runtime tape**
— the model's forward traces once through the `:autograd` Tracer into a
real `DxirFunction` (arbitrary arity — the `grad {}` 1–4 ceiling was
never the IR's), `DxirReverseTransform` produces the gradient function,
and execution is a backend choice (interpreter on host, StableHLO →
`PjrtSession` on GPU). The value-tape stays a debugging fallback. The
`:nn` module of immutable functional components landed F0–F8 in eleven
§ (deep audit, substrate, layers, initializers, optimizers, conv stack,
BatchNorm/Dropout, Embedding/EmbeddingBag, GRU, and the F8 end-to-end
training certification with the compiled-GPU step and PyTorch
convergence parity); threefry-keyed randomness, functional
optimizer/batch-norm state, dense-v1 embedding gradients with the
row-sparse form a recorded design. The running record and the
consolidated Phase F deferral list live in that doc's §4.

## Suggested § sequencing

**Position at §0.4.433 (2026-09-20) — the book is CLOSED; this section
is kept as the historical sequencing record. The end-state and the one
consolidated tails list live at the top of this document. Position as
last swept at §0.4.428:** Phase 0 ✅ (§0.4.365) → Phase A ✅
in substance (§0.4.366–397, §0.4.400/409/414/427 — the scalar-param family
is closed: `Float`/`FloatScalar`/`DoubleScalar` lower, `DScalar`-interface
refuses by name to the tape; §0.4.428 — A2's `view`/`withChange`/`meld`
sugar E2E + host `split` (named deferrals: `split` in `grad {}`,
multi-index `view(IntArray)`, `withChange` beyond rank 3); remaining
tails: gather/scatter axis+list forms, the
mixed rank-increase+stretch broadcast) →
B1–B4 ✅ (§0.4.372/387/394/398/401/403/404/406/407) → B5 ✅ (§0.4.415
`customVjp`/`customVjp2` + §0.4.416 `customJvp`/`customVjpJvp` and their
2-arg forms, ratified 2026-09-20) → C1–C3 ✅
(§0.4.395/396/402/405) → C5 ✅ (§0.4.411 — host surface; §0.4.426 — the
`grad {}` surface v1 through the B5 gate, INTEGRAL region op deferred by
name) → D1 ✅
(§0.4.408) → D2 v1 ✅ (§0.4.413 — IR-level reparameterized gradients;
FIR/`grad {}` spelling is the recorded tail) → E1a ✅ (§0.4.417 — the
`:core` host CSR `SparseTensor`; E ratified by Pedro 2026-09-20 per the
audit's recommendations).

**Remaining, in recommended order:**
1. **Phase E sparse — COMPLETE (§0.4.417–420)** per
   [SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md)'s ratified slicing:
   E1a the `:core` host CSR `SparseTensor` (§0.4.417), E1b `SPARSE_MATMUL`
   + fused SDDMM values-adjoint with the pinned GPU emit refusal
   (§0.4.418), E1c-pre `ZEROS_LIKE` param-addressed structural zeros
   lifting the §0.4.400 one-integer-param synthesis gate (§0.4.419), E1c
   the `grad {}` sparse surface with the GNN-shaped E2E cert (§0.4.420).
   The arc STOPS here by ratified scope: matdiv SKIPPED, GPU = pinned
   refusal (ELL-padded emission is the recorded tail), row-sparse
   embedding gradients deferred to Phase F. F model layer remains a
   product decision. (The §0.4.419 en-passant finding — EMBEDDING_GRAD had
   no forward tangent, so fwd-over-rev hessians through embedding gradient
   bodies refused — CLOSED §0.4.423, which gave the whole fused-adjoint
   family tangent arms; see item 2a below.)
2. **D2 `grad {}` surface — DONE §0.4.421.** The existing host spellings
   (`RandomKey(k0, k1).normalVector<Sym>(n)` / uniform / matrix siblings)
   lower to the zero-operand RNG ops inside `grad {}` lambdas (FIR arm:
   receiver must be a DIRECT literal `RandomKey(k0, k1)` constructor call,
   dims Int literals — the recorded v1 contract; anything else falls back
   loudly, pinned), and the synthesis replays the literal attrs through
   new `io.tlaloc.core.ops` twins `rng{Uniform,Normal}{Vector,Matrix}` —
   the same `:core/Random.kt` kernels as host and interpreter, bit-exact
   (E2E: reparameterized loss with cloned same-stream ε, deterministic
   across calls; linear uniform pin EXACT).
   **Explicit-threefry StableHLO emission — DONE §0.4.422.** The §0.4.408
   emit refusal flips: RNG_UNIFORM/RNG_NORMAL emit the Threefry-2x32
   block as explicit integer ops (iota counters, 20 ARX rounds, key
   schedule folded at emit time from the literal attrs — JAX's own
   approach, never rng_bit_generator). GPU-certified on the GB10
   (PjrtRngSmokeTest): uniform draws BIT-EXACT against the host kernels
   on raw f32 bits (even/odd end-pad lane/rank-2, three keys); normal
   draws (Box-Muller in the host's own f64 intermediates) at tolerance —
   the bits layer is exact, backend libm log/cos is not a bit contract
   (measured 0.0 diff on the GB10 regardless); the reparameterized
   GRADIENT graph (containing a cloned draw) compiles and runs on GPU at
   0.0 vs the interpreter, and GradientEmissionCoverageTest's RNG
   exclusion is LIFTED.
   **Runtime-key operand form — DONE §0.4.432 (IR/interpreter/emitter;
   the FIR surface is the remaining tail).** The creation ops accept an
   ALTERNATIVE form: exactly two scalar-I32 operands carrying the key
   words, `key0`/`key1` attrs absent (the forms are exclusive, both-at-
   once and any other arity refused by name), `dims` still a literal
   attr in both forms — the shape must be static, the stream need not
   be. Interpreter reads keys at execution time (an Int-carrying const
   verbatim — exact for any 32-bit word; anything else through the F32
   value domain guarded STRICTLY to |key| < 2^24, since 2^24+1 rounds
   INTO the domain — beyond it a loud named refusal). Differentiation
   needed NO new arms: keys are integers, RngDrawRule's empty
   contribution list and the forward structural-zero tangent already
   ignore operands, integer key params take the §0.4.54 typed-zero
   gradient, and a cloned draw drags its key operand clones through the
   ordinary usedByAdjoint transitive walk — same key SSA values, same
   stream, pinned bit-exact in DxirRngTest. Emission: key splats
   broadcast from the rank-0 SSA values and the key schedule EMITS
   (ks2's two xors over the raw 0x1BD11BDA constant, the five injection
   adds) instead of folding; the ARX core is untouched and the attr
   form's MLIR is byte-identical to §0.4.422's. GPU-certified on the
   GB10 (PjrtRngSmokeTest): keys as EXECUTABLE INPUTS (f32 scalars CAST
   to i32 in-graph — XLA cannot fold the schedule) BIT-EXACT vs the
   host kernels including the odd end-pad lane; high-bit keys as I32
   const operands BIT-EXACT too. Remaining D tails, recorded: (a) the
   FIR surface — RandomKey-typed vals / lambda params / computed key
   words inside `grad {}` lowering to the operand form (the §0.4.421
   literal-only fallback stays the loud gate until then); (b) an i32
   host-buffer lane in PjrtSession (F32-only v1), which would let
   high-bit runtime keys ride as executable inputs rather than consts.

2a. **Fused-adjoint forward tangents — DONE §0.4.423.** The whole family
   joins forward mode: EMBEDDING_GRAD (linear in upstream), the four
   conv adjoints (bilinear in operands 0/1 — d f(a,b) = f(da,b) +
   f(a,db), shape-only template riding as its clone), AVGPOOL2D_GRAD
   (linear), MAXPOOL2D_GRAD (linear in upstream; the tie mask locally
   constant, the maxpool subgradient convention), SCATTER_ADD (linear in
   base and value). Second order now composes THROUGH gradient bodies of
   embedding/conv/pool/gather losses — hessians of those surfaces were
   unreachable before (the transform refused loudly). Certified
   (DxirFusedAdjointTangentTest): hand-exact HVPs for embedding
   (2·count⊙v), avgpool (window-sum/8), maxpool (2·v at each argmax,
   exact zeros elsewhere, shuffled-grid inputs), gather (2·v on the
   gathered row, exact zeros elsewhere); conv (both adjoints in one
   body) against central differences of the gradient AND the symmetry
   identity ⟨u,Hv⟩ = ⟨v,Hu⟩.
3. **Recorded tails on the books** (each its own §-sized slice when
   pulled): A-phase tails above; C4's grouped-conv §0.4.429 named
   deferrals (transposed-conv grouped VJP, grouped adjoint emission,
   the user surface). (B2's reverse-assembled tall Jacobians closed
   §0.4.412; grouped/depthwise conv landed at IR level §0.4.429; B3's
   multi-result COARSENED tangents + IF-inside-primal_body splice
   closed §0.4.430; D's cauchy/exponential/chiSquare distributions
   closed §0.4.431 — D3, gamma implicit reparam the remaining named
   deferral with the worked formula on file.)

Certification discipline per CLAUDE-memory: solo full-suite runs, count
gate updated per §, GPU smokes for anything touching the emitter.

The close-out (§0.4.433) supersedes the per-item deferral scatter above:
"Remaining tails, consolidated" at the top of this document is the one
authoritative list of what stays open, and the canonical end-of-book
suite number is 1944.
