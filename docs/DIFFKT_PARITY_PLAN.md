# DiffKT Parity Plan

**Status: ACTIVE (opened 2026-07-19, post-§0.4.364).** Goal per Pedro:
support everything [facebookresearch/diffkt](https://github.com/facebookresearch/diffkt)
supports that Tlaloc doesn't yet.

## Where parity already stands (closed §0.4.359–364)

| DiffKT capability | Tlaloc status |
|---|---|
| Reverse-mode AD (vjp/pullback) | ✅ `DxirReverseTransform` + runtime synthesis + compile-time probe |
| Forward-mode AD (jvp/pushforward) | ✅ `DxirForwardTransform` (§0.4.361) — **IR-level only, no user intrinsic yet** |
| Higher-order (hessian-vector) | ✅ full nesting matrix certified at IR level (§0.4.401): fwd∘rev, fwd∘fwd, rev∘fwd, rev∘rev + a third-order spot check; fused-adjoint refusals pinned |
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
        their rules, and SLICE_LIKE is now the one member still without one
        (its window offset is a runtime SUM of prior templates' extents, which
        no literal-offset adjoint can carry — second-order through a concat
        window still errors loudly).
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
      - Host twins: `sliceLikeStart` / `sliceLikeAfter{1,2,3}` — fixed-arity per
        PRIOR count, because synthesis builds positional `IrCall` arguments and
        cannot build an `IrVararg` (the documented reason for the whole `…RankN`
        shim family). Bounded at 4 concat operands; a wider concat is slice 2's
        concern.
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
    - **`view`/indexing, `withChange`, `meld`/`split`**: same
      runtime-extent boundary; sequenced after the mechanism above lands.
      (`stats` left this list — landed host-level in §0.4.397, A5c-3(iv).)
- **A3. NN ops in lambdas** — split by wiring readiness:
  - **A3a ✅ (§0.4.368)**: `softmax(axis)` + `logSoftmax(axis)` E2E through
    `grad {}`. SOFTMAX was fully wired below the surface (interpreter,
    emitter, VJP, JVP, cost) — only the host fn + FIR arm + `irSoftmax`
    synthesis were missing. `logSoftmax` lowers to `LOG(SOFTMAX(x))` (both
    fully-ruled ops; **LOGSUMEXP stays emitter-only** — no VJP/interp/JVP),
    which also forced tensor `irLog`/`irExp` (were scalar-only).
  - **A3b ✅ (§0.4.370 IR-level; §0.4.400 embedding E2E)** — the two
    self-contained halves landed, then embedding got its front-end:
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
    - **`crossEntropyLoss`/`nllLoss`** ✅ E2E through `grad {}`: composed in
      FIR onto existing fully-ruled ops (no new VjpRule). `crossEntropyLoss` =
      `NEG(SUM(MUL(oneHot, LOG(SOFTMAX(logits, -1)))))` (sum-reduction
      convention — total of the per-sample cross-entropies), `nllLoss` skips
      the softmax. `:core` host twins added (`crossEntropyLoss`/`nllLoss`,
      both returning a scalar so the body ends in `.toFloat()`). Certified
      E2E: both gradients synthesise with no fallback and match the analytic
      references (CE: da = softmax·Σb − b, db = −logSoftmax; NLL: da = −b,
      db = −a).
  - **A3b (deferred) — `conv2d`/`maxPool`/`avgPool` in `grad {}`**: this is a
    multi-§ architectural effort, NOT a clean scope widen. Blockers:
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
       Still deferred: the fused ops have neither a VjpRule nor a forward
       tangent — as with `EMBEDDING_GRAD`, differentiating through a gradient body
       that contains them fails loudly rather than silently. (CONV_TRANSPOSE2D's
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
    - **DEFERRED tail — `FloatScalar`-typed params**: the FIR lowers
      `grad { a, s: FloatScalar -> a * s }` to the IDENTICAL dxir (pinned), but
      synthesis cannot BOX the rank-0 gradient back into the returned pair's
      `FloatScalar` slot, so the type guard keeps the original call (the
      long-documented "DScalar boxing" fallback in
      `TlalocIrGenerationExtension`). Closing it means synthesis-side
      `FloatScalar(x)` construction + `.v` unwrap on entry for scalar-class
      params generally (it predates this slice: an ALL-FloatScalar `grad {}`
      falls back the same way). `DScalarMixingGradientTest` pins the fallback
      and says exactly how to flip the pin when boxing lands.

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
  - v1 scope: single-argument `f`, straight-line bodies, host F32. Still
    open in B2's neighbourhood: reverse-assembled (tall) Jacobians for
    m ≪ n, and multi-arg `jacobian2`/`hessian2`. (The seeded-cotangent
    user surface closed in §0.4.398 below.)
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
    region work), multi-arg `vjp2`.
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
  - **Still ruleless, deliberately**: SLICE_LIKE — its window offset is a
    runtime SUM of prior templates' extents along the axis, which no
    literal-`low` adjoint can express; second-order through a concat window
    still errors loudly. (A PAD_LIKE with prior-template offsets is the shape
    of the fix, sequenced when B4 demands it.)
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
- **B3. Forward transform through regions**: IF/WHILE bodies + COARSENED
  (tangent of a coarsened op = forward transform of its `primal_body`) —
  mirrors reverse-mode's history.
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
    bodies carrying MAXPOOL2D_GRAD, EMBEDDING_GRAD, or the symbolic-concat
    SLICE_LIKE fail with "no VJP rule registered for <op>", asserted by
    name. The fused conv/pool/embedding adjoints stay VjpRule-less by
    design (their second derivative would need the adjoint-of-adjoint
    expansion B5/C-era work can decide on); SLICE_LIKE keeps §0.4.399's
    recorded PAD_LIKE fix shape.
  - **Third order** ✅ for free: F(F(R(Σx⁴)))(x, u, v, 0) = 24·x⊙u⊙v pinned.
  - Deferred tails: nesting through region-bearing bodies rides on B3
    (forward-v1 refuses regions before any composition question arises);
    user-facing n-th-order intrinsics (`reverseDerivative{2..4}` spellings)
    are a synthesis-surface question, not an IR one — the IR compositions
    they'd lower to are what this slice certified.
- **B5. User-defined custom derivatives**: a user-facing custom-VJP/JVP
  registration (DiffKT lets users supply derivatives for opaque functions;
  our coarsener `gradient_body` machinery is the internal analogue —
  surface it).

### Phase C — op families DiffKT has that the IR lacks

- ✅ **C1. Special functions — DONE (§0.4.402)**: `LGAMMA` and `DIGAMMA`,
  tensor and scalar, full vertical; `TRIGAMMA` (= polygamma(1)) landed as an
  INTERNAL op — DIGAMMA's adjoint/tangent emit it — with general
  `POLYGAMMA(n)` deliberately out of scope (see deferred tail).
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
  - Deferred tail: general `POLYGAMMA(n)` (needs a polygamma(n) host kernel
    family + `d polygamma(n) = polygamma(n+1)`; DiffKT's own examples use
    only ψ and ψ₁, so parity pressure is low), and with it TRIGAMMA's own
    VjpRule (second-order reverse through DIGAMMA — today it refuses loudly,
    pinned in DxirLgammaDigammaGradTest).
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
| `reverseDerivative` / `primalAndReverseDerivative` (1/2-arg, List, n-th `reverseDerivative{1..4}`, `reverseDiff`) | ✅/🟡 | `grad {}` covers 1st-order; n-th-order nesting certified at IR level (§0.4.401), intrinsic spellings still open |
| `forwardDerivative` (all arities, n-th, `forwardDiff`) / `primalAndForwardDerivative` | 🟡 | `DxirForwardTransform` §0.4.361; no user intrinsic → B1 |
| `jvp` / `primalAndJvp` | 🟡 | same → B1 |
| `vjp` / `primalAndVjp` / `primalAndPullback` (user-supplied cotangent, `vf(primal)` form) | ✅ | `vjp {}` / `valueAndVjp {}` §0.4.398 — seeded single-pass pullback, tensor-valued `f` |
| Jacobian assembly | 🟡 | DiffKT has **no** jacobian intrinsic — `reverseDerivative(x, f: tensor→tensor)` identity-seeds and builds the full Jacobian (`identityGradientOfSameKind`). B2 = that seeding loop |
| `reverseDerivativeTransposed` | ❌ | transposed-Jacobian convention variant; fold into B2 |
| Arbitrary nesting (fwd∘fwd, rev∘rev, …) | ✅/🟡 | full matrix certified at IR level + refusals pinned (§0.4.401); user-facing n-th-order intrinsic spellings still open |
| `ifThenElse(cond, a, b)` (scalar + tensor, differentiable) | ✅ | `where` §0.4.364; scalar branches also via IF regions + coarsening |
| `Wrappable`/`Wrapper` (derivatives through user data structures; examples lean on this) | 🟡 | Tlaloc's K2 plugin lowers data-class params structurally — different mechanism, same end; certify in B5 |
| `integral(a, b, f)` — Romberg quadrature with FTC-wired fwd/rev derivatives | ❌ | genuinely novel; **new C5** |
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
`irCoreScalarCall` path; `trigamma` public but internal-only, general
`polygamma(n)` deferred) · `sigmoid(DScalar)` ✅ (same A5b).

#### Tensor ops (top-level files + `Operations` interface)

| DiffKT | Tlaloc | Notes |
|---|---|---|
| `plus minus times div unaryMinus` (elementwise) | ✅ | §0.4.364 tensor⊗tensor; A5a (§0.4.376) `Float×DTensor` on both operand orders; **A5c (§0.4.378/379) full implicit broadcasting** — NumPy right-alignment in the interpreter, the emitter, the host ops and the adjoints, so `[N,1] ⊙ [N,C]` and `[C] ⊙ [N,C]` differentiate. `DScalar×DTensor` ✅ A5c-3(iv) (§0.4.397): host overloads both orders + Float-param `grad {}` E2E; `FloatScalar`-typed params still fall to the tape (synthesis DScalar boxing — deferred tail) |
| `pow(Float/Int/DScalar/tensor-exponent)` | ✅ | A5b (§0.4.377): `:core/ops` host `pow` (tensor / Float / Int exponents) + FIR entries for `io.tlaloc.core.ops.pow` and `kotlin.math.pow` + a tensor synthesis arm; PowRule/interpreter/emitter/forward already shipped. `DScalar` exponent still open |
| `eq ne lt le gt ge` (tensor masks) | ✅ | §0.4.364 tensor⊗tensor; §0.4.397 Float-scalar rhs (`a gt 1.0f` + computed rank-0) E2E through `grad {}` |
| `relu reluGrad sigmoid tanh exp ln sqrt abs` (tensor) | ✅ | `reluGrad` is public in DiffKT; ours is internal — fine |
| `sin cos tan atan` (tensor) | ✅ | sin/cos ✅; tan/atan ✅ C2 (§0.4.395 — full vertical incl. `stablehlo.tan` / `atan2(x, 1)` emission certified on the GB10; audit: **no** floor/ceil/round/atan2 in DiffKT — those stay ours-optional) |
| `lgamma digamma polygamma` (tensor) | ✅ | C1 (§0.4.402 — lgamma/digamma full vertical incl. the first CHLO emissions, GB10-certified; trigamma internal for DIGAMMA's adjoint; general polygamma(n) deferred) |
| `sum()` full-reduce | ✅ | |
| `sum(axes, keepDims)` | 🟡 | `reduction_dims` IR exists → A1 |
| `mean()` | 🟡 | dispatch arm exists but **no map entry** — not reachable → A1 |
| `FloatTensor.max/min(axes)` | ➖/🟡 | DiffKT only has these on **FloatTensor — not differentiable**; Tlaloc's MAX/MIN have VJPs → A1 exceeds parity |
| `stats()` = (mean, variance) | ✅ | §0.4.397 host sugar, biased variance (÷N); host-level only — DiffKT's `stats` is a convenience, and a Pair-returning body has no `grad {}` lowering (loss contract is scalar) |
| `matmul` (incl. generalized shape-block form) | ✅ | any rank ≥ 2 |
| `innerProduct` | ✅ | DOT |
| `outerProduct` | ✅ | §0.4.369: host + FIR (matmul on unsqueezed) + IR-level grad; §0.4.375: grad{} E2E (A4b — `Lit<Int>` placeholder atom for reshape-created unit axes types MatmulRule's transpose forward + squeeze backward) |
| `matdiv` | ➖ | **sparse-only** in DiffKT (dense explicitly unsupported) → E |
| `conv2d(hStride, vStride, Same/Valid/Explicit padding)` | ✅ | §0.4.362 **exceeds**: DiffKT has no groups/dilation, NHWC only |
| `maxPool / avgPool / maxPoolWithIndices` | ✅ | §0.4.363 **exceeds**: DiffKT pooling is non-overlapping only (stride=window, divisibility required, no padding) → C4 reclassified beyond-parity |
| `batchNorm` (raw op, training-stats variant) | 🟡 | BATCHNORM OpKind exists; VJP + surface unaudited — fold into A3 |
| `softmax(axis) / logSoftmax / logSoftmaxGrad` | 🟡 | SOFTMAX/LOGSUMEXP + VJPs exist → A3 |
| `crossEntropyLoss / crossEntropyLossFromOneHot / nllLossFromOneHot` | ✅ | §0.4.370: `crossEntropyLoss`/`nllLoss` composed in FIR from logSoftmax, E2E through `grad {}` (CROSS_ENTROPY OpKind stays emitter-only) |
| `embedding(table, indices, paddingIndex)` | ✅ | §0.4.370 IR-level (EmbeddingRule + EMBEDDING_GRAD + interpreter + forward tangent) → §0.4.400 E2E through `grad {}` (host op + FIR arm + I32-index-param synthesis + scatter+add emission, GPU-smoked); `paddingIndex` not modelled, indices rank-1 params only |
| `reshape / flatten(startDim) / squeeze / unsqueeze / expand / broadcastTo` | 🟡 | reshape/squeeze/unsqueeze/flatten/transpose ✅ A2a (§0.4.367); `broadcastTo`/`expand` rank-increasing ✅ A2b (§0.4.371) + in-place size-1 stretch ✅ A2b (§0.4.373, runtime-extent `SUM_TO` adjoint) + 2nd-order-through-broadcast ✅ (§0.4.399, `BROADCAST_LIKE`) — mixed rank-increase+stretch still deferred |
| `transpose(axes) / leftTranspose / rightTranspose` | 🟡 | TRANSPOSE + VJP → A2 (left/right = sugar) |
| `concat / stack / split / meld` | 🟡 | CONCAT/SPLIT + VJPs → A2 (`meld` = flatten-and-concat sugar; inverse `split`) |
| `slice / view(index/range/axis) / withChange` (functional update) | 🟡 | single-axis `slice(start,end,axis)` ✅ A2b (§0.4.374, runtime-extent `PAD_TO` adjoint), E2E through `grad {}`; multi-axis `view`/`withChange` scatter sugar still A2 |
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
wrapper, `Wrapper.wrapRandomKey`. All ❌ → Phase D (D1 key+uniform+
gaussian; D2 reparam; cauchy/chiSquare are inverse-CDF sugar over
uniform; gamma reparam needs C1's digamma — landed, §0.4.402).

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
