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
      concretely detectable as size-1). **DEFERRED — 2nd-order through in-place
      broadcast**: SUM_TO has no VjpRule, so `forward(reverse(f))` (an HVP) through
      an in-place stretch errors loudly. Its reverse (broadcast the T-shaped
      upstream back up to `value`'s runtime shape U) needs a runtime-extent
      broadcast-to-template op (the mirror of SUM_TO — a `BROADCAST_LIKE(upstream,
      template=value)` reading U from `value`'s runtime dims); 1st-order is this
      slice.
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
        including "no VjpRule" (second-order through it errors, as it does for
        those two).
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
    - **`view`/indexing, `withChange`, `meld`/`split`, `stats`**: same
      runtime-extent boundary; sequenced after the mechanism above lands.
- **A3. NN ops in lambdas** — split by wiring readiness:
  - **A3a ✅ (§0.4.368)**: `softmax(axis)` + `logSoftmax(axis)` E2E through
    `grad {}`. SOFTMAX was fully wired below the surface (interpreter,
    emitter, VJP, JVP, cost) — only the host fn + FIR arm + `irSoftmax`
    synthesis were missing. `logSoftmax` lowers to `LOG(SOFTMAX(x))` (both
    fully-ruled ops; **LOGSUMEXP stays emitter-only** — no VJP/interp/JVP),
    which also forced tensor `irLog`/`irExp` (were scalar-only).
  - **A3b ✅ (partial, §0.4.370)** — the two self-contained halves landed:
    - **`embedding` VjpRule** ✅: EMBEDDING was wired below the surface
      (emitter-as-gather + cost model) but had no reverse rule and no
      interpreter arm. Added: an EMBEDDING interpreter arm (rank-2 table +
      int index tensor → gathered rows), a fused [OpKind.EMBEDDING_GRAD]
      adjoint op (interpreter arm: scatter-ADD each upstream row back to the
      vocab slot its index selected, collisions summing — result type `[V,D]`
      carries the dims, indices non-differentiable), the EmbeddingRule VjpRule
      (mirrors GatherRule's fused scatter-add), and the EMBEDDING forward-mode
      tangent (linear in the table). Certified IR-level: a hand-pinned
      collision-summing dTable + the JVP⇄VJP cross-identity. **NOT reachable
      from `grad {}`** (no FIR/synthesis arm) — IR-level only, like the pre-A3a
      softmax state; EMBEDDING_GRAD's StableHLO scatter+add-region emission is
      also deferred (interpreter arm suffices for the IR-level cert).
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
  - **A5c-3 remainder (pending).**
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
    (iv) `DScalar × DTensor` mixing, and comparisons against a scalar literal
    (`a gt 1.0f` — `COMPARE_DIRECTION_MAP` still lowers both sides verbatim, and
    the comparison host ops still use the strict `elementwise`).

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
  single argument, straight-line bodies. Certified E2E through the K2
  plugin (dy = 2⟨x,dx⟩ for Σx², the (y,dy) pair, Σexp(x)·dx). Multi-arg
  `jvp2`/`valueAndJvp2` (params `(x1,x2,dx1,dx2)`) is a clean follow-up.
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
`tanh sigmoid pow` ✅ A5b (§0.4.377 — the five-overload `:core` scalar host set
+ `UNARY_OP_MAP` entries; scalar SIGMOID synthesises via the new
`irCoreScalarCall`, scalar TANH via `kotlin.math.tanh`, scalar POW via the
`kotlin.math.pow` map entry) · `tan atan` ❌ (C2) ·
`lgamma digamma polygamma` ❌ (C1) · `sigmoid(DScalar)` ✅ (same A5b).

#### Tensor ops (top-level files + `Operations` interface)

| DiffKT | Tlaloc | Notes |
|---|---|---|
| `plus minus times div unaryMinus` (elementwise) | ✅ | §0.4.364 tensor⊗tensor; A5a (§0.4.376) `Float×DTensor` on both operand orders; **A5c (§0.4.378/379) full implicit broadcasting** — NumPy right-alignment in the interpreter, the emitter, the host ops and the adjoints, so `[N,1] ⊙ [N,C]` and `[C] ⊙ [N,C]` differentiate. `DScalar×DTensor` still open (A5c-3) |
| `pow(Float/Int/DScalar/tensor-exponent)` | ✅ | A5b (§0.4.377): `:core/ops` host `pow` (tensor / Float / Int exponents) + FIR entries for `io.tlaloc.core.ops.pow` and `kotlin.math.pow` + a tensor synthesis arm; PowRule/interpreter/emitter/forward already shipped. `DScalar` exponent still open |
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
| `outerProduct` | ✅ | §0.4.369: host + FIR (matmul on unsqueezed) + IR-level grad; §0.4.375: grad{} E2E (A4b — `Lit<Int>` placeholder atom for reshape-created unit axes types MatmulRule's transpose forward + squeeze backward) |
| `matdiv` | ➖ | **sparse-only** in DiffKT (dense explicitly unsupported) → E |
| `conv2d(hStride, vStride, Same/Valid/Explicit padding)` | ✅ | §0.4.362 **exceeds**: DiffKT has no groups/dilation, NHWC only |
| `maxPool / avgPool / maxPoolWithIndices` | ✅ | §0.4.363 **exceeds**: DiffKT pooling is non-overlapping only (stride=window, divisibility required, no padding) → C4 reclassified beyond-parity |
| `batchNorm` (raw op, training-stats variant) | 🟡 | BATCHNORM OpKind exists; VJP + surface unaudited — fold into A3 |
| `softmax(axis) / logSoftmax / logSoftmaxGrad` | 🟡 | SOFTMAX/LOGSUMEXP + VJPs exist → A3 |
| `crossEntropyLoss / crossEntropyLossFromOneHot / nllLossFromOneHot` | ✅ | §0.4.370: `crossEntropyLoss`/`nllLoss` composed in FIR from logSoftmax, E2E through `grad {}` (CROSS_ENTROPY OpKind stays emitter-only) |
| `embedding(table, indices, paddingIndex)` | 🟡 | §0.4.370: EmbeddingRule VjpRule + EMBEDDING_GRAD adjoint + interpreter + forward tangent, **IR-level only** (no `grad {}` FIR/synthesis arm yet); `paddingIndex` not modelled |
| `reshape / flatten(startDim) / squeeze / unsqueeze / expand / broadcastTo` | 🟡 | reshape/squeeze/unsqueeze/flatten/transpose ✅ A2a (§0.4.367); `broadcastTo`/`expand` rank-increasing ✅ A2b (§0.4.371) + in-place size-1 stretch ✅ A2b (§0.4.373, runtime-extent `SUM_TO` adjoint) — mixed rank-increase+stretch + 2nd-order-through-broadcast deferred |
| `transpose(axes) / leftTranspose / rightTranspose` | 🟡 | TRANSPOSE + VJP → A2 (left/right = sugar) |
| `concat / stack / split / meld` | 🟡 | CONCAT/SPLIT + VJPs → A2 (`meld` = flatten-and-concat sugar; inverse `split`) |
| `slice / view(index/range/axis) / withChange` (functional update) | 🟡 | single-axis `slice(start,end,axis)` ✅ A2b (§0.4.374, runtime-extent `PAD_TO` adjoint), E2E through `grad {}`; multi-axis `view`/`withChange` scatter sugar still A2 |
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
