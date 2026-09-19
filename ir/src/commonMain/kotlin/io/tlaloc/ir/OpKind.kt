package io.tlaloc.ir

enum class OpKind {
    // Elementwise unary
    //
    // STEP: Heaviside step. Returns 1 where input > 0, 0 elsewhere. At x = 0 the
    // output is 0 — matching XLA's `stablehlo.compare GT` + `select 1 0` lowering
    // exactly, not the mathematical `H(0) = 1/2` convention. ReLU's adjoint
    // (§0.4.7) uses STEP as its mask; wiring it as a first-class op keeps the
    // registry + tape + stablehlo lowerings in agreement on that x=0 behaviour.
    NEG, ABS, EXP, LOG, SQRT, RSQRT, TANH, SIGMOID, RELU, GELU, SILU, STEP,

    // §0.4.166 — Trigonometric primitives. Added for the CartPole port (per
    // docs/CARTPOLE_PORT_PLAN.md), whose physics step uses `sin(θ)` and `cos(θ)`
    // for the pole's angular state. Mutual gradients: `d/dx sin = cos`,
    // `d/dx cos = -sin`. Lowered to `stablehlo.sine` / `stablehlo.cosine`.
    SIN, COS,

    // §0.4.204 — Elementwise sign function. Returns 1 / -1 / 0 for x>0 / x<0 / x=0.
    // Added for the CartPole NN port: the policy output is `a = sign(tanh(...) - ε)`
    // which discretises the action to {-1, +1}. Gradient is identically 0 (the
    // function is non-differentiable at 0 and constant elsewhere); SignRule emits
    // a zero const at x's shape.
    SIGN,

    // Boolean negation (Stage B.1 prerequisite for F3 canonicalisation per
    // docs/STAGE_B_PLAN.md §4.4 — F3 swaps an IF's then/else branches and wraps the
    // predicate in NOT to produce a canonical branch order). Operand + result are both
    // Bool; same encoding as STEP (`0f` is false, `1f` is true). Single op kind keeps
    // the dxir surface honest: every negation flows through one node, and downstream
    // canonical-form detection (F1 collapse, CSE) sees one shape. Lowering to
    // StableHLO is `stablehlo.not` on a Bool tensor; deferred post-Stage-B alongside
    // IF/WHILE.
    NOT,

    // §0.4.50 — Bool × Bool → Bool logical-and. Companion to [NOT]; introduced to let
    // the FIR lowering hoist an `if (break_cond) break` at the tail of a while-body into
    // the WHILE's cond region as `LAND(original_cond, NOT(break_cond))`. Same encoding
    // as STEP/NOT (0f/1f). Lowering to StableHLO deferred alongside IF/WHILE.
    LAND,

    // Elementwise binary
    ADD, SUB, MUL, DIV, POW,

    // Reduction
    SUM, MEAN, MAX, MIN, ARGMAX, SOFTMAX, LOGSUMEXP,

    // Linear algebra
    MATMUL, DOT, CONV2D, CONV_TRANSPOSE2D,

    // §0.4.363 — 2-D window pooling (DiffKT-gap item 4, pooling half).
    // NCHW, attrs: `window` [kh, kw], `window_strides` [sh, sw], `padding`
    // [[top, bottom], [left, right]]. Lowered to `stablehlo.reduce_window`
    // (max with -inf init / add with 0 init; AVGPOOL2D then divides by the
    // FULL window size kh·kw, padding included — the count_include_pad
    // convention, documented at the interpreter arm).
    MAXPOOL2D, AVGPOOL2D,

    // §0.4.389 — MAXPOOL2D's adjoint, fused (Phase A3b).
    // MAXPOOL2D_GRAD(upstream, x, y) → dX at x's shape, attrs copied off the
    // primal (`window`, `window_strides`, `padding`). `y` is the pooled value
    // (MAXPOOL2D(x)), which the rule materialises as its own node so no engine
    // recomputes the window max and the emitter gets a ready SSA value.
    //
    // The spelling this replaces is nearest-upsample-and-mask:
    // `RESHAPE([N,C,Ho,Wo] → [N,C,Ho,1,Wo,1]) → BROADCAST → [N,C,Ho,kh,Wo,kw] →
    // RESHAPE → [N,C,H,W]`, applied to both `y` and the upstream, then
    // `WHERE(x == U(y), U(dY), 0)`. Those rank-6 intermediates are why the plan
    // kept maxpool LAST: their types bake `n`, `c`, `Ho`, `Wo`, so under
    // `grad {}`'s -1 sentinels the reshape targets are meaningless, and no rank-6
    // shape witness exists for the synthesis to type them with (blocker: the
    // single-representative `tensorIrType`). Computing the mask directly needs no
    // upsample at all — invert the window per input element, as AVGPOOL2D_GRAD
    // does — so the body stays rank-4 and the result type is simply `x`'s.
    //
    // Semantics: `dX[n,c,iy,ix] = Σ over the windows (y,x) covering (iy,ix) with
    // x[n,c,iy,ix] == Y[n,c,y,x] of dY[n,c,y,x]`. Ties route the FULL upstream to
    // EVERY within-window winner (the MaxRule/JAX-select convention the upsample
    // spelling had); XLA's `select_and_scatter` picks a single winner instead,
    // which is why the emitter expands to compare+select rather than to it —
    // exact ties are common after a relu, and the backends must not disagree.
    // Unlike the conv/avgpool templates, `x` here is a VALUE operand: its
    // elements are compared against the window max. Host twin: `maxPool2dGrad`.
    MAXPOOL2D_GRAD,

    // §0.4.386 — AVGPOOL2D's adjoint, fused and runtime-extent (Phase A3b).
    // AVGPOOL2D_GRAD(upstream, xTemplate) → dX at xTemplate's shape, attrs
    // copied off the primal (`window`, `window_strides`, `padding`).
    //
    // The spelling this replaces was `RESHAPE([N,C,·,·] → [N·C,1,·,·]) →
    // CONV_TRANSPOSE2D(1/(kh·kw) splat, lhs_dilation = stride, padding solved
    // from the extents) → RESHAPE back`, which folds channels into the batch
    // dim so a single-channel splat kernel applies depthwise without grouped-conv
    // support. Both halves of that are sentinel-hostile: the solved padding AND
    // the reshape's `n * c` target, which under -1 dims is 1 — a reshape that
    // silently claims a shape the data does not have. Fusing removes the channel
    // fold entirely (the adjoint is per-channel, so a direct window scatter needs
    // no grouping trick) and leaves nothing but literal attrs.
    //
    // Semantics: `dX[n,c,iy,ix] = (Σ over the outputs whose window covers
    // (iy,ix) of dY[n,c,y,x]) / (kh·kw)` — the count_include_pad mirror of the
    // primal, which also divides by the FULL window. `xTemplate` contributes
    // SHAPE ONLY; its values are never read. Host twin: `avgPool2dGrad`.
    AVGPOOL2D_GRAD,

    // §0.4.385 — the conv adjoints, fused and runtime-extent (Phase A3b).
    // Both exist for one reason: the classical spellings SOLVE their `padding`
    // from the primal's extents —
    //   dX = CONV_TRANSPOSE2D(dY, W) with lhs_dilation = stride and
    //        window_reversal, padding = [[kEff−1−p_low, p_low + H − dilSize], …]
    //   dW = CONV2D(Xᵀ, dYᵀ) with window_strides = rhs_dilation and
    //        rhs_dilation = stride, padding =
    //        [[p_low, (k−1)·d + dilSize − H − p_low], …]
    // where kEff and dilSize come from the kernel's and dY's spatial dims. Under
    // `grad {}` every one of those extents is a -1 sentinel, so solving at
    // TRANSFORM time bakes arithmetic garbage — `[[-3,1],[-3,1]]` where
    // `[[1,1],[1,1]]` is correct for a stride-1 padding-1 conv — and nothing
    // downstream objects: the interpreter, the emitter and the host twins all
    // honour the attrs they are handed, so the gradient is silently wrong.
    // These ops instead carry the tensor whose extents are the TARGET as a
    // shape-only template operand (the PAD_TO / SUM_TO / SLICE_LIKE convention)
    // and solve the padding at EXECUTION time from its runtime dims. Everything
    // they do carry as attrs (`window_strides`, `padding`, `rhs_dilation`) is a
    // literal fact off the primal conv, so it survives sentinels unchanged.
    //
    // CONV2D_DATA_ADJOINT(upstream, kernel, xTemplate) → dX at xTemplate's
    // shape: the lhs-dilated, tap-reversed transposed conv.
    // CONV2D_KERNEL_ADJOINT(x, upstream, wTemplate) → dW at wTemplate's shape:
    // the batch↔feature transpose trick (Xᵀ ⋆ dYᵀ with stride and rhs_dilation
    // swapping roles, transposed back), FUSED so the result type is simply the
    // kernel's and no rank-4 TRANSPOSE nodes are left in the gradient body.
    // In both, the template contributes SHAPE ONLY — its values are never read
    // (dX's conv needs only dY and W; dW's needs only X and dY).
    // Host twins: `conv2dDataAdjoint` / `conv2dKernelAdjoint`.
    CONV2D_DATA_ADJOINT, CONV2D_KERNEL_ADJOINT,

    // §0.4.391 — CONV_TRANSPOSE2D's own adjoints (Phase A3b), so that
    // differentiating THROUGH a transposed conv works: `grad { x, w ->
    // x.convTranspose2d(w, …) }` was a loud "no VJP rule registered" until here,
    // even though the primal's FIR arm and host twin shipped in §0.4.384.
    //
    //   CONV_TRANSPOSE2D_DATA_ADJOINT(upstream, kernel, xTemplate)   → dX
    //   CONV_TRANSPOSE2D_KERNEL_ADJOINT(x, upstream, wTemplate)      → dW
    //
    // Attrs are the primal's own literals (`window_strides`, `padding`,
    // `lhs_dilation`, `rhs_dilation`, `window_reversal`), so nothing here reads an
    // extent and both are sentinel-safe by construction — the same property
    // §0.4.385's conv adjoints were built for.
    //
    // Unlike those, these need NO padding solve at all. A transposed conv's tap
    // maps input↔output through `yDil = yo·s + ky·d − p_low` with `yDil` a
    // multiple of the lhs dilation `L`, so inverting that one equation per tap —
    // `yo = (iy·L + p_low − ky·d) / s`, kept only when it divides evenly and lands
    // in range — absorbs the padding, both dilations, the strides and the reversal
    // in one step. Verified against central differences over six configurations
    // (lhs_dilation 1 and 2, window_strides 1 and 2, rhs_dilation, reversal,
    // asymmetric padding, and all combined) to ~1e-9 before any Kotlin was written.
    // The templates are shape-only for dX and a VALUE operand for dW (its gather
    // reads `x`), as with MAXPOOL2D_GRAD.
    //
    // Interpreter + host only in v1: there is NO StableHLO arm, so a GPU-targeted
    // build fails loudly at emit ("lowering not yet implemented"), the
    // EMBEDDING_GRAD precedent. No shipped cert regresses — nothing could emit
    // this graph before, since the rule did not exist. The emitting identities are
    // known and recorded in the plan: dX = strided-slice(undilate by L) of
    // CONV2D_DATA_ADJOINT(dy, kernel with axes 0/1 swapped), and dW = the swap of
    // CONV2D_KERNEL_ADJOINT over an interior-dilated x — the latter needs interior
    // `stablehlo.pad`, which the emitter's PAD arm does not do yet.
    CONV_TRANSPOSE2D_DATA_ADJOINT, CONV_TRANSPOSE2D_KERNEL_ADJOINT,

    // Shape
    RESHAPE, TRANSPOSE, BROADCAST, CONCAT, SPLIT, SLICE, GATHER, SCATTER,

    // §0.4.373 — runtime-extent unbroadcast (the reverse mirror of BROADCAST's
    // in-place size-1 stretch). SUM_TO(value, template) → template's shape:
    // NumPy unbroadcast — sum `value` over the leading (value.rank −
    // template.rank) axes AND over every aligned axis where template == 1 but
    // value > 1 (keeping those axes size-1). The `template` operand contributes
    // SHAPE ONLY — its values are never read. This is what BroadcastRule emits
    // for the in-place size-1 stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`): which
    // aligned axes were size-1-stretched is unknowable under the -1 sentinel
    // dims of `grad {}`, so the adjoint reads the extent from the primal
    // operand's ACTUAL runtime shape at execution instead of baking it as an
    // attr. The host twin is `sumToLike(value, template)` (mirror of
    // `stretchLike`).
    SUM_TO,

    // §0.4.374 — runtime-extent zero-pad-to-template (the reverse mirror of
    // SLICE). PAD_TO(value, template) → template's shape, attr `low` (List<Int>,
    // one per axis): place `value` into a zero tensor of the template's shape at
    // offset `low` per axis; the trailing pad `high[i] = template.dim[i] −
    // low[i] − value.dim[i]` is derived from the template's ACTUAL runtime shape,
    // never baked as an attr. This is what SliceRule emits as `slice`'s adjoint:
    // the upstream is zero-padded back into the sliced operand's window, but the
    // operand's extent (needed for `high`) is a -1 sentinel under `grad {}`, so
    // the extent is read from the primal operand's runtime shape at execution.
    // `low` IS a compile-time literal (the user's `slice` start offsets, and 0
    // on the non-sliced axes) so it rides as an attr. The `template` operand
    // contributes SHAPE ONLY — its values are never read. Host twin:
    // `padToLike(value, template, low)`.
    PAD_TO,

    // Phase A2b — runtime-extent window slice (the adjoint half of CONCAT, and
    // the reverse mirror of PAD_TO's "place into a window"). SLICE_LIKE(value,
    // thisTemplate, priorTemplate₀, …, priorTemplateₖ₋₁) → thisTemplate's shape,
    // attr `axis` (Int): extract from `value` the window along `axis` that starts
    // at `Σⱼ priorTemplateⱼ.dims[axis]` and runs for `thisTemplate.dims[axis]`
    // elements, taking every other axis whole. BOTH bounds are read from the
    // templates' ACTUAL runtime shapes, never baked as attrs: this is what
    // ConcatRule emits under `grad {}`, where an operand's window offset is the
    // cumulative sum of the PRIOR operands' runtime axis extents and its length is
    // its own — all -1 sentinels at transform time. With no prior templates the
    // window starts at 0. Every template contributes SHAPE ONLY — its values are
    // never read. Host twins: `sliceLikeStart(value, thisTemplate, axis)` and
    // `sliceLikeAfter{1,2,3}(value, thisTemplate, prior…, axis)`.
    SLICE_LIKE,

    // §0.4.360 — shape-plumbing activation (the DiffKT-gap item 2 surface).
    //
    // WHERE(pred: Bool tensor, a, b) — elementwise select; the differentiable
    // routing primitive (`stablehlo.select`). Gradients flow to a/b through the
    // 0/1 mask; pred gets none.
    //
    // COMPARE(a, b) — elementwise comparison producing a Bool tensor; the
    // `direction` attr is one of EQ/NE/LT/LE/GT/GE (`stablehlo.compare`'s
    // spelling, one op kind instead of six). Non-differentiable (zero rule).
    //
    // PAD(x) — zero edge-padding with `low`/`high` List<Int> attrs
    // (`stablehlo.pad`, interior fixed at 0 in v1). First-class both for users
    // and because SLICE's adjoint IS a pad (and PAD's adjoint is a slice).
    WHERE, COMPARE, PAD,

    // §0.4.45 — SCATTER_ADD(base: rank-1, idx: i32-scalar, value: scalar) → rank-1.
    // Output[k] = base[k] for k != idx, output[idx] = base[idx] + value. Semantically
    // equivalent to ADD(base, SCATTER(BROADCAST(0, base.type), idx, value)) but one op
    // instead of three. Emitted by GatherRule to fuse the "zero-fill + one-hot scatter
    // + elementwise ADD with accumulator" chain into a single rank-1 op; with the
    // accompanying gradAccum fusion in DxirReverseTransform, consecutive gather
    // adjoints compose into a linear SCATTER_ADD chain whose base operand threads
    // the accumulator forward without redundant allocations. Cuts per-call transient
    // FloatArray allocs from ~3N (3 per gather) to ~N (one per gather).
    SCATTER_ADD,

    // Normalization
    LAYERNORM, RMSNORM, BATCHNORM,

    // Attention
    SCALED_DOT_PRODUCT_ATTENTION,

    // Misc
    EMBEDDING, CROSS_ENTROPY, CAST,

    // §0.4.370 — reverse of EMBEDDING w.r.t. its table (the DiffKT-parity
    // embedding VjpRule's fused adjoint). EMBEDDING_GRAD(indices, upstream) →
    // dTable [V, D]: scatter-ADD each upstream row upstream[p, :] back to vocab
    // slot indices[p]. Result type [V, D] carries the vocab/embed dims; indices
    // are non-differentiable so no gradient flows to them. Interpreter arm only
    // in v1 (IR-level certification); the StableHLO emission via scatter+add
    // region is deferred until `embedding` is reachable from `grad {}`.
    EMBEDDING_GRAD,

    // Structured control flow (Stage B substrate per docs/STAGE_B_PLAN.md §3.1).
    //
    // IF: 1 boolean-scalar predicate operand + 2 regions [then, else]. Each region has a
    // single block with no args; the terminator yields op.types-many values per branch
    // (both branches must agree on types and arity). Multi-result IFs are valid.
    //
    // WHILE: N loop-carried operands + 2 regions [cond, body]. Both regions' single
    // blocks take N args matching operand types. condRegion's terminator yields one
    // boolean scalar (the loop predicate); bodyRegion's terminator yields N values
    // matching op.types (== operand types). Single back-edge only — no `break` for
    // Stage B; multi-back-edge form deferred per plan §4.6.
    //
    // Lowering to StableHLO is deferred post-Stage-B. Stage B's coarsening pass closes
    // these regions into straight-line dxir before the SCT reverse transform sees them.
    IF, WHILE,

    // §0.4.31 — Stage C.3b.1 SOI splice op. Carries nested functions as attrs:
    //   attrs["primal_body"]: DxirFunction       — post-coarsening simplified primal
    //   attrs["gradient_body"]: DxirFunction     — pre-computed VJP: (upstream, *primal_operands) → (d_operand_i, …)
    //   attrs["reads_primal_indices"]: Set<Int>  — operand indices the gradient_body dereferences
    //
    // Operands map positionally to primal_body.params; types match primal_body.returns types.
    // Interpretation nests DxirInterpreter.evalFunction on primal_body. Differentiation
    // (C.3b.2) splices gradient_body into the outer gradient function during DxirReverseTransform.
    // StableHLO lowering is deferred — the COARSENED op is a compile-time artefact that the
    // downstream passes (grad + synthesis) consume before code-gen.
    COARSENED,

    // Sharding (SDY-equivalent lowering points)
    SHARD_CONSTRAINT, MANUAL_COMPUTATION,

    // Collectives — inserted by Shardy's export passes or written explicitly in a manual
    // computation. We only need enough kinds here to express adjoint duality for
    // GradShardingVerify; a full set would include ALL_TO_ALL, BROADCAST-across-replicas,
    // etc. For forward lowering these remain as StableHLO custom_call / stablehlo.collective
    // placeholders until the emitter grows dedicated handling.
    ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER,
}
