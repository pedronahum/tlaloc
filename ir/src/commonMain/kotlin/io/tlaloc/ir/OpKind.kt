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

    // §0.4.395 — Phase C2 trig tails (DiffKT parity; the audit pinned TAN/ATAN as
    // the ONLY missing trig ops — DiffKT has no floor/ceil/round/atan2). Gradients:
    // `d/dx tan = 1 + tan²(x)` (= sec²; the tan-recompute form keeps the rule
    // sentinel-safe and lets CSE share the primal's TAN), `d/dx atan = 1/(1 + x²)`.
    // Lowered to `stablehlo.tan` and — since StableHLO has no unary atan — to
    // `stablehlo.atan2(x, splat 1.0)`.
    TAN, ATAN,

    // §0.4.402 — Phase C1 special functions (DiffKT parity; its Dirichlet example
    // depends on lgamma/digamma). LGAMMA = ln|Γ(x)|, DIGAMMA = ψ(x) = (ln Γ)′,
    // TRIGAMMA = ψ₁(x) = ψ′(x). Gradients: `d lgamma = digamma`, `d digamma =
    // trigamma`. Host/interpreter evaluate through the shared Double kernels in
    // `:core/SpecialFunctions.kt` (Lanczos g=7; recurrence-to-asymptotic
    // series). Lowered to `chlo.lgamma`, `chlo.digamma`, and
    // `chlo.polygamma(splat 1.0, x)` — the GB10's XLA parses + legalizes CHLO
    // (certified in PjrtLgammaDigammaSmokeTest).
    LGAMMA, DIGAMMA, TRIGAMMA,

    // §0.4.405 — general polygamma ψ⁽ⁿ⁾(x), closing C1's recorded deferral. The
    // order n rides as a compile-time integer `order` attr (n ≥ 2 by invariant:
    // the FIR normalises the user's polygamma(0)/polygamma(1) to DIGAMMA /
    // TRIGAMMA, and the gradient rules only ever emit order + 1 — TrigammaRule
    // emits POLYGAMMA(2), PolygammaRule POLYGAMMA(order+1) — so the whole
    // ψ-ladder differentiates to any depth). Interpreter/host share the
    // `:core/SpecialFunctions.kt` polygamma kernel (recurrence past 10 + n +
    // differentiated Bernoulli series + cot-derivative-polynomial reflection);
    // lowered as `"chlo.polygamma"(splat n.0, x)` — the generic-form spelling
    // §0.4.402's spike certified against the GB10's XLA.
    POLYGAMMA,

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
    // All three engines have an arm. The interpreter and host twins invert the tap
    // equation directly; the emitter cannot (StableHLO has no primitive for it) and
    // instead goes through §0.4.393's identity
    //     convT(x, w; s, L, d, p) ≡ conv(dilate(x, L), swap01(w); s, d, p)
    // composing pieces that already existed: dX = strided-slice (undilate by L) of
    // CONV2D_DATA_ADJOINT against the channel-swapped kernel, dW = the channel swap
    // of CONV2D_KERNEL_ADJOINT over an interior-dilated x. Certified against real
    // XLA on the GB10 (grads 6.0e-8 vs the interpreter).
    // Emitter scope limit: `window_reversal` is rejected there. With a reversed
    // primal the data side would need `!r` and the kernel side a compensating flip,
    // and that identity is unverified — nothing user-reachable sets the attr, and
    // the interpreter and host twins handle any reversal, so it fails loudly rather
    // than shipping a plausible-looking wrong kernel.
    CONV_TRANSPOSE2D_DATA_ADJOINT, CONV_TRANSPOSE2D_KERNEL_ADJOINT,

    // Shape. (A SPLIT kind lived here until §0.4.453 — demoted §0.4.448, DELETED
    // §0.4.454: nothing ever constructed it outside the emitter arm and IR-
    // plumbing tests — per-piece SLICE is the sanctioned spelling, and the
    // :core/:nn host split() surfaces plus the FIR stack/split folds never
    // emitted it. Re-introduction would need: an interpreter arm, a VjpRule
    // with per-index adjoint routing, a forward tangent, a KotlinSourceRenderer
    // arm or named refusal, and an oracle story — none of which buys anything
    // SLICE compositions don't already have.)
    RESHAPE, TRANSPOSE, BROADCAST, CONCAT,

    SLICE, GATHER, SCATTER,

    // §0.4.396 — REVERSE (DiffKT `flip`, Phase C3): reverse element order along
    // the axes listed in the `dimensions` attr (List<Int>, compile-time user
    // literals), all other axes untouched (`stablehlo.reverse`). Shape- and
    // type-preserving. The op is an involution and SELF-ADJOINT: its VJP is
    // REVERSE(upstream, same axes) and its forward tangent REVERSE(tangent,
    // same axes) — flipping is linear and its permutation matrix is its own
    // transpose. No extent is ever read (axis POSITIONS only), so both rules
    // are sentinel-safe by construction with no runtime-extent template needed.
    // NOTE (plan): the conv adjoints' `window_reversal` special-casing could
    // eventually be re-expressed as an explicit REVERSE of the kernel's spatial
    // axes; deliberately not attempted here — the fused adjoints are certified
    // as they stand.
    REVERSE,

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
    // `stretchLike`). §0.4.399 — VjpRule: BROADCAST_LIKE(upstream,
    // template=value), so reverse-mode differentiates THROUGH it.
    SUM_TO,

    // §0.4.399 — runtime-extent broadcast-to-template: the forward twin of
    // SUM_TO, and its VJP. BROADCAST_LIKE(value, template) → template's shape:
    // NumPy right-aligned broadcast — `value`'s shape must be
    // broadcast-compatible with the template's (each aligned axis equal or
    // size-1; missing leading axes replicated), and `value` is stretched up to
    // the template's ACTUAL runtime shape. The `template` operand contributes
    // SHAPE ONLY — its values are never read. This is what SumToRule emits as
    // SUM_TO's adjoint: the upstream (shaped like SUM_TO's template) must be
    // broadcast back up to the value operand's shape, which is a -1 sentinel
    // under `grad {}` — so the target extents are read from the primal value
    // operand's runtime shape at execution. Its own VJP is SUM_TO(upstream,
    // template=value): the pair is closed under differentiation to any order.
    // Host twin: `broadcastToLike(value, template)` (rank-polymorphic mirror
    // of `sumToLike`).
    BROADCAST_LIKE,

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
    // `padToLike(value, template, low)`. §0.4.399 — VjpRule: SLICE_AT(upstream,
    // template=value, low), so reverse-mode differentiates THROUGH it.
    PAD_TO,

    // §0.4.399 — runtime-extent window extraction at a LITERAL offset: the
    // reverse mirror of PAD_TO, and its VJP. SLICE_AT(value, template) →
    // template's shape, attr `low` (List<Int>, one per axis): cut out of
    // `value` the window of the template's ACTUAL runtime shape starting at
    // offset `low` per axis. This is what PadToRule emits as PAD_TO's adjoint:
    // the upstream (shaped like PAD_TO's template) must be sliced back down to
    // the value operand's window, whose extents are -1 sentinels under
    // `grad {}` — so they are read from the primal value operand's runtime
    // shape at execution, while `low` (already a literal on the PAD_TO node)
    // rides along verbatim. Differs from SLICE_LIKE, whose offset is the SUM
    // of prior templates' runtime extents (a concat window); here the offset
    // is a compile-time literal (a slice start). Its own VJP is
    // PAD_TO(upstream, template=value, low): the pair is closed under
    // differentiation to any order. The `template` operand contributes SHAPE
    // ONLY — its values are never read. Host twin:
    // `sliceAtLike(value, template, low)` (+ `sliceAtLikeRank{1,2,3}` shims).
    SLICE_AT,

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
    // `sliceLikeAfter{1..7}(value, thisTemplate, prior…, axis)` (§0.4.425
    // lifted the twin family from 3 priors to 7).
    // §0.4.404 — VjpRule: PAD_LIKE(upstream, outTemplate=value, same priors,
    // axis), so reverse-mode differentiates THROUGH it (second order through a
    // symbolic concat window).
    SLICE_LIKE,

    // §0.4.404 — runtime-extent window placement at a prior-template offset:
    // SLICE_LIKE's transpose, and its VJP. PAD_LIKE(value, outTemplate,
    // priorTemplate₀, …, priorTemplateₖ₋₁) → outTemplate's shape, attr `axis`
    // (Int): place `value` into a zero tensor of outTemplate's ACTUAL runtime
    // shape at offset `Σⱼ priorTemplateⱼ.dims[axis]` along `axis` (every other
    // axis at 0). This is what SliceLikeRule emits as SLICE_LIKE's adjoint:
    // the upstream (shaped like the concat window) must be zero-padded back
    // into the concatenated value's extent at the window's offset — and that
    // offset is a runtime SUM of the PRIOR templates' axis extents, which no
    // literal `low` can carry (the reason PAD_TO/SLICE_AT could not serve and
    // SLICE_LIKE stayed ruleless from §0.4.399 until now). Its own VJP is
    // SLICE_LIKE(upstream, thisTemplate=value, same priors, axis): the pair is
    // closed under differentiation to any order, like SUM_TO ⇄ BROADCAST_LIKE
    // and PAD_TO ⇄ SLICE_AT. Every template contributes SHAPE ONLY — its
    // values are never read. With no prior templates the window sits at 0.
    // Host twins: `padLikeStart(value, outTemplate, axis)` and
    // `padLikeAfter{1..7}(value, outTemplate, prior…, axis)` (§0.4.425
    // lifted the twin family from 3 priors to 7, in lockstep with SLICE_LIKE's).
    PAD_LIKE,

    // §0.4.419 — Phase E1c-pre: the PARAM-ADDRESSED structural zero.
    // ZEROS_LIKE(template) → template's shape AND dtype, all zeros. The
    // runtime-extent-family treatment (SUM_TO/PAD_TO/BROADCAST_LIKE): the
    // template contributes SHAPE ONLY — its values are never read — and the
    // extents are its ACTUAL runtime dims, so nothing extent-derived is baked.
    //
    // Why it exists: §0.4.54's structural zero (the gradient of a
    // non-differentiable integer param) reached tensor land in §0.4.400 as an
    // anonymous DxirConst(0) whose sentinel-dimmed DxirType cannot say WHICH
    // param it zeroes — so the `grad {}` synthesis admitted exactly ONE
    // integer-typed param per lambda and rejected several as ambiguous. A CSR
    // sparse operand carries TWO integer params (colIdx, rowPtr), so Phase E1c
    // cannot exist under that restriction. DxirReverseTransform now emits the
    // structural zero as ZEROS_LIKE on the CLONED PARAM ITSELF: the zero names
    // its param by construction, and any number of integer params is
    // unambiguous. Dtype-generic in the IR (zeros of the template's dtype);
    // the synthesis's v1 scope is the index-tensor case that needs it
    // (`intZerosLike` host twin, I32 rank-1/2).
    //
    // Differentiation: it creates a constant, so no gradient flows anywhere —
    // its VjpRule contributes nothing (the template is shape-only), and its
    // forward tangent is ZEROS_LIKE of the same template (d/dx 0 = 0). Both
    // arms exist so higher-order transforms compose THROUGH gradient bodies
    // that contain it. Emission: emit-time dims are always concrete, so it
    // folds to a static `stablehlo.constant dense<0>` with the template's SSA
    // value unreferenced (the BROADCAST_LIKE/SLICE_LIKE precedent).
    ZEROS_LIKE,

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
    // §0.4.448 — audit finding C, DEMOTED (don't complete): LAYERNORM is
    // coarsener-recognized/emission-only. Sanctioned layers: the cost model
    // and the StableHLO emitter (emitLayerNorm). NO interpreter arm, NO
    // VjpRule, NO forward tangent — DxirInterpreter and both AD transforms
    // refuse it by name with the sanctioned alternative (spell layernorm via
    // ops, the §0.4.390 batchNorm desugaring precedent; the coarsener owns
    // the fused semantics) — see passes/DemotedOpKinds.kt. REJECTED
    // alternative: completing the kind would duplicate certified coarsener
    // work.
    LAYERNORM, RMSNORM, BATCHNORM,

    // Attention
    // §0.4.448 — audit finding C, DEMOTED (don't complete): SDPA is
    // coarsener-recognized/emission-only. Sanctioned layers:
    // FlashAttentionRecognizer (accepts pre-fused SDPA ops), the cost model,
    // and the StableHLO emitter (emitSdpa). NO interpreter arm, NO VjpRule,
    // NO forward tangent — DxirInterpreter and both AD transforms refuse it
    // by name with the sanctioned alternative (the FlashAttention
    // composition: MATMUL/softmax/MATMUL) — see passes/DemotedOpKinds.kt.
    SCALED_DOT_PRODUCT_ATTENTION,

    // §0.4.465 — Phase H1a (docs/INFERENCE_SERVING_AUDIT.md §2 gap 1):
    // attention over a block-table-indexed KV PAGE POOL — vLLM's core serving
    // primitive, and the single biggest item in the inference arc.
    //
    // INFERENCE-ONLY BY DESIGN — and that is a scope decision, not a gap.
    // Paged attention exists to serve a KV cache that was built by *previous*
    // decode steps; there is no training graph in which a page pool is a
    // differentiable intermediate (the pool is mutated state across steps, and
    // its block table is an integer allocator artifact). So this kind has NO
    // VjpRule and NO forward tangent, and BOTH AD transforms refuse it BY NAME
    // with the training spelling in the message (SDPA / the FlashAttention
    // MATMUL-softmax-MATMUL composition, which the coarseners own with
    // certification) — see [io.tlaloc.ir.passes.INFERENCE_ONLY_OP_KINDS].
    // It differs from the DEMOTED kinds in exactly one way: it DOES carry a
    // real interpreter arm and real StableHLO emission, because inference has
    // to actually run.
    //
    // PAGED_ATTENTION(query, keyCache, valueCache, blockTables, seqLens) → out
    //   query      [numSeqs, numHeads, headDim]      F32/BF16
    //   keyCache   [numBlocks, blockSize, numKvHeads, headDim]
    //   valueCache [numBlocks, blockSize, numKvHeads, headDim]
    //   blockTables[numSeqs, maxBlocksPerSeq]        I32
    //   seqLens    [numSeqs]                         I32
    //   out        [numSeqs, numHeads, headDim]
    // attrs: `scale: Double` (the softmax temperature) — and NOTHING ELSE.
    //
    // The DECODE query shape [numSeqs, numHeads, headDim] (one query token per
    // sequence) is deliberate and matches vLLM's `paged_attention_v1` decode
    // kernel signature. REJECTED alternative: the unified [numTokens, numHeads,
    // headDim] ragged form (vLLM's chunked-prefill path) — it additionally
    // needs a `queryStartLoc` operand AND intra-chunk causal masking, which is
    // a different mask algebra, not a different shape; folding both into one
    // kind would make the mask a mode flag. The prefill/chunked form is a NAMED
    // DEFERRAL (Phase H tail), and prefill today rides the existing dense
    // FlashAttention/GQA path, which is what vLLM's own TPU backend does for
    // the non-paged prefill leg.
    //
    // `blockSize` and `numKvHeads` are NOT attrs: they are keyCache.dims[1] and
    // keyCache.dims[2], and the house sentinel-dims rule forbids baking
    // dim-derived values where an attr could then disagree with the operand.
    // GQA grouping is likewise DERIVED: numHeads / numKvHeads must divide
    // evenly, and query head h reads kv head `h / group` — heads grouped
    // CONTIGUOUSLY per kv head, the [numSeqs, numKvHeads, group, headDim]
    // reshape convention shared by the interpreter and the emitter.
    //
    // blockTables and seqLens are integer tensors: non-differentiable,
    // structural-zero slots by the §0.4.54/§0.4.400 convention — moot here,
    // since the whole op refuses differentiation.
    //
    // Emission is the GATHER-COMPOSED REFERENCE FORM (correctness first):
    // gather the pages named by the block table into a dense
    // [numSeqs, maxBlocksPerSeq*blockSize, numKvHeads, headDim] window, mask
    // past seqLen with -inf, then dense GQA attention. A fused vendor/KPTX
    // paged kernel with recognizer claiming is Phase H4 — the whole point of
    // the coarse kind being a kind.
    PAGED_ATTENTION,

    // §0.4.466 — Phase H1b (docs/INFERENCE_SERVING_AUDIT.md §2 gap 2): the
    // KV-cache write. vLLM's decode step computes one new K and one new V per
    // token and deposits them into the page pool at the slots its allocator
    // named, THEN attends over the pool with PAGED_ATTENTION. This is that
    // deposit.
    //
    // KV_CACHE_WRITE(cache, newKv, slotMapping) → updatedCache
    //   cache       [numBlocks, blockSize, numKvHeads, headDim]
    //   newKv       [numTokens, numKvHeads, headDim]
    //   slotMapping [numTokens]                        I32 — FLAT slots:
    //                                                  blockIdx*blockSize + offset
    //   out         [numBlocks, blockSize, numKvHeads, headDim]
    // attrs: NONE (see [io.tlaloc.ir.KvCacheWriteAttrs]).
    //
    // THE NAME. vLLM calls this `reshape_and_cache`, after the reshape its CUDA
    // kernel does on the way in ([numTokens, numKvHeads*headDim] → the paged
    // layout). That names an implementation detail of one kernel, not the
    // semantics. KV_CACHE_WRITE is the house name because the house names ops
    // for what they mean; the reshape is the emitter's business, and H4's fused
    // kernel may not do one at all.
    //
    // ONE POOL PER OP, and the decode graph calls it twice (K then V).
    // REJECTED: fusing both pools into one two-result op, the literal shape of
    // `reshape_and_cache`. That shape exists to save a kernel launch, which is
    // a concern of the kernel and not of the IR; here it would buy a
    // multi-result op (the §0.4.448 result-index landmine) in exchange for
    // nothing, and it would forbid the perfectly ordinary graph that writes
    // only K. A fused recognizer pattern over the adjacent pair is available
    // in H4 if a kernel ever wants it.
    //
    // FUNCTIONAL, NOT IN-PLACE: the op RETURNS an updated pool. The house IR is
    // value-semantics throughout and an aliasing/donating op would be a new and
    // load-bearing concept in it. REJECTED alternative: in-place mutation with
    // the cache as an inout operand — it would make every pass that reorders or
    // CSEs ops responsible for a memory model none of them has. The cost of the
    // functional form is a notional full-pool copy, and the answer to that is
    // XLA's BUFFER DONATION: the serving loop donates the cache buffer and XLA
    // writes in place under the hood, no copy, no IR concept. Wiring donation
    // through the manifest + PjrtSession is a NAMED FOLLOW-ON (Phase H3).
    //
    // A NEGATIVE slot means PADDING — skip the token. That is vLLM's -1
    // convention, and it is what lets a bucketed static-shape decode graph
    // carry slack lanes without a recompile per real batch size. Non-negative
    // slots must be DISTINCT (see KvCacheWriteAttrs for why the emission
    // depends on it).
    //
    // NOT a duplicate of SCATTER, and this was checked before the kind was
    // added. The existing SCATTER/SCATTER_ADD take a SCALAR I32 index and
    // replace ONE row: expressing a KV write with them needs numTokens GATHERs
    // (to pull each slot out of the slotMapping tensor) plus numTokens
    // SCATTERs, an O(numTokens) op explosion for something a single StableHLO
    // scatter expresses — and it would lose the recognizable single kind that
    // H4's fused kernel and H3's buffer donation both need to claim. The
    // general "vectorized scatter" kind that WOULD subsume this is a
    // differentiable surface needing a VjpRule; this one is inference-only and
    // needs none. They are different ops, not one op twice.
    //
    // INFERENCE-ONLY BY DESIGN — see [io.tlaloc.ir.passes.INFERENCE_ONLY_OP_KINDS].
    // A KV page pool is state mutated across decode steps and addressed by
    // integer allocator bookkeeping; it is not a differentiable intermediate in
    // any training graph, and the training spelling of "put these values
    // somewhere" is the differentiable SCATTER family. Both AD transforms
    // refuse it BY NAME. It DOES carry a real interpreter arm and real
    // StableHLO emission, because serving has to actually run it.
    KV_CACHE_WRITE,

    // §0.4.472 — Phase H5 (docs/INFERENCE_SERVING_AUDIT.md §2 gap 5): the
    // KV-quant read. A quantized page pool is stored as small integer CODES
    // plus a scale per kv head; this op turns that pair back into the float
    // pool PAGED_ATTENTION reads.
    //
    // DEQUANTIZE_KV(codes, scales) → pool
    //   codes  [numBlocks, blockSize, numKvHeads, headDim]   integer
    //   scales [numKvHeads] or [1]                           float
    //   out    [numBlocks, blockSize, numKvHeads, headDim]   float
    // attrs: `kv_quant_dtype: String` — the format tag, REQUIRED and the only
    // attr (see [io.tlaloc.ir.DequantizeKvAttrs]). It is not dim-derived: it
    // is a model-config literal, and it fixes the legal CODE RANGE, which is
    // the one fact about a quantized pool that the operand types cannot carry
    // (an int4 pool and an int8 pool have identical types).
    //
    // out[b, s, h, d] = codes[b, s, h, d] * scales[h] — the symmetric-absmax
    // contract, derived with its error bound in
    // [io.tlaloc.ir.inference.KvQuantPool] (|x − x̂| ≤ scale/2 per element,
    // i.e. absmax/254 for int8). The host codec there and this op are pinned
    // to agree elementwise.
    //
    // REJECTED: folding the dequantization into PAGED_ATTENTION as optional
    // scale operands — it makes that op's arity a mode flag, duplicates the
    // formula inside the arc's most intricate emission, and hides the
    // quantization from every other consumer of a pool. As its own kind it is
    // one visible node: CSE shares it between a layer's K and V paths, and a
    // fused kernel can claim the DEQUANTIZE_KV → PAGED_ATTENTION pair the way
    // §0.4.471's inference lane already claims a single op.
    //
    // FP8 IS REFUSED BY NAME here, not merely absent: int8/int4 are
    // integer-code formats, and `value = code * scale` is simply not fp8's
    // dequantization (its code is a bit pattern with an exponent field). fp8
    // KV-quant waits on a narrow DType — bf16's §0.4.455 tour — and the
    // predicate that says so is KvQuantDtype.isIntegerCoded. The codes
    // themselves ride an I32 tensor in v1 for the same reason (no I8 DType
    // exists), so v1 buys the CONTRACT and not yet the bytes; the serving
    // manifest states that in `ServingKvQuant.codeDtype` rather than leaving
    // it to a doc.
    //
    // INFERENCE-ONLY BY DESIGN — see [io.tlaloc.ir.passes.INFERENCE_ONLY_OP_KINDS].
    // Quantization is a lossy, staircase-shaped map whose useful derivative is
    // zero almost everywhere; training through it is the straight-through
    // estimator, which is a TRAINING-TIME FICTION chosen per recipe (clip
    // range, STE variant) and not a fact about this op. Inventing one here
    // would be exactly the adjoint shortcut the house forbids. Both AD
    // transforms refuse it by name. It DOES carry a real interpreter arm and
    // real StableHLO emission, because serving has to actually run it.
    DEQUANTIZE_KV,

    // Misc
    EMBEDDING, CROSS_ENTROPY, CAST,

    // §0.4.370 — reverse of EMBEDDING w.r.t. its table (the DiffKT-parity
    // embedding VjpRule's fused adjoint). §0.4.400 — grew a third operand:
    // EMBEDDING_GRAD(indices, upstream, tableTemplate) → dTable [V, D]:
    // scatter-ADD each upstream row upstream[p, :] back to vocab slot
    // indices[p]. The template joins the SUM_TO/PAD_TO shape-only-operand
    // family — its VALUES are never read (the interpreter and emitter size the
    // result off the concrete result type), but under `grad {}`'s -1 sentinel
    // dims it is the only sound source of the vocab extent, so the synthesis
    // forwards it to the host twin `embeddingGrad`. Indices are
    // non-differentiable so no gradient flows to them.
    EMBEDDING_GRAD,

    // §0.4.418 — Phase E1b: sparse×dense matmul at IR level (DiffKT
    // `SparseFloatTensor` parity per docs/SPARSE_PARITY_AUDIT.md §2, ratified
    // 2026-09-20). No sparse dtype exists anywhere in the IR — the rank-2 CSR
    // operand rides as its THREE dense component tensors (the audit's
    // rejected-alternative reasoning: a first-class sparse dtype would touch
    // the dims model, IrType atoms, CSE keys and every layer's assumptions).
    //
    // SPARSE_MATMUL(values [nnz] F32, colIdx [nnz] I32, rowPtr [N+1] I32,
    // dense [C, D]) → dense [N, D]: the GNN kernel A · B with A the CSR
    // [N, C]. N is read off rowPtr's RUNTIME extent (rowPtr has N+1 entries —
    // the SUM_TO/PAD_TO runtime-extent house pattern) and C off the dense
    // operand's own runtime shape; nothing extent-derived is ever baked as an
    // attr, so the op is sentinel-safe by construction. The walk is E1a's
    // `SparseTensor.matmul(dense)` SpMM bit-for-bit: per output row a Double
    // accumulator collects `values[k] · dense[colIdx[k], :]` in increasing-k
    // (= increasing-column, canonical CSR) order, then narrows to F32.
    //
    // attr `transposed = true` (absent = false) is the ADJOINT form
    // SparseMatmulRule emits for d_dense = Aᵀ · upstream: the SAME three CSR
    // components, operand 3 the [N, D] multiplicand, plus a FIFTH shape-only
    // template operand (the primal dense operand, values never read) carrying
    // the output row extent C — which no component's runtime shape can supply
    // (rowPtr gives N, colIdx gives nnz) and which is a -1 sentinel under
    // `grad {}`. Materialising transposed component tensors at RULE-BUILD
    // time was rejected: a host transpose there reads extents that are
    // sentinels at transform time (the conv-adjoint padding-solve failure
    // mode §0.4.385 exists to avoid), while the attr keeps the transpose at
    // EXECUTION time where the extents are real. The transposed walk scatters
    // `values[k] · dense[row(k), :]` into output row colIdx[k], scanning
    // source rows in order — per output element the SAME Double-add sequence
    // as E1a's `transpose().matmul(dense)` (the canonical counting-sort
    // transpose orders each output row's entries by source row), so it is
    // bit-for-bit that spelling too, with no transpose ever materialised.
    //
    // colIdx/rowPtr are integer tensors: non-differentiable, structural-zero
    // slots (§0.4.54/§0.4.400). NO StableHLO emission by ratified decision
    // (audit GPU option 1, the §0.4.408 RNG precedent): StableHLO/XLA has no
    // sparse types, per-row segments have irregular lengths (a faithful CSR
    // SpMM needs a WHILE over rows or ELL-style max-degree padding — a
    // DIFFERENT format with its own memory blowup on skewed degrees), and a
    // densify-and-matmul fallback would be a silently-O(N²) behaviour fork
    // (the §0.4.392 principle). The emitter refuses loudly by name.
    SPARSE_MATMUL,

    // §0.4.418 — SPARSE_MATMUL's fused values-adjoint (Phase E1b), the
    // EMBEDDING_GRAD / CONV2D_*_ADJOINT fused-adjoint precedent.
    // SPARSE_MATMUL_VALUES_ADJOINT(upstream [N, D], dense [C, D],
    // colIdx [nnz] I32, rowPtr [N+1] I32) → d_values [nnz]: the SDDMM masked
    // to the sparsity pattern — `d_values[k] = Σ_j upstream[row(k), j] ·
    // dense[colIdx[k], j]` (Double accumulator per stored entry,
    // increasing-j order). Only STORED positions get an adjoint entry: the
    // structural zeros are not inputs, so no gradient exists for them — and
    // the fusion is what keeps the adjoint O(nnz·D) instead of materialising
    // the dense [N, C] outer product `upstream · denseᵀ` and re-masking it.
    // nnz is colIdx's RUNTIME extent, N rowPtr's minus one — the
    // runtime-extent pattern again; nothing extent-derived is baked. The
    // formula is symmetric under swapping (upstream ↔ dense) TOGETHER with
    // the index roles (row(k) ↔ colIdx[k]), which is exactly the transposed
    // primal's values-adjoint — so SparseMatmulRule covers both forms with
    // this one kind, operands swapped. Same ratified emit refusal as
    // SPARSE_MATMUL. Host twin: `sparseMatmulValuesAdjoint`.
    SPARSE_MATMUL_VALUES_ADJOINT,

    // §0.4.408 — Phase D1: stateless PRNG draws (DiffKT `RandomKey` parity).
    // Zero-operand creation ops; attrs carry everything: `key0`/`key1` (Int —
    // the two 32-bit words of the :core `RandomKey`) and `dims` (List<Int>,
    // the concrete result extents, which must equal the result type's — these
    // ops have no FIR lowering, so no -1 sentinel can ever reach them, and the
    // literal-attr shape is the invariant that keeps it that way). The
    // interpreter arms call the SAME `:core/Random.kt` kernels the host
    // surface uses (`uniformFloats` / `normalFloats` — threefry-2x32 in JAX's
    // classic counter layout, pinned bit-for-bit against JAX 0.10 with
    // `jax_threefry_partitionable=False`), so host and interpreter agree
    // bit-for-bit by construction.
    //
    // Differentiation (§0.4.413, Phase D2 v1 — flips §0.4.408's loud
    // refusals): both ops differentiate as CONSTANTS. Reverse: RngDrawRule
    // returns the empty contribution list (piecewise-constant in the key,
    // zero operands — the SIGN/COMPARE zero-gradient convention at arity 0);
    // forward: structural-zero (lazy-null) tangent. This is the
    // reparameterization trick's contract — `sample = loc + scale ⊙ ε` lets
    // d loss/d loc and d loss/d scale flow through ordinary ADD/MUL rules
    // while ε contributes nothing — and when an adjoint READS ε (MulRule's
    // d scale), the cloned RNG op re-draws from the SAME literal key attrs:
    // deterministic, same key → same ε, pinned in DxirRngTest.
    //
    // Emission (§0.4.422 — flips the §0.4.408 refusal): EXPLICIT-threefry,
    // JAX's own approach — never `stablehlo.rng_bit_generator`, whose
    // XLA-internal counter layout does NOT reproduce the JAX-style
    // split-halves stream these kernels pin and would silently fork the
    // random stream between engines. The key words and dims are literal
    // attrs, so the emitted graph is static: iota counters, the 20 ARX
    // rounds as add/shift/or/xor over i32, the key schedule folded at emit
    // time. Uniform draws are BIT-EXACT against the host kernels on any
    // backend (integer ops + the bitcast mantissa trick — GPU-certified in
    // PjrtRngSmokeTest); normal draws run Box-Muller in the same f64
    // intermediates as the host but their log/cos are backend libm calls,
    // so they certify at tolerance, never bit-pinned.
    //
    // Runtime-key operand form (§0.4.432 — lifts §0.4.421's literal-only
    // restriction at the IR level): as an ALTERNATIVE to the zero-operand
    // literal-attr form, the ops accept exactly TWO operands — scalar I32
    // key words — with `key0`/`key1` attrs ABSENT (the forms are exclusive;
    // carrying both is refused). The `dims` attr stays literal in BOTH
    // forms: the result SHAPE must be static (the type system and emitter
    // demand concrete extents), but the STREAM need not be — a runtime key
    // is an ordinary SSA value. Interpreter: key operands are read at
    // execution time (an Int-carrying DxirConst is read verbatim; any other
    // node rides the F32 value domain, guarded to |key| < 2^24 with a loud
    // named refusal beyond it). Differentiation: keys are integers, hence
    // non-differentiable — the existing arms already treat draws as
    // constants regardless of form (RngDrawRule's empty contribution list,
    // the forward transform's structural-zero tangent), and a cloned draw
    // drags its key operand clones into the gradient body through the
    // ordinary usedByAdjoint transitive walk: same key SSA values → same
    // stream. Emission: the key splats broadcast from the rank-0 SSA values
    // and the key schedule (ks2 = k0 ^ k1 ^ 0x1BD11BDA, the five injection
    // adds) EMITS as i32 ops instead of folding — the ARX rounds are
    // identical, so the bit stream stays exact on any backend. The FIR
    // surface (RandomKey-typed vals / lambda params inside grad{}) is the
    // recorded remaining tail — see docs/DIFFKT_PARITY_PLAN.md Phase D.
    RNG_UNIFORM, RNG_NORMAL,

    // §0.4.415 — Phase B5 (customVjp): runtime shape assert on a USER-supplied
    // gradient_body return. CHECK_SHAPE_LIKE(value, template) → value verbatim,
    // after asserting value's runtime dims equal template's. The `template`
    // operand is the COARSENED primal operand the contribution accumulates onto
    // (SHAPE ONLY — its values are never read); `value` is the user vjpFn's
    // returned d_operand. Emitted by `handleCoarsenedAdjoint` around each
    // gradient_body return of a `user_gradient = true` COARSENED whose shapes
    // are not statically decidable (under `grad {}`'s -1 sentinels they never
    // are): a machine-built gradient_body honours the shape contract by
    // construction, but a user body is the user's assertion, and a silently
    // wrong-shaped d_x is the failure mode the design doc's §4.1 names. Both
    // concrete shapes at transform time skip the op (equal) or fail the
    // transform loudly (unequal). Host twin `checkShapeLike`; interpreter arm
    // asserts and passes through. NO StableHLO emission by design (the RNG
    // refusal precedent): StableHLO has no assert, and silently dropping the
    // check on GPU would fork host/device behaviour — customVjp gradient
    // bodies are host/interpreter-certified in v1, with GPU emission of user
    // gradient bodies a recorded Phase B5 tail.
    CHECK_SHAPE_LIKE,

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
    // §0.4.460 — Phase G3a UN-DEMOTES SHARD_CONSTRAINT (reversing §0.4.448
    // finding C's demotion, deliberately): reading showed it is a VALUE
    // IDENTITY with layout metadata (emitShardConstraint asserts shape
    // preservation and emits `sdy.sharding_constraint` — the value passes
    // through untouched), so the identity adjoint (upstream passes through,
    // VjpRegistry.ShardConstraintRule) and identity tangent are honest and
    // cheap. The interpreter evaluates it as identity; the KotlinSourceRenderer
    // renders the pass-through (the metadata has no host-math meaning).
    // Re-applying the SAME constraint to the adjoint value is a NAMED DEFERRAL
    // (JAX's with_sharding_constraint transpose does; it needs mesh carryover
    // into AD-built functions) — GradShardingVerify's param-boundary
    // identity-dual check governs gradient layouts meanwhile.
    SHARD_CONSTRAINT, MANUAL_COMPUTATION,

    // Collectives — inserted by Shardy's export passes or written explicitly in a manual
    // computation. A full set would include ALL_TO_ALL, BROADCAST-across-replicas, etc.
    // §0.4.460 — Phase G3a promotes ALL_REDUCE to a REAL OP (reversing
    // §0.4.448 finding C's demotion, deliberately — the intra-job
    // coordinator's first brick). Attribute convention and validation live in
    // [io.tlaloc.ir.AllReduceAttrs]: `replica_groups: List<List<Int>>`
    // (absent = [[0]], the single-replica program) and `reduction: String`
    // (absent = "sum"; v1 supports sum ONLY — mean scales, max/min need
    // subgradient routing, both refuse by name). Shape-preserving. Layers:
    // interpreter (single-process SPMD semantics: |group(0)| × value — exact
    // identity when replica_count == 1; the multi-replica arm is exercised
    // only via unit semantics until G2b/G4), AllReduceRule (all-reduce-sum is
    // SELF-ADJOINT in the replicated per-replica-upstream view: the adjoint
    // is the same ALL_REDUCE on the upstream — NOTE this is a different level
    // than GradShardingVerify.adjointOf's varying→invariant sharded-pipeline
    // table, where the dual is identity; both are recorded, neither replaces
    // the other), forward tangent (linear: the same op on the tangent),
    // StableHLO emission (`"stablehlo.all_reduce"` with the stablehlo.add
    // reduction region + dense replica_groups). KotlinSourceRenderer refuses
    // by name (no single-process host spelling — rendering ×|group| would
    // bake a distribution fact into host math).
    // ALL_GATHER / REDUCE_SCATTER remain GradShardingVerify-duality-only
    // kinds (no interpreter arm, no rules, no emission) until a slice needs
    // them.
    ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER,
}
