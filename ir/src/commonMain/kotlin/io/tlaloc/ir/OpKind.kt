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

    // Shape
    RESHAPE, TRANSPOSE, BROADCAST, CONCAT, SPLIT, SLICE, GATHER, SCATTER,

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
