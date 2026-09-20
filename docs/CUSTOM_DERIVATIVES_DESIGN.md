# Custom Derivatives Design — Phase B5 (§0.4.410)

**Status: RATIFIED (Pedro, 2026-09-20) with the recommended answers to all
four §6 questions — Candidate A call-form named `customVjp`/`customVjp2`;
forward-mode = refuse-unless-jvpFn; the JVP⇄VJP debug oracle ships
(`io.tlaloc.autograd.checkCustomVjp`); Candidate B deferred. v1 LANDED at
§0.4.415; the forward side (Candidate C — `customJvp`/`customJvp2`/
`customVjpJvp`/`customVjpJvp2`) LANDED at §0.4.416** — see §7 below for
what the implementation taught that this design had not anticipated, and
§8 for the forward-side decisions (recorded per the §0.4.383 precedent).
Companion to [DIFFKT_PARITY_PLAN.md](DIFFKT_PARITY_PLAN.md) Phase B5.

## 1. What "custom derivatives" means, in both systems

**DiffKT** has no registration API at all. Its `customReverse` example is
the whole mechanism: the user writes an operation as a function that
type-dispatches on its argument, and for a `ReverseTensor` input returns an
anonymous `ReverseTensor` subclass whose `backpropagate()` override pushes
whatever gradient the user wrote (`x.pushback(myCos(x.primal) * upstream)`).
Custom derivatives are an emergent property of OO dispatch on a runtime
tape. That is unportable verbatim: Tlaloc's differentiation happens at
compile time, in dxir, with no tape in the compiled path.

**Tlaloc already has the internal analogue, certified for 300+ sections**:
`OpKind.COARSENED` — an op carrying `attrs["primal_body"]` (a
`DxirFunction`: the fused computation) and `attrs["gradient_body"]` (a
`DxirFunction` with signature `(upstream…, *primal_operands) →
(d_operand_i, …)`), plus `attrs["reads_primal_indices"]`.
`DxirReverseTransform.handleCoarsenedAdjoint` splices `gradient_body`
inline into the outer gradient function (id-remapped clone, contributions
accumulated by the standard ADD convention; multi-result supported since
§0.4.179 with K upstream params). Since §0.4.403, `DxirForwardTransform`
has the mirror arm: the tangent of a COARSENED is the forward transform of
its `primal_body`, spliced inline. Today only `PhiCalculus.coarsenFunction`
ever CONSTRUCTS such a node — with a MACHINE-derived `gradient_body`. B5
is: let the USER supply the `gradient_body`.

**Why a user would want it** (the use cases pin the design):

1. **Opaque/faster adjoint** — the user knows a cheaper or numerically
   better VJP than composition would produce (classic: `logSumExp`-style
   stabilizations, iterative solvers via implicit function theorem).
2. **Non-lowerable primal** — `f` contains something the FIR lowering
   rejects (an unknown library call throws `LoweringException` today, and
   the whole body falls back / fails). A custom derivative CANNOT fix this
   one alone: the PRIMAL body must still be lowerable to run in the value
   stream. What it fixes is the *derivative* being unavailable
   (fused-adjoint ops like `EMBEDDING_GRAD` refuse rev∘rev today; a user
   custom-VJP of an embedding-bearing block is the escape hatch the
   §0.4.401 refusal pins anticipate).
3. **Derivative that differs from the math of the body by intent**
   (straight-through estimators, gradient clipping/stopping). Note
   `stopGradient` sugar falls out of B5 for free (`customVjp(f = identity,
   vjpFn = zeros)`).

## 2. Candidate user surfaces

### Candidate A — `customVjp(f, vjpFn)` intrinsic call-form (RECOMMENDED for v1)

```kotlin
// :autograd declarations (stubs; the K2 plugin rewrites call sites):
fun <A, R> customVjp(f: (A) -> R, vjpFn: (R, A) -> A): (A) -> R
fun <A, B, R> customVjp2(f: (A, B) -> R, vjpFn: (R, A, B) -> Pair<A, B>): (A, B) -> R
```

Usage — the call-form is used INSIDE a differentiated lambda (or as the
whole lambda):

```kotlin
val g = grad { x: DTensor<Rank1<Sym>, F32> ->
    val stableExp = customVjp(
        f = { t -> exp(t) },
        vjpFn = { upstream, t -> upstream * exp(t) },   // user's adjoint
    )
    stableExp(x).sum().toFloat()
}
```

**Lowering**: the FIR arm, on meeting `customVjp(f, vjpFn)` applied inside
a body it is lowering, recursively lowers BOTH lambda arguments with
`FirLambdaToDxirLowering` and emits ONE `COARSENED` node: `primal_body` =
lowered `f`, `gradient_body` = an ADAPTER of lowered `vjpFn` — the param
order `(upstream, x)` already matches `handleCoarsenedAdjoint`'s
`(upstream, *primal_operands)` contract exactly, so for the 1-arg form the
adapter is the identity; `customVjp2`'s `Pair` return unboxes to the
2-return `(d_a, d_b)` convention (synthesis already boxes/unboxes pairs for
2-return functions — the same seam, run backwards). `reads_primal_indices`
= the operand indices `vjpFn` actually references (computed from the
lowered body — a param with no uses is not a read; this keeps the
§0.4.386-style clone discipline honest).

- Reverse mode: `handleCoarsenedAdjoint` splices the user's `gradient_body`
  — ZERO new transform code.
- Primal-only evaluation (no differentiation): the interpreter already
  nests `evalFunction` on `primal_body`; synthesis decomposes COARSENED
  today (`decomposeCoarsened` exists as the §0.4.403 test oracle) or the
  splice-free direct path — the plugin's existing COARSENED handling
  applies, since `grad {}` bodies have carried COARSENED nodes since
  §0.4.33.
- Both lambdas must be same-compilation-unit lowerable literals (the FIR
  lowering walks source bodies). v1 scope: 1–2 tensor args, tensor return,
  straight-line `vjpFn`.

**Why call-form and not a value**: `customVjp(f, vjpFn)` OUTSIDE a
differentiated context returns a plain composed function whose stub just
applies `f` (so host-side behavior is coherent), but the derivative
attachment only exists where the FIR lowering can SEE the call-form
applied. A first-class "function-with-derivative" VALUE flowing through
variables and across module boundaries is the deserialized-body problem
(§5) — out of v1 scope, and the FIR arm should refuse loudly when the
customVjp result escapes rather than being applied in place (assignment to
a local val applied later in the same body is fine — FIR data flow within
one lowered body is visible; escaping the lambda is not).

### Candidate B — annotation-driven registration (DEFERRED — multi-§, cross-module blocked)

```kotlin
@CustomAdjoint(of = "com.example.mySin")
fun mySinVjp(upstream: DTensor<S, F32>, x: DTensor<S, F32>): DTensor<S, F32> =
    upstream * cos(x)
```

Call sites stay untouched (`mySin(x)` inside `grad {}` starts working) —
the closest spelling to DiffKT's "differentiate through opaque library
functions" ergonomics, and the right END STATE. But it requires the FIR
lowering, on an unknown call, to (a) resolve the annotation pairing across
module boundaries — feasible, annotations survive in metadata — and (b)
LOWER BOTH BODIES, which are NOT in metadata for a dependency module.
Cross-module needs serialized dxir riding in the jar (the coarsening-cache
machinery — `CACHE_DIR_PROPERTY`, CAS-versioned entries — is the natural
precedent/vehicle, but wiring it as a distribution format is its own
design). Same-module annotation support without that is Candidate A with
worse discoverability of failure (action at a distance when the annotation
is misspelled). Recommendation: land A first; B rides on a later
"serialized dxir in artifacts" decision.

### Candidate C — `customJvp` / the forward side (LANDED §0.4.416)

`customJvp(f, jvpFn)` with `jvpFn: (A, A) -> R` (primal, tangent → tangent)
is the forward twin. It was NOT required for v1 — but the forward
SEMANTICS question it answers WAS v1-blocking; see §4. Landed at §0.4.416
together with the combined `customVjpJvp(f, vjpFn, jvpFn)` and both 2-arg
spellings; see §8 for the as-built record (param convention, the reverse
mirror refusal, the pipeline decompose steps).

## 3. What the FIR checker can verify at compile time

The `TlalocIntrinsicCallChecker` precedent (probe-lower at check time, red
squiggles at the call site) extends naturally:

- **Type pairing** — free, from generics: `vjpFn: (R, A) -> A` against
  `f: (A) -> R` is enforced by the resolver itself; a wrong-shaped
  `vjpFn` that still unifies (e.g. everything `DTensor<Shape, F32>`) is a
  RUNTIME shape question (below), not a checker one.
- **Lowerability of both bodies** — probe-lower `f` and `vjpFn` at check
  time; failures diagnose WHICH body and why (the current probe
  discipline).
- **Splice-compatibility of `vjpFn`** — straight-line, single-result ops
  only (the `handleCoarsenedAdjoint` limitation is pre-existing and
  documented); probe and refuse with the limitation named.
- **Differentiability of the COMPOSITION** — the existing per-intrinsic
  probes (reverse/forward/composed) run on the body containing the
  COARSENED node exactly as they do today.
- **NOT verifiable statically**: extent agreement between `f`'s operands
  and `vjpFn`'s returns (runtime quantities under `grad {}`'s -1
  sentinels), and mathematical CORRECTNESS of the user's adjoint (that is
  the point of the feature; see §4 for the debug oracle).

## 4. Failure modes, and the two design decisions inside them

1. **Shape contract violations** (`vjpFn` returns d_x of the wrong extent):
   undetectable statically, so it must fail LOUDLY at runtime. The house
   precedent is `conv2dDataAdjoint` asserting the solved conv lands on its
   template: the splice (or a thin wrapper op) should assert each
   `gradient_body` return's runtime dims equal its operand's before
   accumulation. Cheap, and turns silent wrong-gradient into a named error.
2. **Mathematically wrong `vjpFn`**: unverifiable by construction — the
   MEMORY.md rule ("coarsener adjoints honour the math") applies to
   MACHINE-built bodies; a user body is the user's assertion. But Tlaloc
   can offer what DiffKT cannot: since §0.4.403 BOTH an automatic tangent
   (forward-of-`primal_body`) and the user's adjoint coexist on the same
   node, so a **debug-mode JVP⇄VJP cross-identity check**
   (`⟨ȳ, J·v⟩ == ⟨user-VJP(ȳ), v⟩` at random seeds) is implementable with
   ZERO new math — as a test-time helper (`checkCustomVjp(f, vjpFn, at)`)
   or a system-property-gated runtime probe. Only meaningful when `f` is
   itself automatically differentiable (use case 1, not use case 3 —
   straight-through estimators FAIL this check by design, so it must be
   opt-in, never a default gate). **Product decision: ship the helper?**
   (Recommended: yes, as a `:autograd` test utility.)
3. **Forward-mode semantic fork — THE design decision.** §0.4.403's
   COARSENED forward arm differentiates `primal_body` AUTOMATICALLY,
   ignoring `gradient_body`. For machine-coarsened nodes the two agree by
   construction. For a USER node they need not (use case 3 deliberately
   diverges; use case 1's stabilized adjoint may differ numerically). If
   `jvp {}` silently auto-differentiates a body whose reverse mode honours
   the user's intent, forward and reverse DISAGREE — the §0.4.392
   principle says that fork must not be silent. Resolution: a
   `user_gradient = true` attr on user-built COARSENED nodes;
   `DxirForwardTransform` REFUSES them loudly by name unless the user also
   supplied a `jvpFn` (Candidate C's attr `tangent_body`, spliced by the
   same arm — mechanical once decided). **Product decision: refuse-unless-
   supplied (recommended) vs auto-with-documented-divergence.**
4. **Closure capture** in `f`/`vjpFn` (referencing outer `val`s): the FIR
   lowering's capture story predates B5 (`grad {}` lambdas capture today —
   `Capture.kt`); whatever it supports applies verbatim, but the v1 cert
   must include one captured-value case to pin it.
5. **Nesting**: rev-over-custom-rev works mechanically (the splice inlines
   `gradient_body` as ORDINARY ops; the second reverse pass differentiates
   them if ruled — same status as any gradient body). fwd-over-custom is
   decision 3. Multi-result `f` is supported by `handleCoarsenedAdjoint`
   (K upstreams) but NOT by the forward splice (refused since §0.4.403) —
   v1 gates on single-result and says so.

## 5. Slicing: one-§ vs multi-§

**One § (v1, after ratification)**: Candidate A `customVjp` (1-arg +
2-arg), FIR arm lowering both lambdas → COARSENED with `user_gradient`
attr, forward-transform refusal arm, checker probes, runtime shape asserts
on splice returns, certs: analytic custom adjoint (a deliberately
NON-mathematical vjpFn pinning that the user body is what runs — e.g.
`vjpFn = 3·upstream` where composition would give `2x·upstream`),
stopGradient sugar, captured-value case, rev∘custom nesting, the refusal
pins. No emitter work (COARSENED is decomposed/spliced before codegen —
pre-existing invariant).

**One § each, later**: `customJvp` + the combined `customVjpJvp` (flips
the forward refusal) — LANDED §0.4.416, all four spellings in one § since
the 2-arg widening proved purely mechanical (§8); the debug cross-check
helper if not folded into v1 (it was — §0.4.415); multi-result `f`
reverse-only.

**Multi-§, explicitly deferred**: Candidate B annotations (same-module),
cross-module (needs the serialized-dxir decision), `Wrappable`-style
data-structure params (the §0.4.365 audit's B5 note), scalar/`DScalar`
slots (rides the FloatScalar boxing tail from §0.4.397).

## 6. Open product questions for Pedro (RESOLVED — ratified 2026-09-20)

1. API spelling: **Candidate A call-form, named `customVjp` / `customVjp2`.**
2. Forward-mode policy for user nodes: **refuse-unless-jvpFn** — the
   `user_gradient = true` attr on user-built COARSENED nodes makes
   `DxirForwardTransform` refuse loudly by name unless a `tangent_body`
   (Candidate C's `jvpFn`, a recorded tail) is also present.
3. Debug oracle: **ships in v1** as `io.tlaloc.autograd.checkCustomVjp` —
   pure host math (central differences of `f` vs the user's `vjpFn` in the
   inner-product identity), no plugin, no IR; opt-in, and expected to FAIL
   for straight-through estimators by design.
4. Candidate B: **deferred** on the serialized-dxir decision, as recommended.

## 7. What v1 (§0.4.415) taught — corrections and decisions this design left open

- **The shape-contract check (§4.1) landed as a thin check op**,
  `OpKind.CHECK_SHAPE_LIKE(value, template)` — a value-identity that asserts
  the value's RUNTIME dims equal the template's. `handleCoarsenedAdjoint`
  wraps each `gradient_body` return of a `user_gradient` node: statically
  concrete-and-equal shapes skip the wrap, concrete-and-UNEQUAL ones fail
  the TRANSFORM loudly (check-time visible through the intrinsic probes),
  and anything sentinel-bearing — every `grad {}` body — gets the runtime
  op. Interpreter arm + host twin `checkShapeLike` + synthesis arm +
  `CheckShapeLikeRule` (identity VJP, so rev∘custom nests through it) + a
  forward tangent arm (hessian composes over it). Machine-coarsened bodies
  stay byte-identical — they honour the contract by construction.
  **Deliberate non-emission**: the emitter REFUSES `CHECK_SHAPE_LIKE` by
  name (the RNG-refusal precedent — StableHLO has no assert, and silently
  dropping the check would fork host/device behaviour). customVjp gradient
  bodies are host/interpreter-certified in v1; the honest GPU story is a
  recorded tail.
- **§2's "the plugin's existing COARSENED handling applies" was too
  optimistic**: a customVjp COARSENED sits MID-BODY, and under `grad {}`'s
  sentinels the reduction adjoints carry their operand as a runtime shape
  template — so the reverse transform CLONES the COARSENED into the
  gradient function, where synthesis has no arm. Machine coarsening never
  hit this (it wraps the WHOLE body; nothing consumes the node's result).
  Fix: `TlalocIrGenerationExtension` runs `decomposeCoarsened` (the Layer-4
  CPU-baseline pass, exactly its job) over the gradient function when a
  COARSENED survived — the gradient splice already happened, so what
  remains is pure value recomputation. Correspondingly,
  `computeUsedByAdjoint` marks ALL of a user node's operands used (each may
  be dereferenced as a CHECK template).
- **Capture (§4.4) is narrower than "whatever Capture.kt supports"** —
  that file is the runtime TAPE tracer, not FIR closure capture, which
  never existed for lambda-external values. v1 supports capturing outer
  local `val`s whose lowered value is a compile-time constant (re-emitted
  inline in the inner body); any other capture refuses loudly naming the
  restriction (a non-const capture would need an extra COARSENED operand
  WITH a gradient slot the user's vjpFn does not return — deferred).
- **Escape (§2) includes re-binding**: `val g = f` refuses with the same
  named diagnostic as passing `f` out — any reference to a customVjp-bound
  val outside the invoke-receiver position is an escape. Named arguments
  resolve by NAME at the FIR checker stage (the K2 unwrap landmine bites
  the IR stage, not FIR), so `customVjp(vjpFn = …, f = …)` is safe.
- **The v1 cert list all landed**: the 3·upstream non-mathematical vjpFn
  E2E (scalar + tensor, no fallback), `customVjp2`'s Pair unboxing (7/11
  E2E), stopGradient sugar E2E (∇ Σ x·sg(x) = x), a captured literal val
  E2E, rev∘custom nesting at IR level (the second reverse differentiates
  the spliced user ops), the forward refusal (IR pin + a compile-time E2E
  ERROR naming `user_gradient` — `NOT_DIFFERENTIABLE` is error-severity, so
  `jvp {}` over a customVjp body fails the BUILD), the escape refusal, and
  the shape assert at all four layers (transform-time, interpreter, host
  `checkShapeLike`, and an E2E runtime failure via two same-static-type
  operands of different runtime extents with a swapping vjpFn — the
  mismatch no static type can see).

**Recorded tails** (post-§0.4.416): GPU emission of user gradient bodies;
multi-result `f` (reverse-only); non-const captures; Candidate B;
`vjp {}`/`jacobianReverse` over custom-derivative bodies (the decompose
step runs on the `grad`, `jvp`, and `jacobian`/`hessian` branches — the
seeded-cotangent branches still fall back loudly to `pluginMissing`).

## 8. What the forward side (§0.4.416) decided and taught

- **The `tangent_body` param convention**, left open by §2's Candidate C
  sketch: `(primal_0 … primal_N-1, tangent_0 … tangent_N-1) → (dy)` —
  DxirForwardTransform's OWN emission order (params then d_params), so the
  user's declared `jvpFn` signature IS the splice contract with an identity
  adapter at every arity, the exact analogue of vjpFn's `(upstream, x…)`.
  `customJvp2`'s `jvpFn: (A, B, A, B) -> R` returns the single result
  tangent directly — no Pair unboxing anywhere on the forward side, which
  is why the 2-arg widening was purely mechanical and all four spellings
  landed in one § (the §0.4.406 arity-agnosticism, again).
- **Construction-time validation**: `validateCoarsenedShape` checks a
  present `tangent_body` (2·N params typed like the operands twice over,
  single return typed like the single result, `user_gradient` required —
  machine coarsening never stores tangents), and `gradient_body` becomes
  optional for EXACTLY the customJvp-only shape (`user_gradient` +
  `tangent_body`, nothing else).
- **The mirror refusal**: reverse mode over a customJvp-only node refuses
  loudly naming `customJvp` in `handleCoarsenedAdjoint` — the same
  no-silent-fork sentence as the forward refusal, pointing at
  `customVjpJvp` / `jvp {}`. Since the `grad` checker probe runs the
  reverse transform, `grad {}` over a customJvp body fails the BUILD
  (`NOT_DIFFERENTIABLE` is error-severity), the §0.4.415 symmetry.
- **The forward shape contract is the reverse one mirrored**: a
  sentinel-typed user tangent wraps in `CHECK_SHAPE_LIKE` against the
  node's own value clone (the one template whose runtime dims dy must
  match); statically concrete types skip the wrap — their extents were
  already pinned by the construction-time type equality.
- **Semantic fork, explicit**: `customVjpJvp`'s two bodies are SEPARATE
  assertions. Certified with deliberately INCONSISTENT bodies (vjpFn = 5·u,
  jvpFn = 3·dt, primal math = 2x — three different answers, each mode
  pinned to ITS own) and with CONSISTENT bodies through the JVP⇄VJP
  cross-identity E2E on the tensor path.
- **hessian composes through a customVjpJvp node** — the open question in
  the §5 slicing. forward-over-reverse meets the COARSENED **again** inside
  the reverse-produced body (the §7 value-recomputation clone carries all
  attrs), and the forward arm splices `tangent_body` there — certified at
  IR level (H·v of Σ c(x)² = 12x²·v through the mid-gradient-body splice)
  and E2E (`hessian` of Σ x² over the sentinel tensor path = 2I).
- **Two more pipeline decompose steps**: the §7 "COARSENED survives into
  the function being synthesised" story applies to the forward branches
  too — the forward transform clones the node as the VALUE stream — so the
  plugin's `jvp`/`valueAndJvp` and `jacobian`/`hessian` branches now run
  `decomposeCoarsened` before synthesis when a COARSENED survived (no-op
  otherwise, byte-identical pre-B5). This incidentally opens synthesis to
  machine-COARSENED forward bodies that previously fell back.
