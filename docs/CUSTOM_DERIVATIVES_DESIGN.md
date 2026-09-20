# Custom Derivatives Design — Phase B5 (§0.4.410)

**Status: DESIGN ONLY — the API shape is a PRODUCT decision awaiting
Pedro's ratification. Nothing in this document is implemented.**
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

### Candidate C — `customJvp` / the forward side

`customJvp(f, jvpFn)` with `jvpFn: (A, A) -> R` (primal, tangent → tangent)
is the forward twin. It is NOT required for v1 — but the forward
SEMANTICS question it answers IS v1-blocking; see §4.

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
the forward refusal); the debug cross-check helper if not folded into v1;
multi-result `f` reverse-only.

**Multi-§, explicitly deferred**: Candidate B annotations (same-module),
cross-module (needs the serialized-dxir decision), `Wrappable`-style
data-structure params (the §0.4.365 audit's B5 note), scalar/`DScalar`
slots (rides the FloatScalar boxing tail from §0.4.397).

## 6. Open product questions for Pedro (blocking implementation)

1. API spelling: Candidate A call-form (recommended) — name bikeshed
   (`customVjp` / `withCustomVjp` / `customGrad`)?
2. Forward-mode policy for user nodes: refuse-unless-jvpFn (recommended)
   or auto-differentiate-primal with a documented divergence?
3. Ship the JVP⇄VJP debug oracle helper in v1?
4. Is same-module Candidate B worth its § before the cross-module story
   exists, or does A cover the demand?
