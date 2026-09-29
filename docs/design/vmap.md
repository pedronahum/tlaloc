# vmap: compile-time batching

Status: design for branch `feat/vmap`. Progress and decisions taken while building it
are in [../work-log/vmap-progress.md](../work-log/vmap-progress.md).

## Goal

`vmap` takes a function over one example and returns the same function over a batch,
generated at compile time. It composes with the differentiation intrinsics in both
orders: `vmap { grad { } }` gives per-example gradients, `grad { vmap { } }`
differentiates a batched computation.

## What exists today (the parts vmap builds on)

- **Recognition.** `TlalocIntrinsicCallChecker` (FIR) matches calls by FQN in
  `io.tlaloc.autograd`, lowers the lambda literal with `FirLambdaToDxirLowering.lower`
  into a `DxirFunction`, probes the transform it will need (reverse, forward, seeded),
  reports failures at the call (`LAMBDA_NOT_LOWERABLE`, `NOT_DIFFERENTIABLE`, ...) and
  records the function in `TlalocLoweringHandoff` keyed by file and source range.
- **Rewrite.** `TlalocIrGenerationExtension` (IR) takes the handoff for each matching
  call, runs the transform (`DxirReverseTransform`, `DxirForwardTransform`), and
  `DxirToIrSynthesis.synthesise` builds a Kotlin lambda of `:core` host calls typed by
  the call site. A captured runtime value becomes a trailing input-only parameter,
  bound at the call site (`capturedBindings`).
- **Dimensions.** The plugin writes every dimension of a lambda parameter as `-1`.
  Synthesized code reads extents at run time: from shape-template operands
  (`SUM_TO`, `BROADCAST_LIKE`, the template form of `BROADCAST`, ...), from
  `param.dims[i]`, or from host functions that broadcast. The capture API, tests and
  the StableHLO emitter use concrete dimensions.
- **Rules.** Reverse rules live in `VjpRegistry` (`Vjp.kt`), forward rules in
  `DxirForwardTransform.tangentOf`. Both are per `OpKind`.
- **Engines.** `DxirInterpreter` / `DxirInterpreterF64` evaluate concrete-dim DXIR;
  `StablehloEmitter` emits it; `PjrtSession.runOn` / `runOnHost` run it.
  Both engines already support canonical batched `MATMUL` (`[..., m, k] x [..., k, n]`,
  equal ranks) and NumPy right-aligned implicit broadcasting in elementwise binaries.
  All linear-algebra ops are rank 2 only in the interpreter, the emitter, synthesis and
  the reverse rules.
- **Nesting.** A transformation intrinsic called inside another one's lambda is not
  lowered today (`unresolved call` / unsupported call). The nesting matrix is certified
  at the DXIR level (`DxirNestingMatrixTest`) and through `hessian` (forward over
  reverse), not through user-written nested lambdas.

## API

In `:autograd`, package `io.tlaloc.autograd`, every declaration marked
`@ExperimentalTlalocApi`.

```kotlin
object Batch : IndexName { override val name = "batch" }   // the user's axis name

val perExampleLoss = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Named<Feat, Sym>>, F32> ->
    (x * x).sum()
}
// perExampleLoss: (DTensor<Rank2<Named<Batch, Sym>, Named<Feat, Sym>>, F32>)
//                     -> DTensor<Rank1<Named<Batch, Sym>>, F32>
```

### The batch axis in the type

The batch axis is a named axis, `Named<N, A>`, prepended to every batched argument and
every result. `N` is an `IndexName` object and `A` is `Sym` or `Bounded<B>`:

```kotlin
class BatchAxis<N : IndexName, A : ShapeAtom>
fun <N : IndexName> batchAxis(name: N): BatchAxis<N, Sym>
fun <N : IndexName, B : DimBound> batchAxis(name: N, bound: B): BatchAxis<N, Bounded<B>>
```

| Per-example type | Batched type |
|---|---|
| `DTensor<ScalarShape, D>`, `Float` (F32), `Double` (F64) | `DTensor<Rank1<Named<N, A>>, D>` |
| `DTensor<Rank1<S0>, D>` | `DTensor<Rank2<Named<N, A>, S0>, D>` |
| `DTensor<Rank2<S0, S1>, D>` | `DTensor<Rank3<Named<N, A>, S0, S1>, D>` |
| `DTensor<Rank3<S0, S1, S2>, D>` | `DTensor<Rank4<Named<N, A>, S0, S1, S2>, D>` |
| `DTensor<Shape, D>` (rank erased by an op such as `sum(axis)`) | `DTensor<Shape, D>` |

Kotlin has no type-level functions, so this mapping is a set of overloads, one per
(input kind, output kind) pair, generated into `VmapIntrinsics.kt`. Overloads that differ
only in the lambda's return type are resolved with `@OverloadResolutionByLambdaReturnType`
(checked on Kotlin 2.4.20 with a prototype before this design was written). The lambda's
parameter must be written with its type, as for `grad`.

Consequences:

- **A batch mismatch is a compile error.** Passing a
  `DTensor<Rank2<Named<Time, Sym>, ...>>` to a function vmapped over `Batch` is a Kotlin
  argument type mismatch at the call's file, line and column. Two arguments of `vmap2`
  batched over one axis share `N` and `A`, so they cannot come from different batch axes.
- **A `Bounded` batch axis** (`batchAxis(Batch, MaxBatch)`) gives
  `Named<Batch, Bounded<MaxBatch>>`, so the batched function accepts only tensors whose
  batch axis carries that bound. The bound is a type, not a run-time check: the
  synthesized function runs at any batch size.
- **Two batched arguments** (`vmap2(axis, Batched, Batched)`) share the batch axis's type,
  which does not make their extents equal; the synthesized function checks at run time
  that both have the same batch size (`checkBatchAxes`) and throws otherwise, since a
  batch of one would broadcast against the other.
- **The batch name must be new.** A per-example type that already has an axis named `N`
  is refused at compile time (`VMAP_AXIS_NAME_CLASH`): after batching, two axes would
  carry the same name, which `contract` cannot disambiguate.
- **Rank limit.** Per-example rank 0 to 3, because synthesis handles tensors up to rank 4.
  A rank-erased `DTensor<Shape, D>` output stays erased.

### Batched and broadcast arguments (`in_axes`)

- `vmap(axis) { x -> ... }`: the one argument is batched along a new leading axis.
- `vmap2(axis, a, b) { x, w -> ... }` where `a` and `b` are the marker objects
  `Batched` or `Broadcast` (types `InAxis.Batched` / `InAxis.Broadcast`). A `Broadcast`
  argument is passed unchanged and shared by every example; its type is not rewritten.
  The markers choose the overload, so the returned function's type follows from them.
  At least one argument must be `Batched`; `vmap2(axis, Broadcast, Broadcast)` does not
  exist.
- A runtime value the lambda captures (a local or a parameter declared outside it) is
  broadcast, exactly as it is an input-only parameter for `grad`. This is the closure
  form of `in_axes=None` and needs no marker. Unlike `grad`, `vmap` also accepts a
  captured `DTensor` (a weight matrix is the usual case); synthesis takes its Kotlin type
  from the declaration it binds. `grad` still refuses captured tensors, unchanged.
- Only the leading axis can be batched (JAX's `in_axes=0` or `None`). Batching another
  axis means transposing it to the front first. This is a limitation of the first
  version; the type encoding would need an overload per position.

### Outputs (`out_axes`)

Every output is batched along a new leading axis (JAX's `out_axes=0`). An output that
does not depend on a batched input is broadcast along the batch axis, so the result
type is the same whatever the body reads. Multiple outputs (a `Pair` from
`valueAndGrad` inside `vmap`) are not supported in the first version: the lambda
returns one tensor or one scalar.

## The transform

`DxirVmapTransform.apply(fn, batched: List<Boolean>, batchSize: Int, batchName: String?)`
in `:ir` (`io.tlaloc.ir.passes`). It rewrites a per-example function into a batched one:

- Each param with `batched[i]` gets type `[batchSize] + dims`, axis names
  `[batchName] + names` (unnamed axes stay `null`). The others keep their type.
  `batchSize` is `-1` for plugin-lowered functions and concrete elsewhere.
- Every node is tracked as `(value, isBatched)`. A batched value has the batch axis at
  position 0. An op whose operands are all unbatched is copied unchanged, so
  computation that does not depend on the batch runs once, not once per example.
- Every return is batched; an unbatched return is materialized.

It is dtype-generic: the dtype comes from the node types, and every constant it creates
is built in the node's dtype (a `Double` for F64).

### Materializing an unbatched value

Some rules need an unbatched operand `u` (shape `s`) as a batched one (shape `[B] + s`):

- Concrete dims: `BROADCAST(u) {broadcast_dimensions = [1..r]} : [B] + s`.
- `-1` dims: synthesized code learns `B` only from a batched runtime value. The transform
  keeps one batched zero vector `z = SUM(BROADCAST(0, p), dims = [1..r_p]) : [B]` built
  from the first batched param `p` (just `BROADCAST(0, p)` when `p` is per-example
  scalar), and materializes `u` as `ADD(RESHAPE(z) : [B, 1, ..., 1], u)`, which the
  right-aligned broadcasting of `ADD` expands to `[B] + s`. The one value that changes is
  `-0.0`, which becomes `+0.0`.

Elementwise binaries avoid materializing: they broadcast an unbatched operand
implicitly, and `x · W` shares an unbatched rank-2 `W`. `CONCAT`, `POW`, `COMPARE`,
`WHERE`, the linear algebra and the other `MATMUL` combinations materialize theirs.

### Batching rules

`r` is the per-example rank of the operand, and axis attributes are per-example, so a
batched op shifts them by one.

| Op kinds | Rule |
|---|---|
| Elementwise unary: `NEG ABS EXP LOG SQRT RSQRT TANH SIGMOID RELU GELU SILU STEP SIN COS TAN ATAN LGAMMA DIGAMMA TRIGAMMA POLYGAMMA SIGN NOT CAST` | Same op on the batched operand; attrs unchanged |
| Elementwise binary: `ADD SUB MUL DIV LAND` | A batched operand whose per-example rank is below the result's gets unit axes inserted after the batch axis (`RESHAPE`); an unbatched operand is left as is and broadcasts right-aligned, except an unbatched scalar, which is splatted against the batched operand (synthesized code multiplies a tensor by a scalar only after a splat) |
| `POW COMPARE WHERE` | Every operand must have the result's per-example shape (refused otherwise); unbatched operands are materialized (their host twins do not broadcast); a constant with `-1` extents becomes a scalar splat against a batched operand |
| `BROADCAST` | Batched value: `broadcast_dimensions` become `[0] + (d + 1)`. Template form: the template is materialized if the value is batched and it is not; an unbatched value with a batched template keeps the batched template |
| Reductions: `SUM MEAN MAX MIN` | `reduction_dims` shift by one; an empty list (all axes) becomes `[1..r]`; a per-example scalar is its own reduction |
| `SOFTMAX LOGSUMEXP ARGMAX` | `axis` shifts by one (a negative axis is normalized first); a per-example scalar operand is refused (its axis would be the batch axis) |
| `TRANSPOSE` | `permutation` becomes `[0] + (p + 1)` |
| `RESHAPE` | Target `[B] + target`, with attribute `leading_kept` (the number of leading axes the reshape leaves alone, 1 per vmap). Under `-1` extents a batched flatten `[B, -1, -1] -> [B, -1]` cannot be told from other reshapes of the same ranks without it; synthesis calls the host twin `flattenFrom` |
| `REVERSE` | `dimensions` shift by one |
| `SLICE` | A full slice of the batch axis is prepended to `start_indices` / `limit_indices` / `strides`; `slice_axis` shifts by one |
| `PAD` | A zero pad is prepended to `low` / `high` / `interior` |
| `CONCAT` | All operands materialized, `dimension` shifts by one |
| `MATMUL` (canonical, no dimension attrs) | A batched lhs against an unbatched rank-2 rhs (`x · W`, W shared): a `MATMUL` of a batched lhs and the rank-2 rhs, NumPy `matmul` semantics, so W is not copied per example (interpreters, emitter as one `dot_general`, host twin `matmulSharedRhs`); its reverse rule gives `W̄` as one `[k, n]` product by folding the leading axes into the rows (`RESHAPE` with `merge_leading`, host twin `mergeLeading`), so `grad { vmap { } }` makes no per-example copy of `W̄` either. Every other combination: both operands batched (an unbatched one materialized), then the canonical batched matmul one rank higher (`matmulBatched`). An inner vmap's shared rhs stays shared under an outer vmap, and is refused if the outer one batches it |
| `MATMUL` with `lhs/rhs_contracting_dims` (`contract`) | Refused in the first version |
| `DOT` (rank 1 x rank 1) | `SUM(MUL(a, b), dims = [1])` |
| Runtime-extent ops: `SUM_TO BROADCAST_LIKE PAD_TO SLICE_AT SLICE_LIKE PAD_LIKE CHECK_SHAPE_LIKE ZEROS_LIKE` | Value and templates are batched together (unbatched ones materialized). `SUM_TO` and `BROADCAST_LIKE` align right, so a template of lower rank than the value gets unit axes after the batch axis and the result drops them; `low` / `axis` attrs shift by one. These appear in gradients, so `vmap { grad { } }` needs them |
| `EMBEDDING` | Batched indices with an unbatched table: indices `[B, N]` give `[B, N, D]` (the op already takes rank-2 indices). A batched table is refused |
| `CHOLESKY TRIANGULAR_SOLVE TRIANGLE` | Batched along leading dimensions, which `stablehlo.cholesky` and `stablehlo.triangular_solve` take natively (TRIANGLE is an iota mask over the last two axes). Interpreter, emitter, synthesis (host twins `choleskyBatched`, `triangularSolveBatched`, `scaleTrianglesBatched`) and the reverse and forward rules take leading batch dimensions; an unbatched `TRIANGULAR_SOLVE` operand is materialized. `solveSpd`, `logDetSpd` and `invSpd` are lowered to these ops, so they batch too |
| `SOLVE DET` (LU, lowered as a `stablehlo.while` loop) | Batched: the emitter's batched LU runs the loop once over all matrices (the column index is shared; the pivot row is per matrix, so rows move by one-hot selects); leading axes are flattened to one batch axis and restored. Host twins `solveBatched`, `detBatched`; the reverse and forward rules take leading axes (`DetRule`'s scaled identity spreads one scale per matrix) |
| `QR_Q QR_R EIGH_W EIGH_V` (Householder and Jacobi, `stablehlo.while` loops) | Batched the same way: one loop over all matrices, the column (QR) or the rotation pair (Jacobi) shared, each matrix's reflection or rotation its own; the final sort of eigenvalues and the sign normalization of eigenvectors run per matrix. Host twins `qrQBatched`, `qrRBatched`, `eighValuesBatched`, `eighVectorsBatched`; the reverse and forward rules take leading axes |
| `IF` | Unbatched condition: an `IF` whose yields are batched consistently. Batched condition: both branches are evaluated and selected with `WHERE`; under `-1` extents the predicate stays the 0/1 float mask the host uses (a batched `STEP` / `COMPARE` keeps its operand's dtype), since synthesis types mask tensors, not Bool ones |
| `WHILE` (a `for` loop) | Not batched as a loop: the IR phase first coarsens the body as it does for `jvp {}` (`PhiCalculus` closes or unrolls a loop with a constant trip count; a surviving `COARSENED` is inlined), then batches the straight-line result. A loop that does not coarsen away is refused. Only in a top-level `vmap` lambda: a lambda nested in another intrinsic must be straight-line |
| Constants and params | A constant is unbatched |

Refused by name (compile error `VMAP_NO_BATCHING_RULE` at the call, naming the op kind):
`CONV2D CONV_TRANSPOSE2D` and their adjoints, `MAXPOOL2D AVGPOOL2D` and their gradients,
`GATHER SCATTER SCATTER_ADD`, `EMBEDDING_GRAD`, `SPARSE_MATMUL`, `SPARSE_MATMUL_VALUES_ADJOINT`,
`RNG_UNIFORM RNG_NORMAL`, `CROSS_ENTROPY`, `LAYERNORM RMSNORM BATCHNORM`,
`SCALED_DOT_PRODUCT_ATTENTION PAGED_ATTENTION KV_CACHE_WRITE DEQUANTIZE_KV MOSAIC_KERNEL`,
`WHILE` (after coarsening), `COARSENED`, `SHARD_CONSTRAINT MANUAL_COMPUTATION ALL_REDUCE ALL_GATHER REDUCE_SCATTER`,
and `CROSS_ENTROPY`.
Several of these have no synthesis arm either, so a `grad {}` body cannot contain them
today. There is no sequential fallback: an op without a rule is an error, never a loop
over examples. An op whose operands are all unbatched is copied unchanged whatever its
kind, since it does not depend on the example: an `RNG_UNIFORM` draw with a literal key
gives every example the same numbers, as a loop over examples would.

### Where the rule table lives

The transform's `when` over `OpKind` is the rule table; an `OpKind` not listed above hits
its `else` branch, which throws `VmapUnsupportedException(kind, detail)`. The FIR checker
runs the transform on the lowered lambda (as it probes the reverse transform for `grad`)
and reports the exception as `VMAP_NO_BATCHING_RULE`, so the error appears when the file
compiles, at the call.

## Composition

### Nested intrinsic calls

`FirLambdaToDxirLowering` learns to lower a transformation intrinsic applied inside a
lambda, in two spellings:

```kotlin
grad { w: W -> loss(w, x) }(w0)                   // applied in place
val g = grad { w: W -> loss(w, x) }; g(w0)        // bound to a val, then applied
```

The inner lambda is lowered in the same lowering context, transformed at once (reverse
for `grad`, forward for `jvp`, `DxirVmapTransform` for `vmap`/`vmap2`) and its body inlined
into the outer function with its parameters bound to the argument nodes. Supported inner
intrinsics: `grad`, `jvp`, `vmap`, `vmap2`, with a straight-line body. Others (`grad2`,
`hessian`, `jacobian`, `valueAnd*`, `vjp`) and a loop or branch in the inner lambda
refuse by name in the first version. A `grad` lambda that applies a nested intrinsic may
capture a runtime tensor (the batch a nested `vmap` maps over); `jvp` still cannot carry
a captured value.

A value the inner lambda reads from the outer lambda (an outer parameter or local) is
lowered as a trailing input of the inner function and bound to the outer node. For
`grad` it is input-only (no gradient), for `jvp` its tangent is zero, for `vmap` it is
broadcast. The FIR checker skips a nested intrinsic call (one whose enclosing
`callsOrAssignments` contain an intrinsic call): the outer call owns it, and reports any
failure inside it.

### vmap of grad: per-example gradients

```kotlin
val perExampleGrads = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Feat>, F32> ->
    grad { w: DTensor<Rank2<Feat, Out>, F32> -> loss(w, x) }(w0)   // w0 captured: broadcast
}
```

The inner gradient function (w and x both inputs, x input-only) is inlined, then the
vmap transform batches it: `x` is batched, `w0` is not. The gradient program contains
runtime-extent ops, which is why they have batching rules. The result has type
`[B] + shape(w)`: one gradient per example.

### grad of vmap

```kotlin
val g = grad { w: DTensor<Rank2<Feat, Out>, F32> ->
    vmap(batchAxis(Batch)) { x: DTensor<Rank1<Feat>, F32> -> loss(w, x) }(xs).sum().toFloat()
}
```

The vmap transform batches the primal (`w` broadcast, `xs` batched), then the reverse
transform differentiates the batched program. This needs reverse rules for the ops the
rules above produce: rank-mismatched `MATMUL` gets one, and every other op they emit
already has one.

### jvp and vmap

`vmap { jvp { } }` inlines the forward program then batches it. `jvp { vmap { } }`
batches the primal then runs the forward transform. The rules above produce no op
without a forward rule.

### Nested vmap

`vmap(batchAxis(A)) { xs -> vmap(batchAxis(B)) { x -> f(x) }(xs) }`: the inner call is
batched over `B` (its axis 0) and inlined; the outer transform prepends `A`, giving
`[A, B, ...]`. The two names must differ (`VMAP_AXIS_NAME_CLASH`).

### Order matters in exactly one way

`vmap { grad { } }` differentiates each example separately. `grad { vmap { }.sum() }`
differentiates the sum over the batch, which equals the sum of the per-example
gradients. Both are certified against loops over examples.

One difference: an `if` whose condition depends on the example is batched as a `WHERE`
over both branches. Under `grad { vmap { } }` the unselected branch is differentiated
too, with a zero cotangent, so a branch whose derivative is infinite or NaN where it is not
selected (`sqrt` of a negative) makes the gradient NaN. The unbatched `grad` differentiates
only the taken branch, so a loop of per-example gradients stays finite. JAX's `vmap` has the
same behaviour; the remedy is the same too: keep the unselected branch's input in its
domain (`sqrt(where(c, x, 1))`). `vmap { grad { } }` is not affected: the per-example
gradient's branches are selected after they are differentiated.

## Readable source

`toKotlinSource()` prints a batched function with concrete extents as it prints a
gradient: one `val` per op over the `:core` host twins, the batched ops through the new
twins (`matmulBatched`, `choleskyBatched`, `triangularSolveBatched`,
`scaleTrianglesBatched`, the broadcasting binaries), with
`@file:OptIn(ExperimentalTlalocApi::class)` added when one of them is used. The printed
source compiles without the plugin and returns the interpreter's bits
(`VmapReadableSourceTest`). Per-example gradients of `sum(tanh(x · w))`, `x` batched
(`vmap` of the reverse transform, batch 4):

```kotlin
fun loss_grad_vmap(w: DTensor<Rank2<Sym, Sym>, F32>, x: DTensor<Rank3<Sym, Sym, Sym>, F32>): DTensor<Rank3<Sym, Sym, Sym>, F32> {
    val v2: DTensor<Rank3<Sym, Sym, Sym>, F32> = reshapeToRank3(matmulSharedRhs<Shape>(x, w), 4, 1, 2) // %2 = MATMUL(%1, %0)
    val v3: DTensor<ScalarShape, F32> = Tensors.f32Scalar(1.0f) // %3 = const : f32
    val v4: DTensor<Rank2<Sym, Sym>, F32> = stretchToRank2(v3.reshape(1, 1), 1, 2) // %4 = BROADCAST(%3)
    val v5: DTensor<Rank3<Sym, Sym, Sym>, F32> = v2.tanh() // %5 = TANH(%2)
    val v6: DTensor<Rank3<Sym, Sym, Sym>, F32> = (v5 * v5) // %6 = MUL(%5, %5)
    val v7: DTensor<Rank2<Sym, Sym>, F32> = broadcastDims(1.0f, intArrayOf(1, 2)) // %7 = const : f32[1,2]
    val v8: DTensor<Rank3<Sym, Sym, Sym>, F32> = reshapeToRank3(minusBroadcast<Shape>(v7, v6), 4, 1, 2) // %8 = SUB(%7, %6)
    val v9: DTensor<Rank3<Sym, Sym, Sym>, F32> = reshapeToRank3(timesBroadcast<Shape>(v4, v8), 4, 1, 2) // %9 = MUL(%4, %8)
    val v10: DTensor<Rank3<Sym, Sym, Sym>, F32> = transposePerm3(x, 0, 2, 1) // %10 = TRANSPOSE(%1)
    val v11: DTensor<Rank3<Sym, Sym, Sym>, F32> = reshapeToRank3(matmulBatched<Shape>(v10, v9), 4, 3, 2) // %11 = MATMUL(%10, %9)
    return v11
}
```

The shared weight `w` stays one matrix (`v2`); the per-example gradients `x_bᵀ · ḡ_b` are
one batched matmul (`v11`); the gradient's `1 − tanh²` stays per-example-shaped (`v7`) and
broadcasts against the batch.

`dumpGradSource` covers `vmap {}` calls. A plugin-lowered tensor lambda has `-1`
extents, which the printer refuses by name, so the dump prints that refusal, as it does
for tensor `grad {}` lambdas.

## Verification

- **The core property**, a reusable helper: for a per-example `DxirFunction` `f`,
  `vmap(f)(xs)` equals `stack_i f(x_i)`, on `DxirInterpreter` (F32 and F64) and through
  PJRT, at batch sizes 1, 7 and 64, for every op with a rule.
- Plugin: vmapped lambdas compile, and their results equal a loop over examples of the
  unbatched lambda, also at batch sizes 1, 7 and 64 and with a `Bounded` batch axis at
  several sizes.
- Per-example gradients equal a loop of single-example `grad` calls; `grad` of a vmapped
  loss equals the gradient of the loop version; `jvp` in both orders; nested vmap.
- Negative compilation: a batch-axis mismatch (Kotlin's argument type mismatch) and an op
  without a rule (`VMAP_NO_BATCHING_RULE`) at the call's file, line and column.
- JAX parity against `jax.vmap` if `harness/python` has JAX; skipped by name otherwise.

## Not in this run

- Batching a non-leading axis (`in_axes=1`), `out_axes` other than 0, several outputs.
- vmap in the Tracer-capture API and `:nn`; `boundedProgram` export of a vmapped function.
- Batching rules for convolution, pooling, gathers and scatters, RNG, sparse products,
  attention and serving ops, `WHILE`, collectives; `contract` with explicit dimensions.
- `hessian`, `jacobian`, `vjp` and `valueAnd*` as nested (inner) intrinsics.
- IREE execution of batched programs.
