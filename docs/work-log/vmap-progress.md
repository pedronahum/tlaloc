# vmap progress log

Design: [../design/vmap.md](../design/vmap.md). Branch `feat/vmap` from `main` at `7c05785`.

## Status

| Step | State |
|---|---|
| Phase 1: investigation + design | done |
| 1. Elementwise + broadcasting, one batched arg, interpreter | done, 738d0bb |
| 2. Reductions, shape ops, matmul, softmax, losses | done, 738d0bb (interpreter); plugin in the next commit |
| 3. StableHLO + PJRT | done, db52e68 (GB10) |
| Plugin surface: `vmap`, `vmap2`, `batchAxis`, diagnostics | done (commit after db52e68) |
| 4. Composition: vmap{grad}, grad{vmap}, jvp, nested vmap | done (commit after a9d7de8) |
| 5. Mixed batched / broadcast arguments | done with the plugin surface: `vmap2` markers, captured values (tensors too) |
| 6. Linear algebra | cholesky, triangularSolve, TRIANGLE, and so solveSpd, logDetSpd, invSpd: done (commit after 488107d). solve, det, qr, eigh (while-lowered): refused by name |
| 7. Readable source | done (commit after b3937db): concrete-dim batched functions print and recompile bit-identically; `dumpGradSource` covers vmap (tensor lambdas print the sentinel refusal) |
| 8. examples/per-example-gradients | done (commit after 81d396f): run against this checkout published to a scratch Maven repo (`-Dmaven.repo.local`), exit 0 |

## Decisions

- **Batch axis type.** `Named<N, A>` prepended, `A` = `Sym` or `Bounded<B>`, chosen with
  `batchAxis(name)` / `batchAxis(name, bound)`. The mapping from per-example to batched
  type is a generated set of overloads resolved with `@OverloadResolutionByLambdaReturnType`.
  A prototype on Kotlin 2.4.20 (scratch compile, not in the repo) resolved rank-1 → rank-1,
  rank-1 → scalar tensor, rank-1 → Float, and rejected a `Named<Time, _>` argument for a
  `Batch` function with Kotlin's argument type mismatch.
- **Rank-erased outputs.** Many host ops return `DTensor<Shape, D>`; such an output stays
  erased after batching (an extra overload), because the rank cannot be recovered.
- **Batching is a DXIR pass** (`DxirVmapTransform`), like the forward and reverse
  transforms, so the interpreter, the emitter and synthesis all see ordinary DXIR ops.
  Nothing new at run time except host twins synthesis needs (batched matmul).
- **`-1` dimensions.** Synthesized code gets the batch extent only from a batched runtime
  value, so materializing an unbatched value goes through a zero vector built from the
  first batched param and an implicit-broadcast `ADD` (design doc, "Materializing").
- **Nested intrinsics** are lowered and transformed during FIR lowering and inlined, so
  `vmap { grad { } }` batches a gradient program and `grad { vmap { } }` differentiates a
  batched one.
- **Unbatched matmul operand, revised.** Materialized (one copy per example) and the
  canonical batched `MATMUL` used, instead of a new rank-mismatched `MATMUL`: the canonical
  form already has interpreter, emitter and reverse-rule support, the mixed form would have
  touched six components. Follow-up if the copy costs.
- **No axis names in batched DXIR types.** The emitter infers a `MATMUL` contraction from a
  shared axis name when no dimension attrs are given; a batch name on both operands would
  be taken for one. The name lives only in the Kotlin type.
- **Tensor captures, `vmap` only.** `FirLambdaToDxirLowering.lower` keeps its public
  signature; an internal overload admits a captured `DTensor` for `vmap`, and synthesis
  types it from the bound declaration. `grad` still refuses captured tensors.
- **Batched flatten.** `RESHAPE` gets `leading_kept`; synthesis maps a batched flatten to
  the new host twin `flattenFrom`. Batched matmul maps to `matmulBatched`. Both
  `@ExperimentalTlalocApi` in `:core`.
- **Tests compare exactly** except where the per-example loop calls a host twin that sums
  in another order than the ops it lowers to (`crossEntropyLoss`: 1 ulp, tolerance 1e-6).

- **Nested intrinsics.** `grad`, `jvp`, `vmap`, `vmap2` applied inside another intrinsic's
  lambda (in place, or through a local `val`) are lowered in the same lowering context
  (`lowerNestedLambda`): a value of the enclosing lambda becomes a trailing parameter of
  the nested function bound to that value; a value from outside both goes through the
  enclosing lowering's capture handling. The nested function is transformed at once and
  inlined. The FIR checker skips an intrinsic call whose `callsOrAssignments` contain
  another intrinsic call. A `grad` lambda that contains a nested intrinsic may capture a
  tensor (the batch); every other `grad` lambda refuses one as before.
- **Shaped constants.** Rules for `-1`-extent programs re-emit a tensor-typed constant
  (a splat whose extents synthesis infers by matching axes against the parameters) as a
  scalar splatted against a batched operand: batching made the axis match pick the wrong
  parameter (`vmap { jvp { tanh } }` failed with [1,2,3] vs [1,2,4]).

- **Batched linear algebra.** CHOLESKY, TRIANGULAR_SOLVE and TRIANGLE take leading batch
  axes everywhere: the interpreters (one kernel call per matrix; a rank-2 operand is the
  single call it was), the emitter (`stablehlo.cholesky` / `triangular_solve` take batch
  dims; TRIANGLE's iotas and CHOLESKY's symmetrizing transpose move to the last two axes),
  the reverse and forward rules (their `transpose2` / `matmul2` helpers swap and multiply the
  last two axes; the rank-2 path emits exactly the op it did), and synthesis (new host
  twins `choleskyBatched`, `triangularSolveBatched`, `scaleTrianglesBatched`). The SPD
  composites are lowered to these, so they batch with no rule of their own.

- **Forward rules' shaped constants (a pre-existing jvp bug, fixed).** `jvp2 { x, w ->
  (x matmul w).tanh().sum() }` over all-`Sym` rectangular matrices failed at run time
  (`[2, 3]` vs `[2, 4]`, no vmap involved): the forward rules for `tanh`, `sigmoid`, `tan`,
  `atan`, `pow`, `rsqrt` and the extremum mask emitted a shaped constant, and synthesis
  sized it by matching axes against the parameters. Under `-1` extents they now emit a
  scalar `BROADCAST` against a value of the right shape; concrete-extent programs keep the
  shaped constant (identical DXIR for every concrete-dim test). `JvpRectangularTest`
  checks jvp2 against ⟨∇f, d⟩ from grad2; `jvp { vmap { tanh } }` now passes too.
- **JAX parity** (`VmapJaxParityTest`, the JAX 0.10.0 in `~/.local/venvs/iree`, skipped by
  name without one): batched forward, vmap(grad), grad(vmap(mean)) and vmap(grad(logdet))
  equal jax.vmap / jax.grad at F64 to 1e-12.

## Open problems

- `jvp { }` cannot carry a captured runtime value, so `jvp { vmap { }(xs) }` with `xs`
  captured is refused by name; pass the batch as a parameter (`jvp2`).

- `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes` (a device timing
  floor, listed in CAPABILITIES as failing under load) fails in some full runs and in 1 of 3
  runs alone on this branch. Nothing on the branch touches KPTX or `:benchmarks`.

## Next step

All eight steps are in. Remaining, by value: JAX parity for vmap (if harness/python has
JAX); the forward rules' shaped constants (jvp { vmap { tanh } }); the while-lowered linear
algebra; a rank-mismatched MATMUL to avoid the per-example weight copy.
