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
| 4. Composition: vmap{grad}, grad{vmap}, jvp, nested vmap | next |
| 5. Mixed batched / broadcast arguments | done with the plugin surface: `vmap2` markers, captured values (tensors too) |
| 6. Linear algebra | |
| 7. Readable source | |
| 8. examples/per-example-gradients | |

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

## Open problems

- `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes` (a device timing
  floor, listed in CAPABILITIES as failing under load) fails in some full runs and in 1 of 3
  runs alone on this branch. Nothing on the branch touches KPTX or `:benchmarks`.

## Next step

Step 4: lower a nested intrinsic call (`grad { }(w)`, `vmap(..) { }(xs)`) inside another
intrinsic's lambda in `FirLambdaToDxirLowering`, transform it there and inline it; skip
the nested call in the FIR checker; batching rules for the runtime-extent ops gradients
contain (SUM_TO, BROADCAST_LIKE, PAD_TO, SLICE_AT, ...).
