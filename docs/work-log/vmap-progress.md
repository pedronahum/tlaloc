# vmap progress log

Design: [../design/vmap.md](../design/vmap.md). Branch `feat/vmap` from `main` at `7c05785`.

## Status

| Step | State |
|---|---|
| Phase 1: investigation + design | done |
| 1. Elementwise + broadcasting, one batched arg, interpreter | in progress |
| 2. Reductions, shape ops, matmul, softmax, losses | |
| 3. StableHLO + PJRT | |
| 4. Composition: vmap{grad}, grad{vmap}, jvp, nested vmap | |
| 5. Mixed batched / broadcast arguments | |
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
- **Unbatched matmul operand.** Rank-mismatched `MATMUL` with NumPy semantics instead of
  copying the weight per example.
- **Nested intrinsics** are lowered and transformed during FIR lowering and inlined, so
  `vmap { grad { } }` batches a gradient program and `grad { vmap { } }` differentiates a
  batched one.

## Open problems

## Next step

Write `DxirVmapTransform` with the elementwise and broadcasting rules and the reusable
"vmap equals stacking" test helper in `:ir`.
