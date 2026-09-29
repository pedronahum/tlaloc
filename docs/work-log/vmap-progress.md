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
| 6. Linear algebra | cholesky, triangularSolve, TRIANGLE, and so solveSpd, logDetSpd, invSpd: done (81d396f). solve and det (LU while loop), qr and eigh (Householder, Jacobi): done in later commits. Every linear-algebra op batches |
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
- **Unbatched matmul operand, revised twice.** First materialized (one copy per example) with
  the canonical batched `MATMUL`. Then, for `x · W` (batched lhs, unbatched rank-2 W, the
  per-example-gradient and mini-batch case): a `MATMUL` of a batched lhs and the rank-2 rhs
  (NumPy semantics) in the interpreters, the emitter (one `dot_general`), synthesis
  (`matmulSharedRhs`) and the renderer, with a reverse rule whose `W̄` folds the leading axes
  into the rows (`RESHAPE` + `merge_leading`, `mergeLeading`): no node of shape `[B] + W`
  in the batched program or its gradient (a structural test). Other combinations still copy.
  On the GB10 the F32 shared dot runs at XLA's default precision (TF32), like every F32
  MATMUL Tlaloc emits; `PjrtVmapTest` uses `TestBackend.defaultDotRelTolerance` for F32
  programs with a MATMUL (F64 stays at 1e-12).
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

- **Batched LU.** `emitLuBatched` is a second function beside `emitLu` (whose text is
  unchanged): the loop's column `k` is shared, the pivot row is per matrix, so the row swap
  gathers row `p` with a one-hot select and a sum (a `-0.0` there becomes `+0.0`) instead
  of a `dynamic_slice`. Leading axes are flattened to one and restored. On the GB10 it
  matches the interpreter to 1e-5 (F32) and 1e-12 (F64, two leading axes).

- **Emitter: a one-element input under the empty broadcast form (pre-existing, fixed).**
  The softmax adjoint un-reduces a single row `[1, 1] → [1, 10]` with an empty
  `broadcast_dimensions`; the interpreter splats it, the emitter threw ("length 0 must equal
  input rank 2"), so the gradient of a one-row softmax could not be emitted, with or without
  vmap. Equal rank now takes the identity mapping; a lower-rank one-element input is
  reshaped to a scalar and splat. Found by the benchmark's single-example loop.
- **Measured (GB10, `PjrtVmapBenchTest`, `TLALOC_VMAP_BENCH=1`).** Per-example gradients of
  W1 for `sum(softmax(tanh(x · W1) · W2) ⊙ y)`, W1 64×256, full-precision dots: one batched
  program against a loop that runs the single-example gradient program per example —
  batch 16: 1.35 ms vs 19.63 ms (14.5×); 64: 3.40 vs 46.49 ms (13.7×); 256: 8.72 vs
  242.04 ms (27.8×). After the shared-weight MATMUL, a second run: 16: 1.33 vs 12.68 ms
  (9.5×); 64: 2.74 vs 72.09 ms (26.3×); 256: 10.59 vs 249.42 ms (23.6×). The loop's time
  is mostly one dispatch and host round trip per example, which varies between runs
  (the loop at batch 64 took 46 ms in one run and 72 in the other); read the ratios as
  "10 to 28×", not as a precise figure.

- **Review of the branch (a read-only agent).** Fixed: `vmap2` with two batched arguments
  checks their batch sizes at run time (`checkBatchAxes`, inserted at the top of the
  synthesized lambda); SOFTMAX / LOGSUMEXP / ARGMAX of a per-example scalar are refused (the
  axis landed on the batch axis); a nested `jvp` gives an integer capture a typed zero
  tangent (it was `0.0f` typed I32); the batched host twins handle empty matrices. Documented:
  the batched-`if` NaN under `grad { vmap { } }` (design doc, "Order matters"), that a
  `Bounded` batch axis is a type and not a run-time check, and that an all-unbatched op is
  copied whatever its kind. Doc corrections: MATMUL materializes; POW/COMPARE/WHERE refuse
  mixed shapes; ARGMAX shifts `axis`.

- **Batched QR and eigh.** `emitQrBatched` / `emitEighBatched` beside the unchanged
  rank-2 emitters, the same pattern as LU. The reverse rules needed two broadcasting fixes
  beyond the rank checks: `eighF`'s eigenvalue row and `EighWRule`'s column scaling are
  reshaped to `[..., 1, n]` when batched (right-aligned broadcasting otherwise pairs them
  with the wrong axes). On the GB10 they match the interpreter to 1e-5 (F32) and 1e-12
  (F64, tall QR under two leading axes).

- **Loops and example-dependent `if` in a `vmap` lambda.** The IR phase's vmap branch
  coarsens a loop-bearing body first (PhiCalculus + region lift + decomposeCoarsened, the
  `jvp {}` pipeline; the FIR probe skips such bodies, as `grad`'s does), then batches the
  straight-line result: `for (i in 0 until 3) s = (s * x).tanh()` works, without Symja on
  the classpath. A batched `if` condition failed in synthesis (`no IrType for bool[-1]`): a
  batched STEP / COMPARE under `-1` extents now keeps its operand's float dtype (the 0/1 mask
  the host uses). Loops inside a nested intrinsic's lambda stay refused by name.

- **Second review (a read-only agent, commits 1991aa2..58fbb8f).** Fixed: the shared-rhs
  MATMUL reverse rule ran on MATMULs with dimension attributes (named `contract`); it is now
  for canonical ones only, and those keep their "matching ranks" refusal. A merge RESHAPE
  batched by an outer vmap kept merging from axis 0 on the host path (wrong extents for
  `vmap { grad { vmap { x · W } } }`): the RESHAPE now carries `merge_from`, which vmap
  shifts, and the host twin is `mergeAxes(x, from, count)` (was `mergeLeading`, unreleased).
  The interpreters' DET of empty batched matrices gives one per matrix. The batched LU's
  row gather sums with −0.0 so a row keeps its bits (−0.0 included). The cost model
  multiplies each linear-algebra op's flops by the number of matrices (rank-2 values
  unchanged). The review found the batched emitters faithful to the rank-2 algorithms and
  no change to existing rank-2 behaviour.

- **Named `contract`.** A MATMUL whose dimension attributes describe the canonical product
  (the plugin's lowering of `Rank2<M, K> contract Rank2<K, N>` and of the rank-3 batched
  contract) now batches as a canonical MATMUL (attributes dropped); other contractions stay
  refused by name.

## Open problems

- `jvp { }` cannot carry a captured runtime value, so `jvp { vmap { }(xs) }` with `xs`
  captured is refused by name; pass the batch as a parameter (`jvp2`).

- `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes` (a device timing
  floor, listed in CAPABILITIES as failing under load) fails in some full runs and in about
  1 of 3 runs alone on this branch; on `main` (a worktree at 7c05785) it failed 1 of 3 runs
  alone too. Nothing on the branch touches KPTX or `:benchmarks`.

## Next step

All eight steps are in. Remaining, by value: JAX parity for vmap (if harness/python has
JAX); the forward rules' shaped constants (jvp { vmap { tanh } }); the while-lowered linear
algebra; a rank-mismatched MATMUL to avoid the per-example weight copy.
