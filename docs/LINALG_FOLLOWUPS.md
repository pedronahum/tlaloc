# Linear algebra: open issues

Known gaps in the linear-algebra ops (`io.tlaloc.core.ops.cholesky`, `solve`, `det`,
`qrQ`/`qrR`, `eighValues`/`eighVectors`, … and `:nn`'s `rk4`), to address later. The
history of the work is in [work-log/linalg-progress.md](work-log/linalg-progress.md).

## 1. Review the template-`BROADCAST` IrType rule in synthesis

`DxirToIrSynthesis.deriveResultIrType` gives a two-operand `BROADCAST` whose template
(operand 1) has the result's rank the template's IrType. Before, such a node fell back
to the call's first tensor parameter's IrType, which is wrong when that parameter has
another rank (a vector of hyperparameters before a matrix: the GP example failed to
compile). Every emitter of a templated `BROADCAST` targets its template's own type, and
no existing test changed, but the rule applies to every gradient that splats or
stretches against a template.

- To do: a second review of the rule, and a test that a templated `BROADCAST` whose
  result shape differs from its template's is refused rather than mistyped.
- Regression test: `LinalgGradientTest`, "a GP likelihood over a vector of
  hyperparameters".

## 2. Settle the API names

`solveSpd`, `logDetSpd`, `invSpd`, `identityLike`, `qrQ`/`qrR` and
`eighValues`/`eighVectors`. QR and eigh are two functions each because the forward
transform and the IR synthesis handle single-result ops only; a body that uses both
factors computes the factorization twice. The host-only `qr()` and `eigh()` return a
`Pair`. Rename before the names reach a release, or add multi-result support and a
single differentiable `qr()` / `eigh()`.

## 3. `grad {}` over F64 tensors

`grad {}` differentiates F32 tensors only, for every op: synthesis accepts F32 tensors
(`isAcceptedTensorType`), the host tensor ops are F32, and the interpreter stores
values as `FloatArray`. The linear-algebra functions have F64 host overloads, and their
F64 derivative rules are checked by running F64 gradient graphs through
`PjrtSession.runOnF64` (`PjrtLinalgTest`). Making F64 differentiable is a change to the
synthesis, the host ops and the interpreter.

## 4. Performance of the LU, QR and eigh lowerings

StableHLO has no LU, QR or eigensolver, and no custom-call target was confirmed for
these, so the emitter writes each as a `stablehlo.while` loop: LU `n` steps of `O(n²)`,
QR `n` steps of `O(m²)`, Jacobi `20·n(n−1)/2` steps of `O(n)` work over the whole matrix.
Correct on the GB10, but far slower than vendor routines for large `n`. Options: a
confirmed XLA custom call per backend (cuSolver on CUDA), or blocked algorithms.
Nothing has been measured.

## 5. Named arguments in the conv2d and pooling FIR arms

The FIR argument list is in source order, so an arm that reads literal arguments by
position mis-folds named arguments given in another order. The linear-algebra arm had
this bug and now reads `resolvedArgumentMapping` by parameter name. The conv2d,
`convTranspose2d` and pooling arms in `FirLambdaToDxirLowering` read by position; their
comment says K2 unwraps named arguments without reordering them, which leaves source
order, so `x.conv2d(w, strideW = 2, strideH = 1, …)` is likely mis-folded. Not
reproduced yet: write the failing test first.

## 6. Smaller limitations

- `det`'s gradient is NaN at a singular matrix; JAX's cofactor-based rule is finite at
  rank `n − 1`. A `logAbsDet` would also be useful.
- `eighVectors`' derivative is infinite at a repeated eigenvalue (as in JAX). `eigh`
  runs a fixed 20 Jacobi sweeps rather than testing convergence. The sign rule for
  eigenvectors (largest-magnitude entry positive) is discontinuous where two entries of
  a column tie with opposite signs.
- Rank-2 only: no batch axes. Left-side triangular solves only. QR needs rows ≥ columns.
- `rk4` works on the capture API only; inside `grad {}` the integration loop is written
  in the lambda. Its reverse pass stores every step (no adjoint-ODE solve), and the
  captured integrator has not been run on PJRT.
- The new MLIR is validated by XLA through PJRT; `RoundTripTest` (the StableHLO
  reference parser) skips on a machine without `stablehlo-translate`.
