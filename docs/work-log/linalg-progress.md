# Differentiable linear algebra: progress log

Plan: [linalg-plan.md](linalg-plan.md). Branch `feat/linalg`.

## Status

| Op | State | Commit |
|---|---|---|
| `TRIANGLE` (`tril`, `triu`, `scaleTriangles`) | done | 1 |
| `triangularSolve` | done | 1 |
| `cholesky` | done | 1 |
| `solveSpd` | done | 2 |
| `logDetSpd` | done | 3 |
| `invSpd`, `identityLike` | done | 4 |
| `examples/gaussian-process` | not started | |
| Tier 2: `solve`, `det` (LU), `qr` | not started | |
| Tier 3: `eigh`, RK4 | not started | |

## Commit 1: `TRIANGLE`, `triangularSolve`, `cholesky`

One commit, not three: Cholesky's reverse and forward rules are built from
triangular solves and triangle masks, and the triangular-solve rules from masks,
so none of the three is complete without the others below it.

Where each piece lives:

- Kernels (Double, shared by the host twins and the interpreter):
  `core/.../LinalgKernels.kt`.
- Public API: `core/.../ops/Linalg.kt` (F32) and `LinalgF64.kt` (F64; separate file
  because the overloads erase to the same JVM signatures). Capture API:
  `autograd/.../TracedLinalg.kt`.
- IR: `OpKind.CHOLESKY/TRIANGULAR_SOLVE/TRIANGLE`; rules in `Vjp.kt`
  (`CholeskyRule`, `TriangularSolveRule`, `TriangleRule`) and
  `DxirForwardTransform.tangentOf`; interpreter arms; cost model; Kotlin renderer.
- StableHLO: `emitCholesky` (symmetrize, `stablehlo.cholesky`, mask the upper
  triangle), `emitTriangularSolve`, `emitTriangle` (iota/compare/select).
- Plugin: `LINALG_OP_SET` arm in `FirLambdaToDxirLowering`; `irLinalg` plus IrType
  forward/backward derivation in `DxirToIrSynthesis`. The F32 host twin is picked
  by receiver dtype (`linalgF32Symbol`) since the F64 twin shares the name.

Tests (all pass):

- `ir`: `DxirLinalgGradTest` (10: forward values incl. 1×1, NaN on indefinite,
  Hilbert(6); VJP and JVP vs Double finite differences for every triangular-solve
  flag; forward-over-reverse through Cholesky), `LinalgJaxParityTest` (2).
- `core`: `LinalgTest` (7, F32 and F64 host, Hilbert(8)).
- `autograd`: `TracedLinalgTest` (capture API gradient vs finite differences).
- `compiler-plugin`: `LinalgGradientTest` (`grad`, `grad2`, `jvp` through K2),
  `PrintedGradientGoldenTest` (printed gradient compiles, bit-identical).
- `stablehlo`: `GradientEmissionCoverageTest` gains 10 cases.
- `runtime-pjrt`: `PjrtLinalgTest` (3, GB10): F32 device vs interpreter; F64
  gradient/tangent graphs vs F64 finite differences to 1e-7; Hilbert(5) F64.

Mutation checks (CONTRIBUTING: a new test must fail without the fix): changing Φ's
diagonal ½ → 1 in the Cholesky VJP or JVP, dropping the triangle mask or a
transpose flag in the triangular-solve rules — each makes `DxirLinalgGradTest` fail.

## Commit 2: `solveSpd`

An FIR composite (`cholesky`, `L⁻¹·B`, `L⁻ᵀ·Y`); host twins F32/F64 and a
capture-API twin built the same way. No new DXIR kind and no new rule: the
derivative is the primitives'. Tests: host (textbook system, 1×1, Hilbert(8)
residual), JAX `cho_solve` parity (value, dA, dB), plugin `grad2` and `jvp2`
against finite differences and against the implicit-differentiation formulas
`B̄ = A⁻¹·X̄`, `Ā = −sym(B̄·Xᵀ)`, F64 gradient graph on PJRT.

Decision: no dedicated SOLVE_SPD kind with a hand-written implicit rule. The
composite's reverse pass costs four triangular solves plus two small matmuls
against two for a direct rule, but gives forward mode and every higher order for
free and cannot disagree with the primitives.

## Commit 3: `logDetSpd`

FIR composite: `2·Σ log(rowsum(TRIANGLE(L, 0, 1, 0)))`, the doubling as `s + s` so
no constant is needed. The row sum reads the diagonal exactly (the rest of each row
is an exact zero). The host twin sums in Double; the `grad {}` value comes from F32
ops, and the two can differ in the last bit (documented in the KDoc). Tests: host
(textbook det 36, 1×1, Hilbert(8) against the exact rational determinant), JAX
value/gradient/full 16×16 Hessian at IR level, plugin `grad` (= sym(A)⁻¹ and finite
differences), `jvp`, `hessian` (against finite differences of the exact gradient),
capture API, F64 gradient and Hessian-vector product on PJRT.

## Commit 4: `identityLike`, `invSpd`

`identityLike` lowers to `TRIANGLE(BROADCAST(1, A), 0, 1, 0)`: the two-operand
BROADCAST takes `A` as a shape-only template, so the identity has `A`'s runtime
extents under `grad {}`'s symbolic dims. `invSpd` is `solveSpd(identityLike())`.
The capture API records the identity as a constant leaf. Tests: host (A·A⁻¹ = I,
1×1), JAX value and gradient, plugin `grad`/`jvp` of Σ(A⁻¹)³ and the zero derivative
of `identityLike`, device F32 against the interpreter and F64 gradient on PJRT.

## Decisions

- **F64.** `grad {}` handles F32 tensors only, for every op (`isAcceptedTensorType`),
  and the interpreter computes in F32. Making F64 differentiable is a separate,
  cross-cutting change. F64 here = host forward functions + F64 StableHLO gradient
  graphs checked on PJRT. Recorded in the plan.
- **Symmetrized input** for Cholesky (JAX's `symmetrize_input=True`), so the
  derivative is defined in every entry and finite differences agree with it.
- **Non-SPD input → all NaN**, what XLA returns; no exception (a `grad {}` body
  cannot throw by value).
- **No default arguments** on the new functions: the plugin reads literal flags by
  position (K2 does not reorder named arguments before the lowering runs; the conv2d
  arm documents the same constraint). `triangularSolve(b, lower)` and
  `triangularSolve(b, lower, transposeA, unitDiagonal)`.
- **Left-side solves only.** A right-side solve is a transpose away.
- **Captured tensors are not supported inside `grad {}`** (existing limitation, not
  new). Data must be passed as lambda parameters; affects the GP example.

## Open problems

- none yet

## Next step

`examples/gaussian-process`, run against a scratch Maven repository (`-Dmaven.repo.local`) so `~/.m2` is not touched.
