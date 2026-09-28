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
| `examples/gaussian-process` | done | 5 |
| Tier 2: `solve`, `det` (LU) | done | 6 |
| Tier 2: `qr` (`qrQ`, `qrR`) | done | 7 |
| Tier 3: `eigh` (`eighValues`, `eighVectors`) | done | 8 |
| Tier 3: RK4 integrator (`rk4`, `rk4Trajectory` in `:nn`) | done | 9 |

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

## Commit 5: `examples/gaussian-process`

The example first failed to compile: synthesis rejected a TRIANGLE whose operand, a
two-operand stretch BROADCAST in the log-determinant's adjoint, had no derived
IrType and fell back to the first tensor parameter's (rank 1, the hyperparameter
vector). Every earlier plugin test had a square matrix first, so the fallback was
right by accident. Fix: a same-rank two-operand BROADCAST takes its template's
IrType. While fixing it I found that commit 1 had inserted its CHOLESKY/TRIANGLE arms
into the middle of a multi-kind `when` branch list in two places (the unary list
continued into them); the behaviour was the same because the bodies were identical,
but the structure is now correct. Regression test: `LinalgGradientTest`'s GP
likelihood over a rank-1 first parameter (it failed before the fix).

The example resolves Tlaloc from mavenLocal like the others, but it needs this
checkout's build (the ops are not in `0.1.0-alpha02`); the README says so. I ran it
against a scratch repository (`-Dmaven.repo.local`), so `~/.m2` was not modified.

## Commit 6: `solve`, `det` (LU with partial pivoting)

`SOLVE(A, B)` (attr `transpose_a`) and `DET(A)` are primitives that factor internally;
no LU factor is ever a DXIR value, so no rule differentiates the factorization.
Rules: `B̄ = SOLVE(A, X̄, !t)`, `Ā = −B̄·Xᵀ` (or `−X·B̄ᵀ`); `Ẋ = SOLVE(A, Ḃ − op(Ȧ)·X)`;
`Ā = SOLVE(A, (d̄·d)·I, transposed)` with the identity from a splat against `A`;
`ḋ = d·tr(SOLVE(A, Ȧ))`. Closed under both transforms, so second order works (the
plugin's `hessian` of `det` is tested). One commit for both: they share the LU
lowering.

Kernel: `LinalgKernels.lu` (first row of largest |pivot|, zero pivot leaves zero
multipliers, as `getrf`). Emitter: `emitLu` writes a `stablehlo.while` over k carrying
`(k, A, perm, swaps)`; the pivot is a max reduction then a min reduction over the rows
attaining it (first maximum, same tie rule as the kernel), rows swapped with
`dynamic_slice`/`dynamic_update_slice`, elimination masked to the trailing block. SOLVE
applies P as a 0/1 matrix product at `precision = HIGHEST` (exact) and uses
`stablehlo.triangular_solve` on the packed factor. XLA accepted the loop on the first
run; f64 device results equal the kernel to 1e-13, and a permutation matrix, a singular
matrix (det exactly 0) and 1×1 are checked on the device.

Known limitations: O(n) sequential loop iterations on the GPU (documented in KDoc);
`det` gradient is NaN at a singular matrix (JAX's cofactor rule is finite at rank n−1).

## Commit 7: QR

Two single-result kinds, `QR_Q` and `QR_R`, not one two-result kind: the forward
transform refuses multi-result ops other than IF/COARSENED, and the synthesis builds
one value per node. Cost: a body using both factors runs the factorization twice.
Householder in LAPACK's convention (matches NumPy to 5e-16 and JAX's values, signs
included). VJPs: `QR_Q` gives `(Q̄ − Q·copyltu(Q̄ᵀQ))·R⁻ᵀ`, `QR_R` gives
`Q·copyltu(R·triu(R̄)ᵀ)·R⁻ᵀ`; they add up to the standard QR adjoint (I first wrote
`M = RᵀR̄ − Q̄ᵀQ`; the numpy check against finite differences showed it wrong and
`M = R·R̄ᵀ − Q̄ᵀQ` right before any Kotlin was written). JVP: JAX's `qr_jvp_rule`.
Mutation checks catch a wrong `copyltu`, a dropped sign, and a wrong `Ṙ`; a
mutation that kept the diagonal in Ω was equivalent (it cancels).

Limitations: rows ≥ columns only (a wide matrix is refused by name); full column
rank for derivatives (they divide by `R`).

## Commit 8: `eigh`

`EIGH_W` / `EIGH_V`, single-result for the QR reason. Algorithm: cyclic Jacobi on
`sym(A)` with a fixed `EIGH_SWEEPS = 20`, chosen so that the interpreter and the
StableHLO loop run exactly the same rotations (a convergence test would need an extra
reduction per sweep in the loop; a rotation with `A[p][q] = 0` is the identity, so the
surplus sweeps change nothing). Eigenvector signs: largest-magnitude entry positive,
so `V` is a continuous function of `A` where eigenvalues are distinct and finite
differences are meaningful; JAX/LAPACK signs are arbitrary, so the JAX parity test
compares `w`, `|V|`, and the gradient of `Σw³ + Σ(V⊙V)⊙W`, which is sign-blind.
Rules: JAX's JVP and its adjoint (checked in numpy before writing Kotlin: 1.7e-9).
`F = 1/(I + D) − I` (JAX's spelling) built from DXIR broadcasting (`w − wᵀ` as
`SUB([n], [n, 1])`). The degenerate-eigenvalue behaviour is pinned: at `sym(A) = I`
the eigenvalue gradient is finite and the eigenvector gradient is not.

Limitations: fixed sweeps (not adaptive; for large or badly scaled matrices 20 sweeps
may not converge to rounding); on the GPU `20·n(n−1)/2` sequential loop steps.

## Commit 9: RK4

`:nn/Ode.kt`. A reusable integrator cannot be called inside `grad { }`: the plugin lowers
only calls it knows by name, and a library function taking `f` as a lambda is not one
(an `inline` function is still a call at the FIR stage, where the plugin runs). So
`rk4` is built on the capture API (`Tracer`), the route `:nn` training uses: the
captured function is the unrolled integrator and both transforms differentiate it
with respect to `y0` and every leaf `f` reads. A host `DTensor` overload computes the
same steps. `grad { }` users write the loop in the lambda, which the plugin already
differentiates (`examples/differentiable-physics`).

Tests (`OdeTest`): the host result equals RK4's exact amplification factor
`R(−dt)ⁿ` for linear decay (which pins the coefficients) and its error ratio for
halved steps is RK4's 19.75; a captured damped oscillator's gradient with respect to
`M` and `y0`, and its tangent, against finite differences of a Double RK4; gradient
descent through the captured integrator recovers a damping coefficient (1.0 → 0.3).
First attempt of the fit used step 2, above 2/L″ ≈ 0.87, and diverged; step 0.5.
Mutating a coefficient in either overload fails the tests.

Not done: the captured integrator was not run on PJRT (its graph is MATMUL/ADD/MUL,
all GPU-certified already; `:runtime-pjrt` tests do not depend on `:nn`).

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

Docs: CAPABILITIES.md, README op table, summary at the top of this log.
