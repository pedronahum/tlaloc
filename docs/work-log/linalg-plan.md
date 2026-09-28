# Differentiable linear algebra: plan

Branch `feat/linalg`. Baseline at `c13e72f`: `./gradlew test --continue` green,
2,726 test results, 101 skipped, 0 failures. Known flakes before any change:
`LlamaDecoderPytorchBenchTest` (CPU timing under load) and `BGDHyperOptTest`.

## How an op travels today

A new differentiable op touches these places (the TAN/ATAN commit `eb84271` is the
template):

| Layer | File |
|---|---|
| IR node | `ir/.../OpKind.kt` |
| Reverse rule | `ir/.../passes/Vjp.kt` (`VjpRegistry`) |
| Forward rule | `ir/.../passes/DxirForwardTransform.kt` (`tangentOf`) |
| CPU interpreter | `ir/.../passes/DxirInterpreter.kt` |
| StableHLO | `stablehlo/.../StablehloEmitter.kt` |
| Cost model | `ir/.../recognizer/cost/CostModel.kt` |
| Public API + host twin | `core/.../ops/*.kt` |
| Plugin front end | `compiler-plugin/.../FirLambdaToDxirLowering.kt` (FQN → DXIR ops) |
| Plugin back end | `compiler-plugin/.../DxirToIrSynthesis.kt` (DXIR → calls to host twins, result IrType derivation) |
| Readable reverse | `ir/.../render/KotlinSourceRenderer.kt` |

`grad {}` in user code: the plugin lowers the lambda to DXIR, runs the reverse
transform, and synthesizes Kotlin IR that calls the `:core` host twin of every op in
the gradient body. So every op that can appear in a gradient body needs a host twin
and a synthesis arm, and a user-facing function can be differentiated only if the FIR
lowering knows its FQN.

## Constraints found while orienting

1. **F64 tensors do not go through `grad {}` for any op today.** Synthesis accepts
   F32 tensors only (`isAcceptedTensorType`), the host tensor ops are F32 only, and
   the interpreter stores every value as a `FloatArray` (its KDoc says F64 gets F32
   precision). Making F64 tensors differentiable is a cross-cutting change to every op
   and is not part of this work. What F64 gets here:
   - F64 host kernels for every forward op (`DTensor<Rank2<..>, F64>.cholesky()` etc.),
     computed in Double;
   - DXIR ops that are dtype-generic, emitted as `f64` StableHLO; the F64 gradient
     graphs (from `DxirReverseTransform`/`DxirForwardTransform`) run through
     `PjrtSession.runOnF64` on the GPU and are checked against central finite
     differences in F64 there. That is the F64 gradient certification.
   - F32 gradients (interpreter, plugin) are checked against central finite
     differences of the F64 host forward, so the reference is not itself F32-noisy.
2. StableHLO has `cholesky` and `triangular_solve`. XLA's CPU and GPU compilers
   expand both (they are HLO ops, not custom calls). I use nothing else from outside
   the op set, and emit no `custom_call`.
3. Rank-2 only (no batch dimensions). A batched operand is refused by name.

## Op design

Primitive DXIR ops (each with VJP, JVP, interpreter, StableHLO, host twin, synthesis):

- `CHOLESKY(A) → L`. Lower factor of `sym(A) = (A + Aᵀ)/2`, upper triangle zero. Reading
  the symmetrized input (JAX's `symmetrize_input=True`) makes the function defined on
  every square matrix near an SPD one, so its derivative is well-defined and finite
  differences in any entry agree with it. Non-SPD input gives NaN (XLA's convention;
  the interpreter matches).
  - JVP (Murray 2016): `L̇ = L·Φ(L⁻¹·sym(Ȧ)·L⁻ᵀ)`, Φ = lower triangle with the diagonal
    halved.
  - VJP (adjoint of the JVP): `Ā = sym(L⁻ᵀ·Φ(Lᵀ·L̄)·L⁻¹)`.
  - StableHLO: `(A + Aᵀ)·0.5`, `stablehlo.cholesky lower = true`, then the upper triangle
    masked to zero (StableHLO leaves it implementation-defined).
- `TRIANGULAR_SOLVE(A, B) → X`, attrs `lower`, `transpose_a`, `unit_diagonal`. Solves
  `op(A)·X = B`, left side only; reads only the named triangle of `A`.
  - VJP: `B̄ = op(A)⁻ᵀ·X̄` (one solve with the other transpose), `Ā = mask(−B̄·Xᵀ)` or
    `mask(−X·B̄ᵀ)` when `transpose_a`, `mask` keeping the triangle that was read
    (strict when `unit_diagonal`).
  - JVP: `Ẋ = op(A)⁻¹·(Ḃ − op(mask(Ȧ))·X)`.
- `TRIANGLE(A)`, attrs `lower`, `diagonal`, `upper` (floats): scales the strictly-lower
  part, the diagonal and the strictly-upper part. Linear and self-adjoint, so its VJP
  and JVP are itself. It is `tril`/`triu` for users, and Φ and the triangle masks for
  the rules above.

Composites lowered in the FIR from those primitives (their derivatives come from the
primitives, so `hessian` works without new rules):

- `solveSpd(A, B)`: `L = cholesky(A)`, `Y = L⁻¹B`, `X = L⁻ᵀY`.
- `logDetSpd(A)`: `2·Σ log(diag(L))`, the diagonal taken as the row sum of
  `TRIANGLE(L, 0, 1, 0)`. Its gradient through Murray's rule is `A⁻¹` (tested).
- `invSpd(A)`: `solveSpd(A, I)`, with `I = TRIANGLE(ones like A, 0, 1, 0)`.

## Order

Dependencies put `triangularSolve` first: Cholesky's rules call it.

1. `TRIANGLE` + `triangularSolve` (plus `tril`/`triu`), one commit.
2. `cholesky`.
3. `solveSpd`.
4. `logDetSpd` (with the `hessian` test).
5. `invSpd`.
6. `examples/gaussian-process`.
7. Tier 2: general `solve`/`det` via LU with partial pivoting, then `qr`.
   StableHLO has no LU or QR; the emitter would write them as `stablehlo.while` loops.
   The derivative of `solve` is by implicit differentiation, so `SOLVE` is a primitive
   whose VJP calls `SOLVE` with the transpose; nothing differentiates the LU loop.
8. Tier 3: `eigh`, RK4.

## Verification per op

- Forward: known matrices, a 1×1 case, an ill-conditioned but valid case (Hilbert).
- Reverse and forward rules at IR level against central finite differences of the F64
  host forward; the F64 gradient graphs on PJRT against F64 finite differences.
- Through the real K2 plugin: `grad`, `jvp`, and `hessian` for `logDetSpd`.
- Interpreter vs StableHLO on the GB10 (skips by name without a GPU).
- JAX goldens (generated once with jax 0.10.0 on the CPU, script recorded) for forward
  values and gradients.
