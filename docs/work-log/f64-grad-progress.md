# F64 under `grad {}`: progress log

Plan: [f64-grad-plan.md](f64-grad-plan.md). Branch `feat/f64-grad` from `main` (`6934819`).

Baseline on `main`: 2,881 tests, 0 failures, 101 skipped
(`./gradlew cleanJvmTest cleanTest test --continue --no-build-cache`).

## F32 unchanged: how it is checked

A temporary patch (not committed; kept at
`$SCRATCH/f64check.patch` during the run and described here) wraps
`DxirToIrSynthesis.synthesise` so that, when `TLALOC_F64CHECK_DIR` is set, every call writes
one file named by the SHA-256 of its content: the DXIR handed to synthesis (ids renumbered),
its `toKotlinSource` rendering (or the refusal), and `dump()` of the synthesized Kotlin IR.
Running `:compiler-plugin:test` with it on `main` gave 343 distinct files (337 with
synthesized IR). The same run on the branch must produce the same 343 files for F32 (new F64
tests add files; none may disappear or change).

## Log

### Step 1 — scalar `Double` (done)

Scalar `Double` lambdas already lowered at F64 end to end (literals keep their kind,
synthesis emits `kotlin.Double` arithmetic, the alpha01 fold fixes hold). Added
`F64ScalarGradientTest`: `grad`, `grad2`, `valueAndGrad`, `valueAndGrad2`, `jvp`,
`valueAndJvp`, `vjp`, exp/log/sin/cos/tanh/sqrt, `customVjp`/`customJvp`, all within 1e-13
of analytic Double derivatives, and a hidden-round-trip test (input `1 + 1e-10`, literal
`0.1`, captured `0.3` and `1e-9`, each of which F32 rounding moves by more than the
tolerance). `F64TestHarness` is shared by the F64 plugin tests. `jvp` takes no captured
runtime value, for any dtype (existing limitation); the test uses a `const val` there.

Suite: 2,884, 0 failures.

### Step 2 — F64 tensors on the host path (done)

- `:core`: `HostOpsF64.kt` and `BroadcastOpsF64.kt`, generated from `HostOps.kt` and
  `BroadcastOps.kt` (every `F32`/`Float`/`FloatArray` spelled at F64, literals widened) and
  trimmed of conv, pooling, batchNorm, embedding, RNG and sparse ops, which stay F32. 152
  public functions. `Tensors.f64Scalar/f64Vector/f64Matrix/f64MatrixOf/f64Zeros/f64Tensor3/
  f64Tensor4`, `hostF64()`. ABI baseline: additions only.
- Synthesis: `hostFunctions` replaces all 45 `referenceFunctions` lookups and drops the host
  twins of the other float dtype (tag = dtype argument of the first `DTensor<_, F32|F64>`
  receiver, parameter or result; untagged functions kept). `tensorDtype` is set per function
  by `floatTensorDtypeOf`, which refuses a function holding both F32 and F64 tensors by name.
  The per-op `dtype != F32` gates read `tensorDtype`; tensor constants, TRIANGLE scales,
  `toFloat`→`toDouble`, GATHER and DET results use `Double` in an F64 function
  (`exactDouble` refuses a Float value there).
- FIR: `toDouble` bridge; `literalArgAt` keeps a literal next to an F64 tensor as a Double
  and refuses a Float literal there; scalar×tensor, comparisons, clip, linalg receivers and
  scale arguments, outerProduct accept F64.
- `DxirConst` refuses, at construction, an F64 constant holding a `Float` or `FloatArray`.
  It found one immediately: `where` lowered its predicate test against a `0.0f` splat
  whatever the dtype (fixed). No existing test builds such a constant.
- `DTYPE_MISMATCH` (new FIR checker `TlalocDtypeMixChecker`): a call that fails to resolve
  with F32 and F64 `DTensor` operands is reported at that call's line and column, naming
  the operation and both dtypes. Kotlin's own "none of the candidates" error stays.
- Tests: `F64TensorGradientTest` (7): elementwise + broadcasting + axis reductions;
  matmul + transpose + slice + softmax/logSoftmax; where/comparisons/relu/clip/pow;
  valueAndGrad/jvp/vjp; a hidden-round-trip test on tensors; DTYPE_MISMATCH location; the
  mixed-function refusal. Tolerance 1e-9 against fourth-order differences (1e-14 against
  analytic where available).

F32 check: 343 of 343 reference dumps identical; 21 new dumps, all F64. Suite: 2,891, 0 failures.

Found on the way, F32 as well (not changed): tensor `sin`/`cos` are not lowered under
`grad {}`; comparison masks on a rank-1 operand do not synthesize (`no IrType for body
node … bool[-1]`).

### Step 3 — StableHLO, PJRT, and an F64 reference interpreter (done)

- `DxirInterpreterF64` (`:ir`): `DxirInterpreter` generated at Double width (the RNG and
  KV-dequantize kernels stay F32 and are converted at their boundary). F32-typed nodes are
  rounded to F32 after each op, BF16 to BF16. `DxirInterpreter` itself is unchanged: it
  still computes F64 nodes at F32 precision, as its KDoc says, and existing tests rely on it
  for graphs with F64 nodes (HfMuseGlimmer's F64 RoPE tables), so it does not refuse them.
- StableHLO: a `DoubleArray` rank-N constant prints every digit (`denseFromDoubleArray`);
  scalar Doubles already did.
- `PjrtSession.runOnHost(fn, inputs: List<Any>)`: each param at its own dtype (FloatArray
  F32, DoubleArray F64, IntArray I32), returns likewise; a dtype/array mismatch is refused
  by name.
- Tests: `DxirInterpreterF64Test` (5; gradients against fourth-order differences at 1e-9,
  low-bit inputs and constants at 1e-15, F32 elementwise graphs bit-identical to
  `DxirInterpreter`, the F64-const guard); `F64ConstEmitTest`; `PjrtF64GradTest` (3, GB10):
  the elementwise and matmul/softmax gradient graphs and the forward-over-reverse HVP agree
  with `DxirInterpreterF64` within 9e-16 of the largest entry (tolerance 1e-12), constants
  reach the device with every digit, mixed F32/F64 params through `runOnHost`.
  `F64TensorGradientTest` now also checks the compiled `grad {}` gradient against
  `DxirInterpreterF64` on the same graph (1e-13). So: host `grad {}` ≈ F64 interpreter ≈
  PJRT.

Suite: 2,900; one failure, `KptxPagedAttentionBenchTest`'s dispatch floor (timing, the
flake CAPABILITIES documents), which passed re-run alone.

### Step 4 — linear algebra under `grad {}` in F64 (done)

Steps 2 and 3 were enough for every linalg op: FIR accepts F64 receivers, synthesis picks
the `LinalgF64.kt` twin, TRIANGLE scales are Doubles. New here:

- **`jacobian`, `hessian`, `jacobianReverse` over F64** (`JacobianIntrinsicsF64.kt`,
  `:autograd`): overloads taking `(DTensor<S, F64>) -> R` and returning
  `DTensor<Rank2<Sym, Sym>, F64>`, which Kotlin picks over the generic ones for an F64
  tensor lambda; `assembleJacobianForwardF64`, `assembleHessianForwardF64`,
  `assembleJacobianReverseF64` assemble in Double. The existing signatures are unchanged
  (their F32 return type could not be generalized without breaking source). The plugin
  routes F64 params to the F64 helpers and refuses by name `jacobian2`/`hessian2`/
  `jacobianReverse2` over F64 and any of them over a `Double` scalar.
- **Parametrized existing tests.** `LinalgGradientTest` (plugin) and `LinalgJaxParityTest`
  (IR) run every test at F32 and at F64 (18 + 18). The F32 run is the original program,
  parsed and compared exactly as before (a printed Float widened to Double, which is what
  the old `Double − Float` comparison did); the F64 run is the same program through
  `F64Source.of` (a textual F32→F64 rewrite) with F64 tolerances: 1e-8 against the
  existing second-order differences (justified in the test), 1e-11 against the committed
  float64 JAX goldens (evaluated by `DxirInterpreterF64`). The F64 `det` Hessian is checked
  against the exact Hessian: the nested differences the F32 run uses are good to 1e-6 only.
  Mutation check: rounding the F64 host `cholesky` through Float fails 5 F64 tests and no
  F32 test.
- `F64HessianJaxTest`: `hessian { x.logDetSpd() }` at F64 through the plugin against live
  `jax.hessian` (x64, jax 0.10.0 in `~/.local/venvs/iree`), 1e-11; skips by name without a
  JAX Python.
- `F64TensorGradientTest`: F64 `jacobian`/`jacobianReverse`/`hessian` against analytic
  (1e-14), the `Double`-scalar refusal.

F32 check: 343 of 343 reference dumps identical. Suite: 2,921, 0 failures.

## Decisions

- **Host path = the `grad {}` interpreter.** Synthesized `grad {}` code calls
  `io.tlaloc.core.ops` host functions; F64 support means F64 twins of those functions.
  They live in separate files (`HostOpsF64.kt`, `BroadcastOpsF64.kt`) because each erases to
  its F32 twin's JVM signature and `@JvmName` is not available in `commonMain`, the same
  reason `LinalgF64.kt` exists.
- **PJRT leg.** The plugin's derivative DXIR cannot be recovered at run time (the canonical
  serializer refuses ops with attrs), so the PJRT checks run the `DxirReverseTransform`
  graph of the same program through the general `PjrtSession` entry, not `runOnF64`.

## Next step

Step 5: the remaining ops by likely scientific use: losses (crossEntropyLoss/nllLoss have
F64 twins; check them), the special functions, tensor sin/cos (not lowered at F32 either),
concat/stack/flip/pad, where F64 twins exist; conv/pool/batchNorm/embedding/RNG/sparse are
refused (no F64 host op, and the FIR arms refuse non-F32 by name). Then parametrize more
existing plugin tests with `F64Source`.
