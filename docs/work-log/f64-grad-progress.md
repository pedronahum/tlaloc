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

Step 3: StableHLO emission of F64 rank-N constants at full precision; a `PjrtSession`
entry that takes each param at its own dtype; an F64 evaluation mode of `DxirInterpreter`
so interpreter and PJRT can be compared on the same gradient graph.
