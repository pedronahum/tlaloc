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
(From the printer change on, 8 of the 343 are F64 functions whose dump changes; see below.)

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

### Step 5a — op coverage at F64 (done)

`F64OpCoverageTest`: 20 cases, each a loss written once as a plain function (F64 host ops,
forward only) and once inside `grad {}`; the compiled gradient against fourth-order
differences of the plain function (1e-8), and the F64 value against the same loss at F32
(1e-5). Covered: sigmoid, sqrt, log, neg, exp, pow (scalar and tensor exponent), tan, atan,
lgamma, digamma, polygamma, maximum, minimum, transpose(perm), reshape, flatten, squeeze,
unsqueeze, flip, broadcastTo, axis and full sum/mean/max/min, concat, stack, slice, view,
where with gt/ge/lt/le, crossEntropyLoss, nllLoss, softmax, logSoftmax, relu, sign,
outerProduct, get. All passed on the first run except a typo in the test and one F32 bug
(below).

Refused at F64 (no F64 host op, so the call does not resolve): conv2d, convTranspose2d,
avgPool2d, maxPool2d, batchNorm, embedding, sparse matmul, `contract`, RNG draws (F32).

Found, F32 as well (not changed): the `tanh` adjoint after a shape-changing `reshape`
splats its constant 1 over the parameter's shape, `elementwiseBroadcast: shapes [3, 4] and
[4, 3] are not broadcast-compatible` at run time (`grad { x: Rank2 -> (x.reshape(4, 3) *
x.reshape(4, 3).tanh()).sum() }`).

Suite: 2,922, 0 failures.

### Step 5b — conv, pooling, batchNorm, named contract at F64; named refusals (done)

- `ConvOpsF64.kt` (conv2d, convTranspose2d and their adjoints, avg/max pooling and their
  gradients, batchNorm) and `NamedOpsF64.kt` (`contract`, ranks 1–4), generated from the
  F32 files like the others. The conv engines already accumulated in Double. FIR: the
  conv/pool/batchNorm arms take F32 or F64, batchNorm's `eps` is a Double at F64.
- `F64OpCoverageTest` gains conv2d, convTranspose2d, avgPool2d + maxPool2d, batchNorm (24
  cases, all within 1e-8). Named `contract` at F64 (rank 2, `grad2`) against the analytic
  gradient in `F64TensorGradientTest`.
- `DTYPE_UNSUPPORTED`: `embedding`, `embeddingGrad` and the three sparse matmuls with F64
  operands are refused by name at the call. RNG draws are F32; using one in an F64
  expression is a `DTYPE_MISMATCH`.

Found, F32 as well (not changed): `DOT` (a rank-1 named `contract`) has no synthesis arm,
so `grad { v -> (v contract w) }` does not compile at any dtype; `transpose()` drops axis
names in the DXIR, so a transposed named operand cannot be contracted.

F32 check: 343 of 343 reference dumps identical. Suite: 2,924; one failure, the
`KptxPagedAttentionBenchTest` dispatch floor, which passed re-run alone.

### Step 6 — `examples/gaussian-process` in F64 (done)

Switched to F64 rather than adding a mode: a GP's kernel matrix gets ill-conditioned as the
noise shrinks, and double precision is the usual choice. Ran against this checkout
published to a scratch Maven repo (`-Dmaven.repo.local`): gradient against the Double
reference's differences 8.3e-11 of the largest entry (F32 measured 3.8e-7), same fit
(ℓ = 1.693, σf = 0.874, σn = 0.116), final |gradient| 7.7e-6 (F32: 5.6e-5). The check
tightened from 1e-3 to 1e-8. README and `examples/README.md` updated.

Suite: 2,924; `KptxPagedAttentionBenchTest`'s dispatch floor failed in the full run and in
a run of the benchmarks module (load average about 5 from other work on the machine), and
passed alone.

### Hidden F32 path in the loop closed forms (fixed)

`PhiCalculus` read the constant coefficients of the C7 (`d ← a·d + b[i]`) and C9
(`d ← a·dᵇ`) closed forms through `Float` before handing them to the symbolic engine as
Doubles. A `grad {}` over a Double loop that closes through C9 was off by 1.3e-7
(`grad { x: Double -> var d = x; for (i in 0 until 3) d = 0.9 * d.pow(1.5); d }` gave
0.8770684471477485 for 0.8770685575110583). The coefficients are now Doubles; for F32
programs, whose constants are Floats, the values the engine receives are unchanged (a
Float widened exactly), and the 343 F32 dumps are identical. Pinned by
`PhiCalculusF64Test` (IR, C7 and C9 at 1e-12; the C9 case fails without the fix, the C7
case is closed by an earlier corollary on that loop) and by a plugin test in
`F64ScalarGradientTest`. The sweep for other `toFloat()` in the IR passes found only
dtype-aware arms (`floatLiteralForDtype`, the F32 fold arm, `SymjaEngine`'s literal
lowering).

Suite: 2,927, 0 failures.

### Readable F64 gradients; bounded-dim checks for the F64 factories (done)

- `toKotlinSource` renders F64: `DTensor<…, F64>` types, `Tensors.f64*`/`broadcastDims`
  constants with every digit (`Double.toString`, shortest round-trip), TRIANGLE scales as
  Doubles, `ZEROS_LIKE` as `broadcastLike(0.0, …)`, Bool masks at the function's float dtype.
  A `grad { x: Double -> … }` gradient dumped by `dumpGradSource` now compiles without the
  plugin and gives the compiled gradient's bits (`F64ScalarGradientTest`); it was SKIPPED.
- `BOUNDED_DIM_EXCEEDED` reads a rank-1 factory's `doubleArrayOf(...)`/`DoubleArray(n)`
  length (rows/cols/`dN` were matched by parameter name already). New test in
  `BoundedDimCompileCheckTest`; it fails without the change.

F32 check: the 335 reference dumps of F32 functions are identical (printed source and
refusal text included). The other 8 reference dumps are scalar-`Double` functions whose
printed source replaced the old "dtype f64" refusal: expected. Suite: 2,929, 0 failures.

### 65 existing plugin test classes run at F32 and F64; two-argument assembly at F64 (done)

- A scripted edit (`compileAndRun` rewrites the stub and program through `F64Source.of` when
  the test's `precision` is F64; each `@Test` becomes an F32 run of the untouched body and an
  F64 run of the same body) applied to 64 files: the shape, reduction, broadcast, concat,
  slice, flip, softmax, loss, pow, where/compare, clip, conv, conv-transpose, pooling,
  batchNorm, CNN block, outer-product, matmul (square, rectangular, two-param), NN, special
  function, tan/atan, scalar exp/log/sin/cos/tanh/sigmoid/abs, customVjp, jvp/jvp2/jvpIf/
  jvpLoop, vjp, jacobian/hessian/jacobianReverse (1, 2 and 3 argument), Brachistochrone,
  Hookean spring, HMC (4), BGDHyperOpt, integral, view/withChange/meld, in-place stretch,
  literal-compare, sign, CartPole (4) and matmul-recognition tests; `ThreeArgIntrinsicTest`
  by hand. Each F64 run keeps the file's own assertions: they compare against exact or
  analytic values at the F32 tolerance, so these runs pin that every program compiles and
  computes the right numbers at F64; precision is pinned by the dedicated F64 tests. The F32
  runs are the original tests: the program text, parsing and tolerances are unchanged, and
  the reference dumps of F32 functions are identical.
- The one F64 run removed: `MultiIndexParamGradientTest` (it uses `embedding`, F32-only).
- `jacobian2`, `hessian2`, `jacobianReverse2` over F64 tensors: overloads and
  `assemble*2*F64` helpers (generated from the F32 ones); the plugin routes F64 to them. The
  refusal is now only for a `Double` scalar parameter.

F32 check: the 335 F32 reference dumps are identical. Suite: 3,061, 0 failures.

### Review fixes (done)

A read-only review of the branch by a subagent found, none affecting F32:
- `DTensor<F64> * DScalar` accepted a `FloatScalar` (an implicit F32→F64 promotion).
  The F64 overloads now take `DoubleScalar`. (The F32 `DScalar` overloads, which narrow a
  `DoubleScalar`, are older and unchanged.)
- `DTYPE_UNSUPPORTED` never fired for the sparse ops: K2 resolves a single-candidate call
  with mismatched arguments to `FirResolvedErrorReference`, a subtype of
  `FirResolvedNamedReference`, which the checker treated as resolved. Fixed, and the
  checker also names a `Float`/`FloatScalar` operand next to an F64 tensor
  (`DTYPE_MISMATCH`), only when an F64 tensor is involved, so F32 programs keep Kotlin's
  messages. Tested (`F64TensorGradientTest`).
- The renderer's mask dtype counted F64 scalars; it now follows the tensors as synthesis
  does, and the scalars only in a function without float tensors.
- Before the review, the renderer's per-function state moved from the shared `object` to
  one instance per call (two concurrent renders in one daemon could have raced).
- Cosmetic: a KDoc orphaned by an insertion, unused imports in the generated files.

Checked and fine by the same review: `tensorDtype` is set before any lookup and
`synthesise` is not re-entered; every lookup goes through `hostFunctions`; the F32 arms of
every changed FIR/synthesis branch are unchanged; `DxirInterpreterF64` differs from a
mechanical substitution of `DxirInterpreter` only in the intended snaps and conversions;
the generated host files have no F32-tuned epsilons or bit tricks.

Suite: 3,068, 0 failures. F32 dumps identical.

### PJRT at F64 for conv, pooling, special functions and linalg; hidden F32 constants in the emitter (fixed)

`PjrtF64OpsTest` (GB10): the graphs of the F32 smoke tests (conv, transposed conv, avg and
max pooling, flip, tan/atan, lgamma/digamma) and the linear-algebra kinds (cholesky, solve,
det, qrQ, qrR, eighValues), loss and gradient, through `runOnHost` against
`DxirInterpreterF64`, 1e-11. It found the avg-pool gradient 1.0e-8 off on the device: the
emitter wrote the gradient's 1/(window area) as `(1.0f / n).toString()`, `0.11111111`,
into an f64 graph. Fixed, and the two other places that printed a Float into a graph of
any dtype: the LayerNorm/RMSNorm epsilon and the attention scale 1/√dₖ. At F32 all three
print exactly what they did (asserted for the avg-pool gradient in `F64ConstEmitTest`);
at F64 they print the Double. After the fix every result agrees within 3e-16 of the largest
entry, 1.1e-13 for lgamma/digamma. `batch_norm_inference`'s epsilon stays an f32 attribute
(StableHLO requires it); paged attention is inference-only F32/BF16. (A first version of the
test used Σ Q² for QR, which is constant; both sides returned rounding noise near 1e-17.)

Suite: 3,071, 0 failures.

### Source compatibility and the other runtimes (checked)

- Every example (`differentiable-physics`, `fine-tune`, `gaussian-process`, `gpu-inference`,
  `gpu-training`, `java-inference`, `lora-finetune`, `mnist`, `named-indices`, `quickstart`,
  `readable-gradients`, `spark-inference`, the three `internals` projects) compiles against
  this branch published to a scratch Maven repo: the new same-named F64 overloads break no
  existing call.
- `runOnIree` and the one-shot `runOnPjrt` refuse a non-F32 param or return by name; F64
  graphs go through `PjrtSession.runOnHost` (or `runOnF64`).
- Cleaned the generated `HostOpsF64.kt`: comments copied from the F32 file that spoke of
  narrowing, and `x.toDouble()` on Doubles; added a header.

### Existing tests, unmodified (checked)

The branch changes 70 existing test files (the F32/F64 parametrization and three comment
edits). With `main`'s version of each of those 70 files put back (`git show main:<file>`),
`:compiler-plugin:test`, `:ir:jvmTest`, `:autograd:jvmTest` and `:runtime-pjrt:jvmTest` ran
1,781 tests against the branch's production code: 0 failures, 13 skipped (the usual
MLIR-tool and TPU skips). The branch's files were then restored. So every existing test
passes as written on `main`, and the F32 halves of the parametrized tests are the same
programs.

`BoundedProgram`, `capture` and the Tracer API refuse F64 by name (their specs and tapes
are F32/I32), so none of them computes an F64 input at F32 without saying so.

### Cost of F64 (measured once, not a test)

GB10, load average 2 to 3 from other work, medians of 7 after 3 warm-up calls:

| | F32 | F64 |
|---|---|---|
| host `matmul`, 128×128 | 0.47 ms | 0.48 ms |
| host `matmul`, 256×256 | 3.62 ms | 3.61 ms |
| PJRT, gradient of `Σ tanh(A·B)`, 512×512, incl. transfers | 2.42 ms | 5.54 ms |
| PJRT, same, 2048×2048 | 20.1 ms | 183 ms |

The host kernels accumulate in Double at both widths, so F64 costs the same there. On the
GB10, F64 dots are about 9× slower than the F32 ones at 2048 (which XLA runs as TF32 by
default): the three 2048³ products are about 51 GFLOP, 280 GFLOP/s at F64.

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

Remaining optional work, in order: `checkCustomVjp` at F64; the Tracer-capture API stays F32
(document); then the final-hour docs (CHANGELOG, CAPABILITIES, README) and the summary.
